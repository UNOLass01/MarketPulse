"""Dataset assembly: join features to forward-shifted price, label, and split.

Every function here is pure -- no DB, no network. Callers fetch rows via
``storage.repositories.training_data`` and pass them in; this module only
ever transforms in-memory data, which is what makes the leakage-sensitive
join/split logic testable with nothing more than synthetic rows.

The join, exclusion, and split rules encoded below map directly onto the
phase-3 plan's non-negotiables:

- "Training query excludes the most recent H of data" -- rows whose
  ``feature_ts + H`` has no matching future price are dropped as unlabelable
  (``_attach_forward_price``).
- "Drop rows flagged insufficient_history or has_gap" -- ``assemble_dataset``.
- "Temporal split 70/15/15, chronological, with an embargo gap of H between
  segments" -- ``_split_symbol``: a contiguous split would let a training
  example's label (which reads price at ``t + H``) peek at validation's raw
  price, so the last H of every segment except the final one is embargoed.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import pandas as pd

from marketpulse.exceptions import PermanentError
from marketpulse.features.registry import FEATURE_NAMES, ordered_values
from marketpulse.ml.labeling import LABELS, derive_theta, label_from_return


class InsufficientDataError(PermanentError):
    """Not enough labelable, clean history to train on.

    Raised instead of silently proceeding on a thin sample (phase-3 plan:
    "fail loudly, don't train on a thin sample").
    """


@dataclass(frozen=True)
class FeatureRow:
    """One persisted feature vector joined to the tick it was computed from.

    ``price`` is the price of the same tick ``feature_ts`` was computed
    from -- features and raw ticks are 1:1 by ``(symbol, observed_at)``, so
    this is a plain lookup, not an approximation.
    """

    symbol: str
    feature_ts: datetime
    price: float
    feature_values: dict[str, float | None]
    insufficient_history: bool
    has_gap: bool


@dataclass(frozen=True)
class SplitConfig:
    train_frac: float = 0.70
    val_frac: float = 0.15
    min_rows_per_split: int = 200
    min_class_frac: float = 0.05

    def __post_init__(self) -> None:
        if not 0 < self.train_frac < 1 or not 0 < self.val_frac < 1:
            raise ValueError("train_frac and val_frac must be in (0, 1)")
        if self.train_frac + self.val_frac >= 1:
            raise ValueError("train_frac + val_frac must leave a non-empty test fraction")


@dataclass(frozen=True)
class Dataset:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    theta: float
    horizon: timedelta
    feature_set_version: int
    symbols: tuple[str, ...] = field(default_factory=tuple)

    def row_counts(self) -> dict[str, int]:
        return {
            "train": len(self.train),
            "validation": len(self.validation),
            "test": len(self.test),
        }

    def class_distribution(self, split: str = "train") -> dict[str, int]:
        frame = getattr(self, split)
        counts = frame["label"].value_counts()
        return {str(label): int(count) for label, count in counts.items()}

    def reference_feature_stats(self) -> dict[str, dict[str, float]]:
        """Per-feature summary stats of the *training* split, persisted on
        promotion as the baseline Phase 6's drift monitor compares live
        feature windows against (CLAUDE.md: reference is tied to the
        Production model, not to "yesterday's data").
        """
        stats: dict[str, dict[str, float]] = {}
        for name in FEATURE_NAMES:
            column = self.train[name].dropna()
            if column.empty:
                continue
            stats[name] = {
                "mean": float(column.mean()),
                "std": float(column.std(ddof=0)),
                "min": float(column.min()),
                "max": float(column.max()),
                "p50": float(column.quantile(0.5)),
            }
        return stats


_ID_COLUMNS = ("symbol", "feature_ts", "price", "insufficient_history", "has_gap")


def _rows_to_frame(rows: Sequence[FeatureRow]) -> pd.DataFrame:
    """Expand each row's ``feature_values`` mapping into the registry's fixed
    column order. Uses ``ordered_values`` (raises on a missing key) rather
    than a permissive ``.get`` -- a feature that was never computed must be
    visible as a bug, not silently indistinguishable from a legitimate
    insufficient-history ``None``.
    """
    records = []
    for row in rows:
        record: dict[str, object] = {
            "symbol": row.symbol,
            "feature_ts": pd.Timestamp(row.feature_ts),
            "price": float(row.price),
            "insufficient_history": row.insufficient_history,
            "has_gap": row.has_gap,
        }
        record.update(zip(FEATURE_NAMES, ordered_values(row.feature_values), strict=True))
        records.append(record)
    columns = list(_ID_COLUMNS) + list(FEATURE_NAMES)
    frame = pd.DataFrame.from_records(records, columns=columns)
    # A feature that's legitimately None on every row of a slice (e.g. a
    # delta-based feature like roc_1m when the tick cadence is coarser than
    # its window -- see features.windows' coverage docstring) makes pandas
    # infer an all-None column as dtype=object, not float64-with-NaN. Force
    # it: LightGBM's Dataset (and MLflow's signature inference) both reject
    # object-dtype columns outright, even ones that are just floats + None.
    frame[list(FEATURE_NAMES)] = frame[list(FEATURE_NAMES)].astype("float64")
    return frame


def _attach_forward_price(
    frame: pd.DataFrame, *, horizon: timedelta, tolerance: timedelta
) -> pd.DataFrame:
    """Attach ``future_price``/``future_ts``: the earliest known price at or
    after ``feature_ts + horizon``, found via an as-of forward join against
    the same price series.

    Ticks are not evenly spaced (poll cadence varies, gaps happen), so "the
    row H later" is not a fixed row offset -- it has to be located by time.
    ``search_ts = feature_ts + horizon`` is a monotonic shift of the already
    time-sorted ``feature_ts``, so both sides of the as-of join stay sorted
    without any re-sort-then-realign step, and row order is untouched.
    """
    frame = frame.sort_values("feature_ts").reset_index(drop=True)
    lookup = frame[["feature_ts", "price"]].rename(
        columns={"feature_ts": "future_ts", "price": "future_price"}
    )
    probe = pd.DataFrame({"search_ts": frame["feature_ts"] + horizon})
    joined = pd.merge_asof(
        probe,
        lookup,
        left_on="search_ts",
        right_on="future_ts",
        direction="forward",
        tolerance=tolerance,
    )
    return frame.assign(
        future_price=joined["future_price"].to_numpy(),
        future_ts=joined["future_ts"].to_numpy(),
    )


def _split_symbol(
    frame: pd.DataFrame, *, split: SplitConfig, horizon: timedelta
) -> dict[str, pd.DataFrame]:
    """Chronological 70/15/15 split for a single symbol's rows, with an
    embargo gap of ``horizon`` trimmed off the end of train and validation.

    Boundaries are picked by row position on the time-sorted sequence (never
    by shuffling), then each segment's tail is trimmed so no row whose label
    used a price from the *next* segment survives in this one.
    """
    frame = frame.sort_values("feature_ts").reset_index(drop=True)
    n = len(frame)
    if n < 3:
        return {"train": frame.iloc[0:0], "validation": frame.iloc[0:0], "test": frame.iloc[0:0]}

    train_idx = max(1, int(n * split.train_frac))
    val_idx = max(train_idx + 1, int(n * (split.train_frac + split.val_frac)))
    val_idx = min(val_idx, n - 1)

    train_boundary = frame["feature_ts"].iloc[train_idx - 1]
    val_boundary = frame["feature_ts"].iloc[val_idx - 1]

    ts = frame["feature_ts"]
    train = frame[ts <= train_boundary - horizon]
    validation = frame[(ts > train_boundary) & (ts <= val_boundary - horizon)]
    test = frame[ts > val_boundary]
    return {"train": train, "validation": validation, "test": test}


def _check_sufficiency(dataset_frames: dict[str, pd.DataFrame], split: SplitConfig) -> None:
    for name, frame in dataset_frames.items():
        if len(frame) < split.min_rows_per_split:
            raise InsufficientDataError(
                f"'{name}' split has {len(frame)} rows, below the minimum of "
                f"{split.min_rows_per_split} -- refusing to train on a thin sample"
            )
        # reindex over the full label set (not just labels present) so a
        # class that's entirely absent -- not merely thin -- is caught too;
        # value_counts() alone would silently omit a zero-count class.
        class_fracs = frame["label"].value_counts(normalize=True).reindex(LABELS, fill_value=0.0)
        thin_classes = class_fracs[class_fracs < split.min_class_frac]
        if not thin_classes.empty:
            raise InsufficientDataError(
                f"'{name}' split has class(es) below the minimum share of "
                f"{split.min_class_frac:.0%}: {thin_classes.to_dict()}"
            )


def assemble_dataset(
    rows: Sequence[FeatureRow],
    *,
    horizon: timedelta,
    feature_set_version: int,
    forward_price_tolerance: timedelta,
    theta: float | None = None,
    theta_quantile: float = 0.4,
    split: SplitConfig | None = None,
) -> Dataset:
    """Join, label, and chronologically split ``rows`` into a :class:`Dataset`.

    Each symbol present in ``rows`` is joined to its own forward-shifted
    price and split independently (a per-symbol timeline), then same-named
    segments are concatenated -- this keeps the embargo/chronology guarantee
    correct even when ``rows`` mixes multiple symbols with interleaved but
    not identical timestamps.

    ``forward_price_tolerance`` bounds how far past ``feature_ts + horizon``
    the as-of join may reach for a match -- it should be on the order of the
    tick pipeline's own gap threshold (``FeaturesSettings.gap_threshold_seconds``),
    not the horizon itself. A tolerance as wide as ``horizon`` would let a
    row with no real H-ahead price silently borrow one from up to ``2H``
    later, understating how much of the recent tail is actually unlabelable.
    """
    if not rows:
        raise InsufficientDataError("cannot assemble a dataset from zero feature rows")

    split = split or SplitConfig()
    tolerance = forward_price_tolerance

    frame = _rows_to_frame(rows)
    joined = pd.concat(
        [
            _attach_forward_price(group, horizon=horizon, tolerance=tolerance)
            for _, group in frame.groupby("symbol", sort=False)
        ],
        ignore_index=True,
    )

    # Rows with no future price within tolerance are the most-recent-H tail
    # (or sit next to a gap) -- not labelable yet, excluded rather than
    # imputed (CLAUDE.md: never fabricate a value that teaches something false).
    labelable = joined.dropna(subset=["future_price"]).copy()
    labelable = labelable[~labelable["insufficient_history"] & ~labelable["has_gap"]]

    if labelable.empty:
        raise InsufficientDataError(
            "no rows survived the labelable/insufficient_history/has_gap filters"
        )

    labelable["forward_return"] = (labelable["future_price"] - labelable["price"]) / labelable[
        "price"
    ]

    resolved_theta = theta
    if resolved_theta is None:
        resolved_theta = derive_theta(labelable["forward_return"].tolist(), quantile=theta_quantile)

    labelable["label"] = labelable["forward_return"].map(
        lambda r: label_from_return(r, resolved_theta)
    )

    per_symbol_splits = [
        _split_symbol(group, split=split, horizon=horizon)
        for _, group in labelable.groupby("symbol", sort=False)
    ]
    combined = {
        name: pd.concat([s[name] for s in per_symbol_splits], ignore_index=True)
        .sort_values("feature_ts")
        .reset_index(drop=True)
        for name in ("train", "validation", "test")
    }

    _check_sufficiency(combined, split)

    symbols = tuple(sorted(labelable["symbol"].unique()))
    return Dataset(
        train=combined["train"],
        validation=combined["validation"],
        test=combined["test"],
        theta=resolved_theta,
        horizon=horizon,
        feature_set_version=feature_set_version,
        symbols=symbols,
    )
