"""Forward-return labeling: UP / DOWN / STABLE with a deadband threshold.

Pure functions only -- no I/O, no DB. Labels depend on price at ``t + H``,
which doesn't exist yet at ingest time, so they are computed here, at
training time, from stored history (CLAUDE.md rule #4), never written back
into the streaming/feature-ingestion path.
"""

from collections.abc import Sequence

import numpy as np

LABEL_DOWN = "DOWN"
LABEL_STABLE = "STABLE"
LABEL_UP = "UP"

# Alphabetical, and therefore also the fixed class index order used by
# training/evaluation -- DOWN=0, STABLE=1, UP=2. Any code that needs a class
# index (LightGBM's multiclass objective, confusion matrices, baselines) must
# go through LABEL_TO_INDEX rather than re-deriving an ordering, the same
# discipline ``features.registry`` applies to feature column order.
LABELS: tuple[str, ...] = (LABEL_DOWN, LABEL_STABLE, LABEL_UP)
LABEL_TO_INDEX: dict[str, int] = {label: index for index, label in enumerate(LABELS)}
INDEX_TO_LABEL: dict[int, str] = dict(enumerate(LABELS))


def forward_return(current_price: float, future_price: float) -> float:
    """Simple percentage return from ``current_price`` to ``future_price``.

    Raises ``ValueError`` on a non-positive ``current_price`` rather than
    returning ``inf``/``nan`` -- a zero or negative price is a data-quality
    bug, not a legitimate return of zero.
    """
    if current_price <= 0:
        raise ValueError(f"current_price must be positive, got {current_price}")
    return (future_price - current_price) / current_price


def label_from_return(forward_ret: float, theta: float) -> str:
    """UP if ``forward_ret > theta``, DOWN if ``< -theta``, else STABLE.

    ``theta`` itself must be non-negative; the boundary values ``+theta`` and
    ``-theta`` are inclusive of STABLE (strict ``>``/``<``), so a return
    exactly at the deadband edge is never classified as a directional move.
    """
    if theta < 0:
        raise ValueError(f"theta must be non-negative, got {theta}")
    if forward_ret > theta:
        return LABEL_UP
    if forward_ret < -theta:
        return LABEL_DOWN
    return LABEL_STABLE


def derive_theta(forward_returns: Sequence[float], *, quantile: float = 0.4) -> float:
    """theta = the ``quantile`` quantile of |forward_return|, derived from the
    empirical distribution rather than picked by hand (phase-3 plan).

    A ``quantile`` of 0.4 means ~40% of examples fall inside the deadband
    (STABLE) and the remaining 60% split across UP/DOWN -- a deliberately
    data-driven choice so the label distribution reflects how this symbol
    actually moves, not an arbitrary fixed percentage threshold.
    """
    if not forward_returns:
        raise ValueError("cannot derive theta from an empty return series")
    if not 0 < quantile < 1:
        raise ValueError(f"quantile must be in (0, 1), got {quantile}")
    abs_returns = np.abs(np.asarray(list(forward_returns), dtype=float))
    return float(np.quantile(abs_returns, quantile))
