# Phase 2 — Feature Layer

**Objective:** raw observations become versioned, leakage-free model inputs.
**Complexity:** Medium-High · **Effort:** ~2 days · **Depends on:** Phase 1
**Exit criterion:** features computed **online** (streaming consumer) are byte-identical to features computed **offline** (batch from the same raw rows). If they differ, the anti-skew design has failed and everything downstream is unreliable.

## Tasks

### Feature registry
- [x] `features/registry.py` — canonical **ordered** feature name list + `FEATURE_SET_VERSION` constant
- [x] Serving and training both read order from here; never rely on DataFrame column order

### Pure feature functions (no I/O, no randomness)
- [x] `features/technical.py` — MA 5/15/60m, EMA, **price-to-MA ratios** (ratios, not raw levels — price is non-stationary)
- [x] Volatility: rolling std of returns, high-low range, realised vol 1h/24h
- [x] Momentum: ROC 1/5/15m, RSI-style oscillator, consecutive-direction streak
- [x] Volume: change, and volume relative to its own MA
- [x] `features/temporal.py` — hour/day-of-week **sine-cosine** encoded (so hour 23 and 0 are adjacent)
- [x] `features/pipeline.py` — compose window → `FeatureVector`
- [x] Guard all ratio/division features against zero denominators → null, not inf

### Window state
- [x] `features/windows.py` — bounded ring buffers, memory is O(symbols × max_window), not O(time)
- [x] **Warm-up from DB on consumer restart** — query last N observations per symbol and rebuild state. Skipping this silently produces a NaN gap on every restart.
- [x] Gap detection: timestamp discontinuity sets `has_gap` on affected rows
- [x] Insufficient history → explicit nulls + `insufficient_history` flag, never zeros

### Storage
- [x] `features` table, partitioned monthly on `feature_ts`
- [x] Unique `(symbol_id, feature_ts, feature_set_version)` — versions coexist, no destructive migration
- [x] Index `(symbol_id, feature_ts DESC)` for the API hot path
- [x] `storage/repositories/features.py` — upsert + latest-per-symbol query

### Wiring + seeding
- [x] Consumer computes and persists features after each raw insert
- [x] `scripts/seed_historical.py` — backfill OHLCV history, tagged with a provenance column distinguishing seeded from streamed rows

## Tests
- [x] **Look-ahead leakage (highest priority):** for each feature, compute over a full sequence, then over the sequence truncated at `t`; assert the value at `t` is identical. Parametrised across every feature.
- [x] **Online/offline parity:** same raw rows through both paths → identical vectors
- [x] Restart warm-up: kill consumer mid-stream, restart, assert no NaN gap in features
- [x] Insufficient history returns null + flag (assert **not** zero)
- [x] Zero-denominator cases return null, not inf/NaN
- [x] Gap detection flags rows across an injected time hole
- [x] Cyclical encoding: hour 23 and hour 0 are close in encoded space
- [x] Window memory stays bounded after 10k observations

## Watch out for
- Any pandas `rolling(center=True)` or `.shift(-n)` is an automatic bug.
- If a feature function can't be tested with a plain list of floats, it's in the wrong module.

## Exit criterion result

Ran the exit-criterion test against the real docker-compose stack (Postgres +
RabbitMQ via testcontainers): 300 synthetic ticks published through the real
`Publisher`/topology, consumed through the real `BaseConsumer` with a process
closure identical to `services/consumer/main.py` (the **online** path,
writing to `features`), then the same raw rows read back from Postgres and
replayed through a fresh `WindowStore` + `compute_feature_vector` with no
broker involved (the **offline** path). Every row's `feature_values`,
`insufficient_history`, and `has_gap` compared equal:

```
online rows:   300
offline rows:  300
mismatches:    0
RESULT: PASS
```

Also validated live against the real CoinGecko API and dev stack (not just
synthetic data): producer → RabbitMQ → consumer → `raw_ticks` + `features`
with no errors, no dead-lettered messages; and `scripts/seed_historical.py`
backfilling real BTC-USD/ETH-USD history, with `insufficient_history`
correctly flipping to `false` once 24h+ of real history accumulates per
symbol.

Two defects were found and fixed during this thoroughness pass, both now
covered by regression tests:

1. **Pre-existing phase-1 bug** (`storage/repositories/ticks.py`): `upsert_tick`'s
   `ON CONFLICT DO NOTHING` only targeted the `(message_id, observed_at)`
   constraint, not the also-unique `(symbol_id, observed_at)` one. A
   provider legitimately reporting the same `observed_at` twice (e.g. its
   own data hasn't refreshed between polls) raised an uncaught
   `IntegrityError` instead of a silent no-op. Fixed by leaving `ON CONFLICT
   DO NOTHING` untargeted so either constraint applies. Regression test:
   `tests/integration/test_pipeline.py::test_same_symbol_and_observed_at_with_different_message_id_still_collapses_to_one_row`.
2. **Window retention/coverage contradiction** (`features/windows.py`):
   `SymbolWindow`'s default retention exactly equalled the widest feature
   window (24h), so eviction always discarded the one observation
   `has_full_coverage` needed to prove that window was fully populated —
   `insufficient_history` could never become `false` for `realised_vol_24h`,
   for *any* symbol, ever. Fixed by widening default retention to 27h (real
   margin past the widest lookback). A second, related bug in the same area
   — `features/pipeline.py`'s internal `_windowed()` helper redeclared its
   own `min_points` default, silently shadowing a deliberate tuning of
   `has_full_coverage`'s default — was also fixed by removing the redundant
   parameter. Regression tests:
   `tests/unit/test_windows.py::test_default_retention_leaves_margin_for_the_widest_feature_window`
   and `test_retention_exactly_equal_to_window_cannot_reliably_report_sufficient_coverage`.

**Phase 2 exit criterion met.**
