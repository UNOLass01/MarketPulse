# Phase 2 — Feature Layer

**Objective:** raw observations become versioned, leakage-free model inputs.
**Complexity:** Medium-High · **Effort:** ~2 days · **Depends on:** Phase 1
**Exit criterion:** features computed **online** (streaming consumer) are byte-identical to features computed **offline** (batch from the same raw rows). If they differ, the anti-skew design has failed and everything downstream is unreliable.

## Tasks

### Feature registry
- [ ] `features/registry.py` — canonical **ordered** feature name list + `FEATURE_SET_VERSION` constant
- [ ] Serving and training both read order from here; never rely on DataFrame column order

### Pure feature functions (no I/O, no randomness)
- [ ] `features/technical.py` — MA 5/15/60m, EMA, **price-to-MA ratios** (ratios, not raw levels — price is non-stationary)
- [ ] Volatility: rolling std of returns, high-low range, realised vol 1h/24h
- [ ] Momentum: ROC 1/5/15m, RSI-style oscillator, consecutive-direction streak
- [ ] Volume: change, and volume relative to its own MA
- [ ] `features/temporal.py` — hour/day-of-week **sine-cosine** encoded (so hour 23 and 0 are adjacent)
- [ ] `features/pipeline.py` — compose window → `FeatureVector`
- [ ] Guard all ratio/division features against zero denominators → null, not inf

### Window state
- [ ] `features/windows.py` — bounded ring buffers, memory is O(symbols × max_window), not O(time)
- [ ] **Warm-up from DB on consumer restart** — query last N observations per symbol and rebuild state. Skipping this silently produces a NaN gap on every restart.
- [ ] Gap detection: timestamp discontinuity sets `has_gap` on affected rows
- [ ] Insufficient history → explicit nulls + `insufficient_history` flag, never zeros

### Storage
- [ ] `features` table, partitioned monthly on `feature_ts`
- [ ] Unique `(symbol_id, feature_ts, feature_set_version)` — versions coexist, no destructive migration
- [ ] Index `(symbol_id, feature_ts DESC)` for the API hot path
- [ ] `storage/repositories/features.py` — upsert + latest-per-symbol query

### Wiring + seeding
- [ ] Consumer computes and persists features after each raw insert
- [ ] `scripts/seed_historical.py` — backfill OHLCV history, tagged with a provenance column distinguishing seeded from streamed rows

## Tests
- [ ] **Look-ahead leakage (highest priority):** for each feature, compute over a full sequence, then over the sequence truncated at `t`; assert the value at `t` is identical. Parametrised across every feature.
- [ ] **Online/offline parity:** same raw rows through both paths → identical vectors
- [ ] Restart warm-up: kill consumer mid-stream, restart, assert no NaN gap in features
- [ ] Insufficient history returns null + flag (assert **not** zero)
- [ ] Zero-denominator cases return null, not inf/NaN
- [ ] Gap detection flags rows across an injected time hole
- [ ] Cyclical encoding: hour 23 and hour 0 are close in encoded space
- [ ] Window memory stays bounded after 10k observations

## Watch out for
- Any pandas `rolling(center=True)` or `.shift(-n)` is an automatic bug.
- If a feature function can't be tested with a plain list of floats, it's in the wrong module.
