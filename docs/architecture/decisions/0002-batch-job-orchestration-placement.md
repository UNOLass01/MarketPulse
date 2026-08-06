# 0002 — Where I/O-heavy batch-job orchestration lives

**Status:** Accepted
**Date:** 2026-08-06

## Context

Phase 4 introduces Airflow DAGs whose task callables must import from
`marketpulse.*` and stay thin (CLAUDE.md rule #9). Some of those tasks —
`dag_feature_backfill` in particular — need real orchestration: chunk a date
range, replay stored raw ticks through window state, call feature
computation, upsert results. That orchestration does I/O (DB reads/writes)
end to end.

`src/marketpulse/features/` is explicitly locked to pure functions only, no
I/O (CLAUDE.md rule #1) — feature *calculation* logic must never be
duplicated or entangled with I/O. So the backfill orchestrator can't live
there, even though it calls into it.

## Options considered

1. **Put it in `features/` anyway.** Rejected outright — directly violates
   rule #1 and would make every future contributor re-litigate whether "just
   this one orchestration file" is exempt.
2. **Put it in `storage/`.** `storage/` already does I/O (repositories,
   partitions), so it's not a purity violation, but `storage/` is documented
   as "engine, ORM models, repositories (all queries)" — a backfill job is a
   cross-cutting operation that *uses* storage, not storage itself.
3. **New `jobs/` package** for I/O-heavy, Airflow-invoked orchestration that
   wires pure domain logic (features, ml) together with storage — mirroring
   the role `ml/pipeline.py` already plays for training, but for operations
   that don't have one obvious owning domain package.

## Decision

New package `src/marketpulse/jobs/`. It holds orchestrators like
`jobs/backfill.py` that are allowed to do I/O and call into `features/`'s
pure functions, `storage/`'s repositories, etc. The rule stays simple:
`features/` never does I/O, no matter who's calling it or why.

`ml/pipeline.py` is treated as the precedent for this shape (I/O + pure
logic wiring for one operational concern) and is *not* moved into `jobs/` —
it's central enough to `ml/` to stay put, and moving it isn't in scope here.

Storage-native lifecycle operations (partition export/verify/drop for
`dag_data_archival`) stay in `storage/archival.py` instead of `jobs/`,
since that work fits option 2's reasoning — it's fundamentally storage
lifecycle management, in the same spirit as `storage/partitions.py`, not a
cross-domain wiring job.

## Consequences

- `jobs/` is for orchestration with no single natural owning package —
  expect it to stay small. A new file here should be able to say "no
  existing package fits" before landing.
- `features/`'s no-I/O rule has zero carve-outs; anything that needs to read
  or write goes in `jobs/` (or elsewhere) and imports the pure functions.
- If `jobs/` grows past feature backfill into several unrelated jobs, split
  it by domain rather than letting it become a second dumping ground.
