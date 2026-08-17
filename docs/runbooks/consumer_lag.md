# Runbook — consumer lag / stale features

**Fired by:** `prediction_distribution_shift`, and manually when `/ready`
reports the `features` dependency unhealthy or `/api/v1/predictions/{symbol}`
starts returning `503 features_stale`.

**What it means:** feature rows have stopped arriving, or are arriving far
behind real time. The API is doing exactly what it should — refusing to serve
a prediction the caller could not tell was stale — so *this is not an API
problem*. Look upstream.

> Check this before believing the model has degraded. A drift or accuracy
> alert with a stalled pipeline behind it is a Phase 1 cause wearing a Phase 3
> symptom.

## Triage, in order

1. **How stale, and for which symbols?**
   ```bash
   curl -s localhost:8000/api/v1/symbols | jq
   curl -s localhost:8000/health/dependencies | jq
   ```
   One symbol behind → provider-side gap for that coin. *All* symbols behind
   → producer or consumer.

2. **Is the producer alive?** It runs as a plain process
   (`python -m services.producer.main`), not a compose service — check the
   terminal or supervisor you started it under. A dead poller is silent, not
   loud: no errors is not good news here. Confirm ticks are actually landing:
   ```bash
   curl -s "localhost:8000/api/v1/ticks/BTC-USD?hours=1" | jq '.ticks | length'
   ```

3. **Is the queue backing up?** RabbitMQ management UI at
   <http://localhost:15672>. Depth on `marketpulse.persist` climbing means the
   consumer is behind; depth at zero with no new ticks means nothing is being
   published.

4. **Is the consumer crash-looping?** Same deal —
   `python -m services.consumer.main`. Check its logs, and check the
   infrastructure it depends on:
   ```bash
   docker compose -f docker/docker-compose.yml ps
   ```
   Restart loops usually mean a DB connection problem, not a message problem.

5. **Are messages failing rather than lagging?** If `marketpulse.dead` is
   growing, this is the wrong runbook — go to
   [`dlq_triage.md`](./dlq_triage.md).

## Resolution

- **Provider outage:** nothing to do but wait; the staleness guard is already
  handling it correctly. Do not raise `MP_SERVING__MAX_FEATURE_AGE_SECONDS`
  to make the alert stop — that only hides the gap from callers.
- **Consumer stuck:** restart it. Warm-up rebuilds window state from Postgres
  (`services/consumer/main.py::_warm_up`), so a restart costs a short catch-up,
  not a feature gap.
- **Consumer genuinely too slow:** check whether feature computation is
  bounded (`MP_FEATURES__MAX_BUFFER_POINTS`) before scaling anything.

## Afterwards

Feature rows written during a partial outage are flagged `has_gap=True` and
are excluded from training automatically (`ml.dataset.assemble_dataset`). No
manual cleanup is needed. If a long gap needs backfilling, trigger
`dag_feature_backfill` rather than editing rows by hand.
