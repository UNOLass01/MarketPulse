# Runbook — dead-letter queue triage

**Fired by:** `data_quality_failed`, and manually when `marketpulse.dead` is
non-empty in the RabbitMQ management UI (<http://localhost:15672>).

**What it means:** messages reached the consumer and were rejected. The
topology classifies failures deliberately (CLAUDE.md rule #10):

| Failure | Route | Why |
|---|---|---|
| Schema violation, malformed payload | **DLQ immediately** | A retry will fail identically. Retrying is just a slower way to lose the message. |
| DB unavailable, broker timeout | **Retry queue** (30s TTL, max 3) | The same message may well succeed in 30 seconds. |
| Still failing after 3 attempts | **DLQ** | Cap exists so one poison message cannot loop forever. |

So: **anything in the DLQ is either malformed or has already been retried
three times.** Neither is fixed by re-publishing it blindly.

## Triage

1. **How many, and are they still arriving?**
   ```bash
   docker compose -f docker/docker-compose.yml exec rabbitmq \
     rabbitmqctl list_queues name messages
   ```
   A fixed count = a past incident. A climbing count = still broken.

2. **Look at one.** Use the management UI's *Get messages* on
   `marketpulse.dead` (use *Nack, requeue* so you do not consume it). Check:
   - `schema_version` — a producer deployed ahead of the consumer sends a
     version the consumer does not know.
   - `payload.price` / `payload.volume` — a provider returning `null` or `0`
     for a coin it has no data for.
   - `provider_observed_at` — naive timestamps fail validation
     (`TickPayload` requires timezone-aware).

3. **Check the consumer's classification logs.** It runs as a plain process
   (`python -m services.consumer.main`), so pipe its output:
   ```bash
   python -m services.consumer.main 2>&1 | grep -i 'dead\|schema\|permanent'
   ```
   Every DLQ routing decision is logged with the reason.

4. **If the messages are fine and the failure was transient** (a Postgres
   restart, say), the retry queue should have absorbed it. Messages in the
   DLQ after a transient outage mean the outage outlasted 3 × 30s — that is a
   Postgres availability problem, not a message problem.

## Resolution

- **Schema drift (producer ahead of consumer):** deploy the consumer. Do not
  loosen the contract to make messages pass; the contract failing is it
  working.
- **Bad provider data:** nothing to replay. The ticks were never valid.
  Confirm the provider's behaviour changed, and if it now legitimately emits
  a field this project rejects, that is a contract change with an ADR, not a
  hotfix.
- **Genuine loss worth recovering:** the ticks are still available from the
  provider's historical endpoint. Re-seed rather than replaying the DLQ:
  ```bash
  make seed        # scripts/seed_historical.py, tagged source='seed'
  ```
  Ingestion is idempotent (`upsert_tick`), so re-seeding an overlapping range
  is safe and will not create duplicates.

## Afterwards

Purge the DLQ only once the cause is understood and recorded. An empty DLQ
with an unexplained cause is a lost incident.

```bash
docker compose -f docker/docker-compose.yml exec rabbitmq \
  rabbitmqctl purge_queue marketpulse.dead
```
