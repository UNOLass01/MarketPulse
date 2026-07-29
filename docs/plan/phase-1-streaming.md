# Phase 1 — Streaming Backbone

**Objective:** live market data flowing end-to-end into PostgreSQL, losslessly.
**Complexity:** Medium · **Effort:** ~2 days · **Depends on:** Phase 0
**Exit criterion:** kill the consumer for 5 minutes under live load, restart it, confirm **zero message loss and zero duplicate rows**. This test *is* the phase.

## Tasks

### Contracts
- [x] `contracts/messages.py` — `TickEnvelope` with `message_id` (UUID), `correlation_id`, `schema_version`, `emitted_at` (UTC), `symbol`, `payload`
- [x] Payload model with price, volume, provider timestamp
- [x] Envelope serialization/deserialization in `messaging/serialization.py`

### Provider abstraction
- [x] `ingestion/providers/base.py` — `MarketDataProvider` protocol (`fetch(symbols) -> list[Observation]`)
- [x] `ingestion/providers/coingecko.py` — implements it; batch multi-symbol call
- [x] Normalise all timestamps to UTC at the provider boundary, never downstream
- [x] Handle 429 with `Retry-After`, 5xx with exponential backoff + jitter

### Producer
- [x] `ingestion/poller.py` — interval loop with jitter
- [x] `ingestion/publisher.py` — persistent delivery (mode 2) **and** publisher confirms
- [x] Bounded in-memory buffer when broker is down; shed oldest + log when full (never unbounded)
- [x] Heartbeat counter emitted every loop iteration, success or failure
- [x] Graceful SIGTERM: stop polling, drain in-flight, close cleanly
- [x] `services/producer/main.py` — wiring only

### RabbitMQ topology
- [x] `messaging/topology.py` — declarative, idempotent declaration of:
  - [x] `market.data` (topic, durable), `market.retry`, `market.dlx`
  - [x] `q.ticks.persist` bound `tick.#`, with `x-dead-letter-exchange`, `x-max-length`, `x-message-ttl`
  - [x] `q.ticks.retry` TTL 30s, DLX back to `market.data`
  - [x] `q.ticks.dead` (no TTL — dead letters persist until triaged)
- [x] `scripts/verify_topology.py` — asserts live broker matches the declaration

### Consumer
- [x] `messaging/connection.py` — connection/channel lifecycle, auto-reconnect with backoff
- [x] `messaging/consumer.py` — base consumer with QoS prefetch, ack/nack, retry-count header
- [x] Error classification: schema violation → reject to DLQ (`dead.validation`); transient → retry queue; duplicate → ack silently
- [x] Retry capped at 3, then → `dead.exhausted`
- [x] `services/consumer/main.py`

### Storage
- [x] `storage/models.py` — `symbols`, `raw_ticks` (partitioned monthly by `observed_at`)
- [x] Unique constraints: `(symbol_id, observed_at)` and `(message_id)` (partitioned tables require the partition key in every unique index, so `message_id`'s constraint carries `observed_at` too)
- [x] BRIN index on `observed_at`
- [x] `storage/repositories/ticks.py` — idempotent upsert (`ON CONFLICT DO NOTHING`)
- [x] Alembic migration + partition creation helper
- [x] `storage/engine.py` — pooled session factory, bounded pool size

## Tests
- [x] **Idempotency:** publish same envelope twice → exactly one row
- [x] **Error classification:** malformed payload → DLQ; simulated DB failure → retry queue (integration, testcontainers)
- [x] Provider mocked at the interface — **never hit the live API in tests**
- [x] Backoff honours `Retry-After`
- [x] Topology declaration is idempotent (declare twice, no error)
- [x] Envelope round-trips through serialize/deserialize unchanged
- [x] Retry-count header increments and caps at 3
- [x] Integration: producer → broker → consumer → row appears in Postgres

## Watch out for
- Persistence without publisher confirms means the producer believes messages were saved that weren't. Both, always.
- `x-max-length` is mandatory. An unbounded queue turns a consumer outage into a broker disk-full outage.

## Exit criterion result

Ran against the live docker-compose stack: synthetic load at 2 msg/s through the
real `Publisher`/topology, real `services.consumer.main` process hard-killed
(`TerminateProcess`, simulating a crash rather than a graceful stop) for a full
5 minutes while the producer kept publishing, then restarted.

```
published:        646
rows in postgres:  646
distinct in pg:    646
missing (lost):    0
unexpected extra:  0
duplicate rows:    0
RESULT: PASS
```

**Phase 1 exit criterion met.**
