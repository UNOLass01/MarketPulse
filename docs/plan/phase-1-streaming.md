# Phase 1 — Streaming Backbone

**Objective:** live market data flowing end-to-end into PostgreSQL, losslessly.
**Complexity:** Medium · **Effort:** ~2 days · **Depends on:** Phase 0
**Exit criterion:** kill the consumer for 5 minutes under live load, restart it, confirm **zero message loss and zero duplicate rows**. This test *is* the phase.

## Tasks

### Contracts
- [ ] `contracts/messages.py` — `TickEnvelope` with `message_id` (UUID), `correlation_id`, `schema_version`, `emitted_at` (UTC), `symbol`, `payload`
- [ ] Payload model with price, volume, provider timestamp
- [ ] Envelope serialization/deserialization in `messaging/serialization.py`

### Provider abstraction
- [ ] `ingestion/providers/base.py` — `MarketDataProvider` protocol (`fetch(symbols) -> list[Observation]`)
- [ ] `ingestion/providers/coingecko.py` — implements it; batch multi-symbol call
- [ ] Normalise all timestamps to UTC at the provider boundary, never downstream
- [ ] Handle 429 with `Retry-After`, 5xx with exponential backoff + jitter

### Producer
- [ ] `ingestion/poller.py` — interval loop with jitter
- [ ] `ingestion/publisher.py` — persistent delivery (mode 2) **and** publisher confirms
- [ ] Bounded in-memory buffer when broker is down; shed oldest + log when full (never unbounded)
- [ ] Heartbeat counter emitted every loop iteration, success or failure
- [ ] Graceful SIGTERM: stop polling, drain in-flight, close cleanly
- [ ] `services/producer/main.py` — wiring only

### RabbitMQ topology
- [ ] `messaging/topology.py` — declarative, idempotent declaration of:
  - [ ] `market.data` (topic, durable), `market.retry`, `market.dlx`
  - [ ] `q.ticks.persist` bound `tick.#`, with `x-dead-letter-exchange`, `x-max-length`, `x-message-ttl`
  - [ ] `q.ticks.retry` TTL 30s, DLX back to `market.data`
  - [ ] `q.ticks.dead` (no TTL — dead letters persist until triaged)
- [ ] `scripts/verify_topology.py` — asserts live broker matches the declaration

### Consumer
- [ ] `messaging/connection.py` — connection/channel lifecycle, auto-reconnect with backoff
- [ ] `messaging/consumer.py` — base consumer with QoS prefetch, ack/nack, retry-count header
- [ ] Error classification: schema violation → reject to DLQ (`dead.validation`); transient → retry queue; duplicate → ack silently
- [ ] Retry capped at 3, then → `dead.exhausted`
- [ ] `services/consumer/main.py`

### Storage
- [ ] `storage/models.py` — `symbols`, `raw_ticks` (partitioned monthly by `observed_at`)
- [ ] Unique constraints: `(symbol_id, observed_at)` and `(message_id)`
- [ ] BRIN index on `observed_at`
- [ ] `storage/repositories/ticks.py` — idempotent upsert (`ON CONFLICT DO NOTHING`)
- [ ] Alembic migration + partition creation helper
- [ ] `storage/engine.py` — pooled session factory, bounded pool size

## Tests
- [ ] **Idempotency:** publish same envelope twice → exactly one row
- [ ] **Error classification:** malformed payload → DLQ; simulated DB failure → retry queue (integration, testcontainers)
- [ ] Provider mocked at the interface — **never hit the live API in tests**
- [ ] Backoff honours `Retry-After`
- [ ] Topology declaration is idempotent (declare twice, no error)
- [ ] Envelope round-trips through serialize/deserialize unchanged
- [ ] Retry-count header increments and caps at 3
- [ ] Integration: producer → broker → consumer → row appears in Postgres

## Watch out for
- Persistence without publisher confirms means the producer believes messages were saved that weren't. Both, always.
- `x-max-length` is mandatory. An unbounded queue turns a consumer outage into a broker disk-full outage.
