# 0001 — Why RabbitMQ over Kafka

**Status:** Accepted
**Date:** 2026-07-29

## Context

The ingestion pipeline needs a broker between the tick poller (producer) and
the consumer that persists to Postgres. Throughput is one small-to-mid-size
crypto tick stream, not a multi-topic, multi-consumer-group event backbone.
The team is a single operator running this on a small Docker Compose stack,
not a managed cluster.

## Options considered

1. **Kafka** — high throughput, log-based replay, strong ecosystem for
   large-scale streaming. Costs: a ZooKeeper/KRaft cluster (or a managed
   service) to operate, heavier resource footprint, more moving parts than
   this project's data volume justifies, steeper operational learning curve.
2. **RabbitMQ** — simpler operational model (single container + volume),
   built-in DLQ and retry semantics via exchanges/queues that map directly
   onto CLAUDE.md rule #10 (schema violations → DLQ, transient errors →
   retry queue), sufficient throughput for one tick stream, management UI
   included in the image used here.
3. **Direct DB writes, no broker** — simplest possible, but couples the
   poller's availability to the DB's and removes any buffering or backpressure
   handling; rejected because Phase 1's exit criterion (kill the consumer for
   5 minutes, zero loss/zero dupes) requires a broker to hold messages while
   the consumer is down.

## Decision

Use RabbitMQ. It maps cleanly onto the retry/DLQ split this project already
requires, and its operational cost fits a single-operator, single-stream
project. Kafka's strengths (multi-consumer-group replay, very high
throughput) aren't needed here and the project's "never add without asking"
list explicitly excludes introducing Kafka.

## Consequences

- Retry queue and DLQ are implemented as RabbitMQ exchanges/queues, not a
  Kafka consumer-group replay pattern.
- If a second independent consumer of the same stream is ever needed, or
  throughput requirements grow by an order of magnitude, this decision
  should be revisited — write a new ADR rather than silently reintroducing
  Kafka.
- Message ordering guarantees are per-queue, not per-partition; this is
  sufficient because there is one producer and one logical consumer group
  per queue.
