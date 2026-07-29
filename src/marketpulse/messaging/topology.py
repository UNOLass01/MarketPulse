"""Declarative, idempotent RabbitMQ topology for the tick pipeline.

::

    market.data  (topic) --tick.#--> q.ticks.persist
    market.retry (topic) --tick.#--> q.ticks.retry   (TTL 30s)
    market.dlx   (topic) --#------>  q.ticks.dead     (no TTL)

Consumer-side flow (see ``messaging.consumer``):

- success                          -> ack
- schema violation                 -> publish to market.dlx (dead.validation), ack
- transient error, retries < cap   -> publish to market.retry (x-retry-count incremented), ack
- transient error, retries == cap  -> publish to market.dlx (dead.exhausted), ack
- q.ticks.retry TTL expiry         -> RabbitMQ dead-letters back into market.data
  (routing key preserved), landing in q.ticks.persist for reprocessing.
- unhandled exception              -> nack(requeue=False); q.ticks.persist's own
  ``x-dead-letter-exchange`` (market.dlx) is the safety net for anything the
  consumer didn't explicitly classify.
"""

from pika.adapters.blocking_connection import BlockingChannel

EXCHANGE_DATA = "market.data"
EXCHANGE_RETRY = "market.retry"
EXCHANGE_DLX = "market.dlx"

QUEUE_PERSIST = "q.ticks.persist"
QUEUE_RETRY = "q.ticks.retry"
QUEUE_DEAD = "q.ticks.dead"

ROUTING_KEY_TICK = "tick.#"
ROUTING_KEY_ALL = "#"

RETRY_TTL_MS = 30_000
# Stale unconsumed ticks are worthless once an hour has passed; better to
# dead-letter them than retry indefinitely.
PERSIST_MESSAGE_TTL_MS = 60 * 60 * 1000
# Mandatory: an unbounded queue turns a consumer outage into a broker
# disk-full outage. Default overflow behaviour (drop-head) sheds the oldest.
PERSIST_MAX_LENGTH = 100_000


def declare_topology(channel: BlockingChannel) -> None:
    """Declare exchanges and queues. Safe to call repeatedly.

    Raises whatever pika raises (channel closes with 406 PRECONDITION_FAILED)
    if a queue/exchange already exists with different arguments — that
    signals a real topology drift, not something to silently paper over.
    """
    channel.exchange_declare(exchange=EXCHANGE_DATA, exchange_type="topic", durable=True)
    channel.exchange_declare(exchange=EXCHANGE_RETRY, exchange_type="topic", durable=True)
    channel.exchange_declare(exchange=EXCHANGE_DLX, exchange_type="topic", durable=True)

    channel.queue_declare(
        queue=QUEUE_PERSIST,
        durable=True,
        arguments={
            "x-dead-letter-exchange": EXCHANGE_DLX,
            "x-max-length": PERSIST_MAX_LENGTH,
            "x-message-ttl": PERSIST_MESSAGE_TTL_MS,
        },
    )
    channel.queue_bind(queue=QUEUE_PERSIST, exchange=EXCHANGE_DATA, routing_key=ROUTING_KEY_TICK)

    channel.queue_declare(
        queue=QUEUE_RETRY,
        durable=True,
        arguments={
            "x-message-ttl": RETRY_TTL_MS,
            "x-dead-letter-exchange": EXCHANGE_DATA,
        },
    )
    channel.queue_bind(queue=QUEUE_RETRY, exchange=EXCHANGE_RETRY, routing_key=ROUTING_KEY_TICK)

    channel.queue_declare(queue=QUEUE_DEAD, durable=True)
    channel.queue_bind(queue=QUEUE_DEAD, exchange=EXCHANGE_DLX, routing_key=ROUTING_KEY_ALL)
