"""Verify the live RabbitMQ broker's topology matches messaging.topology.

Usage: python scripts/verify_topology.py

Declaring an exchange/queue with arguments identical to what already exists
is a no-op in RabbitMQ. If the broker's topology has drifted, RabbitMQ closes
the channel with 406 PRECONDITION_FAILED, which this script reports as a
non-zero exit rather than silently ignoring.
"""

import sys

from marketpulse.config import get_settings
from marketpulse.messaging.connection import ConnectionManager
from marketpulse.messaging.topology import declare_topology


def main() -> int:
    settings = get_settings()
    manager = ConnectionManager(settings.rabbitmq.url)
    try:
        declare_topology(manager.channel())
    except Exception as exc:  # noqa: BLE001 - any failure means "not verified"
        print(f"topology mismatch: {exc}", file=sys.stderr)
        return 1
    finally:
        manager.close()
    print("topology OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
