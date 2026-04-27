"""
worker_main.py — starts RabbitMQ worker only.
"""
from __future__ import annotations
import asyncio
import logging
import os
import signal
import sys

from embedding.config import AppConfig
from embedding.infrastructure.container import WorkerContainer
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


async def run(container: WorkerContainer) -> None:
    await container.start()

    loop = asyncio.get_event_loop()
    stop_event = asyncio.Event()

    # handle Ctrl+C and SIGTERM gracefully
    def _request_stop(*_):
        loop.call_soon_threadsafe(stop_event.set)

    signal.signal(signal.SIGINT,  _request_stop)
    signal.signal(signal.SIGTERM, _request_stop)

    logger.info("Worker running — press Ctrl+C to stop.")
    await stop_event.wait()

    logger.info("Shutting down worker ...")
    await container.stop()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    if not os.environ.get("DB_URL"):
        logger.error("DB_URL is not set. Aborting.")
        sys.exit(1)

    cfg = AppConfig.load(os.getenv("CONFIG_PATH", "config.json"))
    container = WorkerContainer(cfg)

    asyncio.run(run(container))
    logger.info("Stopped.")


if __name__ == "__main__":
    main()