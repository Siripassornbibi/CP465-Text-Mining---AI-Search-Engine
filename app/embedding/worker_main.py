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
load_dotenv()

logger = logging.getLogger(__name__)


def _suppress_multiprocess_noise() -> None:
    try:
        import multiprocessing.util as _util
        _original_exit = _util._exit_function

        def _patched_exit(*args, **kwargs):
            try:
                _original_exit(*args, **kwargs)
            except Exception:
                pass

        _util._exit_function = _patched_exit
    except Exception:
        pass

    original_hook = sys.unraisablehook

    def _quiet_unraisable(args):
        if "_recursion_count" in str(args.exc_value) or "ResourceTracker" in str(args.object):
            return
        original_hook(args)

    sys.unraisablehook = _quiet_unraisable


async def run(container: WorkerContainer) -> None:
    await container.start()

    loop = asyncio.get_running_loop()
    stop_event = asyncio.Event()

    def _request_stop(*_):
        loop.call_soon_threadsafe(stop_event.set)

    # register both signals so Ctrl+C and Docker/k8s SIGTERM both clean up
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, _request_stop)

    logger.info("Worker running — press Ctrl+C to stop.")

    try:
        await stop_event.wait()
    finally:
        # finally block runs even if the coroutine is cancelled externally
        logger.info("Shutting down worker ...")
        try:
            await container.stop()
            logger.info("Postgres connection closed. Stopped.")
        except Exception as e:
            logger.error("Error during shutdown: %s", e)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    _suppress_multiprocess_noise()

    if not os.environ.get("DB_URL"):
        logger.error("DB_URL is not set. Aborting.")
        sys.exit(1)

    cfg = AppConfig.load(os.getenv("CONFIG_PATH", "config.json"))
    container = WorkerContainer(cfg)

    try:
        asyncio.run(run(container))
    except KeyboardInterrupt:
        # suppress the traceback — cleanup already happened in the finally block
        pass


if __name__ == "__main__":
    main()