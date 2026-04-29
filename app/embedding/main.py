"""
main.py — starts FastAPI only.
"""
from __future__ import annotations
import logging
import os
import signal
import sys

import uvicorn

from dotenv import load_dotenv
load_dotenv()

from embedding.config import AppConfig
from embedding.infrastructure.container import ApiContainer
from embedding.adapters.api.app import create_app

logger = logging.getLogger(__name__)


def _suppress_multiprocess_noise() -> None:
    """
    Suppress the ResourceTracker AttributeError that multiprocess (used by
    RAGAS/datasets/sentence-transformers) emits on interpreter shutdown.
    This is a known upstream bug — the ResourceTracker destructor runs after
    threading primitives have already been partially torn down.
    We silence it at interpreter exit so Ctrl+C produces clean output.
    """
    import multiprocessing.util as _util

    _original_exit = _util._exit_function

    def _patched_exit(*args, **kwargs):
        try:
            _original_exit(*args, **kwargs)
        except Exception:
            pass

    _util._exit_function = _patched_exit

    # also silence the __del__ traceback printed by Python's unraisable hook
    original_hook = sys.unraisablehook

    def _quiet_unraisable(args):
        msg = str(args.exc_value)
        # only suppress the specific known noise from multiprocess
        if "_recursion_count" in msg or "ResourceTracker" in str(args.object):
            return
        original_hook(args)

    sys.unraisablehook = _quiet_unraisable


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
    container = ApiContainer(cfg)
    app = create_app(container)

    uv_config = uvicorn.Config(
        app=app,
        host=cfg.api.host,
        port=cfg.api.port,
        log_level="info",
        access_log=True,
    )
    server = uvicorn.Server(uv_config)
    signal.signal(signal.SIGTERM, lambda *_: server.handle_exit(sig=signal.SIGTERM, frame=None))

    logger.info("Starting API on %s:%d ...", cfg.api.host, cfg.api.port)
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    finally:
        logger.info("Stopped.")


if __name__ == "__main__":
    main()