"""
main.py — starts FastAPI only.
"""
from __future__ import annotations
import logging
import os
import signal
import sys

import uvicorn

from embedding.config import AppConfig
from embedding.infrastructure.container import ApiContainer
from embedding.adapters.api.app import create_app
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)


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
        logger.info("Stopped.")
        sys.exit(0)


if __name__ == "__main__":
    main()