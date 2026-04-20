 
"""
main.py — entry point.
Builds the container, wires the FastAPI app, starts uvicorn.
"""
from __future__ import annotations
import logging
import os
import signal
import sys

import uvicorn

from dotenv import load_dotenv


from embedding.config import AppConfig
from embedding.infrastructure.container import Container
from embedding.adapters.api.app import create_app

logger = logging.getLogger(__name__)

def main() -> None:
    load_dotenv()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )

    logger.info("PROGRAM BOOST")

    if not os.environ.get("DB_URL"):
        logger.error("DB_URL is not set. Aborting.")
        sys.exit(1)

    config_path = os.getenv("CONFIG_PATH", "config.json")
    cfg = AppConfig.load(config_path)

    container = Container(cfg)
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

    logger.info("Starting on %s:%d ...", cfg.api.host, cfg.api.port)
    server.run()


if __name__ == "__main__":
    main()