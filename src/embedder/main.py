"""
main.py — programmatic entry point
Starts uvicorn (FastAPI) which in turn boots the RabbitMQ worker
via the app lifespan. Both run in the same asyncio event loop.
"""
 
from __future__ import annotations
 
import logging
import os
import signal
import sys
 
import uvicorn
 
logger = logging.getLogger(__name__)
 
 
def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s — %(message)s",
        datefmt="%H:%M:%S",
    )
 
    config_path = os.getenv("CONFIG_PATH", "config.json")
    host        = os.getenv("HOST", "0.0.0.0")
    port        = int(os.getenv("PORT", "8000"))
    reload      = os.getenv("RELOAD", "false").lower() == "true"
    workers     = int(os.getenv("WORKERS", "1"))  # keep at 1 — RabbitMQ consumer is stateful
 
    # DB_URL guard — fail early before uvicorn starts
    if not os.environ.get("DB_URL"):
        logger.error("DB_URL environment variable is not set. Aborting.")
        sys.exit(1)
 
    logger.info("Starting server  host=%s port=%d config=%s", host, port, config_path)
    logger.info("RabbitMQ worker will start inside the FastAPI lifespan.")
 
    uv_config = uvicorn.Config(
        app="api:app",          # module:variable
        host=host,
        port=port,
        reload=reload,          # True only in dev (incompatible with workers > 1)
        workers=workers,
        log_level="info",
        access_log=True,
    )
 
    server = uvicorn.Server(uv_config)
 
    # forward SIGTERM → graceful shutdown (important in Docker / k8s)
    signal.signal(signal.SIGTERM, lambda *_: server.handle_exit(sig=signal.SIGTERM, frame=None))
 
    server.run()
 
 
if __name__ == "__main__":
    main()