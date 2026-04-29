"""
__main__.py — allows:
    python -m embedding          → starts API
    python -m embedding worker   → starts RabbitMQ worker
"""
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "api"

    if mode == "api":
        from app.embedding.main import main as api_main
        api_main()
    elif mode == "worker":
        from app.embedding.worker_main import main as worker_main
        worker_main()
    else:
        print(f"Unknown mode '{mode}'. Use: api | worker")
        sys.exit(1)


main()