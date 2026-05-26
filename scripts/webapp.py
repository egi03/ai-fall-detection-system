"""Launch the fall-detection web demo on http://localhost:8000.

Usage:
    python scripts/webapp.py
    python scripts/webapp.py --host 0.0.0.0 --port 8080
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> None:
    parser = argparse.ArgumentParser(description="Fall detection web demo")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable uvicorn reload (dev)")
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(
        "src.webapp.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level="info",
    )


if __name__ == "__main__":
    main()
