from __future__ import annotations

import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag.server import build_server  # noqa: E402


def main() -> None:
    port = 8899
    server = build_server("127.0.0.1", port, PROJECT_ROOT)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.2)
    try:
        payload = {
            "chat_id": "http-smoke",
            "messages": [{"type": "text", "content": "یک ماگ فانتزی برای نوجوان می‌خواهم"}],
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/chat",
            data=body,
            headers={"Content-Type": "application/json; charset=utf-8"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as response:
            data = json.loads(response.read().decode("utf-8"))
        assert data["base_random_keys"], data
        print(json.dumps({"status": "ok", "endpoint": "/chat"}, ensure_ascii=False))
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
