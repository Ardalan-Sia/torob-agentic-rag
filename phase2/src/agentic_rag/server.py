from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from .agent import Agent
from .schemas import ChatMessage, ChatRequest


INDEX_HTML = """<!doctype html>
<html lang="fa" dir="rtl">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Torob Agentic RAG</title>
  <style>
    body { font-family: Tahoma, Arial, sans-serif; margin: 32px; background: #f7f7f8; color: #202124; }
    main { max-width: 860px; margin: 0 auto; background: white; border: 1px solid #ddd; border-radius: 8px; padding: 24px; }
    h1 { margin-top: 0; font-size: 24px; }
    label { display: block; margin: 14px 0 6px; font-weight: 700; }
    input, textarea { width: 100%; box-sizing: border-box; padding: 10px; border: 1px solid #bbb; border-radius: 6px; font: inherit; }
    textarea { min-height: 110px; resize: vertical; }
    button { margin-top: 14px; padding: 10px 18px; border: 0; border-radius: 6px; background: #155eef; color: white; font: inherit; cursor: pointer; }
    button:disabled { opacity: .65; cursor: wait; }
    pre { direction: ltr; text-align: left; white-space: pre-wrap; background: #111827; color: #e5e7eb; padding: 16px; border-radius: 6px; overflow: auto; }
    .hint { color: #666; font-size: 14px; }
  </style>
</head>
<body>
  <main>
    <h1>Torob Agentic RAG</h1>
    <p class="hint">This page is only for manual testing. The judge still uses POST /chat.</p>
    <label for="chat_id">chat_id</label>
    <input id="chat_id" value="browser-test">
    <label for="message">message</label>
    <textarea id="message">کمترین قیمت برای ماگ هندوانه فانتزی و کارتونی کد 1375 چقدر است؟</textarea>
    <button id="send">Send to /chat</button>
    <h2>Response</h2>
    <pre id="output">Waiting...</pre>
  </main>
  <script>
    const send = document.getElementById("send");
    const output = document.getElementById("output");
    send.addEventListener("click", async () => {
      send.disabled = true;
      output.textContent = "Loading...";
      const payload = {
        chat_id: document.getElementById("chat_id").value || "browser-test",
        messages: [{ type: "text", content: document.getElementById("message").value }]
      };
      try {
        const res = await fetch("/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json; charset=utf-8" },
          body: JSON.stringify(payload)
        });
        const data = await res.json();
        output.textContent = JSON.stringify(data, null, 2);
      } catch (error) {
        output.textContent = String(error);
      } finally {
        send.disabled = false;
      }
    });
  </script>
</body>
</html>
"""


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def parse_request(payload: dict[str, Any]) -> ChatRequest:
    chat_id = payload.get("chat_id")
    messages = payload.get("messages")
    if not isinstance(chat_id, str) or not chat_id:
        raise ValueError("chat_id must be a non-empty string")
    if not isinstance(messages, list) or not messages:
        raise ValueError("messages must be a non-empty list")
    parsed: list[ChatMessage] = []
    for item in messages:
        if not isinstance(item, dict):
            raise ValueError("each message must be an object")
        msg_type = item.get("type")
        content = item.get("content")
        if msg_type not in {"text", "image"}:
            raise ValueError("message.type must be text or image")
        if not isinstance(content, str):
            raise ValueError("message.content must be a string")
        parsed.append(ChatMessage(type=msg_type, content=content))  # type: ignore[arg-type]
    return ChatRequest(chat_id=chat_id, messages=parsed)


class Handler(BaseHTTPRequestHandler):
    agent: Agent

    def do_GET(self) -> None:  # noqa: N802
        if self.path in {"/", "/index.html"}:
            self._html(INDEX_HTML)
        elif self.path == "/health":
            self._json({"status": "ok"})
        elif self.path == "/metrics":
            self._json(self.agent.monitor.metrics())
        else:
            self._json({"error": "not found"}, status=404)

    def do_POST(self) -> None:  # noqa: N802
        if self.path != "/chat":
            self._json({"error": "not found"}, status=404)
            return
        try:
            raw = self.rfile.read(int(self.headers.get("Content-Length", "0")))
            request = parse_request(json.loads(raw.decode("utf-8")))
            response = self.agent.chat(request)
            self._json(response.to_dict())
        except ValueError as exc:
            self._json({"error": str(exc)}, status=400)
        except Exception as exc:
            self.agent.monitor.event("server.error", {"error": str(exc)})
            self._json({"error": "internal server error"}, status=500)

    def log_message(self, fmt: str, *args: Any) -> None:
        self.agent.monitor.event("server.http", {"message": fmt % args})

    def _json(self, payload: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _html(self, html: str, status: int = 200) -> None:
        body = html.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def build_server(host: str, port: int, root: Path | None = None) -> ThreadingHTTPServer:
    Handler.agent = Agent(root or project_root())
    return ThreadingHTTPServer((host, port), Handler)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    server = build_server(args.host, args.port)
    print(f"Torob Agentic RAG is running at http://{args.host}:{args.port}")
    print("Use POST /chat. Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        server.server_close()
