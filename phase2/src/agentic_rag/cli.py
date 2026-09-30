from __future__ import annotations

import argparse
import uuid
from pathlib import Path

from .agent import Agent
from .schemas import ChatMessage, ChatRequest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chat-id", default=str(uuid.uuid4()))
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[3]
    agent = Agent(root)
    print(f"chat_id={args.chat_id}")
    print("Type /exit to stop.")
    while True:
        text = input("user> ").strip()
        if text in {"/exit", "exit", "quit"}:
            break
        response = agent.chat(
            ChatRequest(args.chat_id, [ChatMessage(type="text", content=text)])
        )
        print(response.to_dict())
