from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag.agent import Agent  # noqa: E402
from agentic_rag.schemas import ChatMessage, ChatRequest  # noqa: E402


def ask(agent: Agent, chat_id: str, text: str):
    response = agent.chat(
        ChatRequest(chat_id=chat_id, messages=[ChatMessage(type="text", content=text)])
    )
    return response.to_dict()


def main() -> None:
    runtime = ROOT / "runtime"
    if runtime.exists():
        shutil.rmtree(runtime)

    agent = Agent(PROJECT_ROOT)

    product = ask(agent, "smoke-product", "یک ماگ فانتزی برای نوجوان می‌خواهم")
    assert product["base_random_keys"], product

    feature = ask(agent, "smoke-feature", "جنس ماگ هندوانه فانتزی و کارتونی کد 1375 چیست؟")
    assert feature["base_random_keys"], feature

    seller = ask(agent, "smoke-seller", "کمترین قیمت برای ماگ هندوانه فانتزی و کارتونی کد 1375 چقدر است؟")
    assert seller["base_random_keys"], seller
    assert seller["member_random_keys"], seller

    first = ask(
        agent,
        "smoke-hunt",
        "دنبال یه میز برای کارهای روزمره هستم. کمک می‌کنید فروشنده خوب پیدا کنم؟",
    )
    assert first["member_random_keys"] is None, first
    second = ask(agent, "smoke-hunt", "بودجه حدود ۱۰ میلیون دارم و مدل کشودار ساده می‌خواهم.")
    assert second["member_random_keys"], second

    print(json.dumps({"status": "ok"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
