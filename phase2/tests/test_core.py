from __future__ import annotations

import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from agentic_rag.schemas import ChatResponse  # noqa: E402
from agentic_rag.server import parse_request  # noqa: E402
from agentic_rag.text_utils import normalize_text  # noqa: E402


class RequestParsingTests(unittest.TestCase):
    def test_valid_text_request(self) -> None:
        request = parse_request(
            {
                "chat_id": "demo",
                "messages": [{"type": "text", "content": "یک ماگ می خواهم"}],
            }
        )
        self.assertEqual(request.chat_id, "demo")
        self.assertEqual(request.messages[0].type, "text")

    def test_empty_messages_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "messages"):
            parse_request({"chat_id": "demo", "messages": []})


class ResponseSchemaTests(unittest.TestCase):
    def test_result_keys_are_limited_to_ten(self) -> None:
        response = ChatResponse("ok", [str(i) for i in range(12)], None).to_dict()
        self.assertEqual(len(response["base_random_keys"]), 10)


class PersianTextTests(unittest.TestCase):
    def test_arabic_characters_are_normalized(self) -> None:
        self.assertEqual(normalize_text("ي ك"), "ی ک")


if __name__ == "__main__":
    unittest.main()
