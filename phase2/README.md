# Agent Service

This service is the conversational layer of Torob Agentic RAG and consumes the retrieval artifacts in:

```text
..\phase1\torob_rag\artifacts\clean\unified_base_products.parquet
..\phase1\torob_rag\artifacts\clean\members_clean.parquet
..\phase1\torob_rag\artifacts\embeddings\clip_v1\image\image_index.parquet
```

The focused design uses one agent, one RAG tool, short-term memory by `chat_id`, and one `/chat` endpoint.

## Run

From the Phase 2 folder:

```powershell
cd "<PROJECT_ROOT>\phase2"
$env:PHASE1_ARTIFACTS_DIR = "..\phase1\torob_rag\artifacts"
python .\run_server.py --host 127.0.0.1 --port 8000
```

Endpoint:

```text
POST /chat
Content-Type: application/json
```

Request:

```json
{
  "chat_id": "demo-1",
  "messages": [
    {"type": "text", "content": "یک ماگ فانتزی برای نوجوان می‌خواهم"}
  ]
}
```

Response:

```json
{
  "message": "بر اساس نتایج بازیابی‌شده...",
  "base_random_keys": ["..."],
  "member_random_keys": null
}
```

## What The Agent Does

- Calls the RAG tool for product, feature, comparison, seller, and image requests.
- Keeps short-term memory per `chat_id`.
- Stores simple semantic memory items for city, budget, style, and product intent.
- Uses real Phase 1 cleaned product/member data.
- Returns at most 10 base/member keys.
- Refuses prompt-injection requests asking for hidden/system context.

## Supported Scenarios

- Ask a feature of a specific product.
- Ask the cheapest/best seller for a product.
- Multi-turn seller hunt with memory.
- Compare products for a stated use case.
- Image message handling: returns one real base key from the Phase 1 image index. The image case is lightweight because the full CLIP/FAISS runtime is optional in the project spec.

## Presentation Demo

The presentation demo starts a temporary `/chat` server, sends the main evaluation scenarios, prints each request/response pair, and stops the server automatically:

```powershell
cd "<PROJECT_ROOT>\phase2"
python .\tests\presentation_demo.py
```

## Tests

Three test levels are included:

```powershell
# Direct Agent smoke test: product, feature, seller, and multi-turn memory
python .\tests\smoke_test.py

# HTTP smoke test: starts a temporary server and calls the real POST /chat endpoint
python .\tests\http_smoke_test.py

# Presentation test: runs and prints all major evaluation scenarios
python .\tests\presentation_demo.py
```

The first two commands print a JSON response with `"status": "ok"` when they
pass. The presentation test prints `Status: PASS` for every scenario and ends
with `All presentation scenarios passed.`

## Monitoring

Runtime files are created automatically:

```text
phase2/logs/events.jsonl
phase2/logs/metrics.json
phase2/runtime/memory.json
```

Metrics:

- Mean Time To Answer
- Tokens Per Message
- Memory Hit Rate

## Notes

No external LLM key is required for the deterministic fallback path. Optional OpenRouter-compatible configuration can be enabled through environment variables when bonus LLM rewriting is needed.
