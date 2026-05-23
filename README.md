# Parallel Memory

Parallel Memory is an AI/ML system for psychological memory analysis:
- tracks how a user’s memory changes over time
- detects narrative/emotional drift between original memory and later recall
- generates alternate timeline simulations
- learns from feedback across users while keeping user data separated

## 1. Current Project Status

Implemented now:
- Multi-user backend scaffold with FastAPI
- Per-user isolated storage
- Global anonymized feedback aggregation
- Drift-analysis + timeline pipeline placeholders
- Retrain trigger candidate builder
- Blackboard memory protocol files
- `UNIFIED_LOG.json` orchestration brain

Planned next:
- Hugging Face routed LLM stack (fast + strong model)
- RAG retrieval layer
- Guardrails/safety layer
- PostgreSQL + vector DB migration

## 2. Repository Structure

```text
PARALLER_MEMORY/
  parallel_memory/
    api.py
    config.py
    models.py
    pipeline.py
    retraining.py
    storage.py
  data/
    users/
    global/
  blackboard/
    UNIFIED_LOG.json
    ...
  architecture.md
  requirements.txt
```

## 3. Data/Infra Design

### Per-user storage (isolated)
- Path: `data/users/<user_id>/`
- Files:
  - `memories/memory_<id>.json`
  - `recalls/recall_<memory_id>_<timestamp>.json`
  - `timelines/timeline_<memory_id>_<timestamp>.json`
  - `feedback/feedback_<timestamp>.json`

### Global learning aggregation
- Path: `data/global/`
- Files:
  - `feedback_aggregate.jsonl` (anonymized learning stream)
  - `retrain_candidates.jsonl` (negative samples for retrain prep)

## 4. Install and Run (Local)

## Requirements
- Python 3.10+

## Setup
```bash
pip install -r requirements.txt
```

## Start API
```bash
uvicorn parallel_memory.api:app --reload --port 8000
```

Open docs:
- `http://127.0.0.1:8000/docs`

## 5. End-to-End API Flow

## Step 1: Store original memory
```bash
curl -X POST "http://127.0.0.1:8000/memories" \
  -H "Content-Type: application/json" \
  -d "{\"user_id\":\"user_123\",\"memory_text\":\"In 2022 I rejected a music scholarship for engineering.\",\"emotion\":\"conflict\",\"confidence\":0.72,\"metadata\":{\"year\":2022}}"
```

Response returns `memory_id`.

## Step 2: Submit later recall and generate analysis
```bash
curl -X POST "http://127.0.0.1:8000/recalls" \
  -H "Content-Type: application/json" \
  -d "{\"user_id\":\"user_123\",\"memory_id\":\"<memory_id>\",\"recall_text\":\"I was forced to reject music and felt regret.\",\"emotion\":\"regret\",\"confidence\":0.55}"
```

Response includes:
- drift scores
- added/omitted keywords
- alternate timeline simulation object

## Step 3: Log feedback
```bash
curl -X POST "http://127.0.0.1:8000/feedback" \
  -H "Content-Type: application/json" \
  -d "{\"user_id\":\"user_123\",\"memory_id\":\"<memory_id>\",\"response_id\":\"resp_001\",\"rating\":0,\"correction\":\"I was uncertain, not fully forced.\",\"notes\":\"Tone too absolute.\"}"
```

This stores user feedback and appends anonymized global learning row.

## Step 4: Check global learning status
```bash
curl "http://127.0.0.1:8000/global/summary"
```

## Step 5: Build retrain candidates
```bash
curl -X POST "http://127.0.0.1:8000/global/retrain-check?min_negative=20"
```

## 6. Available Endpoints

- `GET /health`
- `POST /memories`
- `POST /recalls`
- `POST /feedback`
- `GET /global/summary`
- `POST /global/retrain-check`

## 7. Blackboard System

Two orchestration layers now exist:

1. Structured blackboard folders (`blackboard/history`, `state`, `agents`, ...)
2. Unified protocol file:
   - `blackboard/UNIFIED_LOG.json`

`UNIFIED_LOG.json` tracks:
- meta summary and health
- status board
- tasks
- checkpoints
- knowledge
- improvements
- failures
- event log

If you resume work later, start by reading `blackboard/UNIFIED_LOG.json`.

## 8. Notebook Notes

Notebook file:
- `parallel_memory_feedback_loop.ipynb`

Recent fixes already applied:
- corruption/merge-marker cleanup
- removed mandatory Google Drive dependency
- Hugging Face loading fallback for quantized/non-quantized model loading
- optional GitHub artifact save flow

## 9. What Is Placeholder vs Production

Currently placeholder:
- drift scoring heuristic in `parallel_memory/pipeline.py`
- alternate timeline generation in `parallel_memory/pipeline.py`

Production direction:
- sentence embeddings + learned drift model
- routed Hugging Face LLMs
- RAG memory retrieval
- guardrails and confidence calibration

## 10. Next Implementation Plan

1. Implement `T-003`: Hugging Face routed LLM + RAG integration.
2. Add retrieval index for user memories.
3. Add guardrails policy checks and simulation labeling.
4. Move persistence from file-based JSON/JSONL to PostgreSQL + vector DB.
5. Add test coverage for router and feedback-learning path.

## 11. Quick Verification Script (Optional)

```python
from parallel_memory.storage import create_memory, save_feedback
from parallel_memory.pipeline import process_recall_and_simulation
from parallel_memory.retraining import global_summary

m = create_memory("demo_user", {
    "memory_text": "I declined an art scholarship in 2021.",
    "emotion": "conflicted",
    "confidence": 0.7,
    "metadata": {"year": 2021}
})

result = process_recall_and_simulation(
    "demo_user",
    m["memory_id"],
    "I felt pressured and still regret it.",
    "regret",
    0.5
)

save_feedback("demo_user", {
    "memory_id": m["memory_id"],
    "response_id": "resp_demo",
    "rating": 0,
    "correction": "Confidence should be lower."
})

print(result["drift"])
print(global_summary())
```

## 12. Ethics Defaults

- Simulations must be labeled as speculative.
- System should not claim objective truth about memory.
- Confidence outputs are probabilistic.
- Keep user-level memory data isolated from other users.

<img width="324" height="511" alt="{A42E0AF6-C6A4-45B2-B460-BFB58084AF19}" src="https://github.com/user-attachments/assets/8f5abf37-8526-458f-912a-6baa9163faf7" />
