# Parallel Memory - Multi-User Data + Learning Pipeline

## Folder Structure

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
      .gitkeep
    global/
      .gitkeep
  requirements.txt
```

## Data Rules

- Each user has isolated data under: `data/users/<user_id>/`
- Shared improvement data lives in: `data/global/`
- User-specific memories never need to be exposed to other users.
- Global model-improvement dataset stores anonymized training rows only.

## Per-User Layout

```text
data/users/<user_id>/
  memories/
    memory_<memory_id>.json
  recalls/
    recall_<memory_id>_<timestamp>.json
  timelines/
    timeline_<memory_id>_<timestamp>.json
  feedback/
    feedback_<timestamp>.json
```

## Global Layout

```text
data/global/
  feedback_aggregate.jsonl
  retrain_candidates.jsonl
```

