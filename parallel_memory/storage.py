import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .config import GLOBAL_DIR, USERS_DIR, ensure_base_dirs


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _user_dirs(user_id: str) -> dict[str, Path]:
    base = USERS_DIR / user_id
    dirs = {
        "base": base,
        "memories": base / "memories",
        "recalls": base / "recalls",
        "timelines": base / "timelines",
        "feedback": base / "feedback",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def init_storage() -> None:
    ensure_base_dirs()


def create_memory(user_id: str, payload: dict) -> dict:
    init_storage()
    memory_id = str(uuid4())
    row = {
        "memory_id": memory_id,
        "user_id": user_id,
        "created_at": utc_now(),
        **payload,
    }
    dirs = _user_dirs(user_id)
    out = dirs["memories"] / f"memory_{memory_id}.json"
    out.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    return row


def get_memory(user_id: str, memory_id: str) -> dict:
    fp = USERS_DIR / user_id / "memories" / f"memory_{memory_id}.json"
    if not fp.exists():
        raise FileNotFoundError(f"Memory not found: {memory_id}")
    return json.loads(fp.read_text(encoding="utf-8"))


def create_recall(user_id: str, memory_id: str, payload: dict) -> dict:
    init_storage()
    ts = utc_now().replace(":", "-")
    row = {
        "user_id": user_id,
        "memory_id": memory_id,
        "created_at": utc_now(),
        **payload,
    }
    out = _user_dirs(user_id)["recalls"] / f"recall_{memory_id}_{ts}.json"
    out.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    return row


def create_timeline(user_id: str, memory_id: str, payload: dict) -> dict:
    init_storage()
    ts = utc_now().replace(":", "-")
    row = {
        "timeline_id": str(uuid4()),
        "user_id": user_id,
        "memory_id": memory_id,
        "created_at": utc_now(),
        **payload,
    }
    out = _user_dirs(user_id)["timelines"] / f"timeline_{memory_id}_{ts}.json"
    out.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")
    return row


def save_feedback(user_id: str, payload: dict) -> dict:
    init_storage()
    ts = utc_now().replace(":", "-")
    row = {
        "feedback_id": str(uuid4()),
        "user_id": user_id,
        "created_at": utc_now(),
        **payload,
    }
    out = _user_dirs(user_id)["feedback"] / f"feedback_{ts}.json"
    out.write_text(json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8")

    # Append anonymized row for global model improvement.
    global_line = {
        "feedback_id": row["feedback_id"],
        "created_at": row["created_at"],
        "memory_id": row.get("memory_id"),
        "response_id": row.get("response_id"),
        "rating": row.get("rating"),
        "correction": row.get("correction"),
        "notes": row.get("notes"),
    }
    agg = GLOBAL_DIR / "feedback_aggregate.jsonl"
    with agg.open("a", encoding="utf-8") as f:
        f.write(json.dumps(global_line, ensure_ascii=False) + "\n")
    return row


def iter_feedback_aggregate() -> list[dict]:
    fp = GLOBAL_DIR / "feedback_aggregate.jsonl"
    if not fp.exists():
        return []
    rows: list[dict] = []
    for line in fp.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rows.append(json.loads(line))
    return rows

