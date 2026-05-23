import json
from pathlib import Path

from .config import GLOBAL_DIR, ensure_base_dirs
from .storage import iter_feedback_aggregate


def build_retrain_candidates(min_negative: int = 20) -> dict:
    ensure_base_dirs()
    rows = iter_feedback_aggregate()
    negatives = [r for r in rows if int(r.get("rating", 1)) == 0]
    result = {
        "total_feedback": len(rows),
        "negative_feedback": len(negatives),
        "trigger_retrain": len(negatives) >= min_negative,
        "min_negative_threshold": min_negative,
    }

    out = GLOBAL_DIR / "retrain_candidates.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for row in negatives:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    return result


def global_summary() -> dict:
    ensure_base_dirs()
    rows = iter_feedback_aggregate()
    negative = sum(1 for r in rows if int(r.get("rating", 1)) == 0)
    return {
        "aggregate_file": str(Path(GLOBAL_DIR / "feedback_aggregate.jsonl")),
        "total_feedback": len(rows),
        "negative_feedback": negative,
        "negative_rate": round((negative / len(rows)), 4) if rows else 0.0,
    }

