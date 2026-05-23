import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional

from .database import DatabaseManager, GlobalDatabaseManager
from .config import USERS_DIR, GLOBAL_DIR
from .exceptions import DatabaseError

logger = logging.getLogger("parallel_memory.migration")


class MigrationManager:
    def __init__(self):
        self.users_dir = USERS_DIR
        self.global_dir = GLOBAL_DIR

    def migrate_user_data(self, user_id: str, dry_run: bool = False) -> dict:
        try:
            db = DatabaseManager(user_id)
            user_base = self.users_dir / user_id
            stats = {
                "user_id": user_id,
                "memories": 0,
                "recalls": 0,
                "timelines": 0,
                "feedback": 0,
                "errors": [],
                "dry_run": dry_run,
            }

            if not user_base.exists():
                logger.warning(f"User directory not found: {user_base}")
                return stats

            memories_dir = user_base / "memories"
            if memories_dir.exists():
                for memory_file in memories_dir.glob("memory_*.json"):
                    try:
                        data = json.loads(memory_file.read_text(encoding="utf-8"))
                        if not dry_run:
                            db.create_memory(
                                memory_id=data.get("memory_id"),
                                text=data.get("memory_text"),
                                emotion=data.get("emotion"),
                                confidence=data.get("confidence", 0.5)
                            )
                        stats["memories"] += 1
                    except Exception as e:
                        error_msg = f"Memory migration error {memory_file}: {e}"
                        logger.error(error_msg)
                        stats["errors"].append(error_msg)

            recalls_dir = user_base / "recalls"
            if recalls_dir.exists():
                for recall_file in recalls_dir.glob("recall_*.json"):
                    try:
                        data = json.loads(recall_file.read_text(encoding="utf-8"))
                        drift_info = data.get("drift", {})
                        if not dry_run:
                            db.create_recall(
                                memory_id=data.get("memory_id"),
                                text=data.get("recall_text"),
                                emotion=data.get("emotion"),
                                confidence=data.get("confidence", 0.5),
                                semantic_shift=drift_info.get("semantic_shift_score"),
                                emotion_shift=drift_info.get("emotion_shift_score"),
                                omitted_keywords=drift_info.get("omitted_keywords"),
                                added_keywords=drift_info.get("added_keywords")
                            )
                        stats["recalls"] += 1
                    except Exception as e:
                        error_msg = f"Recall migration error {recall_file}: {e}"
                        logger.error(error_msg)
                        stats["errors"].append(error_msg)

            timelines_dir = user_base / "timelines"
            if timelines_dir.exists():
                for timeline_file in timelines_dir.glob("timeline_*.json"):
                    try:
                        data = json.loads(timeline_file.read_text(encoding="utf-8"))
                        branches = data.get("branches", [])
                        description = "\n".join([b.get("summary", "") for b in branches])
                        if not dry_run:
                            db.create_timeline(
                                memory_id=data.get("memory_id"),
                                scenario=data.get("label", "AI Simulation"),
                                description=description,
                                branch_id=data.get("branch_id")
                            )
                        stats["timelines"] += 1
                    except Exception as e:
                        error_msg = f"Timeline migration error {timeline_file}: {e}"
                        logger.error(error_msg)
                        stats["errors"].append(error_msg)

            feedback_dir = user_base / "feedback"
            if feedback_dir.exists():
                for feedback_file in feedback_dir.glob("feedback_*.json"):
                    try:
                        data = json.loads(feedback_file.read_text(encoding="utf-8"))
                        if not dry_run:
                            db.create_feedback(
                                memory_id=data.get("memory_id"),
                                response_id=data.get("response_id", ""),
                                rating=data.get("rating", 0.5),
                                correction=data.get("correction"),
                                notes=data.get("notes")
                            )
                        stats["feedback"] += 1
                    except Exception as e:
                        error_msg = f"Feedback migration error {feedback_file}: {e}"
                        logger.error(error_msg)
                        stats["errors"].append(error_msg)

            logger.info(f"Migration stats for {user_id}: {stats}")
            return stats

        except Exception as e:
            logger.error(f"User migration failed: {e}")
            raise DatabaseError(f"Failed to migrate user {user_id}: {e}", db_path=str(USERS_DIR / user_id))

    def migrate_global_data(self, dry_run: bool = False) -> dict:
        try:
            db = GlobalDatabaseManager()
            stats = {
                "global_feedback": 0,
                "errors": [],
                "dry_run": dry_run,
            }

            feedback_file = self.global_dir / "feedback_aggregate.jsonl"
            if feedback_file.exists():
                for line in feedback_file.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        data = json.loads(line)
                        if not dry_run:
                            db.add_global_feedback(
                                rating=data.get("rating", 0.5),
                                correction=data.get("correction"),
                                category=None
                            )
                        stats["global_feedback"] += 1
                    except Exception as e:
                        error_msg = f"Global feedback migration error: {e}"
                        logger.error(error_msg)
                        stats["errors"].append(error_msg)

            logger.info(f"Global migration stats: {stats}")
            return stats

        except Exception as e:
            logger.error(f"Global migration failed: {e}")
            raise DatabaseError(f"Failed to migrate global data: {e}", db_path=str(GLOBAL_DIR))

    def archive_json_files(self, user_id: Optional[str] = None) -> int:
        archive_dir = Path("./data_archives")
        archive_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        archived_count = 0

        if user_id:
            user_base = self.users_dir / user_id
            if user_base.exists():
                for subdir in user_base.iterdir():
                    if subdir.is_dir():
                        for json_file in subdir.glob("*.json"):
                            archive_path = archive_dir / f"{user_id}_{timestamp}_{json_file.name}"
                            archive_path.write_text(json_file.read_text(encoding="utf-8"), encoding="utf-8")
                            archived_count += 1
        else:
            for user_dir in self.users_dir.iterdir():
                if user_dir.is_dir():
                    for subdir in user_dir.iterdir():
                        if subdir.is_dir():
                            for json_file in subdir.glob("*.json"):
                                rel_path = json_file.relative_to(self.users_dir)
                                archive_path = archive_dir / timestamp / rel_path
                                archive_path.parent.mkdir(parents=True, exist_ok=True)
                                archive_path.write_text(json_file.read_text(encoding="utf-8"), encoding="utf-8")
                                archived_count += 1

        logger.info(f"Archived {archived_count} JSON files to {archive_dir}")
        return archived_count

    def run_full_migration(self, users: Optional[list[str]] = None) -> dict:
        logger.info("Starting full migration from JSON to SQLite")

        if users is None:
            users = [d.name for d in self.users_dir.iterdir() if d.is_dir()]

        results = {
            "users": {},
            "global": {},
            "total_users": len(users),
        }

        for user_id in users:
            logger.info(f"Migrating user: {user_id}")
            try:
                results["users"][user_id] = self.migrate_user_data(user_id, dry_run=False)
            except Exception as e:
                logger.error(f"Failed to migrate user {user_id}: {e}")
                results["users"][user_id] = {"error": str(e)}

        logger.info("Migrating global data")
        results["global"] = self.migrate_global_data(dry_run=False)

        self.archive_json_files()

        logger.info("Migration complete")
        return results
