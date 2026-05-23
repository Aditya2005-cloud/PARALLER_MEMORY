import sqlite3
import logging
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional, Any
from uuid import uuid4
import json

from .config import DATA_DIR, USERS_DIR, ensure_base_dirs

logger = logging.getLogger("parallel_memory.database")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class DatabaseManager:
    def __init__(self, user_id: Optional[str] = None):
        ensure_base_dirs()
        self.user_id = user_id
        if user_id:
            self.db_path = USERS_DIR / f"{user_id}.db"
        else:
            self.db_path = DATA_DIR / "global.db"
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @contextmanager
    def get_db(self) -> Iterator[sqlite3.Connection]:
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Database error: {e}")
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self.get_db() as db:
            cursor = db.cursor()

            if self.user_id:
                cursor.executescript("""
                    CREATE TABLE IF NOT EXISTS memories (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        text TEXT NOT NULL,
                        emotion TEXT,
                        confidence REAL DEFAULT 0.5,
                        model_version TEXT DEFAULT '1.0',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS embeddings (
                        id TEXT PRIMARY KEY,
                        memory_id TEXT NOT NULL UNIQUE,
                        vector BLOB NOT NULL,
                        model_version TEXT DEFAULT '1.0',
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
                    );

                    CREATE TABLE IF NOT EXISTS recalls (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        memory_id TEXT NOT NULL,
                        text TEXT NOT NULL,
                        emotion TEXT,
                        confidence REAL DEFAULT 0.5,
                        semantic_shift REAL,
                        emotion_shift REAL,
                        omitted_keywords TEXT,
                        added_keywords TEXT,
                        emotion_distribution TEXT,
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
                    );

                    CREATE TABLE IF NOT EXISTS timelines (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        memory_id TEXT NOT NULL,
                        branch_id TEXT,
                        scenario TEXT,
                        description TEXT,
                        emotional_impact TEXT,
                        career_trajectory TEXT,
                        relationship_changes TEXT,
                        confidence REAL,
                        model_version TEXT DEFAULT '1.0',
                        created_at TEXT NOT NULL,
                        FOREIGN KEY (memory_id) REFERENCES memories(id) ON DELETE CASCADE
                    );

                    CREATE TABLE IF NOT EXISTS feedback (
                        id TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        memory_id TEXT NOT NULL,
                        response_id TEXT,
                        rating REAL,
                        correction TEXT,
                        notes TEXT,
                        is_resolved INTEGER DEFAULT 0,
                        created_at TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS exceptions (
                        id TEXT PRIMARY KEY,
                        user_id TEXT,
                        function_name TEXT,
                        error_message TEXT,
                        traceback TEXT,
                        status TEXT DEFAULT 'unresolved',
                        created_at TEXT NOT NULL
                    );

                    CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id);
                    CREATE INDEX IF NOT EXISTS idx_recalls_memory ON recalls(memory_id);
                    CREATE INDEX IF NOT EXISTS idx_recalls_user ON recalls(user_id);
                    CREATE INDEX IF NOT EXISTS idx_timelines_memory ON timelines(memory_id);
                    CREATE INDEX IF NOT EXISTS idx_feedback_memory ON feedback(memory_id);
                    CREATE INDEX IF NOT EXISTS idx_feedback_user ON feedback(user_id);
                """)
            else:
                cursor.executescript("""
                    CREATE TABLE IF NOT EXISTS global_feedback (
                        id TEXT PRIMARY KEY,
                        rating REAL,
                        correction TEXT,
                        category TEXT,
                        count INTEGER DEFAULT 1,
                        model_version TEXT DEFAULT '1.0',
                        last_updated TEXT NOT NULL
                    );

                    CREATE TABLE IF NOT EXISTS retrain_log (
                        id TEXT PRIMARY KEY,
                        trigger_reason TEXT,
                        timestamp TEXT,
                        model_version TEXT,
                        performance_delta REAL,
                        samples_count INTEGER
                    );

                    CREATE INDEX IF NOT EXISTS idx_retrain_timestamp ON retrain_log(timestamp);
                """)

    def create_memory(self, memory_id: str, text: str, emotion: Optional[str] = None,
                     confidence: float = 0.5, model_version: str = "1.0") -> dict:
        with self.get_db() as db:
            cursor = db.cursor()
            now = utc_now()
            cursor.execute("""
                INSERT INTO memories (id, user_id, text, emotion, confidence, model_version, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (memory_id, self.user_id, text, emotion, confidence, model_version, now, now))
            return {
                "id": memory_id,
                "user_id": self.user_id,
                "text": text,
                "emotion": emotion,
                "confidence": confidence,
                "model_version": model_version,
                "created_at": now,
                "updated_at": now,
            }

    def get_memory(self, memory_id: str) -> Optional[dict]:
        with self.get_db() as db:
            cursor = db.cursor()
            cursor.execute("SELECT * FROM memories WHERE id = ? AND user_id = ?", (memory_id, self.user_id))
            row = cursor.fetchone()
            return dict(row) if row else None

    def list_memories(self, limit: int = 100, offset: int = 0) -> list[dict]:
        with self.get_db() as db:
            cursor = db.cursor()
            cursor.execute(
                "SELECT * FROM memories WHERE user_id = ? ORDER BY created_at DESC LIMIT ? OFFSET ?",
                (self.user_id, limit, offset)
            )
            return [dict(row) for row in cursor.fetchall()]

    def store_embedding(self, memory_id: str, vector: bytes, model_version: str = "1.0") -> dict:
        with self.get_db() as db:
            cursor = db.cursor()
            embedding_id = str(uuid4())
            now = utc_now()
            cursor.execute("""
                INSERT OR REPLACE INTO embeddings (id, memory_id, vector, model_version, created_at)
                VALUES (?, ?, ?, ?, ?)
            """, (embedding_id, memory_id, vector, model_version, now))
            return {"id": embedding_id, "memory_id": memory_id, "created_at": now}

    def get_embedding(self, memory_id: str) -> Optional[bytes]:
        with self.get_db() as db:
            cursor = db.cursor()
            cursor.execute("SELECT vector FROM embeddings WHERE memory_id = ?", (memory_id,))
            row = cursor.fetchone()
            return row[0] if row else None

    def create_recall(self, memory_id: str, text: str, emotion: Optional[str] = None,
                     confidence: float = 0.5, semantic_shift: Optional[float] = None,
                     emotion_shift: Optional[float] = None,
                     omitted_keywords: Optional[list] = None,
                     added_keywords: Optional[list] = None,
                     emotion_distribution: Optional[dict] = None) -> dict:
        with self.get_db() as db:
            cursor = db.cursor()
            recall_id = str(uuid4())
            now = utc_now()
            cursor.execute("""
                INSERT INTO recalls (id, user_id, memory_id, text, emotion, confidence,
                                    semantic_shift, emotion_shift, omitted_keywords,
                                    added_keywords, emotion_distribution, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                recall_id, self.user_id, memory_id, text, emotion, confidence,
                semantic_shift, emotion_shift,
                json.dumps(omitted_keywords or []),
                json.dumps(added_keywords or []),
                json.dumps(emotion_distribution or {}),
                now
            ))
            return {
                "id": recall_id,
                "user_id": self.user_id,
                "memory_id": memory_id,
                "text": text,
                "emotion": emotion,
                "confidence": confidence,
                "semantic_shift": semantic_shift,
                "emotion_shift": emotion_shift,
                "created_at": now,
            }

    def get_recalls(self, memory_id: str) -> list[dict]:
        with self.get_db() as db:
            cursor = db.cursor()
            cursor.execute("""
                SELECT * FROM recalls WHERE memory_id = ? AND user_id = ?
                ORDER BY created_at DESC
            """, (memory_id, self.user_id))
            rows = []
            for row in cursor.fetchall():
                r = dict(row)
                r["omitted_keywords"] = json.loads(r["omitted_keywords"]) if r["omitted_keywords"] else []
                r["added_keywords"] = json.loads(r["added_keywords"]) if r["added_keywords"] else []
                r["emotion_distribution"] = json.loads(r["emotion_distribution"]) if r["emotion_distribution"] else {}
                rows.append(r)
            return rows

    def create_timeline(self, memory_id: str, scenario: str, description: str,
                       branch_id: Optional[str] = None, emotional_impact: Optional[str] = None,
                       career_trajectory: Optional[str] = None,
                       relationship_changes: Optional[str] = None,
                       confidence: Optional[float] = None,
                       model_version: str = "1.0") -> dict:
        with self.get_db() as db:
            cursor = db.cursor()
            timeline_id = str(uuid4())
            now = utc_now()
            cursor.execute("""
                INSERT INTO timelines (id, user_id, memory_id, branch_id, scenario,
                                      description, emotional_impact, career_trajectory,
                                      relationship_changes, confidence, model_version, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                timeline_id, self.user_id, memory_id, branch_id or str(uuid4()),
                scenario, description, emotional_impact,
                career_trajectory, relationship_changes, confidence, model_version, now
            ))
            return {"id": timeline_id, "memory_id": memory_id, "created_at": now}

    def get_timelines(self, memory_id: str) -> list[dict]:
        with self.get_db() as db:
            cursor = db.cursor()
            cursor.execute("""
                SELECT * FROM timelines WHERE memory_id = ? AND user_id = ?
                ORDER BY created_at DESC
            """, (memory_id, self.user_id))
            return [dict(row) for row in cursor.fetchall()]

    def create_feedback(self, memory_id: str, response_id: str, rating: float,
                       correction: Optional[str] = None, notes: Optional[str] = None) -> dict:
        with self.get_db() as db:
            cursor = db.cursor()
            feedback_id = str(uuid4())
            now = utc_now()
            cursor.execute("""
                INSERT INTO feedback (id, user_id, memory_id, response_id, rating, correction, notes, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (feedback_id, self.user_id, memory_id, response_id, rating, correction, notes, now))
            return {
                "id": feedback_id,
                "user_id": self.user_id,
                "memory_id": memory_id,
                "response_id": response_id,
                "rating": rating,
                "created_at": now,
            }

    def get_feedback_stats(self) -> dict:
        with self.get_db() as db:
            cursor = db.cursor()
            cursor.execute("SELECT COUNT(*) as total, AVG(rating) as avg_rating FROM feedback WHERE user_id = ?", (self.user_id,))
            row = cursor.fetchone()
            return {"total": row[0], "avg_rating": row[1]} if row else {"total": 0, "avg_rating": 0}

    def log_exception(self, function_name: str, error_message: str, traceback: str) -> dict:
        with self.get_db() as db:
            cursor = db.cursor()
            exc_id = str(uuid4())
            now = utc_now()
            cursor.execute("""
                INSERT INTO exceptions (id, user_id, function_name, error_message, traceback, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (exc_id, self.user_id, function_name, error_message, traceback, now))
            return {"id": exc_id, "created_at": now}

    def get_unresolved_exceptions(self, limit: int = 50) -> list[dict]:
        with self.get_db() as db:
            cursor = db.cursor()
            cursor.execute("""
                SELECT * FROM exceptions WHERE user_id = ? AND status = 'unresolved'
                ORDER BY created_at DESC LIMIT ?
            """, (self.user_id, limit))
            return [dict(row) for row in cursor.fetchall()]


class GlobalDatabaseManager:
    def __init__(self):
        ensure_base_dirs()
        self.db_path = DATA_DIR / "global.db"
        self._init_schema()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @contextmanager
    def get_db(self) -> Iterator[sqlite3.Connection]:
        conn = self._get_connection()
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            logger.error(f"Global database error: {e}")
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self.get_db() as db:
            cursor = db.cursor()
            cursor.executescript("""
                CREATE TABLE IF NOT EXISTS global_feedback (
                    id TEXT PRIMARY KEY,
                    rating REAL,
                    correction TEXT,
                    category TEXT,
                    count INTEGER DEFAULT 1,
                    model_version TEXT DEFAULT '1.0',
                    last_updated TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS retrain_log (
                    id TEXT PRIMARY KEY,
                    trigger_reason TEXT,
                    timestamp TEXT,
                    model_version TEXT,
                    performance_delta REAL,
                    samples_count INTEGER
                );

                CREATE INDEX IF NOT EXISTS idx_retrain_timestamp ON retrain_log(timestamp);
                CREATE INDEX IF NOT EXISTS idx_feedback_rating ON global_feedback(rating);
            """)

    def add_global_feedback(self, rating: float, correction: Optional[str] = None, category: Optional[str] = None) -> dict:
        with self.get_db() as db:
            cursor = db.cursor()
            feedback_id = str(uuid4())
            now = utc_now()
            cursor.execute("""
                INSERT INTO global_feedback (id, rating, correction, category, last_updated)
                VALUES (?, ?, ?, ?, ?)
            """, (feedback_id, rating, correction, category, now))
            return {"id": feedback_id, "created_at": now}

    def get_negative_feedback(self, limit: int = 100) -> list[dict]:
        with self.get_db() as db:
            cursor = db.cursor()
            cursor.execute("""
                SELECT * FROM global_feedback WHERE rating < 0.5
                ORDER BY last_updated DESC LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_feedback_stats(self) -> dict:
        with self.get_db() as db:
            cursor = db.cursor()
            cursor.execute("""
                SELECT
                    COUNT(*) as total,
                    AVG(rating) as avg_rating,
                    SUM(CASE WHEN rating < 0.5 THEN 1 ELSE 0 END) as negative_count
                FROM global_feedback
            """)
            row = cursor.fetchone()
            return {
                "total": row[0],
                "avg_rating": row[1] if row[1] else 0,
                "negative_count": row[2] if row[2] else 0,
            }

    def log_retrain(self, trigger_reason: str, model_version: str, performance_delta: Optional[float] = None,
                   samples_count: Optional[int] = None) -> dict:
        with self.get_db() as db:
            cursor = db.cursor()
            retrain_id = str(uuid4())
            now = utc_now()
            cursor.execute("""
                INSERT INTO retrain_log (id, trigger_reason, timestamp, model_version, performance_delta, samples_count)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (retrain_id, trigger_reason, now, model_version, performance_delta, samples_count))
            return {"id": retrain_id, "timestamp": now}

    def get_retrain_history(self, limit: int = 50) -> list[dict]:
        with self.get_db() as db:
            cursor = db.cursor()
            cursor.execute("""
                SELECT * FROM retrain_log ORDER BY timestamp DESC LIMIT ?
            """, (limit,))
            return [dict(row) for row in cursor.fetchall()]
