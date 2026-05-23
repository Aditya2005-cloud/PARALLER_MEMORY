import pytest
import tempfile
from pathlib import Path
from uuid import uuid4

from parallel_memory.database import DatabaseManager, GlobalDatabaseManager


@pytest.fixture
def temp_db_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def user_db(temp_db_dir, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(temp_db_dir))
    yield DatabaseManager("test_user")


@pytest.fixture
def global_db(temp_db_dir, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(temp_db_dir))
    yield GlobalDatabaseManager()


class TestDatabaseManager:
    def test_create_memory(self, user_db):
        memory_id = str(uuid4())
        result = user_db.create_memory(memory_id, "Test memory", "joy", 0.75)

        assert result["id"] == memory_id
        assert result["text"] == "Test memory"
        assert result["emotion"] == "joy"
        assert result["confidence"] == 0.75

    def test_get_memory(self, user_db):
        memory_id = str(uuid4())
        user_db.create_memory(memory_id, "Test memory", "sadness", 0.5)

        retrieved = user_db.get_memory(memory_id)
        assert retrieved is not None
        assert retrieved["text"] == "Test memory"
        assert retrieved["emotion"] == "sadness"

    def test_list_memories(self, user_db):
        for i in range(5):
            user_db.create_memory(str(uuid4()), f"Memory {i}")

        memories = user_db.list_memories(limit=10)
        assert len(memories) == 5

    def test_store_and_get_embedding(self, user_db):
        memory_id = str(uuid4())
        user_db.create_memory(memory_id, "Test")

        test_vector = b"\x00" * 100
        user_db.store_embedding(memory_id, test_vector)

        retrieved = user_db.get_embedding(memory_id)
        assert retrieved == test_vector

    def test_create_recall(self, user_db):
        memory_id = str(uuid4())
        user_db.create_memory(memory_id, "Original")

        recall_id = user_db.create_recall(
            memory_id=memory_id,
            text="Recall",
            emotion="regret",
            confidence=0.4,
            semantic_shift=0.65,
            emotion_shift=0.72
        )["id"]

        recalls = user_db.get_recalls(memory_id)
        assert len(recalls) == 1
        assert recalls[0]["text"] == "Recall"
        assert recalls[0]["semantic_shift"] == 0.65

    def test_create_timeline(self, user_db):
        memory_id = str(uuid4())
        user_db.create_memory(memory_id, "Test")

        timeline = user_db.create_timeline(
            memory_id=memory_id,
            scenario="Career Branch",
            description="What if you took the job?"
        )

        assert timeline["memory_id"] == memory_id
        assert timeline["id"] is not None

    def test_create_feedback(self, user_db):
        memory_id = str(uuid4())
        user_db.create_memory(memory_id, "Test")

        feedback = user_db.create_feedback(
            memory_id=memory_id,
            response_id="resp_1",
            rating=0.25,
            correction="Too harsh"
        )

        assert feedback["rating"] == 0.25
        assert feedback["correction"] == "Too harsh"

    def test_feedback_stats(self, user_db):
        memory_id = str(uuid4())
        user_db.create_memory(memory_id, "Test")

        user_db.create_feedback(memory_id, "resp_1", 0.9)
        user_db.create_feedback(memory_id, "resp_2", 0.1)

        stats = user_db.get_feedback_stats()
        assert stats["total"] == 2
        assert 0.4 < stats["avg_rating"] < 0.6

    def test_log_exception(self, user_db):
        exc = user_db.log_exception(
            "test_function",
            "Test error message",
            "Traceback here"
        )

        assert exc["id"] is not None
        assert exc["created_at"] is not None


class TestGlobalDatabaseManager:
    def test_add_global_feedback(self, global_db):
        feedback = global_db.add_global_feedback(0.35, "Test correction")

        assert feedback["id"] is not None
        assert feedback["created_at"] is not None

    def test_get_negative_feedback(self, global_db):
        global_db.add_global_feedback(0.1)
        global_db.add_global_feedback(0.9)
        global_db.add_global_feedback(0.3)

        negative = global_db.get_negative_feedback()
        assert len(negative) >= 2

    def test_feedback_stats(self, global_db):
        global_db.add_global_feedback(0.2)
        global_db.add_global_feedback(0.8)

        stats = global_db.get_feedback_stats()
        assert stats["total"] == 2
        assert stats["avg_rating"] == 0.5

    def test_log_retrain(self, global_db):
        retrain = global_db.log_retrain("negative_feedback", "1.0", 0.05, 25)

        assert retrain["id"] is not None
        assert retrain["timestamp"] is not None

    def test_retrain_history(self, global_db):
        global_db.log_retrain("trigger_1", "1.0")
        global_db.log_retrain("trigger_2", "1.1")

        history = global_db.get_retrain_history()
        assert len(history) >= 2
