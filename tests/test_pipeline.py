import pytest
import tempfile
from pathlib import Path
from uuid import uuid4

from parallel_memory.pipeline import PipelineEngine, DriftResult
from parallel_memory.database import DatabaseManager


@pytest.fixture
def temp_db_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def pipeline(temp_db_dir, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(temp_db_dir))
    return PipelineEngine("test_user", device="cpu")


class TestPipelineEngine:
    def test_embed_text(self, pipeline):
        text = "This is a test memory about a life decision."
        embedding = pipeline.embed_text(text)

        assert embedding is not None
        assert len(embedding) == 384
        assert embedding.dtype.name == "float32"

    def test_embed_multiple_texts(self, pipeline):
        texts = [
            "Memory one",
            "Memory two",
            "Memory three"
        ]

        embeddings = pipeline.embedding_manager.embed_texts(texts)
        assert len(embeddings) == 3
        assert all(len(e) == 384 for e in embeddings)

    def test_detect_memory_drift(self, pipeline):
        memory_id = str(uuid4())
        db = DatabaseManager("test_user")
        db.create_memory(memory_id, "I rejected the music scholarship for engineering")

        embedding = pipeline.embed_text("I rejected the music scholarship for engineering")
        db.store_embedding(memory_id, pipeline.embedding_manager.serialize_embedding(embedding))

        drift = pipeline.detect_memory_drift(
            memory_id,
            "I was forced to reject music by my family"
        )

        assert isinstance(drift, DriftResult)
        assert 0 <= drift.semantic_shift_score <= 1
        assert 0 <= drift.emotion_shift_score <= 1
        assert isinstance(drift.omitted_keywords, list)
        assert isinstance(drift.added_keywords, list)
        assert isinstance(drift.emotion_distribution, dict)

    def test_estimate_confidence(self, pipeline):
        memory_id = str(uuid4())
        db = DatabaseManager("test_user")
        db.create_memory(memory_id, "Test memory")

        embedding = pipeline.embed_text("Test memory")
        db.store_embedding(memory_id, pipeline.embedding_manager.serialize_embedding(embedding))

        confidence = pipeline.estimate_confidence(memory_id)

        assert 0 <= confidence <= 1

    def test_classify_emotion(self, pipeline):
        embedding_manager = pipeline.embedding_manager
        text = "I feel regret about my decision."

        embedding = embedding_manager.embed_text(text)
        emotion_dist = pipeline._classify_emotion(embedding)

        assert isinstance(emotion_dist, dict)
        assert len(emotion_dist) == 8
        assert abs(sum(emotion_dist.values()) - 1.0) < 0.01

    def test_tokenize(self, pipeline):
        text = "This is a test memory about decisions and regret."
        tokens = pipeline._tokenize(text)

        assert "this" in tokens
        assert "memory" in tokens
        assert "decisions" in tokens
        assert len(tokens) > 0

    def test_extract_keywords(self, pipeline):
        original = "I rejected a music scholarship for engineering."
        recall = "I was forced to reject music for family reasons."

        omitted = pipeline._extract_omitted_keywords(original, recall)
        added = pipeline._extract_added_keywords(original, recall)

        assert len(omitted) > 0
        assert len(added) > 0

    def test_generate_alternate_timeline(self, pipeline):
        memory_id = str(uuid4())
        db = DatabaseManager("test_user")
        db.create_memory(memory_id, "I turned down a startup opportunity")

        timeline = pipeline.generate_alternate_timeline(
            memory_id,
            "I accepted the startup instead"
        )

        assert timeline is not None
        assert "label" in timeline
        assert "branches" in timeline
        assert len(timeline["branches"]) == 3
        assert "safety_note" in timeline

        for branch in timeline["branches"]:
            assert "branch" in branch
            assert "scenario" in branch
            assert "description" in branch
            assert "emotional_impact" in branch

    def test_process_recall(self, pipeline):
        memory_id = str(uuid4())
        db = DatabaseManager("test_user")
        db.create_memory(memory_id, "I chose safety over passion")

        embedding = pipeline.embed_text("I chose safety over passion")
        db.store_embedding(memory_id, pipeline.embedding_manager.serialize_embedding(embedding))

        result = pipeline.process_recall(
            memory_id,
            "Now I wonder if I should have taken the risk",
            emotion="uncertainty",
            confidence=0.5
        )

        assert "memory_id" in result
        assert "recall_id" in result
        assert "drift" in result
        assert "timeline" in result
        assert result["drift"]["semantic_shift"] is not None
        assert result["drift"]["emotion_shift"] is not None

    def test_drift_detection_with_high_change(self, pipeline):
        memory_id = str(uuid4())
        db = DatabaseManager("test_user")
        db.create_memory(memory_id, "Original memory text")

        embedding = pipeline.embed_text("Original memory text")
        db.store_embedding(memory_id, pipeline.embedding_manager.serialize_embedding(embedding))

        drift = pipeline.detect_memory_drift(memory_id, "Completely different story")

        assert drift.semantic_shift_score > 0.3

    def test_days_since_calculation(self, pipeline):
        from datetime import datetime, timezone, timedelta

        now_iso = datetime.now(timezone.utc).isoformat()
        days = pipeline._days_since(now_iso)

        assert days == 0

        past_iso = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
        days = pipeline._days_since(past_iso)

        assert 6 <= days <= 8
