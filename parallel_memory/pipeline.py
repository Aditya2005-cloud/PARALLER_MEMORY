import logging
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

try:
    import torch
except Exception:
    torch = None

from .database import DatabaseManager
from .embeddings import EmbeddingManager
from .exceptions import ModelInferenceError, EmbeddingError, MemoryError

logger = logging.getLogger("parallel_memory.pipeline")

try:
    from .models_ml import MemoryDriftNN, ConfidenceScorer, EmotionClassifier, ModelStorage
except Exception:
    MemoryDriftNN = None
    ConfidenceScorer = None
    ModelStorage = None

    class EmotionClassifier:  # type: ignore[no-redef]
        emotions = ["joy", "sadness", "regret", "acceptance", "nostalgia", "anger", "shame", "pride"]


@dataclass
class DriftResult:
    semantic_shift_score: float
    emotion_shift_score: float
    omitted_keywords: list[str]
    added_keywords: list[str]
    emotion_distribution: dict


class PipelineEngine:
    _embedding_managers: dict[str, EmbeddingManager] = {}
    _loaded_models: dict[tuple[str, str], object] = {}

    def __init__(self, user_id: str, device: str = "cpu", model_version: str = "1.0"):
        self.user_id = user_id
        self.device = device
        self.model_version = model_version

        self.db = DatabaseManager(user_id)
        em_key = f"{device}"
        if em_key not in self._embedding_managers:
            self._embedding_managers[em_key] = EmbeddingManager(device=device)
        self.embedding_manager = self._embedding_managers[em_key]
        self.model_storage = ModelStorage() if ModelStorage is not None else None

        if torch is None or self.model_storage is None or MemoryDriftNN is None or ConfidenceScorer is None:
            self.drift_model = None
            self.confidence_model = None
            self.emotion_model = None
        else:
            self.drift_model = self._load_model("drift", MemoryDriftNN)
            self.confidence_model = self._load_model("confidence", ConfidenceScorer)
            self.emotion_model = self._load_model("emotion", EmotionClassifier)

    def _load_model(self, model_type: str, model_class):
        cache_key = (model_type, self.model_version)
        if cache_key in self._loaded_models:
            return self._loaded_models[cache_key]
        try:
            if model_type == "drift":
                model = self.model_storage.load_drift_model(self.model_version)
            elif model_type == "confidence":
                model = self.model_storage.load_confidence_model(self.model_version)
            elif model_type == "emotion":
                model = self.model_storage.load_emotion_model(self.model_version)
            else:
                model = model_class()
            self._loaded_models[cache_key] = model
            return model
        except Exception as e:
            logger.warning(f"Failed to load {model_type} model: {e}, creating fresh")
            model = model_class()
            self._loaded_models[cache_key] = model
            return model

    def embed_text(self, text: str) -> np.ndarray:
        try:
            embedding = self.embedding_manager.embed_text(text)
            return embedding
        except EmbeddingError as e:
            logger.error(f"Embedding failed: {e}")
            raise

    def detect_memory_drift(self, memory_id: str, recall_text: str) -> DriftResult:
        try:
            memory = self.db.get_memory(memory_id)
            if not memory:
                raise MemoryError(f"Memory not found: {memory_id}", memory_id)

            prev_embedding_bytes = self.db.get_embedding(memory_id)
            if prev_embedding_bytes:
                prev_embedding = self.embedding_manager.deserialize_embedding(prev_embedding_bytes)
            else:
                prev_embedding = self.embed_text(memory["text"])
                serialized = self.embedding_manager.serialize_embedding(prev_embedding)
                self.db.store_embedding(memory_id, serialized, self.model_version)

            curr_embedding = self.embed_text(recall_text)

            if torch is None or self.drift_model is None:
                semantic_drift = float(1.0 - self.embedding_manager.compute_similarity(prev_embedding, curr_embedding))
                emotion_drift = min(1.0, semantic_drift * 0.8)
            else:
                with torch.no_grad():
                    prev_tensor = torch.tensor(prev_embedding, dtype=torch.float32).unsqueeze(0)
                    curr_tensor = torch.tensor(curr_embedding, dtype=torch.float32).unsqueeze(0)
                    semantic_drift, emotion_drift = self.drift_model(prev_tensor[0], curr_tensor[0])

            emotion_distribution = self._classify_emotion(curr_embedding)

            omitted_kw = self._extract_omitted_keywords(memory["text"], recall_text)
            added_kw = self._extract_added_keywords(memory["text"], recall_text)

            return DriftResult(
                semantic_shift_score=round(semantic_drift, 4),
                emotion_shift_score=round(emotion_drift, 4),
                omitted_keywords=omitted_kw,
                added_keywords=added_kw,
                emotion_distribution=emotion_distribution
            )

        except Exception as e:
            logger.error(f"Drift detection failed: {e}")
            raise ModelInferenceError(f"Failed to detect drift: {e}", model_type="drift")

    def estimate_confidence(self, memory_id: str) -> float:
        try:
            memory = self.db.get_memory(memory_id)
            if not memory:
                return 0.5

            embedding_bytes = self.db.get_embedding(memory_id)
            embedding = self.embedding_manager.deserialize_embedding(embedding_bytes) if embedding_bytes else np.zeros(384)

            recalls = self.db.get_recalls(memory_id)
            recall_count = len(recalls)

            created_at = memory.get("created_at", "")
            days_old = self._days_since(created_at)

            text_length = len(memory.get("text", "").split())

            if torch is None or self.confidence_model is None:
                base = 0.75
                penalty = min(0.5, recall_count * 0.03 + days_old * 0.002)
                confidence = max(0.1, base - penalty)
            else:
                with torch.no_grad():
                    embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
                    confidence = self.confidence_model(embedding_tensor[0], recall_count, days_old, text_length)

            return round(min(1.0, max(0.0, confidence)), 4)

        except Exception as e:
            logger.error(f"Confidence estimation failed: {e}")
            return 0.5

    def _classify_emotion(self, embedding: np.ndarray) -> dict:
        try:
            if torch is None or self.emotion_model is None:
                return {e: 1.0 / len(EmotionClassifier.emotions) for e in EmotionClassifier.emotions}
            with torch.no_grad():
                embedding_tensor = torch.tensor(embedding, dtype=torch.float32).unsqueeze(0)
                emotion_dist = self.emotion_model(embedding_tensor[0])
                return emotion_dist
        except Exception as exc:
            logger.warning(f"Emotion classification failed: {exc}")
            return {e: 1.0 / len(EmotionClassifier.emotions) for e in EmotionClassifier.emotions}

    def _extract_omitted_keywords(self, original: str, recall: str) -> list[str]:
        orig_words = set(self._tokenize(original))
        recall_words = set(self._tokenize(recall))
        omitted = sorted(list(orig_words - recall_words))[:10]
        return omitted

    def _extract_added_keywords(self, original: str, recall: str) -> list[str]:
        orig_words = set(self._tokenize(original))
        recall_words = set(self._tokenize(recall))
        added = sorted(list(recall_words - orig_words))[:10]
        return added

    def _tokenize(self, text: str) -> list[str]:
        tokens = [t.strip(".,!?;:()[]{}\"'").lower() for t in text.split()]
        return [t for t in tokens if len(t) > 3]

    def _days_since(self, iso_datetime: str) -> int:
        try:
            from datetime import datetime
            created = datetime.fromisoformat(iso_datetime.replace("Z", "+00:00"))
            now = datetime.now(created.tzinfo)
            return (now - created).days
        except:
            return 0

    def generate_alternate_timeline(self, memory_id: str, recall_text: str) -> dict:
        try:
            memory = self.db.get_memory(memory_id)
            if not memory:
                raise MemoryError(f"Memory not found: {memory_id}", memory_id)

            scenarios = [
                {
                    "branch": "Career Trajectory",
                    "scenario": "Different Professional Path",
                    "description": f"If the opposite choice was made in '{memory['text'][:50]}...', your career might have shifted significantly. Different skill networks, job opportunities, and professional identity could have emerged.",
                    "emotional_impact": "Mixed - potential regret vs. new opportunities"
                },
                {
                    "branch": "Emotional Consequences",
                    "scenario": "Internal Emotional State",
                    "description": f"This alternate timeline might show reduced regret in one area but increased uncertainty in another. Your emotional trajectory could have taken a fundamentally different shape.",
                    "emotional_impact": "Unpredictable - emotions are path-dependent"
                },
                {
                    "branch": "Relationships",
                    "scenario": "Social & Relationship Network",
                    "description": f"Different geographic locations, routines, and social circles might have stemmed from the alternate choice. Your current relationships might not exist, and entirely new ones could have formed.",
                    "emotional_impact": "Profound - relationships shape life meaning"
                }
            ]

            for scenario in scenarios:
                self.db.create_timeline(
                    memory_id=memory_id,
                    scenario=scenario["scenario"],
                    description=scenario["description"],
                    branch_id=scenario["branch"],
                    emotional_impact=scenario["emotional_impact"],
                    career_trajectory="See description" if scenario["branch"] == "Career Trajectory" else None,
                    relationship_changes="See description" if scenario["branch"] == "Relationships" else None,
                    model_version=self.model_version
                )

            return {
                "label": "AI Simulation (Not Objective Truth)",
                "memory_id": memory_id,
                "branches": scenarios,
                "safety_note": "This output is speculative and probabilistic. These timelines explore possibilities, not predict reality.",
                "model_version": self.model_version
            }

        except Exception as e:
            logger.error(f"Timeline generation failed: {e}")
            raise ModelInferenceError(f"Failed to generate timelines: {e}", model_type="timeline")

    def process_recall(self, memory_id: str, recall_text: str, emotion: Optional[str] = None,
                      confidence: float = 0.5) -> dict:
        try:
            drift = self.detect_memory_drift(memory_id, recall_text)

            recall_embedding = self.embed_text(recall_text)
            serialized_recall_emb = self.embedding_manager.serialize_embedding(recall_embedding)

            recall_row = self.db.create_recall(
                memory_id=memory_id,
                text=recall_text,
                emotion=emotion,
                confidence=confidence,
                semantic_shift=drift.semantic_shift_score,
                emotion_shift=drift.emotion_shift_score,
                omitted_keywords=drift.omitted_keywords,
                added_keywords=drift.added_keywords,
                emotion_distribution=drift.emotion_distribution
            )

            timeline = self.generate_alternate_timeline(memory_id, recall_text)

            return {
                "memory_id": memory_id,
                "recall_id": recall_row["id"],
                "drift": {
                    "semantic_shift": drift.semantic_shift_score,
                    "emotion_shift": drift.emotion_shift_score,
                    "omitted_keywords": drift.omitted_keywords,
                    "added_keywords": drift.added_keywords,
                    "emotion_distribution": drift.emotion_distribution
                },
                "timeline": timeline
            }

        except Exception as e:
            logger.error(f"Recall processing failed: {e}")
            raise
