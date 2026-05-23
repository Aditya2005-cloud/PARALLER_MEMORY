import logging
import numpy as np
from typing import List, Optional
import pickle

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    SentenceTransformer = None

from .exceptions import EmbeddingError, ModelLoadError

logger = logging.getLogger("parallel_memory.embeddings")


class EmbeddingManager:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.embedding_dim = 384
        self._load_model()

    def _load_model(self):
        if SentenceTransformer is None:
            logger.error("sentence-transformers not installed")
            raise ModelLoadError("sentence-transformers not installed", fallback_available=True)

        try:
            self.model = SentenceTransformer(self.model_name, device=self.device)
            logger.info(f"Loaded embedding model: {self.model_name} on device: {self.device}")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise ModelLoadError(f"Failed to load {self.model_name}: {e}", fallback_available=True)

    def embed_text(self, text: str) -> np.ndarray:
        if self.model is None:
            logger.warning("Model not loaded, using fallback embedding")
            return self._fallback_embed(text)

        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding.astype(np.float32)
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            raise EmbeddingError(f"Failed to embed text: {e}", model_name=self.model_name)

    def embed_texts(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        if self.model is None:
            return [self._fallback_embed(t) for t in texts]

        try:
            embeddings = self.model.encode(texts, convert_to_numpy=True, batch_size=batch_size)
            return [e.astype(np.float32) for e in embeddings]
        except Exception as e:
            logger.error(f"Batch embedding error: {e}")
            raise EmbeddingError(f"Failed to embed texts: {e}", model_name=self.model_name)

    def _fallback_embed(self, text: str) -> np.ndarray:
        words = text.lower().split()
        word_hash = sum(hash(w) for w in words)
        rng = np.random.RandomState(word_hash % (2**31))
        return rng.randn(self.embedding_dim).astype(np.float32)

    def compute_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        if len(embedding1) == 0 or len(embedding2) == 0:
            return 0.0
        return float(np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2) + 1e-8))

    def compute_similarities_batch(self, embedding: np.ndarray, embeddings_list: List[np.ndarray]) -> List[float]:
        return [self.compute_similarity(embedding, e) for e in embeddings_list]

    def serialize_embedding(self, embedding: np.ndarray) -> bytes:
        return pickle.dumps(embedding)

    def deserialize_embedding(self, data: bytes) -> np.ndarray:
        return pickle.loads(data)
