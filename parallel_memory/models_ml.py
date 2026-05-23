import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Tuple, Optional
from pathlib import Path

logger = logging.getLogger("parallel_memory.models_ml")


class MemoryDriftNN(nn.Module):
    def __init__(self, embedding_dim: int = 384, hidden_dim: int = 256, dropout: float = 0.2):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=2, dropout=dropout, batch_first=True)

        self.attention_weights = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

        self.drift_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Sigmoid()
        )

    def forward(self, prev_embedding: torch.Tensor, curr_embedding: torch.Tensor,
                prev_sequence: Optional[torch.Tensor] = None) -> Tuple[float, float]:
        if prev_sequence is not None:
            lstm_out, _ = self.lstm(prev_sequence)
            attention = F.softmax(self.attention_weights(lstm_out).squeeze(-1), dim=-1)
            context = torch.sum(lstm_out * attention.unsqueeze(-1), dim=1)
        else:
            context = prev_embedding

        combined = torch.cat([context, curr_embedding], dim=-1)
        drift_scores = self.drift_head(combined)
        return drift_scores[0].item(), drift_scores[1].item()


class ConfidenceScorer(nn.Module):
    def __init__(self, embedding_dim: int = 384, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim + 4, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, embedding: torch.Tensor, recall_count: int, days_old: int,
                text_length: int) -> float:
        features = torch.cat([
            embedding,
            torch.tensor([recall_count, days_old, text_length, 0.0], dtype=torch.float32)
        ], dim=-1)
        confidence = self.net(features)
        return confidence.item()


class EmotionClassifier(nn.Module):
    emotions = ["joy", "sadness", "regret", "acceptance", "nostalgia", "anger", "shame", "pride"]
    emotion_to_idx = {e: i for i, e in enumerate(emotions)}

    def __init__(self, embedding_dim: int = 384, hidden_dim: int = 128, dropout: float = 0.2):
        super().__init__()
        num_emotions = len(self.emotions)
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, num_emotions)
        )

    def forward(self, embedding: torch.Tensor) -> dict:
        logits = self.net(embedding)
        probabilities = F.softmax(logits, dim=-1)
        return {
            emotion: prob.item()
            for emotion, prob in zip(self.emotions, probabilities)
        }


class ModelStorage:
    def __init__(self, model_dir: Optional[Path] = None):
        self.model_dir = model_dir or Path("./models")
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def save_drift_model(self, model: MemoryDriftNN, version: str = "1.0"):
        path = self.model_dir / f"drift_nn_v{version}.pt"
        torch.save(model.state_dict(), path)
        logger.info(f"Drift model saved: {path}")

    def load_drift_model(self, version: str = "1.0", embedding_dim: int = 384) -> MemoryDriftNN:
        path = self.model_dir / f"drift_nn_v{version}.pt"
        if not path.exists():
            logger.warning(f"Model not found: {path}, returning fresh model")
            return MemoryDriftNN(embedding_dim)
        model = MemoryDriftNN(embedding_dim)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        logger.info(f"Drift model loaded: {path}")
        return model

    def save_confidence_model(self, model: ConfidenceScorer, version: str = "1.0"):
        path = self.model_dir / f"confidence_nn_v{version}.pt"
        torch.save(model.state_dict(), path)
        logger.info(f"Confidence model saved: {path}")

    def load_confidence_model(self, version: str = "1.0", embedding_dim: int = 384) -> ConfidenceScorer:
        path = self.model_dir / f"confidence_nn_v{version}.pt"
        if not path.exists():
            logger.warning(f"Model not found: {path}, returning fresh model")
            return ConfidenceScorer(embedding_dim)
        model = ConfidenceScorer(embedding_dim)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        logger.info(f"Confidence model loaded: {path}")
        return model

    def save_emotion_model(self, model: EmotionClassifier, version: str = "1.0"):
        path = self.model_dir / f"emotion_nn_v{version}.pt"
        torch.save(model.state_dict(), path)
        logger.info(f"Emotion model saved: {path}")

    def load_emotion_model(self, version: str = "1.0", embedding_dim: int = 384) -> EmotionClassifier:
        path = self.model_dir / f"emotion_nn_v{version}.pt"
        if not path.exists():
            logger.warning(f"Model not found: {path}, returning fresh model")
            return EmotionClassifier(embedding_dim)
        model = EmotionClassifier(embedding_dim)
        model.load_state_dict(torch.load(path, map_location="cpu"))
        logger.info(f"Emotion model loaded: {path}")
        return model


class TrainingUtils:
    @staticmethod
    def create_drift_training_data(embeddings_prev: list, embeddings_curr: list,
                                   drift_labels: list) -> Tuple[torch.Tensor, torch.Tensor]:
        X = []
        y = []
        for prev, curr, label in zip(embeddings_prev, embeddings_curr, drift_labels):
            X.append(torch.cat([torch.tensor(prev), torch.tensor(curr)]))
            y.append(torch.tensor(label))
        return torch.stack(X), torch.stack(y)

    @staticmethod
    def create_confidence_training_data(embeddings: list, features_list: list,
                                       ratings: list) -> Tuple[torch.Tensor, torch.Tensor]:
        X = []
        y = []
        for emb, features, rating in zip(embeddings, features_list, ratings):
            combined = torch.cat([torch.tensor(emb), torch.tensor(features)])
            X.append(combined)
            y.append(torch.tensor([rating]))
        return torch.stack(X), torch.stack(y)

    @staticmethod
    def create_emotion_training_data(embeddings: list,
                                    emotion_labels: list) -> Tuple[torch.Tensor, torch.Tensor]:
        X = torch.stack([torch.tensor(e, dtype=torch.float32) for e in embeddings])
        y = torch.stack([torch.tensor(EmotionClassifier.emotion_to_idx[label], dtype=torch.long)
                        for label in emotion_labels])
        return X, y
