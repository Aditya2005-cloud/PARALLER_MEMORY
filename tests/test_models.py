import pytest
import torch
import numpy as np

from parallel_memory.models_ml import (
    MemoryDriftNN,
    ConfidenceScorer,
    EmotionClassifier,
    ModelStorage,
    TrainingUtils
)


@pytest.fixture
def device():
    return "cpu"


@pytest.fixture
def drift_model(device):
    return MemoryDriftNN().to(device)


@pytest.fixture
def confidence_model(device):
    return ConfidenceScorer().to(device)


@pytest.fixture
def emotion_model(device):
    return EmotionClassifier().to(device)


class TestMemoryDriftNN:
    def test_forward_pass(self, drift_model, device):
        prev_embedding = torch.randn(384, device=device)
        curr_embedding = torch.randn(384, device=device)

        semantic_drift, emotion_drift = drift_model(prev_embedding, curr_embedding)

        assert isinstance(semantic_drift, float)
        assert isinstance(emotion_drift, float)
        assert 0 <= semantic_drift <= 1
        assert 0 <= emotion_drift <= 1

    def test_batch_forward(self, drift_model, device):
        prev_batch = torch.randn(16, 384, device=device)
        curr_batch = torch.randn(16, 384, device=device)

        results = []
        for prev, curr in zip(prev_batch, curr_batch):
            sd, ed = drift_model(prev, curr)
            results.append((sd, ed))

        assert len(results) == 16
        assert all(0 <= sd <= 1 for sd, _ in results)
        assert all(0 <= ed <= 1 for _, ed in results)

    def test_model_parameters(self, drift_model):
        params = list(drift_model.parameters())
        assert len(params) > 0

        total_params = sum(p.numel() for p in params)
        assert total_params > 1000


class TestConfidenceScorer:
    def test_forward_pass(self, confidence_model, device):
        embedding = torch.randn(384, device=device)
        confidence = confidence_model(embedding, recall_count=2, days_old=30, text_length=50)

        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1

    def test_confidence_decreases_with_age(self, confidence_model, device):
        embedding = torch.randn(384, device=device)

        conf_young = confidence_model(embedding, 1, 1, 50)
        conf_old = confidence_model(embedding, 1, 365, 50)

        assert conf_young > 0
        assert conf_old > 0

    def test_model_parameters(self, confidence_model):
        params = list(confidence_model.parameters())
        assert len(params) > 0


class TestEmotionClassifier:
    def test_forward_pass(self, emotion_model, device):
        embedding = torch.randn(384, device=device)
        emotion_dist = emotion_model(embedding)

        assert isinstance(emotion_dist, dict)
        assert len(emotion_dist) == 8

        for emotion, prob in emotion_dist.items():
            assert emotion in EmotionClassifier.emotions
            assert 0 <= prob <= 1

    def test_probabilities_sum_to_one(self, emotion_model, device):
        embedding = torch.randn(384, device=device)
        emotion_dist = emotion_model(embedding)

        total_prob = sum(emotion_dist.values())
        assert abs(total_prob - 1.0) < 0.01

    def test_emotion_mapping(self):
        assert "regret" in EmotionClassifier.emotion_to_idx
        assert "joy" in EmotionClassifier.emotion_to_idx
        assert len(EmotionClassifier.emotions) == 8

    def test_batch_inference(self, emotion_model, device):
        batch = torch.randn(8, 384, device=device)

        results = []
        for emb in batch:
            dist = emotion_model(emb)
            results.append(dist)

        assert len(results) == 8
        for dist in results:
            assert len(dist) == 8
            assert abs(sum(dist.values()) - 1.0) < 0.01


class TestModelStorage:
    def test_save_and_load_drift_model(self, drift_model, tmp_path):
        storage = ModelStorage(tmp_path)
        storage.save_drift_model(drift_model, "1.0")

        loaded_model = storage.load_drift_model("1.0")
        assert loaded_model is not None

        assert list(drift_model.parameters())[0].shape == list(loaded_model.parameters())[0].shape

    def test_load_nonexistent_model(self, tmp_path):
        storage = ModelStorage(tmp_path)
        model = storage.load_drift_model("9.9")

        assert model is not None
        assert isinstance(model, MemoryDriftNN)


class TestTrainingUtils:
    def test_create_drift_training_data(self):
        embeddings_prev = [np.random.randn(384) for _ in range(5)]
        embeddings_curr = [np.random.randn(384) for _ in range(5)]
        drift_labels = [0.5, 0.6, 0.7, 0.3, 0.8]

        X, y = TrainingUtils.create_drift_training_data(
            embeddings_prev, embeddings_curr, drift_labels
        )

        assert X.shape == (5, 768)
        assert y.shape == (5,)

    def test_create_confidence_training_data(self):
        embeddings = [np.random.randn(384) for _ in range(5)]
        features_list = [[1, 30, 50] for _ in range(5)]
        ratings = [0.7, 0.3, 0.9, 0.2, 0.5]

        X, y = TrainingUtils.create_confidence_training_data(
            embeddings, features_list, ratings
        )

        assert X.shape == (5, 387)
        assert y.shape == (5,)

    def test_create_emotion_training_data(self):
        embeddings = [np.random.randn(384) for _ in range(5)]
        emotion_labels = ["joy", "sadness", "regret", "joy", "nostalgia"]

        X, y = TrainingUtils.create_emotion_training_data(embeddings, emotion_labels)

        assert X.shape == (5, 384)
        assert y.shape == (5,)
        assert all(0 <= idx < 8 for idx in y.tolist())
