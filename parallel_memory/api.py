import logging
from uuid import uuid4
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .models import FeedbackIn, MemoryIn, RecallIn
from .database import DatabaseManager, GlobalDatabaseManager
from .exceptions import (
    MemoryError, EmbeddingError, ModelInferenceError,
    DatabaseError, UserValidationError, ParallelMemoryError
)
from .colab_utils import setup_logging

logger = logging.getLogger("parallel_memory.api")
setup_logging()

app = FastAPI(title="Parallel Memory API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline_cache: dict[str, object] = {}


def validate_user_id(user_id: str) -> bool:
    if not user_id or not isinstance(user_id, str) or len(user_id) == 0:
        raise UserValidationError("Invalid user_id", user_id)
    return True


def get_pipeline(user_id: str):
    if user_id in _pipeline_cache:
        return _pipeline_cache[user_id]
    from .pipeline import PipelineEngine

    engine = PipelineEngine(user_id)
    _pipeline_cache[user_id] = engine
    return engine


@app.on_event("startup")
def startup() -> None:
    logger.info("Parallel Memory API starting up")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "version": "0.2.0"}


@app.post("/memories")
def add_memory(payload: MemoryIn) -> dict:
    try:
        validate_user_id(payload.user_id)

        db = DatabaseManager(payload.user_id)
        memory_id = str(uuid4())

        memory_row = db.create_memory(
            memory_id=memory_id,
            text=payload.memory_text,
            emotion=payload.emotion,
            confidence=payload.confidence,
            model_version="1.0"
        )

        pipeline = get_pipeline(payload.user_id)
        embedding = pipeline.embed_text(payload.memory_text)
        serialized = pipeline.embedding_manager.serialize_embedding(embedding)
        db.store_embedding(memory_id, serialized, "1.0")

        logger.info(f"Memory created: {memory_id} for user {payload.user_id}")

        return {
            "message": "memory stored",
            "data": {
                **memory_row,
                "embedding_dimension": len(embedding),
            }
        }

    except UserValidationError as e:
        logger.error(f"User validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except EmbeddingError as e:
        logger.error(f"Embedding error: {e}")
        raise HTTPException(status_code=500, detail="Failed to embed memory") from e
    except Exception as e:
        logger.error(f"Unexpected error in add_memory: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.post("/recalls")
def add_recall(payload: RecallIn) -> dict:
    try:
        validate_user_id(payload.user_id)

        db = DatabaseManager(payload.user_id)
        memory = db.get_memory(payload.memory_id)
        if not memory:
            raise HTTPException(status_code=404, detail=f"Memory not found: {payload.memory_id}")

        pipeline = get_pipeline(payload.user_id)

        result = pipeline.process_recall(
            memory_id=payload.memory_id,
            recall_text=payload.recall_text,
            emotion=payload.emotion,
            confidence=payload.confidence
        )

        estimated_confidence = pipeline.estimate_confidence(payload.memory_id)

        logger.info(f"Recall processed: {result['recall_id']} for memory {payload.memory_id}")

        return {
            "message": "recall analyzed + alternate timeline generated",
            "label": "AI simulation is probabilistic, not objective truth.",
            "data": result,
            "estimated_confidence": estimated_confidence
        }

    except UserValidationError as e:
        logger.error(f"User validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except MemoryError as e:
        logger.error(f"Memory error: {e}")
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ModelInferenceError as e:
        logger.error(f"Model inference error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process recall") from e
    except Exception as e:
        logger.error(f"Unexpected error in add_recall: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.post("/feedback")
def add_feedback(payload: FeedbackIn) -> dict:
    try:
        validate_user_id(payload.user_id)

        db = DatabaseManager(payload.user_id)
        global_db = GlobalDatabaseManager()

        feedback_row = db.create_feedback(
            memory_id=payload.memory_id,
            response_id=payload.response_id,
            rating=payload.rating,
            correction=payload.correction,
            notes=payload.notes
        )

        global_db.add_global_feedback(
            rating=payload.rating,
            correction=payload.correction,
            category="model_correction" if payload.correction else "rating"
        )

        logger.info(f"Feedback recorded: {feedback_row['id']} for user {payload.user_id}")

        return {
            "message": "feedback stored",
            "data": feedback_row,
            "aggregated": True
        }

    except UserValidationError as e:
        logger.error(f"User validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Unexpected error in add_feedback: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.get("/global/summary")
def get_global_summary() -> dict:
    try:
        global_db = GlobalDatabaseManager()
        stats = global_db.get_feedback_stats()
        return {
            "message": "global summary",
            "data": stats
        }
    except Exception as e:
        logger.error(f"Error getting global summary: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve summary") from e


@app.post("/global/retrain-check")
def retrain_check(min_negative: int = 20) -> dict:
    try:
        global_db = GlobalDatabaseManager()
        negative_samples = global_db.get_negative_feedback(limit=min_negative * 5)
        should_retrain = len(negative_samples) >= min_negative

        return {
            "should_retrain": should_retrain,
            "negative_count": len(negative_samples),
            "threshold": min_negative,
            "retrain_trigger": should_retrain
        }
    except Exception as e:
        logger.error(f"Error in retrain_check: {e}")
        raise HTTPException(status_code=500, detail="Failed to check retrain") from e


@app.get("/user/{user_id}/memories")
def list_user_memories(user_id: str, limit: int = 50, offset: int = 0) -> dict:
    try:
        validate_user_id(user_id)

        db = DatabaseManager(user_id)
        memories = db.list_memories(limit=limit, offset=offset)

        return {
            "message": "user memories",
            "user_id": user_id,
            "count": len(memories),
            "data": memories
        }

    except UserValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Error listing memories: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.get("/user/{user_id}/feedback-stats")
def get_user_feedback_stats(user_id: str) -> dict:
    try:
        validate_user_id(user_id)

        db = DatabaseManager(user_id)
        stats = db.get_feedback_stats()

        return {
            "message": "user feedback stats",
            "user_id": user_id,
            "data": stats
        }

    except UserValidationError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as e:
        logger.error(f"Error getting feedback stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error") from e


@app.get("/errors")
def get_recent_errors(limit: int = 50) -> dict:
    try:
        import os
        from pathlib import Path

        log_dir = Path("./logs")
        if not log_dir.exists():
            return {"message": "no logs yet", "data": []}

        recent_logs = sorted(log_dir.glob("*.log"), key=os.path.getctime, reverse=True)

        return {
            "message": "recent logs",
            "count": len(recent_logs),
            "logs": [str(l.name) for l in recent_logs[:limit]]
        }

    except Exception as e:
        logger.error(f"Error retrieving logs: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve logs") from e

