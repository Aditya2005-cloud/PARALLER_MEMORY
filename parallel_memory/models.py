from typing import Any

from pydantic import BaseModel, Field


class MemoryIn(BaseModel):
    user_id: str
    memory_text: str
    emotion: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RecallIn(BaseModel):
    user_id: str
    memory_id: str
    recall_text: str
    emotion: str | None = None
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FeedbackIn(BaseModel):
    user_id: str
    memory_id: str
    response_id: str
    rating: int = Field(ge=0, le=1)
    correction: str | None = None
    notes: str | None = None

