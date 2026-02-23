"""API schemas for gesture inference endpoints."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class PredictRequest(BaseModel):
    """Inference request payload."""

    landmarks: list[float] | None = Field(default=None, description="Single normalized 63-dim frame.")
    sequence: list[list[float]] | None = Field(
        default=None,
        description="Optional (T, 63) sequence for direct batch prediction.",
    )
    refine_text: bool = True
    speak: bool = False

    @model_validator(mode="after")
    def validate_source(self) -> "PredictRequest":
        if self.landmarks is None and self.sequence is None:
            raise ValueError("Provide either landmarks or sequence.")
        return self


class PredictResponse(BaseModel):
    """Inference response payload."""

    raw_prediction: str
    refined_text: str
    confidence: float
    audio_url: str | None = None
    frame_index: int | None = None
    timestamp: str | None = None


class WSInputMessage(BaseModel):
    """Streaming payload accepted on /ws/asl-stream."""

    type: str = "landmarks"
    landmarks: list[float]
    refine_text: bool = True
    speak: bool = False


class WSOutputMessage(BaseModel):
    """Streaming payload emitted to websocket clients."""

    type: str = "prediction"
    raw_prediction: str
    refined_text: str
    confidence: float
    audio_url: str | None = None
    frame_index: int
    timestamp: str

