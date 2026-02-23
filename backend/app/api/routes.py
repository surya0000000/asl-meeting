"""REST API routes."""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import APIRouter, Request

from backend.app.schemas import PredictRequest, PredictResponse
from backend.app.services.stream_processor import ProcessedPrediction, StreamProcessor


router = APIRouter()


@router.get("/health")
async def health(request: Request) -> dict[str, str | bool]:
    settings = request.app.state.settings
    checkpoint_exists = Path(settings.model_checkpoint_path).exists()
    return {
        "status": "ok",
        "service": settings.app_name,
        "checkpoint_found": checkpoint_exists,
    }


@router.post("/predict", response_model=PredictResponse)
async def predict(payload: PredictRequest, request: Request) -> PredictResponse:
    processor: StreamProcessor = request.app.state.stream_processor

    prediction: ProcessedPrediction | None
    if payload.sequence is not None:
        prediction = await asyncio.to_thread(
            processor.predict_sequence,
            payload.sequence,
            payload.refine_text,
            payload.speak,
        )
    else:
        prediction = await asyncio.to_thread(
            processor.process_landmarks,
            payload.landmarks,
            payload.refine_text,
            payload.speak,
        )

    if prediction is None:
        return PredictResponse(
            raw_prediction="BUFFERING",
            refined_text="Collecting temporal context...",
            confidence=0.0,
            frame_index=None,
            timestamp=None,
            audio_url=None,
        )

    return PredictResponse(**prediction.to_dict())

