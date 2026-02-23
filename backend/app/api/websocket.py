"""WebSocket streaming endpoint for ASL landmark inference."""

from __future__ import annotations

import asyncio
import json
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import ValidationError

from backend.app.schemas import WSInputMessage, WSOutputMessage
from backend.app.services.stream_processor import StreamProcessor


LOGGER = logging.getLogger(__name__)
router = APIRouter()


@router.websocket("/ws/asl-stream")
async def asl_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    await websocket.send_json({"type": "connection", "status": "connected"})

    processor: StreamProcessor = websocket.app.state.stream_processor
    try:
        while True:
            payload = await websocket.receive_text()
            try:
                parsed = WSInputMessage.model_validate_json(payload)
            except ValidationError as exc:
                await websocket.send_json({"type": "error", "detail": exc.errors()})
                continue
            except json.JSONDecodeError:
                await websocket.send_json({"type": "error", "detail": "Invalid JSON payload."})
                continue

            prediction = await asyncio.to_thread(
                processor.process_landmarks,
                parsed.landmarks,
                parsed.refine_text,
                parsed.speak,
            )
            if prediction is None:
                await websocket.send_json({"type": "buffering", "detail": "Insufficient frames."})
                continue

            response = WSOutputMessage(
                raw_prediction=prediction.raw_prediction,
                refined_text=prediction.refined_text,
                confidence=prediction.confidence,
                audio_url=prediction.audio_url,
                frame_index=prediction.frame_index,
                timestamp=prediction.timestamp,
            )
            await websocket.send_text(response.model_dump_json())
    except WebSocketDisconnect:
        LOGGER.info("WebSocket client disconnected.")
    except Exception:
        LOGGER.exception("Unhandled websocket error.")
        try:
            await websocket.close(code=1011)
        except RuntimeError:
            pass

