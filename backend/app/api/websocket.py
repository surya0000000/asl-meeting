"""WebSocket streaming endpoint with per-session async pipeline."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from backend.app.core.session_manager import SessionManager, SessionState
from backend.app.services.onnx_inference_engine import ONNXInferenceEngine
from backend.app.services.speech_engine import SpeechEngine
from backend.app.services.text_refiner import TextRefiner
from ml.config import LANDMARK_DIM, SEQUENCE_LENGTH


LOGGER = logging.getLogger(__name__)
router = APIRouter()


def _coerce_frame(frame: list[float] | np.ndarray) -> np.ndarray:
    arr = np.asarray(frame, dtype=np.float32).reshape(-1)
    if arr.shape[0] < LANDMARK_DIM:
        padded = np.zeros((LANDMARK_DIM,), dtype=np.float32)
        padded[: arr.shape[0]] = arr
        return padded
    if arr.shape[0] > LANDMARK_DIM:
        return arr[:LANDMARK_DIM].astype(np.float32)
    return arr.astype(np.float32)


def _coerce_sequence(sequence: Any) -> np.ndarray:
    arr = np.asarray(sequence, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("Expected sequence shape (T, F).")
    if arr.shape[1] < LANDMARK_DIM:
        padded = np.zeros((arr.shape[0], LANDMARK_DIM), dtype=np.float32)
        padded[:, : arr.shape[1]] = arr
        arr = padded
    elif arr.shape[1] > LANDMARK_DIM:
        arr = arr[:, :LANDMARK_DIM]
    if arr.shape[0] < SEQUENCE_LENGTH:
        padded = np.zeros((SEQUENCE_LENGTH, LANDMARK_DIM), dtype=np.float32)
        padded[-arr.shape[0] :] = arr
        arr = padded
    elif arr.shape[0] > SEQUENCE_LENGTH:
        arr = arr[-SEQUENCE_LENGTH:]
    return arr.astype(np.float32)


def _sequence_from_buffer(state: SessionState) -> np.ndarray:
    frames = list(state.sequence_buffer)
    if not frames:
        return np.zeros((SEQUENCE_LENGTH, LANDMARK_DIM), dtype=np.float32)
    arr = np.stack(frames, axis=0).astype(np.float32)
    return _coerce_sequence(arr)


@router.websocket("/ws/asl-stream")
async def asl_stream(websocket: WebSocket) -> None:
    await websocket.accept()
    manager: SessionManager = websocket.app.state.session_manager
    onnx_engine: ONNXInferenceEngine = websocket.app.state.onnx_inference_engine
    text_refiner: TextRefiner = websocket.app.state.text_refiner
    speech_engine: SpeechEngine = websocket.app.state.speech_engine
    settings = websocket.app.state.settings

    state = await manager.get_or_create(websocket.query_params.get("session_id"))
    await websocket.send_json(
        {
            "type": "connection",
            "status": "connected",
            "session_id": str(state.session_id),
        },
    )

    send_lock = asyncio.Lock()
    background_tasks: set[asyncio.Task[Any]] = set()
    tts_worker_task: asyncio.Task[Any] | None = None

    async def safe_send(payload: dict[str, Any]) -> None:
        async with send_lock:
            await websocket.send_json(payload)

    def track_task(task: asyncio.Task[Any]) -> None:
        background_tasks.add(task)

        def _done(done_task: asyncio.Task[Any]) -> None:
            background_tasks.discard(done_task)
            try:
                done_task.result()
            except asyncio.CancelledError:
                pass
            except Exception:
                LOGGER.exception("Background websocket task failed.")

        task.add_done_callback(_done)

    async def tts_worker(session: SessionState) -> None:
        while True:
            item = await session.tts_queue.get()
            if item is None:
                break
            text, gloss = item
            try:
                audio_payload = await speech_engine.synthesize_audio_message(text=text, gloss=gloss)
                if audio_payload:
                    await safe_send(audio_payload)
            except Exception:
                LOGGER.exception("TTS worker failed for session %s", session.session_id)

    async def handle_post_inference(
        session: SessionState,
        raw_prediction: str | None,
        confidence: float,
    ) -> None:
        prediction_for_aggregation: str | None = raw_prediction
        if confidence < settings.confidence_threshold:
            prediction_for_aggregation = None

        events = session.gloss_aggregator.process(
            prediction=prediction_for_aggregation,
            timestamp=datetime.now(timezone.utc),
        )
        for event in events:
            event_type = event.get("type")
            if event_type == "gloss":
                gloss = str(event.get("gloss", "")).strip().upper()
                if not gloss:
                    continue
                session.add_prediction(gloss)
                session.gloss_accumulator.append(gloss)
                await manager.update_activity(session.session_id)

                raw_sequence = " ".join(session.gloss_accumulator)
                tier1 = text_refiner.refine_tier1(raw_sequence)
                await safe_send(
                    {
                        "type": "subtitle",
                        "session_id": str(session.session_id),
                        "gloss": gloss,
                        "rawTranscript": raw_sequence,
                        "refinedText": tier1,
                        "confidence": confidence,
                    },
                )

                async def refine_task(
                    sequence_snapshot: str = raw_sequence,
                    tier1_snapshot: str = tier1,
                    gloss_snapshot: str = gloss,
                ) -> None:
                    refined = await text_refiner.refine_tier2_async(
                        raw_text=sequence_snapshot,
                        tier1_text=tier1_snapshot,
                    )
                    await safe_send(
                        {
                            "type": "subtitle_refined",
                            "session_id": str(session.session_id),
                            "gloss": gloss_snapshot,
                            "rawTranscript": sequence_snapshot,
                            "refinedText": refined,
                        },
                    )

                track_task(asyncio.create_task(refine_task()))
                await session.tts_queue.put((tier1, gloss))

            elif event_type == "sentence_end":
                finalized = " ".join(session.gloss_accumulator).strip()
                if finalized:
                    text_refiner.add_context(text_refiner.refine_tier1(finalized))
                session.gloss_accumulator.clear()
                await safe_send(
                    {
                        "type": "sentence_end",
                        "session_id": str(session.session_id),
                    },
                )

    tts_worker_task = asyncio.create_task(tts_worker(state), name=f"tts-worker-{state.session_id}")
    try:
        while True:
            payload = await websocket.receive_text()
            try:
                parsed = json.loads(payload)
                if not isinstance(parsed, dict):
                    raise ValueError("Payload must be an object.")
            except Exception:
                await safe_send({"type": "error", "detail": "Invalid JSON payload."})
                continue

            incoming_session = parsed.get("session_id")
            if incoming_session:
                incoming_state = await manager.get_or_create(incoming_session)
                if incoming_state.session_id != state.session_id:
                    state = incoming_state
            await manager.update_activity(state.session_id)

            message_type = str(parsed.get("type", "landmark_sequence"))
            try:
                if message_type == "landmark_sequence":
                    sequence = _coerce_sequence(parsed.get("sequence", []))
                elif message_type == "landmarks":
                    frame = _coerce_frame(parsed.get("landmarks", []))
                    state.sequence_buffer.append(frame)
                    sequence = _sequence_from_buffer(state)
                else:
                    await safe_send({"type": "error", "detail": f"Unsupported message type: {message_type}"})
                    continue
            except Exception as exc:
                await safe_send({"type": "error", "detail": str(exc)})
                continue

            inference = await asyncio.to_thread(onnx_engine.predict, sequence)
            top_predictions = inference.get("top_predictions", [])
            raw_prediction = str(inference.get("raw_prediction", "UNSURE"))
            confidence = float(inference.get("confidence", 0.0))

            await safe_send(
                {
                    "type": "prediction",
                    "session_id": str(state.session_id),
                    "gloss": raw_prediction,
                    "raw_prediction": raw_prediction,
                    "confidence": confidence,
                    "alternatives": [item.get("gloss", "") for item in top_predictions[1:3]],
                    "rawTranscript": " ".join(state.gloss_accumulator),
                    "refinedText": text_refiner.refine_tier1(" ".join(state.gloss_accumulator))
                    if state.gloss_accumulator
                    else raw_prediction,
                },
            )

            track_task(
                asyncio.create_task(
                    handle_post_inference(
                        session=state,
                        raw_prediction=raw_prediction,
                        confidence=confidence,
                    ),
                ),
            )
    except WebSocketDisconnect:
        LOGGER.info("WebSocket client disconnected.")
    except Exception:
        LOGGER.exception("Unhandled websocket error.")
        try:
            await websocket.close(code=1011)
        except RuntimeError:
            pass
    finally:
        try:
            await state.tts_queue.put(None)
        except Exception:
            pass
        if tts_worker_task is not None:
            tts_worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await tts_worker_task
        for task in list(background_tasks):
            task.cancel()
        for task in list(background_tasks):
            with contextlib.suppress(asyncio.CancelledError):
                await task
        await manager.close(state.session_id)

