"""Real-time local inference and optional backend streaming client."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from typing import Any

from ml.camera_capture import AsyncCameraCapture, CameraConfig
from ml.config import MLConfig
from ml.hand_landmarks import HandLandmarkExtractor
from ml.inference_engine import GestureInferenceEngine

try:
    import cv2
except ImportError:  # pragma: no cover - optional runtime dep
    cv2 = None

try:
    import websockets
except ImportError:  # pragma: no cover - optional runtime dep
    websockets = None


LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real-time ASL inference from webcam.")
    parser.add_argument("--camera-index", type=int, default=0)
    parser.add_argument("--show-preview", action="store_true")
    parser.add_argument("--backend-ws-url", type=str, default="")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


async def maybe_send_backend(
    websocket: Any,
    landmarks: list[float],
    prediction: str | None = None,
) -> dict[str, Any] | None:
    payload: dict[str, Any] = {"type": "landmarks", "landmarks": landmarks}
    if prediction:
        payload["local_prediction"] = prediction
    await websocket.send(json.dumps(payload))
    try:
        raw_response = await asyncio.wait_for(websocket.recv(), timeout=0.001)
        return json.loads(raw_response)
    except TimeoutError:
        return None
    except Exception:
        LOGGER.debug("Backend response unavailable for this frame.", exc_info=True)
        return None


async def run_stream(args: argparse.Namespace) -> None:
    camera = AsyncCameraCapture(CameraConfig(device_index=args.camera_index, fps=30))
    extractor = HandLandmarkExtractor()
    ml_config = MLConfig.from_env()
    engine = GestureInferenceEngine(ml_config)
    camera.start()

    websocket = None
    if args.backend_ws_url:
        if websockets is None:
            raise RuntimeError("websockets package is required for backend streaming mode.")
        websocket = await websockets.connect(args.backend_ws_url, max_queue=1)
        LOGGER.info("Connected to backend websocket: %s", args.backend_ws_url)

    frame_count = 0
    start_time = time.perf_counter()
    try:
        async for frame in camera.frames():
            frame_count += 1
            landmarks = extractor.extract(frame)
            if landmarks is None:
                if args.show_preview and cv2 is not None:
                    await asyncio.to_thread(cv2.imshow, "ASL Meeting Copilot", frame)
                    key = await asyncio.to_thread(cv2.waitKey, 1)
                    if key == ord("q"):
                        break
                continue

            result = engine.process_landmarks(landmarks)
            local_prediction = result.label if result is not None else None

            if result is not None:
                LOGGER.info(
                    "Local prediction: %s (confidence=%.3f frame=%d)",
                    result.label,
                    result.confidence,
                    result.frame_index,
                )

            if websocket is not None:
                backend_response = await maybe_send_backend(
                    websocket=websocket,
                    landmarks=landmarks.tolist(),
                    prediction=local_prediction,
                )
                if backend_response:
                    LOGGER.info("Backend response: %s", backend_response)

            if args.show_preview and cv2 is not None:
                preview = frame.copy()
                if result is not None:
                    cv2.putText(
                        preview,
                        f"{result.label} ({result.confidence:.2f})",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2,
                    )
                await asyncio.to_thread(cv2.imshow, "ASL Meeting Copilot", preview)
                key = await asyncio.to_thread(cv2.waitKey, 1)
                if key == ord("q"):
                    break

            if frame_count % 300 == 0:
                elapsed = time.perf_counter() - start_time
                fps = frame_count / max(1e-6, elapsed)
                LOGGER.info("Approx capture throughput: %.2f FPS", fps)
    finally:
        camera.stop()
        extractor.close()
        if websocket is not None:
            await websocket.close()
        if cv2 is not None:
            await asyncio.to_thread(cv2.destroyAllWindows)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    asyncio.run(run_stream(args))


if __name__ == "__main__":
    main()

