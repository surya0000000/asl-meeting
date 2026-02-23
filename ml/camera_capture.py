"""Async-friendly camera capture with a non-blocking frame queue."""

from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import AsyncIterator, Optional

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - optional runtime dependency
    cv2 = None


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class CameraConfig:
    """Configuration for webcam capture."""

    device_index: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 30
    queue_size: int = 3


class AsyncCameraCapture:
    """Capture webcam frames in a background thread and consume asynchronously."""

    def __init__(self, config: CameraConfig | None = None) -> None:
        self.config = config or CameraConfig()
        self._capture: Optional["cv2.VideoCapture"] = None
        self._queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=self.config.queue_size)
        self._running = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Open camera and start producer thread."""
        if cv2 is None:
            raise RuntimeError("opencv-python is required for camera capture.")
        if self._running.is_set():
            return

        self._capture = cv2.VideoCapture(self.config.device_index)
        if not self._capture.isOpened():
            raise RuntimeError(f"Unable to open camera index {self.config.device_index}")

        self._capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(self.config.width))
        self._capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self.config.height))
        self._capture.set(cv2.CAP_PROP_FPS, float(self.config.fps))

        self._running.set()
        self._thread = threading.Thread(target=self._loop, name="camera-capture", daemon=True)
        self._thread.start()
        LOGGER.info("Camera capture started.")

    def _loop(self) -> None:
        assert self._capture is not None
        target_delay = 1.0 / max(1, self.config.fps)
        while self._running.is_set():
            start = time.perf_counter()
            ok, frame = self._capture.read()
            if not ok:
                time.sleep(0.01)
                continue

            while True:
                try:
                    self._queue.put_nowait(frame)
                    break
                except queue.Full:
                    try:
                        self._queue.get_nowait()
                    except queue.Empty:
                        break

            elapsed = time.perf_counter() - start
            if elapsed < target_delay:
                time.sleep(target_delay - elapsed)

    async def read(self, timeout: float = 0.1) -> Optional[np.ndarray]:
        """Get the latest frame asynchronously."""
        if not self._running.is_set():
            return None

        def _read() -> Optional[np.ndarray]:
            try:
                return self._queue.get(timeout=timeout)
            except queue.Empty:
                return None

        return await asyncio.to_thread(_read)

    async def frames(self) -> AsyncIterator[np.ndarray]:
        """Yield frames continuously while camera is running."""
        while self._running.is_set():
            frame = await self.read(timeout=0.2)
            if frame is None:
                await asyncio.sleep(0.001)
                continue
            yield frame

    def stop(self) -> None:
        """Stop producer thread and release camera."""
        self._running.clear()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._capture is not None:
            self._capture.release()
            self._capture = None
        LOGGER.info("Camera capture stopped.")

