"""MediaPipe hand landmark extraction and normalization."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Optional

import numpy as np

try:
    import cv2
    import mediapipe as mp
except ImportError:  # pragma: no cover - exercised when optional deps missing
    cv2 = None
    mp = None


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class HandLandmarkConfig:
    """Settings for MediaPipe hand tracking."""

    max_num_hands: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5


class HandLandmarkExtractor:
    """Extract and normalize MediaPipe hand landmarks from RGB/BGR frames."""

    def __init__(self, config: HandLandmarkConfig | None = None) -> None:
        self.config = config or HandLandmarkConfig()
        self._hands = None
        self._drawing = None
        if mp is None:
            LOGGER.warning(
                "mediapipe is not installed. Hand landmark extraction is unavailable.",
            )
            return
        self._hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=self.config.max_num_hands,
            min_detection_confidence=self.config.min_detection_confidence,
            min_tracking_confidence=self.config.min_tracking_confidence,
        )
        self._drawing = mp.solutions.drawing_utils

    @staticmethod
    def normalize_landmarks(landmarks: np.ndarray) -> np.ndarray:
        """
        Normalize hand landmarks to translation and scale invariant features.

        Args:
            landmarks: Array with shape (21, 3).
        Returns:
            Flat array with shape (63,).
        """
        if landmarks.shape != (21, 3):
            raise ValueError(f"Expected landmarks shape (21, 3), got {landmarks.shape}")

        centered = landmarks - landmarks[0]
        distances = np.linalg.norm(centered, axis=1)
        scale = float(np.max(distances))
        if scale < 1e-6:
            scale = 1.0
        normalized = centered / scale
        return normalized.astype(np.float32).reshape(-1)

    def extract(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        """Extract normalized landmarks from one BGR frame."""
        if self._hands is None or cv2 is None:
            return None

        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None

        first_hand = result.multi_hand_landmarks[0]
        coords = np.array(
            [[lm.x, lm.y, lm.z] for lm in first_hand.landmark],
            dtype=np.float32,
        )
        return self.normalize_landmarks(coords)

    def annotate(self, frame_bgr: np.ndarray) -> np.ndarray:
        """Draw hand skeleton on frame for debugging."""
        if self._hands is None or cv2 is None or mp is None:
            return frame_bgr
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        result = self._hands.process(rgb)
        if result.multi_hand_landmarks and self._drawing is not None:
            for hand in result.multi_hand_landmarks:
                self._drawing.draw_landmarks(
                    frame_bgr,
                    hand,
                    mp.solutions.hands.HAND_CONNECTIONS,
                )
        return frame_bgr

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._hands is not None:
            self._hands.close()

