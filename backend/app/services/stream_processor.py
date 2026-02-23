"""End-to-end stream processing pipeline for incoming landmark frames."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import logging
import threading
import time
from typing import Any

from backend.app.core.config import Settings
from backend.app.services.speech_engine import SpeechEngine
from backend.app.services.text_refiner import TextRefiner
from ml.config import MLConfig
from ml.inference_engine import GestureInferenceEngine


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class ProcessedPrediction:
    """Unified backend prediction payload."""

    raw_prediction: str
    refined_text: str
    confidence: float
    frame_index: int
    timestamp: str
    audio_url: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_prediction": self.raw_prediction,
            "refined_text": self.refined_text,
            "confidence": self.confidence,
            "frame_index": self.frame_index,
            "timestamp": self.timestamp,
            "audio_url": self.audio_url,
        }


class StreamProcessor:
    """Process streaming landmarks into text and speech outputs."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        ml_config = MLConfig(
            input_size=63,
            sequence_length=settings.sequence_length,
            predict_every_n_frames=settings.predict_every_n_frames,
            confidence_threshold=settings.confidence_threshold,
            model_checkpoint_path=settings.checkpoint_path,
            vocabulary=settings.vocabulary,
        )
        self.inference_engine = GestureInferenceEngine(ml_config)
        self.text_refiner = TextRefiner(
            enabled=settings.enable_text_refiner,
            openai_api_key=settings.openai_api_key,
            openai_model=settings.openai_model,
        )
        self.speech_engine = SpeechEngine(
            output_dir=settings.audio_output_path,
            enabled=settings.enable_tts,
        )
        self._phrase_tokens: deque[str] = deque(maxlen=20)
        self._lock = threading.Lock()
        self._last_token = ""
        self._last_emit_time = 0.0
        self._cooldown_seconds = 0.2

    def _update_phrase(self, token: str) -> None:
        now = time.monotonic()
        if token == "UNSURE":
            return
        if token != self._last_token or (now - self._last_emit_time) >= self._cooldown_seconds:
            self._phrase_tokens.append(token)
            self._last_token = token
            self._last_emit_time = now

    def _current_raw_text(self, fallback: str) -> str:
        if self._phrase_tokens:
            return " ".join(self._phrase_tokens)
        return fallback

    def process_landmarks(
        self,
        landmarks: list[float],
        refine_text: bool = True,
        speak: bool = False,
    ) -> ProcessedPrediction | None:
        """Process one normalized frame."""
        with self._lock:
            result = self.inference_engine.process_landmarks(landmarks)
            if result is None:
                return None

            self._update_phrase(result.label)
            raw_text = self._current_raw_text(result.label)
            refined = self.text_refiner.refine(raw_text) if refine_text else raw_text

            audio_url = None
            if speak:
                audio_path = self.speech_engine.synthesize(refined)
                if audio_path is not None:
                    audio_url = f"{self.settings.static_audio_mount}/{audio_path.name}"
                    self.speech_engine.cleanup_old_files(limit=100)

            return ProcessedPrediction(
                raw_prediction=result.label,
                refined_text=refined,
                confidence=result.confidence,
                frame_index=result.frame_index,
                timestamp=result.timestamp,
                audio_url=audio_url,
            )

    def predict_sequence(
        self,
        sequence: list[list[float]],
        refine_text: bool = True,
        speak: bool = False,
    ) -> ProcessedPrediction:
        """Run inference directly on a full sequence payload."""
        with self._lock:
            out = self.inference_engine.predict_sequence(sequence)
            raw_prediction = str(out["label"])
            confidence = float(out["confidence"])
            self._update_phrase(raw_prediction)
            raw_text = self._current_raw_text(raw_prediction)
            refined = self.text_refiner.refine(raw_text) if refine_text else raw_text
            audio_url = None
            if speak:
                audio_path = self.speech_engine.synthesize(refined)
                if audio_path is not None:
                    audio_url = f"{self.settings.static_audio_mount}/{audio_path.name}"
                    self.speech_engine.cleanup_old_files(limit=100)
            return ProcessedPrediction(
                raw_prediction=raw_prediction,
                refined_text=refined,
                confidence=confidence,
                frame_index=-1,
                timestamp="",
                audio_url=audio_url,
            )

