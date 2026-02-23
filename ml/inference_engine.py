"""Real-time gesture inference engine with temporal buffering."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import logging
from pathlib import Path
from typing import Any

import numpy as np

from ml.config import MLConfig
from ml.gesture_model import ModelConfig, build_model
from ml.sequence_buffer import SequenceBuffer

try:
    import torch
except ImportError:  # pragma: no cover - optional fallback path
    torch = None


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class InferenceResult:
    """Output emitted after one temporal prediction step."""

    label: str
    confidence: float
    frame_index: int
    timestamp: str


class GestureInferenceEngine:
    """Stateful streaming inference engine with sliding-window processing."""

    def __init__(self, config: MLConfig | None = None) -> None:
        self.config = config or MLConfig.from_env()
        self.buffer = SequenceBuffer(
            sequence_length=self.config.sequence_length,
            feature_dim=self.config.input_size,
        )
        self.frame_counter = 0
        self.vocabulary = self.config.vocabulary
        self._device = "cpu"
        self._model = None
        self._torch_enabled = torch is not None

        if self._torch_enabled:
            self._init_torch_model()
        else:
            LOGGER.warning("PyTorch unavailable. Falling back to heuristic inference.")

    def _init_torch_model(self) -> None:
        assert torch is not None
        model_config = ModelConfig(
            input_size=self.config.input_size,
            num_classes=len(self.vocabulary),
        )
        self._model = build_model(model_config)

        checkpoint_path = Path(self.config.model_checkpoint_path)
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            model_state = checkpoint.get("model_state_dict", checkpoint)
            self._model.load_state_dict(model_state)
            saved_vocab = checkpoint.get("vocabulary")
            if isinstance(saved_vocab, list) and saved_vocab:
                self.vocabulary = [str(v) for v in saved_vocab]
            LOGGER.info("Loaded model checkpoint from %s", checkpoint_path)
        else:
            LOGGER.warning(
                "Checkpoint %s not found. Using randomly initialized weights.",
                checkpoint_path,
            )
        self._model.eval()

    def reset(self) -> None:
        """Clear temporal context for a fresh stream."""
        self.buffer.clear()
        self.frame_counter = 0

    def _torch_predict(self, sequence: np.ndarray) -> tuple[int, float]:
        assert torch is not None
        assert self._model is not None
        with torch.inference_mode():
            tensor = torch.from_numpy(sequence.astype(np.float32)).unsqueeze(0)
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=1)[0]
            conf, idx = torch.max(probs, dim=0)
        return int(idx.item()), float(conf.item())

    def _heuristic_predict(self, sequence: np.ndarray) -> tuple[int, float]:
        # Lightweight deterministic fallback when torch/checkpoint isn't available.
        # Uses aggregate motion and sign to distribute guesses across vocabulary.
        motion = float(np.mean(np.abs(np.diff(sequence, axis=0)))) if sequence.shape[0] > 1 else 0.0
        energy = float(np.mean(np.abs(sequence)))
        score = int(abs((motion * 1000.0) + (energy * 100.0))) % max(1, len(self.vocabulary))
        confidence = float(min(0.75, 0.35 + motion + energy))
        return score, confidence

    def process_landmarks(self, landmarks: list[float] | np.ndarray) -> InferenceResult | None:
        """
        Consume one normalized frame and emit prediction every N frames.

        Returns:
            InferenceResult when prediction is scheduled and enough context is available,
            otherwise None.
        """
        self.frame_counter += 1
        self.buffer.add(landmarks)

        if not self.buffer.is_full:
            return None

        if self.frame_counter % self.config.predict_every_n_frames != 0:
            return None

        sequence = self.buffer.as_array()
        if self._torch_enabled and self._model is not None:
            idx, confidence = self._torch_predict(sequence)
        else:
            idx, confidence = self._heuristic_predict(sequence)

        label = self.vocabulary[idx]
        if confidence < self.config.confidence_threshold:
            label = "UNSURE"

        return InferenceResult(
            label=label,
            confidence=confidence,
            frame_index=self.frame_counter,
            timestamp=datetime.now(timezone.utc).isoformat(),
        )

    def predict_sequence(self, sequence: list[list[float]] | np.ndarray) -> dict[str, Any]:
        """Predict from an explicit sequence payload (REST convenience)."""
        array = np.asarray(sequence, dtype=np.float32)
        if array.ndim != 2:
            raise ValueError("Expected sequence shape (T, F)")
        if array.shape[1] != self.config.input_size:
            raise ValueError(
                f"Expected feature dimension {self.config.input_size}, got {array.shape[1]}",
            )

        if self._torch_enabled and self._model is not None:
            idx, confidence = self._torch_predict(array)
        else:
            idx, confidence = self._heuristic_predict(array)

        label = self.vocabulary[idx]
        if confidence < self.config.confidence_threshold:
            label = "UNSURE"

        return {
            "label": label,
            "confidence": confidence,
            "vocabulary": self.vocabulary,
        }

