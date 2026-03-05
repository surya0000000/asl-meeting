"""ONNX runtime inference for landmark sequence classification."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from ml import DEFAULT_VOCABULARY
from ml.config import LANDMARK_DIM, SEQUENCE_LENGTH

try:
    import onnxruntime as ort
except ImportError:  # pragma: no cover - runtime optional dependency
    ort = None


LOGGER = logging.getLogger(__name__)


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=-1, keepdims=True)


class ONNXInferenceEngine:
    """Run top-k class predictions using ONNX Runtime."""

    def __init__(
        self,
        model_path: Path = Path("ml/models/gesture_model.onnx"),
        label_map_path: Path = Path("ml/models/label_map.json"),
    ) -> None:
        self.model_path = model_path
        self.label_map_path = label_map_path
        self.labels = self._load_label_map(label_map_path)
        self._session: "ort.InferenceSession | None" = None
        self._input_name = "landmarks"
        self._fallback = True

        if ort is None:
            LOGGER.warning("onnxruntime unavailable. Falling back to heuristic predictions.")
            return
        if not model_path.exists():
            LOGGER.warning("ONNX model not found at %s. Falling back to heuristic predictions.", model_path)
            return

        self._session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self._input_name = self._session.get_inputs()[0].name
        self._fallback = False
        self._warm_up()

    @staticmethod
    def _load_label_map(path: Path) -> dict[int, str]:
        if path.exists():
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    out: dict[int, str] = {}
                    for key, value in payload.items():
                        out[int(key)] = str(value)
                    if out:
                        return out
            except Exception:
                LOGGER.exception("Failed to load label map: %s", path)
        return {idx: gloss for idx, gloss in enumerate(DEFAULT_VOCABULARY)}

    def _warm_up(self) -> None:
        if self._session is None:
            return
        dummy = np.zeros((1, SEQUENCE_LENGTH, LANDMARK_DIM), dtype=np.float32)
        for _ in range(5):
            _ = self._session.run(None, {self._input_name: dummy})
        LOGGER.info("ONNX inference session warmed up.")

    @staticmethod
    def _coerce_sequence(sequence: np.ndarray) -> np.ndarray:
        if sequence.ndim != 2:
            raise ValueError(f"Expected (T,F) sequence, got {sequence.shape}")

        seq = sequence.astype(np.float32)
        # Feature dimension -> 126
        if seq.shape[1] < LANDMARK_DIM:
            padded = np.zeros((seq.shape[0], LANDMARK_DIM), dtype=np.float32)
            padded[:, : seq.shape[1]] = seq
            seq = padded
        elif seq.shape[1] > LANDMARK_DIM:
            seq = seq[:, :LANDMARK_DIM]

        # Length -> 30
        if seq.shape[0] < SEQUENCE_LENGTH:
            padded = np.zeros((SEQUENCE_LENGTH, LANDMARK_DIM), dtype=np.float32)
            padded[-seq.shape[0] :] = seq
            seq = padded
        elif seq.shape[0] > SEQUENCE_LENGTH:
            seq = seq[-SEQUENCE_LENGTH:]
        return seq.astype(np.float32)

    def _fallback_logits(self, sequence: np.ndarray) -> np.ndarray:
        num_classes = max(1, len(self.labels))
        motion = float(np.mean(np.abs(np.diff(sequence, axis=0)))) if sequence.shape[0] > 1 else 0.0
        energy = float(np.mean(np.abs(sequence)))
        center = int(abs(motion * 1000.0 + energy * 100.0)) % num_classes
        logits = np.full((num_classes,), -1.0, dtype=np.float32)
        logits[center] = 2.0
        logits[(center + 1) % num_classes] = 1.0
        logits[(center + 2) % num_classes] = 0.5
        return logits

    def predict(self, sequence: np.ndarray) -> dict[str, Any]:
        """Return top-3 predictions with confidences."""
        seq = self._coerce_sequence(np.asarray(sequence, dtype=np.float32))
        input_batch = np.expand_dims(seq, axis=0).astype(np.float32)

        if self._session is not None and not self._fallback:
            output = self._session.run(None, {self._input_name: input_batch})
            logits = np.asarray(output[0], dtype=np.float32)[0]
        else:
            logits = self._fallback_logits(seq)

        probs = _softmax(logits)
        indices = np.argsort(probs)[::-1][:3]
        top_predictions = [
            {
                "index": int(idx),
                "gloss": self.labels.get(int(idx), f"LABEL_{int(idx)}"),
                "confidence": float(probs[int(idx)]),
            }
            for idx in indices
        ]
        top = top_predictions[0]
        return {
            "top_predictions": top_predictions,
            "raw_prediction": top["gloss"],
            "confidence": top["confidence"],
        }

