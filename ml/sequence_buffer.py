"""Temporal sliding window utilities for gesture sequences."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


@dataclass
class SequenceBuffer:
    """Fixed-size sliding buffer for landmark vectors."""

    sequence_length: int
    feature_dim: int
    _buffer: deque[np.ndarray] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.sequence_length <= 0:
            raise ValueError("sequence_length must be > 0")
        if self.feature_dim <= 0:
            raise ValueError("feature_dim must be > 0")
        self._buffer = deque(maxlen=self.sequence_length)

    def add(self, frame: Iterable[float] | np.ndarray) -> None:
        """Add one frame to the buffer, validating dimensionality."""
        arr = np.asarray(frame, dtype=np.float32).reshape(-1)
        if arr.shape[0] != self.feature_dim:
            raise ValueError(f"Expected feature_dim={self.feature_dim}, got={arr.shape[0]}")
        self._buffer.append(arr)

    def clear(self) -> None:
        """Reset buffer state."""
        self._buffer.clear()

    @property
    def is_full(self) -> bool:
        """True when enough frames are available for sequence inference."""
        return len(self._buffer) == self.sequence_length

    @property
    def size(self) -> int:
        """Current number of buffered frames."""
        return len(self._buffer)

    def as_array(self) -> np.ndarray:
        """Return buffered sequence as (T, F) ndarray."""
        if not self._buffer:
            return np.empty((0, self.feature_dim), dtype=np.float32)
        return np.stack(self._buffer, axis=0).astype(np.float32)

    def as_batched_array(self) -> np.ndarray:
        """Return buffered sequence as (1, T, F) ndarray for model input."""
        sequence = self.as_array()
        if sequence.shape[0] == 0:
            return np.empty((0, 0, self.feature_dim), dtype=np.float32)
        return np.expand_dims(sequence, axis=0)

