"""Synthetic temporal dataset generation for gesture prototyping."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - runtime optional import guard
    torch = None
    Dataset = object  # type: ignore[assignment]


def _seed_from_label(label: str) -> int:
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _gesture_signature(label: str, feature_dim: int) -> np.ndarray:
    rng = np.random.default_rng(_seed_from_label(label))
    return rng.normal(loc=0.0, scale=1.0, size=(feature_dim,)).astype(np.float32)


@dataclass(slots=True)
class SyntheticDataConfig:
    """Configuration for synthetic gesture generation."""

    vocabulary: list[str]
    samples_per_class: int = 256
    sequence_length: int = 30
    feature_dim: int = 63
    noise_scale: float = 0.05


def generate_synthetic_sequences(config: SyntheticDataConfig) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic (N, T, F) gesture sequences and class labels."""
    sequences: list[np.ndarray] = []
    labels: list[int] = []

    time_axis = np.linspace(0.0, 2.0 * np.pi, config.sequence_length, dtype=np.float32)

    for class_idx, label in enumerate(config.vocabulary):
        signature = _gesture_signature(label, config.feature_dim)
        freq_rng = np.random.default_rng(_seed_from_label(f"{label}_freq"))
        phase_rng = np.random.default_rng(_seed_from_label(f"{label}_phase"))
        amplitude_rng = np.random.default_rng(_seed_from_label(f"{label}_amp"))

        frequency = freq_rng.uniform(0.5, 2.5, size=(config.feature_dim,)).astype(np.float32)
        phase = phase_rng.uniform(0.0, np.pi, size=(config.feature_dim,)).astype(np.float32)
        amplitude = amplitude_rng.uniform(0.1, 1.2, size=(config.feature_dim,)).astype(np.float32)

        for _ in range(config.samples_per_class):
            waveform = amplitude * np.sin(time_axis[:, None] * frequency[None, :] + phase[None, :])
            trend = np.linspace(0.0, 1.0, config.sequence_length, dtype=np.float32)[:, None] * signature[None, :] * 0.03
            noise = np.random.normal(
                loc=0.0,
                scale=config.noise_scale,
                size=(config.sequence_length, config.feature_dim),
            ).astype(np.float32)
            sequence = waveform + trend + noise
            sequences.append(sequence.astype(np.float32))
            labels.append(class_idx)

    return np.stack(sequences, axis=0), np.asarray(labels, dtype=np.int64)


class SyntheticGestureDataset(Dataset):  # type: ignore[misc]
    """PyTorch Dataset wrapper for synthetic gesture sequences."""

    def __init__(self, sequences: np.ndarray, labels: np.ndarray) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required to instantiate SyntheticGestureDataset.")
        self._x = torch.from_numpy(sequences.astype(np.float32))
        self._y = torch.from_numpy(labels.astype(np.int64))

    def __len__(self) -> int:
        return int(self._x.shape[0])

    def __getitem__(self, idx: int) -> tuple["torch.Tensor", "torch.Tensor"]:
        return self._x[idx], self._y[idx]

