"""PyTorch dataset for unified ASL landmark sequence training."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd

from ml.data.sources.common import pad_or_trim_feature_dim

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # pragma: no cover - runtime dependency
    torch = None
    Dataset = object  # type: ignore[assignment]


@dataclass(slots=True)
class AugmentationConfig:
    """Augmentation hyperparameters for ASL landmark sequences."""

    time_warp_min: float = 0.85
    time_warp_max: float = 1.15
    gaussian_noise_sigma: float = 0.005
    frame_dropout_min: int = 1
    frame_dropout_max: int = 3
    spatial_jitter_max_abs: float = 0.01
    flip_probability: float = 0.5
    time_warp_probability: float = 0.8
    noise_probability: float = 0.7
    frame_dropout_probability: float = 0.5
    spatial_jitter_probability: float = 0.7


def _resample_sequence(sequence: np.ndarray, target_length: int) -> np.ndarray:
    if target_length <= 0:
        raise ValueError("target_length must be > 0")
    if sequence.shape[0] == target_length:
        return sequence
    if sequence.shape[0] <= 1:
        repeated = np.repeat(sequence, target_length, axis=0)
        return repeated.astype(np.float32)

    old_index = np.linspace(0.0, 1.0, sequence.shape[0], dtype=np.float32)
    new_index = np.linspace(0.0, 1.0, target_length, dtype=np.float32)
    output = np.zeros((target_length, sequence.shape[1]), dtype=np.float32)
    for dim in range(sequence.shape[1]):
        output[:, dim] = np.interp(new_index, old_index, sequence[:, dim])
    return output.astype(np.float32)


class ASLDataset(Dataset):  # type: ignore[misc]
    """
    Unified ASL training dataset.

    Returns:
        (landmark_tensor [seq_len, 126], label_index)
    """

    def __init__(
        self,
        manifest_path: Path = Path("ml/data/splits/manifest.csv"),
        label_map_path: Path = Path("ml/data/splits/label_map.json"),
        split: str | None = "train",
        sequence_length: int = 30,
        training: bool | None = None,
        augmentation_config: AugmentationConfig | None = None,
    ) -> None:
        if torch is None:
            raise RuntimeError("PyTorch is required to use ASLDataset.")

        self.manifest_path = manifest_path
        self.label_map_path = label_map_path
        self.sequence_length = sequence_length
        self.augmentation = augmentation_config or AugmentationConfig()
        self._rng = np.random.default_rng()

        frame = pd.read_csv(manifest_path)
        if split is not None and "split" in frame.columns:
            frame = frame[frame["split"] == split].copy()
        frame = frame[frame["sequence_path"].notna()].copy()
        frame["sequence_path"] = frame["sequence_path"].astype(str)
        frame = frame[frame["sequence_path"] != ""]
        frame = frame.reset_index(drop=True)
        self.frame = frame

        if label_map_path.exists():
            with label_map_path.open("r", encoding="utf-8") as file_handle:
                self.label_map = json.load(file_handle)
        else:
            labels = sorted(frame["gloss_norm"].fillna(frame["gloss"]).astype(str).str.upper().unique().tolist())
            self.label_map = {label: index for index, label in enumerate(labels)}
        self.training = training if training is not None else (split == "train")

    def __len__(self) -> int:
        return int(len(self.frame))

    def _apply_time_warp(self, sequence: np.ndarray) -> np.ndarray:
        factor = float(self._rng.uniform(self.augmentation.time_warp_min, self.augmentation.time_warp_max))
        warped_length = max(2, int(round(sequence.shape[0] * factor)))
        return _resample_sequence(sequence, warped_length)

    def _apply_gaussian_noise(self, sequence: np.ndarray) -> np.ndarray:
        noise = self._rng.normal(
            loc=0.0,
            scale=self.augmentation.gaussian_noise_sigma,
            size=sequence.shape,
        ).astype(np.float32)
        return (sequence + noise).astype(np.float32)

    def _apply_random_flip(self, sequence: np.ndarray) -> np.ndarray:
        reshaped = sequence.reshape(sequence.shape[0], 2, 21, 3).copy()
        reshaped[:, :, :, 0] *= -1.0
        return reshaped.reshape(sequence.shape[0], 126).astype(np.float32)

    def _apply_frame_dropout(self, sequence: np.ndarray) -> np.ndarray:
        if sequence.shape[0] == 0:
            return sequence
        count = int(
            self._rng.integers(
                self.augmentation.frame_dropout_min,
                self.augmentation.frame_dropout_max + 1,
            ),
        )
        count = min(max(1, count), sequence.shape[0])
        indices = self._rng.choice(sequence.shape[0], size=count, replace=False)
        output = sequence.copy()
        output[indices] = 0.0
        return output

    def _apply_spatial_jitter(self, sequence: np.ndarray) -> np.ndarray:
        reshaped = sequence.reshape(sequence.shape[0], 2, 21, 3).copy()
        jitter = self._rng.uniform(
            low=-self.augmentation.spatial_jitter_max_abs,
            high=self.augmentation.spatial_jitter_max_abs,
            size=(1, 2, 1, 3),
        ).astype(np.float32)
        reshaped += jitter
        return reshaped.reshape(sequence.shape[0], 126).astype(np.float32)

    def _augment(self, sequence: np.ndarray) -> np.ndarray:
        output = sequence.astype(np.float32)

        if self._rng.random() < self.augmentation.time_warp_probability:
            output = self._apply_time_warp(output)
        if self._rng.random() < self.augmentation.noise_probability:
            output = self._apply_gaussian_noise(output)
        if self._rng.random() < self.augmentation.flip_probability:
            output = self._apply_random_flip(output)
        if self._rng.random() < self.augmentation.frame_dropout_probability:
            output = self._apply_frame_dropout(output)
        if self._rng.random() < self.augmentation.spatial_jitter_probability:
            output = self._apply_spatial_jitter(output)
        return output.astype(np.float32)

    def _pad_or_truncate_length(self, sequence: np.ndarray) -> np.ndarray:
        if sequence.shape[0] == self.sequence_length:
            return sequence
        if sequence.shape[0] > self.sequence_length:
            return _resample_sequence(sequence, self.sequence_length)
        if sequence.shape[0] == 0:
            return np.zeros((self.sequence_length, 126), dtype=np.float32)
        pad = np.zeros((self.sequence_length - sequence.shape[0], sequence.shape[1]), dtype=np.float32)
        return np.concatenate([sequence, pad], axis=0).astype(np.float32)

    def __getitem__(self, index: int) -> tuple["torch.Tensor", int]:
        row = self.frame.iloc[index]
        sequence_path = Path(str(row["sequence_path"]))
        sequence = np.load(sequence_path).astype(np.float32)
        sequence = pad_or_trim_feature_dim(sequence, feature_dim=126)

        if self.training:
            sequence = self._augment(sequence)
        sequence = self._pad_or_truncate_length(sequence)

        gloss_key = str(row.get("gloss_norm") or row.get("gloss", "")).strip().upper()
        label = self.label_map.get(gloss_key)
        if label is None:
            # Fallback for manifests with lowercase/non-normalized labels.
            label = self.label_map.get(" ".join(gloss_key.split()), 0)

        features = torch.from_numpy(sequence.astype(np.float32))
        return features, int(label)


def load_label_map(label_map_path: Path = Path("ml/data/splits/label_map.json")) -> dict[str, int]:
    """Load label map utility for downstream scripts."""
    with label_map_path.open("r", encoding="utf-8") as file_handle:
        payload = json.load(file_handle)
    return {str(key): int(value) for key, value in payload.items()}

