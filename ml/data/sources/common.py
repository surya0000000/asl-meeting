"""Shared utilities for ASL dataset source ingestion."""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

try:
    import cv2
    import mediapipe as mp
except ImportError:  # pragma: no cover - runtime optional dependency
    cv2 = None
    mp = None


LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class SampleRecord:
    """Canonical output metadata for one sample sequence."""

    sample_id: str
    source: str
    gloss: str
    sequence_path: str
    num_frames: int
    feature_dim: int
    metadata_json: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "source": self.source,
            "gloss": self.gloss,
            "sequence_path": self.sequence_path,
            "num_frames": self.num_frames,
            "feature_dim": self.feature_dim,
            "metadata_json": self.metadata_json,
        }


def slugify_gloss(gloss: str) -> str:
    """Convert gloss text into filesystem-safe token."""
    text = gloss.strip().upper()
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Z0-9_]+", "", text)
    return text or "UNKNOWN"


def ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_manifest(records: list[SampleRecord], manifest_path: Path) -> pd.DataFrame:
    """Persist source records as CSV manifest."""
    ensure_directory(manifest_path.parent)
    frame = pd.DataFrame([record.as_dict() for record in records])
    if frame.empty:
        frame = pd.DataFrame(
            columns=[
                "sample_id",
                "source",
                "gloss",
                "sequence_path",
                "num_frames",
                "feature_dim",
                "metadata_json",
            ],
        )
    frame.to_csv(manifest_path, index=False)
    LOGGER.info("Saved manifest: %s (%d rows)", manifest_path, len(frame))
    return frame


def download_file(url: str, destination: Path, chunk_size: int = 2**20) -> Path:
    """Download file from URL with progress bar."""
    ensure_directory(destination.parent)
    response = requests.get(url, timeout=60, stream=True)
    response.raise_for_status()
    total = int(response.headers.get("content-length", "0"))
    with destination.open("wb") as file_handle:
        with tqdm(total=total, unit="B", unit_scale=True, desc=f"Downloading {destination.name}") as progress:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if not chunk:
                    continue
                file_handle.write(chunk)
                progress.update(len(chunk))
    return destination


def run_yt_dlp(url: str, output_path: Path) -> Path | None:
    """Download one video using yt-dlp. Returns output path or None on failure."""
    ensure_directory(output_path.parent)
    cmd = [
        "yt-dlp",
        "--no-progress",
        "--no-warnings",
        "-f",
        "mp4/best",
        "-o",
        str(output_path),
        url,
    ]
    try:
        completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except FileNotFoundError:
        LOGGER.error("yt-dlp command not available in environment.")
        return None
    if completed.returncode != 0:
        LOGGER.warning("yt-dlp failed for %s: %s", url, completed.stderr.strip())
        return None
    if output_path.exists():
        return output_path
    # yt-dlp may change extension if container differs.
    candidates = sorted(output_path.parent.glob(f"{output_path.stem}.*"))
    return candidates[0] if candidates else None


def pad_or_trim_feature_dim(sequence: np.ndarray, feature_dim: int = 126) -> np.ndarray:
    """
    Normalize any sequence feature dimension to requested width.

    - 63 -> 126 by zero-padding missing second hand.
    - >feature_dim -> truncated.
    """
    if sequence.ndim != 2:
        raise ValueError(f"Expected sequence ndim=2, got {sequence.shape}")
    if sequence.shape[1] == feature_dim:
        return sequence.astype(np.float32)
    if sequence.shape[1] == 63 and feature_dim == 126:
        zeros = np.zeros((sequence.shape[0], 63), dtype=np.float32)
        return np.concatenate([sequence.astype(np.float32), zeros], axis=1)
    if sequence.shape[1] < feature_dim:
        width = feature_dim - sequence.shape[1]
        zeros = np.zeros((sequence.shape[0], width), dtype=np.float32)
        return np.concatenate([sequence.astype(np.float32), zeros], axis=1)
    return sequence[:, :feature_dim].astype(np.float32)


def collapse_to_one_hand(two_hand_sequence: np.ndarray) -> np.ndarray:
    """Collapse (T,126) into dominant hand (T,63)."""
    if two_hand_sequence.ndim != 2 or two_hand_sequence.shape[1] != 126:
        raise ValueError("Expected two-hand sequence shape (T,126).")
    left = two_hand_sequence[:, :63]
    right = two_hand_sequence[:, 63:]
    left_energy = float(np.mean(np.abs(left)))
    right_energy = float(np.mean(np.abs(right)))
    return left if left_energy >= right_energy else right


class VideoLandmarkExtractor:
    """Extract MediaPipe hand landmark sequences from video."""

    def __init__(
        self,
        max_num_hands: int = 2,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.max_num_hands = max_num_hands
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        self._hands = None
        if mp is not None:
            self._hands = mp.solutions.hands.Hands(
                static_image_mode=False,
                max_num_hands=max_num_hands,
                min_detection_confidence=min_detection_confidence,
                min_tracking_confidence=min_tracking_confidence,
            )

    def extract(self, video_path: Path) -> tuple[np.ndarray | None, int]:
        """
        Extract frame-wise hand landmarks.

        Returns:
            sequence: (T,126) float32 or None if extraction failed.
            max_hands_seen: maximum number of hands detected in stream.
        """
        if cv2 is None or self._hands is None:
            LOGGER.error("opencv-python-headless and mediapipe are required for extraction.")
            return None, 0
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            LOGGER.warning("Unable to open video: %s", video_path)
            return None, 0

        frames: list[np.ndarray] = []
        max_hands_seen = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self._hands.process(rgb)

                frame_landmarks = np.zeros((2, 21, 3), dtype=np.float32)
                used_slots: set[int] = set()
                detected_count = 0

                if result.multi_hand_landmarks:
                    for index, hand in enumerate(result.multi_hand_landmarks):
                        coords = np.array(
                            [[lm.x, lm.y, lm.z] for lm in hand.landmark],
                            dtype=np.float32,
                        )
                        slot = detected_count if detected_count < 2 else 1
                        if result.multi_handedness and index < len(result.multi_handedness):
                            label = result.multi_handedness[index].classification[0].label.lower()
                            # Keep deterministic slot ordering: left=0, right=1.
                            if label == "left":
                                slot = 0
                            elif label == "right":
                                slot = 1
                        if slot in used_slots:
                            slot = 0 if 0 not in used_slots else 1
                        frame_landmarks[slot] = coords
                        used_slots.add(slot)
                        detected_count += 1

                max_hands_seen = max(max_hands_seen, detected_count)
                frames.append(frame_landmarks.reshape(-1))
        finally:
            capture.release()

        if not frames:
            return None, 0
        return np.stack(frames, axis=0).astype(np.float32), max_hands_seen

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._hands is not None:
            self._hands.close()


def write_npy(sequence: np.ndarray, path: Path) -> Path:
    """Save sequence array to .npy path."""
    ensure_directory(path.parent)
    np.save(path, sequence.astype(np.float32))
    return path


def infer_gloss_from_filename(path: Path) -> str:
    """Infer gloss token from filename using conservative heuristics."""
    stem = path.stem.upper()
    stem = re.sub(r"[_\-]+", " ", stem)
    parts = [token for token in stem.split() if token and not token.isdigit()]
    if not parts:
        return "UNKNOWN"
    # Keep up to first three informative words.
    return " ".join(parts[:3])


def parse_landmark_csv(csv_path: Path) -> np.ndarray | None:
    """
    Parse landmark CSV into (T, F) float array.

    Supports:
    - Wide format: one frame per row with >=63 numeric columns.
    - Long format: columns including frame + x/y/z (+ optional hand + landmark).
    """
    try:
        frame = pd.read_csv(csv_path)
    except Exception:
        LOGGER.exception("Failed to parse CSV: %s", csv_path)
        return None
    if frame.empty:
        return None

    # Wide format fast path.
    numeric = frame.select_dtypes(include=["number"])
    if numeric.shape[1] >= 63 and "x" not in frame.columns:
        values = numeric.to_numpy(dtype=np.float32)
        feature_dim = 126 if values.shape[1] >= 126 else 63
        return values[:, :feature_dim].astype(np.float32)

    lower_map = {column.lower(): column for column in frame.columns}
    frame_col = lower_map.get("frame") or lower_map.get("frame_id") or lower_map.get("frame_idx")
    x_col = lower_map.get("x")
    y_col = lower_map.get("y")
    z_col = lower_map.get("z")
    if not (frame_col and x_col and y_col and z_col):
        numeric_values = numeric.to_numpy(dtype=np.float32)
        if numeric_values.shape[1] >= 63:
            feature_dim = 126 if numeric_values.shape[1] >= 126 else 63
            return numeric_values[:, :feature_dim].astype(np.float32)
        LOGGER.warning("No compatible landmark columns found in %s", csv_path)
        return None

    landmark_col = (
        lower_map.get("landmark")
        or lower_map.get("landmark_id")
        or lower_map.get("point")
        or lower_map.get("joint")
    )
    hand_col = lower_map.get("hand") or lower_map.get("hand_id") or lower_map.get("side")
    if landmark_col is None:
        # If no landmark index exists, fall back to ordering rows per frame.
        rows = []
        for _, sub in frame.groupby(frame_col, sort=True):
            coords = sub[[x_col, y_col, z_col]].to_numpy(dtype=np.float32).reshape(-1)
            if coords.size >= 126:
                rows.append(coords[:126])
            elif coords.size >= 63:
                padded = np.zeros(126, dtype=np.float32)
                padded[: coords.size] = coords[: min(coords.size, 126)]
                rows.append(padded)
        return np.stack(rows, axis=0) if rows else None

    rows = []
    for _, sub in frame.groupby(frame_col, sort=True):
        output = np.zeros((2, 21, 3), dtype=np.float32)
        if hand_col:
            hand_groups = list(sub.groupby(hand_col))
        else:
            hand_groups = [("hand0", sub)]
        for hand_index, (_, hand_frame) in enumerate(hand_groups[:2]):
            for _, row in hand_frame.iterrows():
                point = int(row[landmark_col]) if not pd.isna(row[landmark_col]) else -1
                if 0 <= point < 21:
                    output[hand_index, point, 0] = float(row[x_col])
                    output[hand_index, point, 1] = float(row[y_col])
                    output[hand_index, point, 2] = float(row[z_col])
        rows.append(output.reshape(-1))
    if not rows:
        return None
    sequence = np.stack(rows, axis=0).astype(np.float32)
    max_dim = 126 if sequence.shape[1] >= 126 else 63
    return sequence[:, :max_dim]


def json_dumps(data: dict[str, Any]) -> str:
    """Serialize metadata payload safely."""
    return json.dumps(data, ensure_ascii=True, sort_keys=True)

