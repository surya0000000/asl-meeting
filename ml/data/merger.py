"""Merge multi-source ASL landmarks into unified train/val/test manifests."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from ml.data.sources.common import ensure_directory, pad_or_trim_feature_dim, slugify_gloss


LOGGER = logging.getLogger(__name__)

SOURCE_PRIORITY = {
    "wlasl": 3,
    "msasl": 2,
    "asllvd": 1,
}


def _canonical_gloss(gloss: str) -> str:
    return " ".join(str(gloss).strip().upper().split())


def _reshape_to_hands(sequence: np.ndarray) -> tuple[np.ndarray, int]:
    """Convert sequence into (T, H, 21, 3) representation."""
    if sequence.ndim != 2:
        raise ValueError(f"Expected 2D sequence, got {sequence.shape}")
    if sequence.shape[1] == 63:
        reshaped = sequence.reshape(sequence.shape[0], 1, 21, 3)
        return reshaped.astype(np.float32), 1
    if sequence.shape[1] >= 126:
        reshaped = sequence[:, :126].reshape(sequence.shape[0], 2, 21, 3)
        return reshaped.astype(np.float32), 2
    padded = pad_or_trim_feature_dim(sequence, feature_dim=126)
    reshaped = padded.reshape(padded.shape[0], 2, 21, 3)
    return reshaped.astype(np.float32), 2


def normalize_landmark_sequence(sequence: np.ndarray) -> np.ndarray:
    """
    Normalize landmark sequence for camera distance invariance.

    Steps:
    - Center each hand on wrist (landmark 0).
    - Scale by palm size (distance between MCP5 and MCP17; fallback wrist->MCP9).
    - Output fixed 126-dim shape with zero-padded missing hand.
    """
    hand_seq, num_hands = _reshape_to_hands(sequence)
    normalized = np.zeros_like(hand_seq, dtype=np.float32)

    for hand_idx in range(num_hands):
        coords = hand_seq[:, hand_idx, :, :]
        frame_activity = np.any(np.abs(coords) > 1e-8, axis=(1, 2))
        if not np.any(frame_activity):
            continue

        wrist = coords[:, 0:1, :]  # (T,1,3)
        centered = coords - wrist

        palm_vec = centered[:, 5, :] - centered[:, 17, :]
        palm_size = np.linalg.norm(palm_vec, axis=1)
        fallback = np.linalg.norm(centered[:, 9, :], axis=1)
        palm_size = np.where(palm_size > 1e-6, palm_size, fallback)
        palm_size = np.where(palm_size > 1e-6, palm_size, 1.0).astype(np.float32)

        centered = centered / palm_size[:, None, None]
        centered[~frame_activity] = 0.0
        normalized[:, hand_idx, :, :] = centered

    out = normalized.reshape(normalized.shape[0], -1)
    out = pad_or_trim_feature_dim(out, feature_dim=126)
    return out.astype(np.float32)


def _collect_manifest_paths(data_root: Path, explicit_paths: Iterable[Path] | None = None) -> list[Path]:
    if explicit_paths:
        return [Path(path) for path in explicit_paths]
    processed_root = data_root / "processed"
    return sorted(path for path in processed_root.glob("*/manifest.csv") if path.exists())


def _load_source_rows(manifest_paths: list[Path]) -> pd.DataFrame:
    frames = []
    for manifest_path in manifest_paths:
        try:
            frame = pd.read_csv(manifest_path)
        except Exception:
            LOGGER.warning("Unable to read manifest: %s", manifest_path)
            continue
        frame["manifest_path"] = str(manifest_path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    merged = pd.concat(frames, axis=0, ignore_index=True)
    if "source" not in merged.columns:
        merged["source"] = merged["manifest_path"].apply(lambda path: Path(path).parent.name)
    return merged


def _dedupe_glosses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate overlapping glosses by priority:
    WLASL > MS-ASL > ASLLVD.
    """
    if df.empty:
        return df
    df = df.copy()
    df["gloss_norm"] = df["gloss"].map(_canonical_gloss)

    preferred_sources = set(SOURCE_PRIORITY.keys())
    preferred = df[df["source"].isin(preferred_sources)].copy()
    retained_indices: set[int] = set()

    for gloss, group in preferred.groupby("gloss_norm"):
        _ = gloss
        available_sources = group["source"].unique().tolist()
        best_source = max(available_sources, key=lambda src: SOURCE_PRIORITY.get(src, 0))
        retained_indices.update(group[group["source"] == best_source].index.tolist())

    non_preferred = df[~df["source"].isin(preferred_sources)]
    preferred_kept = df.loc[sorted(retained_indices)] if retained_indices else df.iloc[0:0]
    merged = pd.concat([preferred_kept, non_preferred], axis=0, ignore_index=True)
    return merged


def _split_group_indices(
    size: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = rng.permutation(size)
    if size == 1:
        return order, np.array([], dtype=int), np.array([], dtype=int)
    if size == 2:
        return order[:1], order[1:2], np.array([], dtype=int)

    train_count = max(1, int(round(size * 0.8)))
    val_count = max(1, int(round(size * 0.1)))
    test_count = size - train_count - val_count
    if test_count < 1:
        if train_count >= val_count and train_count > 1:
            train_count -= 1
        elif val_count > 1:
            val_count -= 1
        test_count = size - train_count - val_count
    if test_count < 1:
        test_count = 1
        if train_count > 1:
            train_count -= 1
        else:
            val_count = max(1, val_count - 1)

    train_end = train_count
    val_end = train_count + val_count
    return order[:train_end], order[train_end:val_end], order[val_end : val_end + test_count]


def stratified_split_per_gloss(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Assign train/val/test split with 80/10/10 ratio per gloss."""
    if df.empty:
        return df
    rng = np.random.default_rng(seed)
    output = df.copy()
    output["split"] = "train"

    for gloss, group in output.groupby("gloss_norm"):
        _ = gloss
        group_indices = group.index.to_numpy()
        local_train, local_val, local_test = _split_group_indices(len(group_indices), rng)
        if len(local_train) > 0:
            output.loc[group_indices[local_train], "split"] = "train"
        if len(local_val) > 0:
            output.loc[group_indices[local_val], "split"] = "val"
        if len(local_test) > 0:
            output.loc[group_indices[local_test], "split"] = "test"
    return output


class ASLDataMerger:
    """Merge and normalize ASL data from all source manifests."""

    def __init__(self, data_root: Path = Path("ml/data")) -> None:
        self.data_root = data_root
        self.splits_root = ensure_directory(data_root / "splits")
        self.sequence_root = ensure_directory(self.splits_root / "sequences")
        self.manifest_output = self.splits_root / "manifest.csv"
        self.label_map_output = self.splits_root / "label_map.json"
        self.asllex_hints_path = data_root / "processed" / "asllex" / "phonology_hints.json"

    def _load_asllex_hints(self) -> dict[str, dict[str, object]]:
        if not self.asllex_hints_path.exists():
            return {}
        try:
            return json.loads(self.asllex_hints_path.read_text(encoding="utf-8"))
        except Exception:
            LOGGER.warning("Unable to parse ASL-LEX hints file: %s", self.asllex_hints_path)
            return {}

    def merge(
        self,
        manifest_paths: Iterable[Path] | None = None,
        seed: int = 42,
    ) -> pd.DataFrame:
        """Merge source manifests, normalize sequences, dedupe, and split."""
        paths = _collect_manifest_paths(self.data_root, manifest_paths)
        merged = _load_source_rows(paths)
        if merged.empty:
            LOGGER.warning("No source manifests found to merge.")
            empty = pd.DataFrame(
                columns=[
                    "sample_id",
                    "source",
                    "gloss",
                    "sequence_path",
                    "num_frames",
                    "feature_dim",
                    "metadata_json",
                    "gloss_norm",
                    "split",
                    "label_index",
                ],
            )
            empty.to_csv(self.manifest_output, index=False)
            self.label_map_output.write_text("{}", encoding="utf-8")
            return empty

        # Keep sequence-bearing rows only.
        merged["sequence_path"] = merged["sequence_path"].fillna("").astype(str)
        merged = merged[merged["sequence_path"] != ""].copy()
        merged = merged[merged["sequence_path"].map(lambda path: Path(path).exists())]
        merged = merged[merged["num_frames"].fillna(0).astype(int) > 0]

        deduped = _dedupe_glosses(merged)
        deduped = deduped.drop_duplicates(subset=["sample_id", "source"], keep="first").reset_index(drop=True)

        # Normalize and re-materialize sequences to unified split directory.
        normalized_paths: list[str] = []
        normalized_frames: list[int] = []
        normalized_dims: list[int] = []
        for _, row in deduped.iterrows():
            source = str(row["source"]).strip().lower()
            gloss = _canonical_gloss(row["gloss"])
            sample_id = str(row["sample_id"])
            input_path = Path(str(row["sequence_path"]))
            sequence = np.load(input_path)
            normalized = normalize_landmark_sequence(sequence)

            output_path = self.sequence_root / source / slugify_gloss(gloss) / f"{sample_id}.npy"
            ensure_directory(output_path.parent)
            np.save(output_path, normalized.astype(np.float32))
            normalized_paths.append(str(output_path))
            normalized_frames.append(int(normalized.shape[0]))
            normalized_dims.append(int(normalized.shape[1]))

        deduped["sequence_path"] = normalized_paths
        deduped["num_frames"] = normalized_frames
        deduped["feature_dim"] = normalized_dims
        deduped["gloss_norm"] = deduped["gloss"].map(_canonical_gloss)

        hints = self._load_asllex_hints()
        deduped["has_asllex_hint"] = deduped["gloss_norm"].map(lambda gloss: gloss in hints)
        deduped["asllex_feature_dim"] = deduped["gloss_norm"].map(
            lambda gloss: int(hints.get(gloss, {}).get("feature_dim", 0)),
        )

        split = stratified_split_per_gloss(deduped, seed=seed)
        labels = sorted(split["gloss_norm"].unique().tolist())
        label_map = {label: idx for idx, label in enumerate(labels)}
        split["label_index"] = split["gloss_norm"].map(label_map).astype(int)

        split.to_csv(self.manifest_output, index=False)
        self.label_map_output.write_text(json.dumps(label_map, ensure_ascii=True, indent=2), encoding="utf-8")
        LOGGER.info(
            "Merged dataset saved: %s (%d samples, %d classes)",
            self.manifest_output,
            len(split),
            len(label_map),
        )
        return split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge ASL source manifests.")
    parser.add_argument("--data-root", type=Path, default=Path("ml/data"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--manifest", type=Path, nargs="*", default=None)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    merger = ASLDataMerger(data_root=args.data_root)
    merger.merge(manifest_paths=args.manifest, seed=args.seed)


if __name__ == "__main__":
    main()

