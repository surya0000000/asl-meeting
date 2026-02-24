"""Fingerspelling source loader for A-Z + SPACE landmark sequences."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import re
import subprocess

from tqdm import tqdm

from ml.data.sources.common import (
    SampleRecord,
    ensure_directory,
    json_dumps,
    parse_landmark_csv,
    save_manifest,
    slugify_gloss,
    write_npy,
)


LOGGER = logging.getLogger(__name__)

DEFAULT_KAGGLE_DATASET = "sttaseen/asl-fingerspelling-dataset"


class FingerspellingSource:
    """Load landmark CSVs from Kaggle fingerspelling dataset."""

    source_name = "fingerspelling"
    valid_labels = {chr(index) for index in range(ord("A"), ord("Z") + 1)} | {"SPACE"}

    def __init__(self, data_root: Path = Path("ml/data")) -> None:
        self.data_root = data_root
        self.raw_root = ensure_directory(data_root / "raw" / self.source_name)
        self.download_root = ensure_directory(self.raw_root / "kaggle")
        self.processed_root = ensure_directory(data_root / "processed" / self.source_name)
        self.sequence_root = ensure_directory(self.processed_root / "sequences")
        self.manifest_path = self.processed_root / "manifest.csv"

    def download_from_kaggle(
        self,
        dataset_slug: str = DEFAULT_KAGGLE_DATASET,
        force: bool = False,
    ) -> Path:
        """Download and unzip fingerspelling dataset via Kaggle CLI."""
        marker = self.download_root / ".download_complete"
        if marker.exists() and not force:
            return self.download_root

        cmd = [
            "kaggle",
            "datasets",
            "download",
            "-d",
            dataset_slug,
            "-p",
            str(self.download_root),
            "--unzip",
            "--force" if force else "--quiet",
        ]
        try:
            completed = subprocess.run(cmd, check=False, capture_output=True, text=True)
        except FileNotFoundError as exc:
            raise RuntimeError(
                "Kaggle CLI not found. Install dependency and configure API credentials.",
            ) from exc
        if completed.returncode != 0:
            raise RuntimeError(
                f"Kaggle download failed for {dataset_slug}: {completed.stderr.strip()}",
            )
        marker.write_text("ok", encoding="utf-8")
        return self.download_root

    @classmethod
    def _resolve_gloss(cls, csv_path: Path) -> str:
        parent_token = csv_path.parent.name.strip().upper()
        if parent_token in cls.valid_labels:
            return parent_token
        stem = csv_path.stem.strip().upper().replace("-", "_")
        if stem == "SPACE":
            return "SPACE"
        if len(stem) == 1 and stem in cls.valid_labels:
            return stem
        for token in re.split(r"[^A-Z]+", stem):
            if token in cls.valid_labels:
                return token
        return "SPACE" if "SPACE" in stem else "UNKNOWN"

    def build(
        self,
        dataset_slug: str = DEFAULT_KAGGLE_DATASET,
        local_path: Path | None = None,
        force_download: bool = False,
    ):
        """Build fingerspelling landmark manifest from raw CSV files."""
        if local_path is None:
            root = self.download_from_kaggle(dataset_slug=dataset_slug, force=force_download)
        else:
            root = local_path
        csv_files = sorted(root.rglob("*.csv"))
        records: list[SampleRecord] = []

        for index, csv_path in enumerate(tqdm(csv_files, desc="Fingerspelling CSV parse")):
            sequence = parse_landmark_csv(csv_path)
            if sequence is None or sequence.size == 0:
                continue
            gloss = self._resolve_gloss(csv_path)
            if gloss not in self.valid_labels:
                # Keep this dataset focused on letter-level and SPACE labels.
                continue
            gloss_slug = slugify_gloss(gloss)
            sample_id = f"fingerspelling_{gloss_slug}_{index:06d}"
            output_path = self.sequence_root / gloss_slug / f"{sample_id}.npy"
            write_npy(sequence, output_path)
            records.append(
                SampleRecord(
                    sample_id=sample_id,
                    source=self.source_name,
                    gloss=gloss,
                    sequence_path=str(output_path),
                    num_frames=int(sequence.shape[0]),
                    feature_dim=int(sequence.shape[1]),
                    metadata_json=json_dumps({"source_csv": str(csv_path)}),
                ),
            )

        return save_manifest(records, self.manifest_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build fingerspelling source manifest.")
    parser.add_argument("--data-root", type=Path, default=Path("ml/data"))
    parser.add_argument("--dataset-slug", type=str, default=DEFAULT_KAGGLE_DATASET)
    parser.add_argument("--local-path", type=Path, default=None)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    source = FingerspellingSource(data_root=args.data_root)
    source.build(
        dataset_slug=args.dataset_slug,
        local_path=args.local_path,
        force_download=args.force_download,
    )


if __name__ == "__main__":
    main()

