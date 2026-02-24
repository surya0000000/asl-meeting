"""WLASL source loader with video download and MediaPipe landmark extraction."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from tqdm import tqdm

from ml.data.sources.common import (
    SampleRecord,
    VideoLandmarkExtractor,
    collapse_to_one_hand,
    download_file,
    ensure_directory,
    json_dumps,
    run_yt_dlp,
    save_manifest,
    slugify_gloss,
    write_npy,
)


LOGGER = logging.getLogger(__name__)

WLASL_METADATA_URL = "https://raw.githubusercontent.com/dxli94/WLASL/master/start_kit/WLASL_v0.3.json"


class WLASLSource:
    """Ingest WLASL dataset and export landmark sequence manifest."""

    source_name = "wlasl"

    def __init__(self, data_root: Path = Path("ml/data")) -> None:
        self.data_root = data_root
        self.raw_root = ensure_directory(data_root / "raw" / self.source_name)
        self.video_root = ensure_directory(self.raw_root / "videos")
        self.processed_root = ensure_directory(data_root / "processed" / self.source_name)
        self.sequence_root = ensure_directory(self.processed_root / "sequences")
        self.manifest_path = self.processed_root / "manifest.csv"
        self.metadata_path = self.raw_root / "WLASL_v0.3.json"

    def download_metadata(self, force: bool = False) -> Path:
        """Download canonical WLASL metadata JSON."""
        if self.metadata_path.exists() and not force:
            return self.metadata_path
        download_file(WLASL_METADATA_URL, self.metadata_path)
        return self.metadata_path

    def load_metadata(self) -> list[dict[str, Any]]:
        self.download_metadata(force=False)
        with self.metadata_path.open("r", encoding="utf-8") as file_handle:
            payload = json.load(file_handle)
        if not isinstance(payload, list):
            raise ValueError("WLASL metadata format unexpected; expected list.")
        return payload

    @staticmethod
    def _resolve_video_url(instance: dict[str, Any]) -> str | None:
        url = str(instance.get("url", "")).strip()
        if url:
            return url
        video_id = str(instance.get("video_id", "")).strip()
        if video_id:
            return f"https://www.youtube.com/watch?v={video_id}"
        return None

    def _select_glosses(
        self,
        items: list[dict[str, Any]],
        top_k: int | None = 2000,
    ) -> list[dict[str, Any]]:
        ordered = sorted(
            items,
            key=lambda item: len(item.get("instances", []) or []),
            reverse=True,
        )
        if top_k is None:
            return ordered
        return ordered[: max(1, top_k)]

    def build(
        self,
        top_k: int | None = 2000,
        max_samples_per_gloss: int | None = None,
        force_redownload: bool = False,
    ):
        """Run full WLASL ingestion and return resulting manifest dataframe."""
        metadata = self.load_metadata()
        selected = self._select_glosses(metadata, top_k=top_k)
        extractor = VideoLandmarkExtractor(max_num_hands=2)
        records: list[SampleRecord] = []

        try:
            for gloss_item in tqdm(selected, desc="WLASL glosses"):
                gloss = str(gloss_item.get("gloss", "UNKNOWN")).strip().upper()
                if not gloss:
                    gloss = "UNKNOWN"
                gloss_slug = slugify_gloss(gloss)
                instances = gloss_item.get("instances", []) or []
                if max_samples_per_gloss is not None:
                    instances = instances[: max_samples_per_gloss]

                for index, instance in enumerate(instances):
                    source_id = str(instance.get("video_id", f"{gloss_slug}_{index}"))
                    sample_id = f"wlasl_{gloss_slug}_{index:05d}"
                    video_path = self.video_root / gloss_slug / f"{source_id}.mp4"
                    sequence_path = self.sequence_root / gloss_slug / f"{sample_id}.npy"

                    if not video_path.exists() or force_redownload:
                        video_url = self._resolve_video_url(instance)
                        if video_url is None:
                            LOGGER.debug("Skipping %s missing video URL.", sample_id)
                            continue
                        downloaded = run_yt_dlp(video_url, video_path)
                        if downloaded is None:
                            continue
                        video_path = downloaded

                    sequence, max_hands = extractor.extract(video_path)
                    if sequence is None or sequence.size == 0:
                        continue

                    if max_hands <= 1:
                        sequence = collapse_to_one_hand(sequence)
                    write_npy(sequence, sequence_path)
                    records.append(
                        SampleRecord(
                            sample_id=sample_id,
                            source=self.source_name,
                            gloss=gloss,
                            sequence_path=str(sequence_path),
                            num_frames=int(sequence.shape[0]),
                            feature_dim=int(sequence.shape[1]),
                            metadata_json=json_dumps(
                                {
                                    "instance": instance,
                                    "source_video": str(video_path),
                                    "max_hands_detected": max_hands,
                                },
                            ),
                        ),
                    )
        finally:
            extractor.close()

        return save_manifest(records, self.manifest_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build WLASL landmark source manifest.")
    parser.add_argument("--data-root", type=Path, default=Path("ml/data"))
    parser.add_argument("--top-k", type=int, default=2000)
    parser.add_argument("--max-samples-per-gloss", type=int, default=None)
    parser.add_argument("--force-redownload", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    source = WLASLSource(args.data_root)
    source.build(
        top_k=args.top_k,
        max_samples_per_gloss=args.max_samples_per_gloss,
        force_redownload=args.force_redownload,
    )


if __name__ == "__main__":
    main()

