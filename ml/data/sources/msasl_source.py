"""MS-ASL source loader with manifest download and landmark extraction."""

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

# Community-hosted mirrors of canonical MS-ASL manifests.
MSASL_MANIFEST_URLS = {
    "train": "https://raw.githubusercontent.com/joeyism/real-time-sign-language-detection/master/MSASL/MSASL_train.json",
    "val": "https://raw.githubusercontent.com/joeyism/real-time-sign-language-detection/master/MSASL/MSASL_val.json",
    "test": "https://raw.githubusercontent.com/joeyism/real-time-sign-language-detection/master/MSASL/MSASL_test.json",
}


class MSASLSource:
    """Ingest MS-ASL (1000 classes / ~25k clips) into landmark sequences."""

    source_name = "msasl"

    def __init__(self, data_root: Path = Path("ml/data")) -> None:
        self.data_root = data_root
        self.raw_root = ensure_directory(data_root / "raw" / self.source_name)
        self.video_root = ensure_directory(self.raw_root / "videos")
        self.manifest_root = ensure_directory(self.raw_root / "manifests")
        self.processed_root = ensure_directory(data_root / "processed" / self.source_name)
        self.sequence_root = ensure_directory(self.processed_root / "sequences")
        self.output_manifest = self.processed_root / "manifest.csv"

    def download_manifests(self, force: bool = False) -> dict[str, Path]:
        """Download train/val/test JSON manifests."""
        outputs: dict[str, Path] = {}
        for split, url in MSASL_MANIFEST_URLS.items():
            destination = self.manifest_root / f"MSASL_{split}.json"
            if destination.exists() and not force:
                outputs[split] = destination
                continue
            outputs[split] = download_file(url, destination)
        return outputs

    def load_entries(self, force_manifest_download: bool = False) -> list[dict[str, Any]]:
        manifests = self.download_manifests(force=force_manifest_download)
        entries: list[dict[str, Any]] = []
        for split, path in manifests.items():
            with path.open("r", encoding="utf-8") as file_handle:
                payload = json.load(file_handle)
            if not isinstance(payload, list):
                LOGGER.warning("Skipping unexpected MS-ASL payload in %s", path)
                continue
            for item in payload:
                if isinstance(item, dict):
                    item["split"] = split
                    entries.append(item)
        return entries

    @staticmethod
    def _resolve_gloss(item: dict[str, Any]) -> str:
        for key in ("gloss", "text", "clean_text", "label_text", "sign"):
            value = item.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip().upper()
        label_value = item.get("label")
        if isinstance(label_value, str) and label_value.strip():
            return label_value.strip().upper()
        if isinstance(label_value, int):
            return f"LABEL_{label_value}"
        return "UNKNOWN"

    @staticmethod
    def _resolve_video_url(item: dict[str, Any]) -> str | None:
        value = item.get("url")
        if isinstance(value, str) and value.strip():
            return value.strip()
        video_id = item.get("video_id") or item.get("id")
        if isinstance(video_id, str) and video_id.strip():
            return f"https://www.youtube.com/watch?v={video_id.strip()}"
        return None

    def build(
        self,
        top_k: int | None = 1000,
        max_samples: int | None = None,
        force_manifest_download: bool = False,
        force_redownload_videos: bool = False,
    ):
        """Run full MS-ASL ingestion pipeline."""
        entries = self.load_entries(force_manifest_download=force_manifest_download)
        if not entries:
            LOGGER.warning("No MS-ASL entries available.")
            return save_manifest([], self.output_manifest)

        # Keep top-k glosses by sample count.
        grouped: dict[str, list[dict[str, Any]]] = {}
        for item in entries:
            grouped.setdefault(self._resolve_gloss(item), []).append(item)
        ordered_glosses = sorted(grouped.keys(), key=lambda gloss: len(grouped[gloss]), reverse=True)
        if top_k is not None:
            ordered_glosses = ordered_glosses[: max(1, top_k)]

        extractor = VideoLandmarkExtractor(max_num_hands=2)
        records: list[SampleRecord] = []
        processed_count = 0
        try:
            for gloss in tqdm(ordered_glosses, desc="MS-ASL glosses"):
                gloss_slug = slugify_gloss(gloss)
                for index, item in enumerate(grouped[gloss]):
                    if max_samples is not None and processed_count >= max_samples:
                        break
                    sample_id = f"msasl_{gloss_slug}_{index:05d}"
                    url = self._resolve_video_url(item)
                    if url is None:
                        continue
                    source_video_id = str(item.get("video_id") or item.get("id") or sample_id)
                    video_path = self.video_root / gloss_slug / f"{source_video_id}.mp4"
                    if not video_path.exists() or force_redownload_videos:
                        downloaded = run_yt_dlp(url, video_path)
                        if downloaded is None:
                            continue
                        video_path = downloaded

                    sequence, max_hands = extractor.extract(video_path)
                    if sequence is None or sequence.size == 0:
                        continue
                    if max_hands <= 1:
                        sequence = collapse_to_one_hand(sequence)

                    sequence_path = self.sequence_root / gloss_slug / f"{sample_id}.npy"
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
                                    "entry": item,
                                    "source_video": str(video_path),
                                    "max_hands_detected": max_hands,
                                },
                            ),
                        ),
                    )
                    processed_count += 1
                if max_samples is not None and processed_count >= max_samples:
                    break
        finally:
            extractor.close()

        return save_manifest(records, self.output_manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build MS-ASL landmark source manifest.")
    parser.add_argument("--data-root", type=Path, default=Path("ml/data"))
    parser.add_argument("--top-k", type=int, default=1000)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--force-manifest-download", action="store_true")
    parser.add_argument("--force-redownload-videos", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    source = MSASLSource(args.data_root)
    source.build(
        top_k=args.top_k,
        max_samples=args.max_samples,
        force_manifest_download=args.force_manifest_download,
        force_redownload_videos=args.force_redownload_videos,
    )


if __name__ == "__main__":
    main()

