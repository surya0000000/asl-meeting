"""ASLLVD source loader for direct landmark CSV ingestion."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import re
from urllib.parse import urljoin

import requests
from tqdm import tqdm

from ml.data.sources.common import (
    SampleRecord,
    download_file,
    ensure_directory,
    infer_gloss_from_filename,
    json_dumps,
    parse_landmark_csv,
    save_manifest,
    slugify_gloss,
    write_npy,
)


LOGGER = logging.getLogger(__name__)

ASLLVD_INDEX_URL = "http://csr.bu.edu/asl/asllvd/asllvd-image.html"


class ASLLVDSource:
    """Load native ASLLVD landmark CSV files into sequence tensors."""

    source_name = "asllvd"

    def __init__(self, data_root: Path = Path("ml/data")) -> None:
        self.data_root = data_root
        self.raw_root = ensure_directory(data_root / "raw" / self.source_name)
        self.csv_root = ensure_directory(self.raw_root / "csv")
        self.processed_root = ensure_directory(data_root / "processed" / self.source_name)
        self.sequence_root = ensure_directory(self.processed_root / "sequences")
        self.manifest_path = self.processed_root / "manifest.csv"
        self.index_html_path = self.raw_root / "index.html"

    def download_index(self, force: bool = False) -> Path:
        """Fetch ASLLVD index page."""
        if self.index_html_path.exists() and not force:
            return self.index_html_path
        response = requests.get(ASLLVD_INDEX_URL, timeout=60)
        response.raise_for_status()
        self.index_html_path.write_text(response.text, encoding="utf-8")
        return self.index_html_path

    def discover_csv_urls(self) -> list[str]:
        """Extract CSV links from ASLLVD index HTML."""
        html_path = self.download_index(force=False)
        html = html_path.read_text(encoding="utf-8", errors="ignore")
        links = re.findall(r'href=["\']([^"\']+\.csv)["\']', html, flags=re.IGNORECASE)
        urls = sorted({urljoin(ASLLVD_INDEX_URL, link) for link in links})
        LOGGER.info("Discovered %d CSV links from ASLLVD index.", len(urls))
        return urls

    def download_csv_files(self, force: bool = False, limit: int | None = None) -> list[Path]:
        urls = self.discover_csv_urls()
        if limit is not None:
            urls = urls[: max(1, limit)]
        outputs: list[Path] = []
        for url in tqdm(urls, desc="ASLLVD CSV download"):
            filename = Path(url).name or f"asllvd_{len(outputs):05d}.csv"
            destination = self.csv_root / filename
            if destination.exists() and not force:
                outputs.append(destination)
                continue
            try:
                outputs.append(download_file(url, destination))
            except Exception:
                LOGGER.warning("Failed downloading ASLLVD CSV: %s", url)
        return outputs

    def build(
        self,
        limit: int | None = None,
        force_redownload: bool = False,
    ):
        """Run ASLLVD ingestion to .npy sequences and manifest."""
        csv_files = self.download_csv_files(force=force_redownload, limit=limit)
        records: list[SampleRecord] = []

        for index, csv_path in enumerate(tqdm(csv_files, desc="ASLLVD parse")):
            sequence = parse_landmark_csv(csv_path)
            if sequence is None or sequence.size == 0:
                continue
            gloss = infer_gloss_from_filename(csv_path)
            gloss_slug = slugify_gloss(gloss)
            sample_id = f"asllvd_{gloss_slug}_{index:05d}"
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
    parser = argparse.ArgumentParser(description="Build ASLLVD landmark source manifest.")
    parser.add_argument("--data-root", type=Path, default=Path("ml/data"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force-redownload", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    source = ASLLVDSource(data_root=args.data_root)
    source.build(limit=args.limit, force_redownload=args.force_redownload)


if __name__ == "__main__":
    main()

