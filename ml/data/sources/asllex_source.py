"""ASL-LEX lexical source parser for phonological augmentation hints."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import re
from typing import Any

import pandas as pd

from ml.data.sources.common import SampleRecord, download_file, ensure_directory, save_manifest, slugify_gloss


LOGGER = logging.getLogger(__name__)

ASLLEX_CSV_URL = "https://asl-lex.org/data/ASL-LEX 2.0.csv"

GLOSS_COLUMN_CANDIDATES = [
    "Gloss",
    "IDGloss",
    "Sign",
    "SignEnglish",
    "Lexical Item",
    "EntryID",
]

PHONOLOGY_PATTERNS = [
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"handshape",
        r"location",
        r"movement",
        r"selected fingers",
        r"sign type",
        r"symmetry",
        r"alternating",
        r"major location",
    )
]


class ASLLexSource:
    """Parse ASL-LEX lexical metadata into augmentation feature vectors."""

    source_name = "asllex"

    def __init__(self, data_root: Path = Path("ml/data")) -> None:
        self.data_root = data_root
        self.raw_root = ensure_directory(data_root / "raw" / self.source_name)
        self.processed_root = ensure_directory(data_root / "processed" / self.source_name)
        self.csv_path = self.raw_root / "asl_lex.csv"
        self.hints_json_path = self.processed_root / "phonology_hints.json"
        self.feature_frame_path = self.processed_root / "phonology_features.csv"
        self.manifest_path = self.processed_root / "manifest.csv"

    def download_csv(self, force: bool = False, csv_url: str = ASLLEX_CSV_URL) -> Path:
        """Download ASL-LEX CSV."""
        if self.csv_path.exists() and not force:
            return self.csv_path
        try:
            download_file(csv_url, self.csv_path)
        except Exception as exc:
            raise RuntimeError(
                "Failed to download ASL-LEX CSV automatically. "
                "Provide --csv-path to a local ASL-LEX CSV export.",
            ) from exc
        return self.csv_path

    @staticmethod
    def _resolve_gloss_column(frame: pd.DataFrame) -> str:
        for candidate in GLOSS_COLUMN_CANDIDATES:
            if candidate in frame.columns and not pd.api.types.is_numeric_dtype(frame[candidate]):
                return candidate
        # fallback: first text-like column
        for column in frame.columns:
            if pd.api.types.is_string_dtype(frame[column]):
                return column
        raise ValueError("Unable to identify gloss column in ASL-LEX CSV.")

    @staticmethod
    def _select_phonology_columns(frame: pd.DataFrame) -> list[str]:
        selected = []
        for column in frame.columns:
            if any(pattern.search(column) for pattern in PHONOLOGY_PATTERNS):
                selected.append(column)
        if not selected:
            # fallback to all non-ID textual/numeric metadata fields
            selected = [column for column in frame.columns if column.lower() not in {"id", "entryid"}]
        return selected

    @staticmethod
    def _encode_feature_frame(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
        subset = frame[columns].copy()
        numeric_columns = [
            column
            for column in subset.columns
            if pd.api.types.is_numeric_dtype(subset[column]) or pd.api.types.is_bool_dtype(subset[column])
        ]
        for column in numeric_columns:
            subset[column] = pd.to_numeric(subset[column], errors="coerce").fillna(0.0)

        categorical_columns = [column for column in subset.columns if column not in numeric_columns]
        if categorical_columns:
            cleaned = subset[categorical_columns].fillna("UNK").astype(str)
            encoded = pd.get_dummies(cleaned, prefix=[slugify_gloss(name) for name in categorical_columns])
            subset = pd.concat([subset[numeric_columns], encoded], axis=1)
        else:
            subset = subset[numeric_columns]
        return subset.astype(float)

    def build(
        self,
        csv_path: Path | None = None,
        force_download: bool = False,
        csv_url: str = ASLLEX_CSV_URL,
    ) -> dict[str, Any]:
        """Build phonological feature vectors and augmentation hints."""
        if csv_path is None:
            csv_path = self.download_csv(force=force_download, csv_url=csv_url)

        frame = pd.read_csv(csv_path)
        if frame.empty:
            raise ValueError("ASL-LEX CSV is empty.")
        gloss_column = self._resolve_gloss_column(frame)
        frame = frame[frame[gloss_column].notna()].copy()
        frame["__gloss__"] = frame[gloss_column].astype(str).str.strip().str.upper()
        frame = frame[frame["__gloss__"] != ""]

        phonology_columns = self._select_phonology_columns(frame)
        encoded = self._encode_feature_frame(frame, phonology_columns)
        encoded["__gloss__"] = frame["__gloss__"].values

        # Collapse multiple lexical entries to one vector per gloss.
        grouped = encoded.groupby("__gloss__", as_index=True).mean(numeric_only=True)
        grouped.to_csv(self.feature_frame_path)

        hints: dict[str, Any] = {}
        records: list[SampleRecord] = []
        for gloss, row in grouped.iterrows():
            vector = row.to_numpy(dtype=float).tolist()
            hints[gloss] = {
                "feature_vector": vector,
                "feature_dim": len(vector),
                "phonology_columns": row.index.tolist(),
            }
            records.append(
                SampleRecord(
                    sample_id=f"asllex_{slugify_gloss(gloss)}",
                    source=self.source_name,
                    gloss=gloss,
                    sequence_path="",
                    num_frames=0,
                    feature_dim=0,
                    metadata_json=json.dumps(hints[gloss], ensure_ascii=True),
                ),
            )

        with self.hints_json_path.open("w", encoding="utf-8") as file_handle:
            json.dump(hints, file_handle, ensure_ascii=True, indent=2)

        save_manifest(records, self.manifest_path)
        LOGGER.info(
            "ASL-LEX hints exported: %s (%d gloss entries)",
            self.hints_json_path,
            len(hints),
        )
        return hints


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build ASL-LEX phonology hint dataset.")
    parser.add_argument("--data-root", type=Path, default=Path("ml/data"))
    parser.add_argument("--csv-path", type=Path, default=None)
    parser.add_argument("--csv-url", type=str, default=ASLLEX_CSV_URL)
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    source = ASLLexSource(data_root=args.data_root)
    source.build(csv_path=args.csv_path, force_download=args.force_download, csv_url=args.csv_url)


if __name__ == "__main__":
    main()

