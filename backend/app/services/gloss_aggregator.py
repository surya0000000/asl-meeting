"""Temporal aggregation of raw predictions into stable gloss events."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(slots=True)
class GlossAggregator:
    """Debounce + deduplicate frame predictions and detect sentence gaps."""

    debounce_frames: int = 5
    dedup_gap_frames: int = 10
    sentence_gap_seconds: float = 1.5

    _candidate: str | None = None
    _candidate_count: int = 0
    _last_emitted_gloss: str | None = None
    _frames_since_last_match: int = 10
    _last_signing_time: datetime | None = None

    def process(self, prediction: str | None, timestamp: datetime | None = None) -> list[dict[str, str]]:
        """
        Process one prediction and return semantic events.

        Events:
            {"type": "gloss", "gloss": "..."}
            {"type": "sentence_end"}
        """
        now = timestamp or datetime.now(timezone.utc)
        events: list[dict[str, str]] = []

        is_valid_gloss = bool(prediction and prediction not in {"UNSURE", "BUFFERING"})

        if self._last_signing_time is not None and (now - self._last_signing_time).total_seconds() > self.sentence_gap_seconds:
            events.append({"type": "sentence_end"})

        if not is_valid_gloss:
            self._candidate = None
            self._candidate_count = 0
            self._frames_since_last_match += 1
            return events

        assert prediction is not None
        self._last_signing_time = now

        if self._candidate == prediction:
            self._candidate_count += 1
        else:
            self._candidate = prediction
            self._candidate_count = 1

        if prediction == self._last_emitted_gloss:
            self._frames_since_last_match = 0
        else:
            self._frames_since_last_match += 1

        can_emit = self._candidate_count >= self.debounce_frames
        is_repeat = prediction == self._last_emitted_gloss
        has_gap = self._frames_since_last_match >= self.dedup_gap_frames

        if can_emit and (not is_repeat or has_gap):
            events.append({"type": "gloss", "gloss": prediction})
            self._last_emitted_gloss = prediction
            self._candidate_count = 0
            self._frames_since_last_match = 0

        return events

