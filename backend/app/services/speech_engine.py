"""Text-to-speech utilities for generated subtitle playback."""

from __future__ import annotations

import logging
from pathlib import Path
import tempfile
import uuid
import wave

try:
    import pyttsx3
except ImportError:  # pragma: no cover - optional runtime dependency
    pyttsx3 = None


LOGGER = logging.getLogger(__name__)


class SpeechEngine:
    """Generate speech audio files from refined text."""

    def __init__(self, output_dir: Path, enabled: bool = True) -> None:
        self.output_dir = output_dir
        self.enabled = enabled
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _write_silence_fallback(self, text: str) -> Path:
        """
        Create a short placeholder wav to keep pipeline functional.

        This is only used when pyttsx3 is unavailable in the runtime environment.
        """
        duration_sec = max(1, min(5, len(text) // 15))
        sample_rate = 22050
        num_frames = duration_sec * sample_rate
        path = self.output_dir / f"tts-fallback-{uuid.uuid4().hex}.wav"
        with wave.open(str(path), "w") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(b"\x00\x00" * num_frames)
        return path

    def synthesize(self, text: str) -> Path | None:
        """Convert text to speech and return output wav path."""
        if not self.enabled:
            return None
        if not text.strip():
            return None

        if pyttsx3 is None:
            LOGGER.warning("pyttsx3 unavailable. Producing silent fallback wav.")
            return self._write_silence_fallback(text)

        output_file = self.output_dir / f"tts-{uuid.uuid4().hex}.wav"
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 170)
            engine.save_to_file(text, str(output_file))
            engine.runAndWait()
            engine.stop()
            if output_file.exists():
                return output_file
        except Exception:
            LOGGER.exception("TTS synthesis failed for text: %s", text)

        # Best-effort fallback path if native synthesis fails.
        return self._write_silence_fallback(text)

    def cleanup_old_files(self, limit: int = 100) -> None:
        """Keep only the newest synthesized files."""
        files = sorted(self.output_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
        for stale in files[limit:]:
            try:
                stale.unlink(missing_ok=True)
            except OSError:
                LOGGER.warning("Unable to delete stale audio file: %s", stale)

