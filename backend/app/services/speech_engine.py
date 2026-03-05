"""Text-to-speech utilities with Edge-TTS primary and pyttsx3 fallback."""

from __future__ import annotations

import asyncio
import base64
import logging
from pathlib import Path
import uuid
import wave
import xml.sax.saxutils as saxutils

try:
    import pyttsx3
except ImportError:  # pragma: no cover - optional runtime dependency
    pyttsx3 = None

try:
    import edge_tts
except ImportError:  # pragma: no cover - optional runtime dependency
    edge_tts = None


LOGGER = logging.getLogger(__name__)


class SpeechEngine:
    """Generate speech using Edge-TTS, with pyttsx3 fallback."""

    def __init__(
        self,
        output_dir: Path,
        enabled: bool = True,
        edge_voice: str = "en-US-AriaNeural",
        rate_percent: int = 90,
    ) -> None:
        self.output_dir = output_dir
        self.enabled = enabled
        self.edge_voice = edge_voice
        self.rate_percent = rate_percent
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _wrap_ssml(text: str, rate_percent: int = 90) -> str:
        escaped = saxutils.escape(text)
        return (
            '<speak version="1.0" xml:lang="en-US">'
            f'<prosody rate="{rate_percent}%"><emphasis level="moderate">{escaped}</emphasis></prosody>'
            "</speak>"
        )

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

    @staticmethod
    def _read_file_bytes(path: Path) -> bytes:
        return path.read_bytes()

    async def _synthesize_edge_audio_bytes(self, text: str) -> bytes | None:
        if edge_tts is None:
            return None
        ssml = self._wrap_ssml(text, rate_percent=self.rate_percent)
        # Edge-TTS streams compressed chunks (commonly mp3).
        # We still transport raw bytes to clients over base64.
        communicate = edge_tts.Communicate(ssml, self.edge_voice, rate="-10%")
        chunks: list[bytes] = []
        try:
            async for chunk in communicate.stream():
                if chunk.get("type") == "audio":
                    data = chunk.get("data")
                    if isinstance(data, bytes):
                        chunks.append(data)
            if chunks:
                return b"".join(chunks)
        except Exception:
            LOGGER.exception("Edge-TTS synthesis failed.")
        return None

    def _synthesize_pyttsx3_to_path(self, text: str) -> Path:
        output_file = self.output_dir / f"tts-{uuid.uuid4().hex}.wav"
        if pyttsx3 is None:
            return self._write_silence_fallback(text)
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", 170)
            engine.save_to_file(text, str(output_file))
            engine.runAndWait()
            engine.stop()
            if output_file.exists():
                return output_file
        except Exception:
            LOGGER.exception("pyttsx3 synthesis failed.")
        return self._write_silence_fallback(text)

    def synthesize(self, text: str) -> Path | None:
        """Sync compatibility API used by REST path."""
        if not self.enabled:
            return None
        if not text.strip():
            return None
        return self._synthesize_pyttsx3_to_path(text)

    async def synthesize_audio_bytes(self, text: str) -> bytes | None:
        """Async TTS returning encoded audio bytes for WebSocket streaming."""
        if not self.enabled or not text.strip():
            return None
        edge_bytes = await self._synthesize_edge_audio_bytes(text)
        if edge_bytes:
            return edge_bytes
        fallback_path = await asyncio.to_thread(self._synthesize_pyttsx3_to_path, text)
        return await asyncio.to_thread(self._read_file_bytes, fallback_path)

    async def synthesize_audio_message(self, text: str, gloss: str) -> dict[str, str] | None:
        """Create WebSocket-ready audio payload with base64 bytes."""
        audio_bytes = await self.synthesize_audio_bytes(text)
        if not audio_bytes:
            return None
        encoded = base64.b64encode(audio_bytes).decode("ascii")
        return {
            "type": "audio",
            "data": encoded,
            "gloss": gloss,
        }

    def cleanup_old_files(self, limit: int = 100) -> None:
        """Keep only the newest synthesized files."""
        files = sorted(self.output_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime, reverse=True)
        for stale in files[limit:]:
            try:
                stale.unlink(missing_ok=True)
            except OSError:
                LOGGER.warning("Unable to delete stale audio file: %s", stale)

