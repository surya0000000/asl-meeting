"""Session management for concurrent WebSocket ASL streams."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import asyncio
import logging
from typing import Any
from uuid import UUID, uuid4

import numpy as np

from backend.app.services.gloss_aggregator import GlossAggregator


LOGGER = logging.getLogger(__name__)


@dataclass
class SessionState:
    """Per-user streaming state."""

    session_id: UUID
    sequence_buffer: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=30))
    prediction_history: list[str] = field(default_factory=list)  # last 50
    gloss_accumulator: list[str] = field(default_factory=list)
    last_activity: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tts_queue: asyncio.Queue[Any] = field(default_factory=asyncio.Queue)
    audio_device: str | None = None
    gloss_aggregator: GlossAggregator = field(default_factory=GlossAggregator)

    def add_prediction(self, gloss: str) -> None:
        self.prediction_history.append(gloss)
        if len(self.prediction_history) > 50:
            self.prediction_history = self.prediction_history[-50:]


class SessionManager:
    """Singleton concurrent session manager with idle cleanup."""

    _instance: "SessionManager | None" = None

    def __new__(cls) -> "SessionManager":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False  # type: ignore[attr-defined]
        return cls._instance

    def __init__(self) -> None:
        if getattr(self, "_initialized", False):
            return
        self._initialized = True
        self._sessions: dict[UUID, SessionState] = {}
        self._lock = asyncio.Lock()
        self.cleanup_task: asyncio.Task[None] | None = None
        self.idle_timeout = timedelta(minutes=10)
        self._cleanup_interval_seconds = 60

    @staticmethod
    def _parse_session_id(session_id: str | UUID | None) -> UUID:
        if isinstance(session_id, UUID):
            return session_id
        if isinstance(session_id, str) and session_id.strip():
            try:
                return UUID(session_id.strip())
            except ValueError:
                LOGGER.warning("Invalid session_id provided, creating new one: %s", session_id)
        return uuid4()

    async def get_or_create(self, session_id: str | UUID | None) -> SessionState:
        """Return existing session state or create a new one."""
        parsed = self._parse_session_id(session_id)
        async with self._lock:
            state = self._sessions.get(parsed)
            if state is None:
                state = SessionState(session_id=parsed)
                self._sessions[parsed] = state
            state.last_activity = datetime.now(timezone.utc)
            return state

    async def update_activity(self, session_id: str | UUID) -> None:
        parsed = self._parse_session_id(session_id)
        async with self._lock:
            state = self._sessions.get(parsed)
            if state is not None:
                state.last_activity = datetime.now(timezone.utc)

    async def close(self, session_id: str | UUID) -> None:
        """Close and remove session state."""
        parsed = self._parse_session_id(session_id)
        async with self._lock:
            state = self._sessions.pop(parsed, None)
        if state is None:
            return
        try:
            state.tts_queue.put_nowait(None)
        except Exception:
            pass

    async def _cleanup_loop(self) -> None:
        try:
            while True:
                await asyncio.sleep(self._cleanup_interval_seconds)
                now = datetime.now(timezone.utc)
                stale: list[UUID] = []
                async with self._lock:
                    for session_id, state in self._sessions.items():
                        if now - state.last_activity > self.idle_timeout:
                            stale.append(session_id)
                    for session_id in stale:
                        self._sessions.pop(session_id, None)
                if stale:
                    LOGGER.info("Cleaned up %d idle sessions.", len(stale))
        except asyncio.CancelledError:
            LOGGER.info("Session cleanup loop cancelled.")
            raise

    def start_cleanup_task(self) -> None:
        if self.cleanup_task is None or self.cleanup_task.done():
            self.cleanup_task = asyncio.create_task(self._cleanup_loop(), name="session-cleanup")

    async def stop_cleanup_task(self) -> None:
        if self.cleanup_task is not None:
            self.cleanup_task.cancel()
            try:
                await self.cleanup_task
            except asyncio.CancelledError:
                pass
            self.cleanup_task = None

