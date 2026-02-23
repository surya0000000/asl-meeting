"""FastAPI application entrypoint for ASL Meeting Copilot backend."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from backend.app.api import rest_router, websocket_router
from backend.app.core.config import get_settings
from backend.app.core.logging import configure_logging
from backend.app.services.stream_processor import StreamProcessor


LOGGER = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    configure_logging(settings.log_level)
    settings.audio_output_path.mkdir(parents=True, exist_ok=True)
    app.state.settings = settings
    app.state.stream_processor = StreamProcessor(settings)
    LOGGER.info("ASL Meeting Copilot backend initialized.")
    yield
    LOGGER.info("ASL Meeting Copilot backend shutting down.")


def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(
        title=settings.app_name,
        version="0.1.0",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.mount(
        settings.static_audio_mount,
        StaticFiles(directory=settings.audio_output_path, check_dir=False),
        name="audio",
    )

    app.include_router(rest_router)
    app.include_router(websocket_router)
    return app


app = create_app()

