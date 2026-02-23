"""API router exports."""

from backend.app.api.routes import router as rest_router
from backend.app.api.websocket import router as websocket_router

__all__ = ["rest_router", "websocket_router"]

