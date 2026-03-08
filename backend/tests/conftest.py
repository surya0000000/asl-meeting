"""Pytest fixtures for backend API tests."""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from backend.app.core.config import get_settings
from backend.app.main import create_app


@pytest.fixture(autouse=True)
def configure_test_env() -> None:
    os.environ["SEQUENCE_LENGTH"] = "2"
    os.environ["PREDICT_EVERY_N_FRAMES"] = "1"
    os.environ["ENABLE_TEXT_REFINER"] = "false"
    os.environ["ENABLE_TTS"] = "false"
    os.environ["GESTURE_VOCAB"] = "HELLO,YES,NO,WAIT,QUESTION,AGREE,DISAGREE,SLOW DOWN,THANK YOU"
    get_settings.cache_clear()


@pytest.fixture
def app():
    get_settings.cache_clear()
    return create_app()


@pytest.fixture
def client(app):
    with TestClient(app) as test_client:
        yield test_client

