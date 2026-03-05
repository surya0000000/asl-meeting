"""Central application settings."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration loaded from environment variables."""

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    app_name: str = "ASL Meeting Copilot API"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"

    model_checkpoint_path: str = "ml/models/gesture_model.pt"
    onnx_model_path_value: str = "ml/models/gesture_model.onnx"
    label_map_path_value: str = "ml/models/label_map.json"
    gesture_vocab: str = "HELLO,YES,NO,WAIT,QUESTION,AGREE,DISAGREE,SLOW DOWN,THANK YOU"
    sequence_length: int = 30
    predict_every_n_frames: int = 5
    confidence_threshold: float = 0.5

    enable_text_refiner: bool = True
    enable_tts: bool = False
    openai_api_key: str = Field(default="", repr=False)
    openai_model: str = "gpt-4o-mini"

    audio_output_dir: str = "backend/audio_output"
    static_audio_mount: str = "/audio"
    allow_cors_origins: str = "*"

    @property
    def vocabulary(self) -> list[str]:
        return [item.strip() for item in self.gesture_vocab.split(",") if item.strip()]

    @property
    def audio_output_path(self) -> Path:
        return Path(self.audio_output_dir)

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.model_checkpoint_path)

    @property
    def onnx_model_path(self) -> Path:
        return Path(self.onnx_model_path_value)

    @property
    def label_map_path(self) -> Path:
        return Path(self.label_map_path_value)

    @property
    def cors_origins(self) -> list[str]:
        if self.allow_cors_origins.strip() == "*":
            return ["*"]
        return [item.strip() for item in self.allow_cors_origins.split(",") if item.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()

