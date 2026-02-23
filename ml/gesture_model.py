"""Temporal PyTorch models for ASL gesture classification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

try:
    import torch
    import torch.nn as nn
except ImportError:  # pragma: no cover - handled at runtime where torch is required
    torch = None
    nn = None


@dataclass(slots=True)
class ModelConfig:
    """Hyperparameters for temporal gesture models."""

    input_size: int = 63
    hidden_size: int = 128
    num_layers: int = 2
    num_heads: int = 4
    dropout: float = 0.2
    num_classes: int = 9
    architecture: Literal["lstm", "transformer"] = "lstm"


def _require_torch() -> None:
    if torch is None or nn is None:
        raise RuntimeError(
            "PyTorch is required for model training/inference. "
            "Install dependencies from ml/requirements.txt."
        )


class GestureLSTM(nn.Module):  # type: ignore[misc]
    """LSTM classifier over landmark sequences."""

    def __init__(self, config: ModelConfig) -> None:
        _require_torch()
        super().__init__()
        self.config = config
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.num_classes),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        output, _ = self.lstm(x)
        final_hidden = output[:, -1, :]
        return self.head(final_hidden)


class GestureTransformer(nn.Module):  # type: ignore[misc]
    """Transformer encoder classifier for temporal landmark embeddings."""

    def __init__(self, config: ModelConfig) -> None:
        _require_torch()
        super().__init__()
        self.config = config
        self.embedding = nn.Linear(config.input_size, config.hidden_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_size,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_size * 4,
            dropout=config.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.num_layers,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_size, config.num_classes),
        )

    def forward(self, x: "torch.Tensor") -> "torch.Tensor":
        h = self.embedding(x)
        z = self.encoder(h)
        return self.head(z[:, -1, :])


def build_model(config: ModelConfig) -> nn.Module:  # type: ignore[valid-type]
    """Factory for temporal model architectures."""
    _require_torch()
    if config.architecture == "transformer":
        return GestureTransformer(config)
    return GestureLSTM(config)

