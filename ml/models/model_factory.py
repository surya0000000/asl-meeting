"""Model factory for ASL landmark sequence architectures."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from ml.config import (
    D_MODEL,
    DROPOUT,
    LANDMARK_DIM,
    N_HEADS,
    N_LAYERS,
)
from ml.models.hybrid_model import SignHybrid, SignHybridConfig
from ml.models.landmark_transformer import SignTransformer, SignTransformerConfig


Architecture = Literal["lstm", "transformer", "hybrid"]


@dataclass(slots=True)
class LSTMConfig:
    """Configuration for legacy LSTM baseline."""

    num_classes: int
    landmark_dim: int = LANDMARK_DIM
    hidden_size: int = D_MODEL
    num_layers: int = 2
    dropout: float = DROPOUT


class SignLSTM(nn.Module):
    """LSTM baseline classifier to preserve architecture compatibility."""

    def __init__(self, config: LSTMConfig) -> None:
        super().__init__()
        self.config = config
        self.lstm = nn.LSTM(
            input_size=config.landmark_dim,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=False,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.hidden_size),
            nn.Linear(config.hidden_size, config.num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.classifier(output[:, -1, :])


def get_model(
    architecture: str,
    num_classes: int,
    sequence_length: int,
) -> nn.Module:
    """
    Build a model for landmark sequence classification.

    Supported architectures:
    - lstm
    - transformer
    - hybrid
    """
    arch = architecture.lower().strip()
    if arch == "lstm":
        return SignLSTM(
            LSTMConfig(
                num_classes=num_classes,
                landmark_dim=LANDMARK_DIM,
                hidden_size=D_MODEL,
                num_layers=2,
                dropout=DROPOUT,
            ),
        )
    if arch == "transformer":
        return SignTransformer(
            SignTransformerConfig(
                num_classes=num_classes,
                sequence_length=sequence_length,
                landmark_dim=LANDMARK_DIM,
                d_model=D_MODEL,
                nheads=N_HEADS,
                num_layers=N_LAYERS,
                dropout=DROPOUT,
                feedforward_dim=512,
            ),
        )
    if arch == "hybrid":
        return SignHybrid(
            SignHybridConfig(
                num_classes=num_classes,
                sequence_length=sequence_length,
                landmark_dim=LANDMARK_DIM,
                d_model=D_MODEL,
                lstm_hidden=D_MODEL,
                lstm_layers=2,
                nheads=N_HEADS,
                transformer_layers=2,
                dropout=DROPOUT,
                feedforward_dim=512,
            ),
        )
    raise ValueError(f"Unsupported architecture: {architecture}")

