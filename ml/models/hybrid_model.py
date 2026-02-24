"""Hybrid BiLSTM + Transformer architecture for ASL landmark sequences."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(slots=True)
class SignHybridConfig:
    """Configuration for SignHybrid."""

    num_classes: int
    sequence_length: int = 30
    landmark_dim: int = 126
    d_model: int = 256
    lstm_hidden: int = 256
    lstm_layers: int = 2
    nheads: int = 8
    transformer_layers: int = 2
    dropout: float = 0.1
    feedforward_dim: int = 512


class SignHybrid(nn.Module):
    """
    Hybrid temporal classifier: BiLSTM for local dynamics + Transformer for global context.

    Input shape: (batch, seq_len, 126)
    Output shape: (batch, num_classes) logits
    """

    def __init__(self, config: SignHybridConfig) -> None:
        super().__init__()
        self.config = config
        self.bilstm = nn.LSTM(
            input_size=config.landmark_dim,
            hidden_size=config.lstm_hidden,
            num_layers=config.lstm_layers,
            dropout=config.dropout if config.lstm_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=True,
        )
        self.temporal_projection = nn.Linear(config.lstm_hidden * 2, config.d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.d_model))
        self.position_embedding = nn.Parameter(
            torch.zeros(1, config.sequence_length + 1, config.d_model),
        )
        self.input_dropout = nn.Dropout(config.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nheads,
            dim_feedforward=config.feedforward_dim,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=config.transformer_layers,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.num_classes),
        )
        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)

    def _get_position_embedding(self, token_count: int) -> torch.Tensor:
        if self.position_embedding.shape[1] == token_count:
            return self.position_embedding
        resized = F.interpolate(
            self.position_embedding.transpose(1, 2),
            size=token_count,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        return resized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning logits only."""
        batch_size = x.shape[0]
        local_features, _ = self.bilstm(x)
        projected = self.temporal_projection(local_features)

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, projected], dim=1)
        tokens = self.input_dropout(tokens + self._get_position_embedding(tokens.shape[1]))

        encoded = self.encoder(tokens)
        cls_rep = encoded[:, 0, :]
        return self.classifier(cls_rep)

