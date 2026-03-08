"""Transformer architecture for ASL landmark sequence classification."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F


@dataclass(slots=True)
class SignTransformerConfig:
    """Configuration for SignTransformer."""

    num_classes: int
    sequence_length: int = 30
    landmark_dim: int = 126
    d_model: int = 256
    nheads: int = 8
    num_layers: int = 4
    dropout: float = 0.1
    feedforward_dim: int = 512


class SignTransformer(nn.Module):
    """
    Transformer encoder with learnable positional encoding and CLS token.

    Input shape: (batch, seq_len, 126)
    Output shape: (batch, num_classes) logits
    """

    def __init__(self, config: SignTransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.input_projection = nn.Linear(config.landmark_dim, config.d_model)
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
        self.encoder = nn.TransformerEncoder(encoder_layer=encoder_layer, num_layers=config.num_layers)
        self.classifier = nn.Sequential(
            nn.LayerNorm(config.d_model),
            nn.Linear(config.d_model, config.num_classes),
        )

        self._init_parameters()

    def _init_parameters(self) -> None:
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.position_embedding, std=0.02)

    def _get_position_embedding(self, sequence_tokens: int) -> torch.Tensor:
        """
        Return positional embeddings sized to current token length.

        Supports sequence lengths different from training config by interpolation.
        """
        if self.position_embedding.shape[1] == sequence_tokens:
            return self.position_embedding
        resized = F.interpolate(
            self.position_embedding.transpose(1, 2),
            size=sequence_tokens,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)
        return resized

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass returning classification logits."""
        batch_size = x.shape[0]
        token_embeddings = self.input_projection(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, token_embeddings], dim=1)

        pos = self._get_position_embedding(tokens.shape[1])
        tokens = self.input_dropout(tokens + pos)

        encoded = self.encoder(tokens)
        cls_rep = encoded[:, 0, :]
        return self.classifier(cls_rep)

