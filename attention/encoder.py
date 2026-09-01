"""Encoder stack: a chain of identical encoder layers with residual
connections and post layer norm, as described in the paper."""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feedforward import PositionwiseFeedForward


class EncoderLayer(nn.Module):
    """One encoder layer: self-attention + feed-forward, each wrapped in
    ``LayerNorm(x + Sublayer(x))`` (post-norm)."""

    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attended = self.self_attn(x, x, x, mask)[0]
        x = self.norm1(x + self.dropout(attended))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class Encoder(nn.Module):
    """Stack of ``n_layers`` encoder layers applied to the source sequence."""

    def __init__(self, layer: EncoderLayer, n_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, mask)
        return x
