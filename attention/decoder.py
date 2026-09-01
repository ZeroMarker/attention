"""Decoder stack: masked self-attention, encoder-decoder cross-attention, and
feed-forward, each wrapped in ``LayerNorm(x + Sublayer(x))`` (post-norm)."""
from __future__ import annotations

import copy

import torch
import torch.nn as nn

from .attention import MultiHeadAttention
from .feedforward import PositionwiseFeedForward


class DecoderLayer(nn.Module):
    """One decoder layer."""

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
        self.cross_attn = MultiHeadAttention(d_model, n_head, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        mem_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        attended = self.self_attn(x, x, x, tgt_mask)[0]
        x = self.norm1(x + self.dropout(attended))

        cross = self.cross_attn(x, memory, memory, mem_mask)[0]
        x = self.norm2(x + self.dropout(cross))

        x = self.norm3(x + self.dropout(self.ffn(x)))
        return x


class Decoder(nn.Module):
    """Stack of ``n_layers`` decoder layers."""

    def __init__(self, layer: DecoderLayer, n_layers: int) -> None:
        super().__init__()
        self.layers = nn.ModuleList([copy.deepcopy(layer) for _ in range(n_layers)])

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        mem_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, memory, tgt_mask, mem_mask)
        return x
