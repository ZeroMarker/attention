"""Attention primitives: scaled dot-product attention and multi-head attention.

Implements the core equations from "Attention Is All You Need":
``Attention(Q, K, V) = softmax(Q K^T / sqrt(d_k)) V``.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .utils import apply_mask, softmax_attention


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
    dropout: nn.Module | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute attention over ``(..., seq_q, d_k)`` queries.

    Args:
        q: ``(..., seq_q, d_k)``.
        k: ``(..., seq_k, d_k)``.
        v: ``(..., seq_k, d_v)``.
        mask: optional boolean mask, ``True`` = attend, broadcastable over keys.
        dropout: optional module applied to the attention weights.

    Returns:
        ``(output, weights)`` where ``output`` is ``(..., seq_q, d_v)`` and
        ``weights`` is the post-softmax ``(..., seq_q, seq_k)`` distribution
        (before dropout).
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    scores = apply_mask(scores, mask)
    weights = softmax_attention(scores)
    attended = weights
    if dropout is not None:
        attended = dropout(attended)
    output = torch.matmul(attended, v)
    return output, weights


class MultiHeadAttention(nn.Module):
    """Multi-head attention with a single final projection.

    The input is linearly projected into ``n_head`` separate query/key/value
    spaces, attention runs per head, and the concatenated heads are projected
    back to ``d_model``.
    """

    def __init__(self, d_model: int, n_head: int, dropout: float = 0.1) -> None:
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def _project_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, _ = x.size()
        return (
            x.view(batch, seq, self.n_head, self.d_k)
            .transpose(1, 2)  # (batch, n_head, seq, d_k)
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor | None = None,
        value: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run multi-head attention.

        Args:
            query: ``(batch, seq_q, d_model)``.
            key/value: default to ``query`` for self-attention.
            mask: optional boolean mask, broadcastable over ``(seq_q, seq_k)``.

        Returns:
            ``(output, attention_weights)`` where ``output`` is
            ``(batch, seq_q, d_model)`` and ``attention_weights`` is
            ``(batch, n_head, seq_q, seq_k)``.
        """
        if key is None:
            key = query
        if value is None:
            value = query

        q = self._project_heads(self.w_q(query))
        k = self._project_heads(self.w_k(key))
        v = self._project_heads(self.w_v(value))

        attended, weights = scaled_dot_product_attention(q, k, v, mask, self.dropout)

        batch, _, seq, _ = attended.size()
        attended = (
            attended.transpose(1, 2)
            .contiguous()
            .view(batch, seq, self.d_model)
        )
        return self.w_o(attended), weights
