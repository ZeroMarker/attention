"""Token embeddings and sinusoidal positional encoding."""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class TokenEmbedding(nn.Module):
    """Learned token embedding, scaled by ``sqrt(d_model)`` per the paper.

    The scaling keeps the magnitude of the embeddings comparable to the
    positional encoding so that neither signal dominates the sum.
    """

    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.embedding(tokens) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding (not learned).

    ``PE(pos, 2i)     = sin(pos / 10000^(2i/d_model))``
    ``PE(pos, 2i + 1) = cos(pos / 10000^(2i/d_model))``

    Stored as a non-persistent buffer so the model can ship without it being
    considered a learnable parameter.
    """

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        positions = torch.arange(max_len).unsqueeze(1)  # (max_len, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10_000.0) / d_model)
        )
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(positions * div_term)
        pe[:, 1::2] = torch.cos(positions * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to ``x`` of shape ``(batch, seq, d_model)``."""
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)
