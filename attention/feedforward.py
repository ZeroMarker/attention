"""Position-wise feed-forward network used inside encoder and decoder layers."""
from __future__ import annotations

import torch
import torch.nn as nn


class PositionwiseFeedForward(nn.Module):
    """A two-layer MLP applied independently at each position.

    ``FFN(x) = linear2(activation(linear1(x)))``
    """

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        dropout: float = 0.1,
        activation: str = "relu",
    ) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = {"relu": nn.ReLU(), "gelu": nn.GELU()}[activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
