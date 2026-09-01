"""Model configuration for the Transformer.

Kept as a single dataclass so every hyperparameter lives in one place and
can be passed to :class:`attention.model.Transformer`.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransformerConfig:
    """Hyperparameters for a Transformer model.

    The default values follow the paper's ``base`` setting
    (d_model=512, 8 heads, 6 layers, d_ff=2048).
    """

    vocab_size: int = 32_000
    d_model: int = 512
    n_head: int = 8
    n_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    max_seq_len: int = 512
    activation: str = "relu"
    pad_id: int = 0

    def __post_init__(self) -> None:
        if self.d_model % self.n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        if self.activation not in {"relu", "gelu"}:
            raise ValueError(f"unsupported activation: {self.activation}")
