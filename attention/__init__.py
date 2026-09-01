"""A from-scratch Transformer, following "Attention Is All You Need"."""
from __future__ import annotations

from .attention import MultiHeadAttention, scaled_dot_product_attention
from .config import TransformerConfig
from .decoder import Decoder, DecoderLayer
from .embeddings import PositionalEncoding, TokenEmbedding
from .encoder import Encoder, EncoderLayer
from .feedforward import PositionwiseFeedForward
from .transformer import Transformer

__all__ = [
    "Decoder",
    "DecoderLayer",
    "Encoder",
    "EncoderLayer",
    "MultiHeadAttention",
    "PositionalEncoding",
    "PositionwiseFeedForward",
    "TokenEmbedding",
    "Transformer",
    "TransformerConfig",
    "scaled_dot_product_attention",
]
