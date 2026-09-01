"""Full encoder-decoder Transformer model.

Wires together token embeddings, sinusoidal positional encoding, the encoder
and decoder stacks, and an output projection to the vocabulary. Handles mask
construction for source padding, target causal masking, and cross-attention.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import TransformerConfig
from .decoder import Decoder, DecoderLayer
from .embeddings import PositionalEncoding, TokenEmbedding
from .encoder import Encoder, EncoderLayer
from .utils import build_cross_attention_mask, build_self_attention_mask


class Transformer(nn.Module):
    """Transformer encoder-decoder model.

    Args:
        config: an :class:`TransformerConfig` describing the model.

    Example::

        from attention import Transformer, TransformerConfig

        model = Transformer(TransformerConfig(vocab_size=1000, d_model=64,
                                              n_head=4, n_layers=2, d_ff=256,
                                              max_seq_len=32))
        src = torch.randint(1, 1000, (2, 16))
        tgt = torch.randint(1, 1000, (2, 16))
        logits = model(src, tgt)  # (batch, tgt_len, vocab_size)
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = TokenEmbedding(config.vocab_size, config.d_model)
        self.positional_encoding = PositionalEncoding(
            config.d_model, config.max_seq_len, config.dropout
        )

        encoder_layer = EncoderLayer(
            config.d_model, config.n_head, config.d_ff, config.dropout, config.activation
        )
        decoder_layer = DecoderLayer(
            config.d_model, config.n_head, config.d_ff, config.dropout, config.activation
        )
        self.encoder = Encoder(encoder_layer, config.n_layers)
        self.decoder = Decoder(decoder_layer, config.n_layers)

        self.output_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def encode(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the encoder over source tokens, returning ``(batch, src_len, d_model)``."""
        if src_mask is None:
            src_mask = build_self_attention_mask(src, self.config.pad_id, causal=False)
        src_emb = self.token_embedding(src)
        src_emb = self.positional_encoding(src_emb)
        return self.encoder(src_emb, src_mask)

    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        mem_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run the decoder over target tokens given encoder ``memory``."""
        if tgt_mask is None:
            tgt_mask = build_self_attention_mask(tgt, self.config.pad_id, causal=True)
        tgt_emb = self.token_embedding(tgt)
        tgt_emb = self.positional_encoding(tgt_emb)
        return self.decoder(tgt_emb, memory, tgt_mask, mem_mask)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
    ) -> torch.Tensor:
        """Run the full model, returning ``(batch, tgt_len, vocab_size)`` logits."""
        src_mask = build_self_attention_mask(src, self.config.pad_id, causal=False)
        tgt_mask = build_self_attention_mask(tgt, self.config.pad_id, causal=True)
        mem_mask = build_cross_attention_mask(src, self.config.pad_id)

        memory = self.encode(src, src_mask)
        decoded = self.decode(tgt, memory, tgt_mask, mem_mask)
        return self.output_proj(decoded)
