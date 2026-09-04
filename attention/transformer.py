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

    @torch.no_grad()
    def generate(
        self,
        src: torch.Tensor,
        bos_id: int,
        eos_id: int | None = None,
        max_new_tokens: int = 64,
    ) -> torch.Tensor:
        """Greedily decode target tokens for a batch of source sequences.

        The returned tensor includes the initial beginning-of-sequence token.
        Once an item emits ``eos_id``, later positions for that item are padded
        with :attr:`TransformerConfig.pad_id`. Decoding stops early when every
        item in the batch has emitted EOS.

        Args:
            src: Source token IDs with shape ``(batch, src_len)``.
            bos_id: Token ID used to start every decoded sequence.
            eos_id: Optional token ID that stops decoding.
            max_new_tokens: Maximum number of tokens to append after BOS.
        """
        if src.ndim != 2:
            raise ValueError("src must have shape (batch, src_len)")
        if src.size(0) == 0:
            raise ValueError("src batch must not be empty")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        if max_new_tokens + 1 > self.config.max_seq_len:
            raise ValueError(
                "requested target length exceeds config.max_seq_len "
                f"({max_new_tokens + 1} > {self.config.max_seq_len})"
            )
        for name, token_id in (("bos_id", bos_id), ("eos_id", eos_id)):
            if token_id is not None and not 0 <= token_id < self.config.vocab_size:
                raise ValueError(
                    f"{name} must be between 0 and {self.config.vocab_size - 1}"
                )

        was_training = self.training
        self.eval()
        try:
            src_mask = build_self_attention_mask(
                src, self.config.pad_id, causal=False
            )
            mem_mask = build_cross_attention_mask(src, self.config.pad_id)
            memory = self.encode(src, src_mask)
            generated = torch.full(
                (src.size(0), 1), bos_id, dtype=src.dtype, device=src.device
            )
            finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)

            for _ in range(max_new_tokens):
                decoded = self.decode(generated, memory, mem_mask=mem_mask)
                next_token = self.output_proj(decoded[:, -1]).argmax(dim=-1)

                if eos_id is not None:
                    next_token = torch.where(
                        finished,
                        torch.full_like(next_token, self.config.pad_id),
                        next_token,
                    )
                    finished |= next_token.eq(eos_id)

                generated = torch.cat((generated, next_token.unsqueeze(1)), dim=1)
                if eos_id is not None and finished.all():
                    break

            return generated
        finally:
            self.train(was_training)
