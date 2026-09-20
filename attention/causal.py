"""Decoder-only causal language model (the "tiny GPT" path).

A decoder-only block is the paper's encoder block — self-attention followed by
a position-wise feed-forward, each wrapped in ``LayerNorm(x + Sublayer(x))`` —
stacked under a causal mask, with the same token embedding and output
projection. Causality lives entirely in the attention mask, so this model
reuses :class:`~attention.encoder.EncoderLayer` and
:class:`~attention.encoder.Encoder` rather than duplicating the block: while
both paths are post-norm, the encoder-decoder model and the LLM path cannot
drift apart.

``forward`` returns next-token logits for every position, which is the training
signal for the objective in ``ROADMAP.md`` §2.1 — compare ``logits[:, :-1]``
with ``tokens[:, 1:]``.

Generation recomputes the whole prefix at every step; a KV cache is §1.3.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .config import TransformerConfig
from .embeddings import PositionalEncoding, TokenEmbedding
from .encoder import Encoder, EncoderLayer
from .utils import build_self_attention_mask


class CausalLM(nn.Module):
    """Decoder-only Transformer language model.

    Args:
        config: a :class:`TransformerConfig`; ``d_model`` must be divisible by
            ``n_head``. Dropping cross-attention needs no new configuration, so
            the decoder-only path shares the encoder-decoder config type.

    Example::

        from attention import CausalLM, TransformerConfig

        config = TransformerConfig(vocab_size=10_000, d_model=256, n_head=8,
                                   n_layers=6, d_ff=1024, max_seq_len=256)
        model = CausalLM(config)

        tokens = torch.randint(0, 10_000, (2, 32))
        logits = model(tokens)  # (2, 32, 10_000)

        loss = F.cross_entropy(
            logits[:, :-1].reshape(-1, config.vocab_size),
            tokens[:, 1:].reshape(-1),
            ignore_index=config.pad_id,
        )
    """

    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config

        self.token_embedding = TokenEmbedding(config.vocab_size, config.d_model)
        self.positional_encoding = PositionalEncoding(
            config.d_model, config.max_seq_len, config.dropout
        )
        block = EncoderLayer(
            config.d_model, config.n_head, config.d_ff, config.dropout, config.activation
        )
        self.blocks = Encoder(block, config.n_layers)
        self.output_proj = nn.Linear(config.d_model, config.vocab_size, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Return next-token logits of shape ``(batch, seq, vocab_size)``.

        Every position attends only to itself and earlier positions, so the
        logits at position ``i`` predict ``tokens[:, i + 1]`` without seeing it.
        """
        if tokens.ndim != 2:
            raise ValueError("tokens must have shape (batch, seq)")
        mask = build_self_attention_mask(tokens, self.config.pad_id, causal=True)
        hidden = self.positional_encoding(self.token_embedding(tokens))
        return self.output_proj(self.blocks(hidden, mask))

    @torch.no_grad()
    def generate(
        self,
        prompt: torch.Tensor,
        eos_id: int | None = None,
        max_new_tokens: int = 64,
    ) -> torch.Tensor:
        """Greedily continue ``prompt`` token by token.

        The returned tensor has shape ``(batch, prompt_len + generated)`` and
        begins with ``prompt`` unchanged. Once an item emits ``eos_id``, its
        later positions hold :attr:`TransformerConfig.pad_id`, and decoding
        stops early once every item in the batch has emitted EOS.

        Args:
            prompt: Token IDs with shape ``(batch, prompt_len)``, ``prompt_len >= 1``.
            eos_id: Optional token ID that stops an item's decoding.
            max_new_tokens: Maximum number of tokens to append to the prompt.
        """
        if prompt.ndim != 2:
            raise ValueError("prompt must have shape (batch, prompt_len)")
        if prompt.size(0) == 0:
            raise ValueError("prompt batch must not be empty")
        if prompt.size(1) == 0:
            raise ValueError("prompt must contain at least one token")
        if max_new_tokens < 1:
            raise ValueError("max_new_tokens must be at least 1")
        if prompt.size(1) + max_new_tokens > self.config.max_seq_len:
            raise ValueError(
                "requested sequence length exceeds config.max_seq_len "
                f"({prompt.size(1) + max_new_tokens} > {self.config.max_seq_len})"
            )
        if eos_id is not None and not 0 <= eos_id < self.config.vocab_size:
            raise ValueError(
                f"eos_id must be between 0 and {self.config.vocab_size - 1}"
            )

        was_training = self.training
        self.eval()
        try:
            generated = prompt
            finished = torch.zeros(prompt.size(0), dtype=torch.bool, device=prompt.device)

            for _ in range(max_new_tokens):
                next_token = self.forward(generated)[:, -1].argmax(dim=-1)

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
