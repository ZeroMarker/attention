"""Attention-mask helpers.

All masks are boolean tensors where ``True`` means "attend" and ``False``
means "mask out" (the corresponding logit is set to ``-inf`` before softmax).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def make_padding_mask(tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    """Return a ``(batch, seq)`` boolean mask, ``True`` for non-pad tokens."""
    return tokens != pad_id


def make_causal_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """Return an ``(seq_len, seq_len)`` lower-triangular boolean mask."""
    mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))
    return mask


def build_self_attention_mask(
    tokens: torch.Tensor,
    pad_id: int,
    causal: bool,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build a ``(batch, 1, seq_q, seq_k)`` mask for a self-attention layer.

    When ``causal`` is True the mask is lower-triangular (a position may only
    attend to itself and earlier positions) and is ANDed with the padding mask.
    Defaults to the device of ``tokens`` so the padding and causal parts
    always live on the same device.
    """
    if device is None:
        device = tokens.device
    seq = tokens.size(-1)
    pad = make_padding_mask(tokens, pad_id)  # (batch, seq)
    pad = pad.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq)
    mask = pad.expand(-1, -1, seq, seq).to(device)
    if causal:
        causal_mask = make_causal_mask(seq, device).unsqueeze(0).unsqueeze(0)
        mask = mask & causal_mask
    return mask

def build_cross_attention_mask(
    src_tokens: torch.Tensor,
    pad_id: int,
    device: torch.device | None = None,
) -> torch.Tensor:
    """Build a ``(batch, 1, 1, src_len)`` mask for encoder-decoder attention."""
    if device is None:
        device = src_tokens.device
    pad = make_padding_mask(src_tokens, pad_id)  # (batch, src_len)
    return pad.unsqueeze(1).unsqueeze(2).to(device)  # (batch, 1, 1, src_len)


def apply_mask(scores: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Set masked-out logits to ``-inf`` while leaving valid ones untouched."""
    if mask is None:
        return scores
    # mask broadcasts over the head dimension: (..., seq_q, seq_k)
    return scores.masked_fill(~mask, float("-inf"))


def softmax_attention(scores: torch.Tensor) -> torch.Tensor:
    """Numerically stable softmax over the last (key) dimension."""
    return F.softmax(scores, dim=-1)
