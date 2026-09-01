"""Tests for attention-mask helpers."""
import torch

from attention.utils import (
    build_cross_attention_mask,
    build_self_attention_mask,
    make_causal_mask,
    make_padding_mask,
)


def test_make_padding_mask():
    tokens = torch.tensor([[1, 2, 0, 0], [3, 0, 0, 0]])
    mask = make_padding_mask(tokens, pad_id=0)
    expected = torch.tensor([[True, True, False, False], [True, False, False, False]])
    assert torch.equal(mask, expected)


def test_make_causal_mask_lower_triangular():
    m = make_causal_mask(4)
    assert torch.equal(m, torch.tril(torch.ones(4, 4, dtype=torch.bool)))


def test_self_attention_mask_non_causal_masks_padding_column():
    tokens = torch.tensor([[1, 2, 0]])
    m = build_self_attention_mask(tokens, 0, causal=False)
    assert m.shape == (1, 1, 3, 3)
    # key position 2 is padding -> masked out for every query
    assert not m[0, 0, :, 2].any()
    assert m[0, 0, 0, 0] and m[0, 0, 0, 1]


def test_self_attention_mask_causal_blocks_future():
    tokens = torch.tensor([[1, 2, 3]])
    m = build_self_attention_mask(tokens, 0, causal=True)
    assert m.shape == (1, 1, 3, 3)
    assert not m[0, 0, 0, 1]  # query 0 cannot attend to key 1
    assert not m[0, 0, 0, 2]
    assert not m[0, 0, 1, 2]
    assert m[0, 0, 1, 1]  # query 1 can attend to key 1
    assert m[0, 0, 2, 2]


def test_cross_attention_mask_masks_padding():
    src = torch.tensor([[1, 2, 0, 0]])
    m = build_cross_attention_mask(src, 0)
    assert m.shape == (1, 1, 1, 4)
    assert m[0, 0, 0, 0] and m[0, 0, 0, 1]
    assert not m[0, 0, 0, 2] and not m[0, 0, 0, 3]
