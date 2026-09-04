"""Tests for scaled dot-product and multi-head attention."""
import torch
import torch.nn.functional as F

from attention.attention import MultiHeadAttention, scaled_dot_product_attention


def test_sdpa_output_shapes():
    q = torch.randn(2, 4, 8, 16)  # batch, head, q_len, d_k
    k = torch.randn(2, 4, 6, 16)
    v = torch.randn(2, 4, 6, 20)
    out, weights = scaled_dot_product_attention(q, k, v)
    assert out.shape == (2, 4, 8, 20)
    assert weights.shape == (2, 4, 8, 6)


def test_sdpa_weights_sum_to_one():
    q = torch.randn(1, 1, 3, 8)
    k = torch.randn(1, 1, 5, 8)
    v = torch.randn(1, 1, 5, 8)
    _, weights = scaled_dot_product_attention(q, k, v)
    assert torch.allclose(weights.sum(-1), torch.ones(1, 1, 3), atol=1e-6)


def test_sdpa_masked_positions_get_zero_weight():
    q = torch.randn(1, 1, 1, 8)
    k = torch.randn(1, 1, 4, 8)
    v = torch.randn(1, 1, 4, 8)
    mask = torch.tensor([[[[True, False, True, False]]]])
    _, weights = scaled_dot_product_attention(q, k, v, mask)
    assert torch.allclose(weights[..., 1], torch.zeros(1), atol=1e-6)
    assert torch.allclose(weights[..., 3], torch.zeros(1), atol=1e-6)
    assert torch.allclose(weights.sum(-1), torch.ones(1, 1, 1), atol=1e-6)


def test_sdpa_fully_masked_row_is_finite_and_zero():
    q = torch.randn(1, 1, 1, 8)
    k = torch.randn(1, 1, 3, 8)
    v = torch.randn(1, 1, 3, 8)
    mask = torch.zeros(1, 1, 1, 3, dtype=torch.bool)
    output, weights = scaled_dot_product_attention(q, k, v, mask)
    assert torch.equal(weights, torch.zeros_like(weights))
    assert torch.equal(output, torch.zeros_like(output))


def test_sdpa_stateful_dropout():
    q = torch.randn(2, 2, 4, 8)
    k = torch.randn(2, 2, 4, 8)
    v = torch.randn(2, 2, 4, 8)
    dropout = torch.nn.Dropout(0.5)
    dropout.train()
    out, _ = scaled_dot_product_attention(q, k, v, None, dropout)
    assert out.shape == (2, 2, 4, 8)


def test_multihead_shapes_and_dtype():
    mha = MultiHeadAttention(32, 4)
    x = torch.randn(2, 10, 32)
    out, weights = mha(x)
    assert out.shape == (2, 10, 32)
    assert weights.shape == (2, 4, 10, 10)
    assert out.dtype == x.dtype


def test_multihead_self_attention_matches_parameters():
    # A single-head, single-layer multi-head attention is a trainable projection;
    # verify it produces the expected attention-weight shape when given a mask.
    mha = MultiHeadAttention(8, 1)
    x = torch.randn(1, 5, 8)
    mask = torch.tril(torch.ones(1, 5, 5, dtype=torch.bool))
    _, weights = mha(x, mask=mask)
    assert weights.shape == (1, 1, 5, 5)
    # causal: position 0 can only attend to position 0
    assert torch.allclose(weights[0, 0, 0], F.softmax(torch.tensor([0.0, -1e9, -1e9, -1e9, -1e9]), dim=-1), atol=1e-5)
