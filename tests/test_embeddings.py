"""Tests for token embeddings and positional encoding."""
import torch

from attention.embeddings import PositionalEncoding, TokenEmbedding


def test_token_embedding_output_shape():
    emb = TokenEmbedding(100, 32)
    tokens = torch.randint(0, 100, (2, 5))
    out = emb(tokens)
    assert out.shape == (2, 5, 32)
    assert out.requires_grad


def test_token_embedding_scaled_by_sqrt_d_model():
    emb = TokenEmbedding(100, 32)
    tokens = torch.tensor([[7, 9]])
    out = emb(tokens)
    # magnitude is scaled up by sqrt(d_model)
    assert torch.allclose(out / torch.sqrt(torch.tensor(32.0)), emb.embedding(tokens), atol=1e-5)


def test_positional_encoding_shape_and_boundary_values():
    pe = PositionalEncoding(8, max_len=10, dropout=0.0)
    out = pe(torch.zeros(1, 10, 8))
    assert out.shape == (1, 10, 8)
    # position 0: even dims -> sin(0) = 0, odd dims -> cos(0) = 1
    assert torch.allclose(out[0, 0, 0::2], torch.zeros(4), atol=1e-6)
    assert torch.allclose(out[0, 0, 1::2], torch.ones(4), atol=1e-6)


def test_positional_encoding_is_not_learnable():
    pe = PositionalEncoding(8, max_len=10, dropout=0.0)
    assert not pe.pe.requires_grad
    assert len(list(pe.parameters())) == 0


def test_positional_encoding_truncates_to_sequence_length():
    pe = PositionalEncoding(16, max_len=64, dropout=0.0)
    out = pe(torch.zeros(1, 5, 16))
    assert out.shape == (1, 5, 16)
