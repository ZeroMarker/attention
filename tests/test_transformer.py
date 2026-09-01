"""End-to-end tests for the Transformer model."""
import torch
import torch.nn.functional as F

from attention import Transformer, TransformerConfig


def make_config(**overrides):
    cfg = dict(
        vocab_size=100,
        d_model=32,
        n_head=4,
        n_layers=2,
        d_ff=64,
        max_seq_len=16,
        dropout=0.0,
        pad_id=0,
    )
    cfg.update(overrides)
    return TransformerConfig(**cfg)


def test_forward_logits_shape():
    model = Transformer(make_config())
    src = torch.randint(1, 100, (3, 8))
    tgt = torch.randint(1, 100, (3, 7))
    logits = model(src, tgt)
    assert logits.shape == (3, 7, 100)


def test_encode_decode_shapes():
    model = Transformer(make_config())
    src = torch.randint(1, 100, (2, 8))
    tgt = torch.randint(1, 100, (2, 6))
    memory = model.encode(src)
    assert memory.shape == (2, 8, 32)
    decoded = model.decode(tgt, memory)
    assert decoded.shape == (2, 6, 32)


def test_padding_loss_is_finite():
    model = Transformer(make_config())
    src = torch.tensor([[5, 6, 0, 0], [7, 8, 9, 0]])
    tgt = torch.tensor([[5, 0, 0, 0], [7, 8, 0, 0]])
    logits = model(src, tgt)
    loss = F.cross_entropy(
        logits.reshape(-1, 100), tgt.reshape(-1), ignore_index=0
    )
    assert torch.isfinite(loss)


def test_causal_no_future_leak():
    model = Transformer(make_config())
    model.eval()
    src = torch.randint(1, 100, (1, 8))
    tgt = torch.randint(1, 100, (1, 6))
    with torch.no_grad():
        memory = model.encode(src)
        base = model.decode(tgt, memory)
    # mutate only the last target token; earlier outputs must be unchanged
    tgt_b = tgt.clone()
    tgt_b[0, -1] = (tgt_b[0, -1] % 99) + 1
    with torch.no_grad():
        changed = model.decode(tgt_b, model.encode(src))
    assert torch.allclose(base[0, :-1], changed[0, :-1], atol=1e-6)


def test_training_step_reduces_loss():
    torch.manual_seed(0)
    model = Transformer(make_config())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = F.cross_entropy

    def make_batch():
        src = torch.randint(1, 100, (4, 8))
        # copy task: the decoder reproduces the source sequence
        return src, src.clone()

    src_eval, tgt_eval = make_batch()
    loss0 = loss_fn(model(src_eval, tgt_eval).reshape(-1, 100), tgt_eval.reshape(-1))

    for _ in range(40):
        src, tgt = make_batch()
        logits = model(src, tgt)
        loss = loss_fn(logits.reshape(-1, 100), tgt.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

    loss_end = loss_fn(model(src_eval, tgt_eval).reshape(-1, 100), tgt_eval.reshape(-1))
    assert loss_end < loss0
