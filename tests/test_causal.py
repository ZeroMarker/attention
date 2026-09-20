"""Tests for the decoder-only causal language model."""
import torch
import torch.nn.functional as F

from attention import CausalLM, TransformerConfig


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


def successor_batch(batch, length, seed):
    """Sequences where ``tokens[i + 1] == tokens[i] % 99 + 1``: the next token is
    a function of the previous one, so the task cannot be memorized per batch."""
    generator = torch.Generator().manual_seed(seed)
    start = torch.randint(1, 100, (batch, 1), generator=generator)
    return (start + torch.arange(length)) % 99 + 1


def test_forward_logits_shape():
    model = CausalLM(make_config())
    tokens = torch.randint(1, 100, (3, 9))
    logits = model(tokens)
    assert logits.shape == (3, 9, 100)


def test_position_logits_ignore_later_tokens():
    model = CausalLM(make_config())
    model.eval()
    tokens = torch.randint(1, 100, (1, 10))
    changed = tokens.clone()
    changed[0, -3:] = (changed[0, -3:] % 99) + 1  # guaranteed different IDs
    with torch.no_grad():
        base = model(tokens)
        other = model(changed)
    assert torch.allclose(base[0, :-3], other[0, :-3], atol=1e-6)


def test_appended_padding_keys_do_not_change_prefix_logits():
    model = CausalLM(make_config())
    model.eval()
    tokens = torch.randint(1, 100, (2, 6))
    padded = torch.cat((tokens, torch.zeros(2, 3, dtype=torch.long)), dim=1)
    with torch.no_grad():
        base = model(tokens)
        wide = model(padded)
    assert torch.allclose(base, wide[:, :6], atol=1e-6)


def test_padding_loss_is_finite():
    model = CausalLM(make_config())
    tokens = torch.tensor([[5, 6, 7, 0, 0], [8, 9, 0, 0, 0]])
    logits = model(tokens)
    loss = F.cross_entropy(
        logits[:, :-1].reshape(-1, 100),
        tokens[:, 1:].reshape(-1),
        ignore_index=0,
    )
    assert torch.isfinite(loss)


def test_training_step_reduces_loss_on_successor_task():
    torch.manual_seed(0)
    model = CausalLM(make_config())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    eval_batch = successor_batch(batch=8, length=12, seed=2)

    def loss_of(tokens):
        logits = model(tokens)
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, 100), tokens[:, 1:].reshape(-1)
        )

    loss0 = loss_of(eval_batch)
    for step in range(60):
        loss = loss_of(successor_batch(batch=8, length=12, seed=100 + step))
        opt.zero_grad()
        loss.backward()
        opt.step()

    assert loss_of(eval_batch) < loss0


def test_generate_continues_prompt_and_restores_mode():
    model = CausalLM(make_config())
    model.train()
    prompt = torch.randint(1, 100, (2, 5))

    generated = model.generate(prompt, max_new_tokens=4)

    assert generated.shape == (2, 9)
    assert torch.equal(generated[:, :5], prompt)
    assert model.training


def test_generate_stops_when_every_sequence_emits_eos():
    model = CausalLM(make_config())
    model.output_proj.weight.data.zero_()  # argmax is token 0 for every item
    prompt = torch.randint(1, 100, (3, 5))

    generated = model.generate(prompt, eos_id=0, max_new_tokens=6)

    assert generated.shape == (3, 6)
    assert torch.equal(generated[:, -1], torch.zeros(3, dtype=torch.long))


def test_generate_rejects_sequence_longer_than_context():
    model = CausalLM(make_config(max_seq_len=6))
    prompt = torch.randint(1, 100, (1, 4))

    try:
        model.generate(prompt, max_new_tokens=4)
    except ValueError as error:
        assert "max_seq_len" in str(error)
    else:
        raise AssertionError("expected generate to reject an oversized sequence")
