# Attention Is All You Need

A readable, from-scratch PyTorch implementation of the Transformer from
[*Attention Is All You Need*](https://arxiv.org/abs/1706.03762)
(Vaswani et al., 2017): the paper's encoder–decoder model **and** the
decoder-only causal language model that the modern LLM path is built on.

> **Status: MVP.** Both model families run training forward passes and batched
> greedy decoding. There is no training loop, tokenizer, KV cache, or sampling
> yet; see [ROADMAP.md](ROADMAP.md).

## Model families

| Class | Family | Typical use |
|---|---|---|
| `Transformer` | encoder–decoder (the paper, T5-style) | seq2seq: translation, summarization |
| `CausalLM` | decoder-only (GPT/LLaMA-style) | next-token prediction; the LLM path |

Both are configured by the same frozen `TransformerConfig`, and they share one
implementation of the post-norm block. A decoder-only block *is* the encoder
block — self-attention plus feed-forward — because causality is carried
entirely by the attention mask, not by the block's parameters. `CausalLM`
therefore reuses `EncoderLayer` and `Encoder` instead of a second,
near-identical layer class that would have to be kept in sync.

## What is implemented

- Scaled dot-product and multi-head attention
- Fixed sinusoidal positional encoding
- Position-wise feed-forward networks with ReLU or GELU
- Post-norm encoder, decoder, and decoder-only stacks with residual connections
- Automatic source-padding, target-padding, causal, and cross-attention masks
- Batched greedy generation with optional EOS early stopping
- Decoder-only causal language model with greedy continuation
- Tests for components, masking, end-to-end shape behavior, training, and
  generation

The implementation intentionally uses explicit PyTorch tensor operations
instead of `torch.nn.Transformer`, making the model useful as a compact
educational reference.

## Architecture

`Transformer` follows the original encoder–decoder architecture:

| Component | Role |
|---|---|
| Scaled dot-product attention | Computes `softmax(QKᵀ / √dₖ)V` |
| Multi-head attention | Runs attention in parallel learned subspaces |
| Position-wise FFN | Applies two linear layers independently at each position |
| Positional encoding | Adds fixed sinusoidal position information |
| Encoder | Repeats self-attention and FFN sublayers |
| Decoder | Repeats causal self-attention, cross-attention, and FFN sublayers |
| Output projection | Maps decoder states to vocabulary logits; no softmax is applied |

Every sublayer uses a residual connection followed by layer normalization,
matching the post-norm layout in the paper.

`CausalLM` drops cross-attention and the source side entirely:

```
tokens → TokenEmbedding → + PositionalEncoding → [EncoderLayer] × n_layers → Linear → logits
             ↑ self-attention runs under a causal mask on every layer
```

Two consequences worth knowing:

- Only the mask changes for the LLM path. The attention mask
  (`build_self_attention_mask(..., causal=True)`) is the single place that
  enforces causality, and the model has no source sequence at all — so
  `generate()` takes a prompt instead of a `src`.
- Positions are still absolute sinusoids. RoPE, RMSNorm, SwiGLU, and GQA are
  §1.2 of the roadmap; the KV cache is §1.3.

## Installation

Python 3.10 or newer is required. The suite is verified on CPython 3.12.3 with
PyTorch 2.13.0 (CPU).

```bash
git clone https://github.com/ZeroMarker/attention.git
cd attention
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

Run the test suite from the repository root:

```bash
python -m pytest
```

The package is imported from the repository root and is not installed, so use
`python -m pytest` rather than a bare `pytest`: the `-m` form puts the working
directory on `sys.path`, while bare `pytest` fails collection with
`ModuleNotFoundError: No module named 'attention'`.

## Quick start

```python
import torch

from attention import Transformer, TransformerConfig

config = TransformerConfig(
    vocab_size=32_000,
    d_model=512,
    n_head=8,
    n_layers=6,
    d_ff=2048,
    dropout=0.1,
    max_seq_len=512,
    pad_id=0,
)
model = Transformer(config)

src_tokens = torch.randint(1, config.vocab_size, (2, 16))
tgt_tokens = torch.randint(1, config.vocab_size, (2, 12))

# Shape: (batch, target_length, vocab_size). Values are raw logits.
logits = model(src_tokens, tgt_tokens)

# The encoder and decoder can also be called independently.
memory = model.encode(src_tokens)           # (2, 16, 512)
decoded = model.decode(tgt_tokens, memory)  # (2, 12, 512)
```

The decoder applies a causal mask automatically. Tokens equal to `pad_id` are
excluded as attention keys in source and target padding masks.

## Training inputs

`Transformer.forward(src, tgt)` returns logits for each token in `tgt`. For
teacher-forced next-token training, pass a right-shifted target to the decoder
and compare its logits with the unshifted labels:

```python
import torch.nn.functional as F

# full_target starts with BOS and ends with EOS/padding
decoder_input = full_target[:, :-1]
labels = full_target[:, 1:]

logits = model(src_tokens, decoder_input)
loss = F.cross_entropy(
    logits.reshape(-1, config.vocab_size),
    labels.reshape(-1),
    ignore_index=config.pad_id,
)
```

This repository provides the model components only; optimizer setup, batching,
checkpointing, and dataset code remain roadmap items. The decoder-only path
uses the same objective over a single sequence — see
[Decoder-only causal LM](#decoder-only-causal-lm).

## Greedy generation

```python
model.eval()
generated = model.generate(
    src_tokens,
    bos_id=1,
    eos_id=2,
    max_new_tokens=64,
)
```

`generate()`:

- encodes the source once, then selects `argmax` at each decoder step;
- returns shape `(batch, generated_length)` including the initial BOS token;
- stops early when every sequence emits `eos_id`, when one is supplied;
- pads positions after EOS with `config.pad_id` while other batch items finish;
- runs without gradient tracking and restores the model's previous train/eval
  mode afterward; and
- requires `max_new_tokens + 1 <= config.max_seq_len`.

Generation currently recomputes decoder attention over the full generated
prefix at each step. A KV cache and alternative decoding strategies are
planned.

## Decoder-only causal LM

`CausalLM` is the LLM path (the "tiny GPT" of `ROADMAP.md` §1.1): the same
modules minus cross-attention, with a causal mask on self-attention everywhere.
It reuses the post-norm encoder block — a decoder-only block is self-attention
plus feed-forward, and causality is carried entirely by the mask — so the two
paths cannot drift apart.

```python
import torch
import torch.nn.functional as F

from attention import CausalLM, TransformerConfig

config = TransformerConfig(
    vocab_size=10_000,
    d_model=256,
    n_head=8,
    n_layers=6,
    d_ff=1024,
    max_seq_len=256,
    pad_id=0,
)
model = CausalLM(config)

tokens = torch.randint(0, config.vocab_size, (4, 64))
logits = model(tokens)  # (4, 64, 10000): next-token logits per position

# Teacher-forced next-token objective.
loss = F.cross_entropy(
    logits[:, :-1].reshape(-1, config.vocab_size),
    tokens[:, 1:].reshape(-1),
    ignore_index=config.pad_id,
)
```

`generate(prompt, eos_id=None, max_new_tokens=64)` greedily continues a prompt,
returning `(batch, prompt_len + generated)` with the prompt preserved verbatim.
It mirrors the encoder–decoder generation contract: early stop once every item
emits `eos_id`, `pad_id` filler for finished items, no gradient tracking, and
the previous train/eval mode restored afterward. It requires
`prompt_len + max_new_tokens <= config.max_seq_len`.

## Configuration

`TransformerConfig` is an immutable dataclass with the following defaults:

| Field | Default | Meaning |
|---|---:|---|
| `vocab_size` | `32000` | Number of token IDs |
| `d_model` | `512` | Embedding and hidden-state width |
| `n_head` | `8` | Number of attention heads |
| `n_layers` | `6` | Number of layers in each stack |
| `d_ff` | `2048` | Feed-forward hidden width |
| `dropout` | `0.1` | Dropout probability |
| `max_seq_len` | `512` | Maximum source or target length |
| `activation` | `"relu"` | FFN activation: `"relu"` or `"gelu"` |
| `pad_id` | `0` | Token ID treated as padding |

`d_model` must be divisible by `n_head`. The defaults match the main dimensions
of the paper's base model; all fields can be overridden directly.

For `CausalLM`, `max_seq_len` is the total context — prompt plus generation —
and `pad_id` positions are excluded as attention keys, so a padded prompt is
safe to feed in.

## Mask convention

Masks are boolean tensors where `True` means “attend” and `False` means
“masked out.” Helper functions in `attention.utils` produce:

| Helper | Shape |
|---|---|
| `make_padding_mask(tokens, pad_id)` | `(batch, sequence)` |
| `make_causal_mask(sequence_length)` | `(sequence, sequence)` |
| `build_self_attention_mask(...)` | `(batch, 1, query, key)` |
| `build_cross_attention_mask(...)` | `(batch, 1, 1, source_length)` |

The `build_*` helpers default to the device of the token tensor, so padding and
causal parts always agree. The same module holds the attention numerics:
`apply_mask` writes `-inf` into masked logits, and `softmax_attention` runs a
numerically stable softmax that maps an all-masked query row to zero weights
instead of `NaN`, keeping batches with an all-padding example finite.

## Repository layout

```text
.
├── attention/
│   ├── __init__.py        # public exports
│   ├── attention.py       # scaled dot-product and multi-head attention
│   ├── causal.py          # decoder-only CausalLM
│   ├── config.py          # TransformerConfig
│   ├── decoder.py         # decoder layer and stack
│   ├── embeddings.py      # token and positional embeddings
│   ├── encoder.py         # self-attention + FFN block and stack (both families)
│   ├── feedforward.py     # position-wise feed-forward network
│   ├── transformer.py     # encoder–decoder model and greedy generation
│   └── utils.py           # attention-mask helpers and numerics
├── tests/
│   ├── test_attention.py    # scaled dot-product and multi-head attention
│   ├── test_causal.py       # decoder-only model, masking, training, generation
│   ├── test_embeddings.py   # token embeddings and positional encoding
│   ├── test_masks.py        # mask helpers
│   └── test_transformer.py  # encoder–decoder end-to-end behavior
├── ROADMAP.md             # longer-term development plan
└── requirements.txt
```

## Roadmap

- [x] Core encoder–decoder Transformer
- [x] Decoder-only causal LM with greedy continuation
- [x] Automatic attention masks
- [x] Batched greedy decoding
- [x] Component and end-to-end tests
- [ ] Training loop and example dataset
- [ ] Tokenizer
- [ ] KV cache
- [ ] Sampling and beam-search decoding

See [ROADMAP.md](ROADMAP.md) for the full model-development, training,
evaluation, and deployment plan.

## License

[MIT](LICENSE) © 2026 Mark Chen
