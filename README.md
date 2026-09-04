# Attention Is All You Need

A readable, from-scratch PyTorch implementation of the encoder–decoder
Transformer from [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762)
(Vaswani et al., 2017).

> **Status: MVP.** The model supports training forward passes and batched
> greedy decoding. A training pipeline, tokenizer, sampling, and beam search
> are not included yet.

## What is implemented

- Scaled dot-product and multi-head attention
- Fixed sinusoidal positional encoding
- Position-wise feed-forward networks with ReLU or GELU
- Post-norm encoder and decoder stacks with residual connections
- Automatic source-padding, target-padding, causal, and cross-attention masks
- Batched greedy generation with optional EOS early stopping
- Tests for components, masking, end-to-end shape behavior, training, and
  generation

The implementation intentionally uses explicit PyTorch tensor operations
instead of `torch.nn.Transformer`, making the model useful as a compact
educational reference.

## Architecture

The model follows the original encoder–decoder architecture:

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

## Installation

Python 3.10 or newer is required.

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
checkpointing, and dataset code remain roadmap items.

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

## Mask convention

Masks are boolean tensors where `True` means “attend” and `False` means
“masked out.” Helper functions in `attention.utils` produce:

| Helper | Shape |
|---|---|
| `make_padding_mask(tokens, pad_id)` | `(batch, sequence)` |
| `make_causal_mask(sequence_length)` | `(sequence, sequence)` |
| `build_self_attention_mask(...)` | `(batch, 1, query, key)` |
| `build_cross_attention_mask(...)` | `(batch, 1, 1, source_length)` |

Fully masked attention rows safely produce zero weights instead of NaNs.

## Repository layout

```text
.
├── attention/
│   ├── __init__.py        # public exports
│   ├── attention.py       # scaled dot-product and multi-head attention
│   ├── config.py          # TransformerConfig
│   ├── decoder.py         # decoder layer and stack
│   ├── embeddings.py      # token and positional embeddings
│   ├── encoder.py         # encoder layer and stack
│   ├── feedforward.py     # position-wise feed-forward network
│   ├── transformer.py     # complete model and greedy generation
│   └── utils.py           # attention-mask helpers
├── tests/                 # unit and integration tests
├── ROADMAP.md             # longer-term development plan
└── requirements.txt
```

## Roadmap

- [x] Core encoder–decoder Transformer
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
