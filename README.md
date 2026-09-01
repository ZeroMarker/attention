# Attention Is All You Need

A from-scratch implementation of the Transformer model, following
[*Attention Is All You Need*](https://arxiv.org/abs/1706.03762)
(Vaswani et al., 2017).

> **Status: MVP.** A working encoder-decoder Transformer forward pass is
> implemented and tested. Training loops and decoding strategies are planned.

## Goal

Build the Transformer end-to-end in Python — from the primitive
(scaled dot-product attention) up to a full encoder–decoder model — with
readable, dependency-light code that makes every component's math
explicit. The primary goal is understanding: each building block is
written out and documented, not hidden behind a framework abstraction.

## Architecture

The model follows the canonical Transformer layout:

| Component                    | Role |
|------------------------------|------|
| **Scaled dot-product attention** | `softmax(QKᵀ / √dₖ)V` — the core attention primitive |
| **Multi-head attention**     | Parallel attention heads projected over the full representation |
| **Position-wise FFN**        | Two linear layers with a ReLU (or GELU) activation, applied per position |
| **Positional encoding**      | Sinusoidal (or learned) encoding so attention can use order |
| **Encoder stack**            | `N` identical layers: multi-head self-attention + FFN, with residual connections and layer norm |
| **Decoder stack**            | `N` identical layers: masked self-attention + encoder-decoder attention + FFN |
| **Output layer**             | Linear projection to vocabulary size + softmax |

## Package layout

```
attention/
├── __init__.py        # public exports
├── config.py          # TransformerConfig dataclass
├── attention.py       # scaled dot-product + multi-head attention
├── embeddings.py      # token embeddings + sinusoidal positional encoding
├── feedforward.py     # position-wise feed-forward network
├── encoder.py         # encoder layer and stack
├── decoder.py         # decoder layer and stack
├── transformer.py     # full Transformer model
└── utils.py           # attention-mask helpers
├── tests/             # unit tests per component
├── requirements.txt
└── README.md
```

## Usage

```python
from attention import Transformer, TransformerConfig

model = Transformer(TransformerConfig(
    vocab_size=32_000,
    d_model=512,
    n_head=8,
    n_layers=6,
    d_ff=2048,
    dropout=0.1,
    max_seq_len=512,
    pad_id=0,
))

# Full encoder-decoder forward pass -> (batch, tgt_len, vocab_size)
logits = model(src_tokens, tgt_tokens)

# Lower-level building blocks
memory = model.encode(src_tokens)          # (batch, src_len, d_model)
decoded = model.decode(tgt_tokens, memory)  # (batch, tgt_len, d_model)
```

`TransformerConfig` defaults to the paper's `base` setting (d_model=512,
8 heads, 6 layers); `big` is d_model=1024, 16 heads, 6 layers. The
decoder self-attention is causally masked for you, and source/target
padding (tokens equal to `pad_id`) is masked automatically.

## Install & test

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
pytest
```

The core forward pass uses only PyTorch; the mask helpers are pure
PyTorch tensor ops built from scratch.

## Roadmap

- [x] Scaled dot-product attention
- [x] Multi-head attention
- [x] Positional encoding
- [x] Encoder / decoder layers and stacks
- [x] Full Transformer forward pass
- [x] Unit tests for each component
- [ ] Training loop and example dataset
- [ ] Greedy / beam-search decoding

## References

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017.

## License

[MIT](LICENSE) © 2026 Mark Chen
