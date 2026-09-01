# Attention Is All You Need

A from-scratch implementation of the Transformer model, following
[*Attention Is All You Need*](https://arxiv.org/abs/1706.03762)
(Vaswani et al., 2017).

> **Status: scaffold.** This repository currently contains only project
> metadata (`.gitignore`, `LICENSE`). No implementation exists yet. The
> sections below describe the *intended* design, not shipped code.

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

Training objectives, data pipelines, and decoding strategies (greedy,
beam search) are planned but out of scope for the core implementation.

## Intended layout

```
attention/
├── model/
│   ├── __init__.py
│   ├── attention.py      # scaled dot-product + multi-head attention
│   ├── embeddings.py     # token embeddings + positional encoding
│   ├── encoder.py        # encoder layer and stack
│   ├── decoder.py        # decoder layer and stack
│   └── transformer.py    # full Transformer model
├── tests/                # unit tests per component
├── requirements.txt
└── README.md
```

## Intended usage

```python
from attention.model import Transformer

model = Transformer(
    vocab_size=32_000,
    d_model=512,
    n_head=8,
    n_layers=6,
    d_ff=2048,
    dropout=0.1,
)

logits = model(src_tokens, tgt_tokens)
```

Configuration mirrors the paper's `base` (d_model=512, 8 heads, 6
layers) and `big` (d_model=1024, 16 heads, 6 layers) settings.

## Roadmap

- [ ] Scaled dot-product attention
- [ ] Multi-head attention
- [ ] Positional encoding
- [ ] Encoder / decoder layers and stacks
- [ ] Full Transformer forward pass
- [ ] Unit tests for each component
- [ ] Training loop and example dataset

## References

- Vaswani et al., [Attention Is All You Need](https://arxiv.org/abs/1706.03762), 2017.

## License

[MIT](LICENSE) © 2026 Mark Chen
