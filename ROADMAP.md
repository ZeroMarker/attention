# LLM Development · Training · Deployment Roadmap

A phased plan for turning the current encoder–decoder Transformer MVP
(`./attention`) into a trained and deployed LLM. Each phase lists concrete
milestones, the repo work it implies, and an exit criterion.

> Current state: **MVP** — a from-scratch encoder–decoder Transformer with
> correct forward pass, masks, and unit tests. No tokenizer, no training
> loop, no inference engine, no serving.

---

## Decision point: model family

The current model is **encoder–decoder** (Vaswani et al., 2017). Most modern
LLMs (GPT, LLaMA, Mistral) are **decoder-only causal LMs** that do next-token
prediction. Choose one track:

- **Decoder-only (recommended for an LLM)** — drop cross-attention, mask
  self-attention causally everywhere, tie embeddings. Building blocks to keep:
  `MultiHeadAttention`, `DecoderLayer` (minus `cross_attn`), `ffn`, masks.
- **Encoder–decoder** (T5-style) — keep today's structure; use for seq2seq
  (translation, summarization). Not the LLM path.

The milestones below assume **decoder-only**.

---

## Phase 1 — Model development

Goal: a scaffold that *trains stably* and generates text.

| # | Milestone | Details | Done when |
|---|---|---|---|
| 1.1 | Causal LM wrapper | New `causal.py`: `DecoderLayer` without `cross_attn`, causal mask on every self-attention layer, `forward(tokens) -> logits (B, S, V)`, `generate()` stub | Loss decreases on a toy copy task |
| 1.2 | Modern building blocks | Replace `LayerNorm`→`RMSNorm`; sinusoidal→**RoPE**; FFN→**SwiGLU**; add **GQA** (grouped-query attention) | Perplexity at least as good as LN/sinusoidal baseline at same budget |
| 1.3 | KV cache | Cache K/V per layer across decode steps; `generate()` streams one token at a time | Cache hits; memory ~ `2 × n_layers × n_kv_head × seq × d_k` per batch |
| 1.4 | Tokenizer | Train a **BPE** tokenizer (vocab 32k–64k, add `<pad/unk/bos/eos>`); tokenize encode/trim/mask | Round-trip `decode(encode(x)) == x` on held-out text |
| 1.5 | Weight tying | Share input embedding and output head (`tie_word_embeddings`) | Params drop by `vocab × d_model`; quality unchanged |
| 1.6 | Config scaling | `attention/config.py`: add `rms_norm`, `rope`, `gqa(kv_heads)`, `context_len`, `tie_embeddings` | Config variants for micro (smoke) / small (e.g. 125M) / base |

Exit criterion: a micro causal LM trains on a small corpus and yields
decreasing loss + fluent-ish samples. Encoder–decoder code stays usable but
is not on the LLM path.

---

## Phase 2 — Pretraining

Goal: train at scale with a reproducible, observable pipeline.

| # | Milestone | Details | Done when |
|---|---|---|---|
| 2.1 | Objective | Teacher-forced next-token prediction; `CrossEntropyLoss(ignore_index=pad)`; pack sequences with segment masks | Loss matches `-log_p` on a known corpus |
| 2.2 | Data pipeline | Streaming reader → tokenize → pack to `context_len` → shuffle → dedup/filter; configurable dataset (e.g. FineWeb / The Pile) | Deterministic across runs with a seed |
| 2.3 | Optimizer | `AdamW` (decoupled weight decay), weight decay on weights only (not norms/biases), cosine LR + linear warmup, grad clip, **bf16** autocast | Loss curve smooth; no NaN with clip |
| 2.4 | Single-GPU train loop | `train.py` + `eval.py`, checkpoint/resume, gradient accumulation | Reproducible run; resume restores loss trace |
| 2.5 | Distributed | DDP (single node) → **FSDP / DeepSpeed ZeRO-3** (multi-node), gradient accumulation, activation checkpointing, `FlashAttention` | Throughput scales with GPUs; loss/grad-norm identical to single-node (small run) |
| 2.6 | Compute budget | Chinchilla target ≈ `20 × params` tokens (e.g. 7B → ~140B tokens) | Budget/hardware plan in `docs/`; MFU recorded |
| 2.7 | Observability | Log loss, val perplexity, grad norm, LR, tokens/s, GPU util; checkpoint every N steps; eval on fixed validation set | Dashboards/run logs reproducible |

Exit criterion: a small-model run matches reference perplexity within a few
percent at a target token budget.

---

## Phase 3 — Evaluation & alignment

Goal: a model that is both capable and safe.

| # | Milestone | Details | Done when |
|---|---|---|---|
| 3.1 | Capability evals | Perplexity + HellaSwag, WinoGrande, ARC-e, GSM8K, MMLU, HumanEval/AIME | Baseline numbers recorded per checkpoint |
| 3.2 | SFT | Instruction tuning on curated data (system/user/assistant), loss only on completions | Model follows format; loss on assistant tokens drops |
| 3.3 | Alignment | **DPO** (simpler) or RLHF (PPO + reward model); best-of-N / rejection sampling | Preference win-rate improves at equal capability |
| 3.4 | Safety | Toxicity/refusal/truthfulness/hallucination checks; red-teaming | Alignment tax quantified; safety regressions blocked |
| 3.5 | Checkpoint selection | Hold-out eval + safety gate before selection | A single promoted checkpoint with artifact hash |

Exit criterion: a checkpoint that beats the base model on the eval suite
without a safety regression.

---

## Phase 4 — Deployment / serving

Goal: a safe, efficient serving stack.

| # | Milestone | Details | Done when |
|---|---|---|---|
| 4.1 | Generation | Greedy + sampling (temperature, top-k, top-p, min-p); seeded + streaming | Deterministic seeded output; stable streaming |
| 4.2 | Serving engine | **vLLM** (PagedAttention, continuous batching) or TGI/TensorRT-LLM; OpenAI-compatible HTTP API | Throughput/latency benchmarks (tokens/s, TTFT, TBT) |
| 4.3 | Quantization | int8 / int4 (GPTQ, AWQ, bitsandbytes), FP8; KV-cache quantization | Quality drop measured on eval suite; memory reduced |
| 4.4 | Hardware | Single → multi-GPU; tensor parallel for >7B | Cost/latency/throughput tradeoff documented |
| 4.5 | Product layer | Auth, rate limiting, request/response streaming (SSE), timeouts | API contract spec + integration test |
| 4.6 | Observability & cost | Latency, tokens/s, GPU util, error rate, cost per 1M tokens; alerting | Dashboards + alert rules |
| 4.7 | Rollout | Canary/shadow, blue-green; eval-gated promotion; rollback | Production release with rollback path |

Exit criterion: a versioned serving deployment with defined SLOs, cost per
token, and a rollback plan.

---

## Phase 5 — Maintain & iterate

- Continual: data refresh (fresh/clean subsets), eval regression gates,
  red-teaming, drift monitoring, safety updates, quant/latency tuning.
- Periodic: re-train or continue-pretrain on new data; re-align; re-eval.

---

## Cross-cutting

- **Reproducibility**: pin seeds, data version, config hash, framework versions;
  log everything; store checkpoints with metadata.
- **Safety**: treat safety as a first-class gate through Phase 3–5, not an
  afterthought; keep a red-team test suite.
- **Cost control**: record token budget, GPU-hours, and $/1M tokens at every
  phase so scaling decisions are evidence-based.

## References

- Vaswani et al., *Attention Is All You Need*, 2017.
- Hoffmann et al., *Training Compute-Optimal LLMs* (Chinchilla), 2022.
- Touvron et al., *LLaMA: Open and Efficient Foundation Language Models*, 2023.
- Su et al., *RoFormer* (RoPE) / Shazeer, *GLU Variants & GQA* / Zhang & Sennrich, *RMSNorm*.
- Huang et al., *FlashAttention*, 2022.
- Kwon et al., *Efficient Memory Management for LLM Serving with PagedAttention* (vLLM), 2023.
- Rafailov et al., *Direct Preference Optimization*, 2023.
