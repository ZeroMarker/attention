# LLM Development · Training · Deployment Roadmap

A phased plan for turning the current Transformer package (`./attention`) into
a trained and deployed LLM. Each phase lists concrete milestones, the repo work
it implies, and an exit criterion.

> Current state: **MVP + §1.1 done** — a from-scratch encoder–decoder
> Transformer plus a decoder-only causal LM (`attention.CausalLM`) with
> correct forward pass, masks, greedy decoding, and unit tests. The Phase 1
> exit criterion has been met once by hand; see [Notes on §1.1](#notes-on-11).
> No tokenizer, training loop, KV cache, optimized inference engine, or
> serving. Apart from the encoder–decoder MVP's greedy decoding (§4.1), no
> milestone below §1.1 is implemented.

---

## Decision point: model family

The repo's original model is **encoder–decoder** (Vaswani et al., 2017). Most
modern LLMs (GPT, LLaMA, Mistral) are **decoder-only causal LMs** that do
next-token prediction.

**Track chosen: decoder-only**, implemented as `attention.CausalLM` (§1.1). The
encoder–decoder model stays in the repo as the seq2seq (T5-style) reference and
is not on the LLM path.

The original plan for §1.1 was to drop `cross_attn` from `DecoderLayer`. That
was unnecessary: a decoder-only block is *exactly* the encoder block —
self-attention plus feed-forward — because causality comes from the attention
mask, not the block's parameters. `CausalLM` therefore reuses `EncoderLayer`,
which keeps one tested post-norm block serving both paths instead of a second
near-identical `DecoderLayer` variant. It also reuses `Encoder` as the stack.

The milestones below assume **decoder-only**.

---

## Phase 1 — Model development

Goal: a scaffold that *trains stably* and generates text.

| # | Milestone | Details | Done when |
|---|---|---|---|
| 1.1 ✅ | Causal LM wrapper | `attention/causal.py`: decoder-only stack (reuses the post-norm encoder block — a decoder-only block *is* self-attention + FFN, with causality carried entirely by the mask, so both paths share one implementation), causal mask on every self-attention layer, `forward(tokens) -> logits (B, S, V)`; `generate(prompt, eos_id, max_new_tokens)` mirrors the encoder-decoder greedy API | Done — see [Notes on §1.1](#notes-on-11) |
| 1.2 | Modern building blocks | Replace `LayerNorm`→`RMSNorm`; sinusoidal→**RoPE**; FFN→**SwiGLU**; add **GQA** (grouped-query attention) | Perplexity at least as good as LN/sinusoidal baseline at same budget |
| 1.3 | KV cache | Cache K/V per layer across decode steps; `generate()` streams one token at a time | Cache hits; memory ~ `2 × n_layers × n_kv_head × seq × d_k` per batch |
| 1.4 | Tokenizer | Train a **BPE** tokenizer (vocab 32k–64k, add `<pad/unk/bos/eos>`); tokenize encode/trim/mask | Round-trip `decode(encode(x)) == x` on held-out text |
| 1.5 | Weight tying | Share input embedding and output head (`tie_word_embeddings`) | Params drop by `vocab × d_model`; quality unchanged |
| 1.6 | Config scaling | `attention/config.py`: add `rms_norm`, `rope`, `gqa(kv_heads)`, `context_len`, `tie_embeddings` | Config variants for micro (smoke) / small (e.g. 125M) / base |

Exit criterion: a micro causal LM trains on a small corpus and yields
decreasing loss + fluent-ish samples. Encoder–decoder code stays usable but
is not on the LLM path.

### Notes on §1.1

**Deviation from the plan: the acceptance task is a *successor* task, not a
copy task.** The original criterion was "loss decreases on a toy copy task".
A copy task at this scale did not converge fast enough to make a stable
regression test: with `d_model=32, n_head=4, n_layers=2, d_ff=64, vocab=100`,
Adam at `lr=1e-3`, batch 8, length 12, the held-out loss over all positions
moved `4.8254 → 4.7384` (40 steps), `→ 4.6963` (80), `→ 4.6181` (150). The
committed test `test_training_step_reduces_loss_on_successor_task` instead
uses sequences where `tokens[i+1] == tokens[i] % 99 + 1`. The next token is
then a deterministic function of the previous one, the task cannot be
memorized per batch, and the same model and learning rate reach
`4.696 → 2.5319` on held-out sequences in 60 steps. The test asserts on
held-out loss only.

**Exit-criterion evidence (ad-hoc run; the script was not committed).** A
char-level micro LM trained from scratch on a 3,560-character repeated-English
corpus (29-token vocab) with `d_model=128, n_head=4, n_layers=4, d_ff=512,
max_seq_len=64, dropout=0` — **800,512 parameters** — under `AdamW` at
`lr=3e-3`, batch 16, random crops of length 64:

| step | 0 | 25 | 50 | 75 | 100 | 125 | 150 |
|---|---|---|---|---|---|---|---|
| loss | 3.6063 | 0.5715 | 0.1693 | 0.0642 | 0.0515 | 0.0593 | 0.0651 |

Greedy continuation of the prompt `the quick` returned
`quick brown fox jumps over the lazy dog. the `. The corpus is tiny and
repetitive, so this run shows the objective optimizes and greedy decoding
emits coherent text — not that the model generalizes. Reproduce it with the
recipe above; there is no committed script.

### Notes on §1.3

`tests/test_causal.py::test_appended_padding_keys_do_not_change_prefix_logits`
already pins the invariant a cache has to preserve: appending padding keys
must not change prefix logits. Keep it green when swapping in cached decode.

---

## Phase 2 — Pretraining

Goal: train at scale with a reproducible, observable pipeline.

| # | Milestone | Details | Done when |
|---|---|---|---|
| 2.1 | Objective | Teacher-forced next-token prediction; `CrossEntropyLoss(ignore_index=pad)`; pack sequences with segment masks. The shape contract is already in place: `logits[:, :-1]` against `tokens[:, 1:]` (README, *Decoder-only causal LM*) | Loss matches `-log_p` on a known corpus |
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
| 4.1 | Generation | Greedy decoding is complete for **both** families — `Transformer.generate(src, bos_id, ...)` and `CausalLM.generate(prompt, ...)`, sharing the early-stop/EOS-mode contract. Still to add: sampling (temperature, top-k, top-p, min-p), seeded generation, and streaming. | Deterministic seeded output; stable streaming |
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
