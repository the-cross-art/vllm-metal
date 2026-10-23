# Independent verification request: LFM2-VL image input

## What you are being asked to do

Determine, from scratch, whether **image input actually works end to end** on
this branch. Start the engine on an LFM2-VL checkpoint, send a real image, get
an answer about that image, and compare the output against native mlx-vlm.

Do not assume it works. Do not assume it is broken. Run it and report what you
observe.

## Background, briefly

`vllm-metal` is a plugin that lets vLLM serve models on Apple Silicon Macs
using MLX. vLLM owns the API server, scheduler and paged KV cache; `mlx-lm` and
`mlx-vlm` own the model implementations; vllm-metal is the layer between them.

This branch adds support for **LFM2-VL / LFM2.5-VL**, a vision-language model
family from Liquid AI, so that image prompts reach the vision tower instead of
being ignored.

The new code is in:

- `vllm_metal/multimodal/lfm2_vl/adapter.py`
- the dispatch in `vllm_metal/v1/model_adapter.py`
- `tools/lfm2_vl_vision_smoke.py`

## Machine

Apple M5 Pro, 48 GB unified memory, macOS 26.7.1, full Xcode installed.

## Setup

Use a **fresh clone in a new directory**. Do not reuse an existing checkout:
other directories on this machine contain working notes that would bias your
investigation.

```bash
git clone https://github.com/the-cross-art/vllm-metal.git ~/lfm2-verify
cd ~/lfm2-verify
git checkout lfm2-vl-verify
./install.sh
source .venv-vllm-metal/bin/activate
uv pip install -e ".[dev]"
python -m vllm_metal.metal.build
```

Always set `PYTORCH_ENABLE_MPS_FALLBACK=0` for runs, as the project's
`AGENTS.md` requires; it stops PyTorch silently falling back to CPU and
masking a Metal failure.

## The task

1. **Unit tests** — confirm the branch's own tests pass:

   ```bash
   PYTORCH_ENABLE_MPS_FALLBACK=0 pytest -q tests/multimodal/ tests/v1/mm tests/attention \
     tests/test_model_adapter.py tests/test_attention_sdpa.py tests/test_paged_attention.py
   ```

2. **End-to-end image serving** — the actual question. A smoke tool exists:

   ```bash
   PYTORCH_ENABLE_MPS_FALLBACK=0 python tools/lfm2_vl_vision_smoke.py
   ```

   You may also drive it manually with `vllm serve` and an OpenAI-style
   `image_url` request, or with `vllm.LLM(...).chat(...)`, whichever you
   prefer. The goal is an answer that demonstrably depends on the image.

3. **Parity** — if you get it running, compare the engine's first-token
   log-probs against native mlx-vlm for the same image and prompt. Report
   EXACT and TOP_K_MATCH counts separately, as the project's `AGENTS.md`
   requires for model changes.

## Checkpoints to try

These all exist on Hugging Face. Try whichever you judge appropriate:

- `LiquidAI/LFM2.5-VL-450M-MLX-4bit`
- `LiquidAI/LFM2.5-VL-3B-MLX-4bit`
- `LiquidAI/LFM2.5-VL-1.6B`
- `mlx-community/LFM2-VL-450M-4bit`

## What to report

Be specific and quote real output. For each thing you tried:

- the exact command
- the full error, if any
- which layer it failed in, and whose code that is — the checkpoint's
  packaging, transformers, vLLM core, vllm-metal, or mlx-vlm
- any workaround you applied, and whether it got you further
- whether a served answer actually depended on the image content, or could
  have been produced from the text prompt alone

End with a clear verdict:

- **WORKS** — image input functions end to end; include the parity numbers
- **BLOCKED** — it cannot currently run; name every blocker and whose it is
- **WORKS WITH CAVEATS** — it runs only after changes outside this branch;
  state exactly what you had to change

If you conclude it is blocked, say whether each blocker is fixable inside
vllm-metal or requires a change in a dependency or in the published
checkpoints.

---

# Findings

> Fill this section in. Replace the placeholders.

## Environment

- vLLM version:
- transformers version:
- mlx / mlx-lm / mlx-vlm versions:

## Unit tests

```
(paste result)
```

## End-to-end attempts

### Attempt 1

- checkpoint:
- command:
- result:

```
(paste output or error)
```

- layer that failed, and whose code:

### Attempt 2

- checkpoint:
- command:
- result:

```
(paste output or error)
```

- layer that failed, and whose code:

## Parity, if reached

- EXACT:
- TOP_K_MATCH:
- KL(engine || mlx-vlm):
- did the answer depend on the image:

## Blockers found

| # | Blocker | Layer | Fixable in vllm-metal? |
|---|---|---|---|
| 1 | | | |
| 2 | | | |

## Verdict

**WORKS / BLOCKED / WORKS WITH CAVEATS**

Reasoning:
