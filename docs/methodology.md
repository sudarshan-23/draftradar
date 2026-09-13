# Methodology

## Hardware & software

- GPU: NVIDIA Tesla T4 (compute capability 7.5, 15GB VRAM), free-tier Google Colab
- vLLM 0.28.0, `dtype="half"` (T4 does not support bfloat16)
- `enforce_eager=True` — CUDA graphs disabled for T4 stability
- Speculative method: `draft_model` (requires no special-trained draft head,
  works with any two same-family models)

## Models

- Draft: Qwen2.5-0.5B-Instruct (fixed across both configs)
- Target (narrow gap): Qwen2.5-1.5B-Instruct
- Target (wide gap): Qwen2.5-3B-Instruct
- `num_speculative_tokens=3` for all speculative runs

## Task families

- **Code**: 10 prompts from OpenAI HumanEval
- **Chat**: 10 prompter-role messages from OpenAssistant/oasst1
- **Tool-calling**: 10 hand-authored prompts (a gated dataset,
  Salesforce/xlam-function-calling-60k, was unavailable without
  authentication, so representative prompts were written manually)

## Metrics

All metrics are read directly from vLLM's `llm.get_metrics()` API
(Prometheus-style counters/histograms), diffed before/after each run to
isolate that run's contribution from the engine's cumulative totals.

- **Acceptance rate** = accepted draft tokens / proposed draft tokens
- **TPS** = total output tokens / measured wall-clock time
- **TTFT** = mean time-to-first-token, from vLLM's own histogram
- Every spec run included a 1-prompt warm-up (discarded from timing) to
  absorb Triton JIT compilation cost before the timed run

## Known caveats

- The first task-family run in a session absorbs extra warm-up/compilation
  cost even with the warm-up prompt; the very first TTFT measurement in a
  fresh engine (code, narrow-gap, ~3.3s TTFT) is inflated by this and should
  not be compared directly against later runs.
- One OSL-sweep data point (256 max_tokens, wide-gap, code) showed unusually
  low throughput likely due to uneven batch completion times / JIT
  recompilation for a new sequence-length shape — noted rather than
  smoothed over.
