# DraftRadar — Speculative Decoding Benchmarking Harness

DraftRadar measures whether vLLM's speculative decoding actually speeds things
up, and for what kind of workload — instead of assuming it's a universal win.

## The problem

Speculative decoding lets a small "draft" model guess ahead while a larger
"target" model verifies the guesses in bulk. It's often treated as a safe
default optimization. The paper *"Speculative Speculative Decoding"*
(arXiv 2603.03251, 2026) argues that draft-model acceptance rate — the thing
that determines whether this actually pays off — varies a lot by task type,
and that variance is under-benchmarked in practice.

## What this project does

Benchmarks vLLM's `draft_model` speculative decoding method across:
- **2 draft/target configurations**: a narrow gap (Qwen2.5-1.5B target /
  0.5B draft) and a wide gap (Qwen2.5-3B target / 0.5B draft)
- **3 task families**: code (HumanEval), chat (OASST1), and tool-calling
  (hand-authored prompts)
- **True no-speculation baselines** for each config, on the same target model

All metrics (acceptance rate, throughput, TTFT, latency) are pulled directly
from vLLM's own Prometheus-style metrics API — not estimated or simulated.
Every run is logged to DuckDB.

## Key results

| Config | Task | Speedup vs. baseline | Acceptance rate |
|---|---|---|---|
| Narrow (1.5B/0.5B) | code | 0.88x | 0.870 |
| Narrow | chat | 0.67x | 0.786 |
| Narrow | tool-calling | 0.83x | 0.721 |
| Wide (3B/0.5B) | code | **1.36x** | 0.891 |
| Wide | chat | 0.99x | 0.567 |
| Wide | tool-calling | 0.88x | 0.611 |

**Only 1 of 6 tested configurations beat plain decoding**: code generation
with the wider draft/target gap. Everything else was neutral or a net
slowdown. Widening the gap improved acceptance rate for code but *hurt* it
for chat and tool-calling — a bigger target model diverged more from the
small draft's guesses on less predictable text.

Throughput also scaled strongly with concurrency (21 → 67 → 133 tokens/sec
across batch sizes 1 → 4 → 10), meaning the benefit is far more pronounced
under real production load than in single-request testing.

**Recommendation**: enable speculative decoding for code-generation traffic
with a wide draft/target size gap; leave it off for chat or tool-calling
traffic on this model family and hardware class.

See [`results/`](./results) for charts and the raw DuckDB file, and
[`docs/methodology.md`](./docs/methodology.md) for setup details and honest
caveats.

## Stack

Python · vLLM 0.28.0 · DuckDB · Matplotlib · Qwen2.5 model family

## How to run

This was built and run on a free Google Colab T4 GPU. Open
[`notebooks/draftradar_benchmark.ipynb`](./notebooks/draftradar_benchmark.ipynb)
in Colab, select a T4 runtime, and run all cells in order.

```bash
pip install -r requirements.txt
```

## Limitations

- Tool-calling prompts are hand-authored (10 examples), not from a canonical
  benchmark — a real limitation, stated plainly.
- Only one draft model size (0.5B) was tested; the target model was varied,
  not the draft.
- Small sample size (10 prompts per task family).
- All runs used `enforce_eager=True` (CUDA graphs disabled) for T4
  compatibility — results may differ on more capable hardware.

## License

MIT