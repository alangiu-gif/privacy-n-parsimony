# Privacy Guard & Token Parsimony Framework

Empirical benchmark suite for the **Privacy Guard & Token Parsimony** framework, accompanying the paper:

> *Privacy Guard & Token Parsimony by Prompt and Context Handling and LLM Routing*
> Alessio Langiu — CNR-ISMAR
> Preprint: [arXiv link] | Code: [GitHub link] | Replication Bundle: [Zenodo DOI]

---

## What This Is

This repository contains all scripts, Colab notebooks, and synthetic datasets used to empirically validate the **Dual-Vault architecture** — a hybrid LLM routing framework that treats context compression (Token Parsimony) and data sanitisation (Zero Leakage) as dual projections of the same operator.

The benchmark evaluates three metrics across a **2×2 test matrix** (user profile: Lazy vs. Expert; secret typology: Personal vs. Institutional):

| Metric | Description | Key result |
|---|---|---|
| **Metric 1** | Token Parsimony (OpEx reduction) | 45% blended cost reduction |
| **Metric 2** | Sanitisation Efficacy (leakage rate) | 0% leakage on 3/4 quadrants |
| **Metric 3** | Answer Quality (LLM-as-a-Judge) | 85% preference for APO-compressed responses |

Full results: [`RESULTS.md`](RESULTS.md)

---

## Architecture

```
User Prompt
    │
    ▼
[Local Privacy Guard — SLM 7B–14B]
    ├── APO: abstractive prompt compression & task decomposition
    └── Scrubbing: PII/secret redaction (+ deterministic layer)
    │
    ▼
[Zero-Trust Router]
    ├── Tier 0 — Untrusted / sanitised-only
    ├── Tier 1 — Commercial standard (NDA)
    ├── Tier 2 — GDPR-compliant (EU jurisdiction)
    └── Tier 3 — On-premise / Zero-Trust
```

---

## Requirements

```bash
pip install openai faker tqdm
```

The scripts use an **OpenAI-compatible API** — works with any local inference server:
- [LM Studio](https://lmstudio.ai/) (recommended for Apple Silicon)
- [Ollama](https://ollama.com/)
- Any OpenAI-compatible endpoint

---

## Configuration

All endpoints and models are configurable via environment variables:

```bash
# Local inference server (Privacy Guard / APO model)
export LOCAL_API_BASE="http://localhost:1234/v1"
export LOCAL_API_KEY="lm-studio"          # or "ollama", or any string

# Cloud / Tier-1 endpoint (for routing benchmarks)
export CLOUD_API_BASE="http://localhost:4000/v1"
export CLOUD_API_KEY="your-api-key"
```

Set the model names inside each script or override via env vars as documented in each file's docstring.

---

## Repository Structure

```
├── benchmark_core.py            # Core 2x2 PoC benchmark (APO + scrubbing)
├── run_metric1_parsimony.py     # Metric 1: Token Parsimony, 1,000-sample run
├── run_metric2_decomposition.py # Metric 2: Task Decomposition OpEx
├── run_metric3_quality.py       # Metric 3: LLM-as-a-Judge quality evaluation
├── evaluate_answer_quality.py   # Lightweight 40-sample quality evaluation
│
├── test_30b_sanitisation.py       # Sanitisation test — 30B parameter model
├── test_frontier_sanitisation.py  # Sanitisation test — Frontier models (e.g., Llama 70B)
├── preliminary_frontier_test.py   # Exploratory scaling test (8B→104B)
├── run_large_scale.py           # Large-scale overnight runner
├── exponential_scale_test.py    # Exponential scaling analysis
│
├── Supplementary_Experiments.ipynb # Analytical projections (LIFO, Latency, Hybrid)
├── run_supplementary_experiments.py# Script for analytical projections plots
│
├── generate_dataset.py          # Synthetic dataset generator (Faker)
├── generate_plots.py            # Plot generation from results
├── plot_exploratory.py          # Exploratory tier plots
├── plot_architecture_v3.py      # System architecture flowchart generator
│
├── print_results.py             # Print Colab/cloud benchmark results
├── print_local_results.py       # Print local benchmark results
│
├── *.ipynb                      # Google Colab notebooks (ready to run)
├── *.pdf                        # Pre-generated result figures
│
└── RESULTS.md                   # Full numerical results
```

---

## Reproducing the Main Results

**Metric 1 — Token Parsimony (1,000 samples):**
```bash
export LOCAL_API_BASE="http://localhost:1234/v1"
python run_metric1_parsimony.py
```

**Metric 2 — Task Decomposition:**
```bash
python run_metric2_decomposition.py
```

**Metric 3 — Answer Quality (LLM-as-a-Judge):**
```bash
export CLOUD_API_BASE="http://localhost:4000/v1"
python run_metric3_quality.py
```

**Quick PoC (40 samples):**
```bash
python benchmark_core.py
```

Alternatively, open the Colab notebooks for a fully hosted execution:
- `Privacy_Guard_2x2_Colab.ipynb` — Metrics 1 & 2
- `Task_Decomposition_Colab.ipynb` — Metric 2
- `Privacy_Guard_Combined_Colab.ipynb` — All metrics

---

## Citation

If you use this benchmark in your research, please cite:

```bibtex
@misc{langiu2025privacyguard,
  title={Privacy Guard \& Token Parsimony by Prompt and Context Handling and LLM Routing},
  author={Langiu, Alessio},
  year={2026},
  eprint={ARXIV_ID},
  archivePrefix={arXiv},
  primaryClass={cs.CR}
}
```

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

## Acknowledgements

The author used large language model assistance (Claude, Anthropic) for manuscript editing, bibliography verification, and development of benchmark scripts. All scientific claims, experimental design, results, and conclusions are solely the responsibility of the author.
