# Benchmark Results: Privacy Guard & Token Parsimony

Full empirical validation of the Dual-Vault framework across three metrics.

---

## Execution Parameters

| Parameter | Value |
|---|---|
| Privacy Guard model (APO & Scrubbing) | `qwen2.5-7b-instruct` (LM Studio / Ollama) |
| Cloud model (Tier 0, large-scale) | `qwen2.5:7b` via Ollama Q4_0 (Google Colab T4) |
| Judge model | `qwen2.5-7b-instruct` |
| Dataset | Synthetic, generated with `Faker` |
| Local hardware | Apple Silicon M4 Max, 128GB RAM |
| Cloud hardware | NVIDIA T4 GPU (Google Colab) |

---

## Metric 1: Token Parsimony (OpEx Reduction)

**Small-scale validation (40 samples, local hardware)**

| Quadrant | Baseline tokens | Dual-Vault tokens | Reduction |
|---|---|---|---|
| Expert / Personal | 51.6 | 37.3 (±4.9) | **27.7%** |
| Expert / Institutional | 39.5 | 34.0 (±0.0) | **13.9%** |
| Lazy / Personal | 141.6 | 59.4 (±1.8) | **58.1%** |
| Lazy / Institutional | 136.8 | 63.0 (±2.6) | **53.9%** |
| **Blended (40 samples)** | — | — | **47.6%** |

- Local APO latency: **0.67s** (±0.17s) per sample

**Large-scale validation (1,000 samples, Colab T4, Ollama Q4_0)**

- Blended OpEx reduction: **45.0%** (mathematically stable across all quadrants)

---

## Metric 2: Sanitisation Efficacy (Zero Leakage)

**Large-scale (1,000 samples, 3,250 injected secrets — 1,300 Personal, 1,950 Institutional)**

| Quadrant | Leakage rate | Leaked / Total secrets |
|---|---|---|
| Expert / Personal | **0%** | 0 / 500 |
| Expert / Institutional | **0%** | 0 / 500 |
| Lazy / Personal | **0%** | 0 / 1,000 |
| Lazy / Institutional | **33.6%** | 420 / 1,250 |
| **Blended** | **12.9%** | 420 / 3,250 |

**Key finding**: the 7B model achieves perfect redaction on 3 out of 4 quadrants. Failure is exclusively concentrated in the Lazy/Institutional quadrant, where cryptographic keys and API tokens are buried inside massive unstructured server log dumps ("Lost in the Middle" effect).

**Parameter scaling exploration (8B → 104B, needle-in-a-haystack)**

- 32B (Qwen 2.5 Coder) and 70B (Llama 3.3): 0% leakage on exploratory task
- 8B, 30B, 104B classes: leakage rates ranging from 33.3% to 100%

**Follow-up statistical validation — Llama 3.3 70B (50 samples, Lazy/Institutional)**

- Leakage rate: **1.20%** (3 / 250 secrets leaked)
- Confirms frontier models approach near-perfect redaction, but a deterministic layer (e.g. Microsoft Presidio) remains necessary to guarantee 100% TPR in enterprise deployments.

---

## Metric 3: Answer Quality (LLM-as-a-Judge)

**Evaluation: 40 samples, judge model `qwen3-coder-30b`**

Comparing raw verbose prompts (Baseline) vs. APO-compressed sanitised prompts (Dual-Vault):

| Outcome | Count | Rate |
|---|---|---|
| Dual-Vault preferred | 34 | **85.0%** |
| Tie | 2 | 5.0% |
| Baseline preferred | 4 | 10.0% |

**Key finding**: aggressive semantic compression by the local APO not only preserves intent but actively improves final Cloud response quality by removing conversational noise.

---

## Task Decomposition OpEx Reduction

**118 samples, simulated server logs (~11,300 tokens/sample)**

| Routing strategy | Cloud cost (118 samples) | OpEx reduction |
|---|---|---|
| Monolithic (full log → Cloud) | $6.71 | — |
| Decomposed (local extraction + selective routing) | $0.16 | **97.54%** |

---

## Architectural Conclusions

1. **APO and Scrubbing must be decoupled**: a 7B model reliably compresses prompts (APO) but fails at strict syntactic redaction on verbose unstructured data. These are conflicting objectives for non-safety-aligned sub-8B models.
2. **Scaling parameter count is not the solution**: the 30B class performs worse than 7B on scrubbing (54.4% leakage). Leakage rate is driven by architecture and alignment, not raw parameter count.
3. **Recommended production architecture**: local APO (7B–14B) + deterministic PII layer (Microsoft Presidio) + tiered cloud routing.
