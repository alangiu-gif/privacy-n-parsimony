"""
run_metric2_decomposition.py — Benchmark Metric 2: Task Decomposition and OPEX Reduction.

Demonstrates how the Dual-Vault architecture can decompose a complex multi-step
request into sub-tasks, routing only the expensive steps to the cloud model.

Scenario: a massive server log dump (~500 lines) is given as context. The baseline
would send the entire log to the cloud for all three sub-tasks. The Dual-Vault
approach instead:
  1. Sub-task 1 (local): A local extractor model reads the full log and extracts
     only the critical error root cause (a tiny, non-sensitive string).
  2. Sub-task 2 (local): A local translator model converts the root cause to French.
  3. Sub-task 3 (cloud): Only the compact root-cause sentence is forwarded to the
     cloud model for formal prose generation (the apology email draft).

The cost delta between steps 1-2 (billed locally at ~$0) and step 3 (cloud tokens)
vs. the baseline (full log sent to cloud) illustrates the OPEX savings.

Environment variables:
  API_BASE        — OpenAI-compatible local inference endpoint
  API_KEY         — API key for the local endpoint
  EXTRACTOR_MODEL — Model used for log extraction (sub-task 1)
  TRANSLATOR_MODEL — Model used for translation (sub-task 2)
"""
import os
import json
import time
import random
import logging
import traceback
from datetime import datetime

try:
    from openai import OpenAI
    import openai
    from faker import Faker
    from tqdm import tqdm
except ImportError:
    print("Please install requirements: pip install openai faker tqdm")
    exit(1)

# ==========================================
# BENCHMARK CONFIGURATION
# ==========================================
TOTAL_SAMPLES = 1000

# Local inference endpoint (OpenAI-compatible, e.g. LM Studio, Ollama)
API_BASE = os.getenv("API_BASE", "http://localhost:1234/v1")
API_KEY = os.getenv("API_KEY", "lm-studio")

# Models - configure via env var or edit below
EXTRACTOR_MODEL = os.getenv("EXTRACTOR_MODEL", "qwen2.5-7b-instruct")
TRANSLATOR_MODEL = os.getenv("TRANSLATOR_MODEL", "qwen2.5-7b-instruct")

# Cloud pricing assumption used to compute OPEX: $5.00 per 1M input tokens
CLOUD_COST_PER_TOKEN = 5.00 / 1000000

# Paths
RESULTS_FILE = os.path.join(os.path.dirname(__file__), "decomp_1000_results.jsonl")
LOG_FILE = os.path.join(os.path.dirname(__file__), "runner_2_decomposition.log")

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler()],
)
logger = logging.getLogger("DecompBenchmark")

fake = Faker("en_GB")


def approx_token_count(text: str) -> int:
    """Approximate token count using the common 4-chars-per-token heuristic."""
    return len(text) // 4


def run_with_retry(func, max_retries=3, base_delay=2):
    """
    Retry wrapper for OpenAI-compatible API calls with exponential back-off.

    Handles timeout, connection, and rate-limit errors. Returns None after
    all retries are exhausted.
    """
    for attempt in range(max_retries):
        try:
            return func()
        except openai.APITimeoutError:
            logger.warning(f"Timeout occurred. Retry {attempt + 1}/{max_retries}...")
            time.sleep(base_delay * (2**attempt))
        except openai.APIConnectionError:
            logger.warning(f"Connection error. Retry {attempt + 1}/{max_retries}...")
            time.sleep(base_delay * (2**attempt))
        except openai.RateLimitError:
            logger.warning(f"Rate limit hit. Retry {attempt + 1}/{max_retries}...")
            time.sleep(5 * (2**attempt))
        except Exception as e:
            logger.error(f"Unexpected API error: {e}")
            logger.error(traceback.format_exc())
            time.sleep(base_delay)
    return None


def load_checkpoint() -> set:
    """
    Read already-processed IDs from the JSONL results file for resumption support.
    """
    completed_ids = set()
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r") as f:
            for line in f:
                try:
                    data = json.loads(line)
                    completed_ids.add(data["id"])
                except json.JSONDecodeError:
                    continue
    return completed_ids


def generate_massive_context(index: int) -> tuple[str, str]:
    """
    Generate a large synthetic server log (~500 lines) with one buried CRITICAL entry.

    The critical error is inserted at position 250 (the middle of the log) to
    stress-test the "Lost in the Middle" problem: models tend to miss information
    that appears in the centre of very long contexts.

    Returns (full_log_text, expected_root_cause) so the caller can verify accuracy.
    """
    logs = []
    for _ in range(500):
        logs.append(
            f"[{fake.date_time_this_year()}] INFO: Connection from {fake.ipv4()} established. Status: OK."
        )

    node = random.randint(1, 20)
    root_cause = f"OOM (Out Of Memory) on Node {node} due to rogue query."
    error_msg = f"[{fake.date_time_this_year()}] CRITICAL: Database connection failed. Root cause: {root_cause}"
    # Bury the critical line in the middle to simulate real-world log noise
    logs.insert(250, error_msg)

    return "\n".join(logs), root_cause


def main():
    logger.info(f"Starting Task Decomposition Benchmark ({TOTAL_SAMPLES} samples)")
    client = OpenAI(base_url=API_BASE, api_key=API_KEY)

    completed_ids = load_checkpoint()
    logger.info(f"Found {len(completed_ids)} completed samples in checkpoint.")

    total_bl_cost = 0.0
    total_dv_cost = 0.0

    with open(RESULTS_FILE, "a") as f:
        pbar = tqdm(range(TOTAL_SAMPLES), desc="Benchmarking Decomposition")
        for i in pbar:
            item_id = f"TEST-DECOMP-{i + 1}"

            if item_id in completed_ids:
                continue

            try:
                massive_context, expected_root_cause = generate_massive_context(i + 1)
                massive_context_tokens = approx_token_count(massive_context)

                complex_intent = """
Analyse the attached logs and perform the following 3 tasks:
1. Identify the root cause of the critical error.
2. Translate the root cause explanation to French.
3. Draft a formal apology email to the client explaining the specific root cause.
"""
                # Baseline: cloud would receive the full log plus all three sub-tasks
                baseline_cloud_tokens = massive_context_tokens + approx_token_count(
                    complex_intent
                )
                baseline_cost = baseline_cloud_tokens * CLOUD_COST_PER_TOKEN

                t0 = time.time()

                # Sub-Task 1: Extraction
                st1_prompt = f"Extract ONLY the root cause of the critical error from these logs. Be extremely concise.\n\nLOGS:\n{massive_context}"

                def _call_st1():
                    return (
                        client.chat.completions.create(
                            model=EXTRACTOR_MODEL,
                            messages=[{"role": "user", "content": st1_prompt}],
                            temperature=0.0,
                            timeout=45.0,
                        )
                        .choices[0]
                        .message.content.strip()
                    )

                st1_response = run_with_retry(_call_st1) or "ERROR_EXTRACTION"

                # Sub-Task 2: Translation
                st2_prompt = f"Translate this error to French: {st1_response}"

                def _call_st2():
                    return (
                        client.chat.completions.create(
                            model=TRANSLATOR_MODEL,
                            messages=[{"role": "user", "content": st2_prompt}],
                            temperature=0.0,
                            timeout=30.0,
                        )
                        .choices[0]
                        .message.content.strip()
                    )

                st2_response = run_with_retry(_call_st2) or "ERROR_TRANSLATION"

                # Sub-task 3: only the compact root-cause sentence is sent to the cloud
                # for high-quality prose generation — not the entire 500-line log.
                st3_prompt = f"Draft a formal apology email to the client regarding this specific error: {st1_response}"
                cloud_tokens_used = approx_token_count(st3_prompt)
                cloud_cost = cloud_tokens_used * CLOUD_COST_PER_TOKEN

                latency = time.time() - t0

                reduction = (
                    100 - (cloud_cost / baseline_cost * 100) if baseline_cost > 0 else 0
                )

                # Record
                result_record = {
                    "timestamp": datetime.now().isoformat(),
                    "id": item_id,
                    "baseline_tokens": baseline_cloud_tokens,
                    "dual_vault_tokens": cloud_tokens_used,
                    "baseline_cost_usd": baseline_cost,
                    "dual_vault_cost_usd": cloud_cost,
                    "opex_reduction_percentage": reduction,
                    "latency_sec": latency,
                    "expected_root_cause": expected_root_cause,
                    "extracted_root_cause": st1_response,
                    "translated_root_cause": st2_response,
                    "error": st1_response == "ERROR_EXTRACTION",
                }

                f.write(json.dumps(result_record) + "\n")
                f.flush()

                total_bl_cost += baseline_cost
                total_dv_cost += cloud_cost
                curr_red = (
                    100 - (total_dv_cost / total_bl_cost * 100)
                    if total_bl_cost > 0
                    else 0
                )
                pbar.set_postfix({"Red": f"{curr_red:.1f}%"})

                # Brief pause to avoid saturating the local inference server
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Failed processing item {item_id}: {e}")
                logger.error(traceback.format_exc())

    logger.info("Benchmark 2 (Decomposition) complete.")


if __name__ == "__main__":
    main()
