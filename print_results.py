import json
import numpy as np
import sys


def stats(file):
    try:
        with open(file, "r") as f:
            data = [json.loads(l) for l in f if l.strip()]
        return data
    except Exception as e:
        return []


matrix = stats(
    "./colab_matrix_results.jsonl"
)
decomp = stats(
    "./colab_decomposition_results.jsonl"
)

if matrix:
    total_bl = sum(d["baseline_tokens"] for d in matrix)
    total_dv = sum(d["dv_tokens"] for d in matrix)
    blended_red = 100 - (total_dv / total_bl * 100) if total_bl > 0 else 0
    total_leaks = sum(d["leaks"] for d in matrix)
    total_secrets = sum(d["total_secrets"] for d in matrix)
    leakage_rate = (total_leaks / total_secrets * 100) if total_secrets > 0 else 0
    print(f"=== MATRIX 2x2 BENCHMARK ({len(matrix)} samples) ===")
    print(f"Total Baseline Tokens: {total_bl}")
    print(f"Total Dual-Vault Tokens: {total_dv}")
    print(f"Blended OpEx Reduction: {blended_red:.2f}%")
    print(f"Total Leaks: {total_leaks}/{total_secrets} ({leakage_rate:.2f}%)")

    print("\nBy Quadrant:")
    for p in ["expert", "lazy"]:
        for t in ["personal", "institutional"]:
            quad = [d for d in matrix if d["profile"] == p and d["typology"] == t]
            if quad:
                avg_red = np.mean([d["reduction_percent"] for d in quad])
                lks = sum(d["leaks"] for d in quad)
                secs = sum(d["total_secrets"] for d in quad)
                avg_lat = np.mean([d["latency"] for d in quad])
                print(
                    f"  [{p.upper()} / {t.upper()}]: Reduction: {avg_red:.1f}% | Latency: {avg_lat:.2f}s | Leaks: {lks}/{secs}"
                )

if decomp:
    total_bl_cost = sum(d["baseline_cost_usd"] for d in decomp)
    total_dv_cost = sum(d["dual_vault_cost_usd"] for d in decomp)
    cost_red = 100 - (total_dv_cost / total_bl_cost * 100) if total_bl_cost > 0 else 0
    avg_lat = np.mean([d["latency_sec"] for d in decomp])
    print(f"\n=== TASK DECOMPOSITION BENCHMARK ({len(decomp)} samples) ===")
    print(f"Total Baseline Cost: ${total_bl_cost:.5f}")
    print(f"Total Dual-Vault Cost: ${total_dv_cost:.5f}")
    print(f"Total Cost Reduction: {cost_red:.2f}%")
    print(f"Average Latency: {avg_lat:.2f}s")
