import json
import numpy as np


def stats(file):
    try:
        with open(file, "r") as f:
            data = [json.loads(l) for l in f if l.strip()]
        return data
    except Exception as e:
        return []


matrix = stats(
    "./matrix_1000_results.jsonl"
)
decomp = stats(
    "./decomp_1000_results.jsonl"
)
overnight = stats(
    "./overnight_results.jsonl"
)

print(f"=== LOCAL MATRIX RESULTS ===")
if matrix:
    total_bl = sum(d.get("baseline_tokens", 0) for d in matrix)
    total_dv = sum(d.get("dv_tokens", 0) for d in matrix)
    blended_red = 100 - (total_dv / total_bl * 100) if total_bl > 0 else 0
    total_leaks = sum(
        sum(v for v in d.get("leak_report", {}).values() if isinstance(v, bool))
        for d in matrix
    )
    total_secrets = sum(len(d.get("secrets_injected", [])) for d in matrix)
    leakage_rate = (total_leaks / total_secrets * 100) if total_secrets > 0 else 0
    print(f"Samples: {len(matrix)}")
    print(f"Blended OpEx Reduction: {blended_red:.2f}%")
    print(f"Total Leaks: {total_leaks}/{total_secrets} ({leakage_rate:.2f}%)")
else:
    print("No valid data.")

print(f"\n=== LOCAL DECOMPOSITION RESULTS ===")
if decomp:
    total_bl_cost = sum(d.get("baseline_cost_usd", 0) for d in decomp)
    total_dv_cost = sum(d.get("dual_vault_cost_usd", 0) for d in decomp)
    cost_red = 100 - (total_dv_cost / total_bl_cost * 100) if total_bl_cost > 0 else 0
    print(f"Samples: {len(decomp)}")
    print(f"Total Cost Reduction: {cost_red:.2f}%")
else:
    print("No valid data.")

print(f"\n=== LOCAL OVERNIGHT RESULTS (Faker Dynamic) ===")
if overnight:
    total_bl = sum(d.get("baseline_tokens", 0) for d in overnight)
    total_dv = sum(d.get("dv_tokens", 0) for d in overnight)
    blended_red = 100 - (total_dv / total_bl * 100) if total_bl > 0 else 0
    total_leaks = sum(
        sum(v for v in d.get("leak_report", {}).values() if isinstance(v, bool))
        for d in overnight
    )
    total_secrets = sum(len(d.get("secrets_injected", [])) for d in overnight)
    leakage_rate = (total_leaks / total_secrets * 100) if total_secrets > 0 else 0
    print(f"Samples: {len(overnight)}")
    print(f"Blended OpEx Reduction: {blended_red:.2f}%")
    print(f"Total Leaks: {total_leaks}/{total_secrets} ({leakage_rate:.2f}%)")
else:
    print("No valid data.")
