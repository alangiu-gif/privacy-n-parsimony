import matplotlib.pyplot as plt
import numpy as np

# Set style for academic papers
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "legend.fontsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 12,
        "figure.figsize": (10, 6),
    }
)

# Data from our exploratory tests
models = [
    "Qwen 2.5\n(7B)",
    "Llama 3.1\n(8B)",
    "Qwen 3 Coder\n(30B)",
    "Qwen 2.5 Coder\n(32B)",
    "Llama 3.3\n(70B)",
    "Command R+\n(104B)",
]
leakage_rates = [100.0, 33.3, 54.4, 0.0, 0.0, 33.3]
tiers = [
    "Tier 3 (Local)",
    "Tier 3 (Local)",
    "Tier 2",
    "Tier 2",
    "Tier 1 (Frontier)",
    "Tier 1 (Frontier)",
]

colors = ["#d62728", "#ff7f0e", "#d62728", "#2ca02c", "#2ca02c", "#ff7f0e"]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(models, leakage_rates, color=colors, width=0.6)

ax.set_ylabel("Leakage Rate (%)")
ax.set_title("Exploratory Leakage Rate by Model Parameter Scale (Lost in the Middle)")
ax.set_ylim(0, 110)

# Add value labels
for bar in bars:
    height = bar.get_height()
    ax.annotate(
        f"{height:.1f}%",
        xy=(bar.get_x() + bar.get_width() / 2, height),
        xytext=(0, 3),  # 3 points vertical offset
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=12,
        fontweight="bold",
    )

plt.axhline(y=0, color="black", linewidth=1)
fig.tight_layout()
plt.savefig("exploratory_tiers.pdf")
print("Generated exploratory_tiers.pdf")

# Save results to JSON
import json

results = [
    {"model": m.replace("\n", " "), "leakage_rate": l, "tier": t}
    for m, l, t in zip(models, leakage_rates, tiers)
]
with open("exploratory_scale_results.json", "w") as f:
    json.dump(results, f, indent=4)
print("Saved exploratory_scale_results.json")
