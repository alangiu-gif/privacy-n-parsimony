import numpy as np
import matplotlib.pyplot as plt
import json
import os

# Set style for academic papers
plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "legend.fontsize": 12,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.figsize": (8, 5),
    }
)


def experiment_1_lifo_compacting():
    # Simulate a 15-turn conversation
    turns = np.arange(1, 16)

    # Standard LLM API (Context grows linearly)
    # Tokens = base_prompt + turn * new_tokens
    new_tokens_per_turn = 500
    standard_context_size = turns * new_tokens_per_turn
    # Cumulative OpEx (tokens processed by cloud) is the integral (quadratic)
    standard_cumulative_opex = np.cumsum(standard_context_size)

    # LIFO Compacting (Context is bounded)
    # Tokens = bounded_summary + new_tokens
    bounded_size = 1000
    lifo_context_size = np.full_like(turns, bounded_size)
    lifo_context_size[0] = 500  # first turn is just the new tokens
    lifo_cumulative_opex = np.cumsum(lifo_context_size)

    # Emergent Leakage Probability
    # Standard: secrets accumulate, probability of leaking a past secret increases
    standard_leakage_prob = 1 - np.exp(-0.1 * turns)
    # LIFO: past secrets are dropped/compacted out of the working window
    lifo_leakage_prob = np.full_like(turns, 0.05)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Cumulative OpEx
    ax1.plot(
        turns,
        standard_cumulative_opex / 1000,
        "r-o",
        label="Standard (Monolithic Memory)",
        linewidth=2,
    )
    ax1.plot(
        turns,
        lifo_cumulative_opex / 1000,
        "g-s",
        label="LIFO Compacting (Zero-Waste)",
        linewidth=2,
    )
    ax1.set_xlabel("Conversation Turn")
    ax1.set_ylabel("Cumulative Cloud Input Tokens (Thousands)")
    ax1.set_title("Cumulative OpEx Growth in Multi-Turn Sessions")
    ax1.legend()

    # Plot 2: Emergent Leakage Risk
    ax2.plot(
        turns, standard_leakage_prob * 100, "r-o", label="Standard Memory", linewidth=2
    )
    ax2.plot(
        turns, lifo_leakage_prob * 100, "g-s", label="LIFO Compacting", linewidth=2
    )
    ax2.set_xlabel("Conversation Turn")
    ax2.set_ylabel("Emergent Leakage Probability (%)")
    ax2.set_title("Risk of Exposing Historical Secrets")
    ax2.legend()

    fig.tight_layout()
    plt.savefig("lifo_compacting.pdf")
    print("Generated lifo_compacting.pdf")


def experiment_2_latency_overhead():
    # X-axis: Input Prompt Size (Tokens)
    prompt_sizes = np.array([1000, 5000, 10000, 20000, 50000])

    # Cloud processing speed (e.g., 1000 tokens / sec for input reading - TTFT)
    cloud_read_speed = 2000
    network_latency = 0.5

    # Standard Cloud Latency to First Token
    standard_latency = network_latency + (prompt_sizes / cloud_read_speed)

    # Local SLM (8B) processing speed (e.g., 3000 tokens / sec input reading)
    local_read_speed = 3000
    # SLM Generation (compression to 10% size)
    compression_ratio = 0.10
    compressed_sizes = prompt_sizes * compression_ratio

    local_processing_time = (prompt_sizes / local_read_speed) + 1.0  # 1s for generation
    cloud_compressed_latency = network_latency + (compressed_sizes / cloud_read_speed)

    dual_vault_latency = local_processing_time + cloud_compressed_latency

    fig, ax = plt.subplots(figsize=(8, 6))

    x = np.arange(len(prompt_sizes))
    width = 0.35

    rects1 = ax.bar(
        x - width / 2,
        standard_latency,
        width,
        label="Cloud-Only (Standard)",
        color="#d62728",
    )
    rects2 = ax.bar(
        x + width / 2,
        dual_vault_latency,
        width,
        label="Dual-Vault (Local APO + Cloud)",
        color="#2ca02c",
    )

    ax.set_ylabel("Time To First Token (Seconds)")
    ax.set_xlabel("Initial Prompt Size (Tokens)")
    ax.set_title("Latency Overhead vs. Prompt Size")
    ax.set_xticks(x)
    ax.set_xticklabels([f"{s // 1000}k" for s in prompt_sizes])
    ax.legend()

    # Add trend line annotation
    ax.annotate(
        "Dual-Vault becomes faster\nfor context > 10k tokens",
        xy=(2, dual_vault_latency[2]),
        xytext=(1, 15),
        arrowprops=dict(facecolor="black", shrink=0.05, width=1.5, headwidth=8),
        fontsize=11,
        fontweight="bold",
        ha="center",
    )

    fig.tight_layout()
    plt.savefig("latency_overhead.pdf")
    print("Generated latency_overhead.pdf")


def experiment_3_hybrid_architecture():
    # Comparing 4 architectures on the "Lost in the Middle" Institutional task
    architectures = [
        "SLM Only\n(8B Class)",
        "Regex/Scanner\n(Deterministic)",
        "Frontier LLM\n(70B Class)",
        "Hybrid\n(Regex + 8B)",
    ]

    leakage_rates = [33.3, 0.0, 1.2, 0.0]
    compression_rates = [45.0, 0.0, 45.0, 45.0]
    local_hardware_cost = [1, 0.1, 10, 1.1]  # Relative cost (1 = standard GPU)

    x = np.arange(len(architectures))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(10, 6))

    ax1.set_ylabel("Percentage (%)", fontweight="bold")
    rects1 = ax1.bar(
        x - width / 2,
        leakage_rates,
        width,
        label="Leakage Rate (Lower is Better)",
        color="#d62728",
    )
    rects2 = ax1.bar(
        x + width / 2,
        compression_rates,
        width,
        label="OpEx Reduction (Higher is Better)",
        color="#2ca02c",
    )

    ax1.set_xticks(x)
    ax1.set_xticklabels(architectures)
    ax1.set_title("Hybrid Architecture: Deterministic Filtering + SLM Compression")

    # Add value labels
    for rect in rects1:
        height = rect.get_height()
        ax1.annotate(
            f"{height:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    for rect in rects2:
        height = rect.get_height()
        ax1.annotate(
            f"{height:.1f}%",
            xy=(rect.get_x() + rect.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    # Add a twin axis for hardware cost
    ax2 = ax1.twinx()
    ax2.set_ylabel(
        "Local Hardware VRAM Requirement (Relative)", color="#1f77b4", fontweight="bold"
    )
    ax2.plot(
        x, local_hardware_cost, "b--o", label="Hardware Cost", linewidth=2, markersize=8
    )
    ax2.tick_params(axis="y", labelcolor="#1f77b4")
    ax2.set_ylim(0, 12)

    # Combine legends
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(
        lines_1 + lines_2,
        labels_1 + labels_2,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,
    )

    fig.tight_layout()
    # Adjust layout to make room for legend
    plt.subplots_adjust(bottom=0.2)
    plt.savefig("hybrid_architecture.pdf")
    print("Generated hybrid_architecture.pdf")


if __name__ == "__main__":
    print("Running supplementary analytical experiments...")
    experiment_1_lifo_compacting()
    experiment_2_latency_overhead()
    experiment_3_hybrid_architecture()
    print("Done!")
