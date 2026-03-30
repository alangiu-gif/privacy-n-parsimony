import json
import numpy as np
import matplotlib.pyplot as plt
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


def safe_divide(n, d):
    return n / d if d > 0 else 0


def load_jsonl(file_path):
    data = []
    if os.path.exists(file_path):
        with open(file_path, "r") as f:
            for line in f:
                if line.strip():
                    data.append(json.loads(line))
    return data


def generate_quadrant_plot():
    matrix_data = load_jsonl("colab_matrix_results.jsonl")
    if not matrix_data:
        print("No matrix data found.")
        return

    quadrants = [
        ("expert", "personal", "Expert\nPersonal"),
        ("expert", "institutional", "Expert\nInstitutional"),
        ("lazy", "personal", "Lazy\nPersonal"),
        ("lazy", "institutional", "Lazy\nInstitutional"),
    ]

    labels = []
    opex_reductions = []
    leakage_rates = []

    for profile, typology, label in quadrants:
        subset = [
            d
            for d in matrix_data
            if d.get("profile") == profile and d.get("typology") == typology
        ]

        if subset:
            avg_red = np.mean([d.get("reduction_percent", 0) for d in subset])
            total_leaks = sum(d.get("leaks", 0) for d in subset)
            total_secrets = sum(d.get("total_secrets", 0) for d in subset)
            leak_rate = safe_divide(total_leaks, total_secrets) * 100

            labels.append(label)
            opex_reductions.append(avg_red)
            leakage_rates.append(leak_rate)

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    rects1 = ax.bar(
        x - width / 2,
        opex_reductions,
        width,
        label="OpEx Reduction (%)",
        color="#2ca02c",
    )
    rects2 = ax.bar(
        x + width / 2, leakage_rates, width, label="Leakage Rate (%)", color="#d62728"
    )

    ax.set_ylabel("Percentage (%)")
    ax.set_title("Token Parsimony vs Data Leakage by Quadrant")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()

    # Add value labels
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(
                f"{height:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, height),
                xytext=(0, 3),  # 3 points vertical offset
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    autolabel(rects1)
    autolabel(rects2)

    fig.tight_layout()
    plt.savefig("quadrant_results.pdf")
    print("Generated quadrant_results.pdf")


def generate_quality_plot():
    # From our 40-sample benchmark
    labels = ["Dual-Vault (APO) Wins", "Baseline (Raw) Wins", "Ties"]
    sizes = [85.0, 10.0, 5.0]
    colors = ["#2ca02c", "#d62728", "#7f7f7f"]
    explode = (0.1, 0, 0)  # explode 1st slice

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(
        sizes,
        explode=explode,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        shadow=False,
        startangle=90,
        textprops={"fontsize": 14},
    )
    ax.axis("equal")  # Equal aspect ratio ensures that pie is drawn as a circle.
    ax.set_title("Answer Quality Evaluation (LLM-as-a-Judge)", pad=20)

    fig.tight_layout()
    plt.savefig("answer_quality.pdf")
    print("Generated answer_quality.pdf")


def generate_decomposition_plot():
    # From our 118-sample benchmark
    labels = ["Baseline (Monolithic)", "Dual-Vault (Decomposition)"]
    costs = [6.71, 0.16]
    colors = ["#d62728", "#2ca02c"]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.bar(labels, costs, color=colors, width=0.5)

    ax.set_ylabel("Projected Cloud Cost (USD)")
    ax.set_title("OpEx Reduction via Task Decomposition (118 log samples)")

    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"${height:.2f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=14,
            fontweight="bold",
        )

    # Add an arrow showing the reduction
    ax.annotate(
        "-97.5% OpEx",
        xy=(0.5, 3.0),
        xytext=(0.5, 4.0),
        arrowprops=dict(facecolor="black", shrink=0.05, width=2, headwidth=10),
        ha="center",
        va="bottom",
        fontsize=14,
        fontweight="bold",
        color="black",
    )

    fig.tight_layout()
    plt.savefig("task_decomposition.pdf")
    print("Generated task_decomposition.pdf")


if __name__ == "__main__":
    generate_quadrant_plot()
    generate_quality_plot()
    generate_decomposition_plot()
