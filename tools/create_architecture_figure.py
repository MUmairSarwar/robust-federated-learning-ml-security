"""Create the repository's architecture figure without running an experiment."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]


def box(ax, x, y, width, height, text, color):
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle="round,pad=0.02,rounding_size=0.02",
        facecolor=color, edgecolor="#1d3557", linewidth=1.5,
    )
    ax.add_patch(patch)
    ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", fontsize=11)


def arrow(ax, start, end):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=15, color="#455a64", linewidth=1.6))


def main():
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    box(ax, 0.02, 0.38, 0.13, 0.22, "MNIST\npublic benchmark", "#dbeafe")
    box(ax, 0.20, 0.38, 0.14, 0.22, "Seeded client\npartitions\n(IID / non-IID)", "#e0f2fe")
    box(ax, 0.39, 0.62, 0.15, 0.20, "Honest clients\nlocal training", "#dcfce7")
    box(ax, 0.39, 0.18, 0.15, 0.24, "Malicious clients\nlabel poisoning or\ncorner backdoor", "#fee2e2")
    box(ax, 0.60, 0.38, 0.15, 0.24, "Server aggregation\nFedAvg · median\ntrimmed mean · fuzzy", "#fef3c7")
    box(ax, 0.81, 0.58, 0.16, 0.20, "Updated global CNN\nnext round", "#ede9fe")
    box(ax, 0.81, 0.18, 0.16, 0.22, "Evaluation\nclean accuracy · loss\nattack success rate", "#f3e8ff")

    arrow(ax, (0.15, 0.49), (0.20, 0.49))
    arrow(ax, (0.34, 0.49), (0.39, 0.72))
    arrow(ax, (0.34, 0.49), (0.39, 0.30))
    arrow(ax, (0.54, 0.72), (0.60, 0.55))
    arrow(ax, (0.54, 0.30), (0.60, 0.45))
    arrow(ax, (0.75, 0.54), (0.81, 0.67))
    arrow(ax, (0.75, 0.46), (0.81, 0.29))
    arrow(ax, (0.89, 0.58), (0.72, 0.62))

    ax.text(0.5, 0.94, "Federated-Learning Security Experiment", ha="center", va="center", fontsize=20, fontweight="bold", color="#17324d")
    ax.text(0.5, 0.06, "Controlled defensive simulation — no real users, services or production data", ha="center", fontsize=10, color="#52606d")
    fig.tight_layout()
    fig.savefig(ROOT / "docs" / "architecture.svg", bbox_inches="tight")
    fig.savefig(ROOT / "docs" / "architecture.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
