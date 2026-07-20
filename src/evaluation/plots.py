"""Result figures."""
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def save_run_plots(history: pd.DataFrame, reliability: pd.DataFrame, output_dir: str):
    figures = Path(output_dir) / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    for column, filename, ylabel in [("clean_accuracy", "accuracy_vs_round.png", "Clean accuracy"),
                                     ("train_loss", "loss_vs_round.png", "Local training loss")]:
        ax = history.plot(x="round", y=column, marker="o", legend=False)
        ax.set_ylabel(ylabel); ax.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(figures / filename, dpi=180); plt.close()
    if "attack_success_rate" in history:
        ax = history.plot(x="round", y="attack_success_rate", marker="o", legend=False)
        ax.set_ylabel("Attack success rate"); ax.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(figures / "attack_success_rate_comparison.png", dpi=180); plt.close()
    if not reliability.empty:
        pivot = reliability.pivot(index="round", columns="client", values="reliability_score")
        ax = pivot.plot(marker="o"); ax.set_ylabel("Reliability score")
        plt.tight_layout(); plt.savefig(figures / "reliability_scores.png", dpi=180); plt.close()


def plot_defense_comparison(csv_path, output_dir):
    table = pd.read_csv(csv_path).dropna(subset=["clean_accuracy"])
    if table.empty:
        return
    ax = table.plot.bar(x="defense", y=["clean_accuracy", "attack_success_rate"])
    plt.tight_layout(); plt.savefig(Path(output_dir) / "figures" / "defense_comparison.png", dpi=180); plt.close()

