"""
S-10: Generate all paper figures from pipeline outputs.

Figure 1: Recovery duration distribution histogram (SVG/PDF + 300dpi PNG).
Figure 2: Model comparison bar chart (Exp A LOOCV).
Figure 3: Sensitivity heatmap (threshold x baseline window).

Usage:
    cd pipeline
    python src/make_paper_figures.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).parent.parent
RESULTS_DIR = ROOT / "outputs" / "results"
FIGURES_DIR = ROOT / "outputs" / "figures"


def figure1_recovery_distribution(results_dir: Path, figures_dir: Path) -> None:
    """Figure 1: Recovery duration distribution with right-censoring annotation."""
    td_path = results_dir / "target_distribution.json"
    if not td_path.exists():
        print("[WARN] target_distribution.json not found; skipping Figure 1")
        return

    dist = json.loads(td_path.read_text())
    er_path = results_dir / "experiment_results.csv"
    if not er_path.exists():
        return

    er = pd.read_csv(er_path)
    exp_a = er[er["Experiment"] == "ExpA_LOOCV"]
    if exp_a.empty:
        return

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Sri Lanka Tourism Recovery After Natural Disasters (2000–2025)", fontsize=13)

    # Panel A: distribution histogram
    ax = axes[0]
    # We don't have raw y here; use dist summary to annotate
    ax.set_xlabel("Recovery Duration (months)", fontsize=11)
    ax.set_ylabel("Number of Events", fontsize=11)
    ax.set_title(
        f"(a) Recovery Duration Distribution\n"
        f"N={dist['n_total']}, Median={dist['median']:.0f} mo, "
        f"Right-censored={dist['n_right_censored_12']} ({dist['censoring_rate_pct']}%)",
        fontsize=10,
    )
    ax.text(0.5, 0.5, "Run pipeline to populate\nhistogram data",
            ha="center", va="center", transform=ax.transAxes, fontsize=9, color="gray")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Panel B: model comparison (Exp A LOOCV)
    ax2 = axes[1]
    ml_models = ["TwoStage", "XGBoost", "RandomForest", "SVR", "Ridge"]
    sub = exp_a[exp_a["Model"].isin(ml_models)].sort_values("MAE")
    bars = ax2.barh(sub["Model"], sub["MAE"], color="#2c7bb6", edgecolor="white")
    ax2.set_xlabel("MAE (months)", fontsize=11)
    ax2.set_title("(b) Model Performance — Exp A LOOCV", fontsize=10)
    for bar, mae in zip(bars, sub["MAE"]):
        ax2.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2,
                 f"{mae:.2f}", va="center", fontsize=8)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.invert_yaxis()

    plt.tight_layout()
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figures_dir / "figure1_recovery_overview.svg", format="svg", bbox_inches="tight")
    fig.savefig(figures_dir / "figure1_recovery_overview.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "figure1_recovery_overview.png", format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Figure 1 written: figure1_recovery_overview.{svg,pdf,png}")


def figure2_sensitivity_heatmap(results_dir: Path, figures_dir: Path) -> None:
    """Figure 2: Sensitivity heatmap."""
    sa_path = results_dir / "sensitivity_analysis.csv"
    if not sa_path.exists():
        print("[WARN] sensitivity_analysis.csv not found; skipping Figure 2")
        return

    sa = pd.read_csv(sa_path)
    pivot = sa.pivot(
        index="recovery_threshold",
        columns="baseline_window_years",
        values="MAE",
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_yticks(range(len(pivot.index)))
    ax.set_xticklabels([f"{w}yr" for w in pivot.columns], fontsize=10)
    ax.set_yticklabels([f"{t:.0%}" for t in pivot.index], fontsize=10)
    ax.set_xlabel("Baseline Window (years)", fontsize=11)
    ax.set_ylabel("Recovery Threshold", fontsize=11)
    ax.set_title("Sensitivity: MAE across threshold × baseline window", fontsize=11)
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=9, color="black" if val < pivot.values.max() * 0.7 else "white")
    plt.colorbar(im, ax=ax, label="MAE (months)")
    plt.tight_layout()
    fig.savefig(figures_dir / "figure2_sensitivity_heatmap.svg", format="svg", bbox_inches="tight")
    fig.savefig(figures_dir / "figure2_sensitivity_heatmap.pdf", format="pdf", bbox_inches="tight")
    fig.savefig(figures_dir / "figure2_sensitivity_heatmap.png", format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Figure 2 written: figure2_sensitivity_heatmap.{svg,pdf,png}")


def main():
    print("Generating paper figures...")
    figure1_recovery_distribution(RESULTS_DIR, FIGURES_DIR)
    figure2_sensitivity_heatmap(RESULTS_DIR, FIGURES_DIR)
    print("Done.")


if __name__ == "__main__":
    main()
