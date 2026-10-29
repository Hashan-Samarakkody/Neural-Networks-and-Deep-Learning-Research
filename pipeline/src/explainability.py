"""
SHAP-based explainability for tree models (XGBoost, RandomForest).

Exports:
    - Beeswarm summary plot (vector PDF + SVG, 300 DPI PNG)
    - Dependence plots for each feature
    - Feature importance ranking table (CSV)

Uses TreeSHAP for exact computation on tree-based models.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


FEATURE_DISPLAY_NAMES = {
    "DSI": "Disaster Severity Index",
    "TWDI": "Tourism-Weighted Disaster Impact",
    "Season_sin": "Seasonality (Sine)",
    "Season_cos": "Seasonality (Cosine)",
    "global_tourism_idx": "Global Tourism Index",
    "gdp_growth": "GDP Growth Rate",
    "Economic_Condition_Index": "Economic Condition Index",
    "external_shock_flag": "External Shock Flag",
}


def compute_shap_values(model, X: np.ndarray, feature_names: list) -> tuple:
    """
    Compute TreeSHAP values for XGBoost or RandomForest.

    Returns (shap_values_array, shap_explainer).
    """
    if not SHAP_AVAILABLE:
        raise ImportError("shap is not installed. Run: pip install shap")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return shap_values, explainer


def plot_beeswarm(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list,
    output_dir: str | Path,
    experiment_name: str,
    model_name: str,
) -> None:
    """
    Export SHAP beeswarm summary plot in PDF, SVG, and PNG formats.
    """
    if not SHAP_AVAILABLE:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    display_names = [FEATURE_DISPLAY_NAMES.get(f, f) for f in feature_names]

    fig, ax = plt.subplots(figsize=(10, 6))
    plt.sca(ax)
    shap.summary_plot(
        shap_values,
        X,
        feature_names=display_names,
        show=False,
        plot_type="dot",
        max_display=len(feature_names),
    )
    plt.title(
        f"SHAP Feature Importance — {model_name} ({experiment_name})",
        fontsize=13, fontweight="bold", pad=12
    )
    plt.tight_layout()

    stem = f"shap_beeswarm_{experiment_name}_{model_name}"
    fig.savefig(output_dir / f"{stem}.pdf", format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.svg", format="svg", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.png", format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_dependence(
    shap_values: np.ndarray,
    X: np.ndarray,
    feature_names: list,
    output_dir: str | Path,
    experiment_name: str,
    model_name: str,
) -> None:
    """
    Export SHAP dependence plot for each feature.
    """
    if not SHAP_AVAILABLE:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, feat in enumerate(feature_names):
        display = FEATURE_DISPLAY_NAMES.get(feat, feat)
        fig, ax = plt.subplots(figsize=(7, 5))
        plt.sca(ax)
        shap.dependence_plot(
            i, shap_values, X,
            feature_names=[FEATURE_DISPLAY_NAMES.get(f, f) for f in feature_names],
            show=False,
            ax=ax,
        )
        ax.set_title(
            f"SHAP Dependence: {display} — {model_name} ({experiment_name})",
            fontsize=11
        )
        plt.tight_layout()

        stem = f"shap_dep_{experiment_name}_{model_name}_{feat}"
        fig.savefig(output_dir / f"{stem}.pdf", format="pdf", dpi=300, bbox_inches="tight")
        fig.savefig(output_dir / f"{stem}.png", format="png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def feature_importance_table(
    shap_values: np.ndarray,
    feature_names: list,
    output_dir: str | Path,
    experiment_name: str,
    model_name: str,
) -> pd.DataFrame:
    """
    Compute mean |SHAP| per feature and export as CSV.
    """
    mean_abs = np.abs(shap_values).mean(axis=0)
    df = pd.DataFrame({
        "Feature": feature_names,
        "Display_Name": [FEATURE_DISPLAY_NAMES.get(f, f) for f in feature_names],
        "Mean_Abs_SHAP": mean_abs,
        "Rank": pd.Series(mean_abs).rank(ascending=False).astype(int).values,
    }).sort_values("Mean_Abs_SHAP", ascending=False)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"shap_importance_{experiment_name}_{model_name}.csv"
    df.to_csv(out_path, index=False)
    return df


def run_shap_analysis(
    model,
    X_train: np.ndarray,
    feature_names: list,
    figures_dir: str | Path,
    results_dir: str | Path,
    experiment_name: str,
    model_name: str = "XGBoost",
) -> pd.DataFrame | None:
    """
    Full SHAP analysis pipeline: compute, plot beeswarm, plot dependence, export table.
    Returns feature importance DataFrame.
    """
    if not SHAP_AVAILABLE:
        print("[WARN] shap not installed; skipping SHAP analysis.")
        return None

    shap_values, _ = compute_shap_values(model, X_train, feature_names)
    plot_beeswarm(shap_values, X_train, feature_names, figures_dir, experiment_name, model_name)
    plot_dependence(shap_values, X_train, feature_names, figures_dir, experiment_name, model_name)
    importance_df = feature_importance_table(
        shap_values, feature_names, results_dir, experiment_name, model_name
    )
    return importance_df
