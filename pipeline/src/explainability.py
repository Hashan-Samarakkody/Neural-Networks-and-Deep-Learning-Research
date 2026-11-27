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


# ── Permutation importance, grouped importance, VIF ──────────────────────────

# Correlated feature groups for grouped importance
_FEATURE_GROUPS = {
    "economy": ["Economic_Condition_Index", "gdp_growth", "global_tourism_idx"],
    "hazard": ["DSI", "TWDI"],
    "seasonality": ["Season_sin", "Season_cos"],
    "shock": ["external_shock_flag"],
}


def permutation_importance(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: list,
    n_repeats: int = 30,
    seed: int = 42,
) -> pd.DataFrame:
    """
    R-19: Permutation importance on held-out test set.
    Drop in MAE when feature is randomly shuffled.
    """
    from sklearn.metrics import mean_absolute_error as _mae

    rng = np.random.default_rng(seed)
    base_mae = _mae(y_test, model.predict(X_test))

    rows = []
    for i, feat in enumerate(feature_names):
        deltas = []
        for _ in range(n_repeats):
            X_perm = X_test.copy()
            X_perm[:, i] = rng.permutation(X_perm[:, i])
            perm_mae = _mae(y_test, model.predict(X_perm))
            deltas.append(perm_mae - base_mae)
        rows.append({
            "Feature": feat,
            "Display_Name": FEATURE_DISPLAY_NAMES.get(feat, feat),
            "mean_mae_increase": round(float(np.mean(deltas)), 4),
            "std_mae_increase": round(float(np.std(deltas)), 4),
        })

    df = pd.DataFrame(rows).sort_values("mean_mae_increase", ascending=False).reset_index(drop=True)
    df["Rank"] = range(1, len(df) + 1)
    return df


def grouped_importance(
    perm_df: pd.DataFrame,
    feature_groups: dict | None = None,
) -> pd.DataFrame:
    """
    R-19: Sum permutation importance within predefined correlated groups.
    """
    groups = feature_groups or _FEATURE_GROUPS
    rows = []
    for group_name, features in groups.items():
        subset = perm_df[perm_df["Feature"].isin(features)]
        rows.append({
            "group": group_name,
            "features": ", ".join(features),
            "sum_mean_mae_increase": round(float(subset["mean_mae_increase"].sum()), 4),
            "n_features": len(subset),
        })
    return pd.DataFrame(rows).sort_values("sum_mean_mae_increase", ascending=False)


def compute_vif(X: np.ndarray, feature_names: list) -> pd.DataFrame:
    """
    R-19: Variance Inflation Factor for each feature.
    VIF > 10 indicates severe multicollinearity.
    """
    from sklearn.linear_model import LinearRegression

    n_features = X.shape[1]
    vif_values = []
    for i in range(n_features):
        y_i = X[:, i]
        X_others = np.delete(X, i, axis=1)
        if X_others.shape[1] == 0 or len(y_i) <= X_others.shape[1]:
            vif_values.append(float("nan"))
            continue
        lr = LinearRegression().fit(X_others, y_i)
        ss_res = np.sum((y_i - lr.predict(X_others)) ** 2)
        ss_tot = np.sum((y_i - y_i.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        vif = 1 / (1 - r2) if r2 < 1 else float("inf")
        vif_values.append(round(float(vif), 2))

    return pd.DataFrame({
        "Feature": feature_names,
        "Display_Name": [FEATURE_DISPLAY_NAMES.get(f, f) for f in feature_names],
        "VIF": vif_values,
    }).sort_values("VIF", ascending=False)


def run_r19_analysis(
    model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_train: np.ndarray,
    feature_names: list,
    results_dir: str | Path,
    experiment_name: str,
    model_name: str,
) -> None:
    """
    R-19: Run permutation importance, grouped importance, and VIF; save CSVs.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    perm_df = permutation_importance(model, X_test, y_test, feature_names)
    perm_df.to_csv(results_dir / f"permutation_importance_{experiment_name}_{model_name}.csv", index=False)

    group_df = grouped_importance(perm_df)
    group_df.to_csv(results_dir / "grouped_importance.csv", index=False)

    vif_df = compute_vif(X_train, feature_names)
    vif_df.to_csv(results_dir / "vif.csv", index=False)


def run_ablation_no_index(
    df_A: "pd.DataFrame",
    cfg: dict,
    results_dir: "Path",
    seed: int = 42,
) -> "pd.DataFrame":
    """
    S-18: Ablation study removing DSI and TWDI (the index features).
    Runs LOOCV with XGBoost on reduced feature set.
    """
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    import xgboost as xgb
    from preprocess import FeatureBuilder
    from feature_engineering import FEATURE_COLS

    no_index_cols = [f for f in FEATURE_COLS if f not in ("DSI", "TWDI")]
    n = len(df_A)
    loo = LeaveOneOut()
    preds, trues = [], []
    for tr, te in loo.split(np.arange(n)):
        fb = FeatureBuilder(cfg)
        df_tr = df_A.iloc[tr]
        df_te = df_A.iloc[te]
        fb.fit(df_tr)
        X_tr = fb.transform(df_tr)[no_index_cols].values
        X_te = fb.transform(df_te)[no_index_cols].values
        y_tr = df_tr["recovery_months"].values
        m = xgb.XGBRegressor(random_state=seed, verbosity=0, nthread=1)
        m.fit(X_tr, y_tr)
        preds.append(float(np.clip(m.predict(X_te)[0], 0, 12)))
        trues.append(float(df_A.iloc[te[0]]["recovery_months"]))

    yt, yp = np.array(trues), np.array(preds)
    df_out = pd.DataFrame([{
        "experiment": "ExpA_LOOCV_no_index",
        "features_removed": "DSI,TWDI",
        "n_features": len(no_index_cols),
        "MAE": round(float(mean_absolute_error(yt, yp)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(yt, yp))), 4),
        "R2": round(float(r2_score(yt, yp)), 4),
    }])
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    df_out.to_csv(Path(results_dir) / "ablation_no_index.csv", index=False)
    return df_out
