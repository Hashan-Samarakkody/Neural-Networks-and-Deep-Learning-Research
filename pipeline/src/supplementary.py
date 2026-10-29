"""
Supplementary outputs for Q1/Q2 journal compliance:

1. Event-level table — one row per event: date, hazard, location, DSI, TWDI,
   pre-disaster baseline, recovery months, is_right_censored, external_shock_flag,
   all 8 engineered features.

2. Feature correlation matrix — Pearson r table showing ECI-GlobalTourism
   collinearity (addressed by Ridge regularization).

3. Hazard-type performance breakdown — MAE/RMSE/R2 split by Disaster Type
   for the best model under LOOCV.

4. RQ1-RQ4 summary — explicit mapping from research questions to results.

5. Target distribution histogram — PNG/SVG plot.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path


FEATURE_COLS = [
    "DSI", "TWDI", "Season_sin", "Season_cos",
    "global_tourism_idx", "gdp_growth", "Economic_Condition_Index", "external_shock_flag",
]


def export_event_level_table(
    df_events: pd.DataFrame,
    X: pd.DataFrame,
    results_dir: Path,
) -> None:
    """
    Export one row per event with all features + target + metadata.
    Supplementary Table S1 for reviewers.
    """
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    cols_meta = [
        c for c in [
            "disaster_id", "start_year", "start_month", "disaster_type", "disaster_subtype",
            "location", "deaths", "total_affected", "damage_usd_000", "duration_days",
            "n_events_clustered", "pre_disaster_baseline", "recovery_months", "is_right_censored",
        ] if c in df_events.columns
    ]

    meta = df_events[cols_meta].copy()
    feature_cols_present = [c for c in FEATURE_COLS if c in X.columns]
    features = X[feature_cols_present].copy()

    out = pd.concat([meta.reset_index(drop=True), features.reset_index(drop=True)], axis=1)
    out.to_csv(results_dir / "supplementary_event_table.csv", index=False)
    print(f"      Event-level table: {len(out)} rows → supplementary_event_table.csv")


def export_correlation_matrix(
    X: pd.DataFrame,
    results_dir: Path,
    figures_dir: Path,
) -> pd.DataFrame:
    """
    Pearson correlation matrix for all 8 features.
    Exports CSV + heatmap PNG to document collinearity addressed by Ridge.
    """
    results_dir = Path(results_dir)
    figures_dir = Path(figures_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    feature_cols_present = [c for c in FEATURE_COLS if c in X.columns]
    corr = X[feature_cols_present].corr(method="pearson")
    corr.to_csv(results_dir / "feature_correlation_matrix.csv")

    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(feature_cols_present)))
    ax.set_yticks(range(len(feature_cols_present)))
    display = {
        "DSI": "DSI", "TWDI": "TWDI", "Season_sin": "Sin", "Season_cos": "Cos",
        "global_tourism_idx": "GlobalTour", "gdp_growth": "GDP",
        "Economic_Condition_Index": "ECI", "external_shock_flag": "Shock",
    }
    labels = [display.get(c, c) for c in feature_cols_present]
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(feature_cols_present)):
        for j in range(len(feature_cols_present)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(corr.values[i, j]) > 0.6 else "black")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Feature Pearson Correlation Matrix\n(ECI-GlobalTourism collinearity addressed by Ridge regularisation)",
                 fontsize=11, pad=10)
    plt.tight_layout()
    fig.savefig(figures_dir / "feature_correlation_matrix.pdf", format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / "feature_correlation_matrix.png", format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"      Correlation matrix → feature_correlation_matrix.csv + .png")
    return corr


def export_hazard_type_performance(
    df_events: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    results_dir: Path,
    experiment_name: str = "ExpA_LOOCV",
) -> pd.DataFrame:
    """
    Break MAE/RMSE/R2 down by Disaster Type (P8/P10 requirement).
    df_events must be row-aligned with y_true/y_pred.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    dtype_col = next((c for c in ["disaster_type", "Disaster Type"] if c in df_events.columns), None)
    if dtype_col is None:
        print("      [WARN] No disaster_type column — skipping hazard breakdown.")
        return pd.DataFrame()

    df_eval = df_events[[dtype_col]].rename(columns={dtype_col: "Disaster Type"}).copy().reset_index(drop=True)
    df_eval["y_true"] = np.array(y_true)
    df_eval["y_pred"] = np.array(y_pred)
    df_eval["abs_error"] = np.abs(df_eval["y_true"] - df_eval["y_pred"])

    rows = []
    for htype, grp in df_eval.groupby("Disaster Type"):
        n = len(grp)
        if n < 2:
            continue
        mae = float(mean_absolute_error(grp["y_true"], grp["y_pred"]))
        rmse = float(np.sqrt(mean_squared_error(grp["y_true"], grp["y_pred"])))
        try:
            r2 = float(r2_score(grp["y_true"], grp["y_pred"]))
        except Exception:
            r2 = float("nan")
        rows.append({
            "Hazard_Type": htype,
            "N": n,
            "MAE": round(mae, 4),
            "RMSE": round(rmse, 4),
            "R2": round(r2, 4),
        })

    df_out = pd.DataFrame(rows).sort_values("MAE")
    df_out.to_csv(results_dir / f"hazard_type_performance_{experiment_name}.csv", index=False)
    print("      Hazard-type performance breakdown:")
    for _, row in df_out.iterrows():
        print(f"        {row['Hazard_Type']}: N={row['N']} MAE={row['MAE']} R2={row['R2']}")
    return df_out


def export_target_distribution_plot(
    y: np.ndarray,
    figures_dir: Path,
    dist_summary: dict,
) -> None:
    """
    Histogram of recovery duration distribution with right-censoring annotation.
    """
    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    bins = range(0, 14)
    counts, edges, patches = ax.hist(
        y, bins=bins, edgecolor="white", color="#2c7bb6", align="left", rwidth=0.8
    )
    # Highlight right-censored bar (month=12) in different color
    for i, (left, right) in enumerate(zip(edges[:-1], edges[1:])):
        if int(left) == 12:
            patches[i].set_facecolor("#d7191c")

    ax.set_xlabel("Recovery Duration (months)", fontsize=12)
    ax.set_ylabel("Number of Events", fontsize=12)
    ax.set_title(
        "Distribution of Tourism Recovery Duration\n"
        f"N={dist_summary['n_total']}, Median={dist_summary['median']} months, "
        f"Right-censored (12 mo) = {dist_summary['n_right_censored_12']} "
        f"({dist_summary['censoring_rate_pct']}%)",
        fontsize=11,
    )
    ax.set_xticks(range(0, 13))
    ax.text(11.5, max(counts) * 0.9, "Right-censored\n(recovery not\nachieved ≤12 mo)",
            ha="center", va="top", fontsize=8, color="#d7191c",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#d7191c"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(figures_dir / "target_distribution.pdf", format="pdf", dpi=300, bbox_inches="tight")
    fig.savefig(figures_dir / "target_distribution.png", format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("      Target distribution plot exported.")


def export_rq_summary(
    dist_summary: dict,
    summary_A: pd.DataFrame,
    summary_B: pd.DataFrame,
    wilcoxon_tests: pd.DataFrame,
    shap_importance: pd.DataFrame | None,
    results_dir: Path,
) -> None:
    """
    Write a structured RQ1-RQ4 summary JSON that maps each research question
    to the quantitative answer from the pipeline.
    """
    import json
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    def best_row(df, primary_sort="MAE"):
        # Exclude naive baselines from best model selection
        ml_models = ["XGBoost", "RandomForest", "SVR", "Ridge"]
        sub = df[df["Model"].isin(ml_models)]
        if sub.empty:
            return {}
        row = sub.sort_values(primary_sort).iloc[0]
        return row.to_dict()

    best_a = best_row(summary_A)
    best_b = best_row(summary_B)

    xgb_vs_naive = {}
    if not wilcoxon_tests.empty:
        for _, r in wilcoxon_tests.iterrows():
            if "Naive" in str(r.get("model_b", "")):
                xgb_vs_naive[r.get("model_b", "")] = {
                    "p_value": r.get("p_value"),
                    "significant": r.get("significant_at_0.05"),
                }

    top_features = []
    if shap_importance is not None and not shap_importance.empty:
        top_features = shap_importance.head(3)[["Feature", "Mean_Abs_SHAP"]].to_dict(orient="records")

    rq_summary = {
        "RQ1": {
            "question": "What is the distribution of tourism recovery duration after natural disasters in Sri Lanka (2000-2025)?",
            "answer": {
                "n_events": dist_summary.get("n_total"),
                "median_recovery_months": dist_summary.get("median"),
                "mean_recovery_months": dist_summary.get("mean"),
                "right_censored_n": dist_summary.get("n_right_censored_12"),
                "censoring_rate_pct": dist_summary.get("censoring_rate_pct"),
                "note": "Right-censoring at 12 months; events where recovery not achieved within observation window treated as censored, not as 12-month recoveries.",
            },
        },
        "RQ2": {
            "question": "Which disaster and economic features most strongly predict recovery duration?",
            "answer": {
                "method": "TreeSHAP (Experiment B, XGBoost)",
                "top_3_features": top_features,
                "note": "DSI and TWDI capture hazard severity; external_shock_flag captures systematic macro-collapses (COVID, Easter attacks, 2022 crisis) outside physical disaster mechanism.",
            },
        },
        "RQ3": {
            "question": "Can ML models predict recovery duration better than naive historical baselines?",
            "answer": {
                "experiment_A_best_model": best_a.get("Model"),
                "experiment_A_best_MAE": best_a.get("MAE"),
                "experiment_B_best_model": best_b.get("Model"),
                "experiment_B_best_MAE": best_b.get("MAE"),
                "wilcoxon_vs_naive": xgb_vs_naive,
                "note": "Significance tested with Wilcoxon signed-rank (non-parametric). 95% bootstrap CIs reported for all point estimates.",
            },
        },
        "RQ4": {
            "question": "How robust are predictions across varying recovery thresholds and baseline windows?",
            "answer": {
                "thresholds_tested": [0.80, 0.90, 1.00],
                "baseline_windows_tested": [3, 6, 12],
                "main_threshold": 0.90,
                "main_baseline_window": 3,
                "note": "See sensitivity_analysis.csv for full grid results. Main threshold (90%) chosen per UNWTO recovery benchmark.",
            },
        },
    }

    with open(results_dir / "rq_summary.json", "w") as f:
        json.dump(rq_summary, f, indent=2)
    print("      RQ1-RQ4 summary → rq_summary.json")
