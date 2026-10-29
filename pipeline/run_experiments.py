"""
Master execution script for the Tourism Recovery Duration pipeline.

Experiment A: Pure natural hazards (N=67), LOOCV validation.
Experiment B: All events including compound shocks (N=86), chronological split (train<=2018, test>=2019).

Usage:
    cd pipeline
    python run_experiments.py

Outputs written to outputs/results/ and outputs/figures/.
"""

import sys
import json
import traceback
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from data_loader import load_config, load_arrivals, load_emdat, validate_data
from baseline_counterfactual import build_recovery_targets
from feature_engineering import build_feature_matrix, FEATURE_COLS
from validation import chronological_split
from train import run_loocv, run_chronological_split, naive_baselines_loocv, naive_baselines_split
from evaluate import (
    summarize_experiment, conformal_pi_from_loocv, wilcoxon_comparison,
    target_distribution_summary, performance_by_hazard_type,
)
from explainability import run_shap_analysis
from supplementary import (
    export_event_level_table, export_correlation_matrix,
    export_target_distribution_plot, export_rq_summary,
    export_hazard_type_performance,
)


CFG_PATH = ROOT / "configs" / "model_config.yaml"
SEED = 42
np.random.seed(SEED)


def _p(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def main():
    _p("=" * 70)
    _p("Tourism Recovery Duration Prediction Pipeline")
    _p("Fixed: Seasonal baseline | Chronological split | Log1p DSI |")
    _p("       Conformal PI | Nested CV | 8 features | No data leakage")
    _p("=" * 70)

    cfg = load_config(CFG_PATH)
    figures_dir = ROOT / cfg["paths"]["figures_dir"]
    results_dir = ROOT / cfg["paths"]["results_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    _p("\n[1/6] Loading data...")
    arrivals = load_arrivals(ROOT / cfg["paths"]["arrivals_raw"])
    emdat = load_emdat(ROOT / cfg["paths"]["emdat_raw"], cfg)
    validate_data(emdat, arrivals)
    _p(f"      Loaded {len(arrivals)} monthly arrival records ({arrivals['year'].min()}-{arrivals['year'].max()})")
    _p(f"      Loaded {len(emdat)} physical disaster events (after taxonomy filter + clustering)")

    _p("\n[2/6] Computing seasonally adjusted recovery targets...")
    df_with_targets = build_recovery_targets(emdat, arrivals, cfg)
    df_with_targets = df_with_targets.dropna(subset=["recovery_months"])

    dist = target_distribution_summary(df_with_targets["recovery_months"].values)
    _p(f"      N events: {dist['n_total']}")
    _p(f"      Median recovery: {dist['median']} months | Mean: {dist['mean']} months")
    _p(f"      Right-censored at 12 months: {dist['n_right_censored_12']} ({dist['censoring_rate_pct']}%)")
    _p(f"      Zero-month recoveries: {dist['n_zero_months']}")
    with open(results_dir / "target_distribution.json", "w") as f:
        json.dump(dist, f, indent=2)

    _p("\n[3/6] Engineering features (8-feature matrix)...")
    X, y = build_feature_matrix(df_with_targets, cfg)
    X_arr = X.values
    y_arr = y.values
    _p(f"      Feature matrix shape: {X.shape}")
    _p(f"      Features: {', '.join(FEATURE_COLS)}")

    shock_cfg = cfg["external_shocks"]
    crisis_years = set()
    for rng in shock_cfg.values():
        crisis_years.update(range(rng[0], rng[1] + 1))

    mask_clean = ~df_with_targets["start_year"].isin(crisis_years)
    X_A = X_arr[mask_clean.values]
    y_A = y_arr[mask_clean.values]
    df_A = df_with_targets[mask_clean.values].reset_index(drop=True)

    X_B = X_arr
    y_B = y_arr
    df_B = df_with_targets.reset_index(drop=True)

    _p(f"      Experiment A (pure natural hazards): N = {len(X_A)}")
    _p(f"      Experiment B (all events incl. shocks): N = {len(X_B)}")

    _p("\n[4/6] Experiment A -- LOOCV on pure natural hazards...")
    results_A = run_loocv(X_A, y_A, cfg)
    naive_A = naive_baselines_loocv(y_A)
    results_A.update(naive_A)

    xgb_res_A = results_A["XGBoost"]
    pi_A = conformal_pi_from_loocv(
        xgb_res_A["y_true"], xgb_res_A["y_pred"],
        alpha=cfg["conformal_prediction"]["alpha"]
    )
    _p(f"      XGBoost conformal PI half-width: +/-{pi_A['pi_half_width_months']} months")
    _p(f"      Empirical coverage: {pi_A['empirical_coverage'] * 100:.1f}%")

    summary_A = summarize_experiment(results_A, "ExpA_LOOCV")
    _p("\n      --- Experiment A Results ---")
    _p(summary_A[["Model", "MAE", "RMSE", "R2", "PI_half_width", "PI_empirical_coverage"]].to_string(index=False))

    _p("\n[5/6] Experiment B -- Chronological split (train <=2018, test >=2019)...")
    train_idx, test_idx = chronological_split(
        pd.DataFrame(X_B), pd.Series(y_B), df_B,
        train_cutoff_year=cfg["experiment"]["train_cutoff_year"]
    )
    _p(f"      Train N={len(train_idx)}, Test N={len(test_idx)}")

    results_B = run_chronological_split(X_B, y_B, train_idx, test_idx, cfg)
    naive_B = naive_baselines_split(y_B[train_idx], y_B[test_idx])
    results_B_eval = {k: {"y_true": v["y_true"], "y_pred": v["y_pred"]} for k, v in results_B.items()}
    results_B_eval.update(naive_B)

    xgb_res_B = results_B_eval["XGBoost"]
    pi_B = conformal_pi_from_loocv(
        xgb_res_B["y_true"], xgb_res_B["y_pred"],
        alpha=cfg["conformal_prediction"]["alpha"]
    )

    summary_B = summarize_experiment(results_B_eval, "ExpB_Chronological")
    _p("\n      --- Experiment B Results ---")
    _p(summary_B[["Model", "MAE", "RMSE", "R2", "PI_half_width", "PI_empirical_coverage"]].to_string(index=False))

    _p("\n[6/6] Wilcoxon signed-rank tests (XGBoost vs others)...")
    y_true_A = np.array(results_A["XGBoost"]["y_true"])
    tests = []
    for opponent in ["RandomForest", "Ridge", "SVR", "NaiveMean", "NaiveMedian"]:
        if opponent in results_A:
            w = wilcoxon_comparison(
                y_true_A,
                np.array(results_A["XGBoost"]["y_pred"]),
                np.array(results_A[opponent]["y_pred"]),
                "XGBoost", opponent,
            )
            tests.append(w)
            sig = "YES" if w["significant_at_0.05"] else "NO"
            _p(f"      XGBoost vs {opponent}: p={w['p_value']} {sig}")
    tests_df = pd.DataFrame(tests)

    all_results = pd.concat([summary_A, summary_B], ignore_index=True)
    all_results.to_csv(results_dir / "experiment_results.csv", index=False)
    tests_df.to_csv(results_dir / "wilcoxon_tests.csv", index=False)

    pi_summary = pd.DataFrame([
        {"Experiment": "A (LOOCV)", **pi_A},
        {"Experiment": "B (Chronological)", **pi_B},
    ])
    pi_summary.to_csv(results_dir / "conformal_prediction_intervals.csv", index=False)

    # --- Supplementary outputs ---
    _p("\n[SUPP] Supplementary outputs for Q1/Q2 compliance...")

    X_df = pd.DataFrame(X_B, columns=FEATURE_COLS)
    export_event_level_table(df_B, X_df, results_dir)
    export_correlation_matrix(X_df, results_dir, figures_dir)
    export_target_distribution_plot(y_B, figures_dir, dist)

    if "XGBoost" in results_A:
        xgb_a = results_A["XGBoost"]
        export_hazard_type_performance(
            df_A, np.array(xgb_a["y_true"]), np.array(xgb_a["y_pred"]),
            results_dir, "ExpA_LOOCV",
        )

    shap_df = None
    xgb_model_B = results_B.get("XGBoost", {}).get("model", None)
    if xgb_model_B is not None:
        _p("\n      Running SHAP analysis on XGBoost (Experiment B)...")
        shap_df = run_shap_analysis(
            xgb_model_B, X_B[train_idx], FEATURE_COLS,
            figures_dir, results_dir,
            experiment_name="ExpB", model_name="XGBoost",
        )
        if shap_df is not None:
            _p("      SHAP feature ranking:")
            for _, row in shap_df.iterrows():
                _p(f"        {int(row['Rank']):2d}. {row['Display_Name']}: {row['Mean_Abs_SHAP']:.4f}")

    export_rq_summary(dist, summary_A, summary_B, tests_df, shap_df, results_dir)

    # --- Sensitivity analysis ---
    _p("\n[SENS] Sensitivity analysis (threshold x baseline window)...")
    try:
        from sensitivity_analysis import run_full_sensitivity
        from xgboost import XGBRegressor

        shock_cfg2 = cfg["external_shocks"]
        crisis_years_set = set()
        for rng in shock_cfg2.values():
            crisis_years_set.update(range(rng[0], rng[1] + 1))

        xgb_best_params = {}
        if "XGBoost" in results_A:
            params_list = results_A["XGBoost"].get("best_params", [])
            if params_list:
                xgb_best_params = params_list[-1]
            # Strip non-constructor params that RandomizedSearchCV may inject
            xgb_best_params = {
                k: v for k, v in xgb_best_params.items()
                if k not in ("random_state", "verbosity", "tree_method", "objective", "nthread")
            }

        sens_cfg = cfg.get("sensitivity_analysis", {})
        sens_df = run_full_sensitivity(
            X_A, y_A, df_A, arrivals,
            XGBRegressor, xgb_best_params,
            thresholds=sens_cfg.get("thresholds", [0.80, 0.90, 1.00]),
            baseline_windows=sens_cfg.get("baseline_windows", [3, 6, 12]),
            crisis_years=crisis_years_set,
            results_dir=results_dir,
            seed=SEED,
        )
        _p(f"      Sensitivity grid: {len(sens_df)} combinations. Main setting highlighted.")
    except Exception as e:
        _p(f"      [WARN] Sensitivity analysis skipped: {e}")
        traceback.print_exc()

    _p("\n" + "=" * 70)
    _p("Pipeline complete. Outputs written to:")
    _p(f"  Results: {results_dir.resolve()}")
    _p(f"  Figures: {figures_dir.resolve()}")
    _p("=" * 70)
    _p("\nNOTE: All reported metrics come from this deterministic run (seed=42).")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(1)
