"""
Master execution script for the Tourism Recovery Duration pipeline.

Experiment A: Pure natural hazards (crisis years 2019-2022 excluded), LOOCV.
Experiment B: All events including compound shocks, chronological split
              (train <= 2018, test >= 2019).

Usage:
    cd pipeline
    python run_experiments.py

Outputs written to outputs/results/ and outputs/figures/.
"""

import sys
import json
import subprocess
import traceback
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PINNED = {
    "xgboost": ("2.1.1", "!="),   # != means warn if differs
    "numpy": ("2.0.2", "!="),
    "shap": ("0.46.0", "!="),
}


def _assert_versions():
    import importlib.metadata
    mismatches = []
    for pkg, (expected, op) in PINNED.items():
        try:
            actual = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            actual = "not installed"
        if op == "!=" and actual != expected:
            mismatches.append(f"  {pkg}: expected {expected}, got {actual}")
    if mismatches:
        print("[VERSION WARNING] Library versions differ from frozen run:")
        for m in mismatches:
            print(m)
        print("  Results may not reproduce exactly. See docs/VERSION_SENSITIVITY.md")


def _get_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from data_loader import load_config, load_arrivals, load_emdat, validate_data
from baseline_counterfactual import build_recovery_targets
from feature_engineering import FEATURE_COLS
from preprocess import FeatureBuilder
from validation import (
    chronological_split, buffered_loocv_split,
    leave_one_year_out_split, expanding_window_split,
)
from train import (
    run_loocv, run_chronological_split, run_multisplit,
    naive_baselines_loocv, naive_baselines_split,
)
from evaluate import (
    summarize_experiment, conformal_pi_from_loocv, wilcoxon_comparison,
    target_distribution_summary, holm_correction,
)
from explainability import run_shap_analysis
from confounding import flag_confounded_events, run_confounding_variants
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


def _save_predictions(results: dict, df_events: pd.DataFrame, experiment_name: str, results_dir: Path):
    """R-16: save per-event predictions for all models."""
    rows = []
    for model_name, res in results.items():
        for idx, (yt, yp) in enumerate(zip(res["y_true"], res["y_pred"])):
            event_id = str(df_events.iloc[idx]["disaster_id"]) if idx < len(df_events) else str(idx)
            onset = str(df_events.iloc[idx]["start_date"]) if idx < len(df_events) else ""
            rows.append({
                "experiment": experiment_name,
                "model": model_name,
                "event_id": event_id,
                "onset": onset,
                "y_true": yt,
                "y_pred": yp,
            })
    pd.DataFrame(rows).to_csv(results_dir / f"predictions_{experiment_name}.csv", index=False)


def main():
    _p("=" * 70)
    _p("Tourism Recovery Duration Prediction Pipeline")
    _p("Fold-safe features | Chronological split | Log1p DSI |")
    _p("Conformal PI | Nested CV | 8 features | No data leakage")
    _p("=" * 70)

    _assert_versions()
    run_start = datetime.utcnow()
    cfg = load_config(CFG_PATH)
    figures_dir = ROOT / cfg["paths"]["figures_dir"]
    results_dir = ROOT / cfg["paths"]["results_dir"]
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ───────────────────────────────────────────────────────
    _p("\n[1/6] Loading data...")
    arrivals = load_arrivals(ROOT / cfg["paths"]["arrivals_raw"])
    emdat = load_emdat(ROOT / cfg["paths"]["emdat_raw"], cfg)
    validate_data(emdat, arrivals)
    _p(f"      {len(arrivals)} monthly arrival records ({arrivals['year'].min()}-{arrivals['year'].max()})")
    _p(f"      {len(emdat)} physical disaster events (after taxonomy filter + clustering)")

    # ── 2. Recovery targets ────────────────────────────────────────────────
    _p("\n[2/6] Computing seasonally adjusted recovery targets...")
    df_all = build_recovery_targets(emdat, arrivals, cfg)

    # R-03: drop events with insufficient baseline history
    n_insufficient = df_all["baseline_insufficient"].sum()
    if n_insufficient:
        _p(f"      Dropping {n_insufficient} events with insufficient baseline history (R-03)")
        dropped = df_all[df_all["baseline_insufficient"]][["disaster_id", "start_year", "start_month"]]
        _p(f"      Dropped: {dropped.to_dict('records')}")

    df_with_targets = (
        df_all[~df_all["baseline_insufficient"]]
        .dropna(subset=["recovery_months"])
        .sort_values("start_date")
        .reset_index(drop=True)
    )

    dist = target_distribution_summary(df_with_targets["recovery_months"].values)
    _p(f"      N events (valid target): {dist['n_total']}")
    _p(f"      Median recovery: {dist['median']} months | Mean: {dist['mean']} months")
    _p(f"      Right-censored at 12 months: {dist['n_right_censored_12']} ({dist['censoring_rate_pct']}%)")
    _p(f"      Zero-month recoveries: {dist['n_zero_months']}")
    with open(results_dir / "target_distribution.json", "w") as f:
        json.dump(dist, f, indent=2)

    # ── 3. Feature matrix (reference only; LOOCV uses fold-safe FeatureBuilder) ──
    _p("\n[3/6] Reference feature matrix (fold-safe inside loops)...")
    # Build once to get reference X for SHAP and supplementary tables
    fb_ref = FeatureBuilder(cfg)
    fb_ref.fit(df_with_targets)
    df_feat_ref = fb_ref.transform(df_with_targets)
    df_feat_ref.to_csv(results_dir / "reference_fullsample.csv", index=False)
    X_ref = df_feat_ref[FEATURE_COLS].values
    y_ref = df_with_targets["recovery_months"].values
    _p(f"      Reference feature matrix shape: {X_ref.shape}")
    _p(f"      Features: {', '.join(FEATURE_COLS)}")

    shock_cfg = cfg["external_shocks"]
    crisis_years = set()
    for rng in shock_cfg.values():
        crisis_years.update(range(rng[0], rng[1] + 1))

    mask_clean = ~df_with_targets["start_year"].isin(crisis_years)
    df_A = df_with_targets[mask_clean].reset_index(drop=True)
    df_B = df_with_targets.reset_index(drop=True)
    y_A = df_A["recovery_months"].values
    y_B = df_B["recovery_months"].values
    X_A = df_feat_ref[mask_clean.values][FEATURE_COLS].values

    _p(f"      Experiment A (pure natural hazards): N = {len(df_A)}")
    _p(f"      Experiment B (all events incl. shocks): N = {len(df_B)}")

    # R-04: event-flow table
    event_flow = {
        "raw_emdat_records": len(emdat),
        "physical_disasters_after_cluster": len(emdat),
        "events_baseline_insufficient": int(n_insufficient),
        "events_valid_target": dist["n_total"],
        "events_right_censored": dist["n_right_censored_12"],
        "events_zero_months": dist["n_zero_months"],
        "events_in_crisis_window": int((~mask_clean).sum()),
        "exp_A_N": int(len(df_A)),
        "exp_B_N": int(len(df_B)),
    }

    # ── 4. Experiment A (LOOCV) ────────────────────────────────────────────
    _p("\n[4/6] Experiment A -- LOOCV on pure natural hazards...")
    results_A = run_loocv(df_A, cfg)   # TwoStage integrated; fold-safe FeatureBuilder
    naive_A = naive_baselines_loocv(y_A, df=df_A)
    results_A.update(naive_A)

    xgb_res_A = results_A["XGBoost"]
    pi_A = conformal_pi_from_loocv(
        xgb_res_A["y_true"], xgb_res_A["y_pred"],
        alpha=cfg["conformal_prediction"]["alpha"]
    )
    _p(f"      XGBoost conformal PI half-width (heuristic): +/-{pi_A['pi_half_width_months']} months")
    _p(f"      Empirical coverage: {pi_A['empirical_coverage'] * 100:.1f}%")

    if "TwoStage" in results_A:
        from two_stage_model import stage1_loocv_accuracy
        s1_acc = stage1_loocv_accuracy(X_A, y_A)
        _p(f"      Stage-1 accuracy: {s1_acc['stage1_accuracy']} | "
           f"F1(non-zero): {s1_acc['stage1_f1_nonzero']} | "
           f"AUC: {s1_acc['stage1_auc']} | "
           f"majority_rate: {s1_acc['stage1_majority_rate']}")

    summary_A = summarize_experiment(results_A, "ExpA_LOOCV")
    _p("\n      --- Experiment A Results ---")
    _p(summary_A[["Model", "MAE", "RMSE", "R2"]].to_string(index=False))

    # R-16: save per-event predictions
    _save_predictions(results_A, df_A, "ExpA_LOOCV", results_dir)

    # ── S-06: Additional validation schemes ───────────────────────────────
    _p("\n[VAL] Additional validation schemes (buffered LOOCV / LOYO / expanding window)...")
    extra_summaries = []
    try:
        buf_splits = buffered_loocv_split(df_A, buffer_months=12)
        _p(f"      Buffered LOOCV: {len(buf_splits)} folds")
        res_buf = run_multisplit(df_A, buf_splits, cfg)
        extra_summaries.append(summarize_experiment(res_buf, "ExpA_BufferedLOOCV"))
    except Exception as e:
        _p(f"      [WARN] Buffered LOOCV skipped: {e}")
    try:
        loyo_splits = leave_one_year_out_split(df_A)
        _p(f"      LOYO: {len(loyo_splits)} folds")
        res_loyo = run_multisplit(df_A, loyo_splits, cfg)
        extra_summaries.append(summarize_experiment(res_loyo, "ExpA_LOYO"))
    except Exception as e:
        _p(f"      [WARN] LOYO skipped: {e}")
    try:
        ew_splits = expanding_window_split(df_A)
        _p(f"      Expanding window: {len(ew_splits)} folds")
        res_ew = run_multisplit(df_A, ew_splits, cfg)
        extra_summaries.append(summarize_experiment(res_ew, "ExpA_ExpandingWindow"))
    except Exception as e:
        _p(f"      [WARN] Expanding window skipped: {e}")

    # ── 5. Experiment B (chronological split) ──────────────────────────────
    _p("\n[5/6] Experiment B -- Chronological split (train <=2018, test >=2019)...")
    X_dummy = pd.DataFrame(np.zeros((len(df_B), 1)))
    y_dummy = pd.Series(y_B)
    train_idx, test_idx = chronological_split(
        X_dummy, y_dummy, df_B,
        train_cutoff_year=cfg["experiment"]["train_cutoff_year"]
    )
    _p(f"      Train N={len(train_idx)}, Test N={len(test_idx)}")

    results_B = run_chronological_split(df_B, train_idx, test_idx, cfg)
    naive_B = naive_baselines_split(
        y_B[train_idx], y_B[test_idx],
        df_train=df_B.iloc[train_idx], df_test=df_B.iloc[test_idx],
    )
    results_B_eval = {k: {"y_true": v["y_true"], "y_pred": v["y_pred"]} for k, v in results_B.items()}
    results_B_eval.update(naive_B)

    event_flow["exp_B_train_N"] = int(len(train_idx))
    event_flow["exp_B_test_N"] = int(len(test_idx))

    # Empirical test-set coverage (not a conformal guarantee; ISS-C02)
    xgb_res_B = results_B_eval.get("XGBoost", {})
    if xgb_res_B:
        y_true_B = np.array(xgb_res_B["y_true"])
        y_pred_B = np.array(xgb_res_B["y_pred"])
        alpha_B = cfg["conformal_prediction"]["alpha"]
        residuals_B = np.abs(y_true_B - y_pred_B)
        pi_B = {
            "empirical_test_coverage": float(np.mean(residuals_B <= np.quantile(residuals_B, 1 - alpha_B))),
            "note": "Empirical test-set coverage only — not a conformal guarantee (ISS-C02).",
        }

    summary_B = summarize_experiment(results_B_eval, "ExpB_Chronological")
    _p("\n      --- Experiment B Results ---")
    _p(summary_B[["Model", "MAE", "RMSE", "R2"]].to_string(index=False))

    # R-16: save per-event predictions
    _save_predictions(results_B_eval, df_B.iloc[test_idx].reset_index(drop=True), "ExpB_Chronological", results_dir)

    # ── S-04: Survival analysis ────────────────────────────────────────────
    _p("\n[SURV] Survival analysis (CoxPH + CoxnetAFT)...")
    from survival import export_survival_results
    surv_df = export_survival_results(df_B, cfg, train_idx, test_idx, results_dir)
    _p(f"      Survival models: {surv_df[['model','c_index','integrated_brier_score']].to_string(index=False)}")

    # ── 6. Wilcoxon tests ──────────────────────────────────────────────────
    _p("\n[6/6] Wilcoxon signed-rank tests...")
    y_true_A = np.array(results_A["XGBoost"]["y_true"])
    tests = []
    opponents = ["RandomForest", "Ridge", "SVR", "TwoStage",
                 "NaiveMean", "NaiveMedian", "NaiveLastKnown", "NaivePrevious_Oracle"]
    for opponent in opponents:
        if opponent in results_A:
            w = wilcoxon_comparison(
                y_true_A,
                np.array(results_A["XGBoost"]["y_pred"]),
                np.array(results_A[opponent]["y_pred"]),
                "XGBoost", opponent,
            )
            tests.append(w)
            _p(f"      XGBoost vs {opponent}: p_raw={w['p_raw']}")
    # S-07: apply Holm correction across all tests
    p_raws = [t["p_raw"] for t in tests]
    p_holms = holm_correction(p_raws)
    for t, ph in zip(tests, p_holms):
        t["p_holm"] = round(ph, 6)
        t["significant_at_0.05"] = bool(ph < 0.05)
        sig = "YES" if t["significant_at_0.05"] else "NO"
        _p(f"      (Holm) XGBoost vs {t['model_b']}: p_raw={t['p_raw']} p_holm={t['p_holm']} {sig}")
    tests_df = pd.DataFrame(tests)

    # ── Save results ───────────────────────────────────────────────────────
    all_results = pd.concat([summary_A, summary_B] + extra_summaries, ignore_index=True)
    all_results.to_csv(results_dir / "experiment_results.csv", index=False)
    tests_df.to_csv(results_dir / "wilcoxon_tests.csv", index=False)
    if "s1_acc" in locals():
        with open(results_dir / "two_stage_stage1.json", "w") as f:
            json.dump(s1_acc, f, indent=2)

    if xgb_res_B:
        pi_summary = pd.DataFrame([
            {"Experiment": "A (LOOCV)", "type": "conformal_PI_heuristic", **pi_A},
            {"Experiment": "B (Chronological)", "type": "empirical_test_coverage_only", **pi_B},
        ])
        pi_summary.to_csv(results_dir / "conformal_prediction_intervals.csv", index=False)

    # R-04: save event-flow table
    with open(results_dir / "event_flow.json", "w") as f:
        json.dump(event_flow, f, indent=2)
    _p(f"\n      Event flow: {event_flow}")

    # ── Supplementary outputs ──────────────────────────────────────────────
    _p("\n[SUPP] Supplementary outputs...")
    X_df = pd.DataFrame(X_ref, columns=FEATURE_COLS)
    export_event_level_table(df_B, X_df, results_dir)
    export_correlation_matrix(X_df, results_dir, figures_dir)
    export_target_distribution_plot(y_B, figures_dir, dist)

    if "XGBoost" in results_A:
        xgb_a = results_A["XGBoost"]
        export_hazard_type_performance(
            df_A, np.array(xgb_a["y_true"]), np.array(xgb_a["y_pred"]),
            results_dir, "ExpA_LOOCV",
        )

    # ── S-09: Feature diagnostics for Exp B ───────────────────────────────
    _p("\n[DIAG] Feature diagnostics (Exp B fold-safe vs full-sample reference)...")
    try:
        fb_B_diag = results_B.get("XGBoost", {}).get("feature_builder", fb_ref)
        X_B_full = fb_ref.transform(df_B)[FEATURE_COLS].values
        X_B_train_diag = fb_B_diag.transform(df_B.iloc[train_idx])[FEATURE_COLS].values
        diag = {
            "reference_fullsample_shape": list(X_B_full.shape),
            "fold_train_shape": list(X_B_train_diag.shape),
            "feature_means_reference": {
                f: round(float(np.mean(X_B_full[:, i])), 4)
                for i, f in enumerate(FEATURE_COLS)
            },
            "feature_means_fold_train": {
                f: round(float(np.mean(X_B_train_diag[:, i])), 4)
                for i, f in enumerate(FEATURE_COLS)
            },
            "feature_std_reference": {
                f: round(float(np.std(X_B_full[:, i])), 4)
                for i, f in enumerate(FEATURE_COLS)
            },
            "note": "fold_train uses FeatureBuilder fit on train_idx only (fold-safe); reference uses all events",
        }
        with open(results_dir / "feature_diagnostics.json", "w") as f:
            json.dump(diag, f, indent=2)
        _p("      feature_diagnostics.json written.")
    except Exception as e:
        _p(f"      [WARN] Feature diagnostics skipped: {e}")

    shap_df = None
    xgb_model_B = results_B.get("XGBoost", {}).get("model", None)
    if xgb_model_B is not None:
        _p("\n      Running SHAP analysis on XGBoost (Experiment B)...")
        # R-01: use training-fold features for SHAP
        fb_B = results_B["XGBoost"].get("feature_builder", fb_ref)
        X_B_train = fb_B.transform(df_B.iloc[train_idx])[FEATURE_COLS].values
        shap_df = run_shap_analysis(
            xgb_model_B, X_B_train, FEATURE_COLS,
            figures_dir, results_dir,
            experiment_name="ExpB", model_name="XGBoost",
        )
        if shap_df is not None:
            _p("      SHAP feature ranking:")
            for _, row in shap_df.iterrows():
                _p(f"        {int(row['Rank']):2d}. {row['Display_Name']}: {row['Mean_Abs_SHAP']:.4f}")

    export_rq_summary(dist, summary_A, summary_B, tests_df, shap_df, results_dir)

    # ── S-02: Baseline method comparison ──────────────────────────────────
    _p("\n[BASE] Baseline method comparison (seasonal_mean / seasonal_trend / yoy_growth)...")
    try:
        from baseline_counterfactual import build_recovery_targets as _brt
        base_rows = []
        for bmethod in ["seasonal_mean", "seasonal_trend", "yoy_growth"]:
            cfg_b = {**cfg, "baseline": {**cfg.get("baseline", {}), "method": bmethod}}
            df_bm = _brt(emdat, arrivals, cfg_b)
            df_bm = df_bm[~df_bm["baseline_insufficient"]].dropna(subset=["recovery_months"])
            y_bm = df_bm["recovery_months"].values
            base_rows.append({
                "method": bmethod,
                "n_events": len(y_bm),
                "median_recovery": float(np.median(y_bm)),
                "mean_recovery": round(float(np.mean(y_bm)), 3),
                "n_right_censored": int(np.sum(y_bm == 12)),
                "n_zero": int(np.sum(y_bm == 0)),
            })
            _p(f"      {bmethod}: N={len(y_bm)} median={np.median(y_bm):.1f} mean={np.mean(y_bm):.2f}")
        pd.DataFrame(base_rows).to_csv(results_dir / "baseline_method_comparison.csv", index=False)
    except Exception as e:
        _p(f"      [WARN] Baseline method comparison skipped: {e}")

    # ── S-01: Confounding sensitivity ─────────────────────────────────────
    _p("\n[CONF] Confounding window sensitivity...")
    try:
        from xgboost import XGBRegressor as _XGB
        df_A_flagged = flag_confounded_events(df_A, cfg)
        conf_df = run_confounding_variants(
            df_A_flagged, X_A, y_A, cfg, _XGB, {}, results_dir, seed=SEED,
        )
        _p(f"      Confounding variants: {conf_df[['variant','n_events','MAE','R2']].to_string(index=False)}")
        # Carry confounded_by back for event table
        df_A = df_A_flagged
        df_B = flag_confounded_events(df_B, cfg)
    except Exception as e:
        _p(f"      [WARN] Confounding sensitivity skipped: {e}")

    # ── Sensitivity analysis ───────────────────────────────────────────────
    _p("\n[SENS] Sensitivity analysis (threshold x baseline window)...")
    try:
        from sensitivity_analysis import run_full_sensitivity
        from xgboost import XGBRegressor

        # R-15: use config default params, not last-fold params
        xgb_default_params = {}

        sens_cfg = cfg.get("sensitivity_analysis", {})
        sens_df = run_full_sensitivity(
            X_A, y_A, df_A, arrivals,
            XGBRegressor, xgb_default_params,
            thresholds=sens_cfg.get("thresholds", [0.80, 0.90, 1.00]),
            baseline_windows=sens_cfg.get("baseline_windows", [3, 6, 12]),
            crisis_years=crisis_years,
            results_dir=results_dir,
            seed=SEED,
            persistence_values=sens_cfg.get("persistence_values", [1]),
        )
        _p(f"      Sensitivity grid: {len(sens_df)} combinations.")
    except Exception as e:
        _p(f"      [WARN] Sensitivity analysis skipped: {e}")
        traceback.print_exc()

    # ── Run manifest ───────────────────────────────────────────────────────
    import platform, importlib.metadata
    run_end = datetime.utcnow()
    manifest = {
        "commit": _get_commit_hash(),
        "python_version": sys.version,
        "platform": platform.platform(),
        "seed": SEED,
        "config_path": str(CFG_PATH),
        "run_start_utc": run_start.isoformat(),
        "run_end_utc": run_end.isoformat(),
        "run_duration_seconds": (run_end - run_start).total_seconds(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in ["pandas", "numpy", "scikit-learn", "xgboost", "shap", "scipy"]
            if importlib.metadata.version(p)
        },
    }
    with open(results_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # ── S-10: paper_numbers.json ───────────────────────────────────────────
    try:
        from paper_numbers import extract_paper_numbers
        pn = extract_paper_numbers(results_dir)
        _p(f"\n[PAPER] paper_numbers.json written ({len(pn)} keys).")
    except Exception as e:
        _p(f"      [WARN] paper_numbers.json skipped: {e}")

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
