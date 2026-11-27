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
    "pandas": "2.2.2", "numpy": "2.0.2", "scikit-learn": "1.5.2", "xgboost": "2.1.1",
    "shap": "0.46.0", "matplotlib": "3.9.2", "scipy": "1.14.1", "scikit-survival": "0.23.1",
}


def _assert_versions():
    """Abort unless every library has the pinned version (set ALLOW_VERSION_MISMATCH=1 to override)."""
    import importlib.metadata, os
    mismatches = []
    for pkg, expected in PINNED.items():
        try:
            actual = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            actual = "not installed"
        if actual != expected:
            mismatches.append(f"  {pkg}: pinned {expected}, installed {actual}")
    if mismatches:
        msg = "Library versions differ from requirements.txt:\n" + "\n".join(mismatches)
        if os.environ.get("ALLOW_VERSION_MISMATCH") == "1":
            print("[VERSION WARNING] " + msg + "\n  Override active: results are NOT the frozen results. See docs/VERSION_SENSITIVITY.md")
            return False
        print("[VERSION ERROR] " + msg + "\n  Install with: pip install -r requirements.txt")
        sys.exit(2)
    return True


def _cached(name, fn):
    """Checkpoint expensive stages (outputs/cache, git-ignored) so an interrupted run can resume."""
    import pickle
    cdir = ROOT / "outputs" / "cache"; cdir.mkdir(parents=True, exist_ok=True)
    f = cdir / f"{name}.pkl"
    if f.exists():
        return pickle.loads(f.read_bytes())
    out = fn()
    f.write_bytes(pickle.dumps(out))
    return out


def _get_commit_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "src"))

from data_loader import load_config, load_arrivals, load_emdat, validate_data, _EMDAT_FLOW
from baseline_counterfactual import build_recovery_targets
from feature_engineering import FEATURE_COLS
from preprocess import FeatureBuilder
from validation import (
    chronological_split, buffered_loocv_split,
    leave_one_year_out_split, expanding_window_split,
)
from train import (
    run_loocv, run_chronological_split, run_multisplit,
    naive_baselines_loocv, naive_baselines_split, naive_baselines_multisplit,
)
from evaluate import (
    summarize_experiment, conformal_pi_from_loocv, wilcoxon_comparison,
    target_distribution_summary, holm_correction,
)
from explainability import run_shap_analysis
from confounding import flag_confounded_events, run_confounding_variants
from paper_extras import compute_paper_extras
from make_paper_figures import make_figure1
from default_models import loocv_default_xgb, loo_zero_baseline, permutation_importance_loocv, run_ablation
from supplementary import (
    export_event_level_table, export_correlation_matrix,
    export_target_distribution_plot, export_rq_summary,
    export_hazard_type_performance, export_dtii_report,
)


CFG_PATH = ROOT / "configs" / "model_config.yaml"
SEED = 42
np.random.seed(SEED)


def _p(*args, **kwargs):
    print(*args, **kwargs)
    sys.stdout.flush()


def _get_regime(year: int, confounded_by: str = "") -> str:
    """Assign regime label to an event."""
    if confounded_by and "civil_war_end" in str(confounded_by):
        return "confounded_civil_war"
    if 2019 <= year <= 2022:
        return "easter_covid_crisis"
    if year >= 2023:
        return "post_crisis_recovery"
    return "baseline"


def _save_predictions(results: dict, df_events: pd.DataFrame, experiment_name: str, results_dir: Path):
    """Save per-event predictions with regime column for all models."""
    rows = []
    for model_name, res in results.items():
        for idx, (yt, yp) in enumerate(zip(res["y_true"], res["y_pred"])):
            if idx < len(df_events):
                ev = df_events.iloc[idx]
                event_id = str(ev.get("disaster_id", idx))
                onset = str(ev.get("start_date", ""))
                year = int(ev.get("start_year", 0))
                confounded_by = str(ev.get("confounded_by", ""))
                regime = _get_regime(year, confounded_by)
            else:
                event_id = str(idx)
                onset = ""
                regime = "unknown"
            rows.append({
                "experiment": experiment_name,
                "model": model_name,
                "event_id": event_id,
                "onset": onset,
                "regime": regime,
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

    versions_pinned = _assert_versions()
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

    n_insufficient = df_all["baseline_insufficient"].sum()
    if n_insufficient:
        _p(f"      Dropping {n_insufficient} events with insufficient baseline history")
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
    df_A = flag_confounded_events(df_A, cfg)
    df_B = flag_confounded_events(df_B, cfg)
    y_A = df_A["recovery_months"].values
    y_B = df_B["recovery_months"].values
    X_A = df_feat_ref[mask_clean.values][FEATURE_COLS].values

    _p(f"      Experiment A (pure natural hazards): N = {len(df_A)}")
    _p(f"      Experiment B (all events incl. shocks): N = {len(df_B)}")

    event_flow = {
        "raw_emdat_records": _EMDAT_FLOW.get("raw_emdat_records", len(emdat)),
        "physical_disasters_before_cluster": _EMDAT_FLOW.get("physical_disasters_before_cluster", len(emdat)),
        "physical_disasters_after_cluster": _EMDAT_FLOW.get("physical_disasters_after_cluster", len(emdat)),
        "events_baseline_insufficient": int(n_insufficient),
        "events_valid_target": dist["n_total"],
        "events_right_censored": dist["n_right_censored_12"],
        "events_zero_months": dist["n_zero_months"],
        "events_in_crisis_window": int((~mask_clean).sum()),
        "exp_A_N": int(len(df_A)),
        "exp_B_N": int(len(df_B)),
    }
    _n_clustered = event_flow["physical_disasters_after_cluster"]
    _n_valid = event_flow["events_valid_target"]
    _n_insuff = event_flow["events_baseline_insufficient"]
    assert _n_clustered == _n_valid + _n_insuff, (
        f"Event flow arithmetic error: {_n_clustered} clustered != "
        f"{_n_valid} valid + {_n_insuff} insufficient"
    )

    # ── 4. Experiment A (LOOCV) ────────────────────────────────────────────
    _p("\n[4/6] Experiment A -- LOOCV on pure natural hazards...")
    results_A = _cached('results_A', lambda: run_loocv(df_A, cfg))   # TwoStage integrated; fold-safe FeatureBuilder
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

    _save_predictions(results_A, df_A, "ExpA_LOOCV", results_dir)

    # ── Additional validation schemes ─────────────────────────────────────
    _p("\n[VAL] Additional validation schemes (buffered LOOCV / LOYO / expanding window)...")
    extra_summaries = []
    try:
        buf_splits = buffered_loocv_split(df_A, buffer_months=12)
        _p(f"      Buffered LOOCV: {len(buf_splits)} folds")
        res_buf = _cached('res_buf', lambda: run_multisplit(df_A, buf_splits, cfg))
        res_buf.update(naive_baselines_multisplit(df_A, buf_splits))
        extra_summaries.append(summarize_experiment(res_buf, "ExpA_BufferedLOOCV"))
    except Exception as e:
        _p(f"      [WARN] Buffered LOOCV skipped: {e}")
    try:
        loyo_splits = leave_one_year_out_split(df_A)
        _p(f"      LOYO: {len(loyo_splits)} folds")
        res_loyo = _cached('res_loyo', lambda: run_multisplit(df_A, loyo_splits, cfg))
        res_loyo.update(naive_baselines_multisplit(df_A, loyo_splits))
        extra_summaries.append(summarize_experiment(res_loyo, "ExpA_LOYO"))
    except Exception as e:
        _p(f"      [WARN] LOYO skipped: {e}")
    try:
        ew_splits = expanding_window_split(df_A)
        _p(f"      Expanding window: {len(ew_splits)} folds")
        res_ew = _cached('res_ew', lambda: run_multisplit(df_A, ew_splits, cfg))
        res_ew.update(naive_baselines_multisplit(df_A, ew_splits))
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

    results_B = _cached('results_B', lambda: run_chronological_split(df_B, train_idx, test_idx, cfg))
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

    _save_predictions(results_B_eval, df_B.iloc[test_idx].reset_index(drop=True), "ExpB_Chronological", results_dir)

    # ── Survival analysis ──────────────────────────────────────────────────
    _p("\n[SURV] Survival analysis (CoxPH, RandomSurvivalForest, DiscreteHazard)...")
    from survival import run_survival_experiments
    surv_df = run_survival_experiments(df_A, df_B, train_idx, test_idx, cfg, results_dir)
    _p(surv_df[["experiment", "model", "c_index", "integrated_brier_score"]].to_string(index=False))
    if surv_df["c_index"].isna().all():
        raise RuntimeError("All survival models failed; see survival_results.csv notes")

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
    def _regime_counts(df_ev):
        regs = df_ev.apply(
            lambda r: _get_regime(int(r.get("start_year", 0)), str(r.get("confounded_by", ""))),
            axis=1,
        )
        vc = regs.value_counts()
        return {
            "n_baseline": int(vc.get("baseline", 0)),
            "n_easter_covid_crisis": int(vc.get("easter_covid_crisis", 0)),
            "n_post_crisis_recovery": int(vc.get("post_crisis_recovery", 0)),
            "n_confounded_civil_war": int(vc.get("confounded_civil_war", 0)),
        }
    rc_A = _regime_counts(df_A)
    rc_B = _regime_counts(df_B)
    for col, val in rc_A.items():
        all_results.loc[all_results["Experiment"].str.startswith("ExpA"), col] = val
    for col, val in rc_B.items():
        all_results.loc[all_results["Experiment"].str.startswith("ExpB"), col] = val
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

    with open(results_dir / "event_flow.json", "w") as f:
        json.dump(event_flow, f, indent=2)
    _p(f"\n      Event flow: {event_flow}")

    # ── Supplementary outputs ──────────────────────────────────────────────
    _p("\n[SUPP] Supplementary outputs...")
    X_df = pd.DataFrame(X_ref, columns=FEATURE_COLS)
    df_B_ev = df_B.copy()
    for c in ("matched_district", "dtii_rule", "DTII", "TWDI"):
        if c in df_feat_ref.columns:
            df_B_ev[c] = df_feat_ref[c].values
    export_event_level_table(df_B_ev, X_df, results_dir)
    export_dtii_report(df_B_ev, results_dir)
    export_correlation_matrix(X_df, results_dir, figures_dir)
    export_target_distribution_plot(y_B, figures_dir, dist)

    if "XGBoost" in results_A:
        xgb_a = results_A["XGBoost"]
        export_hazard_type_performance(
            df_A, np.array(xgb_a["y_true"]), np.array(xgb_a["y_pred"]),
            results_dir, "ExpA_LOOCV",
        )

    # ── Feature diagnostics for Exp B ─────────────────────────────────────
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

    # ── Permutation importance, VIF, grouped importance, ablation ─────────
    _p("\n[IMP] Permutation importance, VIF, grouped importance, ablation...")
    try:
        from explainability import run_r19_analysis
        xgb_model_A_last = None
        # Use Exp B XGBoost for permutation importance (has held-out test set)
        if xgb_model_B is not None:
            fb_B_imp = results_B.get("XGBoost", {}).get("feature_builder", fb_ref)
            X_B_te = fb_B_imp.transform(df_B.iloc[test_idx])[FEATURE_COLS].values
            y_B_te = y_B[test_idx]
            X_B_tr = fb_B_imp.transform(df_B.iloc[train_idx])[FEATURE_COLS].values
            run_r19_analysis(
                xgb_model_B, X_B_te, y_B_te, X_B_tr, FEATURE_COLS,
                results_dir, "ExpB", "XGBoost",
            )
            _p("      permutation_importance_ExpB_XGBoost.csv, grouped_importance.csv, vif.csv written.")
        abl_df = run_ablation(df_A, cfg, results_dir, seed=SEED)
        _p("      ablation_features.csv:\n" + abl_df[["variant", "MAE", "R2"]].to_string(index=False))
        perm_A = permutation_importance_loocv(df_A, cfg, seed=SEED)
        perm_A.to_csv(results_dir / "permutation_importance_ExpA_XGBoost.csv", index=False)
        _p("      permutation_importance_ExpA_XGBoost.csv written.")
    except Exception as e:
        raise RuntimeError(f"importance/ablation step failed: {e}") from e

    # ── Baseline method comparison ─────────────────────────────────────────
    _p("\n[BASE] Baseline method comparison (seasonal_mean / seasonal_trend / yoy_growth)...")
    from baseline_counterfactual import build_recovery_targets as _brt
    base_rows = []
    ids_A = set(df_A["disaster_id"])
    for bmethod in ["seasonal_mean", "seasonal_trend", "yoy_growth"]:
        cfg_b = {**cfg, "baseline": {**cfg.get("baseline", {}), "method": bmethod}}
        df_bm = _brt(emdat, arrivals, cfg_b)
        df_bm = df_bm[~df_bm["baseline_insufficient"]].dropna(subset=["recovery_months"]).sort_values("start_date").reset_index(drop=True)
        y_bm = df_bm["recovery_months"].values
        df_bmA = df_bm[df_bm["disaster_id"].isin(ids_A)].reset_index(drop=True)
        yA_bm = df_bmA["recovery_months"].values
        xr = loocv_default_xgb(df_bmA, cfg, seed=SEED)
        zr = loo_zero_baseline(yA_bm)
        base_rows.append({
            "method": bmethod, "n_events": len(y_bm), "median_recovery": float(np.median(y_bm)),
            "mean_recovery": round(float(np.mean(y_bm)), 3), "n_right_censored": int(np.sum(y_bm == 12)),
            "n_zero": int(np.sum(y_bm == 0)), "ExpA_N": int(len(yA_bm)),
            "ExpA_zero_share": round(float(np.mean(yA_bm == 0)), 4), "ExpA_n_censored": int(np.sum(yA_bm == 12)),
            "ExpA_XGBoost_default_MAE": xr["MAE"], "ExpA_XGBoost_default_R2": xr["R2"],
            "ExpA_median_baseline_MAE": zr["MAE"],
        })
        _p(f"      {bmethod}: zero={base_rows[-1]['n_zero']} cens={base_rows[-1]['n_right_censored']} XGB-default MAE={xr['MAE']}")
    base_df = pd.DataFrame(base_rows)
    base_df.to_csv(results_dir / "baseline_method_comparison.csv", index=False)

    # ── Confounding sensitivity ────────────────────────────────────────────
    _p("\n[CONF] Confounding window sensitivity...")
    conf_df = run_confounding_variants(df_A, cfg, results_dir, seed=SEED)
    _p(conf_df[["variant", "n_events", "MAE", "R2", "MAE_median_baseline"]].to_string(index=False))

    # ── Sensitivity analysis ───────────────────────────────────────────────
    _p("\n[SENS] Sensitivity analysis (threshold x baseline window x persistence)...")
    from sensitivity_analysis import run_full_sensitivity
    sens_cfg = cfg["sensitivity_analysis"]
    sens_df = run_full_sensitivity(
        df_A, arrivals, cfg, crisis_years,
        thresholds=sens_cfg["thresholds"], baseline_windows=sens_cfg["baseline_windows"],
        persistence_values=sens_cfg["persistence_values"], results_dir=results_dir, seed=SEED,
    )
    _p(f"      Sensitivity grid: {len(sens_df)} combinations.")

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
            for p in ["pandas", "numpy", "scikit-learn", "xgboost", "shap", "scipy", "scikit-survival", "matplotlib"]
        },
        "pinned_versions_ok": bool(versions_pinned),
    }
    with open(results_dir / "run_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    # ── Macro lag sensitivity ──────────────────────────────────────────────
    _p("\n[MACLAG] GDP-growth lag sensitivity (0/1/2 years), default-setting XGBoost...")
    lag_rows = []
    for lag in [0, 1, 2]:
        r = loocv_default_xgb(df_A, cfg, gdp_lag=lag, seed=SEED)
        lag_rows.append({"gdp_lag_years": lag, "MAE": r["MAE"], "RMSE": r["RMSE"], "R2": r["R2"], "n_events": r["N"]})
        _p(f"      gdp_lag={lag}: MAE={r['MAE']}")
    pd.DataFrame(lag_rows).to_csv(results_dir / "macro_lag_sensitivity.csv", index=False)

    # Consistency: the same nominal setting must give the same number in every analysis
    main_sens = float(sens_df[sens_df.is_main_setting].MAE.iloc[0])
    ref_vals = {
        "sensitivity main cell": main_sens,
        "confounding all": float(conf_df[conf_df.variant == "all"].MAE.iloc[0]),
        "ablation full": float(abl_df[abl_df.variant == "full"].MAE.iloc[0]),
        "macro lag 0": float(lag_rows[0]["MAE"]),
        "baseline comparison seasonal_mean": float(base_df[base_df.method == "seasonal_mean"].ExpA_XGBoost_default_MAE.iloc[0]),
    }
    assert max(ref_vals.values()) - min(ref_vals.values()) < 1e-9, f"Default-XGBoost analyses disagree: {ref_vals}"
    _p(f"      Consistency check passed: default-setting XGBoost MAE = {main_sens} in all analyses.")

    # Paper statistics, Figure 1 and paper_numbers.json
    compute_paper_extras(df_B, df_A, train_idx, test_idx, arrivals, cfg, results_dir)
    make_figure1(results_dir, figures_dir)
    _p("      paper_extras.json and figure1.{svg,pdf,png} written.")

    from paper_numbers import extract_paper_numbers
    pn = extract_paper_numbers(results_dir)
    _p(f"\n[PAPER] paper_numbers.json written ({len(pn)} keys).")

    from make_readme import main as _readme
    _readme(results_dir)

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
