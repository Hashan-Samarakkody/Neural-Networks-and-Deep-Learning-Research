"""
Sensitivity analysis: recovery threshold x baseline window x persistence.

Targets are recomputed with the same function as the main pipeline
(baseline_counterfactual.compute_recovery_months), so the baseline rule, the
no-future-years rule and the persistence rule are identical. Each cell is evaluated
with the shared default-setting XGBoost LOOCV (default_models) and with the LOO
median baseline. Output: sensitivity_analysis.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

from baseline_counterfactual import compute_recovery_months
from default_models import loocv_default_xgb, loo_zero_baseline


def recompute_targets(df_events, arrivals, window, threshold, persistence, crisis_years, method="seasonal_mean"):
    months, cens = [], []
    for _, r in df_events.iterrows():
        t, _, c, _ = compute_recovery_months(
            arrivals, int(r["start_year"]), int(r["start_month"]),
            K=window, threshold=threshold, max_months=12,
            crisis_years=crisis_years, method=method, persistence=persistence,
        )
        months.append(float(t)); cens.append(bool(c))
    return np.array(months), np.array(cens)


def run_full_sensitivity(
    df_A: pd.DataFrame,
    arrivals: pd.DataFrame,
    cfg: dict,
    crisis_years: set,
    thresholds: list,
    baseline_windows: list,
    persistence_values: list,
    results_dir: Path,
    seed: int = 42,
) -> pd.DataFrame:
    rows = []
    main_t, main_w, main_p = cfg["experiment"]["recovery_threshold"], cfg["experiment"]["baseline_window_years"], cfg["recovery"]["persistence_months"]
    for thresh in thresholds:
        for window in baseline_windows:
            for persistence in persistence_values:
                y, cens = recompute_targets(df_A, arrivals, window, thresh, persistence, crisis_years)
                xgb = loocv_default_xgb(df_A, cfg, y=y, seed=seed)
                med = loo_zero_baseline(y)
                rows.append({
                    "recovery_threshold": thresh, "baseline_window_years": window,
                    "persistence_months": persistence,
                    "zero_share": round(float(np.mean(y == 0)), 4), "n_censored": int(cens.sum()),
                    "n_changed_vs_main": None,
                    "MAE": xgb["MAE"], "RMSE": xgb["RMSE"], "R2": xgb["R2"], "N": xgb["N"],
                    "MAE_median_baseline": med["MAE"],
                    "is_main_setting": bool(thresh == main_t and window == main_w and persistence == main_p),
                })
                print(f"      Sensitivity: thr={thresh} K={window} persist={persistence} "
                      f"zero={rows[-1]['zero_share']} cens={rows[-1]['n_censored']} MAE={xgb['MAE']}")
    df = pd.DataFrame(rows)
    main = df[df.is_main_setting]
    if len(main):
        y_main = df_A["recovery_months"].values.astype(float)
        y_chk, _ = recompute_targets(df_A, arrivals, main_w, main_t, main_p, crisis_years)
        assert np.array_equal(y_chk, y_main), "Sensitivity main cell targets differ from the main pipeline targets"
        ref = df_A["recovery_months"].values.astype(float)
        for i in df.index:
            r = df.loc[i]
            y_i, _ = recompute_targets(df_A, arrivals, int(r.baseline_window_years), r.recovery_threshold,
                                       int(r.persistence_months), crisis_years)
            df.loc[i, "n_changed_vs_main"] = int(np.sum(y_i != ref))
    results_dir = Path(results_dir); results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / "sensitivity_analysis.csv", index=False)
    return df
