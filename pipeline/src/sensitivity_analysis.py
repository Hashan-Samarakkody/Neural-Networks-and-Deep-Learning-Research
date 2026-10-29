"""
Sensitivity analysis: vary recovery threshold and baseline window.

Tests robustness of findings across:
    - Recovery thresholds: 0.80, 0.90, 1.00 (main = 0.90)
    - Baseline windows: 3, 6, 12 non-crisis reference years (main = 3)

Exports CSV comparing MAE/RMSE/R2 across all combinations for best model.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def _build_arrivals_index(arrivals: pd.DataFrame) -> dict:
    """Pre-index arrivals by (year, month) for O(1) lookup."""
    idx = {}
    for _, row in arrivals.iterrows():
        idx[(int(row["year"]), int(row["month"]))] = float(row["tourist_arrivals"])
    return idx


def compute_recovery_months_threshold(
    event_row: pd.Series,
    arrivals_idx: dict,
    all_years: list,
    baseline_window: int,
    recovery_threshold: float,
    max_months: int = 12,
    crisis_years: set = None,
) -> tuple[int, bool]:
    """
    Compute recovery duration for one event under given threshold + baseline window.
    arrivals_idx: pre-built dict from _build_arrivals_index().
    Returns (recovery_months, is_right_censored).
    """
    if crisis_years is None:
        crisis_years = {2019, 2020, 2021, 2022}

    event_year = int(event_row.get("start_year", event_row.get("Start Year", 0)))
    event_month = int(event_row.get("start_month", event_row.get("Start Month", 1)))

    ref_years = sorted(
        [y for y in all_years if y < event_year and y not in crisis_years],
        reverse=True,
    )[:baseline_window]

    if not ref_years:
        return np.nan, False

    baseline_vals = [
        arrivals_idx[(ry, event_month)]
        for ry in ref_years
        if (ry, event_month) in arrivals_idx
    ]

    if not baseline_vals:
        return np.nan, False

    target = recovery_threshold * np.mean(baseline_vals)

    for t in range(1, max_months + 1):
        total_month = event_month + t - 1
        check_year = event_year + total_month // 12
        check_month = total_month % 12 + 1
        val = arrivals_idx.get((check_year, check_month))
        if val is not None and val >= target:
            return t, False

    return max_months, True


def run_sensitivity_loocv(
    X: np.ndarray,
    y: np.ndarray,
    df_events: pd.DataFrame,
    arrivals_idx: dict,
    all_years: list,
    model_class,
    best_params: dict,
    threshold: float,
    baseline_window: int,
    crisis_years: set,
    seed: int = 42,
) -> dict:
    """
    Re-run LOOCV after recomputing targets under alternative threshold/window.
    Returns metrics dict.
    """
    new_targets = []
    for _, row in df_events.iterrows():
        months, _ = compute_recovery_months_threshold(
            row, arrivals_idx, all_years, baseline_window, threshold,
            crisis_years=crisis_years,
        )
        new_targets.append(months)

    y_sens = np.array(new_targets, dtype=float)
    valid_mask = ~np.isnan(y_sens)
    X_v = X[valid_mask]
    y_v = y_sens[valid_mask]

    if len(X_v) < 5:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan, "N": 0}

    loo = LeaveOneOut()
    preds, trues = [], []
    for train_idx, test_idx in loo.split(X_v):
        model = model_class(**best_params, random_state=seed)
        model.fit(X_v[train_idx], y_v[train_idx])
        p = float(model.predict(X_v[test_idx])[0])
        preds.append(np.clip(p, 0, 12))
        trues.append(y_v[test_idx[0]])

    y_true = np.array(trues)
    y_pred = np.array(preds)
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = float(r2_score(y_true, y_pred))
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4), "N": int(len(y_v))}


def run_full_sensitivity(
    X_A: np.ndarray,
    y_A: np.ndarray,
    df_A: pd.DataFrame,
    arrivals: pd.DataFrame,
    best_model_class,
    best_params: dict,
    thresholds: list,
    baseline_windows: list,
    crisis_years: set,
    results_dir: Path,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Grid over thresholds × baseline_windows. Export CSV.
    Returns DataFrame with one row per combination.
    """
    # Pre-index once; all 9 combos reuse the same dict
    arrivals_idx = _build_arrivals_index(arrivals)
    all_years = sorted(arrivals["year"].unique().tolist())

    rows = []
    for thresh in thresholds:
        for window in baseline_windows:
            metrics = run_sensitivity_loocv(
                X_A, y_A, df_A, arrivals_idx, all_years,
                best_model_class, best_params,
                thresh, window, crisis_years, seed,
            )
            row = {
                "recovery_threshold": thresh,
                "baseline_window_years": window,
                **metrics,
                "is_main_setting": (thresh == 0.90 and window == 3),
            }
            rows.append(row)
            print(
                f"      Sensitivity: threshold={thresh:.0%} window={window}yr "
                f"MAE={metrics['MAE']} RMSE={metrics['RMSE']} R2={metrics['R2']} N={metrics['N']}"
            )

    df = pd.DataFrame(rows)
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(results_dir / "sensitivity_analysis.csv", index=False)
    return df
