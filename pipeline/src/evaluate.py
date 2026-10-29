"""
Evaluation: MAE, RMSE, R², and non-parametric Conformal Prediction Intervals.

Conformal Prediction (Split Conformal, Section 3.4 of spec):
    1. Compute absolute non-conformity scores on calibration folds:
       alpha_i = |y_i - y_hat_i|
    2. Calculate the (1-alpha) quantile:
       q_hat = Quantile({alpha_i}, ceil((n+1)(1-alpha)) / n)
    3. Prediction interval: [y_hat_new - q_hat, y_hat_new + q_hat]

This replaces the invalid parametric ±1.96*std assumption.

Wilcoxon signed-rank test for pairwise model comparison.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true: list | np.ndarray, y_pred: list | np.ndarray) -> dict:
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)}


def compute_metrics_with_ci(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> dict:
    """
    Compute MAE, RMSE, R² with 95% bootstrap confidence intervals.
    """
    rng = np.random.default_rng(seed)
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    n = len(y_true)

    boot_mae, boot_rmse, boot_r2 = [], [], []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt, yp = y_true[idx], y_pred[idx]
        boot_mae.append(mean_absolute_error(yt, yp))
        boot_rmse.append(np.sqrt(mean_squared_error(yt, yp)))
        if np.std(yt) > 0:
            boot_r2.append(r2_score(yt, yp))

    def ci(vals):
        return (round(np.percentile(vals, 2.5), 4), round(np.percentile(vals, 97.5), 4))

    base = compute_metrics(y_true, y_pred)
    return {
        **base,
        "MAE_95CI": ci(boot_mae),
        "RMSE_95CI": ci(boot_rmse),
        "R2_95CI": ci(boot_r2) if boot_r2 else (None, None),
    }


def conformal_prediction_interval(
    calibration_residuals: np.ndarray,
    alpha: float = 0.05,
) -> float:
    """
    Compute q_hat from calibration absolute residuals.

    q_hat = Quantile({|y_i - y_hat_i|}, ceil((n+1)(1-alpha)) / n)
    The prediction interval for a new point is [y_new - q_hat, y_new + q_hat].
    """
    n = len(calibration_residuals)
    level = np.ceil((n + 1) * (1 - alpha)) / n
    level = min(level, 1.0)
    q_hat = np.quantile(np.abs(calibration_residuals), level)
    return float(q_hat)


def conformal_pi_from_loocv(
    y_true: list | np.ndarray,
    y_pred: list | np.ndarray,
    alpha: float = 0.05,
    max_months: int = 12,
) -> dict:
    """
    Compute conformal PI width from LOOCV residuals.
    Each fold's residual is used as a calibration score.
    """
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    residuals = y_true - y_pred
    q_hat = conformal_prediction_interval(residuals, alpha=alpha)
    coverage = float(np.mean(np.abs(residuals) <= q_hat))
    return {
        "conformal_q_hat": round(q_hat, 4),
        "pi_lower_offset": round(-q_hat, 4),
        "pi_upper_offset": round(q_hat, 4),
        "pi_half_width_months": round(q_hat, 4),
        "empirical_coverage": round(coverage, 4),
        "target_coverage": 1 - alpha,
    }


def wilcoxon_comparison(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    model_a: str,
    model_b: str,
) -> dict:
    """
    Wilcoxon signed-rank test comparing absolute errors of two models.
    H0: median(|e_A| - |e_B|) = 0
    """
    ae_a = np.abs(y_true - pred_a)
    ae_b = np.abs(y_true - pred_b)
    diffs = ae_a - ae_b
    if np.all(diffs == 0):
        return {
            "test": "Wilcoxon signed-rank",
            "model_a": model_a,
            "model_b": model_b,
            "statistic": 0.0,
            "p_value": 1.0,
            "significant_at_0.05": False,
        }
    stat, p = stats.wilcoxon(diffs, alternative="two-sided", zero_method="wilcox")
    return {
        "test": "Wilcoxon signed-rank",
        "model_a": model_a,
        "model_b": model_b,
        "statistic": round(float(stat), 4),
        "p_value": round(float(p), 6),
        "significant_at_0.05": bool(p < 0.05),
    }


def summarize_experiment(
    results: dict[str, dict],
    experiment_name: str,
    alpha: float = 0.05,
    n_bootstrap: int = 200,
) -> pd.DataFrame:
    """
    Aggregate metrics across all models for one experiment.
    Returns a formatted summary DataFrame.
    """
    rows = []
    for model_name, res in results.items():
        y_true = np.array(res["y_true"])
        y_pred = np.array(res["y_pred"])
        m = compute_metrics_with_ci(y_true, y_pred, n_bootstrap=n_bootstrap)
        pi = conformal_pi_from_loocv(y_true, y_pred, alpha=alpha)
        rows.append({
            "Experiment": experiment_name,
            "Model": model_name,
            "MAE": m["MAE"],
            "MAE_95CI": str(m["MAE_95CI"]),
            "RMSE": m["RMSE"],
            "RMSE_95CI": str(m["RMSE_95CI"]),
            "R2": m["R2"],
            "R2_95CI": str(m["R2_95CI"]),
            "PI_half_width": pi["pi_half_width_months"],
            "PI_empirical_coverage": pi["empirical_coverage"],
            "N": len(y_true),
        })
    df = pd.DataFrame(rows)
    return df.sort_values(["Experiment", "MAE"])


def performance_by_hazard_type(
    df_events: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> pd.DataFrame:
    """
    MAE/RMSE/R2 breakdown by Disaster Type (aligned with df_events rows).
    Requires 'Disaster Type' column in df_events.
    """
    dtype_col = next((c for c in ["disaster_type", "Disaster Type"] if c in df_events.columns), None)
    if dtype_col is None:
        return pd.DataFrame()
    df = df_events[[dtype_col]].rename(columns={dtype_col: "Disaster Type"}).copy().reset_index(drop=True)
    df["y_true"] = np.array(y_true)
    df["y_pred"] = np.array(y_pred)
    rows = []
    for htype, grp in df.groupby("Disaster Type"):
        if len(grp) < 2:
            continue
        mae = mean_absolute_error(grp["y_true"], grp["y_pred"])
        rmse = np.sqrt(mean_squared_error(grp["y_true"], grp["y_pred"]))
        try:
            r2 = r2_score(grp["y_true"], grp["y_pred"])
        except Exception:
            r2 = float("nan")
        rows.append({"Hazard_Type": htype, "N": len(grp),
                     "MAE": round(mae, 4), "RMSE": round(rmse, 4), "R2": round(r2, 4)})
    return pd.DataFrame(rows).sort_values("MAE")


def target_distribution_summary(y: np.ndarray) -> dict:
    """Report statistics of the recovery target distribution."""
    return {
        "n_total": int(len(y)),
        "n_zero_months": int(np.sum(y == 0)),
        "n_right_censored_12": int(np.sum(y == 12)),
        "mean": round(float(np.mean(y)), 3),
        "median": round(float(np.median(y)), 3),
        "std": round(float(np.std(y)), 3),
        "min": round(float(np.min(y)), 3),
        "max": round(float(np.max(y)), 3),
        "censoring_rate_pct": round(100.0 * np.mean(y == 12), 1),
    }
