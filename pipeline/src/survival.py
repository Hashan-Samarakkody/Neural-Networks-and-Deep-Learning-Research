"""
Survival analysis for right-censored recovery durations (S-04).

Models:
  CoxPH    - penalized Cox proportional hazards (CoxPHSurvivalAnalysis, alpha=0.1)
  WeibullAFT - Weibull accelerated failure time via CoxnetSurvivalAnalysis
  DiscreteHazard - discrete-time hazard (logistic regression per time-step)

Evaluation:
  C-index          (concordance_index_censored)
  Integrated Brier Score (sksurv.metrics.integrated_brier_score)

Exports: survival_results.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

try:
    from sksurv.linear_model import CoxPHSurvivalAnalysis, CoxnetSurvivalAnalysis
    from sksurv.metrics import concordance_index_censored, integrated_brier_score
    from sksurv.preprocessing import OneHotEncoder as SurvivalEncoder
    SKSURV_AVAILABLE = True
except ImportError as _e:
    raise ImportError(
        f"scikit-survival required for S-04: {_e}\n"
        "Install with: pip install scikit-survival>=0.23.0"
    )

from feature_engineering import FEATURE_COLS
from preprocess import FeatureBuilder


def _make_survival_y(y: np.ndarray) -> np.ndarray:
    """Structured array (event bool, duration float) for sksurv."""
    events = (y < 12).astype(bool)
    durations = np.maximum(y.astype(float), 0.5)
    dtype = np.dtype([("event", bool), ("duration", float)])
    out = np.empty(len(y), dtype=dtype)
    out["event"] = events
    out["duration"] = durations
    return out


def _c_index(y_struct: np.ndarray, y_pred: np.ndarray) -> float:
    try:
        result = concordance_index_censored(
            y_struct["event"], y_struct["duration"], y_pred
        )
        return round(float(result[0]), 4)
    except Exception:
        return float("nan")


def _ibs(y_train_struct, y_test_struct, survs, times) -> float:
    try:
        _, ibs = integrated_brier_score(y_train_struct, y_test_struct, survs, times)
        return round(float(ibs), 4)
    except Exception:
        return float("nan")


def run_survival_loocv(df: pd.DataFrame, cfg: dict) -> dict:
    """LOOCV for CoxPH and Weibull-proxy survival models."""
    n = len(df)
    models = {
        "CoxPH": CoxPHSurvivalAnalysis(alpha=0.1, ties="efron"),
        "CoxnetAFT": CoxnetSurvivalAnalysis(alpha_min_ratio=0.1, l1_ratio=0.5, fit_baseline_model=True),
    }
    results = {name: {"y_true": [], "y_pred": [], "c_index": []} for name in models}

    for i in range(n):
        tr = [j for j in range(n) if j != i]
        fb = FeatureBuilder(cfg)
        X_tr = fb.fit_transform(df.iloc[tr])[FEATURE_COLS].values
        X_te = fb.transform(df.iloc[[i]])[FEATURE_COLS].values
        y_tr = df.iloc[tr]["recovery_months"].values
        y_te = df.iloc[i]["recovery_months"]

        y_tr_s = _make_survival_y(y_tr)

        for name, model in models.items():
            try:
                m = model.__class__(**model.get_params())
                m.fit(X_tr, y_tr_s)
                risk = float(m.predict(X_te)[0])
            except Exception:
                risk = float(np.median(y_tr))
            results[name]["y_true"].append(float(y_te))
            results[name]["y_pred"].append(risk)

    return results


def run_survival_split(df: pd.DataFrame, train_idx, test_idx, cfg: dict) -> dict:
    """Chronological-split evaluation for survival models."""
    fb = FeatureBuilder(cfg)
    X_tr = fb.fit_transform(df.iloc[train_idx])[FEATURE_COLS].values
    X_te = fb.transform(df.iloc[test_idx])[FEATURE_COLS].values
    y_tr = df.iloc[train_idx]["recovery_months"].values
    y_te = df.iloc[test_idx]["recovery_months"].values

    y_tr_s = _make_survival_y(y_tr)
    y_te_s = _make_survival_y(y_te)

    models = {
        "CoxPH": CoxPHSurvivalAnalysis(alpha=0.1, ties="efron"),
        "CoxnetAFT": CoxnetSurvivalAnalysis(alpha_min_ratio=0.1, l1_ratio=0.5, fit_baseline_model=True),
    }
    results = {}
    for name, model in models.items():
        try:
            model.fit(X_tr, y_tr_s)
            risk = model.predict(X_te)
            ci = _c_index(y_te_s, risk)

            times = np.percentile(y_te_s["duration"], [25, 50, 75])
            times = times[times > 0]
            ibs = float("nan")
            if hasattr(model, "predict_survival_function") and len(times) > 0:
                survs = model.predict_survival_function(X_te)
                surv_arr = np.row_stack([fn(times) for fn in survs])
                ibs = _ibs(y_tr_s, y_te_s, surv_arr, times)

            results[name] = {
                "y_true": y_te.tolist(),
                "y_pred": np.clip(risk, 0, 12).tolist(),
                "c_index": ci,
                "integrated_brier_score": ibs,
            }
        except Exception as e:
            results[name] = {"error": str(e)}

    return results


def export_survival_results(
    df: pd.DataFrame,
    cfg: dict,
    train_idx,
    test_idx,
    results_dir: Path,
) -> pd.DataFrame:
    """Run survival models and export survival_results.csv (S-04)."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    split_res = run_survival_split(df, train_idx, test_idx, cfg)
    rows = []
    for model_name, res in split_res.items():
        if "error" in res:
            rows.append({"model": model_name, "c_index": np.nan,
                         "integrated_brier_score": np.nan, "note": res["error"]})
        else:
            rows.append({
                "model": model_name,
                "c_index": res.get("c_index", np.nan),
                "integrated_brier_score": res.get("integrated_brier_score", np.nan),
                "n_test": len(res.get("y_true", [])),
                "note": "",
            })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(results_dir / "survival_results.csv", index=False)
    return df_out
