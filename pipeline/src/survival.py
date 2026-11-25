"""
R-10: Survival analysis for right-censored recovery durations.

Models:
  - Penalised Cox proportional hazards (lifelines CoxPHFitter, penalizer=0.1)
  - Weibull accelerated failure time (lifelines WeibullAFTFitter)

Evaluation:
  - Concordance index (C-index)
  - Integrated Brier Score (IBS) via manual computation

Usage:
    from survival import run_survival_loocv, run_survival_split
"""

import warnings
import numpy as np
import pandas as pd

try:
    from lifelines import CoxPHFitter, WeibullAFTFitter
    from lifelines.utils import concordance_index
    LIFELINES_AVAILABLE = True
except ImportError:
    LIFELINES_AVAILABLE = False
    warnings.warn("lifelines not installed; survival models unavailable. pip install lifelines")

from feature_engineering import FEATURE_COLS
from preprocess import FeatureBuilder


def _brier_score_at_t(times, events, predictions, t):
    """Brier score at time t (Equation 20 of Graf et al. 1999)."""
    n = len(times)
    bs = 0.0
    for i in range(n):
        s_pred = float(predictions.iloc[i]) if hasattr(predictions, "iloc") else float(predictions[i])
        if times[i] <= t and events[i] == 1:
            bs += (0 - s_pred) ** 2
        elif times[i] > t:
            bs += (1 - s_pred) ** 2
    return bs / n


def _integrated_brier_score(times, events, predict_fn, t_range):
    """Trapezoidal IBS over t_range."""
    scores = [_brier_score_at_t(times, events, predict_fn(t), t) for t in t_range]
    return float(np.trapz(scores, t_range) / (t_range[-1] - t_range[0]))


def _prepare_survival_df(df_feat: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Add duration/event columns expected by lifelines."""
    sdf = df_feat[FEATURE_COLS].copy()
    sdf["duration"] = y.astype(float)
    # event=0 when right-censored (T_rec==12); event=1 otherwise
    sdf["event"] = (y < 12).astype(int)
    # lifelines requires duration > 0
    sdf["duration"] = sdf["duration"].clip(lower=0.5)
    return sdf


def _run_one_model_loocv(fitter_cls, fitter_kwargs, df_all, cfg):
    """LOOCV for a single lifelines fitter. Returns y_true, y_pred (median survival time)."""
    n = len(df_all)
    y_true, y_pred = [], []
    for i in range(n):
        train_idx = [j for j in range(n) if j != i]
        fb = FeatureBuilder(cfg)
        df_train_feat = fb.fit_transform(df_all.iloc[train_idx])
        df_test_feat = fb.transform(df_all.iloc[[i]])
        y_train = df_all.iloc[train_idx]["recovery_months"].values
        y_test = df_all.iloc[i]["recovery_months"]

        sdf_train = _prepare_survival_df(df_train_feat, y_train)
        sdf_test = df_test_feat[FEATURE_COLS].copy()
        sdf_test["duration"] = float(y_test)
        sdf_test["event"] = int(y_test < 12)
        sdf_test["duration"] = max(float(y_test), 0.5)

        try:
            fitter = fitter_cls(**fitter_kwargs)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitter.fit(sdf_train, duration_col="duration", event_col="event")
            median_pred = fitter.predict_median(sdf_test.drop(columns=["duration", "event"]))
            pred = float(median_pred.iloc[0]) if hasattr(median_pred, "iloc") else float(median_pred)
            pred = np.clip(pred, 0, 12)
        except Exception:
            pred = float(np.median(y_train))

        y_true.append(float(y_test))
        y_pred.append(float(pred))

    return np.array(y_true), np.array(y_pred)


def run_survival_loocv(df: pd.DataFrame, cfg: dict) -> dict:
    """
    LOOCV for Cox and Weibull AFT survival models.
    Returns dict with keys "CoxPH" and "WeibullAFT", each containing y_true/y_pred.
    """
    if not LIFELINES_AVAILABLE:
        print("[WARN] lifelines not installed; skipping survival models.")
        return {}

    results = {}
    models = {
        "CoxPH": (CoxPHFitter, {"penalizer": 0.1}),
        "WeibullAFT": (WeibullAFTFitter, {"penalizer": 0.1}),
    }
    for name, (cls, kwargs) in models.items():
        y_true, y_pred = _run_one_model_loocv(cls, kwargs, df, cfg)
        results[name] = {"y_true": y_true.tolist(), "y_pred": y_pred.tolist()}

    return results


def run_survival_split(df: pd.DataFrame, train_idx, test_idx, cfg: dict) -> dict:
    """Chronological-split evaluation for survival models."""
    if not LIFELINES_AVAILABLE:
        return {}

    results = {}
    fb = FeatureBuilder(cfg)
    df_train_feat = fb.fit_transform(df.iloc[train_idx])
    df_test_feat = fb.transform(df.iloc[test_idx])
    y_train = df.iloc[train_idx]["recovery_months"].values
    y_test = df.iloc[test_idx]["recovery_months"].values

    sdf_train = _prepare_survival_df(df_train_feat, y_train)
    sdf_test_x = df_test_feat[FEATURE_COLS].copy()

    models = {
        "CoxPH": (CoxPHFitter, {"penalizer": 0.1}),
        "WeibullAFT": (WeibullAFTFitter, {"penalizer": 0.1}),
    }
    for name, (cls, kwargs) in models.items():
        try:
            fitter = cls(**kwargs)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fitter.fit(sdf_train, duration_col="duration", event_col="event")
            median_preds = fitter.predict_median(sdf_test_x)
            y_pred = np.clip(median_preds.values.flatten(), 0, 12)
        except Exception as e:
            print(f"[WARN] {name} split failed: {e}")
            y_pred = np.full(len(y_test), np.median(y_train))

        results[name] = {
            "y_true": y_test.tolist(),
            "y_pred": y_pred.tolist(),
            "model": fitter if "fitter" in dir() else None,
        }

    return results


def survival_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    C-index (concordance index) for survival predictions.
    Uses event indicator: event=1 if T_rec < 12.
    """
    events = (np.array(y_true) < 12).astype(int)
    if not LIFELINES_AVAILABLE or events.sum() == 0:
        return {"c_index": float("nan")}
    try:
        c = concordance_index(y_true, y_pred, events)
    except Exception:
        c = float("nan")
    return {"c_index": round(float(c), 4)}
