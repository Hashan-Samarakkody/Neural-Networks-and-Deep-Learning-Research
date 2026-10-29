"""
Model training with nested cross-validation hyperparameter search.

Models:
    - Ridge (Linear Regression with regularization; handles ECI-GlobalTourism collinearity)
    - Random Forest
    - Support Vector Regression (SVR)
    - XGBoost

Naive baselines:
    - Historical mean recovery duration
    - Historical median recovery duration
    - Previous event's recovery duration (leave-one-out mean of remaining)

Nested CV:
    Inner folds (5-fold CV) tune hyperparameters.
    Outer loop is LOOCV (Exp A) or single chronological split (Exp B).
    This prevents test-set contamination during hyperparameter search.
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, KFold
from math import prod
import xgboost as xgb
import yaml

SEED = 42
# Cap parallel workers: avoids spawning 64 processes on many-core machines
_N_JOBS = min(4, max(1, (os.cpu_count() or 4) // 2))

MODEL_REGISTRY = {
    "Ridge": {
        "class": Ridge,
        "default_params": {"random_state": None},
        "grid_key": "ridge",
        "needs_scaling": True,
    },
    "RandomForest": {
        "class": RandomForestRegressor,
        "default_params": {"random_state": SEED, "n_jobs": 1},
        "grid_key": "random_forest",
        "needs_scaling": False,
    },
    "SVR": {
        "class": SVR,
        "default_params": {},
        "grid_key": "svr",
        "needs_scaling": True,
    },
    "XGBoost": {
        "class": xgb.XGBRegressor,
        "default_params": {
            "random_state": SEED,
            "verbosity": 0,
            "tree_method": "hist",
            "objective": "reg:squarederror",
            "nthread": 1,
        },
        "grid_key": "xgboost",
        "needs_scaling": False,
    },
}


def _build_param_grid(cfg: dict, grid_key: str) -> dict:
    grids = cfg.get("hyperparameter_grids", {})
    raw = grids.get(grid_key, {})
    processed = {}
    for k, v in raw.items():
        processed[k] = [None if x is None else x for x in v]
    return processed


def tune_model(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: dict,
    n_inner_splits: int = 5,
) -> tuple:
    """
    Inner-fold GridSearchCV to find best hyperparameters.

    Returns (best_estimator, best_params, scaler_or_None).
    """
    meta = MODEL_REGISTRY[name]
    param_grid = _build_param_grid(cfg, meta["grid_key"])

    scaler = None
    X_fit = X_train
    if meta["needs_scaling"]:
        scaler = StandardScaler()
        X_fit = scaler.fit_transform(X_train)

    inner_cv = KFold(n_splits=n_inner_splits, shuffle=True, random_state=SEED)
    estimator = meta["class"](**meta["default_params"])

    if param_grid:
        n_combos = prod(len(v) for v in param_grid.values())
        n_iter = min(30, n_combos)
        gs = RandomizedSearchCV(
            estimator,
            param_grid,
            n_iter=n_iter,
            cv=inner_cv,
            scoring="neg_mean_absolute_error",
            n_jobs=_N_JOBS,
            refit=True,
            random_state=SEED,
        )
        gs.fit(X_fit, y_train)
        best_model = gs.best_estimator_
        best_params = gs.best_params_
    else:
        estimator.fit(X_fit, y_train)
        best_model = estimator
        best_params = {}

    return best_model, best_params, scaler


def predict_clipped(model, scaler, X_test: np.ndarray, max_months: int = 12) -> np.ndarray:
    X_in = scaler.transform(X_test) if scaler is not None else X_test
    preds = model.predict(X_in)
    return np.clip(preds, 0, max_months)


def run_loocv(
    X: np.ndarray,
    y: np.ndarray,
    cfg: dict,
    n_inner_splits: int = 5,
) -> dict[str, dict]:
    """
    Full LOOCV training loop for Experiment A.

    Returns dict: {model_name: {"y_true": [...], "y_pred": [...], "best_params": [...]}}
    """
    from validation import loocv_split
    from sklearn.model_selection import LeaveOneOut

    results = {name: {"y_true": [], "y_pred": [], "best_params": []} for name in MODEL_REGISTRY}

    loo = LeaveOneOut()
    n = len(X)

    for fold_i, (train_idx, test_idx) in enumerate(loo.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        for name in MODEL_REGISTRY:
            model, params, scaler = tune_model(name, X_train, y_train, cfg, n_inner_splits)
            pred = predict_clipped(model, scaler, X_test)
            results[name]["y_true"].extend(y_test.tolist())
            results[name]["y_pred"].extend(pred.tolist())
            results[name]["best_params"].append(params)

        if (fold_i + 1) % 10 == 0 or fold_i == 0:
            import sys
            print(f"      LOOCV fold {fold_i + 1}/{n}", flush=True)

    return results


def run_chronological_split(
    X: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    cfg: dict,
    n_inner_splits: int = 5,
) -> dict[str, dict]:
    """
    Single chronological train/test split for Experiment B.

    Returns dict: {model_name: {"y_true": [...], "y_pred": [...], "best_params": {...}}}
    """
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    results = {}
    for name in MODEL_REGISTRY:
        model, params, scaler = tune_model(name, X_train, y_train, cfg, n_inner_splits)
        pred = predict_clipped(model, scaler, X_test)
        results[name] = {
            "y_true": y_test.tolist(),
            "y_pred": pred.tolist(),
            "best_params": params,
            "model": model,
            "scaler": scaler,
        }

    return results


def naive_baselines_loocv(y: np.ndarray) -> dict[str, dict]:
    """
    Compute naive baselines under LOOCV:
    - HistoricalMean: mean of all other fold events
    - HistoricalMedian: median of all other fold events
    - PreviousEvent: most recent event before current (by index order, not time-strict)
    """
    results = {
        "NaiveMean": {"y_true": [], "y_pred": []},
        "NaiveMedian": {"y_true": [], "y_pred": []},
        "NaivePrevious": {"y_true": [], "y_pred": []},
    }
    n = len(y)
    for i in range(n):
        other = np.delete(y, i)
        results["NaiveMean"]["y_true"].append(y[i])
        results["NaiveMean"]["y_pred"].append(float(np.mean(other)))
        results["NaiveMedian"]["y_true"].append(y[i])
        results["NaiveMedian"]["y_pred"].append(float(np.median(other)))
        prev_pred = float(y[i - 1]) if i > 0 else float(np.mean(other))
        results["NaivePrevious"]["y_true"].append(y[i])
        results["NaivePrevious"]["y_pred"].append(prev_pred)

    return results


def naive_baselines_split(
    y_train: np.ndarray, y_test: np.ndarray
) -> dict[str, dict]:
    return {
        "NaiveMean": {
            "y_true": y_test.tolist(),
            "y_pred": [float(np.mean(y_train))] * len(y_test),
        },
        "NaiveMedian": {
            "y_true": y_test.tolist(),
            "y_pred": [float(np.median(y_train))] * len(y_test),
        },
        "NaivePrevious": {
            "y_true": y_test.tolist(),
            "y_pred": [float(y_train[-1])] * len(y_test),
        },
    }
