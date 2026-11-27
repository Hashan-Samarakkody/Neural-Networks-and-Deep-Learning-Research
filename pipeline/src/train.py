"""
Model training with nested cross-validation hyperparameter search.

Models:
    Ridge, RandomForest, SVR, XGBoost, TwoStage (zero-inflated)

Naive baselines:
    NaiveMean, NaiveMedian,
    NaivePrevious_Oracle  — previous event T_rec (look-ahead, non-ex-ante)
    NaiveLastKnown        — most recent event whose 12-month window closed (ex-ante)

Nested CV:
    Inner folds: KFold without shuffle (preserves temporal order).
    Model scaler inside sklearn Pipeline so inner CV refits scaler on each inner split.
    Outer loop: LOOCV (Exp A) or single chronological split (Exp B).
    FeatureBuilder fit inside every outer fold on training indices only.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV, KFold, LeaveOneOut
from math import prod
import xgboost as xgb

SEED = 42
_N_JOBS = min(4, max(1, (os.cpu_count() or 4) // 2))

MODEL_REGISTRY = {
    "Ridge": {
        "class": Ridge,
        "default_params": {},
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
    return {k: [None if x is None else x for x in v] for k, v in raw.items()}


def tune_model(
    name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    cfg: dict,
    n_inner_splits: int = 5,
) -> tuple:
    """
    Inner-fold RandomizedSearchCV.

    Model scaler is inside a sklearn Pipeline so the inner CV refits
    the scaler on each inner training split, not on the outer training fold.
    Returns (best_estimator, best_params_stripped, None).
    scaler=None because Pipeline owns the scaler.
    """
    meta = MODEL_REGISTRY[name]
    param_grid = _build_param_grid(cfg, meta["grid_key"])

    # no shuffle — preserve temporal order inside inner folds
    inner_cv = KFold(n_splits=n_inner_splits, shuffle=False)

    if meta["needs_scaling"]:
        estimator = Pipeline([
            ("scaler", StandardScaler()),
            ("model", meta["class"](**meta["default_params"])),
        ])
        # Prefix grid keys for Pipeline
        param_grid = {f"model__{k}": v for k, v in param_grid.items()}
    else:
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
        gs.fit(X_train, y_train)
        best_model = gs.best_estimator_
        # Strip Pipeline prefix so saved params are human-readable
        best_params = {
            k.replace("model__", ""): v for k, v in gs.best_params_.items()
        }
    else:
        estimator.fit(X_train, y_train)
        best_model = estimator
        best_params = {}

    return best_model, best_params, None  # scaler=None; Pipeline owns it


def predict_clipped(model, scaler, X_test: np.ndarray, max_months: int = 12) -> np.ndarray:
    # scaler is None when model is a Pipeline or unscaled estimator
    X_in = scaler.transform(X_test) if scaler is not None else X_test
    return np.clip(model.predict(X_in), 0, max_months)


def run_loocv(
    df: pd.DataFrame,
    cfg: dict,
    n_inner_splits: int = 5,
) -> dict:
    """
    Full LOOCV for Experiment A (pure natural hazards).

    Fits FeatureBuilder inside every fold on training indices only.
    TwoStage model is integrated here so it also uses fold-safe features.

    Returns {model_name: {"y_true", "y_pred", "best_params"}}.
    """
    from preprocess import FeatureBuilder
    from feature_engineering import FEATURE_COLS
    from two_stage_model import _fit as ts_fit, _predict as ts_predict

    model_names = list(MODEL_REGISTRY.keys()) + ["TwoStage"]
    results = {n: {"y_true": [], "y_pred": [], "best_params": []} for n in model_names}

    loo = LeaveOneOut()
    n = len(df)

    for fold_i, (train_idx, test_idx) in enumerate(loo.split(df)):
        df_tr = df.iloc[train_idx]
        df_te = df.iloc[test_idx]

        fb = FeatureBuilder(cfg)
        fb.fit(df_tr)
        X_tr = fb.transform(df_tr)[FEATURE_COLS].values
        X_te = fb.transform(df_te)[FEATURE_COLS].values
        y_tr = df_tr["recovery_months"].values
        y_te = df_te["recovery_months"].values

        for name in MODEL_REGISTRY:
            model, params, scaler = tune_model(name, X_tr, y_tr, cfg, n_inner_splits)
            pred = predict_clipped(model, scaler, X_te)
            results[name]["y_true"].extend(y_te.tolist())
            results[name]["y_pred"].extend(pred.tolist())
            results[name]["best_params"].append(params)

        # TwoStage uses same fold-safe features
        ts_sc, ts_clf, ts_reg = ts_fit(X_tr, y_tr)
        ts_pred = _predict_ts(ts_sc, ts_clf, ts_reg, X_te)
        results["TwoStage"]["y_true"].extend(y_te.tolist())
        results["TwoStage"]["y_pred"].extend(ts_pred.tolist())

        if (fold_i + 1) % 10 == 0 or fold_i == 0:
            print(f"      LOOCV fold {fold_i + 1}/{n}", flush=True)

    return results


def _predict_ts(scaler, clf, reg, X_test):
    from two_stage_model import _predict
    return _predict(scaler, clf, reg, X_test)


def run_chronological_split(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    cfg: dict,
    n_inner_splits: int = 5,
) -> dict:
    """
    Single chronological train/test split for Experiment B.

    Fits FeatureBuilder on training events only (start_year <= 2018).
    TwoStage integrated.
    Returns {model_name: {"y_true", "y_pred", "best_params", "model"}}.
    """
    from preprocess import FeatureBuilder
    from feature_engineering import FEATURE_COLS
    from two_stage_model import _fit as ts_fit, _predict as ts_predict

    df_tr = df.iloc[train_idx]
    df_te = df.iloc[test_idx]

    fb = FeatureBuilder(cfg)
    fb.fit(df_tr)
    X_tr = fb.transform(df_tr)[FEATURE_COLS].values
    X_te = fb.transform(df_te)[FEATURE_COLS].values
    y_tr = df_tr["recovery_months"].values
    y_te = df_te["recovery_months"].values

    results = {}
    for name in MODEL_REGISTRY:
        model, params, scaler = tune_model(name, X_tr, y_tr, cfg, n_inner_splits)
        pred = predict_clipped(model, scaler, X_te)
        results[name] = {
            "y_true": y_te.tolist(),
            "y_pred": pred.tolist(),
            "best_params": params,
            "model": model,
            "scaler": scaler,
            "feature_builder": fb,
        }

    ts_sc, ts_clf, ts_reg = ts_fit(X_tr, y_tr)
    ts_pred = _predict_ts(ts_sc, ts_clf, ts_reg, X_te)
    results["TwoStage"] = {
        "y_true": y_te.tolist(),
        "y_pred": ts_pred.tolist(),
        "best_params": {},
        "model": None,
        "scaler": None,
        "feature_builder": fb,
    }

    return results


def run_multisplit(
    df: pd.DataFrame,
    splits: list,
    cfg: dict,
    n_inner_splits: int = 5,
) -> dict:
    """
    Generic multi-split runner for buffered LOOCV, LOYO, expanding-window.

    splits: list of (train_idx, test_idx) arrays.
    Returns {model_name: {"y_true", "y_pred", "best_params"}}.
    """
    from preprocess import FeatureBuilder
    from feature_engineering import FEATURE_COLS
    from two_stage_model import _fit as ts_fit

    model_names = list(MODEL_REGISTRY.keys()) + ["TwoStage"]
    results = {n: {"y_true": [], "y_pred": [], "best_params": []} for n in model_names}

    for fold_i, (train_idx, test_idx) in enumerate(splits):
        df_tr = df.iloc[train_idx]
        df_te = df.iloc[test_idx]

        fb = FeatureBuilder(cfg)
        fb.fit(df_tr)
        X_tr = fb.transform(df_tr)[FEATURE_COLS].values
        X_te = fb.transform(df_te)[FEATURE_COLS].values
        y_tr = df_tr["recovery_months"].values
        y_te = df_te["recovery_months"].values

        for name in MODEL_REGISTRY:
            model, params, scaler = tune_model(
                name, X_tr, y_tr, cfg,
                n_inner_splits=min(n_inner_splits, max(2, len(train_idx) - 1)),
            )
            pred = predict_clipped(model, scaler, X_te)
            results[name]["y_true"].extend(y_te.tolist())
            results[name]["y_pred"].extend(pred.tolist())
            results[name]["best_params"].append(params)

        ts_sc, ts_clf, ts_reg = ts_fit(X_tr, y_tr)
        ts_pred = _predict_ts(ts_sc, ts_clf, ts_reg, X_te)
        results["TwoStage"]["y_true"].extend(y_te.tolist())
        results["TwoStage"]["y_pred"].extend(ts_pred.tolist())

    return results


def naive_baselines_loocv(y: np.ndarray, df: pd.DataFrame = None) -> dict:
    """
    Naive baselines under LOOCV.

    NaiveMean, NaiveMedian: leave-one-out mean/median of remaining events.
    NaivePrevious_Oracle: previous event T_rec (non-ex-ante look-ahead).
    NaiveLastKnown: T_rec of most recent event whose window closed before onset.
    """
    results = {
        "NaiveMean": {"y_true": [], "y_pred": []},
        "NaiveMedian": {"y_true": [], "y_pred": []},
        "NaivePrevious_Oracle": {"y_true": [], "y_pred": []},
        "NaiveLastKnown": {"y_true": [], "y_pred": []},
    }
    n = len(y)

    # Pre-compute onset dates for NaiveLastKnown
    onsets = None
    if df is not None and "start_date" in df.columns:
        onsets = pd.to_datetime(df["start_date"].values)

    for i in range(n):
        other = np.delete(y, i)
        results["NaiveMean"]["y_true"].append(y[i])
        results["NaiveMean"]["y_pred"].append(float(np.mean(other)))
        results["NaiveMedian"]["y_true"].append(y[i])
        results["NaiveMedian"]["y_pred"].append(float(np.median(other)))

        # Oracle: previous by index (may be future-peeking)
        prev_oracle = float(y[i - 1]) if i > 0 else float(np.mean(other))
        results["NaivePrevious_Oracle"]["y_true"].append(y[i])
        results["NaivePrevious_Oracle"]["y_pred"].append(prev_oracle)

        # LastKnown: most recent event j < i whose 12-month window closed
        if onsets is not None:
            lk_pred = float("nan")  # NaN when no prior closed window
            current_onset = onsets[i]
            for j in range(i - 1, -1, -1):
                if j == i:
                    continue
                window_close = onsets[j] + pd.DateOffset(months=12)
                if window_close <= current_onset:
                    lk_pred = float(y[j])
                    break
            results["NaiveLastKnown"]["y_true"].append(y[i])
            results["NaiveLastKnown"]["y_pred"].append(lk_pred)
        else:
            results["NaiveLastKnown"]["y_true"].append(y[i])
            results["NaiveLastKnown"]["y_pred"].append(float(np.median(other)))

    return results


def naive_baselines_split(
    y_train: np.ndarray,
    y_test: np.ndarray,
    df_train: pd.DataFrame = None,
    df_test: pd.DataFrame = None,
) -> dict:
    """Naive baselines for chronological split."""
    n_test = len(y_test)
    train_median = float(np.median(y_train))

    # NaiveLastKnown for split: last training event whose window closed before each test onset
    lk_preds = [train_median] * n_test
    if df_train is not None and df_test is not None and "start_date" in df_train.columns:
        train_onsets = pd.to_datetime(df_train["start_date"].values)
        test_onsets = pd.to_datetime(df_test["start_date"].values)
        for k, t_onset in enumerate(test_onsets):
            for j in range(len(y_train) - 1, -1, -1):
                window_close = train_onsets[j] + pd.DateOffset(months=12)
                if window_close <= t_onset:
                    lk_preds[k] = float(y_train[j])
                    break

    return {
        "NaiveMean": {
            "y_true": y_test.tolist(),
            "y_pred": [float(np.mean(y_train))] * n_test,
        },
        "NaiveMedian": {
            "y_true": y_test.tolist(),
            "y_pred": [train_median] * n_test,
        },
        "NaivePrevious_Oracle": {
            "y_true": y_test.tolist(),
            "y_pred": [float(y_train[-1])] * n_test,
        },
        "NaiveLastKnown": {
            "y_true": y_test.tolist(),
            "y_pred": lk_preds,
        },
    }
