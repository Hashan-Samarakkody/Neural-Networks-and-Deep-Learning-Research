"""
Shared default-setting XGBoost LOOCV helper.

Every sensitivity analysis (threshold/window grid, confounding variants, feature
ablation, macro lag, baseline-method comparison, Experiment A permutation importance)
uses this single helper so that the same nominal setting always gives the same number.

Setting: XGBRegressor with library defaults except random_state=seed, nthread=1,
verbosity=0; fold-safe FeatureBuilder fitted on each training fold; predictions
clipped to [0, 12].
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

from feature_engineering import FEATURE_COLS, get_macro_features
from preprocess import FeatureBuilder


def _metrics(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 4),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        "R2": round(float(r2_score(y_true, y_pred)), 4) if len(y_true) > 1 else float("nan"),
        "N": int(len(y_true)),
    }


def _features(fb: FeatureBuilder, df: pd.DataFrame, cols: list[str], gdp_lag: int) -> np.ndarray:
    feat = fb.transform(df)
    if gdp_lag:
        lagged = []
        for y in df["start_year"].astype(int):
            try:
                lagged.append(get_macro_features(int(y) - gdp_lag).get("gdp_growth", 0.0))
            except KeyError:
                lagged.append(0.0)
        feat["gdp_growth"] = lagged
    return feat[cols].values


def loocv_default_xgb(
    df: pd.DataFrame,
    cfg: dict,
    y: np.ndarray | None = None,
    drop_features: tuple = (),
    gdp_lag: int = 0,
    seed: int = 42,
    return_predictions: bool = False,
):
    """LOOCV with default-setting XGBoost; returns metrics (and predictions)."""
    df = df.reset_index(drop=True).copy()
    if y is not None:
        df["recovery_months"] = np.asarray(y, dtype=float)
    cols = [c for c in FEATURE_COLS if c not in drop_features]
    yv = df["recovery_months"].values.astype(float)
    preds = []
    n = len(df)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        fb = FeatureBuilder(cfg).fit(df.iloc[tr])
        X_tr = _features(fb, df.iloc[tr], cols, gdp_lag)
        X_te = _features(fb, df.iloc[[i]], cols, gdp_lag)
        m = XGBRegressor(random_state=seed, verbosity=0, nthread=1)
        m.fit(X_tr, yv[tr])
        preds.append(float(np.clip(m.predict(X_te)[0], 0, 12)))
    preds = np.array(preds)
    out = _metrics(yv, preds)
    if return_predictions:
        return out, yv, preds
    return out


def loo_zero_baseline(y: np.ndarray) -> dict:
    """LOO median baseline (predicts the median of the other events)."""
    y = np.asarray(y, dtype=float)
    preds = np.array([np.median(np.delete(y, i)) for i in range(len(y))])
    return _metrics(y, preds)


def permutation_importance_loocv(
    df: pd.DataFrame,
    cfg: dict,
    n_repeats: int = 20,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Experiment A permutation importance for default-setting XGBoost.

    For each LOOCV fold the held-out value of one feature is replaced by values
    drawn from the training rows; the MAE increase is averaged over folds and repeats.
    """
    df = df.reset_index(drop=True)
    cols = list(FEATURE_COLS)
    yv = df["recovery_months"].values.astype(float)
    rng = np.random.default_rng(seed)
    n = len(df)
    base_err = np.zeros(n)
    perm_err = {c: np.zeros((n, n_repeats)) for c in cols}
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        fb = FeatureBuilder(cfg).fit(df.iloc[tr])
        X_tr = fb.transform(df.iloc[tr])[cols].values
        X_te = fb.transform(df.iloc[[i]])[cols].values
        m = XGBRegressor(random_state=seed, verbosity=0, nthread=1).fit(X_tr, yv[tr])
        base_err[i] = abs(yv[i] - float(np.clip(m.predict(X_te)[0], 0, 12)))
        for k, c in enumerate(cols):
            Xp = np.repeat(X_te, n_repeats, axis=0)
            Xp[:, k] = rng.choice(X_tr[:, k], size=n_repeats, replace=True)
            p = np.clip(m.predict(Xp), 0, 12)
            perm_err[c][i] = np.abs(yv[i] - p)
    rows = []
    for c in cols:
        inc = perm_err[c].mean(axis=1) - base_err
        rows.append({"Feature": c, "mean_mae_increase": round(float(inc.mean()), 4),
                     "std_mae_increase": round(float(inc.std(ddof=1)), 4)})
    out = pd.DataFrame(rows).sort_values("mean_mae_increase", ascending=False).reset_index(drop=True)
    out["Rank"] = np.arange(1, len(out) + 1)
    return out


ABLATIONS = {
    "full": (),
    "no_global_tourism_idx": ("global_tourism_idx",),
    "no_hazard_severity": ("DSI", "TWDI"),
    "no_economy": ("global_tourism_idx", "gdp_growth", "Economic_Condition_Index"),
    "no_seasonality": ("Season_sin", "Season_cos"),
}


def run_ablation(df: pd.DataFrame, cfg: dict, results_dir, seed: int = 42) -> pd.DataFrame:
    from pathlib import Path
    rows = []
    for name, drop in ABLATIONS.items():
        r = loocv_default_xgb(df, cfg, drop_features=drop, seed=seed)
        rows.append({"variant": name, "features_removed": ",".join(drop) or "none",
                     "n_features": len(FEATURE_COLS) - len(drop), **r})
    out = pd.DataFrame(rows)
    out.to_csv(Path(results_dir) / "ablation_features.csv", index=False)
    return out
