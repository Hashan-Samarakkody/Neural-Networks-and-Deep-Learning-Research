"""
Survival analysis for the right-censored recovery target.

Time and event definition (consistent with the target definition):
  t* = first recovered month (1..12). Recovered events: event=1, time=T_rec+1.
  Censored events (no recovery within 12 months): event=0, time=12.

Models: penalised Cox proportional hazards, random survival forest, and a
discrete-time hazard model (logistic regression on person-month rows).
Metrics: Harrell concordance index and integrated Brier score (when defined).
Evaluation: Experiment B (chronological split) and Experiment A (LOOCV, pooled
risk scores; no censored events, C-index only).
Output: survival_results.csv (a model that fails is reported with its error, never silently).
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression

from sksurv.linear_model import CoxPHSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_censored, integrated_brier_score

from feature_engineering import FEATURE_COLS
from preprocess import FeatureBuilder

MAXT = 12
MODELS = ("CoxPH", "RandomSurvivalForest", "DiscreteHazard")


def make_y(y: np.ndarray):
    y = np.asarray(y, dtype=float)
    event = y < MAXT
    time = np.where(event, y + 1.0, float(MAXT))
    out = np.empty(len(y), dtype=[("event", bool), ("time", float)])
    out["event"] = event
    out["time"] = time
    return out


class DiscreteHazard:
    """Logistic discrete-time hazard on person-month rows; risk = 1 - S(12)."""

    def __init__(self, C: float = 1.0):
        self.C = C

    def fit(self, X, ys):
        rows, lab = [], []
        for x, ev, t in zip(X, ys["event"], ys["time"]):
            last = int(t)
            for m in range(1, last + 1):
                rows.append(np.r_[x, m]); lab.append(int(ev and m == last))
        self.clf = LogisticRegression(C=self.C, max_iter=2000).fit(np.array(rows), np.array(lab))
        return self

    def survival(self, X, times):
        out = np.ones((len(X), len(times)))
        for i, x in enumerate(X):
            S = 1.0; surv = {}
            for m in range(1, MAXT + 1):
                h = self.clf.predict_proba(np.r_[x, m].reshape(1, -1))[0, 1]
                S *= (1 - h); surv[m] = S
            out[i] = [surv[int(np.floor(t))] if t >= 1 else 1.0 for t in times]
        return out

    def predict(self, X):
        return 1.0 - self.survival(X, [MAXT])[:, 0]


def _fit_predict(name, X_tr, ys_tr, X_te):
    """Return risk scores and (optionally) survival function callable on grid."""
    if name == "CoxPH":
        m = CoxPHSurvivalAnalysis(alpha=0.1, ties="efron").fit(X_tr, ys_tr)
        return m.predict(X_te), (lambda times: np.vstack([fn(times) for fn in m.predict_survival_function(X_te)]))
    if name == "RandomSurvivalForest":
        m = RandomSurvivalForest(n_estimators=200, min_samples_leaf=3, random_state=42, n_jobs=1).fit(X_tr, ys_tr)
        return m.predict(X_te), (lambda times: np.vstack([fn(times) for fn in m.predict_survival_function(X_te)]))
    m = DiscreteHazard().fit(X_tr, ys_tr)
    return m.predict(X_te), (lambda times: m.survival(X_te, times))


def _cindex(ys, risk):
    return round(float(concordance_index_censored(ys["event"], ys["time"], risk)[0]), 4)


def _ibs(ys_tr, ys_te, surv_fn):
    """IBS over months 2..9; the test set is administratively censored at tau (training follow-up)."""
    tau = ys_tr["time"].max() - 0.5
    ys_t = ys_te.copy()
    over = ys_t["time"] > tau
    ys_t["time"][over] = tau
    ys_t["event"][over] = False
    times = np.array([t for t in np.arange(1.0, 12.0) if ys_t["time"].min() < t < tau])
    if len(times) < 2:
        return float("nan"), "fewer than two evaluation times inside the follow-up"
    try:
        S = surv_fn(times)
        return round(float(integrated_brier_score(ys_tr, ys_t, S, times)), 4), (
            f"IBS over months {int(times.min())}-{int(times.max())}; test censored at {tau:.1f}")
    except Exception as e:  # reported, not hidden
        return float("nan"), f"IBS undefined: {e}"


def run_survival_experiments(df_A, df_B, train_idx, test_idx, cfg, results_dir) -> pd.DataFrame:
    rows = []
    # Experiment B: chronological split
    fb = FeatureBuilder(cfg).fit(df_B.iloc[train_idx])
    X_tr = fb.transform(df_B.iloc[train_idx])[FEATURE_COLS].values
    X_te = fb.transform(df_B.iloc[test_idx])[FEATURE_COLS].values
    ys_tr = make_y(df_B.iloc[train_idx]["recovery_months"].values)
    ys_te = make_y(df_B.iloc[test_idx]["recovery_months"].values)
    for name in MODELS:
        try:
            risk, surv = _fit_predict(name, X_tr, ys_tr, X_te)
            ci = _cindex(ys_te, risk); ibs, note = _ibs(ys_tr, ys_te, surv)
        except Exception as e:
            ci, ibs, note = float("nan"), float("nan"), f"model failed: {e}"
        rows.append({"experiment": "ExpB_Chronological", "model": name, "c_index": ci,
                     "integrated_brier_score": ibs, "n_train": len(ys_tr), "n_test": len(ys_te),
                     "n_test_censored": int((~ys_te["event"]).sum()), "note": note})
    # Experiment A: LOOCV pooled risk scores
    n = len(df_A); yA = df_A["recovery_months"].values
    ysA = make_y(yA)
    for name in MODELS:
        risks = np.zeros(n)
        try:
            for i in range(n):
                tr = [j for j in range(n) if j != i]
                fbA = FeatureBuilder(cfg).fit(df_A.iloc[tr])
                Xa = fbA.transform(df_A.iloc[tr])[FEATURE_COLS].values
                Xt = fbA.transform(df_A.iloc[[i]])[FEATURE_COLS].values
                risks[i] = _fit_predict(name, Xa, ysA[tr], Xt)[0][0]
            ci, note = _cindex(ysA, risks), "no censored events in Experiment A; IBS not defined"
        except Exception as e:
            ci, note = float("nan"), f"model failed: {e}"
        rows.append({"experiment": "ExpA_LOOCV", "model": name, "c_index": ci,
                     "integrated_brier_score": float("nan"), "n_train": n - 1, "n_test": n,
                     "n_test_censored": 0, "note": note})
    out = pd.DataFrame(rows)
    results_dir = Path(results_dir); results_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(results_dir / "survival_results.csv", index=False)
    return out
