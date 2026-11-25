"""
Two-stage zero-inflated model for tourism recovery duration prediction.

Stage 1: LogisticRegression — classifies P(T_rec=0) vs P(T_rec>0).
Stage 2: Ridge — predicts duration on the T_rec>0 subset only.

Combined rule:
    stage_1 == 0  →  predict 0
    stage_1 == 1  →  clip(stage_2_pred, 1, 12)

Hyperparameters are fixed (C=1, alpha=1) because the non-zero training
subset in LOOCV has only ~12 events — too small for meaningful inner CV.
# ponytail: fixed hparams; add nested CV when N_nonzero > 30
"""

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

SEED = 42
_MAX = 12


def _fit(X_train: np.ndarray, y_train: np.ndarray):
    """Fit scaler + stage-1 clf + stage-2 reg. Returns (scaler, clf, reg|None)."""
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X_train)
    y_binary = (y_train > 0).astype(int)
    if len(np.unique(y_binary)) < 2:
        # ponytail: single-class fallback; rare edge case in LOOCV folds near start
        clf = DummyClassifier(strategy="most_frequent")
    else:
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=SEED, solver="lbfgs")
    clf.fit(Xs, y_binary)
    mask = y_train > 0
    reg = None
    if mask.sum() >= 2:
        reg = Ridge(alpha=1.0)
        reg.fit(Xs[mask], y_train[mask])
    return scaler, clf, reg


def _predict(scaler, clf, reg, X_test: np.ndarray) -> np.ndarray:
    Xs = scaler.transform(X_test)
    binary = clf.predict(Xs)
    out = np.zeros(len(X_test))
    nz = binary == 1
    if nz.any() and reg is not None:
        out[nz] = np.clip(reg.predict(Xs[nz]), 1, _MAX)
    elif nz.any():
        out[nz] = 1.0  # fallback: minimum non-zero
    return out


def run_two_stage_loocv(X: np.ndarray, y: np.ndarray) -> dict:
    """LOOCV for two-stage model. Returns {model_name: {y_true, y_pred}}."""
    y_true, y_pred = [], []
    for train_idx, test_idx in LeaveOneOut().split(X):
        scaler, clf, reg = _fit(X[train_idx], y[train_idx])
        y_true.append(float(y[test_idx[0]]))
        y_pred.append(float(_predict(scaler, clf, reg, X[test_idx])[0]))
    return {"TwoStage": {"y_true": y_true, "y_pred": y_pred}}


def run_two_stage_split(
    X: np.ndarray, y: np.ndarray,
    train_idx: np.ndarray, test_idx: np.ndarray,
) -> dict:
    """Chronological split for two-stage model."""
    scaler, clf, reg = _fit(X[train_idx], y[train_idx])
    preds = _predict(scaler, clf, reg, X[test_idx])
    return {"TwoStage": {"y_true": y[test_idx].tolist(), "y_pred": preds.tolist()}}


def stage1_loocv_accuracy(X: np.ndarray, y: np.ndarray) -> dict:
    """S-12: LOOCV accuracy + AUC + majority_rate for stage-1 classifier."""
    true_bin, pred_bin, pred_proba = [], [], []
    for train_idx, test_idx in LeaveOneOut().split(X):
        scaler, clf, _ = _fit(X[train_idx], y[train_idx])
        Xs_te = scaler.transform(X[test_idx])
        true_bin.append(int(y[test_idx[0]] > 0))
        pred_bin.append(int(clf.predict(Xs_te)[0]))
        if hasattr(clf, "predict_proba"):
            proba = clf.predict_proba(Xs_te)[0]
            pred_proba.append(float(proba[1]) if len(proba) > 1 else float(proba[0]))
        else:
            pred_proba.append(float(pred_bin[-1]))
    tb, pb = np.array(true_bin), np.array(pred_bin)
    majority_rate = round(float(max(tb.mean(), 1 - tb.mean())), 4)
    try:
        auc = round(float(roc_auc_score(tb, np.array(pred_proba))), 4)
    except Exception:
        auc = float("nan")
    return {
        "stage1_accuracy": round(float(accuracy_score(tb, pb)), 4),
        "stage1_f1_nonzero": round(float(f1_score(tb, pb, zero_division=0)), 4),
        "stage1_auc": auc,
        "stage1_majority_rate": majority_rate,
        "stage1_n_zero": int((tb == 0).sum()),
        "stage1_n_nonzero": int((tb == 1).sum()),
    }
