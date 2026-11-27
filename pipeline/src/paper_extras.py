"""
Statistics quoted in the paper that are derived from pipeline data (not model fits).
Written to outputs/results/paper_extras.json so that every paper number is reproducible
from the repository.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path

from feature_engineering import FEATURE_COLS
from preprocess import FeatureBuilder

MAX_M = 12


def km_curve(y: np.ndarray, censored: np.ndarray):
    """Kaplan-Meier estimate of P(not recovered by t); event = recovery at T_rec."""
    xs, ys, S = [0], [1.0], 1.0
    for k in range(MAX_M + 1):
        n = int((y >= k).sum()); ev = int(((y == k) & ~censored).sum())
        if n > 0 and ev > 0:
            S *= 1 - ev / n
        xs.append(k); ys.append(S)
    return xs, ys


def compute_paper_extras(df_all, df_A, train_idx, test_idx, arrivals, cfg, results_dir) -> dict:
    results_dir = Path(results_dir)
    out = {}
    y = df_all["recovery_months"].values.astype(float)
    cens = df_all["is_right_censored"].values.astype(bool)
    crisis = df_all["external_shock_flag"].values == 1 if "external_shock_flag" in df_all else None
    if crisis is None:
        yrs = set()
        for lo, hi in cfg["external_shocks"].values():
            yrs.update(range(lo, hi + 1))
        crisis = df_all["start_year"].isin(yrs).values

    out["n_events"] = int(len(y)); out["n_zero"] = int((y == 0).sum()); out["n_censored"] = int(cens.sum())
    out["mean_target"] = round(float(y.mean()), 4)
    yA = df_A["recovery_months"].values.astype(float)
    out["expA"] = {"n": int(len(yA)), "zero_share": round(float((yA == 0).mean()), 4),
                   "mean_target": round(float(yA.mean()), 4),
                   "predict_zero_MAE": round(float(np.abs(yA).mean()), 4),
                   "nonzero_by_year": {int(k): int(v) for k, v in
                                       df_A[df_A.recovery_months > 0].start_year.value_counts().sort_index().items()}}
    tr = df_all.iloc[train_idx]; te = df_all.iloc[test_idx]
    out["expB"] = {"n_train": int(len(tr)), "n_test": int(len(te)),
                   "train_zero_share": round(float((tr.recovery_months == 0).mean()), 4),
                   "train_mean": round(float(tr.recovery_months.mean()), 4),
                   "test_mean": round(float(te.recovery_months.mean()), 4),
                   "test_censored": int(te.is_right_censored.sum())}
    cy = df_all[crisis]
    out["crisis_window"] = {
        "n": int(len(cy)), "n_censored": int(cy.is_right_censored.sum()),
        "censored_by_year": {int(k): int(v) for k, v in cy.groupby("start_year").is_right_censored.sum().items()},
        "n_events_by_year": {int(k): int(v) for k, v in cy.groupby("start_year").size().items()},
        "n_outside": int((~crisis).sum()), "n_censored_outside": int(df_all[~crisis].is_right_censored.sum()),
    }
    ci_, cc_ = df_all[~crisis], df_all[crisis]
    out["km_not_recovered_at_12"] = {
        "outside_crisis": round(km_curve(ci_.recovery_months.values.astype(float), ci_.is_right_censored.values.astype(bool))[1][-1], 4),
        "crisis_window": round(km_curve(cc_.recovery_months.values.astype(float), cc_.is_right_censored.values.astype(bool))[1][-1], 4)}
    out["km_curves"] = {"outside_crisis": km_curve(ci_.recovery_months.values.astype(float), ci_.is_right_censored.values.astype(bool)),
                        "crisis_window": km_curve(cc_.recovery_months.values.astype(float), cc_.is_right_censored.values.astype(bool))}
    # event spacing
    d = df_all.sort_values(["start_year", "start_month"])
    idx = d.start_year * 12 + d.start_month
    gap = idx.diff().iloc[1:]
    out["event_spacing"] = {"pairs": int(len(gap)), "within_12_months": int((gap <= 12).sum()),
                            "consecutive_pairs_same_target": int((d.recovery_months.diff().iloc[1:] == 0).sum())}
    # arrivals
    ann = arrivals.groupby("year").tourist_arrivals.sum()
    pc = (ann.pct_change() * 100)
    piv = arrivals.pivot(index="month", columns="year", values="tourist_arrivals")
    ratio = (piv[2023] / piv[2018])
    out["arrivals"] = {"annual_2018": int(ann[2018]), "annual_2021": int(ann[2021]),
                       "pct_change_2007": round(float(pc[2007]), 1), "pct_change_2008": round(float(pc[2008]), 1),
                       "ratio_2023_to_2018_min": round(float(ratio.min()), 3), "ratio_2023_to_2018_max": round(float(ratio.max()), 3)}
    # fold-safe diagnostics for Experiment B
    fb = FeatureBuilder(cfg).fit(tr)
    Xtr = fb.transform(tr)[FEATURE_COLS]; Xte = fb.transform(te)[FEATURE_COLS]
    diag = {}
    for c in FEATURE_COLS:
        lo, hi = float(Xtr[c].min()), float(Xtr[c].max())
        diag[c] = {"train_min": round(lo, 4), "train_max": round(hi, 4),
                   "test_min": round(float(Xte[c].min()), 4), "test_max": round(float(Xte[c].max()), 4),
                   "share_test_outside_train_range": round(float(((Xte[c] < lo) | (Xte[c] > hi)).mean()), 4)}
    out["expB_feature_diagnostics_foldsafe"] = diag
    out["corr_global_index_year_train"] = round(float(np.corrcoef(Xtr["global_tourism_idx"], tr["start_year"])[0, 1]), 4)
    out["corr_eci_gdp_train_foldsafe"] = round(float(np.corrcoef(Xtr["Economic_Condition_Index"], Xtr["gdp_growth"])[0, 1]), 4)
    # regime-split errors from saved predictions
    reg = {}
    for exp in ("ExpA_LOOCV", "ExpB_Chronological"):
        p = results_dir / f"predictions_{exp}.csv"
        if p.exists():
            pr = pd.read_csv(p)
            g = pr.assign(ae=(pr.y_true - pr.y_pred).abs()).groupby(["regime", "model"]).ae.agg(["mean", "size"]).reset_index()
            reg[exp] = {r: {m: {"MAE": round(float(v), 4), "n": int(n)} for m, v, n in
                            zip(sub.model, sub["mean"], sub["size"])} for r, sub in g.groupby("regime")}
    out["mae_by_regime"] = reg
    (results_dir / "paper_extras.json").write_text(json.dumps(out, indent=2, default=float))
    return out
