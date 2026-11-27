"""Tests for the revision-5 fixes: persistence in sensitivity, survival, naive multisplit,
confounding flags, default-model consistency, paper extras."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

import run_experiments as R
from data_loader import load_arrivals, load_emdat
from baseline_counterfactual import build_recovery_targets, compute_recovery_months
from sensitivity_analysis import recompute_targets
from confounding import flag_confounded_events
from default_models import loocv_default_xgb, loo_zero_baseline
from survival import make_y, DiscreteHazard, run_survival_experiments
from train import naive_baselines_multisplit
from paper_extras import compute_paper_extras, km_curve


@pytest.fixture(scope="module")
def data():
    cfg = R.load_config(R.CFG_PATH)
    arr = load_arrivals(R.ROOT / cfg["paths"]["arrivals_raw"])
    em = load_emdat(R.ROOT / cfg["paths"]["emdat_raw"], cfg)
    d = build_recovery_targets(em, arr, cfg)
    d = d[~d.baseline_insufficient].dropna(subset=["recovery_months"]).sort_values("start_date").reset_index(drop=True)
    crisis = set(range(2019, 2023))
    A = d[~d.start_year.isin(crisis)].reset_index(drop=True)
    return cfg, arr, d, A, crisis


def _toy_arrivals():
    rows = []
    for y in range(2000, 2012):
        for m in range(1, 13):
            rows.append({"year": y, "month": m, "tourist_arrivals": 100000.0})
    df = pd.DataFrame(rows)
    # event in 2010-03; month 4 recovers (100%), month 5 drops again, month 6+ recover
    df.loc[(df.year == 2010) & (df.month == 3), "tourist_arrivals"] = 10000.0
    df.loc[(df.year == 2010) & (df.month == 4), "tourist_arrivals"] = 100000.0
    df.loc[(df.year == 2010) & (df.month == 5), "tourist_arrivals"] = 10000.0
    return df


def test_persistence_changes_recovery_month():
    arr = _toy_arrivals()
    t1 = compute_recovery_months(arr, 2010, 3, K=3, threshold=0.9, crisis_years=set(), persistence=1)[0]
    t2 = compute_recovery_months(arr, 2010, 3, K=3, threshold=0.9, crisis_years=set(), persistence=2)[0]
    assert t1 == 0 and t2 > t1


def test_sensitivity_recompute_uses_persistence():
    arr = _toy_arrivals()
    ev = pd.DataFrame([{"start_year": 2010, "start_month": 3}])
    y1, _ = recompute_targets(ev, arr, 3, 0.9, 1, set())
    y2, _ = recompute_targets(ev, arr, 3, 0.9, 2, set())
    assert y2[0] > y1[0]


def test_survival_target_definition_and_models():
    y = np.array([0, 3, 11, 12, 12], dtype=float)
    ys = make_y(y)
    assert list(ys["event"]) == [True, True, True, False, False]
    assert list(ys["time"]) == [1.0, 4.0, 12.0, 12.0, 12.0]
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 3)); yy = np.clip((X[:, 0] * 3 + 4).round(), 0, 12)
    m = DiscreteHazard().fit(X, make_y(yy))
    S = m.survival(X[:3], [1, 5, 11])
    assert np.all(np.diff(S, axis=1) <= 1e-12)


def test_survival_experiments_not_nan(data, tmp_path):
    cfg, arr, d, A, _ = data
    tr = np.where(d.start_year <= 2018)[0]; te = np.where(d.start_year > 2018)[0]
    out = run_survival_experiments(A, d, tr, te, cfg, tmp_path)
    assert out["c_index"].notna().all()


def test_naive_multisplit_is_ex_ante():
    df = pd.DataFrame({"start_year": [2000, 2000, 2001, 2003], "start_month": [1, 6, 5, 1],
                       "recovery_months": [5.0, 0.0, 2.0, 7.0]})
    splits = [(np.array([0, 1, 2]), np.array([3]))]
    res = naive_baselines_multisplit(df, splits)
    # most recent training event closed >= 12 months before onset is event 2 (2001-05 -> closed by 2002-05)
    assert res["NaiveLastKnown"]["y_pred"] == [2.0]
    assert res["NaiveMedian"]["y_pred"] == [2.0]
    # a test event within 12 months of every training event falls back to the earlier-event median
    df2 = pd.DataFrame({"start_year": [2000, 2000], "start_month": [1, 3], "recovery_months": [4.0, 9.0]})
    r2 = naive_baselines_multisplit(df2, [(np.array([0]), np.array([1]))])
    assert r2["NaiveLastKnown"]["y_pred"] == [4.0]


def test_confounding_flags_include_tail(data):
    cfg, _, _, A, _ = data
    f = flag_confounded_events(A, cfg)
    assert f.loc[f.start_year == 2008, "confounded_by"].str.contains("civil_war_end").all()
    assert f.loc[f.start_year == 2023, "post_crisis_recovery"].eq(1).all()


def test_default_xgb_helper_is_deterministic(data):
    cfg, _, _, A, _ = data
    sub = A.iloc[:14].reset_index(drop=True)
    r1 = loocv_default_xgb(sub, cfg); r2 = loocv_default_xgb(sub, cfg)
    assert r1 == r2
    assert loo_zero_baseline(sub.recovery_months.values)["N"] == len(sub)


def test_paper_extras_keys(data, tmp_path):
    cfg, arr, d, A, _ = data
    tr = np.where(d.start_year <= 2018)[0]; te = np.where(d.start_year > 2018)[0]
    ex = compute_paper_extras(d, A, tr, te, arr, cfg, tmp_path)
    for k in ("expA", "expB", "km_not_recovered_at_12", "event_spacing", "arrivals", "expB_feature_diagnostics_foldsafe"):
        assert k in ex
    assert ex["expA"]["predict_zero_MAE"] == pytest.approx(A.recovery_months.mean(), abs=1e-3)
    x, y = km_curve(np.array([0., 0., 5.]), np.array([False, False, False]))
    assert y[-1] == 0.0
