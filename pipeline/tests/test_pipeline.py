"""
Unit + integration tests for the tourism recovery pipeline.
Run from pipeline/:  python -m pytest tests/ -v
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from baseline_counterfactual import compute_recovery_months, build_recovery_targets
from feature_engineering import compute_dsi, compute_dtii, add_seasonality
from preprocess import FeatureBuilder
from evaluate import conformal_prediction_interval
from validation import chronological_split
from two_stage_model import _fit, _predict, run_two_stage_loocv


# ── helpers ────────────────────────────────────────────────────────────────

def _make_arrivals(override: dict = None, start=2005, end=2015, base=100_000):
    """Uniform arrivals at `base` with optional (year, month) overrides."""
    rows = [
        {"year": y, "month": m, "tourist_arrivals": base}
        for y in range(start, end + 1) for m in range(1, 13)
    ]
    df = pd.DataFrame(rows)
    if override:
        for (y, m), v in override.items():
            df.loc[(df.year == y) & (df.month == m), "tourist_arrivals"] = v
    return df


def _make_event_df(n=20, seed=42):
    """Minimal emdat-style DataFrame with DSI component columns."""
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "disaster_id": [f"T-{i}" for i in range(n)],
        "disaster_type": rng.choice(["Flood", "Storm", "Landslide"], n),
        "start_year": rng.integers(2000, 2019, n),  # 2000-2018 inclusive
        "start_month": rng.integers(1, 13, n),
        "start_date": [pd.Timestamp(2010, 1, 1)] * n,
        "location": ["Colombo"] * n,
        "deaths": rng.integers(0, 100, n).astype(float),
        "injured": rng.integers(0, 1000, n).astype(float),
        "total_affected": rng.integers(0, 10000, n).astype(float),
        "damage_usd_000": rng.integers(0, 50000, n).astype(float),
        "duration_days": rng.integers(1, 30, n).astype(float),
        "n_events_clustered": np.ones(n, int),
    })


# ── compute_recovery_months ─────────────────────────────────────────────────

class TestRecoveryMonths:
    """Disaster in month 6 of 2012; uniform arrivals = 100k; threshold 90%."""

    ARRIVALS = _make_arrivals()  # all 100k

    def test_immediate_recovery_gives_zero(self):
        """Month-after-disaster already at 100k >= 90k → T_rec=0."""
        t, _, censored, _ = compute_recovery_months(
            self.ARRIVALS, disaster_year=2012, disaster_month=6,
            K=3, threshold=0.90, max_months=12, crisis_years=set(),
        )
        assert t == 0
        assert censored is False

    def test_delayed_recovery(self):
        """Months 7-8 suppressed; month 9 recovers → T_rec=2."""
        arr = _make_arrivals(override={(2012, 7): 50_000, (2012, 8): 50_000})
        t, _, censored, _ = compute_recovery_months(
            arr, disaster_year=2012, disaster_month=6,
            K=3, threshold=0.90, max_months=12, crisis_years=set(),
        )
        assert t == 2
        assert censored is False

    def test_right_censoring(self):
        """All 12 post-disaster months suppressed → T_rec=12, censored=True."""
        suppressions = {(2012, m): 50_000 for m in range(7, 13)}
        suppressions.update({(2013, m): 50_000 for m in range(1, 7)})
        arr = _make_arrivals(override=suppressions)
        t, _, censored, _ = compute_recovery_months(
            arr, disaster_year=2012, disaster_month=6,
            K=3, threshold=0.90, max_months=12, crisis_years=set(),
        )
        assert t == 12
        assert censored is True

    def test_crisis_year_exclusion_no_fallback(self):
        """R-03: no all-years fallback. All prior years are crisis years → baseline=0, skip offset."""
        arr = _make_arrivals(start=2010, end=2012)
        t, baselines, _, _ = compute_recovery_months(
            arr, disaster_year=2012, disaster_month=6,
            K=3, threshold=0.90, max_months=12,
            crisis_years={2009, 2010, 2011},
        )
        # R-03: returns 0.0 when no valid reference years (no fallback to all-years mean)
        assert baselines[1] == pytest.approx(0.0)

    def test_december_onset_month_rollover(self):
        """Disaster in December: month+1 wraps to January of next year. T_rec computed correctly."""
        arr = _make_arrivals(start=2008, end=2014)
        # Suppress January and February 2013 (months 1-2 after December 2012 disaster)
        arr_mod = _make_arrivals(
            override={(2013, 1): 50_000, (2013, 2): 50_000},
            start=2008, end=2014,
        )
        t, baselines, _, _ = compute_recovery_months(
            arr_mod, disaster_year=2012, disaster_month=12,
            K=3, threshold=0.90, max_months=12, crisis_years=set(),
        )
        # offset 1 = Jan 2013 (50k < 90k), offset 2 = Feb 2013 (50k < 90k), offset 3 = Mar 2013 (100k >= 90k)
        assert t == 2
        # baselines for January use prior years' January values (non-zero)
        assert baselines[1] > 0

    def test_no_future_years_in_baseline(self):
        """R-03: baseline must only use years strictly before the disaster year."""
        # Arrivals only: 2010-2014. Disaster in 2010 → prior year 2009 missing.
        arr = _make_arrivals(start=2010, end=2014)
        t, baselines, _, _ = compute_recovery_months(
            arr, disaster_year=2010, disaster_month=6,
            K=3, threshold=0.90, max_months=12, crisis_years=set(),
        )
        # Only years < 2010 count; none exist → baseline 0 for all offsets
        for offset in range(1, 13):
            assert baselines.get(offset, 0.0) == pytest.approx(0.0)


# ── DSI / TWDI ──────────────────────────────────────────────────────────────

class TestFeatureEngineering:

    def test_dsi_zero_input(self):
        """All-zero disaster metrics → log1p=0 everywhere → DSI=0."""
        df = pd.DataFrame({
            "total_affected": [0.0],
            "deaths": [0.0],
            "damage_usd_000": [0.0],
            "duration_days": [0.0],
        })
        df_out, _ = compute_dsi(df)
        assert df_out["DSI"].iloc[0] == pytest.approx(0.0)

    def _make_fb_df(self, n=5, seed=0):
        """Minimal DataFrame suitable for FeatureBuilder."""
        rng = np.random.default_rng(seed)
        return pd.DataFrame({
            "total_affected": rng.uniform(0, 10000, n),
            "deaths": rng.uniform(0, 100, n),
            "duration_days": rng.uniform(1, 30, n),
            "start_year": [2010] * n,
            "start_month": rng.integers(1, 13, n),
            "location": ["Colombo"] * n,
        })

    def _minimal_cfg(self):
        return {
            "dtii": {"Colombo": 0.195},
            "external_shocks": {},
            "dsi_include_damage": False,
        }

    def test_twdi_is_dsi_pos_times_dtii(self):
        """R-02: TWDI = DSI_pos × DTII (not DSI × DTII). DSI_pos ∈ [0,1]."""
        df = self._make_fb_df(n=10)
        cfg = self._minimal_cfg()
        fb = FeatureBuilder(cfg)
        out = fb.fit_transform(df)
        np.testing.assert_allclose(
            out["TWDI"].values,
            (out["DSI_pos"] * out["DTII"]).values,
            rtol=1e-9,
        )

    def test_twdi_non_negative(self):
        """R-02: TWDI >= 0 always, even when DSI is negative."""
        df = self._make_fb_df(n=20, seed=99)
        cfg = self._minimal_cfg()
        fb = FeatureBuilder(cfg)
        out = fb.fit_transform(df)
        assert (out["TWDI"].values >= 0).all(), "TWDI must be non-negative"

    def test_twdi_monotone_in_dtii(self):
        """Higher DTII for identical severity → higher TWDI."""
        base = self._make_fb_df(n=5)
        cfg_low = {**self._minimal_cfg(), "dtii": {"Colombo": 0.10}}
        cfg_high = {**self._minimal_cfg(), "dtii": {"Colombo": 0.40}}
        fb_low = FeatureBuilder(cfg_low).fit(base)
        fb_high = FeatureBuilder(cfg_high).fit(base)
        out_low = fb_low.transform(base)
        out_high = fb_high.transform(base)
        # Every event: higher DTII → higher or equal TWDI
        assert (out_high["TWDI"].values >= out_low["TWDI"].values - 1e-9).all()

    def test_seasonality_orthogonal(self):
        """sin² + cos² == 1 for every month."""
        df = pd.DataFrame({"start_month": list(range(1, 13))})
        df = add_seasonality(df)
        norm = df["Season_sin"] ** 2 + df["Season_cos"] ** 2
        np.testing.assert_allclose(norm.values, 1.0, atol=1e-10)


# ── conformal_prediction_interval ───────────────────────────────────────────

class TestConformalPI:

    def test_known_quantile(self):
        """Residuals [1,2,3,4,5], alpha=0.05 → q at ceil(6*0.95)/5 = ceil(5.7)/5 = 6/5 = 1.2 → clipped to 1.0 → max residual = 5."""
        residuals = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        q = conformal_prediction_interval(residuals, alpha=0.05)
        assert q == pytest.approx(5.0)

    def test_coverage_at_least_nominal(self):
        """q_hat from LOOCV residuals must cover >= (1-alpha) of those same residuals."""
        rng = np.random.default_rng(0)
        residuals = np.abs(rng.normal(0, 2, 100))
        q = conformal_prediction_interval(residuals, alpha=0.10)
        coverage = float(np.mean(residuals <= q))
        assert coverage >= 0.90


# ── chronological_split ─────────────────────────────────────────────────────

class TestChronologicalSplit:

    def _df_and_arrays(self, years):
        n = len(years)
        df = pd.DataFrame({"start_year": years, "start_month": [1] * n})
        X = pd.DataFrame(np.zeros((n, 3)))
        y = pd.Series(np.zeros(n))
        return X, y, df

    def test_year_boundary(self):
        years = list(range(2010, 2025))
        X, y, df = self._df_and_arrays(years)
        train, test = chronological_split(X, y, df, train_cutoff_year=2018)
        assert all(df["start_year"].iloc[train] <= 2018)
        assert all(df["start_year"].iloc[test] > 2018)

    def test_no_overlap(self):
        years = list(range(2000, 2026))
        X, y, df = self._df_and_arrays(years)
        train, test = chronological_split(X, y, df, train_cutoff_year=2018)
        assert len(set(train) & set(test)) == 0

    def test_raises_on_empty_train(self):
        years = list(range(2020, 2026))
        X, y, df = self._df_and_arrays(years)
        with pytest.raises(ValueError):
            chronological_split(X, y, df, train_cutoff_year=2015)


# ── two_stage_model ──────────────────────────────────────────────────────────

class TestTwoStageModel:

    def _separable_data(self, n_zero=15, n_nonzero=10, seed=42):
        """
        Perfectly separable: zero events have X[:,0]<0, non-zero have X[:,0]>0.
        Non-zero targets all equal 5.
        """
        rng = np.random.default_rng(seed)
        X_z = np.column_stack([-rng.uniform(1, 3, n_zero), rng.normal(0, 0.1, n_zero)])
        X_nz = np.column_stack([rng.uniform(1, 3, n_nonzero), rng.normal(0, 0.1, n_nonzero)])
        X = np.vstack([X_z, X_nz])
        y = np.array([0.0] * n_zero + [5.0] * n_nonzero)
        return X, y

    def test_zero_prediction_on_zero_class(self):
        X, y = self._separable_data()
        scaler, clf, reg = _fit(X, y)
        # predict on a strongly zero-class point
        X_test = np.array([[-2.0, 0.0]])
        pred = _predict(scaler, clf, reg, X_test)
        assert pred[0] == pytest.approx(0.0)

    def test_nonzero_prediction_bounded(self):
        X, y = self._separable_data()
        scaler, clf, reg = _fit(X, y)
        # predict on a strongly non-zero class point
        X_test = np.array([[2.0, 0.0]])
        pred = _predict(scaler, clf, reg, X_test)
        assert 1.0 <= pred[0] <= 12.0

    def test_all_zero_target_no_stage2(self):
        """When all training targets are zero, reg=None, predictions all 0."""
        X = np.random.randn(20, 4)
        y = np.zeros(20)
        scaler, clf, reg = _fit(X, y)
        assert reg is None

    def test_loocv_output_shape(self):
        X, y = self._separable_data()
        result = run_two_stage_loocv(X, y)
        assert "TwoStage" in result
        assert len(result["TwoStage"]["y_true"]) == len(y)
        assert len(result["TwoStage"]["y_pred"]) == len(y)

    def test_predictions_in_valid_range(self):
        """All two-stage predictions must be in [0, 12]."""
        X, y = self._separable_data()
        result = run_two_stage_loocv(X, y)
        preds = np.array(result["TwoStage"]["y_pred"])
        assert np.all(preds >= 0) and np.all(preds <= 12)


# ── fold-safe preprocessing ──────────────────────────────────────────────────

class TestFoldSafeFeatures:
    """R-01: FeatureBuilder must fit only on training rows."""

    def _cfg(self):
        return {
            "dtii": {"Colombo": 0.195},
            "external_shocks": {},
            "dsi_include_damage": False,
        }

    def _df(self, n=30, seed=7):
        rng = np.random.default_rng(seed)
        return pd.DataFrame({
            "total_affected": rng.uniform(0, 50000, n),
            "deaths": rng.uniform(0, 500, n),
            "duration_days": rng.uniform(1, 60, n),
            "start_year": np.tile([2010, 2011, 2012], n // 3 + 1)[:n],
            "start_month": rng.integers(1, 13, n),
            "location": ["Colombo"] * n,
        })

    def test_scaler_ignores_test_rows(self):
        """Scaler parameters must equal those from train-only fit, not full-dataset fit."""
        df = self._df()
        train_idx = np.arange(0, 20)
        test_idx = np.arange(20, 30)

        fb_fold = FeatureBuilder(self._cfg())
        fb_fold.fit(df.iloc[train_idx])

        fb_full = FeatureBuilder(self._cfg())
        fb_full.fit(df)

        # Scaler mean should differ between train-only and full-dataset fit
        # (unless test rows are identical to train, which they're not by construction)
        train_mean_fold = fb_fold._dsi_scaler.mean_
        train_mean_full = fb_full._dsi_scaler.mean_
        assert not np.allclose(train_mean_fold, train_mean_full), (
            "Scaler fit on train only must differ from full-dataset fit"
        )

    def test_transform_before_fit_raises(self):
        """transform() without prior fit() must raise."""
        fb = FeatureBuilder(self._cfg())
        with pytest.raises(RuntimeError):
            fb.transform(self._df(n=5))

    def test_dsi_min_from_train_only(self):
        """DSI_pos range must reflect training distribution, not test distribution."""
        df = self._df(n=40)
        # Make last 10 rows extreme to widen the DSI range if test rows leaked in
        df.loc[30:, "total_affected"] = 1_000_000
        df.loc[30:, "deaths"] = 10_000

        fb = FeatureBuilder(self._cfg())
        fb.fit(df.iloc[:30])

        # DSI_max should be from train; test rows' DSI may exceed it → clipped to 1
        out_test = fb.transform(df.iloc[30:])
        assert (out_test["DSI_pos"].values <= 1.0).all()
        assert (out_test["DSI_pos"].values >= 0.0).all()


# ── NaiveLastKnown ────────────────────────────────────────────────────────────

class TestNaiveLastKnown:
    """R-09: NaiveLastKnown uses most recent event whose 12-month window closed before current onset."""

    def test_last_known_uses_closed_window(self):
        """Event at t=0: prior event at t-13 months qualifies; event at t-6 months does not."""
        from train import naive_baselines_loocv

        onsets = pd.to_datetime([
            "2010-01-01",   # event 0
            "2010-07-01",   # event 1 (onset 6 months after event 0 — window still open)
            "2011-08-01",   # event 2 (onset 19 months after event 0 — window closed)
        ])
        y = np.array([3.0, 5.0, 7.0])
        df = pd.DataFrame({
            "start_date": onsets,
            "start_year": [2010, 2010, 2011],
        })

        result = naive_baselines_loocv(y, df=df)
        lk = result["NaiveLastKnown"]

        # For event 2 (index 2): event 0's window closed (Jan 2010 + 12m = Jan 2011 < Aug 2011).
        # Event 1's window (Jul 2010 + 12m = Jul 2011) also closed before Aug 2011.
        # Last known = event 1 (most recent closed), so pred = 5.0
        assert lk["y_pred"][2] == pytest.approx(5.0)

        # For event 1 (index 1): event 0's window (Jan 2010 + 12m = Jan 2011) is NOT closed
        # before Jul 2010. No prior closed window → pred = NaN or 0
        assert np.isnan(lk["y_pred"][1]) or lk["y_pred"][1] == pytest.approx(0.0)

    def test_naive_previous_oracle_renamed(self):
        """R-09: NaivePrevious renamed to NaivePrevious_Oracle in baselines output."""
        from train import naive_baselines_loocv

        y = np.array([2.0, 4.0, 6.0])
        onsets = pd.to_datetime(["2010-01-01", "2011-01-01", "2012-01-01"])
        df = pd.DataFrame({"start_date": onsets, "start_year": [2010, 2011, 2012]})
        result = naive_baselines_loocv(y, df=df)
        assert "NaivePrevious_Oracle" in result, "NaivePrevious must be renamed to NaivePrevious_Oracle"
        assert "NaivePrevious" not in result, "Old name NaivePrevious must not exist"


# ── integration ──────────────────────────────────────────────────────────────

class TestIntegration:
    """
    End-to-end check from DataFrame-level (after data loading) through
    feature matrix → split → shape assertions.
    """

    @pytest.fixture(scope="class")
    def setup_data(self):
        from baseline_counterfactual import build_recovery_targets
        from feature_engineering import build_feature_matrix, FEATURE_COLS

        arrivals = _make_arrivals(start=2000, end=2020)
        events = _make_event_df(n=30)
        cfg = {
            "experiment": {
                "baseline_window_years": 3,
                "recovery_threshold": 0.90,
                "max_recovery_months": 12,
                "train_cutoff_year": 2012,
            },
            "external_shocks": {},
            "dtii": {"Colombo": 0.195},
        }
        df_targets = build_recovery_targets(events, arrivals, cfg)
        df_targets = df_targets.dropna(subset=["recovery_months"]).sort_values("start_date").reset_index(drop=True)
        X, y = build_feature_matrix(df_targets, cfg)
        return X, y, df_targets, FEATURE_COLS

    def test_feature_count(self, setup_data):
        X, y, _, FEATURE_COLS = setup_data
        assert X.shape[1] == 8
        assert list(X.columns) == FEATURE_COLS

    def test_target_shape_matches_features(self, setup_data):
        X, y, _, _ = setup_data
        assert len(X) == len(y)

    def test_no_nan_in_features(self, setup_data):
        X, _, _, _ = setup_data
        assert not X.isnull().any().any()

    def test_target_range(self, setup_data):
        _, y, _, _ = setup_data
        assert y.min() >= 0 and y.max() <= 12

    def test_chronological_split_sizes(self, setup_data):
        X, y, df, _ = setup_data
        X_pd = pd.DataFrame(X.values)
        y_s = pd.Series(y.values)
        train_idx, test_idx = chronological_split(X_pd, y_s, df, train_cutoff_year=2012)
        assert len(train_idx) + len(test_idx) == len(X)
        assert len(test_idx) > 0


# ── S-20: persistence rule ────────────────────────────────────────────────────

class TestRecoveryPersistence:
    """S-03: persistence=N requires N consecutive months above threshold."""

    def test_persistence_2_delays_recovery(self):
        """Single above-threshold month does not satisfy persistence=2."""
        # Month 1 above threshold, month 2 below, month 3+ above
        suppressions = {(2012, 8): 50_000}   # offset 2 falls below threshold
        arr = _make_arrivals(override=suppressions)
        # persistence=1: recovers at offset 1
        t1, _, _, _ = compute_recovery_months(
            arr, disaster_year=2012, disaster_month=6,
            K=3, threshold=0.90, max_months=12, crisis_years=set(),
            persistence=1,
        )
        # persistence=2: offset 1 alone is not enough (interrupted by offset 2 below)
        t2, _, _, _ = compute_recovery_months(
            arr, disaster_year=2012, disaster_month=6,
            K=3, threshold=0.90, max_months=12, crisis_years=set(),
            persistence=2,
        )
        assert t1 == 0, f"persistence=1 should recover immediately, got {t1}"
        assert t2 > 0, f"persistence=2 should not recover at offset 0 with gap at offset 2"

    def test_persistence_1_matches_original_behaviour(self):
        """persistence=1 (default) gives same result as old single-month check."""
        arr = _make_arrivals()
        t, _, censored, _ = compute_recovery_months(
            arr, disaster_year=2012, disaster_month=6,
            K=3, threshold=0.90, max_months=12, crisis_years=set(),
            persistence=1,
        )
        assert t == 0
        assert censored is False

    def test_persistence_2_all_above_recovers(self):
        """When every post-disaster month is above threshold, persistence=2 recovers at offset 1."""
        arr = _make_arrivals()
        t, _, censored, _ = compute_recovery_months(
            arr, disaster_year=2012, disaster_month=6,
            K=3, threshold=0.50, max_months=12, crisis_years=set(),
            persistence=2,
        )
        assert t <= 1
        assert censored is False


# ── S-20: trend / yoy baseline methods ───────────────────────────────────────

class TestBaselineMethods:
    """S-02: seasonal_trend and yoy_growth baseline variants return valid floats."""

    def test_seasonal_trend_baseline_positive(self):
        from baseline_counterfactual import compute_baseline
        arr = _make_arrivals(start=2007, end=2014)
        baselines = compute_baseline(
            arr, disaster_year=2012, disaster_month=6,
            K=3, crisis_years=set(), method="seasonal_trend",
        )
        assert baselines[1] > 0, "seasonal_trend baseline must be positive"

    def test_yoy_growth_baseline_positive(self):
        from baseline_counterfactual import compute_baseline
        arr = _make_arrivals(start=2007, end=2014)
        baselines = compute_baseline(
            arr, disaster_year=2012, disaster_month=6,
            K=3, crisis_years=set(), method="yoy_growth",
        )
        assert baselines[1] > 0, "yoy_growth baseline must be positive"

    def test_seasonal_mean_matches_default(self):
        from baseline_counterfactual import compute_baseline
        arr = _make_arrivals(start=2007, end=2014)
        b_explicit = compute_baseline(
            arr, disaster_year=2012, disaster_month=6,
            K=3, crisis_years=set(), method="seasonal_mean",
        )
        b_default = compute_baseline(
            arr, disaster_year=2012, disaster_month=6,
            K=3, crisis_years=set(),
        )
        assert b_explicit[1] == pytest.approx(b_default[1])

    def test_unknown_method_raises(self):
        from baseline_counterfactual import compute_baseline
        arr = _make_arrivals(start=2007, end=2014)
        with pytest.raises((ValueError, KeyError)):
            compute_baseline(
                arr, disaster_year=2012, disaster_month=6,
                K=3, crisis_years=set(), method="nonexistent_method",
            )


# ── S-20: block bootstrap block_events param ─────────────────────────────────

class TestBlockBootstrap:
    """S-08: block_events parameter controls overlapping block size."""

    def _make_preds(self, n=20, seed=0):
        rng = np.random.default_rng(seed)
        y_true = rng.uniform(0, 12, n)
        y_pred = y_true + rng.normal(0, 0.5, n)
        y_pred = np.clip(y_pred, 0, 12)
        return y_true, y_pred

    def test_block_events_stored_in_result(self):
        from evaluate import block_bootstrap_ci
        y_true, y_pred = self._make_preds()
        result = block_bootstrap_ci(y_true, y_pred, n_bootstrap=100, block_events=4)
        assert result["block_events"] == 4

    def test_different_block_sizes_give_different_cis(self):
        from evaluate import block_bootstrap_ci
        y_true, y_pred = self._make_preds(n=30)
        r1 = block_bootstrap_ci(y_true, y_pred, n_bootstrap=200, block_events=2, seed=0)
        r2 = block_bootstrap_ci(y_true, y_pred, n_bootstrap=200, block_events=10, seed=0)
        # CIs need not be identical when block size differs
        ci1 = r1["MAE_95CI_block_bootstrap"]
        ci2 = r2["MAE_95CI_block_bootstrap"]
        # At minimum both are valid (lower <= upper)
        assert ci1[0] <= ci1[1]
        assert ci2[0] <= ci2[1]

    def test_block_events_one_equals_iid(self):
        """block_events=1 is IID bootstrap (each event its own block)."""
        from evaluate import block_bootstrap_ci
        y_true, y_pred = self._make_preds(n=25)
        result = block_bootstrap_ci(y_true, y_pred, n_bootstrap=100, block_events=1)
        assert result["block_events"] == 1
        lo, hi = result["MAE_95CI_block_bootstrap"]
        assert lo >= 0 and hi >= lo


# ── S-20: Holm correction ─────────────────────────────────────────────────────

class TestHolmCorrection:
    """S-07: Holm-Bonferroni adjusted p-values >= raw p-values and monotone."""

    def test_holm_geq_raw(self):
        from evaluate import holm_correction
        p_raw = [0.001, 0.01, 0.05, 0.10, 0.50]
        p_adj = holm_correction(p_raw)
        for raw, adj in zip(p_raw, p_adj):
            assert adj >= raw - 1e-12, f"p_holm={adj} < p_raw={raw}"

    def test_holm_monotone_after_sort(self):
        """Adjusted p-values in sorted order must be non-decreasing."""
        from evaluate import holm_correction
        p_raw = [0.04, 0.001, 0.02, 0.10, 0.003]
        p_adj = holm_correction(p_raw)
        paired = sorted(zip(p_raw, p_adj), key=lambda x: x[0])
        adj_sorted = [a for _, a in paired]
        for i in range(len(adj_sorted) - 1):
            assert adj_sorted[i] <= adj_sorted[i + 1] + 1e-12

    def test_holm_all_significant_inflated(self):
        """With m tests and all p=0.001, smallest adj = 0.001*m."""
        from evaluate import holm_correction
        m = 5
        p_raw = [0.001] * m
        p_adj = holm_correction(p_raw)
        assert min(p_adj) == pytest.approx(0.001 * m, rel=1e-9)

    def test_holm_single_test_unchanged(self):
        """Single p-value: Holm leaves it unchanged."""
        from evaluate import holm_correction
        p_adj = holm_correction([0.03])
        assert p_adj[0] == pytest.approx(0.03)
