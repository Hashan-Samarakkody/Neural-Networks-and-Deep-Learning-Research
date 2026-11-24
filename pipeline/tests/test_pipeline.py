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

from baseline_counterfactual import compute_recovery_months
from feature_engineering import compute_dsi, compute_twdi, compute_dtii, add_seasonality
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
        t, _, censored = compute_recovery_months(
            self.ARRIVALS, disaster_year=2012, disaster_month=6,
            K=3, threshold=0.90, max_months=12, crisis_years=set(),
        )
        assert t == 0
        assert censored is False

    def test_delayed_recovery(self):
        """Months 7-8 suppressed; month 9 recovers → T_rec=2."""
        arr = _make_arrivals(override={(2012, 7): 50_000, (2012, 8): 50_000})
        t, _, censored = compute_recovery_months(
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
        t, _, censored = compute_recovery_months(
            arr, disaster_year=2012, disaster_month=6,
            K=3, threshold=0.90, max_months=12, crisis_years=set(),
        )
        assert t == 12
        assert censored is True

    def test_crisis_year_exclusion(self):
        """If the only valid prior years are crisis years, falls back to overall mean."""
        arr = _make_arrivals(start=2010, end=2012)
        # 2009 and 2010 and 2011 are all crisis years → no reference years
        t, baselines, _ = compute_recovery_months(
            arr, disaster_year=2012, disaster_month=6,
            K=3, threshold=0.90, max_months=12,
            crisis_years={2009, 2010, 2011},
        )
        # Fallback should still produce a numeric baseline (overall mean)
        assert baselines[1] > 0


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

    def test_twdi_is_product(self):
        """TWDI = DSI * DTII exactly."""
        df = pd.DataFrame({
            "total_affected": [1000.0],
            "deaths": [10.0],
            "damage_usd_000": [500.0],
            "duration_days": [5.0],
            "location": ["Colombo"],
        })
        dtii_cfg = {"Colombo": 0.195}
        df, _ = compute_dsi(df)
        df = compute_dtii(df, dtii_cfg)
        df = compute_twdi(df)
        assert df["TWDI"].iloc[0] == pytest.approx(df["DSI"].iloc[0] * 0.195)

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
