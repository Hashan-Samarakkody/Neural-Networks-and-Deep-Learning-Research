"""
Fold-safe feature building.

FeatureBuilder.fit(df_train)   — learns DSI scaler, DSI_pos range, ECI scaler
                                   from training rows only.
FeatureBuilder.transform(df)   — applies without refitting.

Usage in LOOCV:
    fb = FeatureBuilder(cfg)
    fb.fit(df.iloc[train_idx])
    X_train = fb.transform(df.iloc[train_idx])[FEATURE_COLS].values
    X_test  = fb.transform(df.iloc[test_idx])[FEATURE_COLS].values
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from feature_engineering import (
    FEATURE_COLS, compute_dtii, add_seasonality,
    add_macro_features, add_external_shock_flag,
)

# damage_usd_000 is missing for 62 of 80 physical records (77.5 %); use 3-component DSI.
# Set dsi_include_damage: true in config to revert to 4-component (sensitivity).
_DSI3 = ["total_affected", "deaths", "duration_days"]
_DSI4 = ["total_affected", "deaths", "damage_usd_000", "duration_days"]


class FeatureBuilder:
    """Fit on training data, transform any split. Thread-unsafe (no locks)."""

    def __init__(self, cfg: dict):
        self._dtii = cfg["dtii"]
        self._shock = cfg.get("external_shocks", {})
        self._components = _DSI4 if cfg.get("dsi_include_damage", False) else _DSI3
        self._dsi_scaler: StandardScaler | None = None
        self._dsi_min: float | None = None
        self._dsi_max: float | None = None
        self._eci_scaler: StandardScaler | None = None

    # ------------------------------------------------------------------ #

    def fit(self, df_train: pd.DataFrame) -> "FeatureBuilder":
        df = df_train.copy()
        for c in self._components:
            if c not in df.columns:
                df[c] = 0.0
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

        log_df = df[self._components].apply(np.log1p)
        self._dsi_scaler = StandardScaler().fit(log_df)

        z = self._dsi_scaler.transform(log_df)
        dsi_tr = z.mean(axis=1)
        self._dsi_min = float(dsi_tr.min())
        self._dsi_max = float(dsi_tr.max())

        dfm = add_macro_features(df)
        self._eci_scaler = StandardScaler().fit(self._eci_input(dfm))
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self._dsi_scaler is None:
            raise RuntimeError("FeatureBuilder.fit() must be called before transform().")
        df = df.copy()
        for c in self._components:
            if c not in df.columns:
                df[c] = 0.0
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

        # DSI (signed z-score mean)
        log_df = df[self._components].apply(np.log1p)
        z = self._dsi_scaler.transform(log_df)
        df["DSI"] = z.mean(axis=1)

        # DSI_pos ∈ [0,1]; larger = more severe; monotone in every component
        span = max(self._dsi_max - self._dsi_min, 1e-9)
        df["DSI_pos"] = ((df["DSI"] - self._dsi_min) / span).clip(0.0, 1.0)

        # TWDI = DSI_pos × DTII  → always ≥ 0 (fixes sign-inversion for DSI < 0)
        df = compute_dtii(df, self._dtii)
        df["TWDI"] = df["DSI_pos"] * df["DTII"]

        df = add_seasonality(df)
        df = add_macro_features(df)
        df["Economic_Condition_Index"] = (
            self._eci_scaler.transform(self._eci_input(df)).mean(axis=1)
        )
        df = add_external_shock_flag(df, {"external_shocks": self._shock})
        return df

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    @staticmethod
    def _eci_input(df: pd.DataFrame) -> pd.DataFrame:
        eci = df[["gdp_growth", "inflation", "exr_lkr_usd"]].copy()
        eci["inflation"] = -eci["inflation"]
        eci["exr_lkr_usd"] = -eci["exr_lkr_usd"]
        return eci
