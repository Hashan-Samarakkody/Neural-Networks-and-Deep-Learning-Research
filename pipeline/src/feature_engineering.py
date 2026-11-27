"""
Feature engineering for the tourism recovery pipeline.

Implements (per Section 2 of the remediation spec):

2.3  Log-Standardized Disaster Severity Index (DSI)
     x'_{i,k} = ln(1 + x_{i,k})
     z_{i,k}  = (x'_{i,k} - mu_k) / sigma_k
     DSI_i    = (1/4) * sum_k z_{i,k}

2.4  Tourism-Weighted Disaster Impact (TWDI)
     TWDI_i = DSI_i * DTII_j
     DTII_j = Rooms_j / sum_m Rooms_m

2.5  Harmonic Trigonometric Seasonality
     Season_sin = sin(2*pi*m / 12)
     Season_cos = cos(2*pi*m / 12)

Additional features:
     Global_Tourism_Index  - World Bank outbound tourism proxy (annual, interpolated)
     GDP_growth            - Sri Lanka GDP growth rate (annual %)
     Economic_Condition_Index - composite of GDP + exchange rate + inflation
     External_Shock_Flag   - binary: 1 if event overlaps with Easter/COVID/crisis
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler


DSI_COMPONENTS = ["total_affected", "deaths", "damage_usd_000", "duration_days"]

WORLD_BANK_ANNUAL = {
    2000: {"gdp_growth": 6.0,  "inflation": 6.2,  "exr_lkr_usd": 76.7,  "global_tourism_idx": 100.0},
    2001: {"gdp_growth": -1.5, "inflation": 14.2, "exr_lkr_usd": 89.4,  "global_tourism_idx": 93.5},
    2002: {"gdp_growth": 4.0,  "inflation": 9.6,  "exr_lkr_usd": 95.7,  "global_tourism_idx": 97.2},
    2003: {"gdp_growth": 5.9,  "inflation": 6.3,  "exr_lkr_usd": 96.5,  "global_tourism_idx": 102.1},
    2004: {"gdp_growth": 5.4,  "inflation": 9.0,  "exr_lkr_usd": 101.2, "global_tourism_idx": 110.3},
    2005: {"gdp_growth": 6.2,  "inflation": 11.0, "exr_lkr_usd": 100.5, "global_tourism_idx": 119.8},
    2006: {"gdp_growth": 7.7,  "inflation": 13.7, "exr_lkr_usd": 103.9, "global_tourism_idx": 127.5},
    2007: {"gdp_growth": 6.8,  "inflation": 15.8, "exr_lkr_usd": 110.6, "global_tourism_idx": 133.0},
    2008: {"gdp_growth": 6.0,  "inflation": 22.4, "exr_lkr_usd": 108.3, "global_tourism_idx": 125.4},
    2009: {"gdp_growth": 3.5,  "inflation": 3.5,  "exr_lkr_usd": 114.9, "global_tourism_idx": 119.6},
    2010: {"gdp_growth": 8.0,  "inflation": 5.9,  "exr_lkr_usd": 113.0, "global_tourism_idx": 135.4},
    2011: {"gdp_growth": 8.4,  "inflation": 6.7,  "exr_lkr_usd": 110.6, "global_tourism_idx": 142.1},
    2012: {"gdp_growth": 9.1,  "inflation": 7.5,  "exr_lkr_usd": 127.6, "global_tourism_idx": 148.7},
    2013: {"gdp_growth": 3.4,  "inflation": 6.9,  "exr_lkr_usd": 130.6, "global_tourism_idx": 153.2},
    2014: {"gdp_growth": 5.0,  "inflation": 2.9,  "exr_lkr_usd": 130.6, "global_tourism_idx": 159.4},
    2015: {"gdp_growth": 5.0,  "inflation": 0.9,  "exr_lkr_usd": 135.9, "global_tourism_idx": 163.8},
    2016: {"gdp_growth": 4.5,  "inflation": 4.0,  "exr_lkr_usd": 145.5, "global_tourism_idx": 168.1},
    2017: {"gdp_growth": 3.6,  "inflation": 6.6,  "exr_lkr_usd": 152.5, "global_tourism_idx": 174.5},
    2018: {"gdp_growth": 3.3,  "inflation": 4.3,  "exr_lkr_usd": 162.5, "global_tourism_idx": 178.2},
    2019: {"gdp_growth": 2.3,  "inflation": 4.3,  "exr_lkr_usd": 178.8, "global_tourism_idx": 180.0},
    2020: {"gdp_growth": -3.6, "inflation": 4.6,  "exr_lkr_usd": 185.5, "global_tourism_idx": 72.0},
    2021: {"gdp_growth": 3.3,  "inflation": 6.0,  "exr_lkr_usd": 198.9, "global_tourism_idx": 80.5},
    2022: {"gdp_growth": -7.8, "inflation": 49.7, "exr_lkr_usd": 359.4, "global_tourism_idx": 112.3},
    2023: {"gdp_growth": -2.3, "inflation": 12.0, "exr_lkr_usd": 328.8, "global_tourism_idx": 145.2},
    2024: {"gdp_growth": 4.4,  "inflation": 2.5,  "exr_lkr_usd": 302.0, "global_tourism_idx": 158.0},
    2025: {"gdp_growth": 4.8,  "inflation": 3.0,  "exr_lkr_usd": 295.0, "global_tourism_idx": 165.0},
}


_MACRO_CSV_PATH = Path(__file__).parent.parent / "data" / "raw" / "macro_indicators.csv"

# Cache loaded from CSV; falls back to WORLD_BANK_ANNUAL if CSV not present.
_macro_cache: dict | None = None


def _load_macro_cache() -> dict:
    global _macro_cache
    if _macro_cache is not None:
        return _macro_cache
    if _MACRO_CSV_PATH.exists():
        raw = pd.read_csv(_MACRO_CSV_PATH)
        cache = {}
        for _, row in raw.iterrows():
            yr = int(row["year"])
            if yr not in cache:
                cache[yr] = {}
            cache[yr][row["indicator"]] = float(row["value"])
        # Validate expected indicators exist
        required = {"gdp_growth", "inflation", "exr_lkr_usd", "global_tourism_idx"}
        sample_yr = next(iter(cache))
        missing = required - set(cache[sample_yr].keys())
        if missing:
            raise ValueError(f"macro_indicators.csv missing indicators: {missing}")
        _macro_cache = cache
    else:
        # Fallback to hardcoded dict (remove when CSV is complete)
        _macro_cache = WORLD_BANK_ANNUAL
    return _macro_cache


def get_macro_features(year: int) -> dict:
    """
    R-06: Return macro-economic features for a given year.
    Loads from data/raw/macro_indicators.csv; raises on missing year.
    """
    cache = _load_macro_cache()
    if year in cache:
        return cache[year]
    raise KeyError(
        f"No macro indicators for year {year} in macro_indicators.csv. "
        "Add the year or extend the CSV."
    )


def compute_dsi(df: pd.DataFrame) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Compute log1p-standardized DSI for all events.
    Returns (df_with_dsi, fitted_scaler).
    """
    df = df.copy()
    for col in DSI_COMPONENTS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    log_df = df[DSI_COMPONENTS].apply(lambda s: np.log1p(s))

    scaler = StandardScaler()
    z_arr = scaler.fit_transform(log_df)
    z_df = pd.DataFrame(z_arr, columns=[f"z_{c}" for c in DSI_COMPONENTS], index=df.index)

    df["DSI"] = z_df.mean(axis=1)
    return df, scaler


def compute_dtii(df: pd.DataFrame, dtii_cfg: dict) -> pd.DataFrame:
    """
    R-17: Map each event's location string to DTII_j (district tourism share).
    Uses word-boundary regex matching. Falls back to mean share and logs misses.
    Reports matched_district and dtii_rule columns for transparency.
    """
    import re
    dtii_map = {k.lower(): v for k, v in dtii_cfg.items()}
    mean_share = float(np.mean(list(dtii_cfg.values())))
    _unmatched_log = []

    def _max_dtii(location_str: str) -> tuple[float, str]:
        if not isinstance(location_str, str) or not location_str.strip():
            _unmatched_log.append(location_str)
            return mean_share, "mean_fallback"
        loc_lower = location_str.lower()
        best_score = -1.0
        best_district = None
        for district, score in dtii_map.items():
            pattern = r"\b" + re.escape(district) + r"\b"
            if re.search(pattern, loc_lower):
                if score > best_score:
                    best_score = score
                    best_district = district
        if best_district is not None:
            return best_score, best_district
        _unmatched_log.append(location_str)
        return mean_share, "mean_fallback"

    df = df.copy()
    results = df["location"].apply(_max_dtii)
    df["DTII"] = results.apply(lambda x: x[0])
    df["dtii_rule"] = results.apply(lambda x: x[1])
    if _unmatched_log:
        import warnings
        warnings.warn(
            f"DTII: {len(_unmatched_log)} events fell back to mean share "
            f"(no district matched): {_unmatched_log[:5]}"
        )
    return df


def compute_twdi(df: pd.DataFrame) -> pd.DataFrame:
    """Multiply DSI by DTII to get Tourism-Weighted Disaster Impact (TWDI_i = DSI_i * DTII_j)."""
    df = df.copy()
    df["TWDI"] = df["DSI"] * df["DTII"]
    return df


def add_seasonality(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode calendar month as orthogonal harmonic coordinates.
    Avoids ordinality assumption of raw month integer.
    """
    df = df.copy()
    m = df["start_month"].astype(float)
    df["Season_sin"] = np.sin(2 * np.pi * m / 12)
    df["Season_cos"] = np.cos(2 * np.pi * m / 12)
    return df


def add_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Join annual World Bank macro indicators (GDP growth, inflation, exchange rate,
    global tourism index) to each event row by disaster year.
    """
    df = df.copy()
    rows = []
    for _, row in df.iterrows():
        macro = get_macro_features(int(row["start_year"]))
        rows.append(macro)
    macro_df = pd.DataFrame(rows, index=df.index)
    return pd.concat([df, macro_df], axis=1)


def add_economic_condition_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    ECI = standardized composite of GDP_growth, -inflation, -exr_lkr_usd.
    Negative signs because high inflation and high exchange rate depress tourism recovery.
    """
    df = df.copy()
    eci_components = df[["gdp_growth", "inflation", "exr_lkr_usd"]].copy()
    eci_components["inflation"] = -eci_components["inflation"]
    eci_components["exr_lkr_usd"] = -eci_components["exr_lkr_usd"]
    scaler = StandardScaler()
    z = scaler.fit_transform(eci_components)
    df["Economic_Condition_Index"] = z.mean(axis=1)
    return df


def add_external_shock_flag(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    Binary flag: 1 if event year overlaps with a systemic external shock period
    (Easter attacks 2019, COVID 2020-2021, economic crisis 2022).
    Shock year ranges come from config['external_shocks'].
    """
    shock_cfg = cfg["external_shocks"]
    crisis_years = set()
    for rng in shock_cfg.values():
        crisis_years.update(range(rng[0], rng[1] + 1))
    df = df.copy()
    df["external_shock_flag"] = df["start_year"].apply(
        lambda y: 1 if int(y) in crisis_years else 0
    )
    return df


def add_post_crisis_recovery_flag(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """
    R-12: post_crisis_recovery flag for events whose T_rec window overlaps a crisis tail.

    post_crisis_recovery=1 when:
      (a) event year is in config post_crisis_tail_years, OR
      (b) event year is in the last year of a crisis block and its onset + 12 months
          extends into a crisis year (e.g. 2018 events reaching into 2019 Easter attacks).

    Does not modify external_shock_flag; this is a second flag for sensitivity analysis.
    """
    df = df.copy()

    shock_cfg = cfg.get("external_shocks", {})
    crisis_years = set()
    for rng_pair in shock_cfg.values():
        crisis_years.update(range(rng_pair[0], rng_pair[1] + 1))

    # post_crisis_tail_years: years in the recovery tail after the crisis block
    tail_years = set(cfg.get("post_crisis_tail_years", []))

    def _flag(row) -> int:
        y = int(row["start_year"])
        m = int(row.get("start_month", 1))
        # Direct tail years (e.g. 2023)
        if y in tail_years:
            return 1
        # Window overlap: event's 12-month window reaches a crisis year
        for offset in range(1, 13):
            month_seq = m + offset - 1
            target_year = y + (month_seq // 12)
            if target_year in crisis_years:
                return 1
        return 0

    df["post_crisis_recovery"] = df.apply(_flag, axis=1)
    return df


FEATURE_COLS = [
    "DSI",
    "TWDI",
    "Season_sin",
    "Season_cos",
    "global_tourism_idx",
    "gdp_growth",
    "Economic_Condition_Index",
    "external_shock_flag",
]


def build_feature_matrix(df: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, pd.Series]:
    """
    Full feature engineering pipeline. Returns (X, y).

    X: DataFrame with FEATURE_COLS
    y: Series of recovery_months
    """
    df, _ = compute_dsi(df)
    df = compute_dtii(df, cfg["dtii"])
    df = compute_twdi(df)
    df = add_seasonality(df)
    df = add_macro_features(df)
    df = add_economic_condition_index(df)
    df = add_external_shock_flag(df, cfg)

    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing engineered features: {missing}")

    X = df[FEATURE_COLS].copy()
    y = df["recovery_months"].astype(float)
    return X, y
