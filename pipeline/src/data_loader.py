"""
Data ingestion and validation for the tourism recovery pipeline.

Loads EM-DAT disaster events and SLTDA monthly tourist arrivals.
Applies hazard taxonomy filtering (excludes epidemics / biological events).
Clusters compound events within 30 days into single composite records.
"""

import warnings
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

warnings.filterwarnings("ignore")


def load_config(config_path: str | Path) -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_arrivals(path: str | Path) -> pd.DataFrame:
    """
    Load monthly tourist arrivals.

    Expected CSV columns: year (int), month (int), tourist_arrivals (int/float)
    Returns DataFrame indexed by (year, month) with column 'tourist_arrivals'.
    """
    df = pd.read_csv(path)
    required = {"year", "month", "tourist_arrivals"}
    missing = required - set(df.columns.str.lower())
    if missing:
        raise ValueError(f"tourist_arrivals.csv missing columns: {missing}")
    df.columns = df.columns.str.lower()
    df["year"] = df["year"].astype(int)
    df["month"] = df["month"].astype(int)
    df["tourist_arrivals"] = pd.to_numeric(df["tourist_arrivals"], errors="coerce")
    df = df.dropna(subset=["tourist_arrivals"])
    df = df.sort_values(["year", "month"]).reset_index(drop=True)
    return df


def load_emdat(path: str | Path, cfg: dict) -> pd.DataFrame:
    """
    Load and clean EM-DAT Sri Lanka disaster data.

    Applies:
    - Column rename / type coercion
    - Hazard taxonomy filter (physical disasters only)
    - Compound event clustering (events within 30-day window → single record)
    - Duration computation in days
    """
    raw = pd.read_excel(path, sheet_name="EM-DAT Data", header=0)
    raw.columns = raw.columns.str.strip()

    col_map = {
        "DisNo.": "disaster_id",
        "Disaster Type": "disaster_type",
        "Disaster Subtype": "disaster_subtype",
        "Start Year": "start_year",
        "Start Month": "start_month",
        "Start Day": "start_day",
        "End Year": "end_year",
        "End Month": "end_month",
        "End Day": "end_day",
        "Total Deaths": "deaths",
        "No. Injured": "injured",
        "No. Affected": "affected",
        "Total Affected": "total_affected",
        "Total Damage ('000 US$)": "damage_usd_000",
        "Location": "location",
        "CPI": "cpi",
        "Disaster Subgroup": "disaster_subgroup",
    }
    df = raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns})

    numeric_cols = [
        "start_year", "start_month", "start_day",
        "end_year", "end_month", "end_day",
        "deaths", "injured", "affected", "total_affected", "damage_usd_000",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["start_year"] = df["start_year"].astype("Int64")
    df["start_month"] = df["start_month"].fillna(1).astype("Int64")
    df = df[df["start_year"].notna() & (df["start_year"] >= 2000)]

    physical_types = set(cfg["hazard_taxonomy"]["physical_types"])
    df = df[df["disaster_type"].isin(physical_types)].copy()

    df["deaths"] = df["deaths"].fillna(0)
    df["injured"] = df["injured"].fillna(0)
    df["total_affected"] = df["total_affected"].fillna(df["affected"].fillna(0))
    df["damage_usd_000"] = df["damage_usd_000"].fillna(0)

    df = _compute_duration(df)
    df = _cluster_compound_events(df, cfg["hazard_taxonomy"]["compound_cluster_days"])
    df = df.reset_index(drop=True)
    return df


def _compute_duration(df: pd.DataFrame) -> pd.DataFrame:
    def safe_date(row, prefix):
        try:
            y = int(row[f"{prefix}_year"]) if pd.notna(row[f"{prefix}_year"]) else None
            m = int(row[f"{prefix}_month"]) if pd.notna(row[f"{prefix}_month"]) else 1
            d = int(row[f"{prefix}_day"]) if pd.notna(row.get(f"{prefix}_day", np.nan)) else 1
            if y is None:
                return None
            m = max(1, min(12, m))
            d = max(1, min(28, d))
            return pd.Timestamp(y, m, d)
        except Exception:
            return None

    df["start_date"] = df.apply(lambda r: safe_date(r, "start"), axis=1)
    df["end_date"] = df.apply(lambda r: safe_date(r, "end"), axis=1)

    def duration_days(row):
        if row["start_date"] is None:
            return 1
        end = row["end_date"] if row["end_date"] is not None else row["start_date"]
        delta = (end - row["start_date"]).days + 1
        return max(1, delta)

    df["duration_days"] = df.apply(duration_days, axis=1)
    return df


def _cluster_compound_events(df: pd.DataFrame, window_days: int) -> pd.DataFrame:
    """
    Events starting within `window_days` of each other → merged into single record.
    Summed: deaths, injured, total_affected, damage_usd_000.
    Kept from first event: disaster_id, disaster_type, start_date, start_year, start_month.
    Duration spans first start to last end.
    """
    df = df.sort_values("start_date").reset_index(drop=True)
    clusters = []
    used = set()

    for i, row in df.iterrows():
        if i in used:
            continue
        group = [i]
        ref_date = row["start_date"]
        for j, other in df.iloc[i + 1:].iterrows():
            if j in used:
                continue
            if other["start_date"] is None:
                continue
            if (other["start_date"] - ref_date).days <= window_days:
                group.append(j)
                used.add(j)
        used.add(i)

        sub = df.loc[group]
        merged = {
            "disaster_id": sub.iloc[0]["disaster_id"],
            "disaster_type": sub.iloc[0]["disaster_type"],
            "disaster_subtype": sub.iloc[0].get("disaster_subtype", ""),
            "start_date": sub["start_date"].min(),
            "start_year": int(sub.iloc[0]["start_year"]),
            "start_month": int(sub.iloc[0]["start_month"]),
            "location": "; ".join(sub["location"].dropna().unique()),
            "deaths": sub["deaths"].sum(),
            "injured": sub["injured"].sum(),
            "total_affected": sub["total_affected"].sum(),
            "damage_usd_000": sub["damage_usd_000"].sum(),
            "duration_days": (
                (sub["end_date"].max() - sub["start_date"].min()).days + 1
                if sub["end_date"].notna().any() else sub["duration_days"].max()
            ),
            "n_events_clustered": len(group),
        }
        clusters.append(merged)

    return pd.DataFrame(clusters)


def validate_data(emdat: pd.DataFrame, arrivals: pd.DataFrame) -> None:
    """Raise if critical validation checks fail."""
    if emdat.empty:
        raise ValueError("EM-DAT physical disaster dataframe is empty after filtering.")
    if arrivals.empty:
        raise ValueError("Tourist arrivals dataframe is empty.")

    min_arr = arrivals["year"].min()
    max_arr = arrivals["year"].max()
    min_dis = emdat["start_year"].min()
    max_dis = emdat["start_year"].max()

    if min_dis < min_arr or max_dis > max_arr:
        print(
            f"[WARN] Arrivals span {min_arr}-{max_arr}; "
            f"disasters span {min_dis}-{max_dis}. "
            "Some events may lack coverage."
        )

    missing_arrivals = arrivals["tourist_arrivals"].isna().sum()
    if missing_arrivals > 0:
        print(f"[WARN] {missing_arrivals} missing monthly arrival values.")
