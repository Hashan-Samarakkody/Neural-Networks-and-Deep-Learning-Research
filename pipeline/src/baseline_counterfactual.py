"""
Seasonally adjusted counterfactual arrival baseline.

Formula (Section 2.1 of remediation spec):
    A_base(m, t0) = (1/K) * sum_{k=1}^{K} A_{y-k, m}

where:
    m       = calendar month of interest (t0 + t)
    y       = disaster year
    K       = number of non-crisis reference years (default 3)
    A_{y,m} = tourist arrivals in year y, month m

A year is considered a "crisis year" if the external_shock_flag is active in
any month of that year (COVID, Easter attacks, economic crisis).

Recovery target (Section 2.2):
    T_rec = min({t in {1,...,12} : A_{t0+t} >= 0.90 * A_base(m_{t0+t})} ∪ {12})

Special case:
    If A_{t0+1} >= 0.90 * A_base(m_{t0+1}), then T_rec = 0
    (arrival barely dropped; event had minimal tourism impact)
"""

import numpy as np
import pandas as pd


CRISIS_YEARS = {2019, 2020, 2021, 2022}


def compute_baseline(
    arrivals: pd.DataFrame,
    disaster_year: int,
    disaster_month: int,
    K: int = 3,
    crisis_years: set | None = None,
) -> dict[int, float]:
    """
    Compute the same-month historical counterfactual for each post-disaster month.

    Returns a dict: {month_offset (1..12): baseline_arrivals}
    month_offset 0 = disaster month itself (used for T_rec = 0 check).
    """
    if crisis_years is None:
        crisis_years = CRISIS_YEARS

    arr_idx = arrivals.set_index(["year", "month"])["tourist_arrivals"]

    def _baseline_for_month(target_month: int, target_year: int) -> float:
        valid_years = []
        y = target_year - 1
        while len(valid_years) < K:
            if y < arrivals["year"].min():
                break
            if y not in crisis_years:
                try:
                    val = arr_idx.loc[(y, target_month)]
                    if pd.notna(val) and val > 0:
                        valid_years.append(val)
                except KeyError:
                    pass
            y -= 1

        if not valid_years:
            all_month = arrivals[arrivals["month"] == target_month]["tourist_arrivals"]
            return float(all_month.mean()) if len(all_month) > 0 else 0.0

        return float(np.mean(valid_years))

    baselines = {}
    for offset in range(0, 13):
        month_seq = disaster_month + offset - 1
        target_month = (month_seq % 12) + 1
        target_year = disaster_year + (month_seq // 12)
        baselines[offset] = _baseline_for_month(target_month, target_year)

    return baselines


def compute_recovery_months(
    arrivals: pd.DataFrame,
    disaster_year: int,
    disaster_month: int,
    K: int = 3,
    threshold: float = 0.90,
    max_months: int = 12,
    crisis_years: set | None = None,
) -> tuple[int, dict, bool]:
    """
    Compute T_rec for a single disaster event.

    Returns:
        (T_rec, baselines_dict, is_right_censored)

    T_rec = 0  if month immediately after disaster meets threshold
    T_rec = 12 if recovery not achieved within max_months (right-censored)
    """
    if crisis_years is None:
        crisis_years = CRISIS_YEARS

    baselines = compute_baseline(
        arrivals, disaster_year, disaster_month, K=K, crisis_years=crisis_years
    )

    arr_idx = arrivals.set_index(["year", "month"])["tourist_arrivals"]

    def get_arrivals(year, month):
        try:
            return float(arr_idx.loc[(year, month)])
        except KeyError:
            return np.nan

    for t in range(1, max_months + 1):
        month_seq = disaster_month + t - 1
        target_month = (month_seq % 12) + 1
        target_year = disaster_year + (month_seq // 12)
        actual = get_arrivals(target_year, target_month)
        baseline = baselines.get(t, 0.0)

        if pd.isna(actual) or baseline <= 0:
            continue

        if actual >= threshold * baseline:
            if t == 1:
                return 0, baselines, False
            return t - 1, baselines, False

    return max_months, baselines, True


def build_recovery_targets(
    emdat: pd.DataFrame,
    arrivals: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """
    Compute T_rec for every disaster event in emdat.

    Returns emdat with added columns:
        recovery_months, pre_disaster_baseline, is_right_censored
    """
    K = cfg["experiment"]["baseline_window_years"]
    threshold = cfg["experiment"]["recovery_threshold"]
    max_months = cfg["experiment"]["max_recovery_months"]

    shock_cfg = cfg["external_shocks"]
    crisis_years = set()
    for rng in shock_cfg.values():
        crisis_years.update(range(rng[0], rng[1] + 1))

    records = []
    for _, row in emdat.iterrows():
        year = int(row["start_year"])
        month = int(row["start_month"])

        t_rec, baselines, censored = compute_recovery_months(
            arrivals, year, month, K=K,
            threshold=threshold, max_months=max_months,
            crisis_years=crisis_years,
        )

        records.append({
            **row.to_dict(),
            "recovery_months": t_rec,
            "pre_disaster_baseline": baselines.get(0, np.nan),
            "is_right_censored": censored,
        })

    return pd.DataFrame(records)
