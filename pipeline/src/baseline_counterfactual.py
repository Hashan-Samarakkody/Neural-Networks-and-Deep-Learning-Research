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


def _baseline_seasonal_trend(vals: list[float]) -> float:
    """Log-linear trend fitted to K reference values; extrapolate one step."""
    if len(vals) < 2:
        return float(np.mean(vals)) if vals else 0.0
    log_vals = np.log(np.maximum(vals, 1e-6))
    xs = np.arange(len(log_vals), dtype=float)
    slope, intercept = np.polyfit(xs, log_vals, 1)
    return float(np.exp(intercept + slope * len(log_vals)))


def _baseline_yoy_growth(vals: list[float]) -> float:
    """Mean of values scaled by mean year-on-year growth rate."""
    if len(vals) < 2:
        return float(np.mean(vals)) if vals else 0.0
    ratios = [vals[i] / vals[i - 1] for i in range(1, len(vals)) if vals[i - 1] > 0]
    mean_growth = float(np.mean(ratios)) if ratios else 1.0
    return float(np.mean(vals)) * mean_growth


def compute_baseline(
    arrivals: pd.DataFrame,
    disaster_year: int,
    disaster_month: int,
    K: int = 3,
    crisis_years: set | None = None,
    method: str = "seasonal_mean",
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
        # Never use disaster_year or later as reference (R-03: no future leakage)
        y = min(target_year - 1, disaster_year - 1)
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
            return 0.0

        if method == "seasonal_trend":
            return _baseline_seasonal_trend(list(reversed(valid_years)))
        elif method == "yoy_growth":
            return _baseline_yoy_growth(list(reversed(valid_years)))
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
    method: str = "seasonal_mean",
    persistence: int = 1,
) -> tuple[int, dict, bool, dict]:
    """
    Compute T_rec for a single disaster event.

    Returns:
        (T_rec, baselines_dict, is_right_censored, extras)

    extras contains: depth, months_below_threshold, pre_event_ratio.
    T_rec = 0  if month immediately after disaster meets threshold
    T_rec = 12 if recovery not achieved within max_months (right-censored)
    persistence: consecutive months above threshold required (S-03).
    """
    if crisis_years is None:
        crisis_years = CRISIS_YEARS

    baselines = compute_baseline(
        arrivals, disaster_year, disaster_month, K=K, crisis_years=crisis_years, method=method,
    )

    arr_idx = arrivals.set_index(["year", "month"])["tourist_arrivals"]

    def get_arrivals(year, month):
        try:
            return float(arr_idx.loc[(year, month)])
        except KeyError:
            return np.nan

    pre_event = baselines.get(0, 0.0)
    event_month_seq = disaster_month - 1
    event_actual_month = (event_month_seq % 12) + 1
    event_actual_year = disaster_year + (event_month_seq // 12)
    actual_at_event = get_arrivals(event_actual_year, event_actual_month)
    pre_event_ratio = (
        float(actual_at_event / pre_event)
        if (pre_event > 0 and not pd.isna(actual_at_event))
        else np.nan
    )

    months_below = 0
    max_depth = 0.0
    consec = 0

    for t in range(1, max_months + 1):
        month_seq = disaster_month + t - 1
        target_month = (month_seq % 12) + 1
        target_year = disaster_year + (month_seq // 12)
        actual = get_arrivals(target_year, target_month)
        baseline = baselines.get(t, 0.0)

        if pd.isna(actual) or baseline <= 0:
            continue

        ratio = actual / baseline
        if ratio < threshold:
            months_below += 1
            depth = 1.0 - ratio
            if depth > max_depth:
                max_depth = depth
            consec = 0
        else:
            consec += 1
            if consec >= persistence:
                extras = {
                    "depth": round(max_depth, 4),
                    "months_below_threshold": months_below,
                    "pre_event_ratio": round(pre_event_ratio, 4) if not pd.isna(pre_event_ratio) else np.nan,
                }
                if t == 1 and consec == 1:
                    return 0, baselines, False, extras
                return t - consec, baselines, False, extras

    extras = {
        "depth": round(max_depth, 4),
        "months_below_threshold": months_below,
        "pre_event_ratio": round(pre_event_ratio, 4) if not pd.isna(pre_event_ratio) else np.nan,
    }
    return max_months, baselines, True, extras


def _count_reference_years(
    arrivals: pd.DataFrame,
    disaster_year: int,
    disaster_month: int,
    K: int,
    crisis_years: set,
    max_months: int = 12,
) -> int:
    """Return the minimum number of non-crisis reference years available across all offsets."""
    arr_idx = arrivals.set_index(["year", "month"])["tourist_arrivals"]
    min_count = K  # start at K; will be reduced if any offset has fewer

    for offset in range(1, max_months + 1):
        month_seq = disaster_month + offset - 1
        target_month = (month_seq % 12) + 1
        target_year = disaster_year + (month_seq // 12)

        count = 0
        y = target_year - 1
        while count < K:
            if y < arrivals["year"].min():
                break
            if y not in crisis_years:
                try:
                    val = arr_idx.loc[(y, target_month)]
                    if pd.notna(val) and val > 0:
                        count += 1
                except KeyError:
                    pass
            y -= 1
        min_count = min(min_count, count)

    return min_count


def build_recovery_targets(
    emdat: pd.DataFrame,
    arrivals: pd.DataFrame,
    cfg: dict,
) -> pd.DataFrame:
    """
    Compute T_rec for every disaster event in emdat.

    Returns emdat with added columns:
        recovery_months, pre_disaster_baseline, is_right_censored,
        n_reference_years_min, baseline_insufficient
    baseline_insufficient=True when fewer than baseline.min_reference_years
    non-crisis reference years exist for any post-disaster offset; those events
    get recovery_months=NaN and are dropped before modelling (R-03).
    """
    K = cfg["experiment"]["baseline_window_years"]
    threshold = cfg["experiment"]["recovery_threshold"]
    max_months = cfg["experiment"]["max_recovery_months"]
    min_ref = cfg.get("baseline", {}).get("min_reference_years", 2)
    method = cfg.get("baseline", {}).get("method", "seasonal_mean")
    persistence = cfg.get("recovery", {}).get("persistence_months", 1)

    shock_cfg = cfg["external_shocks"]
    crisis_years = set()
    for rng in shock_cfg.values():
        crisis_years.update(range(rng[0], rng[1] + 1))

    records = []
    for _, row in emdat.iterrows():
        year = int(row["start_year"])
        month = int(row["start_month"])

        n_ref = _count_reference_years(arrivals, year, month, K, crisis_years, max_months)
        insufficient = n_ref < min_ref

        if insufficient:
            t_rec, baselines, censored, extras = np.nan, {}, False, {}
        else:
            t_rec, baselines, censored, extras = compute_recovery_months(
                arrivals, year, month, K=K,
                threshold=threshold, max_months=max_months,
                crisis_years=crisis_years, method=method, persistence=persistence,
            )

        records.append({
            **row.to_dict(),
            "recovery_months": t_rec,
            "pre_disaster_baseline": baselines.get(0, np.nan),
            "is_right_censored": censored,
            "n_reference_years_min": n_ref,
            "baseline_insufficient": insufficient,
            "depth": extras.get("depth", np.nan),
            "months_below_threshold": extras.get("months_below_threshold", np.nan),
            "pre_event_ratio": extras.get("pre_event_ratio", np.nan),
        })

    return pd.DataFrame(records)
