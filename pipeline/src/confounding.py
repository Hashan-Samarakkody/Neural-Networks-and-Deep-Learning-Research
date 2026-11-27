"""
Confounding windows and sensitivity variants.

Windows (config confounding_windows):
  civil_war_end       2007-2009  conflict period and its end
  easter_covid_crisis 2019-2022  already excluded from Experiment A
Post-crisis tail (config post_crisis_tail_years, default 2023): events whose recovery
window follows the 2022 crisis.

Variants (default-setting XGBoost LOOCV, fold-safe features, with the LOO median baseline):
  all                          Experiment A sample
  excluded_civil_war           without civil-war-window events
  excluded_civil_war_and_tail  without civil-war-window and post-crisis-tail events
  flagged_civil_war            civil-war-window events only
  flagged_tail                 post-crisis-tail events only
"""

import numpy as np
import pandas as pd
from pathlib import Path

from default_models import loocv_default_xgb, loo_zero_baseline


def flag_confounded_events(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    windows = cfg.get("confounding_windows", {})
    tail = set(cfg.get("post_crisis_tail_years", []))
    df = df.copy()

    def _flags(year):
        hits = [name for name, (lo, hi) in windows.items() if lo <= int(year) <= hi]
        if int(year) in tail:
            hits.append("post_crisis_tail")
        return ",".join(hits) if hits else ""

    df["confounded_by"] = df["start_year"].apply(_flags)
    df["post_crisis_recovery"] = df["start_year"].apply(lambda y: int(int(y) in tail))
    return df


def run_confounding_variants(df_A: pd.DataFrame, cfg: dict, results_dir: Path, seed: int = 42) -> pd.DataFrame:
    df = flag_confounded_events(df_A, cfg).reset_index(drop=True)
    cw = df["confounded_by"].str.contains("civil_war_end", na=False).values
    tail = df["confounded_by"].str.contains("post_crisis_tail", na=False).values
    variants = {
        "all": np.ones(len(df), dtype=bool),
        "excluded_civil_war": ~cw,
        "excluded_civil_war_and_tail": ~(cw | tail),
        "flagged_civil_war": cw,
        "flagged_tail": tail,
    }
    rows = []
    for name, mask in variants.items():
        sub = df[mask].reset_index(drop=True)
        if len(sub) < 6:
            rows.append({"variant": name, "n_events": int(len(sub)), "note": "too few events"})
            continue
        r = loocv_default_xgb(sub, cfg, seed=seed)
        z = loo_zero_baseline(sub["recovery_months"].values)
        rows.append({
            "variant": name, "n_events": int(len(sub)),
            "zero_share": round(float((sub["recovery_months"] == 0).mean()), 4),
            "MAE": r["MAE"], "RMSE": r["RMSE"], "R2": r["R2"],
            "MAE_median_baseline": z["MAE"], "note": "",
        })
    out = pd.DataFrame(rows)
    results_dir = Path(results_dir); results_dir.mkdir(parents=True, exist_ok=True)
    out.to_csv(results_dir / "confounding_sensitivity.csv", index=False)
    return out
