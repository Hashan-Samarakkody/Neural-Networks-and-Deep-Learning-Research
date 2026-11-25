"""
Confounding window detection and sensitivity analysis (S-01).

Two known confounding periods distort recovery signals:
  civil_war_end (2007-2009): post-conflict normalisation inflates recovery speed.
  easter_covid_crisis (2019-2022): already excluded from Exp A via external_shocks;
      included here for completeness so confounded_by is populated for all events.

Three experiment variants:
  all        - full Exp A sample (N=46, crisis-year events already excluded)
  excluded   - Exp A minus civil_war_end-flagged events
  flagged_only - civil_war_end events only (sensitivity check)
"""

import numpy as np
import pandas as pd
from pathlib import Path


def flag_confounded_events(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    windows = cfg.get("confounding_windows", {})
    df = df.copy()

    def _flags(year):
        hits = [name for name, (lo, hi) in windows.items() if lo <= int(year) <= hi]
        return ",".join(hits) if hits else ""

    df["confounded_by"] = df["start_year"].apply(_flags)
    return df


def run_confounding_variants(
    df_A: pd.DataFrame,
    X_A: np.ndarray,
    y_A: np.ndarray,
    cfg: dict,
    model_class,
    best_params: dict,
    results_dir: Path,
    seed: int = 42,
) -> pd.DataFrame:
    from sklearn.model_selection import LeaveOneOut
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    df_flagged = flag_confounded_events(df_A, cfg)
    is_civil_war = df_flagged["confounded_by"].str.contains("civil_war_end", na=False)

    variants = {
        "all": np.ones(len(df_A), dtype=bool),
        "excluded": (~is_civil_war).values,
        "flagged_only": is_civil_war.values,
    }

    rows = []
    for name, mask in variants.items():
        X_v = X_A[mask]
        y_v = y_A[mask]
        n = len(y_v)

        if n < 5:
            rows.append({
                "variant": name,
                "n_events": n,
                "MAE": np.nan, "RMSE": np.nan, "R2": np.nan,
                "note": "too_few_events",
            })
            continue

        loo = LeaveOneOut()
        preds, trues = [], []
        for tr, te in loo.split(X_v):
            m = model_class(**best_params, random_state=seed)
            m.fit(X_v[tr], y_v[tr])
            preds.append(float(np.clip(m.predict(X_v[te])[0], 0, 12)))
            trues.append(float(y_v[te[0]]))

        yt, yp = np.array(trues), np.array(preds)
        rows.append({
            "variant": name,
            "n_events": n,
            "MAE": round(float(mean_absolute_error(yt, yp)), 4),
            "RMSE": round(float(np.sqrt(mean_squared_error(yt, yp))), 4),
            "R2": round(float(r2_score(yt, yp)), 4),
            "note": "",
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(results_dir / "confounding_sensitivity.csv", index=False)
    return df_out
