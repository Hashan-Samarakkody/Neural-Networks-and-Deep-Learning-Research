"""
S-10: Extract all in-text paper statistics to paper_numbers.json.

All numbers reported in the paper text, tables, and figures are sourced
exclusively from this module so they remain reproducible from outputs/.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path


def extract_paper_numbers(results_dir: Path) -> dict:
    results_dir = Path(results_dir)
    numbers = {}

    # Target distribution
    td_path = results_dir / "target_distribution.json"
    if td_path.exists():
        numbers["target_distribution"] = json.loads(td_path.read_text())

    # Experiment results
    er_path = results_dir / "experiment_results.csv"
    if er_path.exists():
        er = pd.read_csv(er_path)
        ml_models = ["XGBoost", "RandomForest", "SVR", "Ridge", "TwoStage"]

        def _best(exp_label, metric="MAE"):
            sub = er[(er["Experiment"] == exp_label) & (er["Model"].isin(ml_models))]
            if sub.empty:
                return {}
            row = sub.sort_values(metric).iloc[0]
            return row.to_dict()

        numbers["exp_A_best"] = _best("ExpA_LOOCV")
        numbers["exp_B_best"] = _best("ExpB_Chronological")

        for exp in er["Experiment"].unique():
            numbers.setdefault("all_experiments", {})[exp] = (
                er[er["Experiment"] == exp][["Model", "MAE", "RMSE", "R2"]]
                .to_dict(orient="records")
            )

    # Wilcoxon / permutation tests
    wt_path = results_dir / "wilcoxon_tests.csv"
    if wt_path.exists():
        wt = pd.read_csv(wt_path)
        numbers["permutation_tests"] = wt.to_dict(orient="records")

    # Survival
    sv_path = results_dir / "survival_results.csv"
    if sv_path.exists():
        numbers["survival"] = pd.read_csv(sv_path).to_dict(orient="records")

    # Event flow
    ef_path = results_dir / "event_flow.json"
    if ef_path.exists():
        numbers["event_flow"] = json.loads(ef_path.read_text())

    # Sensitivity
    sa_path = results_dir / "sensitivity_analysis.csv"
    if sa_path.exists():
        sa = pd.read_csv(sa_path)
        main = sa[sa.get("is_main_setting", pd.Series([False] * len(sa), dtype=bool))]
        numbers["sensitivity_main"] = main.to_dict(orient="records")
        numbers["sensitivity_full"] = sa.to_dict(orient="records")

    # Baseline comparison
    bm_path = results_dir / "baseline_method_comparison.csv"
    if bm_path.exists():
        numbers["baseline_comparison"] = pd.read_csv(bm_path).to_dict(orient="records")

    # Confounding
    cf_path = results_dir / "confounding_sensitivity.csv"
    if cf_path.exists():
        numbers["confounding_variants"] = pd.read_csv(cf_path).to_dict(orient="records")

    # Statistics derived from data and saved predictions (paper_extras.py)
    for key, fname, kind in (
        ("paper_extras", "paper_extras.json", "json"),
        ("stage1", "two_stage_stage1.json", "json"),
        ("run_manifest", "run_manifest.json", "json"),
        ("ablation", "ablation_features.csv", "csv"),
        ("macro_lag", "macro_lag_sensitivity.csv", "csv"),
        ("permutation_importance_ExpA", "permutation_importance_ExpA_XGBoost.csv", "csv"),
        ("permutation_importance_ExpB", "permutation_importance_ExpB_XGBoost.csv", "csv"),
        ("grouped_importance", "grouped_importance.csv", "csv"),
        ("vif", "vif.csv", "csv"),
        ("shap_ExpB_XGBoost", "shap_importance_ExpB_XGBoost.csv", "csv"),
    ):
        pth = results_dir / fname
        if pth.exists():
            numbers[key] = json.loads(pth.read_text()) if kind == "json" else pd.read_csv(pth).to_dict(orient="records")

    out_path = results_dir / "paper_numbers.json"
    out_path.write_text(json.dumps(numbers, indent=2, default=str))
    return numbers


if __name__ == "__main__":
    import sys
    rdir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("outputs/results")
    nums = extract_paper_numbers(rdir)
    print(f"paper_numbers.json written with {len(nums)} top-level keys")
