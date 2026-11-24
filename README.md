# Predicting Tourism Recovery Duration Following Natural Disasters in Sri Lanka

Regression pipeline for estimating the number of months for Sri Lanka tourist arrivals to recover to 90% of pre-disaster seasonal baseline after a physical natural disaster (2000–2025).

## Setup

```bash
cd pipeline
pip install -r requirements.txt
```

## Data

Place the following files in `pipeline/data/raw/`:

| File | Source |
|------|--------|
| `Disaster data.xlsx` | EM-DAT Sri Lanka physical disasters (2000–2025) |
| `tourist_arrivals.csv` | SLTDA monthly tourist arrivals — columns: `year`, `month`, `tourist_arrivals` |

## Run

```bash
cd pipeline
python run_experiments.py
```

Outputs are written to `pipeline/outputs/`:

| Path | Contents |
|------|----------|
| `outputs/results/experiment_results.csv` | MAE / RMSE / R² with 95% bootstrap CI for all models |
| `outputs/results/wilcoxon_tests.csv` | Wilcoxon signed-rank p-values |
| `outputs/results/target_distribution.json` | Recovery duration distribution statistics |
| `outputs/results/rq_summary.json` | RQ1–RQ4 answers |
| `outputs/results/sensitivity_analysis.csv` | 9-cell threshold × baseline-window grid |
| `outputs/results/shap_importance_ExpB_XGBoost.csv` | TreeSHAP feature ranking |
| `outputs/figures/` | SHAP beeswarm, dependence plots, correlation matrix, distribution histogram |

## Reproducibility

All experiments use `seed=42` (global `np.random.seed` and per-estimator `random_state`). Results are deterministic given the same input data.

## Experiments

| Experiment | Dataset | Validation | N (events) |
|------------|---------|------------|------------|
| A | Pure physical disasters (crisis years 2019–2022 excluded) | LOOCV | 51 |
| B | All events including crisis-era shocks | Chronological split (train ≤2018, test ≥2019) | 65 (42 train / 23 test) |

## Key Findings

- **Target distribution:** 58.5% of events have T_rec = 0 months (immediate recovery); median = 0; mean = 3.45; 20% right-censored at 12 months.
- **Exp A best ML model:** XGBoost (MAE = 1.83). NaivePrevious baseline beats all ML models (MAE = 1.49) due to zero-inflation.
- **Exp B best model:** Ridge (MAE = 2.51, R² = 0.69); high R² reflects External_Shock_Flag step-level effect in test period.
- **Statistical significance:** XGBoost significantly outperforms NaiveMean (p = 0.0001) but not NaiveMedian (p = 0.669).
- **Top SHAP features:** global_tourism_idx, Economic_Condition_Index, gdp_growth.

## Configuration

All experimental settings are in `pipeline/configs/model_config.yaml`:
- Recovery threshold: 90% of seasonal baseline (UNWTO benchmark)
- Baseline window: 3 non-crisis reference years
- Crisis years: 2019 (Easter attacks), 2020–2022 (COVID + economic crisis)
- Hyperparameter grids for XGBoost, RandomForest, SVR, Ridge

## Known Limitations

1. **Zero-inflation:** 58.5% zero-month recoveries. Two-stage classifier+regressor not yet implemented; ordinary regression is suboptimal for this target distribution.
2. **Small N:** N=51/23 makes bootstrap CIs wide. Findings are exploratory.
3. **Exp B conformal PI:** Empirical test-set coverage reported — not a valid conformal prediction guarantee (requires separate calibration partition).
4. **Static DTII:** District tourism importance shares are point-in-time; do not reflect hotel openings/closures 2000–2025.
