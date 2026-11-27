# Version Sensitivity

Frozen run commit: `e3225d1` (branch `master`, tag `paper-results-v3` pending).

## Pinned environment

| Package | Frozen version |
|---------|---------------|
| xgboost | 2.1.1 |
| numpy | 2.0.2 |
| shap | 0.46.0 |
| scikit-learn | 1.5.2 |
| pandas | 2.2.2 |
| scipy | 1.14.1 |
| scikit-survival | >=0.23.0 |

Install with:

```bash
pip install -r pipeline/requirements.txt
```

## Known sensitivities

**XGBoost 2.x vs 3.x** — Default `nthread` handling changed between versions. The
pipeline forces `nthread=1` explicitly to ensure deterministic output regardless of
CPU count. Upgrading to 3.x may shift leaf values by < 0.01 months MAE.

**NumPy 2.x** — SHAP 0.46.0 uses internal NumPy APIs that differ from 0.52.x. Using
shap>=0.50 on the same model weights may change SHAP values by ±0.01. The final ranking
is stable.

**Reproducibility guarantee** — With the pinned environment above, `seed=42`, and the
data files in `pipeline/data/raw/`, running `python pipeline/run_experiments.py` should
reproduce all numbers in `pipeline/outputs/results/` to 4 decimal places.
