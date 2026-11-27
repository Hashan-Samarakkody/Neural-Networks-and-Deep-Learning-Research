# Version Sensitivity

Authoritative environment (pinned in `requirements.txt`, enforced at start-up by `run_experiments.py`):
Python 3.12, pandas 2.2.2, numpy 2.0.2, scikit-learn 1.5.2, xgboost 2.1.1, shap 0.46.0, matplotlib 3.9.2,
scipy 1.14.1, scikit-survival 0.23.1. The pipeline aborts if any version differs (override: `ALLOW_VERSION_MISMATCH=1`;
the manifest then records `pinned_versions_ok: false`). The frozen results in `outputs/` come from a run in this
environment (see `run_manifest.json`; run time about 32 minutes on one CPU).

## Measured differences between two runs

Run 1 (earlier "frozen" run, commit 4489a73, Windows, xgboost 3.4.1, numpy 2.2.6, shap 0.52.0, scikit-learn 1.9.1) versus
Run 2 (pinned environment, Linux). Every model except XGBoost reproduced to four decimals; the models that use XGBoost differ:

| Quantity | Run 1 (xgboost 3.4.1) | Run 2 (pinned 2.1.1) |
|---|---|---|
| ExpA_LOOCV XGBoost MAE | 0.933 | 1.061 |
| ExpB_Chronological XGBoost MAE | 5.383 | 5.614 |

Holm-adjusted permutation p-values of XGBoost against each comparator also changed (for example against the random forest 0.58 in run 1 and 1.00
in run 2; against the two-stage model 1.00 and 0.94); the comparisons with the mean baseline, the last-known baseline and Ridge stayed significant.
SHAP values changed (global tourism index 0.53 versus 0.58); the top-ranked feature did not change.

## Rules

* Report results only from the pinned environment.
* XGBoost-dependent conclusions (its MAE, its SHAP values, tests that involve it) must not be stated without the library version.
* The statement in earlier versions of this file that XGBoost 3.x shifts MAE by less than 0.01 months was wrong.
