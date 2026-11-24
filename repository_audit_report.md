# MASTER PROMPT 1: COMPLETE REPOSITORY AUDIT REPORT
## Tourism Recovery Duration Prediction — Sri Lanka

**Audit Date:** 2026-11-24  
**Auditors:** 5-expert panel (ML Engineer, ML Professor, Tourism Economics, Disaster Risk, Statistician)  
**Pipeline Version:** post-remediation (seed=42, N=65 events, 8 features)  
**Scope:** All 11 Python source files, 1 YAML config, all CSV/JSON outputs

---

## PHASE 1: COMPLETE REPOSITORY INSPECTION

### File Inventory

| File | Lines | Status |
|------|-------|--------|
| `src/data_loader.py` | 207 | Clean |
| `src/baseline_counterfactual.py` | 170 | Clean, 1 minor issue |
| `src/feature_engineering.py` | 214 | Clean |
| `src/train.py` | 245 | 1 moderate issue |
| `src/evaluate.py` | 218 | 2 issues |
| `src/validation.py` | 59 | Clean |
| `src/explainability.py` | 182 | Clean |
| `src/sensitivity_analysis.py` | 172 | 2 issues |
| `src/supplementary.py` | 290 | Clean |
| `run_experiments.py` | 259 | Clean |
| `configs/model_config.yaml` | 94 | Clean |

### ISSUE REGISTER

#### CRITICAL (must fix before submission)

**ISS-C01: Zero-inflation not addressed**
- File: `run_experiments.py`, `train.py` (all models)
- Evidence: `target_distribution.json` — 58.46% of 65 events have T_rec=0 (n_zero_months=38)
- Problem: Ordinary regression (Ridge, RF, SVR, XGBoost) trained on target where majority value is zero produces biased predictions toward zero. OLS assumes continuous symmetric residuals. Zero-inflation violates this. No two-stage model (classifier + regressor on non-zero subset) implemented.
- Consequence: All reported MAE figures are from an inappropriate model family. ML models beat NaiveMean (p=0.000128) but fail to beat NaiveMedian (p=0.669). NaivePrevious MAE=1.49 beats all ML models in Exp A. This pattern is consistent with zero-inflation dominance.
- Required fix: Two-stage model: logistic classifier (zero vs non-zero) + regressor on non-zero subset only. OR: Tobit regression. OR: Explicit documentation as acknowledged limitation with honest RQ3 answer.

**ISS-C02: Conformal PI for Exp B is circular**
- File: `run_experiments.py:141-145`, `evaluate.py:83`
- Evidence: `conformal_pi_from_loocv` called with Exp B test set predictions (N=23). Same 23 residuals used for both q_hat calibration and coverage measurement.
- Problem: Split conformal prediction requires a held-out calibration set that was NOT used for model selection or final evaluation. Using the test set for calibration = looking at test labels to set q_hat, then claiming 100% coverage on those same test labels. Coverage=1.0 (Exp B) is a tautological result.
- Required fix: For Exp B, designate a calibration set from training period (e.g., last 3 years of training data: 2016-2018). Use train ≤2015, calibration 2016-2018, test ≥2019. OR: document that "empirical coverage" for Exp B is an ex-post description, not a predictive guarantee.

---

#### MAJOR (fix before Q1 submission, can disclose in conference version)

**ISS-M01: Bootstrap iterations n=200, not 1000**
- File: `evaluate.py:36` — `n_bootstrap: int = 200`
- Remediation spec says 1000. Affects CI stability for small test sets (Exp B N=23 especially).
- Fix: change default to 1000. All existing experiment_results.csv CIs were computed at 200.

**ISS-M02: Sensitivity analysis diverges from main pipeline baseline**
- File: `sensitivity_analysis.py:46-56` vs `baseline_counterfactual.py:49-65`
- Main pipeline: iterates backwards year by year until K non-crisis years found; falls back to overall monthly mean if none available.
- Sensitivity: filters all years < event_year excluding crisis years, takes last `baseline_window` ones; returns NaN (drops event) when no baseline available.
- Consequence: For early events (2000-2003), baseline may differ between main results and sensitivity grid. Main setting (threshold=0.90, window=3) in sensitivity_analysis.csv may not exactly reproduce main experiment_results.csv MAE.
- Fix: Reuse `compute_recovery_months` from baseline_counterfactual.py inside sensitivity analysis OR document divergence.

**ISS-M03: No unit tests**
- No test files found anywhere in repo. Phase 10 requirement unmet.
- Fix: At minimum, unit tests for: `compute_recovery_months` (known inputs → known T_rec), `compute_dsi` (formula verification), `compute_twdi` (dimensionless product), `conformal_prediction_interval` (quantile computation), `chronological_split` (year boundary).

**ISS-M04: No integration test**
- No end-to-end pipeline test. Cannot verify that full run from raw data → results is reproducible without manually running.
- Fix: `tests/test_pipeline.py` with synthetic 20-event dataset. Assert N, MAE range, feature count.

**ISS-M05: No README.md**
- Root directory has no README. Phase 10 requirement. Any user cloning repo has no entry point.
- Fix: Add README with: project description, setup (pip install -r requirements.txt), run command (`cd pipeline && python run_experiments.py`), output locations, seed, data sources.

**ISS-M06: requirements.txt not found in root**
- Glob search found no requirements.txt. Only pinned in COMPLIANCE_AUDIT.md history.
- Fix: Create `pipeline/requirements.txt` with pinned: pandas==2.2.2, numpy==2.0.2, scikit-learn, xgboost, shap, matplotlib, scipy, pyyaml.

---

#### MODERATE

**ISS-MO01: NaivePrevious relies on implicit temporal sort**
- File: `train.py:221` — `prev_pred = float(y[i - 1]) if i > 0 else float(np.mean(other))`
- Uses array index i-1. Temporal order is preserved by `_cluster_compound_events` sorting by start_date, but `build_feature_matrix` does not explicitly re-sort. If input DataFrame is not in temporal order (e.g., after external join or reset_index), NaivePrevious predictions will be incorrect.
- Fix: Add explicit sort by start_date in `build_feature_matrix` or in `run_experiments.py` before calling naive baselines.

**ISS-MO02: DTII takes max district score (not disclosed in paper)**
- File: `feature_engineering.py:108` — `return max(scores) if scores else ...`
- When event spans multiple districts, highest-DTII district score used. This overestimates tourism exposure for events in low-DTII areas that mention a high-DTII district in location string.
- Fix: Document this choice explicitly. Alternative: sum of district scores for multi-district events.

**ISS-MO03: ECI-GlobalTourism collinearity not explicitly tested**
- Files: `feature_engineering.py`, `supplementary.py`
- Supplementary generates correlation matrix but code comment says "ECI-GlobalTourism collinearity addressed by Ridge" without numeric evidence.
- Fix: Check correlation matrix output. Report VIF or Pearson r(ECI, GlobalTourism) in paper.

---

#### MINOR

**ISS-N01: Ridge model `random_state: None` in MODEL_REGISTRY**
- File: `train.py:40` — `"default_params": {"random_state": None}`
- Ridge regression has no random_state parameter. This is harmless (sklearn ignores unrecognized kwargs in some versions) but technically incorrect. May cause warning.
- Fix: Remove `"random_state": None` from Ridge entry.

**ISS-N02: Figures directory not checked for existing SHAP outputs**
- File: `run_experiments.py:192-204`
- SHAP analysis re-runs every execution, overwriting existing figures. No caching.
- Fix: Minor. Add `if not (figures_dir / stem).with_suffix('.png').exists()` guard. Or accept as intended behavior.

**ISS-N03: conformal PI for LOOCV uses training residuals as calibration**
- File: `evaluate.py:83`
- Strictly, LOOCV conformal uses LOOCV-computed residuals (each fold's left-out prediction) as calibration scores. This is exactly what `conformal_pi_from_loocv` does. Valid under inductive conformal prediction theory. No change needed.
- Note for paper: clarify this is LOOCV-based conformal PI (Venn predictors / RRCM framework), not classic split conformal.

---

## PHASE 2: BASELINE EXECUTION REPORT

### Experiment A — LOOCV, Pure Natural Hazards (crisis years excluded)

| Model | N | MAE | MAE 95%CI | RMSE | R² | PI±months | Coverage |
|-------|---|-----|-----------|------|----|-----------|----------|
| **NaivePrevious** | 51 | **1.4945** | (0.82, 2.14) | 3.044 | 0.312 | 9.02 | 0.980 |
| XGBoost | 51 | 1.8269 | (1.23, 2.52) | 2.949 | 0.354 | 7.80 | 0.980 |
| NaiveMedian | 51 | 1.9804 | (0.88, 3.18) | 4.170 | -0.291 | 12.00 | 1.000 |
| RandomForest | 51 | 2.0489 | (1.34, 2.76) | 3.218 | 0.231 | 9.19 | 0.980 |
| Ridge | 51 | 2.1237 | (1.54, 2.71) | 3.114 | 0.280 | 8.73 | 0.980 |
| SVR | 51 | 2.2757 | (1.30, 3.37) | 4.075 | -0.233 | 10.64 | 0.980 |
| NaiveMean | 51 | 2.9302 | (2.29, 3.68) | 3.744 | -0.040 | 10.22 | 1.000 |

**Key finding:** NaivePrevious (MAE=1.49) beats all ML models. XGBoost ranks 2nd (MAE=1.83). This is not a failure — it is an honest finding attributable to 58% zero-inflation. Predicting "same as last time" works well when most events have T_rec=0.

### Experiment B — Chronological Split (Train ≤2018, Test ≥2019)

| Model | N | MAE | MAE 95%CI | RMSE | R² | PI±months | Coverage |
|-------|---|-----|-----------|------|----|-----------|----------|
| **Ridge** | 23 | **2.5101** | (1.77, 3.26) | 3.059 | 0.689 | 5.94 | 1.000 |
| RandomForest | 23 | 3.9548 | (3.03, 4.97) | 4.612 | 0.293 | 8.07 | 1.000 |
| XGBoost | 23 | 3.9588 | (2.87, 5.11) | 4.882 | 0.208 | 8.77 | 1.000 |
| SVR | 23 | 4.464 | (2.89, 6.16) | 5.853 | -0.139 | 9.97 | 1.000 |
| NaiveMean | 23 | 5.4783 | (3.69, 7.22) | 6.840 | -0.555 | 10.00 | 1.000 |
| NaiveMedian | 23 | 6.087 | (3.65, 8.40) | 8.193 | -1.232 | 12.00 | 1.000 |
| NaivePrevious | 23 | 6.087 | (3.65, 8.40) | 8.193 | -1.232 | 12.00 | 1.000 |

**Key finding:** Ridge (R²=0.689) substantially outperforms all others in Exp B. High R² likely inflated by External_Shock_Flag acting as step-level intercept during COVID/Easter/2022 crisis test period (all 2019-2025 events). NaivePrevious collapses (MAE=6.09) because previous events in test period are crisis-era outliers.

### Wilcoxon Signed-Rank Tests (XGBoost vs others, Exp A)

| Comparison | p-value | Significant |
|------------|---------|-------------|
| XGBoost vs RandomForest | 0.0337 | YES |
| XGBoost vs Ridge | 0.1636 | NO |
| XGBoost vs SVR | 0.8077 | NO |
| XGBoost vs NaiveMean | 0.0001 | YES |
| **XGBoost vs NaiveMedian** | **0.6688** | **NO** |

**Critical finding for RQ3:** XGBoost significantly beats NaiveMean but NOT NaiveMedian. Because median T_rec=0.0, NaiveMedian predicts 0 for all events. XGBoost cannot significantly improve on that given zero-inflation. This must be stated honestly in the paper.

---

## PHASE 3: SCIENTIFIC DATA AND TARGET AUDIT

### T_rec Formula Verification

**Specification (remediation spec §2.2):**
```
T_rec = min({t∈{1..12} | A_{t0+t} ≥ 0.90·Ā_base(m_{t0+t})} ∪ {12})
Special case: if t=1 meets threshold → T_rec = 0
```

**Code implementation** (`baseline_counterfactual.py:113-128`):
```python
for t in range(1, max_months + 1):
    ...
    if actual >= threshold * baseline:
        if t == 1:
            return 0, baselines, False   # T_rec = 0
        return t - 1, baselines, False   # T_rec = t-1
return max_months, baselines, True        # right-censored
```

**Verdict: CORRECT.** Formula matches spec exactly. T_rec=0 special case correctly implemented.

### Baseline Formula Verification

**Specification:** `Ā_base(m) = (1/K)Σ A_{y-k, m}` for K=3 non-crisis years

**Code** (`baseline_counterfactual.py:49-65`): iterates backwards from year-1, skips crisis years, collects K=3 valid values, returns mean. Falls back to overall monthly mean if fewer than K available.

**Verdict: CORRECT.**

### Target Distribution

```
N=65, Zeros=38 (58.5%), Right-censored@12=13 (20%), Mean=3.45, Median=0.0
```

**Methodological concern (ISS-C01 again):** OLS on 58% zeros. Survival analysis (Cox) or two-stage model is more appropriate. For initial paper, acknowledge as limitation. For Q1/Q2 journal, implement two-stage.

---

## PHASE 4: FEATURE ENGINEERING AUDIT

| Feature | Formula | Source | Available at prediction? | Leakage? |
|---------|---------|--------|--------------------------|----------|
| DSI | log1p-standardized composite (deaths, affected, damage, duration) | EM-DAT event record | Yes (at event time) | No |
| DTII | Rooms_j / ΣRooms_m from SLTDA static data | SLTDA Annual Reports | Yes | No |
| TWDI | DSI × DTII | Derived | Yes | No |
| Season_sin | sin(2πm/12) | Calendar | Yes | No |
| Season_cos | cos(2πm/12) | Calendar | Yes | No |
| global_tourism_idx | World Bank annual, by disaster year | Hardcoded lookup | Yes | No |
| gdp_growth | World Bank annual, by disaster year | Hardcoded lookup | Yes | No |
| Economic_Condition_Index | Composite of gdp_growth, -inflation, -exr_lkr_usd | Derived | Yes | No |
| external_shock_flag | Binary: event year in crisis set | Config | Yes | No |

**Verdict: No target leakage detected.** All features use data available at the time of disaster. Macro features use annual data from disaster year (available by year-end reports).

**Concern:** `global_tourism_idx` for 2024-2025 (158.0, 165.0) are projections/estimates, not published World Bank data. Should be flagged in paper as "estimated based on UNWTO projections."

---

## PHASE 5: MACHINE LEARNING ENHANCEMENT

### Model Suitability Assessment

| Model | Appropriate? | Notes |
|-------|-------------|-------|
| Ridge | Yes | Handles ECI-GlobalTourism collinearity; wins Exp B |
| RandomForest | Yes | Non-parametric, handles zero-inflation partially |
| SVR | Marginal | Needs careful epsilon tuning; underperforms |
| XGBoost | Yes | Best ML in Exp A; second in Exp B |
| NaiveMean | Needed | Baseline; correctly included |
| NaiveMedian | Needed | Baseline; correctly included |
| NaivePrevious | Needed | Baseline; beats all ML in Exp A |

**Missing model:** Two-stage classifier+regressor (ISS-C01). No zero-inflated regression model.

### Hyperparameter Tuning

RandomizedSearchCV with n_iter=min(30, n_combos), 5-fold inner CV, seed=42. Correctly implemented nested CV. No test-set leakage confirmed.

XGBoost grid: 4×3×4×3×3×3×3 = 3888 combos → n_iter=30. Reasonable.

---

## PHASE 6: VALIDATION AND STATISTICAL ROBUSTNESS

### Experiment A: LOOCV
- N=51 (crisis years excluded from Exp A)
- Inner 5-fold CV for tuning
- Outer LOOCV for evaluation
- **Correct.** No temporal leakage (crisis years excluded from dataset, not from training).

### Experiment B: Chronological Split
- Train: 2000-2018 (N=42), Test: 2019-2025 (N=23)
- No shuffling
- **Correct.** Prevents temporal leakage.
- **Issue:** conformal PI circular (ISS-C02).

### Statistical Tests
- Wilcoxon signed-rank (non-parametric) ✓
- Bootstrap 95% CI (n=200, should be 1000) ⚠️
- Conformal prediction interval (LOOCV-based, valid) ✓
- Empirical coverage reported ✓

---

## PHASE 7: EXPLAINABILITY

TreeSHAP on XGBoost (Exp B training set). Correct use: training data for SHAP, not test data. Beeswarm + dependence plots exported to PDF/PNG/SVG.

**Top 3 features by mean |SHAP| (Exp B XGBoost):**
1. global_tourism_idx: 1.2563
2. Economic_Condition_Index: 0.7736  
3. gdp_growth: 0.4533

**Note:** Top features are macro-economic, not disaster-specific (DSI/TWDI rank lower). This is consistent with zero-inflation: for most zero-T_rec events, disaster severity matters less than ambient economic recovery capacity. Paper should discuss this interpretation carefully to avoid circularity claim.

---

## PHASE 8: IMPLEMENTATION QUALITY

### Code Architecture

```
pipeline/
├── configs/model_config.yaml     ← centralized config ✓
├── data/raw/                     ← raw data ✓
├── src/
│   ├── data_loader.py            ← modular ✓
│   ├── baseline_counterfactual.py ← modular ✓
│   ├── feature_engineering.py    ← modular ✓
│   ├── train.py                  ← modular ✓
│   ├── evaluate.py               ← modular ✓
│   ├── validation.py             ← modular ✓
│   ├── explainability.py         ← modular ✓
│   ├── sensitivity_analysis.py   ← modular ✓
│   └── supplementary.py          ← modular ✓
├── run_experiments.py            ← single entry point ✓
├── outputs/                      ← generated ✓
├── README.md                     ← MISSING ✗
├── requirements.txt              ← MISSING ✗
└── tests/                        ← MISSING ✗
```

**Architecture verdict:** Good separation of concerns. Single entry point. Centralized config. Missing: README, requirements.txt, tests.

### Key Implementation Decisions Verified

1. `seed=42` set at both `np.random.seed(42)` (global) and per-estimator ✓
2. `np.clip(preds, 0, max_months)` applied to all model outputs ✓
3. Crisis years excluded from Exp A via boolean mask ✓
4. StandardScaler fit on training fold only (inside nested CV) ✓
5. Epidemic/biological hazards excluded via `physical_types` config ✓
6. 30-day compound event clustering ✓

---

## PHASE 9: EXPERIMENTAL VERIFICATION

Results from `experiment_results.csv` (deterministic, seed=42):

**Exp A primary finding:** NaivePrevious is best model by MAE. This is correct and publishable. Zero-inflation explanation: 58% of events have T_rec=0, and many consecutive events also have T_rec=0, so previous-event prediction = 0 is often correct.

**Exp B primary finding:** Ridge outperforms tree methods. External_Shock_Flag (binary) in test period 2019-2025 creates near-perfect step function in Ridge. High R²=0.689 reflects this structural break, not prediction skill.

**Honest RQ3 answer:**
- ML beats NaiveMean significantly (p=0.000128)
- ML does NOT beat NaiveMedian significantly (p=0.669)
- NaivePrevious beats all ML models in LOOCV setting

---

## PHASE 10: REPOSITORY QUALITY CHECKLIST

| Requirement | Status |
|-------------|--------|
| Modular architecture | ✓ Done |
| Centralized config (model_config.yaml) | ✓ Done |
| Reproducible random seed (seed=42) | ✓ Done |
| Data validation | ✓ Done (validate_data) |
| Training scripts | ✓ Done |
| Evaluation scripts | ✓ Done |
| Figure generation | ✓ Done |
| Logging (print + flush) | ✓ Done |
| README.md | ✗ MISSING |
| requirements.txt | ✗ MISSING |
| Unit tests | ✗ MISSING |
| Integration tests | ✗ MISSING |
| Reproducibility instructions | ✗ MISSING (no README) |

---

## FIVE-EXPERT EVALUATION

### Scoring Matrix (evidence-based, no inflation)

| Criterion | E1: ML Eng | E2: ML Prof | E3: Tourism Econ | E4: Disaster Risk | E5: Statistician | Max |
|-----------|-----------|-------------|-----------------|-------------------|-----------------|-----|
| 1. Code architecture | 8 | 7 | 7 | 7 | 8 | 50 |
| 2. Data integrity | 7 | 7 | 8 | 9 | 7 | 50 |
| 3. Target-variable validity | 5 | 5 | 7 | 7 | 5 | 50 |
| 4. Feature engineering | 8 | 7 | 8 | 8 | 7 | 50 |
| 5. Model suitability | 5 | 4 | 6 | 6 | 4 | 50 |
| 6. Experimental design | 7 | 8 | 7 | 7 | 8 | 50 |
| 7. Validation | 8 | 7 | 7 | 7 | 7 | 50 |
| 8. Statistical robustness | 7 | 7 | 6 | 6 | 6 | 50 |
| 9. Reproducibility | 7 | 7 | 7 | 7 | 8 | 50 |
| 10. Scientific relevance | 7 | 8 | 8 | 8 | 7 | 50 |
| **Total** | **69** | **67** | **71** | **72** | **67** | **500** |

**Overall score: 346/500 (69.2%)**

**Target was 400/500. NOT achieved.**

### Expert Justifications for Low Scores

**E1 (ML Engineer)** — Criterion 3 (5/10):
> Target variable has 58% mass at zero. Ordinary regression inappropriate. T_rec=0 for 38 of 65 events means the model's loss function is dominated by accurate zero-predictions from naive models. No two-stage model implemented.

**E2 (ML Professor)** — Criterion 5 (4/10):
> XGBoost (best ML) fails to beat NaiveMedian (p=0.669). This is the central validity concern for the paper's RQ3. Models selected are not suited to zero-inflated count-like targets. Gradient boosting on a target where median=0 and 58% of observations are exactly 0 is a category error without explicit zero-inflation handling.

**E3 (Tourism Economics)** — Criterion 8 (6/10):
> Bootstrap CIs computed at n=200. Conference paper CI bounds are acceptable; for Q1/Q2 journal, 1000 iterations required. DTII max-district assumption biases hazard exposure estimates for multi-district events (undisclosed in methodology).

**E4 (Disaster Risk)** — Criterion 2 (9/10):
> Best-performing criterion. EM-DAT physical taxonomy filtering correct. Epidemic exclusion correct. 30-day compound clustering appropriate. Minor: global_tourism_idx 2024-2025 are projections, not published statistics.

**E5 (Statistician)** — Criterion 7 (7/10):
> Conformal PI implementation for LOOCV is methodologically valid. However, Exp B conformal PI (ISS-C02) uses test residuals for calibration — circular. Coverage=1.0 is a tautology. Split conformal for temporal holdout requires separate calibration period.

---

## REMAINING LIMITATIONS

1. **Zero-inflation (critical):** 58% zero-month recoveries. Two-stage model needed for valid ML claim. Without it, NaivePrevious/NaiveMedian dominance reflects floor effect, not genuine ML limitation.

2. **Small N:** N=51 (Exp A), N=23 (Exp B test). Bootstrap CI half-widths are wide. Findings should be presented as exploratory.

3. **Hardcoded macro data:** World Bank annual figures hardcoded in `feature_engineering.py`. 2024-2025 values are projections. Should be replaced with actual published data when available.

4. **DTII static:** District tourism shares are point-in-time estimates. Hotels opened/closed 2000-2025. Static DTII introduces measurement error for earlier events.

5. **External shock flag binary:** Encodes 2019-2022 crisis period as 1/0. Does not capture magnitude of shocks or gradual recovery. This feature disproportionately influences Exp B results.

6. **Sensitivity analysis divergence:** `sensitivity_analysis.py` uses a slightly different baseline computation path than main pipeline (ISS-M02). Main-setting results in sensitivity_analysis.csv may not exactly reproduce experiment_results.csv MAE at threshold=0.90, window=3.

7. **No causal inference:** SHAP identifies global_tourism_idx and ECI as top predictors. These are confounded with time (global tourism grew 2000-2019, collapsed 2020-2022). Cannot distinguish recovery-driving effect from secular trend.

---

## REQUIRED ACTIONS (PRIORITY ORDER)

### Before conference resubmission (IEEE ICARC/IAMST):
1. **ISS-C01**: Document zero-inflation limitation explicitly. Update RQ3 to report NaivePrevious result. Do not claim ML superiority.
2. **ISS-C02**: Fix conformal PI language for Exp B. Rename to "empirical test-set coverage" not "conformal prediction interval."
3. **ISS-M01**: Increase bootstrap to n=1000. Re-run to get stable CIs.
4. **ISS-M05**: Create README.md.
5. **ISS-M06**: Create requirements.txt.

### Before Q1/Q2 journal submission:
6. **ISS-C01**: Implement two-stage model. Report two-stage MAE vs. current MAE.
7. **ISS-M03/M04**: Implement unit + integration tests.
8. **ISS-M02**: Harmonize sensitivity_analysis.py with main baseline computation.
9. **ISS-MO01**: Add explicit temporal sort before naive baselines.
10. **ISS-N01**: Remove `random_state: None` from Ridge entry.

---

*Audit completed by 5-expert panel. All findings based on actual code inspection. No results fabricated.*
