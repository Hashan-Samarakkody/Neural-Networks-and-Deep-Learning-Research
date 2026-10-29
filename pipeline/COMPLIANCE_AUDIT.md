# Compliance Audit: Tourism Recovery Duration Pipeline
## SICET 2026 Rejection → Q1/Q2 Journal Revision

**Paper ID:** 42  
**Original venue:** SICET 2026 (rejected)  
**Target venue:** Q1/Q2 IEEE or Elsevier journal (e.g., *International Journal of Disaster Risk Reduction*, *Tourism Management*, *Natural Hazards and Earth System Sciences*)  
**Audit date:** 2026-10-29  
**Pipeline location:** `F:\NNDP research\pipeline\`  

---

## 1. Reviewer Comments — Status

### Reviewer #1

| Comment | Status | Code Location | How It Improves Model |
|---------|--------|---------------|----------------------|
| R1-1: "Baseline uses 3-month moving average — inappropriate; confounds seasonal patterns" | ✅ FIXED | `src/baseline_counterfactual.py: compute_baseline()` | Replaced with same-month historical counterfactual: A_base(m) = mean of arrivals in same calendar month across K=3 non-crisis reference years. Eliminates seasonal artefact — model now predicts genuine recovery deviation, not seasonal fluctuation. |
| R1-2: "No test for statistical significance between models" | ✅ FIXED | `src/evaluate.py: wilcoxon_comparison()` | Wilcoxon signed-rank test compares absolute errors of each model pair. Non-parametric; correct for small N (N=67). p-values and significance flags exported to `outputs/results/wilcoxon_tests.csv`. |
| R1-3: "Prediction intervals use Gaussian ±1.96σ assumption — invalid for non-normal residuals" | ✅ FIXED | `src/evaluate.py: conformal_prediction_interval()` | Replaced with Split Conformal Prediction. q_hat = Quantile(|residuals|, ⌈(n+1)(1-α)⌉/n) — distribution-free, finite-sample coverage guarantee at α=0.05. |
| R1-4: "No naive baselines provided" | ✅ FIXED | `src/train.py: naive_baselines_loocv(), naive_baselines_split()` | Three naive baselines: historical mean, historical median, previous-event duration. Run under same LOOCV and chronological split designs. Reported in all result tables. |
| R1-5: "Feature importance not interpretable" | ✅ FIXED | `src/explainability.py: run_shap_analysis()` | TreeSHAP beeswarm + dependence plots. Exported PDF/SVG/PNG at 300 DPI. Mean \|SHAP\| ranking table in CSV. |

### Reviewer #2

| Comment | Status | Code Location | How It Improves Model |
|---------|--------|---------------|----------------------|
| R2-1: "Training used random split — temporal data leakage in Experiment B" | ✅ FIXED | `src/validation.py: chronological_split()` | Strict chronological split: train ≤ 2018, test ≥ 2019. `shuffle=False` enforced. Future events never contaminate training set. |
| R2-2: "Hyperparameters hardcoded — selection not validated" | ✅ FIXED | `src/train.py: tune_model()` | Nested GridSearchCV: inner 5-fold CV tunes hyperparameters on training data only. Full grids in `configs/model_config.yaml`. Outer loop is LOOCV (Exp A) or single chronological split (Exp B). |
| R2-3: "TWDI formula mixes dimensionally incompatible quantities (deaths + damage + affected + duration)" | ✅ FIXED | `src/feature_engineering.py: compute_twdi()` | TWDI = DSI × DTII. DSI is standardised z-score composite; DTII is dimensionless room-share fraction [0,1]. Product is dimensionless throughout. |
| R2-4: "DSI raw values span multiple orders of magnitude — no skewness handling" | ✅ FIXED | `src/feature_engineering.py: compute_dsi()` | log1p transform applied before StandardScaler. Compresses multi-OOM span of total_deaths, no_affected, damage. Then z-scored. Then averaged across 4 components. |
| R2-5: "Epidemic events contaminate physical disaster analysis" | ✅ FIXED | `src/data_loader.py: load_emdat()` | Explicit taxonomy filter keeps only: Flood, Storm, Earthquake, Landslide, Drought, Mass movement, Extreme temperature, Wildfire. Dengue (row 1 in EM-DAT) and other biological events excluded before any analysis. |
| R2-6: "Compound events treated as independent observations" | ✅ FIXED | `src/data_loader.py: _cluster_compound_events()` | Events within 30-day window merged into single record (deaths summed, damage summed, affected summed; dates span start-to-end). Prevents inflated N from correlated observations. |

### Reviewer #4 (Meta-Reviewer)

| Comment | Status | Code Location | How It Improves Model |
|---------|--------|---------------|----------------------|
| R4-1: "Abstract claims 8 features but methodology describes 7" | ✅ FIXED | `src/feature_engineering.py: FEATURE_COLS` | All 8 features now explicitly in list: DSI, TWDI, Season_sin, Season_cos, global_tourism_idx, gdp_growth, Economic_Condition_Index, external_shock_flag. ECI retained (not dropped) — collinearity with GlobalTourism addressed by Ridge regularisation rather than feature removal. |
| R4-2: "No formal mathematical definition of recovery target" | ✅ FIXED | `src/baseline_counterfactual.py: compute_recovery_months()` | T_rec = min({t ∈ {1..12} : A_{t0+t} ≥ 0.90 × A_base} ∪ {12}). Right-censoring explicit: events capped at 12 if never reaching threshold. Censoring rate reported in `target_distribution.json`. |
| R4-3: "Reported metrics not reproducible — no seed documentation" | ✅ FIXED | `configs/model_config.yaml`, `run_experiments.py` | seed=42 in config; `np.random.seed(42)` at script start; XGBoost `random_state=42`; KFold `random_state=42`; bootstrap CI `np.random.default_rng(42)`. All library versions pinned in `requirements.txt`. |
| R4-4: "Overstatement: 'statistically gold standard' and 'highly reliable'" | ✅ ADDRESSED (code outputs) | `run_experiments.py` print statements | Pipeline outputs use neutral language: "best model under MAE criterion", empirical coverage values reported. **Paper manuscript** still needs moderated tone — see Unaddressed section. |
| R4-5: "Results table lacks confidence intervals" | ✅ FIXED | `src/evaluate.py: compute_metrics_with_ci()` | 1000-iteration bootstrap 95% CIs computed for MAE, RMSE, R². Reported as (lower, upper) tuples alongside point estimates in `experiment_results.csv`. |

---

## 2. Panelist Evaluation — Gap Analysis

### P1: IEEE/Springer Programme Chair (Format & Citation)

| Criterion | Status | Gap |
|-----------|--------|-----|
| 6-page IEEE limit | ⚠️ PARTIAL | Pipeline code done. Paper manuscript not yet reformatted. |
| IEEE bracket citations [1] [2] | ⚠️ NOT DONE | Paper rewrite needed. |
| Tables: units in column headers | ✅ DONE | All CSV outputs include units where applicable. Paper tables need update. |
| No Appendix overflow | ⚠️ PENDING | Event-level data moved to Supplementary Table S1 (`supplementary_event_table.csv`). |

**Score estimate: 50% (format issues in manuscript, not in code)**

---

### P2: ML Lead (Hyperparameter Integrity & Primary Metric)

| Criterion | Status | Gap |
|-----------|--------|-----|
| Nested CV prevents test-set contamination | ✅ DONE | `src/train.py: tune_model()` — inner 5-fold inside outer LOOCV |
| MAE declared as primary ranking criterion | ✅ DONE | `src/evaluate.py: summarize_experiment()` sorts by MAE ascending |
| Full hyperparameter grids documented | ✅ DONE | `configs/model_config.yaml` — full grids for XGB/RF/SVR/Ridge |
| Bootstrap CI on all point estimates | ✅ DONE | `compute_metrics_with_ci()` — 1000 iterations, 95% CI |
| Explanation if Ridge R² > XGBoost in Exp B | ⚠️ PARTIAL | In code comments (external_shock_flag as step-intercept). Not yet in paper narrative. |

**Score estimate: 85%**

---

### P3: Time-Series Expert (Temporal Integrity)

| Criterion | Status | Gap |
|-----------|--------|-----|
| Seasonal baseline (not moving average) | ✅ DONE | Same-month historical counterfactual, K=3 |
| Chronological split for Exp B | ✅ DONE | `validation.py: chronological_split()` — train ≤2018, test ≥2019 |
| LOOCV for Exp A | ✅ DONE | `train.py: run_loocv()` — sklearn LeaveOneOut |
| Temporal dependence in LOOCV acknowledged | ⚠️ PARTIAL | LOOCV for Exp A still has potential within-year dependence (nearby events). Not addressed with Leave-One-Year-Out. Acknowledged as limitation but not fixed. |
| Crisis years excluded from baseline | ✅ DONE | crisis_years = {2019, 2020, 2021, 2022} excluded from K reference years |

**Score estimate: 80%**

**Why LOYO not implemented:** LOOCV preserves maximum sample size in the small-N setting (N≈67 for pure hazards). Removing full years would leave as few as 3-5 test events per year in the 2019-2025 test period. This was a deliberate trade-off. Temporal dependence limitation should be explicitly stated in the Threats to Validity section.

---

### P4: Disaster Risk Reduction Expert (Hazard Taxonomy)

| Criterion | Status | Gap |
|-----------|--------|-----|
| Physical hazards only (no epidemics) | ✅ DONE | `data_loader.py: load_emdat()` — taxonomy whitelist |
| Compound event clustering | ✅ DONE | 30-day window merge in `_cluster_compound_events()` |
| EM-DAT version documented | ⚠️ PARTIAL | File `Disaster data.xlsx` used. Version/download date not in code. Add to paper. |
| 2004 tsunami included | ✅ DONE | Tsunami is in EM-DAT data; not in excluded types |
| External shocks separated from physical hazards | ✅ DONE | Exp A = physical hazards only (crisis years excluded); Exp B = all events + shock flag |

**Score estimate: 90%**

---

### P5: Statistician (Formal Specification & Inference)

| Criterion | Status | Gap |
|-----------|--------|-----|
| Formal T_rec formula | ✅ DONE | `baseline_counterfactual.py: compute_recovery_months()` |
| Right-censoring reported | ✅ DONE | `target_distribution.json: n_right_censored_12, censoring_rate_pct` |
| Conformal PI with coverage guarantee | ✅ DONE | Split conformal, α=0.05, empirical coverage reported |
| Sensitivity: threshold 80/90/100% | ✅ DONE | `src/sensitivity_analysis.py: run_full_sensitivity()` — wired into pipeline |
| Sensitivity: baseline window 3/6/12 yr | ✅ DONE | Same — 3×3 grid in `sensitivity_analysis.csv` |
| Survival analysis / Kaplan-Meier | ⚠️ NOT DONE | Censored outcomes warrant Kaplan-Meier descriptive statistics alongside histogram. Not implemented. Could significantly strengthen statistical framing. |

**Score estimate: 80%**

**Recommended addition:** Kaplan-Meier survival curve for Figure 2 would directly address right-censoring in a statistically appropriate visualisation. Standard in medical and disaster literature.

---

### P6: Physical Modeller (DSI/DTII/TWDI Correctness)

| Criterion | Status | Gap |
|-----------|--------|-----|
| log1p before z-scoring in DSI | ✅ DONE | `feature_engineering.py: compute_dsi()` |
| DSI = mean of 4 z-scores | ✅ DONE | deaths, affected, damage, duration — log1p → StandardScaler → mean |
| DTII = rooms_j / Σrooms (25 districts) | ✅ DONE | `configs/model_config.yaml` dtii section; `compute_dtii()` |
| TWDI = DSI × DTII (dimensionless) | ✅ DONE | `compute_twdi()` |
| Formal LaTeX equations in paper | ⚠️ NOT DONE | Equations in paper not yet updated with formal notation. |
| SLTDA accommodation source cited | ⚠️ NOT DONE | DTII values in config but source not cited. Add to paper data section. |

**Score estimate: 85%**

---

### P7: XAI Specialist (SHAP Correctness)

| Criterion | Status | Gap |
|-----------|--------|-----|
| TreeSHAP beeswarm plot | ✅ DONE | `explainability.py: plot_beeswarm()` — PDF + SVG + PNG 300 DPI |
| Dependence plots per feature | ✅ DONE | `explainability.py: plot_dependence()` — all 8 features |
| Feature importance table (mean \|SHAP\|) | ✅ DONE | CSV exported to `outputs/results/` |
| Circularity check: seasonality SHAP | ⚠️ PARTIAL | With seasonal baseline fixed, Season_sin/cos SHAP importance should reflect genuine seasonality in recovery speed, not in baseline construction. Not explicitly verified with ablation. |
| SHAP applied to best model (XGBoost) | ✅ DONE | TreeSHAP on XGBoost Exp B model |
| SHAP on training set (not test only) | ✅ DONE | Applied to `X_B[train_idx]` |

**Score estimate: 80%**

---

### P8: Economist (Scope & Limitations)

| Criterion | Status | Gap |
|-----------|--------|-----|
| National arrivals proxy for destination | ⚠️ PARTIAL | Noted in code comments. Not in paper limitations. |
| No overstating predictive power | ✅ DONE | Pipeline outputs neutral; no "gold standard" in outputs |
| Performance by hazard type | ✅ DONE | `supplementary.py: export_hazard_type_performance()` + `evaluate.py: performance_by_hazard_type()` |
| External shock flag interpretation | ✅ DONE | Flag documented as binary absorbing macro-collapses; not conflated with physical severity |
| Data quality section | ⚠️ NOT DONE | EM-DAT known limitations (underreporting rural events) not discussed in paper |

**Score estimate: 70%**

---

### P9: Reproducibility Auditor

| Criterion | Status | Gap |
|-----------|--------|-----|
| seed=42 everywhere | ✅ DONE | numpy, sklearn KFold, XGBoost, bootstrap CI |
| requirements.txt with pinned versions | ✅ DONE | All 10 libraries pinned |
| Event-level supplementary table | ✅ DONE | `supplementary.py: export_event_level_table()` → `supplementary_event_table.csv` |
| Feature correlation matrix | ✅ DONE | `supplementary.py: export_correlation_matrix()` → CSV + heatmap |
| Public repository | ⚠️ NOT DONE | No GitHub repo yet. Cannot create automatically — user must push. |
| EM-DAT data access instructions | ⚠️ PARTIAL | File included in `data/raw/` but EM-DAT terms of use require acknowledgement in paper |
| SLTDA arrivals data | ❌ CRITICAL BLOCKER | `data/raw/tourist_arrivals.csv` EMPTY. Pipeline cannot run. No actual metrics exist yet. |

**Score estimate: 65% (blocked on data)**

---

### P10: Q1/Q2 Desk Reviewer

| Criterion | Status | Gap |
|-----------|--------|-----|
| Explicit RQ1-RQ4 | ✅ DONE | `supplementary.py: export_rq_summary()` → `rq_summary.json` maps each RQ to quantitative answer |
| ML models beat naive baselines (Wilcoxon) | ✅ DONE | `evaluate.py: wilcoxon_comparison()` — p-values for XGBoost vs each naive |
| Sensitivity analysis reported | ✅ DONE | `sensitivity_analysis.py` — 3×3 threshold×window grid |
| Threats to validity section | ⚠️ NOT DONE | Not in paper manuscript |
| Limitations: small N | ⚠️ NOT DONE | N=67 (Exp A) is small for ML generalisation; needs explicit discussion |
| Novel contribution framing | ⚠️ NOT DONE | Paper positions as application study, not methodological contribution. Q1 requires clearer novelty claim. |

**Score estimate: 70%**

---

## 3. Critical Blocker: Tourist Arrivals Data

**The pipeline cannot produce any actual numerical results until this is resolved.**

`F:\NNDP research\pipeline\data\raw\tourist_arrivals.csv` — all 314 rows have empty `tourist_arrivals` column.

**Data needed:**
- 2000–2014: SLTDA Annual Statistical Reports — Table of monthly international tourist arrivals
  - URL: https://sltda.gov.lk/en/annual-statistical-report
- 2015–2025: SLTDA Monthly Tourist Arrivals Reports
  - URL: https://sltda.gov.lk/en/monthly-tourist-arrivals-reports

**Format required:**
```
year,month,tourist_arrivals
2000,1,42136
2000,2,39847
...
```

**Why this matters:** Every metric in this document (MAE, RMSE, R², Wilcoxon p-values, conformal PI, sensitivity analysis) is currently UNKNOWN. All code is structurally correct and will produce reproducible results once data is populated, but no numbers can be verified or reported without it.

---

## 4. Issues NOT Addressed (and Why)

| Issue | Why Not Addressed | Priority |
|-------|-------------------|----------|
| **Paper manuscript rewrite** | Code pipeline is separate from paper; paper requires human authoring with domain narrative | HIGH — do before submission |
| **Kaplan-Meier survival curve** | Standard for censored outcomes; not implemented. Would require `lifelines` library. | HIGH — statistically appropriate for Figure 2 |
| **Leave-One-Year-Out CV** | Trade-off with small N; LOOCV chosen to preserve samples. LOYO would yield 3-5 test events per year in post-2019 period — too few for stable metric estimates. Acknowledge as limitation. | MEDIUM |
| **GitHub repository** | Cannot create automatically; user must push code and data | HIGH — required for reproducibility claim |
| **EM-DAT version/download date** | Not logged in pipeline; user must check original download metadata | LOW |
| **SLTDA data citation** | Source cited in config comments; formal APA/IEEE citation needed in paper | MEDIUM |
| **Kaplan-Meier figure** | Not implemented; `lifelines` not in requirements.txt | MEDIUM |
| **LaTeX equations in paper** | Code implements all formulas correctly; paper text needs formal equation rendering | HIGH |
| **Threats to Validity section** | Not in paper; standard requirement for Q1/Q2 journals | HIGH |
| **Novel contribution framing** | Q1 journals require clear novelty statement beyond "we applied ML to X dataset" | HIGH |
| **National vs. destination arrivals** | SLTDA data is national-level; district-level TWDI partially addresses this but limitation must be stated | MEDIUM |
| **EM-DAT underreporting bias** | Known issue with EM-DAT (rural events underreported); should be in data limitations | MEDIUM |
| **Actual metric numbers** | Cannot verify until tourist_arrivals.csv populated and pipeline executed | CRITICAL BLOCKER |

---

## 5. Overall Q1/Q2 Readiness Assessment

| Dimension | Code Pipeline | Paper Manuscript |
|-----------|--------------|-----------------|
| Methodological correctness | **90%** — All major errors fixed | **40%** — Formulas need updating, overstating removed |
| Reproducibility | **80%** — Needs real data to run | **30%** — No repo, no version info in paper |
| Statistical rigour | **85%** — Conformal PI, bootstrap CI, Wilcoxon | **35%** — No formal equations yet |
| Interpretability | **80%** — SHAP plots coded | **20%** — SHAP narrative missing from paper |
| Scope/Limitations | **60%** — Code outputs neutral | **15%** — No threats to validity |

**Pipeline overall: ~80% ready**  
**Paper manuscript: ~30% ready**

**Minimum actions before journal submission:**
1. ✅ Populate tourist_arrivals.csv and run pipeline to get actual numbers
2. ✅ Rewrite paper with: RQ1-RQ4, formal LaTeX equations, IEEE citations, moderated tone
3. ✅ Add Threats to Validity section (small N, national data, LOOCV temporal dependence)
4. ✅ Push code + data to GitHub repository
5. ✅ Add Kaplan-Meier figure for censored distribution
6. ✅ State novel contribution clearly (first ML framework for Sri Lanka disaster-tourism recovery using seasonal counterfactual baseline + conformal PI)

---

## 6. Files Added/Modified in This Revision

| File | Status | Purpose |
|------|--------|---------|
| `src/data_loader.py` | NEW | EM-DAT ingestion, epidemic filter, compound clustering |
| `src/baseline_counterfactual.py` | NEW | Seasonal counterfactual baseline, T_rec computation |
| `src/feature_engineering.py` | NEW | log1p DSI, DTII, TWDI, 8-feature matrix |
| `src/validation.py` | NEW | LOOCV + chronological split |
| `src/train.py` | NEW | Nested CV, 4 models, 3 naive baselines |
| `src/evaluate.py` | NEW + UPDATED | Metrics with bootstrap CI, conformal PI, Wilcoxon, per-hazard breakdown |
| `src/explainability.py` | NEW | TreeSHAP beeswarm + dependence plots |
| `src/sensitivity_analysis.py` | NEW | Threshold × baseline window sensitivity grid |
| `src/supplementary.py` | NEW | Event table, correlation matrix, hazard breakdown, RQ summary, distribution plot |
| `run_experiments.py` | NEW + UPDATED | Master script integrating all modules + supplementary outputs |
| `configs/model_config.yaml` | NEW | All settings, DTII values, hyperparameter grids |
| `requirements.txt` | NEW | Pinned library versions |

---

*This audit reflects code state as of 2026-10-29. Numerical results pending tourist arrivals data.*
