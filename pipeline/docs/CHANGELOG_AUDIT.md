# Audit Changelog

Per-issue record of old vs new behaviour. Every change that alters a number is listed with old and new values where known. Baseline commit: **fb7930c** (24 Nov 2026).

---

## Revision 4 fixes (S-01 to S-20)

Baseline for Revision 4: commit **e3225d1** (frozen run, branch `master`).
Branch: `fix/revision-4`.

| Issue | Commit | Summary |
|-------|--------|---------|
| S-01 | e49aaac | confounding_windows config, confounded_by column, 3 variants, confounding_sensitivity.csv |
| S-02 | f693103 | seasonal_trend + yoy_growth baseline variants, baseline_method_comparison.csv |
| S-03 | f6fd80c | persistence rule (default=1), depth/months_below/pre_event_ratio in event table |
| S-04 | 47d2d5d | scikit-survival replaces lifelines; CoxPH + CoxnetAFT; C-index + IBS; survival_results.csv |
| S-05 | e030382 | version assertion at startup; git commit hash in run_manifest.json; VERSION_SENSITIVITY.md |
| S-06 | fb88545 | buffered LOOCV, LOYO, expanding-window; experiment label in results CSV |
| S-07 | 08b980c | sign-flip permutation test + Holm correction replaces Wilcoxon |
| S-08 | 3b87d2c | block_events param (overlapping blocks) replaces block_months bug |
| S-09 | 1a81007 | feature_diagnostics.json; reference_fullsample.csv |
| S-10 | cfbc853 | paper_numbers.json; make_paper_figures.py; paper_numbers.py |
| S-11 | 95adae0 | 5 thresholds × 6 K × 2 persistence sensitivity grid; zero_share + n_censored |
| S-12 | fa04b3f | AUC + majority_rate in two_stage_stage1.json; METHODS.md |
| S-13 | 8926b3a | regime column (baseline/easter_covid_crisis/post_crisis/confounded) in predictions + results |
| S-14 | 3318306 | supplementary_event_table missing columns + dtii_report.csv |
| S-15 | c50f6b5 | event_flow: raw_emdat_records=86; full 86→80→65→60 flow; arithmetic assertion |
| S-16 | 8351e30 | README Key Findings updated from paper_numbers.json; CHANGELOG_AUDIT.md S-01..S-15 |
| S-17 | c8f5150 | MIT LICENSE, CITATION.cff, docs/DATA_AVAILABILITY.md |
| S-18 | 4d3670f | permutation_importance, grouped_importance, VIF, ablation_no_index CSVs |
| S-19 | cfdce58 | DATA_PROVENANCE.md 2025 provisional status; macro_lag_years sensitivity CSV |
| S-20 | fa9b7a5 | tests: persistence rule, trend baselines, block_events param, Holm correction; 45 tests pass |

---

## R-01 — Fold-safe preprocessing

**Commit:** 47d12d1

| | Old | New |
|-|-----|-----|
| DSI scaler fit | `scaler.fit_transform(log_df)` on all 65 events | `FeatureBuilder.fit(df_train)` inside each LOOCV fold / chronological split |
| ECI scaler fit | `StandardScaler().fit_transform` on all rows | Fitted on training fold only inside `FeatureBuilder` |
| Model scaler | Fitted outside inner CV | sklearn `Pipeline([("scaler", ...), ("model", ...)])` refits on each inner fold |

**Effect on numbers:** All metrics change. New results are in `outputs/results/` from the frozen run (R-05).

---

## R-02 — TWDI sign inversion

**Commit:** 47d12d1

| | Old | New |
|-|-----|-----|
| TWDI formula | `TWDI = DSI × DTII` (DSI is signed z-score; negative for 35/65 events) | `TWDI = DSI_pos × DTII` where `DSI_pos = clip((DSI - min_train)/(max_train - min_train), 0, 1)` |
| Minimum TWDI | Negative (down to approx. −0.6) | 0 always |

**Effect on numbers:** TWDI range changes from [−0.6, +0.6] to [0, 1]. Feature importance changes.

---

## R-03 — Baseline fallback removed

**Commit:** 47d12d1

| | Old | New |
|-|-----|-----|
| Fallback | Averaged all years including future/crisis when no prior non-crisis year found | Returns 0.0; offset skipped |
| Events affected | 2 events (2000-09, 2000-11) got baseline from full-dataset average | Flagged `baseline_insufficient=True`, dropped from modelling |
| N events | 65 (with 2 spurious censored events) | 63 (after dropping 2 insufficient-history events) |

**Old censoring rate:** ~20%. **New censoring rate:** may change slightly after re-run.

---

## R-04 — Event-flow table

**Commit:** 47d12d1

| | Old | New |
|-|-----|-----|
| Event counts in docstrings | N=67, N=86 (stale) | Removed; `event_flow.json` documents every stage |
| event_flow.json | Not present | Written by `run_experiments.py` to `outputs/results/event_flow.json` |

---

## R-05 — Frozen outputs run

**Status:** Pending — requires full pipeline execution on clean clone after all fixes.
Old outputs in `outputs_archive/pre_audit_fb7930c/`.

---

## R-06 — Macro indicators provenance

**Commit:** (this batch)

| | Old | New |
|-|-----|-----|
| Storage | `WORLD_BANK_ANNUAL` dict hardcoded in `feature_engineering.py` | `pipeline/data/raw/macro_indicators.csv` with source, indicator_code, vintage |
| Missing year | Nearest-year fallback | Raises `KeyError` |
| Documentation | None | `docs/DATA_PROVENANCE.md` |

**Values unchanged** — same numbers, different storage.

---

## R-07 — Missing damage flag

**Commit:** 47d12d1

| | Old | New |
|-|-----|-----|
| Damage zero-fill | `fillna(0.0)` silently | `damage_missing` flag added before fill; 3-component DSI default |
| DSI components | 4 (incl. damage_usd_000) | 3 by default (`dsi_include_damage: false`); 4-component available via config |
| Events with damage=0 | 48/65 (zero-filled) | 48/65 (zero-filled but flagged); DSI 3-component avoids the issue |

---

## R-08 — Dependence-aware validation

**Commit:** 47d12d1

| | Old | New |
|-|-----|-----|
| Inner CV | `KFold(shuffle=True)` | `KFold(shuffle=False)` |
| Buffered LOOCV | Not present | `buffered_loocv_split(buffer_months=12)` added |
| Leave-one-year-out | Not present | `leave_one_year_out_split()` added |

---

## R-09 — NaiveLastKnown baseline

**Commit:** 47d12d1

| | Old | New |
|-|-----|-----|
| NaivePrevious | Used previous event's T_rec (look-ahead) | Renamed `NaivePrevious_Oracle`; labelled non-ex-ante |
| NaiveLastKnown | Not present | Added: most recent event whose 12-month window closed before current onset |

**Effect on headline claim:** "No model beats NaivePrevious" — this claim used a look-ahead baseline. With `NaiveLastKnown`, comparison is honest.

---

## R-10 — Survival models

**Commit:** (this batch)

New file: `pipeline/src/survival.py`.
Models: penalised Cox PH (`penalizer=0.1`) and Weibull AFT (`penalizer=0.1`) via lifelines.
Evaluation: concordance index. Results in `outputs/results/survival_results.csv` after frozen run.

---

## R-11 — Block bootstrap

**Commit:** (this batch)

| | Old | New |
|-|-----|-----|
| Bootstrap CI | Row-wise iid resampling | Moving-block bootstrap (block=12 months, 1000 resamples) |
| CI column names | `MAE_95CI`, etc. | `MAE_95CI_block_bootstrap`, etc. |
| Wilcoxon label | "exploratory" | Kept as exploratory; block-bootstrap MAE differences are primary |

---

## R-12 — Post-crisis flags

**Commit:** (this batch)

| | Old | New |
|-|-----|-----|
| 2023 events | `external_shock_flag=0` | `post_crisis_recovery=1` via `add_post_crisis_recovery_flag()` |
| 2018 overlap events | Not flagged | `post_crisis_recovery=1` (window reaches into 2019 Easter attacks) |
| Config | No post-crisis key | `post_crisis_tail_years: [2023]` |

---

## R-13 — Unit tests

**Commit:** 47d12d1

Added tests: month rollover, no future years in baseline, fold-safe scaling, TWDI non-negative, TWDI monotone in DTII, NaiveLastKnown, NaivePrevious_Oracle rename.
GitHub Actions CI: `.github/workflows/ci.yml`.

---

## R-14 — Requirements lock

**Commit:** (this batch)

Added `lifelines==0.29.0` to `requirements.txt`. Created `requirements-lock.txt` with full pip-freeze.

---

## R-15 — Sensitivity analysis

**Commit:** 47d12d1

| | Old | New |
|-|-----|-----|
| XGBoost params | Borrowed from last LOOCV fold (`params_list[-1]`) | Config default params (empty dict → XGBoost defaults) |
| Grid | theta in {0.80, 0.90, 1.00}, K in {3, 6, 12} | theta in {0.80, 0.90, 1.00}, K in {3, 6, 12} (widened in config) |

---

## R-16 — Per-event predictions

**Commit:** 47d12d1

`outputs/results/predictions_{experiment}.csv` with columns: `experiment, model, event_id, onset, y_true, y_pred`.

---

## R-17 — DTII word-boundary matching

**Commit:** (this batch)

| | Old | New |
|-|-----|-----|
| Matching | Substring (`district in location_str`) | Word-boundary regex (`\b{district}\b`) |
| Fallback reporting | Silent | Logs unmatched events; `dtii_rule` column added |

---

## R-18 — Housekeeping

**Commit:** 47d12d1

Removed `pipeline/src/__pycache__/sensitivity_analysis.cpython-312.pyc` from git.

---

## R-19 — Explainability checks

**Commit:** (this batch)

Added to `explainability.py`: `permutation_importance`, `grouped_importance`, `compute_vif`.
Outputs: `permutation_importance_{exp}_{model}.csv`, `grouped_importance.csv`, `vif.csv`.

---

## R-20 — Composite event typing

**Commit:** 47d12d1

| | Old | New |
|-|-----|-----|
| 2004 tsunami type | `Flood` (first record's type) | `mixed` (`composite_type` column) |
| member_types | Not stored | `member_types` column (e.g. `"Flood; Earthquake"`) |
