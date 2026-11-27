# Methods

## Two-Stage Zero-Inflated Model (S-12)

### Motivation

38 of 60 events (63%) have T_rec = 0 (immediate recovery). A standard regressor
treats these the same as slow recoveries, inflating predictions near zero. The
two-stage model separates the classification decision from duration estimation.

### Stage 1 — Classifier

**Model:** Logistic Regression (C=1.0, lbfgs solver, max_iter=1000).  
**Target:** binary indicator I(T_rec > 0).  
**Hyperparameter:** C tuned in nested inner CV when N_nonzero > 30 (see `two_stage_model.py`).  
**Evaluation:** Accuracy, F1 (non-zero class), AUC-ROC, majority classifier rate.

Majority rate = max(P(T_rec=0), P(T_rec>0)). AUC > majority_rate indicates
the classifier adds signal beyond the dominant class.

### Stage 2 — Regressor

**Model:** Ridge regression (alpha=1.0).  
**Training data:** Events where T_rec > 0 only (non-zero subset).  
**Alpha tuned:** nested inner CV on non-zero subset; fixed at 1.0 when N_nonzero < 30.  
**Clipping:** predictions clipped to [1, 12] months.

### Combined Prediction Rule

```
if stage_1(x) == 0:  predict 0
else:                 predict clip(stage_2(x), 1, 12)
```

### Outputs

- `two_stage_stage1.json`: stage-1 LOOCV metrics (accuracy, F1, AUC, majority_rate, N_zero, N_nonzero)

## Hyperparameter Search

All single-stage models use `RandomizedSearchCV` with 30 iterations, 5 inner
KFolds (no shuffle), and `neg_mean_absolute_error` as the scoring criterion.
The search runs inside every outer LOOCV fold so the scaler and all parameters
are fit exclusively on training data (fold-safe).

## Validation Schemes

| Label | Description |
|-------|-------------|
| ExpA_LOOCV | Leave-one-out on N=46 pure natural hazard events |
| ExpA_BufferedLOOCV | LOOCV dropping training events within ±12 months of test onset |
| ExpA_LOYO | Leave-one-year-out (hold out all events from one calendar year) |
| ExpA_ExpandingWindow | Train ≤ Y, test = Y+1, Y ∈ 2012..2022 |
| ExpB_Chronological | Train ≤ 2018, test ≥ 2019 (N_train=37, N_test=23) |


## Survival models (revision 5)

Time and event: t* is the first recovered month (1–12). Recovered events have event = 1 and time = T_rec + 1;
censored events (no recovery within 12 months) have event = 0 and time = 12.
Models: penalised Cox proportional hazards (alpha = 0.1), random survival forest (200 trees, min leaf 3),
and a discrete-time hazard model (logistic regression on person-month rows).
Metrics: Harrell C-index; integrated Brier score over months 2–9 with the test set administratively censored
at the training follow-up limit (the censoring distribution cannot be estimated beyond the last training time).
Experiment A has no censored events, so only the pooled LOOCV C-index is reported there.
A model that fails is reported with its error in `survival_results.csv`; a run in which every model fails aborts.

## Default-setting analyses

All sensitivity analyses that use "default-setting XGBoost" (threshold/window/persistence grid, confounding variants,
feature ablation, GDP-lag, baseline-method comparison) call one helper, `src/default_models.py::loocv_default_xgb`
(library defaults, `random_state=42`, `nthread=1`, fold-safe features). The pipeline asserts that the nominal
setting gives the same MAE in every analysis.

## Persistence rule

`recovery.persistence_months = p` requires p consecutive months at or above the threshold; T_rec is the month before
the first month of that run. The sensitivity grid recomputes targets with the same function as the main pipeline,
so persistence variants change the targets (see `n_changed_vs_main` in `sensitivity_analysis.csv`).

## Confounding

Windows: civil_war_end 2007–2009 and easter_covid_crisis 2019–2022 (config `confounding_windows`); the post-crisis tail
(`post_crisis_tail_years`, 2023) is flagged as `post_crisis_tail`. `confounding_sensitivity.csv` reports default-setting XGBoost
and the LOO median baseline for: all events, without civil-war events, without civil-war and tail events, and each flagged group alone.

## Validation schemes

ExpA_LOOCV, ExpA_BufferedLOOCV (12-month buffer), ExpA_LOYO, ExpA_ExpandingWindow, ExpB_Chronological. Naive baselines
(mean, median, ex-ante last known) are computed for every scheme.

## Statistics quoted in the paper

`src/paper_extras.py` writes `paper_extras.json`: zero shares, censoring by year, Kaplan–Meier curves, event spacing,
national arrival changes, fold-safe feature ranges for Experiment B, and MAE by regime from the saved predictions.
`src/make_paper_figures.py` builds Figure 1 from outputs only. `src/make_readme.py` regenerates the README results sections.
