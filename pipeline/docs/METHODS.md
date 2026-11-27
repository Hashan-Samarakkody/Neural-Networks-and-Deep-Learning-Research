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
