"""
Validation engines for the two experimental designs.

Experiment A (pure natural hazards, N=67):
    Leave-One-Out Cross-Validation (LOOCV) — preserves maximum sample size.

Experiment B (all events including compound shocks, N=86):
    Strict chronological split — train on 2000-2018, test on 2019-2025.
    No random shuffling. Eliminates temporal data leakage.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut, KFold


def loocv_split(X: pd.DataFrame, y: pd.Series):
    """
    Yield (train_idx, test_idx) pairs for LOOCV.
    Each fold trains on N-1 events and tests on the 1 held-out event.
    """
    loo = LeaveOneOut()
    indices = np.arange(len(X))
    for train_idx, test_idx in loo.split(indices):
        yield train_idx, test_idx


def chronological_split(
    X: pd.DataFrame,
    y: pd.Series,
    df_full: pd.DataFrame,
    train_cutoff_year: int = 2018,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Split on Start_Year: train <= train_cutoff_year, test > train_cutoff_year.

    Returns (train_indices, test_indices) as numpy arrays.
    The df_full must be aligned with X (same row order).
    """
    years = df_full["start_year"].values
    train_idx = np.where(years <= train_cutoff_year)[0]
    test_idx = np.where(years > train_cutoff_year)[0]

    if len(train_idx) == 0:
        raise ValueError(f"No training events with start_year <= {train_cutoff_year}")
    if len(test_idx) == 0:
        raise ValueError(f"No test events with start_year > {train_cutoff_year}")

    return train_idx, test_idx


def inner_cv_folds(X_train: np.ndarray, n_splits: int = 5, seed: int = 42):
    """
    Inner k-fold splits for nested hyperparameter search.
    Used inside GridSearchCV to prevent test-set leakage.
    """
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(kf.split(X_train))
