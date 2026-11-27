"""
Validation engines for the two experimental designs.

Experiment A (pure natural hazards):
    LOOCV — preserves maximum sample size.
    buffered_loocv_split — drops events within ±buffer_months of test onset (R-08).
    leave_one_year_out_split — hold out all events from one calendar year.

Experiment B (all events including compound shocks):
    Strict chronological split — train <= 2018, test >= 2019.
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
    R-08: shuffle=False — preserves temporal order inside inner folds.
    """
    kf = KFold(n_splits=n_splits, shuffle=False)
    return list(kf.split(X_train))


def buffered_loocv_split(
    df: pd.DataFrame,
    buffer_months: int = 12,
) -> list:
    """
    R-08: Buffered LOOCV — for each test event, drop from training any event
    whose onset is within ±buffer_months of the test onset.

    Yields (train_idx, test_idx) where test_idx is a 1-element array.
    Requires df["start_date"] (Timestamp).
    """
    if "start_date" not in df.columns:
        raise ValueError("buffered_loocv_split requires df['start_date'].")

    onsets = pd.to_datetime(df["start_date"].values)
    n = len(df)
    buffer = pd.DateOffset(months=buffer_months)
    splits = []

    for i in range(n):
        t = onsets[i]
        train_mask = np.array([
            abs((onsets[j] - t).days) > buffer_months * 30
            for j in range(n)
            if j != i
        ])
        all_others = np.array([j for j in range(n) if j != i])
        train_idx = all_others[train_mask]
        test_idx = np.array([i])
        if len(train_idx) > 0:
            splits.append((train_idx, test_idx))

    return splits


def expanding_window_split(
    df: pd.DataFrame,
    start_year: int = 2012,
    end_year: int = 2022,
) -> list:
    """
    S-06: Expanding-window validation. Train on events up to year Y, test on Y+1.
    Yields (train_idx, test_idx) for Y = start_year..end_year where both sets non-empty.
    """
    years = df["start_year"].values
    splits = []
    for y in range(start_year, end_year + 1):
        train_idx = np.where(years <= y)[0]
        test_idx = np.where(years == y + 1)[0]
        if len(train_idx) >= 5 and len(test_idx) > 0:
            splits.append((train_idx, test_idx))
    return splits


def leave_one_year_out_split(
    df: pd.DataFrame,
) -> list:
    """
    R-08: Leave-one-year-out — hold out all events from one calendar year,
    train on all other years.

    Yields (train_idx, test_idx) for each unique year.
    Requires df["start_year"].
    """
    years = df["start_year"].values
    unique_years = sorted(set(years))
    splits = []

    for yr in unique_years:
        test_idx = np.where(years == yr)[0]
        train_idx = np.where(years != yr)[0]
        if len(test_idx) > 0 and len(train_idx) > 0:
            splits.append((train_idx, test_idx))

    return splits
