"""
src/faircv/mitigation.py
------------------------
Lightweight bias mitigation techniques for FairCV
(FairCV Proposal, Section 12).

Three techniques are implemented
----------------------------------
T1  Sensitive Attribute Removal
    Remove gender and ethnicity columns from the feature matrix before
    training.  This is equivalent to Setting A in the FairCVdb paper.
    Since Setting A is the default throughout the pipeline, T1 is
    represented by *not* using DEMO_COLS.

T2  Attribute Masking (text-level)
    Replace gendered / ethnicity-indicating tokens in biography text with
    [MASK] before SBERT encoding.  Implemented in preprocessing.py
    (``mask_sensitive_attributes``).  The output is directly used in all
    SBERT pipelines by default (``apply_masking=True``).

T3  Sample Reweighting
    Compute inverse-frequency sample weights per (y, sensitive_group) cell,
    then pass them to ``clf.fit(X, y, sample_weight=w)``.
    This forces the model to pay proportionally more attention to under-
    represented (y, group) combinations, reducing demographic parity gap
    without changing the feature set.

All three techniques are documented, non-adversarial, and apply at the
data or training stage -- they do not require model architecture changes.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# T1 -- Sensitive Attribute Removal
# ---------------------------------------------------------------------------

def remove_sensitive_columns(
    df: pd.DataFrame,
    sensitive_cols: list[str],
) -> pd.DataFrame:
    """Drop protected-attribute columns from a DataFrame.

    Parameters
    ----------
    df            : input feature DataFrame
    sensitive_cols: column names to drop (e.g. ['gender', 'ethnicity'])

    Returns
    -------
    pd.DataFrame with sensitive columns removed.
    """
    cols_to_drop = [c for c in sensitive_cols if c in df.columns]
    return df.drop(columns=cols_to_drop)


# ---------------------------------------------------------------------------
# T3 -- Sample Reweighting
# ---------------------------------------------------------------------------

def compute_sample_weights(
    y_train: np.ndarray,
    sensitive_train: np.ndarray,
) -> np.ndarray:
    """Inverse-frequency sample weights per (y, group) cell.

    Each sample receives a weight inversely proportional to its
    (label, group) cell frequency, normalised so the mean weight = 1.

    Parameters
    ----------
    y_train       : binary training labels
    sensitive_train: integer group assignments (gender or ethnicity)

    Returns
    -------
    np.ndarray of shape (n_train,)
        Sample weights for use in clf.fit(X, y, sample_weight=w).

    Notes
    -----
    This matches the reweighting formula in FairCV Proposal Section 12.3:

        w_i = N / (K * n_{y_i, g_i})

    where N = total samples, K = number of (y, g) cells,
    n_{y,g} = cell count.
    """
    y_arr = np.asarray(y_train).ravel()
    g_arr = np.asarray(sensitive_train).ravel()

    pairs = list(zip(y_arr.tolist(), g_arr.tolist()))
    counts = Counter(pairs)
    n_total = len(y_arr)
    n_cells = len(counts)

    weights = np.array(
        [n_total / (n_cells * counts[(y, g)]) for y, g in pairs],
        dtype=float,
    )
    # Normalise so mean weight = 1 (keeps effective learning rate stable)
    weights /= weights.mean()
    return weights


def fit_with_reweighting(
    clf,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    sensitive_tr: np.ndarray,
) -> object:
    """Fit a classifier with inverse-frequency sample reweighting.

    Automatically handles Pipeline objects by routing ``sample_weight``
    to the final estimator step.

    Parameters
    ----------
    clf          : unfitted sklearn estimator or Pipeline
    X_tr         : training feature matrix
    y_tr         : training labels
    sensitive_tr : protected attribute array for weight computation

    Returns
    -------
    Fitted classifier (same object, mutated in place).
    """
    weights = compute_sample_weights(y_tr, sensitive_tr)

    if hasattr(clf, "named_steps"):
        # sklearn Pipeline: route sample_weight to final step
        final_step = list(clf.named_steps.keys())[-1]
        fit_params = {f"{final_step}__sample_weight": weights}
        clf.fit(X_tr, y_tr, **fit_params)
    else:
        clf.fit(X_tr, y_tr, sample_weight=weights)

    return clf


# ---------------------------------------------------------------------------
# Mitigation experiment runner
# ---------------------------------------------------------------------------

def run_mitigation_experiment(
    X_tr: np.ndarray,
    X_te: np.ndarray,
    y_tr: np.ndarray,
    y_te: np.ndarray,
    gender_tr: np.ndarray,
    gender_te: np.ndarray,
    ethnicity_te: np.ndarray,
    clf_factory,          # callable() -> unfitted classifier
    compute_perf_fn,      # callable(y_true, y_pred, y_prob) -> dict
    compute_fairness_fn,  # callable(y_true, y_pred, gender, ethnicity) -> dict
    label: str = "Experiment",
) -> dict:
    """Run one mitigation technique and return a result record.

    Parameters
    ----------
    clf_factory       : zero-argument callable that returns a fresh classifier.
    compute_perf_fn   : metric function from metrics.compute_performance.
    compute_fairness_fn: metric function from metrics.compute_full_fairness.
    label             : descriptive name for the experiment.

    Returns
    -------
    dict with keys:
        label, y_pred, y_prob, perf, fairness
    """
    clf = clf_factory()
    clf = fit_with_reweighting(clf, X_tr, y_tr, gender_tr)

    y_prob = clf.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    perf     = compute_perf_fn(y_te, y_pred, y_prob)
    fairness = compute_fairness_fn(y_te, y_pred, gender_te, ethnicity_te)

    return {
        "label":     label,
        "y_pred":    y_pred,
        "y_prob":    y_prob,
        "perf":      perf,
        "fairness":  fairness,
        "model":     clf,
    }
