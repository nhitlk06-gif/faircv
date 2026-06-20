"""
src/faircv/metrics.py
---------------------
Performance and fairness metrics for FairCV, aligned with the FairCVdb
evaluation protocol (Complement et al., CVPRW 2020).

Performance metrics
-------------------
  Accuracy, Precision, Recall, F1, MAE, ROC-AUC

Fairness metrics
----------------
  Demographic Parity (DP Gap)
      |P(y_hat=1|A=1) - P(y_hat=1|A=0)|
      Measures whether the selection rate is equal across groups.

  Equal Opportunity (EOO Gap)
      |TPR(A=1) - TPR(A=0)|
      Measures whether qualified candidates are equally likely to be
      selected regardless of their protected attribute.

  Disparate Impact (DI)
      min_group_rate / max_group_rate
      Values below 0.80 indicate potential discrimination under EEOC
      four-fifths (80%) rule.

All functions accept numpy arrays and work with binary or multi-class
protected attributes.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, mean_absolute_error, roc_auc_score,
)


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------

def compute_performance(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
) -> dict:
    """Return a dict of standard classification metrics.

    Parameters
    ----------
    y_true : binary ground-truth labels
    y_pred : binary model predictions
    y_prob : positive-class probability scores (required for AUC / MAE)
    """
    m: dict = {
        "Accuracy":  float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall":    float(recall_score(y_true, y_pred, zero_division=0)),
        "F1":        float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_prob is not None:
        m["ROC-AUC"] = float(roc_auc_score(y_true, y_prob))
        m["MAE"]     = float(mean_absolute_error(y_true, y_prob))
    return m


# ---------------------------------------------------------------------------
# Fairness metrics
# ---------------------------------------------------------------------------

def positive_rate(y_pred: np.ndarray, A: np.ndarray, a) -> float:
    """P(y_hat=1 | A=a)."""
    mask = A == a
    return float(y_pred[mask].mean()) if mask.any() else 0.0


def demographic_parity_gap(y_pred: np.ndarray, A: np.ndarray) -> float:
    """|P(y_hat=1|A=1) - P(y_hat=1|A=0)|  (binary protected attribute)."""
    return abs(positive_rate(y_pred, A, 1) - positive_rate(y_pred, A, 0))


def equal_opportunity_gap(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    A: np.ndarray,
) -> float:
    """|TPR(A=1) - TPR(A=0)|."""
    def _tpr(a: int) -> float:
        mask = (A == a) & (y_true == 1)
        return float(y_pred[mask].mean()) if mask.any() else 0.0
    return abs(_tpr(1) - _tpr(0))


def disparate_impact(y_pred: np.ndarray, A: np.ndarray) -> float:
    """min_rate / max_rate across all groups (multi-class aware).

    DI < 0.80 is flagged as a potential discrimination risk under the
    EEOC four-fifths rule.
    """
    groups = np.unique(A)
    rates = {g: float(y_pred[A == g].mean()) for g in groups}
    lo, hi = min(rates.values()), max(rates.values())
    return float(lo / hi) if hi > 0 else 1.0


def compute_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    group_arr: np.ndarray,
    group_labels: dict,
) -> pd.DataFrame:
    """Per-group accuracy, F1, selection rate, TPR, DP Gap, EOO Gap.

    Parameters
    ----------
    group_arr   : integer group assignments (0, 1, ...).
    group_labels: {int -> str} display names.

    Returns
    -------
    pd.DataFrame indexed by group display name.
    """
    rows = []
    for gid, gname in group_labels.items():
        mask = group_arr == gid
        if not mask.any():
            continue
        yt, yp = y_true[mask], y_pred[mask]
        pos_rate = float(yp.mean())
        tpr = float(yt[yp == 1].sum() / yt.sum()) if yt.sum() > 0 else 0.0
        rows.append({
            "Group":      gname,
            "N":          int(mask.sum()),
            "Accuracy":   float(accuracy_score(yt, yp)),
            "F1":         float(f1_score(yt, yp, zero_division=0)),
            "Pos Rate":   pos_rate,
            "TPR":        tpr,
        })
    result = pd.DataFrame(rows).set_index("Group")
    if len(result) >= 2:
        result["DP Gap"]  = result["Pos Rate"].max() - result["Pos Rate"].min()
        result["EOO Gap"] = result["TPR"].max() - result["TPR"].min()
    else:
        result["DP Gap"]  = 0.0
        result["EOO Gap"] = 0.0
    return result


def compute_full_fairness(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    gender_arr: np.ndarray,
    ethnicity_arr: np.ndarray,
    gender_labels: dict,
    eth_labels: dict,
) -> dict:
    """Compute DP Gap, EOO Gap, and Disparate Impact for gender and ethnicity.

    Returns
    -------
    dict with keys:
        DP_Gap_Gender, EOO_Gap_Gender, DI_Gender,
        DP_Gap_Ethnicity, EOO_Gap_Ethnicity, DI_Ethnicity
    """
    g_df = compute_group_metrics(y_true, y_pred, gender_arr, gender_labels)
    e_df = compute_group_metrics(y_true, y_pred, ethnicity_arr, eth_labels)

    return {
        "DP_Gap_Gender":     round(float(g_df["DP Gap"].mean()), 4),
        "EOO_Gap_Gender":    round(float(g_df["EOO Gap"].mean()), 4),
        "DI_Gender":         round(disparate_impact(y_pred, gender_arr), 4),
        "DP_Gap_Ethnicity":  round(float(e_df["DP Gap"].mean()), 4),
        "EOO_Gap_Ethnicity": round(float(e_df["EOO Gap"].mean()), 4),
        "DI_Ethnicity":      round(disparate_impact(y_pred, ethnicity_arr), 4),
    }


# ---------------------------------------------------------------------------
# Before / After comparison helper
# ---------------------------------------------------------------------------

def compare_before_after(
    before: dict,
    after: dict,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Build a Before -> After comparison DataFrame.

    Parameters
    ----------
    before, after : dict
        Performance or fairness metric dicts.
    metrics : list[str], optional
        Subset of keys to include.  Defaults to all shared keys.

    Returns
    -------
    pd.DataFrame with columns ['Metric', 'Before', 'After', 'Delta', 'Better']
    """
    keys = metrics if metrics else sorted(set(before) & set(after))
    rows = []
    for k in keys:
        b, a = float(before.get(k, 0)), float(after.get(k, 0))
        # Lower is better for gap metrics; higher for perf metrics
        lower_better = any(kw in k for kw in ("Gap", "MAE", "DP_", "EOO_"))
        improved = (a < b) if lower_better else (a > b)
        rows.append({
            "Metric": k,
            "Before": round(b, 4),
            "After":  round(a, 4),
            "Delta":  round(a - b, 4),
            "Better": improved,
        })
    return pd.DataFrame(rows)
