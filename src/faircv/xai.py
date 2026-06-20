"""
src/faircv/xai.py
-----------------
Explainability (XAI) layer for FairCV.

Provides SHAP-based feature attributions at three levels of granularity:

1. Global importance
   Mean |SHAP| per feature across the entire test set.
   Shows which features the model relies on most overall.

2. Per-group SHAP disparity  (gender / ethnicity)
   Mean |SHAP| per feature, broken down by protected group.
   The *SHAP gap* = |mean_group1 - mean_group0| per feature exposes
   which features the model uses differently for different demographic
   groups -- the primary XAI signal for fairness auditing.

3. DP Gap decomposition
   SHAP difference: E[phi_j | A=1] - E[phi_j | A=0]
   Each feature's signed contribution to the demographic parity gap in
   score space, summing exactly to the total DP gap (local accuracy).

Implementation note
-------------------
Uses a dependency-free LinearSHAP for logistic / linear models and a
sampling-based approximation for tree / neural models, avoiding a hard
dependency on the ``shap`` package.  If ``shap`` is installed its
TreeExplainer / LinearExplainer will be preferred automatically.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Dependency-free SHAP implementations
# ---------------------------------------------------------------------------

def _logit(model, X: np.ndarray) -> np.ndarray:
    """Return the model score (logit) for the positive class."""
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    p = np.clip(model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


class LinearSHAP:
    """Exact Shapley values for a fitted linear / logistic sklearn model.

    phi_j(x) = w_j * (x_j - E[x_j])

    This satisfies local accuracy exactly:  f(x) = phi_0 + sum_j phi_j.
    """

    def __init__(self, model, background: np.ndarray):
        # Unwrap Pipeline
        est = model
        if hasattr(model, "named_steps"):
            for step in model.named_steps.values():
                if hasattr(step, "coef_"):
                    est = step
                    break
        self.coef  = est.coef_.ravel().astype(float)
        self.bg_mean = np.asarray(background).mean(axis=0)

    def shap_values(self, X: np.ndarray) -> np.ndarray:
        return (np.asarray(X, dtype=float) - self.bg_mean) * self.coef


class SamplingSHAP:
    """Monte-Carlo Shapley values for arbitrary sklearn models.

    Uses the permutation-based estimator of Strumbelj & Kononenko (2014).
    Suitable for RF and MLP where exact SHAP is not available without
    additional dependencies.
    """

    def __init__(self, model, background: np.ndarray,
                 n_perm: int = 64, seed: int = 42, max_bg: int = 128):
        self.model = model
        bg = np.asarray(background, dtype=float)
        if len(bg) > max_bg:
            rng = np.random.default_rng(seed)
            bg  = bg[rng.choice(len(bg), max_bg, replace=False)]
        self.bg  = bg
        self.n_perm = n_perm
        self.rng = np.random.default_rng(seed)

    def shap_values(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        n, d = X.shape
        phi = np.zeros((n, d))
        for _ in range(self.n_perm):
            perm = self.rng.permutation(d)
            bg_row = self.bg[self.rng.integers(len(self.bg), size=n)]
            x_with = bg_row.copy()
            x_wout = bg_row.copy()
            for j in perm:
                x_with[:, j] = X[:, j]
                s_with = _logit(self.model, x_with)
                s_wout = _logit(self.model, x_wout)
                phi[:, j] += (s_with - s_wout)
                x_wout[:, j] = X[:, j]
        return phi / self.n_perm


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_shap_values(
    model,
    X_te: np.ndarray,
    X_tr_background: np.ndarray,
    n_explain: int = 500,
    seed: int = 42,
) -> np.ndarray:
    """Compute SHAP values for a random subset of test rows.

    Selects LinearSHAP for LR models, SamplingSHAP for RF / MLP.
    If the optional ``shap`` package is installed, prefers its
    TreeExplainer for RF.

    Parameters
    ----------
    model            : fitted sklearn estimator or Pipeline
    X_te             : test feature matrix (n_test, d)
    X_tr_background  : training data for background distribution
    n_explain        : number of test rows to explain (random subsample)
    seed             : RNG seed

    Returns
    -------
    np.ndarray, shape (min(n_explain, n_test), d)
        SHAP values in model score (logit) space.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X_te), min(n_explain, len(X_te)), replace=False)
    Xs  = X_te[idx]

    # Detect linear model
    _is_linear = False
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "coef_"):
                _is_linear = True
                break
    elif hasattr(model, "coef_"):
        _is_linear = True

    if _is_linear:
        # Apply preprocessing transforms to background and subset
        bg_proc = _transform_all_but_last(model, X_tr_background)
        xs_proc = _transform_all_but_last(model, Xs)
        explainer = LinearSHAP(model, bg_proc)
        return explainer.shap_values(xs_proc), idx

    # Try shap.TreeExplainer for RF
    try:
        import shap  # optional dependency
        import sklearn.ensemble
        est = model
        if hasattr(model, "named_steps"):
            est = list(model.named_steps.values())[-1]
        if isinstance(est, sklearn.ensemble.RandomForestClassifier):
            explainer = shap.TreeExplainer(est)
            return explainer.shap_values(Xs)[:, :, 1], idx
    except Exception:
        pass

    # Fallback: SamplingSHAP
    explainer = SamplingSHAP(model, X_tr_background, seed=seed)
    return explainer.shap_values(Xs), idx


def global_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
) -> pd.DataFrame:
    """Mean |SHAP| per feature (global importance).

    Returns
    -------
    pd.DataFrame sorted by importance descending,
    columns: ['Feature', 'MeanAbsSHAP']
    """
    imp = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"Feature": feature_names, "MeanAbsSHAP": imp})
        .sort_values("MeanAbsSHAP", ascending=False)
        .reset_index(drop=True)
    )


def group_shap_disparity(
    shap_values: np.ndarray,
    group_arr: np.ndarray,
    group_labels: dict,
    feature_names: list[str],
    shap_idx: np.ndarray,
) -> pd.DataFrame:
    """Per-group mean |SHAP| and SHAP gap.

    Parameters
    ----------
    shap_idx : indices of the rows explained (returned by compute_shap_values)

    Returns
    -------
    pd.DataFrame with one column per group + 'SHAP_Gap' column,
    indexed by feature name.
    """
    A_sub = group_arr[shap_idx]
    records = {}
    for gid, gname in group_labels.items():
        mask = A_sub == gid
        if mask.any():
            records[gname] = np.abs(shap_values[mask]).mean(axis=0)
        else:
            records[gname] = np.zeros(shap_values.shape[1])

    df = pd.DataFrame(records, index=feature_names)
    cols = list(records.keys())
    if len(cols) >= 2:
        df["SHAP_Gap"] = (df[cols[0]] - df[cols[1]]).abs()
    return df.round(6)


def dp_gap_decomposition(
    shap_values: np.ndarray,
    group_arr: np.ndarray,
    feature_names: list[str],
    shap_idx: np.ndarray,
) -> pd.DataFrame:
    """Per-feature contribution to the demographic parity gap.

    Delta_j = E[phi_j | A=1] - E[phi_j | A=0]

    sum(Delta_j) equals the score-level DP gap exactly (up to SHAP
    estimator error), providing a precise, per-feature attribution of
    where the bias enters the model's decision.

    Returns
    -------
    pd.DataFrame columns: ['Feature', 'Delta', 'AbsDelta', 'Share']
    Sorted by |Delta| descending.
    """
    A_sub = group_arr[shap_idx]
    m1 = A_sub == 1
    m0 = A_sub == 0

    if m1.any() and m0.any():
        delta = shap_values[m1].mean(axis=0) - shap_values[m0].mean(axis=0)
    else:
        delta = np.zeros(shap_values.shape[1])

    abs_delta  = np.abs(delta)
    total      = abs_delta.sum() + 1e-12
    share      = abs_delta / total

    df = pd.DataFrame({
        "Feature":  feature_names,
        "Delta":    delta.round(6),
        "AbsDelta": abs_delta.round(6),
        "Share":    share.round(4),
    }).sort_values("AbsDelta", ascending=False).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# Plot helpers (return matplotlib Figure objects for Streamlit)
# ---------------------------------------------------------------------------

INK   = "#1b2330"
COMP  = "#2E86AB"
PROXY = "#E4572E"
GOOD  = "#2E9E5B"
WARN  = "#E0A100"


def plot_global_importance(
    shap_values: np.ndarray,
    feature_names: list[str],
    title: str = "Global Feature Importance (Mean |SHAP|)",
    max_features: int = 10,
) -> plt.Figure:
    """Horizontal bar chart of mean |SHAP| values."""
    imp = np.abs(shap_values).mean(axis=0)
    order = np.argsort(imp)[-max_features:]
    names = [feature_names[i] for i in order]
    vals  = imp[order]

    fig, ax = plt.subplots(figsize=(7, max(3, len(names) * 0.45)))
    ax.barh(range(len(names)), vals, color=COMP, edgecolor="white")
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Mean |SHAP value|", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", color=INK)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    return fig


def plot_group_shap_bars(
    disp_df: pd.DataFrame,
    group_labels: dict,
    title: str = "SHAP by Group",
) -> plt.Figure:
    """Grouped horizontal bar chart: mean |SHAP| per group per feature."""
    group_cols = [v for v in group_labels.values() if v in disp_df.columns]
    n_feat = len(disp_df)
    h      = 0.8 / len(group_cols)
    colors = [COMP, PROXY, GOOD, WARN]

    fig, ax = plt.subplots(figsize=(8, max(4, n_feat * 0.6)))
    y = np.arange(n_feat)
    for k, (col, color) in enumerate(zip(group_cols, colors)):
        offset = (k - len(group_cols) / 2 + 0.5) * h
        ax.barh(y + offset, disp_df[col].values, h * 0.9,
                label=col, color=color, alpha=0.85, edgecolor="white")

    ax.set_yticks(y)
    ax.set_yticklabels(disp_df.index.tolist(), fontsize=9)
    ax.set_xlabel("Mean |SHAP value|", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", color=INK)
    ax.legend(fontsize=8, loc="lower right")
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    return fig


def plot_dp_decomposition(
    dp_df: pd.DataFrame,
    title: str = "Per-Feature DP Gap Decomposition (Delta_j)",
) -> plt.Figure:
    """Signed bar chart showing each feature's contribution to DP gap."""
    names  = dp_df["Feature"].tolist()
    deltas = dp_df["Delta"].tolist()
    colors = [PROXY if d > 0 else COMP for d in deltas]

    fig, ax = plt.subplots(figsize=(8, max(3.5, len(names) * 0.5)))
    ax.barh(range(len(names)), deltas, color=colors, edgecolor="white")
    ax.axvline(0, color="gray", linewidth=0.8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("Delta_j  (E[phi|A=1] - E[phi|A=0])", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", color=INK)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    # Annotate values
    for i, (v, name) in enumerate(zip(deltas, names)):
        ax.text(v + (0.0002 if v >= 0 else -0.0002), i,
                f"{v:+.4f}", va="center",
                ha="left" if v >= 0 else "right", fontsize=7.5)
    fig.tight_layout()
    return fig


def plot_shap_gender_comparison(
    shap_values: np.ndarray,
    group_arr: np.ndarray,
    feature_names: list[str],
    shap_idx: np.ndarray,
    group_labels: dict,
    title: str = "SHAP by Protected Group",
) -> plt.Figure:
    """Side-by-side mean |SHAP| bars per gender / group."""
    A_sub = group_arr[shap_idx]
    groups = sorted(group_labels.keys())
    means  = {gid: np.abs(shap_values[A_sub == gid]).mean(axis=0)
              for gid in groups if (A_sub == gid).any()}

    all_means = np.sum(list(means.values()), axis=0)
    order     = np.argsort(all_means)
    feat_ord  = [feature_names[i] for i in order]

    h       = 0.35
    colors  = [COMP, PROXY, GOOD, WARN]
    fig, ax = plt.subplots(figsize=(9, max(4, len(feat_ord) * 0.55)))
    y = np.arange(len(feat_ord))
    for k, gid in enumerate(groups):
        if gid not in means:
            continue
        offset = (k - len(groups) / 2 + 0.5) * h
        ax.barh(y + offset, means[gid][order], h * 0.85,
                label=group_labels[gid], color=colors[k % len(colors)],
                alpha=0.85, edgecolor="white")

    ax.set_yticks(y)
    ax.set_yticklabels(feat_ord, fontsize=9)
    ax.set_xlabel("Mean |SHAP value|", fontsize=9)
    ax.set_title(title, fontsize=11, fontweight="bold", color=INK)
    ax.legend(fontsize=8)
    for sp in ["top", "right"]:
        ax.spines[sp].set_visible(False)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _transform_all_but_last(model, X: np.ndarray) -> np.ndarray:
    """Apply all pipeline steps except the final estimator."""
    if not hasattr(model, "named_steps"):
        return np.asarray(X)
    steps = list(model.named_steps.values())
    Xt = np.asarray(X, dtype=float)
    for step in steps[:-1]:
        Xt = step.transform(Xt)
    return Xt
