"""
src/faircv/fusion.py
--------------------
Multi-modal fusion strategies for FairCV (Section 8.4 / 10.2-10.4 of the
FairCV proposal).

Three strategies are implemented
---------------------------------
early_fusion(X_text, X_struct, clf)
    F_early = [E_text ; E_meta]   (concatenation, 384+8 = 392-dim)
    Single classifier trained on the joint feature vector.

late_fusion(X_text_tr, X_text_te, X_struct_tr, X_struct_te,
            y_tr, clf_name, beta, best_params)
    Two independent classifiers, one per modality.
    P_final = beta * P_text + (1-beta) * P_struct   (beta=0.5 by default)

hybrid_fusion(X_text_tr_pca, X_text_te_pca,
              X_struct_tr, X_struct_te, y_tr, clf_name, alpha, best_params)
    Feature-level weighted combination after PCA alignment.
    F = alpha * F_text_pca + (1-alpha) * F_struct   (alpha=0.6 by default)

All functions return a dict with:
    y_pred, y_prob, model (or models for late fusion)
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


# ---------------------------------------------------------------------------
# Internal: instantiate classifier by name
# ---------------------------------------------------------------------------

def _make_clf(clf_name: str, best_params: dict, seed: int = 42):
    """Create a fresh, unfitted classifier instance.

    Parameters
    ----------
    clf_name   : 'LR', 'RF', or 'MLP'
    best_params: hyperparameter dict from tune_lr / tune_rf / tune_mlp
    """
    if clf_name == "LR":
        p = best_params.get("lr", {})
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=p.get("C", 1.0), max_iter=1000, solver="lbfgs",
                class_weight="balanced", random_state=seed,
            ),
        )
    if clf_name == "RF":
        p = best_params.get("rf", {})
        return RandomForestClassifier(
            n_estimators=p.get("n_estimators", 200),
            max_depth=p.get("max_depth", 10),
            min_samples_leaf=p.get("min_samples_leaf", 2),
            max_features=p.get("max_features", "sqrt"),
            class_weight="balanced",
            n_jobs=-1, random_state=seed,
        )
    if clf_name == "MLP":
        p = best_params.get("mlp", {})
        return make_pipeline(
            StandardScaler(),
            MLPClassifier(
                hidden_layer_sizes=p.get("hidden_layer_sizes", (64, 32)),
                alpha=p.get("alpha", 1e-3),
                learning_rate_init=p.get("learning_rate_init", 5e-4),
                max_iter=300, early_stopping=True,
                validation_fraction=0.1, n_iter_no_change=15,
                solver="adam", random_state=seed,
            ),
        )
    raise ValueError(f"Unknown classifier '{clf_name}'. Choose from: LR, RF, MLP.")


# ---------------------------------------------------------------------------
# Early Fusion
# ---------------------------------------------------------------------------

def early_fusion(
    X_text_tr: np.ndarray,
    X_text_te: np.ndarray,
    X_struct_tr: np.ndarray,
    X_struct_te: np.ndarray,
    y_tr: np.ndarray,
    clf_name: str = "RF",
    best_params: dict | None = None,
    seed: int = 42,
) -> dict:
    """Concatenation-based early fusion.

    F_early = [E_text ; E_meta]  =>  single joint classifier.

    Returns
    -------
    dict: y_pred, y_prob, model, input_dim
    """
    if best_params is None:
        best_params = {}

    X_tr = np.concatenate([X_text_tr, X_struct_tr], axis=1)
    X_te = np.concatenate([X_text_te, X_struct_te], axis=1)

    clf = _make_clf(clf_name, best_params, seed)
    clf.fit(X_tr, y_tr)

    y_prob = clf.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "y_pred":     y_pred,
        "y_prob":     y_prob,
        "model":      clf,
        "input_dim":  X_tr.shape[1],
        "strategy":   "Early Fusion",
        "classifier": clf_name,
    }


# ---------------------------------------------------------------------------
# Late Fusion
# ---------------------------------------------------------------------------

def late_fusion(
    X_text_tr: np.ndarray,
    X_text_te: np.ndarray,
    X_struct_tr: np.ndarray,
    X_struct_te: np.ndarray,
    y_tr: np.ndarray,
    clf_name: str = "RF",
    beta: float = 0.5,
    best_params: dict | None = None,
    seed: int = 42,
) -> dict:
    """Decision-level late fusion.

    Two independent classifiers, combined as:
        P_final = beta * P_text + (1-beta) * P_struct

    Parameters
    ----------
    beta : float
        Weight on the text-stream probability (default 0.5 = equal weight).

    Returns
    -------
    dict: y_pred, y_prob, model_text, model_struct
    """
    if best_params is None:
        best_params = {}

    clf_text   = _make_clf(clf_name, best_params, seed)
    clf_struct = _make_clf(clf_name, best_params, seed + 1)

    clf_text.fit(X_text_tr, y_tr)
    clf_struct.fit(X_struct_tr, y_tr)

    p_text   = clf_text.predict_proba(X_text_te)[:, 1]
    p_struct = clf_struct.predict_proba(X_struct_te)[:, 1]

    y_prob = beta * p_text + (1.0 - beta) * p_struct
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "y_pred":       y_pred,
        "y_prob":       y_prob,
        "model_text":   clf_text,
        "model_struct": clf_struct,
        "beta":         beta,
        "strategy":     "Late Fusion",
        "classifier":   clf_name,
    }


# ---------------------------------------------------------------------------
# Weighted Hybrid Fusion
# ---------------------------------------------------------------------------

def hybrid_fusion(
    X_text_tr_pca: np.ndarray,
    X_text_te_pca: np.ndarray,
    X_struct_tr: np.ndarray,
    X_struct_te: np.ndarray,
    y_tr: np.ndarray,
    clf_name: str = "RF",
    alpha: float = 0.6,
    best_params: dict | None = None,
    seed: int = 42,
) -> dict:
    """Feature-level weighted hybrid fusion.

    F = alpha * F_text_pca + (1-alpha) * F_struct

    The text stream must already be PCA-projected to the same
    dimensionality as the structured stream
    (use ``preprocessing.pca_align_text`` before calling).

    Parameters
    ----------
    alpha : float
        Weight on the text stream (default 0.6 = text-dominant, per proposal).

    Returns
    -------
    dict: y_pred, y_prob, model, alpha
    """
    if best_params is None:
        best_params = {}

    X_tr = alpha * X_text_tr_pca + (1.0 - alpha) * X_struct_tr
    X_te = alpha * X_text_te_pca + (1.0 - alpha) * X_struct_te

    clf = _make_clf(clf_name, best_params, seed)
    clf.fit(X_tr, y_tr)

    y_prob = clf.predict_proba(X_te)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "y_pred":     y_pred,
        "y_prob":     y_prob,
        "model":      clf,
        "alpha":      alpha,
        "strategy":   "Hybrid Fusion",
        "classifier": clf_name,
    }


# ---------------------------------------------------------------------------
# Structured-only baseline
# ---------------------------------------------------------------------------

def structured_only(
    X_struct_tr: np.ndarray,
    X_struct_te: np.ndarray,
    y_tr: np.ndarray,
    clf_name: str = "RF",
    best_params: dict | None = None,
    seed: int = 42,
) -> dict:
    """Baseline: structured competency features only (no text modality).

    Used as the 'Before SBERT' reference in ablation studies.

    Returns
    -------
    dict: y_pred, y_prob, model
    """
    if best_params is None:
        best_params = {}

    clf = _make_clf(clf_name, best_params, seed)
    clf.fit(X_struct_tr, y_tr)

    y_prob = clf.predict_proba(X_struct_te)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    return {
        "y_pred":     y_pred,
        "y_prob":     y_prob,
        "model":      clf,
        "strategy":   "Structured Only",
        "classifier": clf_name,
    }
