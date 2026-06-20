"""
src/faircv/models.py
--------------------
Model factory for FairCV.

Available estimators
--------------------
lr  -- Logistic Regression (L2, lbfgs).
        Baseline: interpretable, exact SHAP via LinearSHAP.
rf  -- Random Forest.
        Captures non-linear competency interactions.
        Feature importance via mean impurity decrease (Gini).
mlp -- Multi-layer Perceptron.
        More expressive decision boundary than LR.

All estimators are wrapped in a sklearn Pipeline that includes
StandardScaler for scale-sensitive models (LR, MLP).  RF is
scale-invariant and receives raw features.

Hyperparameter search
---------------------
``tune_lr``, ``tune_rf``, ``tune_mlp`` run GridSearch / RandomizedSearch
on Setting A, blind label, 3-5 fold CV, scoring='f1'.
Best params are returned and stored in the pipeline results dict.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, StratifiedKFold,
)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_model(kind: str = "lr", seed: int = 42, **kwargs):
    """Return an unfitted sklearn estimator (or Pipeline).

    Parameters
    ----------
    kind : {'lr', 'rf', 'mlp'}
    seed : int
        Random state for reproducibility.
    **kwargs :
        Override default hyperparameters.  Passed directly to the
        underlying estimator constructor.

    Returns
    -------
    sklearn estimator (Pipeline for lr / mlp, bare classifier for rf)
    """
    if kind == "lr":
        params = dict(C=1.0, max_iter=1000, solver="lbfgs",
                      class_weight="balanced", random_state=seed)
        params.update(kwargs)
        return make_pipeline(StandardScaler(), LogisticRegression(**params))

    if kind == "rf":
        params = dict(n_estimators=200, max_depth=10, min_samples_leaf=2,
                      max_features="sqrt", n_jobs=-1,
                      class_weight="balanced", random_state=seed)
        params.update(kwargs)
        return RandomForestClassifier(**params)

    if kind == "mlp":
        params = dict(hidden_layer_sizes=(64, 32), alpha=1e-3,
                      learning_rate_init=5e-4, max_iter=300,
                      early_stopping=True, validation_fraction=0.1,
                      n_iter_no_change=15, solver="adam", random_state=seed)
        params.update(kwargs)
        return make_pipeline(StandardScaler(), MLPClassifier(**params))

    raise ValueError(f"Unknown model kind '{kind}'. Choose from: lr, rf, mlp.")


# ---------------------------------------------------------------------------
# Hyperparameter tuning helpers
# ---------------------------------------------------------------------------

def tune_lr(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    seed: int = 42,
) -> dict:
    """GridSearchCV over C for Logistic Regression.

    Returns
    -------
    dict with keys:
        best_params, best_score, cv_results_df
    """
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    grid = GridSearchCV(
        LogisticRegression(max_iter=1000, solver="lbfgs",
                           class_weight="balanced", random_state=seed),
        param_grid={"C": [0.001, 0.01, 0.1, 1, 10, 100]},
        cv=cv, scoring="f1", n_jobs=-1,
    )
    grid.fit(X_tr, y_tr)
    return {
        "best_params":    grid.best_params_,
        "best_score":     float(grid.best_score_),
        "cv_results_df":  _cv_df(grid),
    }


def tune_rf(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    n_iter: int = 20,
    seed: int = 42,
) -> dict:
    """RandomizedSearchCV for Random Forest hyperparameters.

    Returns
    -------
    dict with keys:
        best_params, best_score, cv_results_df
    """
    import pandas as pd
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        RandomForestClassifier(n_jobs=-1, class_weight="balanced",
                               random_state=seed),
        param_distributions={
            "n_estimators":     [50, 100, 200, 300],
            "max_depth":        [None, 5, 10, 20],
            "min_samples_leaf": [1, 2, 5, 10],
            "max_features":     ["sqrt", "log2"],
        },
        n_iter=n_iter, cv=cv, scoring="f1",
        n_jobs=-1, random_state=seed,
    )
    search.fit(X_tr, y_tr)
    return {
        "best_params":   search.best_params_,
        "best_score":    float(search.best_score_),
        "cv_results_df": _cv_df(search),
    }


def tune_mlp(
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    n_iter: int = 12,
    seed: int = 42,
) -> dict:
    """RandomizedSearchCV for MLP hyperparameters.

    Returns
    -------
    dict with keys:
        best_params, best_score, cv_results_df
    """
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    search = RandomizedSearchCV(
        MLPClassifier(max_iter=200, early_stopping=True,
                      validation_fraction=0.1, n_iter_no_change=10,
                      solver="adam", random_state=seed),
        param_distributions={
            "hidden_layer_sizes": [(64, 32), (128, 64), (64, 32, 16), (32, 16)],
            "alpha":              [0.0001, 0.001, 0.01],
            "learning_rate_init": [0.001, 0.0005, 0.0001],
        },
        n_iter=n_iter, cv=cv, scoring="f1", random_state=seed,
    )
    search.fit(X_tr, y_tr)
    return {
        "best_params":   search.best_params_,
        "best_score":    float(search.best_score_),
        "cv_results_df": _cv_df(search),
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _cv_df(search_obj):
    import pandas as pd
    res = search_obj.cv_results_
    df = pd.DataFrame({
        "params":     [str(p) for p in res["params"]],
        "mean_score": res["mean_test_score"],
        "std_score":  res["std_test_score"],
    }).sort_values("mean_score", ascending=False).reset_index(drop=True)
    return df


def get_feature_importances(model, feature_names: list) -> dict:
    """Extract feature importances from RF or coefficients from LR.

    Returns
    -------
    dict : {feature_name -> importance_value}
    """
    import pandas as pd

    # Unwrap Pipeline if needed
    est = model
    if hasattr(model, "named_steps"):
        for step in model.named_steps.values():
            if hasattr(step, "feature_importances_") or hasattr(step, "coef_"):
                est = step
                break

    if hasattr(est, "feature_importances_"):
        imp = est.feature_importances_
    elif hasattr(est, "coef_"):
        imp = np.abs(est.coef_.ravel())
    else:
        imp = np.ones(len(feature_names)) / len(feature_names)

    return dict(zip(feature_names, imp.tolist()))
