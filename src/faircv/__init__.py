"""
FairCV -- multi-modal fair recruitment scoring system.

Modules
-------
data        : FairCVdb loader, feature constants, Dataset container.
preprocessing : Binarisation, scaling, feature-stream separation.
models      : Model factory (LR / RF / MLP) + SBERT encoder.
metrics     : Performance (Accuracy, F1, AUC, MAE) and fairness
              (DP Gap, EOO Gap, Disparate Impact) metrics, aligned
              with the FairCVdb evaluation protocol.
fusion      : Early / Late / Weighted-Hybrid fusion strategies.
mitigation  : Bias mitigation: attribute removal, masking, reweighting.
xai         : SHAP-based global and per-group explanations.
pipeline    : End-to-end audit runner returning a results dict consumed
              by the Streamlit app.
"""

__version__ = "1.0.0"
__all__ = [
    "data", "preprocessing", "models", "metrics",
    "fusion", "mitigation", "xai", "pipeline",
]
