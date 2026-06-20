"""
src/faircv/preprocessing.py
---------------------------
Feature preparation for the FairCV pipeline.

Responsibilities
----------------
1. Scale structured features (StandardScaler fit on train only).
2. Compute SBERT text embeddings with attribute masking for bias mitigation.
3. Separate feature streams for fusion experiments:
       Stream 1  -- text embeddings (384-dim from SBERT)
       Stream 2  -- structured competency features (8-dim)
4. PCA-based dimensionality alignment used by the Weighted Hybrid strategy.

All transforms are fit exclusively on training rows to prevent data leakage.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# ---------------------------------------------------------------------------
# Structured feature scaling
# ---------------------------------------------------------------------------

def scale_features(
    X_tr: np.ndarray,
    X_te: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """Fit StandardScaler on train, apply to both splits.

    Returns
    -------
    X_tr_sc, X_te_sc, scaler
    """
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)
    return X_tr_sc, X_te_sc, scaler


def get_split_arrays(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    scale: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler | None]:
    """Extract train/test arrays for a given feature set and label column.

    Follows the FairCVdb predefined split (rows 0-19199 = train).
    Scaler is fit on train only to prevent leakage.

    Returns
    -------
    X_tr, X_te, y_tr, y_te, scaler
        ``scaler`` is None when ``scale=False``.
    """
    train = df[df["split"] == "train"].reset_index(drop=True)
    test  = df[df["split"] == "test"].reset_index(drop=True)

    X_tr = train[feature_cols].values.astype(float)
    X_te = test[feature_cols].values.astype(float)
    y_tr = train[label_col].values
    y_te = test[label_col].values

    scaler = None
    if scale:
        X_tr, X_te, scaler = scale_features(X_tr, X_te)

    return X_tr, X_te, y_tr, y_te, scaler


# ---------------------------------------------------------------------------
# Attribute masking (Bias Mitigation Technique 2)
# ---------------------------------------------------------------------------

_MASK_PATTERNS = [
    # Gendered pronouns
    r"\b(he|she|his|her|him|himself|herself)\b",
    # Common gendered name patterns (simplified)
    r"\b(Mr\.?|Mrs\.?|Ms\.?|Miss|Sir|Madam)\b",
    # Gender indicator words
    r"\b(male|female|man|woman|boy|girl|gentleman|lady)\b",
    # Nationality / ethnicity indicators that may carry protected info
    r"\b(asian|caucasian|hispanic|latino|black|white|african|european)\b",
]
_MASK_RE = re.compile("|".join(_MASK_PATTERNS), flags=re.IGNORECASE)


def mask_sensitive_attributes(texts: list[str]) -> list[str]:
    """Replace gendered and ethnicity-indicating tokens with [MASK].

    This implements Bias Mitigation Technique 2 (Attribute Masking) from
    the FairCV proposal (Section 12.2).  The operation is applied before
    SBERT encoding so the resulting embeddings cannot reconstruct protected
    attributes from surface-level lexical cues.

    Parameters
    ----------
    texts : list[str]
        Raw or lightly preprocessed biography strings.

    Returns
    -------
    list[str]
        Masked texts, same length as input.
    """
    return [_MASK_RE.sub("[MASK]", t) for t in texts]


# ---------------------------------------------------------------------------
# SBERT text encoding
# ---------------------------------------------------------------------------

def encode_sbert(
    texts: list[str],
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 64,
    apply_masking: bool = True,
) -> np.ndarray:
    """Encode biography texts with Sentence-BERT.

    Uses ``all-MiniLM-L6-v2`` (384-dim) by default.  When ``apply_masking``
    is True, sensitive attribute tokens are replaced with [MASK] before
    encoding (Bias Mitigation Technique 2).

    Parameters
    ----------
    texts : list[str]
        Input biography strings (one per candidate).
    model_name : str
        SentenceTransformers model identifier.
    batch_size : int
        Encoding batch size.
    apply_masking : bool
        Whether to apply attribute masking before encoding.

    Returns
    -------
    np.ndarray, shape (n_samples, 384)
        L2-normalised dense embeddings.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "sentence-transformers is required for SBERT encoding.  "
            "Install with: pip install sentence-transformers"
        ) from exc

    if apply_masking:
        texts = mask_sensitive_attributes(texts)

    sbert = SentenceTransformer(model_name)
    embeddings = sbert.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        convert_to_numpy=True,
        normalize_embeddings=True,  # L2-normalise for cosine compatibility
    )
    return embeddings.astype(np.float32)


# ---------------------------------------------------------------------------
# Feature stream separation
# ---------------------------------------------------------------------------

def build_feature_streams(
    df: pd.DataFrame,
    text_embeddings: np.ndarray,
    struct_cols: list[str],
    label_col: str = "y_blind",
) -> dict:
    """Separate and scale the two feature streams for fusion experiments.

    Stream 1  -- SBERT text embeddings (384-dim), scaled.
    Stream 2  -- Structured competency features (8-dim), scaled.

    Parameters
    ----------
    df : pd.DataFrame
        Full raw dataframe with ``split`` column.
    text_embeddings : np.ndarray, shape (n_total, 384)
        Pre-computed SBERT embeddings aligned with ``df`` row order.
    struct_cols : list[str]
        Structured feature column names (e.g. COMPETENCY).
    label_col : str
        Binary target column.

    Returns
    -------
    dict with keys:
        X_text_tr, X_text_te   : scaled text streams
        X_struct_tr, X_struct_te : scaled structured streams
        y_tr, y_te             : binary targets
        text_scaler, struct_scaler : fitted scalers
    """
    is_train = (df["split"] == "train").values
    is_test  = (df["split"] == "test").values

    # -- Text stream --------------------------------------------------------
    X_text_tr_raw = text_embeddings[is_train]
    X_text_te_raw = text_embeddings[is_test]
    X_text_tr, X_text_te, text_scaler = scale_features(X_text_tr_raw, X_text_te_raw)

    # -- Structured stream --------------------------------------------------
    X_struct_tr_raw, X_struct_te_raw, y_tr, y_te, struct_scaler = get_split_arrays(
        df, struct_cols, label_col, scale=True
    )
    X_struct_tr = X_struct_tr_raw
    X_struct_te = X_struct_te_raw

    return {
        "X_text_tr":     X_text_tr,
        "X_text_te":     X_text_te,
        "X_struct_tr":   X_struct_tr,
        "X_struct_te":   X_struct_te,
        "y_tr":          y_tr,
        "y_te":          y_te,
        "text_scaler":   text_scaler,
        "struct_scaler": struct_scaler,
    }


# ---------------------------------------------------------------------------
# PCA alignment helper (Weighted Hybrid Fusion)
# ---------------------------------------------------------------------------

def pca_align_text(
    X_text_tr: np.ndarray,
    X_text_te: np.ndarray,
    n_components: int,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, PCA]:
    """Project text embeddings to ``n_components`` dimensions via PCA.

    Fit on train only.  Used by the Weighted Hybrid strategy to align
    the 384-dim text stream with the 8-dim structured stream before
    feature-level combination.

    Returns
    -------
    X_tr_pca, X_te_pca, pca_model
    """
    pca = PCA(n_components=n_components, random_state=seed)
    X_tr_pca = pca.fit_transform(X_text_tr)
    X_te_pca = pca.transform(X_text_te)
    return X_tr_pca, X_te_pca, pca
