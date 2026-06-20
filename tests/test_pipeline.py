"""
tests/test_pipeline.py
----------------------
Smoke tests for the FairCV pipeline.

All tests run on synthetic in-memory data -- no FairCVdb.csv required.
The suite is intentionally fast (< 30 s on a single CPU core) so it can
run in CI without GPU or large datasets.

Run with:
    python -m pytest tests/ -v
"""
from __future__ import annotations

import os
import sys
import tempfile
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__))), "src"))

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SEED   = 0
N      = 2400   # small synthetic dataset (100 train / 25 test per group)
N_TR   = 1920
N_TE   = 480

COMPETENCY = [
    "Suitability", "Language 1", "Language 2", "Language 3",
    "Experience", "Education", "Recommendation", "Availability",
]


def _synthetic_df(n: int = N, seed: int = SEED) -> pd.DataFrame:
    """Build a minimal synthetic FairCVdb-compatible DataFrame."""
    rng = np.random.default_rng(seed)

    df = pd.DataFrame(
        rng.uniform(0, 1, (n, len(COMPETENCY))),
        columns=COMPETENCY,
    )
    df["gender"]          = rng.integers(0, 2, n)
    df["ethnicity"]       = rng.integers(0, 3, n)
    df["blind_label"]     = (rng.uniform(0, 1, n) > 0.5).astype(float)
    df["gender_label"]    = df["blind_label"]
    df["ethnicity_label"] = df["blind_label"]
    df["split"]           = "train"
    df.loc[N_TR:, "split"] = "test"
    return df


@pytest.fixture(scope="module")
def syn_df():
    return _synthetic_df()


@pytest.fixture(scope="module")
def syn_csv(syn_df, tmp_path_factory):
    tmp = tmp_path_factory.mktemp("data")
    path = str(tmp / "FairCVdb.csv")
    syn_df.to_csv(path, index=False)
    return path


# ---------------------------------------------------------------------------
# data.py
# ---------------------------------------------------------------------------

class TestData:
    def test_load_faircv(self, syn_csv):
        from faircv.data import load_faircv
        df, ds = load_faircv(syn_csv)
        assert len(df) == N
        assert ds.n == N
        assert "y_blind" in df.columns
        assert "split"   in df.columns
        assert ds.X.shape == (N, len(COMPETENCY))

    def test_split_indices(self, syn_csv):
        from faircv.data import load_faircv
        _, ds = load_faircv(syn_csv)
        tr_idx, te_idx = ds.split(test_frac=0.2, seed=SEED)
        assert len(tr_idx) + len(te_idx) == N
        assert len(set(tr_idx) & set(te_idx)) == 0   # disjoint


# ---------------------------------------------------------------------------
# preprocessing.py
# ---------------------------------------------------------------------------

class TestPreprocessing:
    def test_scale_features(self):
        from faircv.preprocessing import scale_features
        rng = np.random.default_rng(SEED)
        X_tr = rng.normal(size=(100, 8))
        X_te = rng.normal(size=(25, 8))
        Xtr_s, Xte_s, scaler = scale_features(X_tr, X_te)
        assert Xtr_s.shape == X_tr.shape
        np.testing.assert_allclose(Xtr_s.mean(axis=0), 0, atol=1e-9)

    def test_mask_sensitive_attributes(self):
        from faircv.preprocessing import mask_sensitive_attributes
        texts = ["He is a great engineer.", "She has a strong background."]
        masked = mask_sensitive_attributes(texts)
        assert "[MASK]" in masked[0]
        assert "[MASK]" in masked[1]
        assert "He" not in masked[0]
        assert "She" not in masked[1]

    def test_get_split_arrays(self, syn_csv):
        from faircv.data import load_faircv
        from faircv.preprocessing import get_split_arrays
        df, _ = load_faircv(syn_csv)
        X_tr, X_te, y_tr, y_te, scaler = get_split_arrays(
            df, COMPETENCY, "y_blind", scale=True
        )
        assert X_tr.shape[0] == N_TR
        assert X_te.shape[0] == N_TE
        assert scaler is not None


# ---------------------------------------------------------------------------
# models.py
# ---------------------------------------------------------------------------

class TestModels:
    @pytest.mark.parametrize("kind", ["lr", "rf", "mlp"])
    def test_make_model_fit_predict(self, kind):
        from faircv.models import make_model
        rng = np.random.default_rng(SEED)
        X_tr = rng.normal(size=(200, 8))
        y_tr = rng.integers(0, 2, 200)
        X_te = rng.normal(size=(50, 8))
        clf = make_model(kind, seed=SEED)
        clf.fit(X_tr, y_tr)
        prob = clf.predict_proba(X_te)
        assert prob.shape == (50, 2)
        np.testing.assert_allclose(prob.sum(axis=1), 1.0, atol=1e-6)

    def test_get_feature_importances_rf(self):
        from faircv.models import make_model, get_feature_importances
        rng = np.random.default_rng(SEED)
        clf = make_model("rf", seed=SEED)
        clf.fit(rng.normal(size=(200, 8)), rng.integers(0, 2, 200))
        imp = get_feature_importances(clf, COMPETENCY)
        assert len(imp) == len(COMPETENCY)
        assert abs(sum(imp.values()) - 1.0) < 0.01   # importances sum to ~1


# ---------------------------------------------------------------------------
# metrics.py
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_compute_performance(self):
        from faircv.metrics import compute_performance
        rng  = np.random.default_rng(SEED)
        y    = rng.integers(0, 2, 100)
        yhat = rng.integers(0, 2, 100)
        prob = rng.uniform(0, 1, 100)
        m = compute_performance(y, yhat, prob)
        for k in ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "MAE"]:
            assert k in m
            assert 0.0 <= m[k] <= 1.0

    def test_dp_gap_binary(self):
        from faircv.metrics import demographic_parity_gap
        y_pred = np.array([1, 1, 0, 0, 1, 0, 1, 0])
        A      = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        gap = demographic_parity_gap(y_pred, A)
        assert 0.0 <= gap <= 1.0

    def test_disparate_impact(self):
        from faircv.metrics import disparate_impact
        y_pred = np.array([1, 1, 0, 0, 1, 0])
        A      = np.array([0, 0, 0, 1, 1, 1])
        di = disparate_impact(y_pred, A)
        assert 0.0 <= di <= 1.0

    def test_compare_before_after(self):
        from faircv.metrics import compare_before_after
        before = {"F1": 0.60, "DP_Gap": 0.15}
        after  = {"F1": 0.58, "DP_Gap": 0.05}
        df = compare_before_after(before, after)
        assert len(df) == 2
        dp_row = df[df["Metric"] == "DP_Gap"].iloc[0]
        assert dp_row["Better"] is True  # DP gap went down


# ---------------------------------------------------------------------------
# mitigation.py
# ---------------------------------------------------------------------------

class TestMitigation:
    def test_compute_sample_weights(self):
        from faircv.mitigation import compute_sample_weights
        rng = np.random.default_rng(SEED)
        y = rng.integers(0, 2, 200)
        A = rng.integers(0, 2, 200)
        w = compute_sample_weights(y, A)
        assert len(w) == 200
        assert w.min() > 0
        np.testing.assert_allclose(w.mean(), 1.0, rtol=1e-6)

    def test_fit_with_reweighting(self):
        from faircv.mitigation import fit_with_reweighting
        from faircv.models import make_model
        rng = np.random.default_rng(SEED)
        X = rng.normal(size=(200, 8))
        y = rng.integers(0, 2, 200)
        A = rng.integers(0, 2, 200)
        clf = make_model("rf", seed=SEED)
        clf = fit_with_reweighting(clf, X, y, A)
        prob = clf.predict_proba(rng.normal(size=(20, 8)))
        assert prob.shape == (20, 2)


# ---------------------------------------------------------------------------
# xai.py
# ---------------------------------------------------------------------------

class TestXAI:
    @pytest.fixture(scope="class")
    def fitted_rf(self):
        from faircv.models import make_model
        rng = np.random.default_rng(SEED)
        X = rng.normal(size=(200, 8))
        y = rng.integers(0, 2, 200)
        clf = make_model("rf", seed=SEED)
        clf.fit(X, y)
        return clf, X, rng.normal(size=(50, 8))

    def test_compute_shap_values(self, fitted_rf):
        from faircv.xai import compute_shap_values
        clf, X_tr, X_te = fitted_rf
        sv, idx = compute_shap_values(clf, X_te, X_tr, n_explain=20, seed=SEED)
        assert sv.shape == (min(20, len(X_te)), 8)
        assert len(idx) == sv.shape[0]

    def test_global_importance(self, fitted_rf):
        from faircv.xai import compute_shap_values, global_importance
        clf, X_tr, X_te = fitted_rf
        sv, _ = compute_shap_values(clf, X_te, X_tr, n_explain=20, seed=SEED)
        gi = global_importance(sv, COMPETENCY)
        assert len(gi) == 8
        assert "MeanAbsSHAP" in gi.columns

    def test_dp_gap_decomposition(self, fitted_rf):
        from faircv.xai import compute_shap_values, dp_gap_decomposition
        clf, X_tr, X_te = fitted_rf
        sv, idx = compute_shap_values(clf, X_te, X_tr, n_explain=20, seed=SEED)
        rng = np.random.default_rng(SEED)
        A = rng.integers(0, 2, len(X_te))
        dp_df = dp_gap_decomposition(sv, A, COMPETENCY, idx)
        assert len(dp_df) == 8
        assert "Delta" in dp_df.columns
        assert "Share" in dp_df.columns


# ---------------------------------------------------------------------------
# fusion.py
# ---------------------------------------------------------------------------

class TestFusion:
    @pytest.fixture(scope="class")
    def stream_data(self):
        rng = np.random.default_rng(SEED)
        return {
            "X_text_tr":   rng.normal(size=(200, 16)),
            "X_text_te":   rng.normal(size=(50,  16)),
            "X_struct_tr": rng.uniform(0, 1, (200, 8)),
            "X_struct_te": rng.uniform(0, 1, (50,  8)),
            "y_tr":        rng.integers(0, 2, 200),
            "y_te":        rng.integers(0, 2, 50),
        }

    def test_early_fusion(self, stream_data):
        from faircv.fusion import early_fusion
        s = stream_data
        res = early_fusion(
            s["X_text_tr"], s["X_text_te"],
            s["X_struct_tr"], s["X_struct_te"],
            s["y_tr"], clf_name="RF",
        )
        assert res["y_pred"].shape == (50,)
        assert res["y_prob"].shape == (50,)

    def test_late_fusion(self, stream_data):
        from faircv.fusion import late_fusion
        s = stream_data
        res = late_fusion(
            s["X_text_tr"], s["X_text_te"],
            s["X_struct_tr"], s["X_struct_te"],
            s["y_tr"], clf_name="RF",
        )
        assert res["y_pred"].shape == (50,)

    def test_hybrid_fusion(self, stream_data):
        from faircv.fusion import hybrid_fusion
        s = stream_data
        res = hybrid_fusion(
            s["X_text_tr"][:, :8], s["X_text_te"][:, :8],
            s["X_struct_tr"], s["X_struct_te"],
            s["y_tr"], clf_name="RF",
        )
        assert res["y_pred"].shape == (50,)


# ---------------------------------------------------------------------------
# pipeline.py (end-to-end smoke test -- no SBERT)
# ---------------------------------------------------------------------------

class TestPipeline:
    def test_run_audit_smoke(self, syn_csv):
        from faircv.pipeline import run_audit
        out = run_audit(syn_csv, use_sbert=False, seed=SEED, n_explain=50)
        assert "master_df"   in out
        assert "shap"        in out
        assert "before_after" in out
        assert "mitigation"  in out
        assert len(out["master_df"]) > 0

    def test_master_df_columns(self, syn_csv):
        from faircv.pipeline import run_audit
        out = run_audit(syn_csv, use_sbert=False, seed=SEED, n_explain=50)
        df  = out["master_df"]
        for col in ["Model", "Setting", "Label", "F1", "Accuracy"]:
            assert col in df.columns, f"Missing column: {col}"

    def test_shap_keys(self, syn_csv):
        from faircv.pipeline import run_audit
        out = run_audit(syn_csv, use_sbert=False, seed=SEED, n_explain=50)
        for m in ["LR", "RF"]:
            assert m in out["shap"], f"SHAP missing for {m}"
            assert "global"    in out["shap"][m]
            assert "dp_decomp" in out["shap"][m]
