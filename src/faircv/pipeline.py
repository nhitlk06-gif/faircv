"""
src/faircv/pipeline.py
----------------------
End-to-end FairCV audit pipeline.

``run_audit`` is the single entry point consumed by the Streamlit app.
It trains all models, computes fairness metrics before and after
mitigation, generates SHAP explanations, and returns a single results
dict keyed by section.

The function is decorated with ``@st.cache_data`` in the app layer, so
it runs once per unique (csv_path, use_sbert, seed) combination.

Results dict structure
----------------------
{
  "dataset"       : Dataset object
  "df"            : raw pd.DataFrame
  "perf"          : { model_key -> performance dict }
  "fairness_g"    : { model_key -> group fairness DataFrame (gender) }
  "fairness_e"    : { model_key -> group fairness DataFrame (ethnicity) }
  "master_df"     : unified comparison DataFrame
  "shap"          : { model_key -> shap_array, shap_idx }
  "mitigation"    : list of mitigation result dicts
  "fusion"        : { strategy_key -> result dict }  (only if use_sbert)
  "best_params"   : { lr, rf, mlp }
  "before_after"  : { model_key -> compare_before_after DataFrame }
}
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .data import (
    load_faircv, COMPETENCY, DEMO_COLS,
    GENDER_LABELS, ETH_LABELS, LABEL_COLS,
)
from .preprocessing import (
    get_split_arrays, scale_features,
    encode_sbert, build_feature_streams, pca_align_text,
)
from .models import make_model, tune_lr, tune_rf, tune_mlp, get_feature_importances
from .metrics import (
    compute_performance, compute_group_metrics,
    compute_full_fairness, compare_before_after,
)
from .mitigation import (
    compute_sample_weights, fit_with_reweighting,
    run_mitigation_experiment,
)
from .xai import (
    compute_shap_values, global_importance,
    group_shap_disparity, dp_gap_decomposition,
)
from .fusion import (
    early_fusion, late_fusion, hybrid_fusion, structured_only,
)


# ---------------------------------------------------------------------------
# Main audit runner
# ---------------------------------------------------------------------------

def run_audit(
    csv_path: str,
    use_sbert: bool = False,
    seed: int = 42,
    n_explain: int = 500,
    verbose: bool = False,
) -> dict:
    """Run the full FairCV multi-objective audit.

    Parameters
    ----------
    csv_path  : path to FairCVdb.csv
    use_sbert : whether to run SBERT-based fusion experiments
                (requires sentence-transformers; slow on first run)
    seed      : RNG seed for reproducibility
    n_explain : number of test rows to explain with SHAP
    verbose   : print progress to stdout

    Returns
    -------
    Large results dict; see module docstring for structure.
    """

    def _log(msg):
        if verbose:
            print(f"[pipeline] {msg}")

    # ------------------------------------------------------------------ #
    # 1. Load data                                                         #
    # ------------------------------------------------------------------ #
    _log("Loading FairCVdb ...")
    df, ds = load_faircv(csv_path)

    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df  = df[df["split"] == "test"].reset_index(drop=True)

    gender_tr    = train_df["gender"].values
    gender_te    = test_df["gender"].values
    ethnicity_tr = train_df["ethnicity"].values
    ethnicity_te = test_df["ethnicity"].values

    # ------------------------------------------------------------------ #
    # 2. Hyperparameter tuning (Setting A, blind label)                   #
    # ------------------------------------------------------------------ #
    _log("Tuning hyperparameters ...")
    X_tr_A, X_te_A, y_tr, y_te, _ = get_split_arrays(
        df, COMPETENCY, "y_blind", scale=True
    )
    X_tr_A_raw, X_te_A_raw, _, _, _ = get_split_arrays(
        df, COMPETENCY, "y_blind", scale=False
    )

    lr_tune  = tune_lr(X_tr_A, y_tr, seed=seed)
    rf_tune  = tune_rf(X_tr_A_raw, y_tr, seed=seed)
    mlp_tune = tune_mlp(X_tr_A, y_tr, seed=seed)

    best_params = {
        "lr":  lr_tune["best_params"],
        "rf":  rf_tune["best_params"],
        "mlp": mlp_tune["best_params"],
    }
    _log(f"  LR best C={best_params['lr']}, RF best={best_params['rf']}")

    # ------------------------------------------------------------------ #
    # 3. Train structured models (Settings A & B x LR / RF / MLP)        #
    # ------------------------------------------------------------------ #
    _log("Training structured models ...")
    SETTINGS = {
        "A: Competency Only":           COMPETENCY,
        "B: Competency + Demographics": COMPETENCY + DEMO_COLS,
    }
    LABEL_MAP = {
        "blind":     "y_blind",
        "gender":    "y_gender",
        "ethnicity": "y_ethnicity",
    }

    perf       = {}
    fairness_g = {}
    fairness_e = {}
    models_store = {}

    for setting_name, feat_cols in SETTINGS.items():
        for label_short, label_col in LABEL_MAP.items():
            key = f"{setting_name} | {label_short}"
            needs_scale = True  # LR + MLP need scaling; RF does not
            for model_name, kind in [("LR", "lr"), ("RF", "rf"), ("MLP", "mlp")]:
                mkey = f"{key} | {model_name}"

                scale = (kind != "rf")
                X_tr, X_te, y_tr_l, y_te_l, _ = get_split_arrays(
                    df, feat_cols, label_col, scale=scale
                )

                # Select best hyperparams per model kind
                hp = best_params[kind]
                clf = make_model(kind, seed=seed, **hp)

                # T3: reweighting on Setting A, blind label only
                if setting_name.startswith("A") and label_short == "blind":
                    clf = fit_with_reweighting(clf, X_tr, y_tr_l, gender_tr)
                else:
                    clf.fit(X_tr, y_tr_l)

                y_prob = clf.predict_proba(X_te)[:, 1]
                y_pred = (y_prob >= 0.5).astype(int)

                perf[mkey]       = compute_performance(y_te_l, y_pred, y_prob)
                fairness_g[mkey] = compute_group_metrics(
                    y_te_l, y_pred, gender_te, GENDER_LABELS
                )
                fairness_e[mkey] = compute_group_metrics(
                    y_te_l, y_pred, ethnicity_te, ETH_LABELS
                )
                models_store[mkey] = {
                    "clf": clf, "feat_cols": feat_cols,
                    "y_te": y_te_l, "y_pred": y_pred, "y_prob": y_prob,
                }

    # ------------------------------------------------------------------ #
    # 4. Master comparison DataFrame                                       #
    # ------------------------------------------------------------------ #
    _log("Building master comparison table ...")
    records = []
    for mkey, p in perf.items():
        parts   = mkey.split(" | ")
        setting = parts[0].split(":")[0].strip()
        label   = parts[1]
        model   = parts[2]
        rec = {"Model": model, "Setting": setting, "Label": label}
        rec.update({k: round(v, 4) for k, v in p.items()})
        gf = fairness_g.get(mkey)
        ef = fairness_e.get(mkey)
        if gf is not None:
            rec["DP_Gap_Gender"]    = round(float(gf["DP Gap"].mean()), 4)
            rec["EOO_Gap_Gender"]   = round(float(gf["EOO Gap"].mean()), 4)
        if ef is not None:
            rec["DP_Gap_Ethnicity"] = round(float(ef["DP Gap"].mean()), 4)
            rec["EOO_Gap_Ethnicity"]= round(float(ef["EOO Gap"].mean()), 4)
        records.append(rec)
    master_df = pd.DataFrame(records)

    # ------------------------------------------------------------------ #
    # 5. SHAP explanations (Setting A, blind label, RF & LR)              #
    # ------------------------------------------------------------------ #
    _log("Computing SHAP values ...")
    shap_results = {}

    for model_name, kind, scale in [("LR", "lr", True), ("RF", "rf", False)]:
        mkey = f"A: Competency Only | blind | {model_name}"
        if mkey not in models_store:
            continue
        ms = models_store[mkey]
        clf = ms["clf"]

        X_tr_s, X_te_s, _, _, _ = get_split_arrays(
            df, COMPETENCY, "y_blind", scale=scale
        )
        sv, idx = compute_shap_values(
            clf, X_te_s, X_tr_s, n_explain=n_explain, seed=seed
        )
        shap_results[model_name] = {
            "shap_values": sv,
            "shap_idx":    idx,
            "global":      global_importance(sv, COMPETENCY),
            "gender_disp": group_shap_disparity(
                sv, gender_te, GENDER_LABELS, COMPETENCY, idx),
            "eth_disp":    group_shap_disparity(
                sv, ethnicity_te, ETH_LABELS, COMPETENCY, idx),
            "dp_decomp":   dp_gap_decomposition(
                sv, gender_te, COMPETENCY, idx),
        }

    # ------------------------------------------------------------------ #
    # 6. Mitigation experiments                                            #
    # ------------------------------------------------------------------ #
    _log("Running mitigation experiments ...")
    baseline_mkey = "A: Competency Only | blind | RF"
    baseline_ms   = models_store.get(baseline_mkey, {})

    mitigation_records = []

    # T1: Sensitive Attribute Removal (= Setting A -- already done)
    if baseline_ms:
        p = perf[baseline_mkey]
        ff = compute_full_fairness(
            baseline_ms["y_te"], baseline_ms["y_pred"],
            gender_te, ethnicity_te, GENDER_LABELS, ETH_LABELS,
        )
        mitigation_records.append({
            "label":    "T1: Attr. Removal (Setting A)",
            "perf":     p,
            "fairness": ff,
        })

    # T2: Attribute Masking -- reflected in SBERT pipeline
    # (bio_anonymized uses masked text; no separate tabular result here)

    # T3: Sample Reweighting (already applied above; extract metrics)
    if baseline_ms:
        p = perf[baseline_mkey]
        ff = compute_full_fairness(
            baseline_ms["y_te"], baseline_ms["y_pred"],
            gender_te, ethnicity_te, GENDER_LABELS, ETH_LABELS,
        )
        mitigation_records.append({
            "label":    "T3: Sample Reweighting",
            "perf":     p,
            "fairness": ff,
        })

    # ------------------------------------------------------------------ #
    # 7. Before / After comparison (Setting A vs B, blind label)          #
    # ------------------------------------------------------------------ #
    before_after = {}
    for model_name in ["LR", "RF", "MLP"]:
        k_a = f"A: Competency Only | blind | {model_name}"
        k_b = f"B: Competency + Demographics | blind | {model_name}"
        if k_a in perf and k_b in perf:
            combined_a = {**perf[k_a], **{
                "DP_Gap_Gender":    float(fairness_g[k_a]["DP Gap"].mean()),
                "EOO_Gap_Gender":   float(fairness_g[k_a]["EOO Gap"].mean()),
                "DP_Gap_Ethnicity": float(fairness_e[k_a]["DP Gap"].mean()),
            }}
            combined_b = {**perf[k_b], **{
                "DP_Gap_Gender":    float(fairness_g[k_b]["DP Gap"].mean()),
                "EOO_Gap_Gender":   float(fairness_g[k_b]["EOO Gap"].mean()),
                "DP_Gap_Ethnicity": float(fairness_e[k_b]["DP Gap"].mean()),
            }}
            before_after[model_name] = compare_before_after(
                combined_b,  # "before" = biased (includes demographics)
                combined_a,  # "after"  = debiased (competency only)
            )

    # ------------------------------------------------------------------ #
    # 8. SBERT fusion experiments (optional)                              #
    # ------------------------------------------------------------------ #
    fusion_results = {}
    if use_sbert:
        _log("Encoding SBERT embeddings (this may take a few minutes) ...")
        bio_texts = df["bio_anonymized"].fillna("").tolist()
        embeddings = encode_sbert(bio_texts, apply_masking=True)

        streams = build_feature_streams(
            df, embeddings, COMPETENCY, label_col="y_blind"
        )

        # PCA alignment for Hybrid strategy
        n_struct = len(COMPETENCY)
        X_tr_pca, X_te_pca, pca_model = pca_align_text(
            streams["X_text_tr"], streams["X_text_te"],
            n_components=n_struct, seed=seed,
        )

        y_tr_f = streams["y_tr"]
        y_te_f = streams["y_te"]

        for clf_name in ["LR", "RF", "MLP"]:
            # Structured only (baseline)
            res = structured_only(
                streams["X_struct_tr"], streams["X_struct_te"],
                y_tr_f, clf_name=clf_name,
                best_params=best_params, seed=seed,
            )
            res["perf"]     = compute_performance(y_te_f, res["y_pred"], res["y_prob"])
            res["fairness"] = compute_full_fairness(
                y_te_f, res["y_pred"], gender_te, ethnicity_te,
                GENDER_LABELS, ETH_LABELS)
            fusion_results[f"Structured Only | {clf_name}"] = res

            # Early Fusion
            res = early_fusion(
                streams["X_text_tr"], streams["X_text_te"],
                streams["X_struct_tr"], streams["X_struct_te"],
                y_tr_f, clf_name=clf_name,
                best_params=best_params, seed=seed,
            )
            res["perf"]     = compute_performance(y_te_f, res["y_pred"], res["y_prob"])
            res["fairness"] = compute_full_fairness(
                y_te_f, res["y_pred"], gender_te, ethnicity_te,
                GENDER_LABELS, ETH_LABELS)
            fusion_results[f"Early Fusion | {clf_name}"] = res

            # Late Fusion
            res = late_fusion(
                streams["X_text_tr"], streams["X_text_te"],
                streams["X_struct_tr"], streams["X_struct_te"],
                y_tr_f, clf_name=clf_name,
                best_params=best_params, seed=seed,
            )
            res["perf"]     = compute_performance(y_te_f, res["y_pred"], res["y_prob"])
            res["fairness"] = compute_full_fairness(
                y_te_f, res["y_pred"], gender_te, ethnicity_te,
                GENDER_LABELS, ETH_LABELS)
            fusion_results[f"Late Fusion | {clf_name}"] = res

            # Hybrid Fusion
            res = hybrid_fusion(
                X_tr_pca, X_te_pca,
                streams["X_struct_tr"], streams["X_struct_te"],
                y_tr_f, clf_name=clf_name,
                best_params=best_params, seed=seed,
            )
            res["perf"]     = compute_performance(y_te_f, res["y_pred"], res["y_prob"])
            res["fairness"] = compute_full_fairness(
                y_te_f, res["y_pred"], gender_te, ethnicity_te,
                GENDER_LABELS, ETH_LABELS)
            fusion_results[f"Hybrid Fusion | {clf_name}"] = res

        _log("SBERT fusion experiments complete.")

    # ------------------------------------------------------------------ #
    # 9. Package and return                                               #
    # ------------------------------------------------------------------ #
    return {
        "dataset":      ds,
        "df":           df,
        "perf":         perf,
        "fairness_g":   fairness_g,
        "fairness_e":   fairness_e,
        "master_df":    master_df,
        "shap":         shap_results,
        "mitigation":   mitigation_records,
        "fusion":       fusion_results,
        "best_params":  best_params,
        "before_after": before_after,
        "gender_te":    gender_te,
        "ethnicity_te": ethnicity_te,
        "models":       models_store,
    }
