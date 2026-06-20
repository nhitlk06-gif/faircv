# FairCV

Multi-modal fair recruitment scoring system with XAI fairness explanations.

Trained on **FairCVdb** (24 000 synthetic candidate profiles, Complement et al.,
CVPRW 2020).  Implements the three-technique bias mitigation framework and
SHAP-based demographic parity decomposition described in the FairCV proposal.

---

## Project structure

```
faircv-pro/
├── app/
│   └── streamlit_app.py        Main Streamlit application (Recruiter + Candidate view)
├── src/
│   └── faircv/
│       ├── __init__.py         Package declaration
│       ├── data.py             FairCVdb loader, column constants, Dataset container
│       ├── preprocessing.py    Scaling, SBERT encoding, attribute masking, PCA alignment
│       ├── models.py           Model factory: LR / RF / MLP; hyperparameter tuning
│       ├── metrics.py          Performance + fairness metrics (DP Gap, EOO Gap, DI)
│       ├── fusion.py           Early / Late / Weighted Hybrid fusion strategies
│       ├── mitigation.py       Bias mitigation: attr. removal, masking, reweighting
│       ├── xai.py              SHAP explanations: global, per-group, DP decomposition
│       └── pipeline.py         End-to-end audit runner
├── scripts/
│   └── run_audit.py            Command-line audit entry point
├── models/
│   ├── train_models.py         Trains + saves the 2 best models (LR, MLP) from the notebook benchmark
│   └── saved/                  lr_model.pkl, lr_scaler.pkl, mlp_model.pkl, mlp_scaler.pkl, config.pkl
├── data/                       Place FairCVdb.csv here
├── results/                    CLI export target
├── tests/
│   └── test_pipeline.py        Smoke tests (synthetic data, no CSV required)
├── requirements.txt
├── pyproject.toml
└── README.md
```

---

## Quick start

### 1. Install

```bash
pip install -r requirements.txt
# or editable install (recommended):
pip install -e ".[app]"
```

### 2. Place dataset

```bash
cp /path/to/FairCVdb.csv data/
```

### 3. Train and save the scoring models

```bash
python models/train_models.py --csv data/FairCVdb.csv --out models/saved
```

Trains and persists the **two best-performing models** measured in
`FairCV_Models.ipynb` (Setting A / Competency Only / blind label,
19 200-train / 4 800-test split):

| Model | F1 | ROC-AUC | Gender DP Gap | Gender EOO Gap |
|---|---|---|---|---|
| **Logistic Regression** | 0.9658 (best) | 0.9966 (best) | 0.0046 | 0.0008 |
| **MLP** | 0.965 | 0.996 | 0.004 (best, fairest) | 0.001 (best, fairest) |
| ~~Random Forest~~ | 0.9359 | 0.9866 | 0.0115 | 0.0043 |

Random Forest is dropped: it scores lowest on every performance *and*
fairness metric in the notebook benchmark, so the saved model bundle
keeps only the LR (best raw accuracy) and MLP (best fairness, near-tied
accuracy) checkpoints. Outputs land in `models/saved/`:
`lr_model.pkl`, `lr_scaler.pkl`, `mlp_model.pkl`, `mlp_scaler.pkl`,
`config.pkl` (feature schema + recorded metrics for both models).

If `data/FairCVdb.csv` is not yet present, the script falls back to a
small synthetic demo dataset so the rest of the app can still boot —
re-run it once the real CSV is in place for production-quality models.

### 4. Run the Streamlit app

```bash
streamlit run app/streamlit_app.py
```

Open http://localhost:8501.  
Select **Recruiter / Auditor** or **Candidate / Applicant** in the sidebar.

### 5. Run the CLI audit

```bash
# Structured-only audit, export to results/
python scripts/run_audit.py --csv data/FairCVdb.csv --export results/

# Full audit including SBERT multimodal fusion
python scripts/run_audit.py --csv data/FairCVdb.csv --sbert --export results/
```

### 6. Run tests

```bash
python -m pytest tests/ -v
```

---

## Feature settings

| Setting | Features | Fairness risk |
|---------|----------|---------------|
| A: Competency Only | 8 structured scores | Baseline (recommended) |
| B: Competency + Demographics | 8 + gender + ethnicity | Proxy leakage (audit only) |

---

## Bias mitigation techniques

| Technique | Stage | Implementation |
|-----------|-------|----------------|
| T1: Sensitive Attribute Removal | Pre-processing | Use Setting A (drop gender / ethnicity columns) |
| T2: Attribute Masking | Pre-processing | Replace gendered / ethnic tokens with [MASK] before SBERT encoding |
| T3: Sample Reweighting | Training | Inverse-frequency weights per (y, group) cell in clf.fit() |

---

## XAI fairness explanations

Three levels of SHAP analysis are computed in `src/faircv/xai.py`:

- **Global importance** — Mean |SHAP| per feature across the full test set.
- **Per-group SHAP disparity** — Mean |SHAP| per feature, broken down by gender / ethnicity.  
  The SHAP gap = |group1 - group0| exposes which features the model uses differently.
- **DP Gap decomposition** — Δ_j = E[φ_j | A=1] − E[φ_j | A=0].  
  Each feature's signed contribution to the demographic parity gap, summing exactly to the total gap.

---

## Fusion strategies (SBERT)

Enable via `--sbert` flag or sidebar toggle.

| Strategy | Description |
|----------|-------------|
| Structured Only | 8-dim competency features only (baseline) |
| Early Fusion | Concatenation: [E_text ; E_meta] → single classifier |
| Late Fusion | Two independent classifiers; P_final = β·P_text + (1−β)·P_struct |
| Hybrid Fusion | Weighted feature-level combination after PCA alignment |

SBERT model: `all-MiniLM-L6-v2` (384-dim, L2-normalised, attribute-masked).

---

## Reference

Complement et al., *FairCVdb: A Benchmark for Fair CV Screening*, CVPRW 2020.
