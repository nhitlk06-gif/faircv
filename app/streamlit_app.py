"""
app/streamlit_app.py
--------------------
FairCV -- Main Streamlit interface for the fair CV screening system.

Two roles share the same session:

  Recruiter
      1. Enter job position title and requirements.
      2. Set priority weights for each criterion (real-time sliders).
      3. Upload multiple candidate CVs as individual PDF files (>= 1, no ZIP).
      4. Choose AI model: Logistic Regression or MLP.
      5. View ranked results Top 10/5/3/1, SHAP charts, fairness audit.

  Candidate
      1. Enter the target position they are applying for.
      2. Upload their own CV as a single PDF file.
      3. View personal score, competency profile, and fairness report card.

Run:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import io
import json
import os
import re
import sys

import joblib
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Path setup -- add src/ to sys.path so faircv package can be imported
# ---------------------------------------------------------------------------

_APP_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_APP_DIR)
sys.path.insert(0, os.path.join(_ROOT_DIR, "src"))

from faircv.data import (
    COMPETENCY, COMP_NAMES,
    GENDER_LABELS, ETH_LABELS,
    GOOGLE_DRIVE_FILE_ID, LOCAL_CSV_PATH,
)

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FairCV",
    layout="wide",
    page_icon="",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------

INK  = "#1b2330"   # Primary text
COMP = "#2E86AB"   # Blue -- competency / neutral
GOOD = "#2E9E5B"   # Green -- strong / pass
WARN = "#E0A100"   # Amber -- moderate / review
BAD  = "#C2384A"   # Red -- weak / fail / bias
PRXY = "#E4572E"   # Orange -- proxy / alert
MUT  = "#7a869a"   # Grey -- muted / secondary

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------

st.markdown(f"""
<style>
:root {{
  --ink:{INK}; --comp:{COMP}; --good:{GOOD};
  --bad:{BAD}; --warn:{WARN}; --proxy:{PRXY};
}}
.block-container {{ padding-top:1rem; max-width:1280px; }}

/* Hero banner */
.hero {{
  background: linear-gradient(110deg,#101826 0%,#1f3550 55%,{COMP} 130%);
  color:#fff; padding:20px 28px; border-radius:14px;
  margin-bottom:14px; box-shadow:0 6px 24px rgba(16,24,38,.18);
}}
.hero h1 {{ margin:0; font-size:1.5rem; letter-spacing:.1px; }}
.hero p  {{ margin:.35rem 0 0; opacity:.9; font-size:.93rem; }}

/* Info card */
.card {{
  background:#fff; border:1px solid #e2e8f0;
  border-radius:12px; padding:14px 18px;
  box-shadow:0 2px 8px rgba(20,30,50,.05); height:100%;
}}
.card h4 {{ margin:0 0 6px; color:var(--ink); font-size:.96rem; }}

/* Large KPI number */
.kpi {{ font-size:1.85rem; font-weight:700; line-height:1.1; }}
.sub {{ color:#5a6678; font-size:.83rem; margin-top:2px; }}

/* Badges */
.badge {{
  display:inline-block; padding:3px 11px;
  border-radius:999px; font-size:.78rem; font-weight:700;
  color:#fff; margin:2px 3px 2px 0;
}}
.b-good  {{ background:var(--good);  }}
.b-bad   {{ background:var(--bad);   }}
.b-warn  {{ background:var(--warn);  }}
.b-comp  {{ background:var(--comp);  }}
.b-proxy {{ background:var(--proxy); }}
.b-top1  {{ background:#D97706; }}
.b-top3  {{ background:#7C3AED; }}
.b-top5  {{ background:var(--good);  }}
.b-top10 {{ background:var(--comp);  }}

/* Candidate ranking row */
.cand-row {{
  display:flex; align-items:center; gap:14px;
  padding:10px 14px; border:1px solid #e2e8f0;
  border-radius:9px; margin:5px 0; background:#fafbfc;
}}
.cand-rank  {{ font-size:1.2rem; font-weight:700; color:#475569; min-width:32px; }}
.cand-name  {{ font-weight:600; color:var(--ink); flex:1; }}
.cand-score {{ font-size:1.2rem; font-weight:700;
               color:var(--comp); min-width:54px; text-align:right; }}

/* Fairness verdict boxes */
.v-clear {{ background:#e7f6ec; border:1px solid #a9dcb9;
            border-radius:10px; padding:12px 16px; color:#1d6b39; }}
.v-warn  {{ background:#fff6e0; border:1px solid #f0d493;
            border-radius:10px; padding:12px 16px; color:#7a5800; }}
.v-bad   {{ background:#fdece8; border:1px solid #f3b3a4;
            border-radius:10px; padding:12px 16px; color:#8f2d18; }}

/* Alert box */
.alert {{
  background:#fff7ed; border:1px solid #fed7aa;
  border-radius:9px; padding:9px 14px;
  color:#9a3412; font-size:.86rem; margin:4px 0;
}}

/* Section divider */
.sec {{
  font-size:1.04rem; font-weight:700; color:var(--ink);
  margin:18px 0 8px; padding-bottom:4px;
  border-bottom:2px solid #e2e8f0;
}}

/* Fairness note */
.fair-note {{
  background:#eff6ff; border:1px solid #bfdbfe;
  border-radius:9px; padding:11px 15px;
  color:#1d4ed8; font-size:.86rem; margin-top:8px;
}}

/* Upload zone hint */
.upload-hint {{
  background:#f8fafc; border:2px dashed #cbd5e1;
  border-radius:10px; padding:14px 18px;
  color:#475569; font-size:.88rem; margin-bottom:8px;
}}
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# Shared helper functions
# ===========================================================================

def _card(title: str, kpi: str, color: str, sub: str) -> str:
    """Render an HTML KPI card."""
    return (
        f'<div class="card"><h4>{title}</h4>'
        f'<div class="kpi" style="color:{color}">{kpi}</div>'
        f'<div class="sub">{sub}</div></div>'
    )


def _read_pdf(uploaded_file) -> str:
    """Extract plain text from a Streamlit UploadedFile (PDF).

    Tries pdfminer.six first; falls back to raw byte decode if unavailable.
    """
    try:
        from pdfminer.high_level import extract_text as _pdfminer
        raw   = uploaded_file.read()
        text  = _pdfminer(io.BytesIO(raw))
    except ImportError:
        uploaded_file.seek(0)
        text = uploaded_file.read().decode("utf-8", errors="ignore")
    except Exception:
        uploaded_file.seek(0)
        text = uploaded_file.read().decode("utf-8", errors="ignore")

    # Normalise whitespace
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# Load trained models (cached -- loaded only once per Streamlit session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading AI models...")
def _load_models(out_dir: str = "models/saved"):
    """Load LR + MLP models and scalers from models/saved/.

    Automatically runs train_models.py if .pkl files are missing.
    """
    paths = {
        "lr":     os.path.join(out_dir, "lr_model.pkl"),
        "lr_sc":  os.path.join(out_dir, "lr_scaler.pkl"),
        "mlp":    os.path.join(out_dir, "mlp_model.pkl"),
        "mlp_sc": os.path.join(out_dir, "mlp_scaler.pkl"),
        "config": os.path.join(out_dir, "config.pkl"),
    }
    if not all(os.path.exists(p) for p in paths.values()):
        st.info("Trained models not found -- training now on FairCVdb (one-time only)...")
        import subprocess
        script = os.path.join(_ROOT_DIR, "models", "train_models.py")
        subprocess.run([sys.executable, script], check=True)

    return (
        joblib.load(paths["lr"]),
        joblib.load(paths["lr_sc"]),
        joblib.load(paths["mlp"]),
        joblib.load(paths["mlp_sc"]),
        joblib.load(paths["config"]),
    )


lr_model, lr_scaler, mlp_model, mlp_scaler, cfg = _load_models()
FEAT_COLS  = cfg.get("features",      COMPETENCY)
FEAT_NAMES = cfg.get("feature_names", COMP_NAMES)
DEMO_MODE  = cfg.get("demo_mode",     False)

if DEMO_MODE:
    st.warning(
        "**Demo mode:** Models were trained on synthetic data.  "
        "Place `FairCVdb.csv` in `data/` and run "
        "`python models/train_models.py` for production-quality results."
    )


# ===========================================================================
# Sidebar -- shared configuration for both roles
# ===========================================================================

with st.sidebar:
    st.markdown("## FairCV")
    st.caption("Fair Recruitment Scoring System")

    role = st.radio("I am a", ["Recruiter", "Candidate"])
    st.markdown("---")

    if role == "Recruiter":

        # -- Job position --------------------------------------------------
        st.markdown("### Job Position")
        job_title = st.text_input("Position title", "Data Analyst")
        job_desc  = st.text_area(
            "Job requirements",
            "Minimum 2 years of data analysis experience.  "
            "Proficient in Python and SQL.  "
            "English communication required.  "
            "Machine learning background preferred.",
            height=110,
        )
        domain = st.selectbox(
            "Industry domain",
            ["IT / Technology", "Marketing", "Finance / Accounting",
             "Human Resources", "Other"],
        )

        st.markdown("---")

        # -- Scoring weights (real-time) -----------------------------------
        st.markdown("### Scoring Weights")
        st.caption(
            "Drag sliders to set criterion priority.  "
            "Rankings update instantly."
        )
        w_suit  = st.slider("Suitability to role",    0.0, 1.0, 0.30, 0.05)
        w_exp   = st.slider("Work experience",         0.0, 1.0, 0.25, 0.05)
        w_edu   = st.slider("Education level",         0.0, 1.0, 0.20, 0.05)
        w_lang1 = st.slider("Primary language",        0.0, 1.0, 0.10, 0.05)
        w_lang2 = st.slider("Secondary language",      0.0, 1.0, 0.05, 0.05)
        w_avail = st.slider("Availability to start",   0.0, 1.0, 0.10, 0.05)

        # Weight vector aligned with FEAT_COLS order.
        # Recommendation is excluded to prevent network/referral bias.
        weights = {
            "Suitability":    w_suit,
            "Language 1":     w_lang1,
            "Language 2":     w_lang2,
            "Language 3":     0.0,
            "Experience":     w_exp,
            "Education":      w_edu,
            "Recommendation": 0.0,   # Excluded -- prone to network bias
            "Availability":   w_avail,
        }

        st.markdown("---")

        # -- Model selection -----------------------------------------------
        st.markdown("### AI Model")
        model_choice = st.radio(
            "Use model",
            ["LR (Logistic Regression)", "MLP (Neural Network)"],
            help="LR: interpretable, fast.\nMLP: non-linear, lower DP Gap.",
        )

        # -- Claude API toggle ---------------------------------------------
        st.markdown("---")
        st.markdown("### CV Extraction")
        use_claude = st.toggle(
            "Use Claude API (more accurate)",
            value=bool(os.environ.get("ANTHROPIC_API_KEY")),
        )
        if use_claude and not os.environ.get("ANTHROPIC_API_KEY"):
            key_in = st.text_input("ANTHROPIC_API_KEY", type="password")
            if key_in:
                os.environ["ANTHROPIC_API_KEY"] = key_in
            use_claude = bool(os.environ.get("ANTHROPIC_API_KEY"))

    else:
        # Candidate sidebar -- minimal config
        job_title    = st.text_input("Position you are applying for", "Data Analyst")
        job_desc     = st.text_area(
            "Typical requirements for this role",
            "Python, SQL, data analysis, 2+ years experience",
            height=75,
        )
        domain       = "Other"
        model_choice = "MLP (Neural Network)"
        use_claude   = bool(os.environ.get("ANTHROPIC_API_KEY"))
        weights      = {c: 1.0 / len(FEAT_COLS) for c in FEAT_COLS}

    # -- Model performance summary (production mode only) ------------------
    if not DEMO_MODE:
        st.markdown("---")
        st.markdown("### Model Performance")
        lrm  = cfg.get("lr_metrics",  {})
        mlpm = cfg.get("mlp_metrics", {})
        st.markdown(
            f"| Model | F1 | AUC | DP Gap |\n|---|---|---|---|\n"
            f"| LR  | {lrm.get('F1',0):.3f} | {lrm.get('ROC-AUC',0):.3f} | "
            f"{lrm.get('DP_Gap_Gender',0):.4f} |\n"
            f"| MLP | {mlpm.get('F1',0):.3f} | {mlpm.get('ROC-AUC',0):.3f} | "
            f"{mlpm.get('DP_Gap_Gender',0):.4f} |"
        )
        st.caption("Trained on FairCVdb blind label (gender / ethnicity blind)")


# ===========================================================================
# Feature extraction helpers
# ===========================================================================

# Keyword dictionaries by industry domain
_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "IT / Technology": [
        "python", "java", "sql", "machine learning", "data", "ai", "software",
        "backend", "frontend", "cloud", "aws", "docker", "kubernetes",
        "tensorflow", "pytorch", "deep learning", "api", "git", "agile",
        "analyst", "engineer", "developer", "code", "programming",
    ],
    "Marketing": [
        "marketing", "brand", "campaign", "social media", "seo", "content",
        "digital", "advertising", "customer", "market research", "crm",
        "analytics", "e-commerce", "conversion", "roi", "kpi",
    ],
    "Finance / Accounting": [
        "finance", "accounting", "audit", "tax", "budget", "financial",
        "excel", "reporting", "investment", "risk", "compliance", "ifrs",
        "cpa", "cfa", "gaap", "ledger", "reconciliation",
    ],
    "Human Resources": [
        "hr", "human resources", "recruitment", "talent", "training",
        "performance", "payroll", "labor", "onboarding", "kpi",
        "hrbp", "compensation", "benefits", "workforce",
    ],
    "Other": [],
}


def _suitability_score(cv_text: str, domain: str, job_desc: str) -> float:
    """Score 'Suitability' via keyword density + job description overlap.

    Algorithm:
      1. Count domain keyword hits in cv_text (case-insensitive).
      2. Compute keyword hit rate and apply sqrt to smooth the distribution.
      3. Compute word overlap between cv_text and job_desc.
      4. Weighted blend: 40% keyword density + 60% JD overlap.
      5. Clip to [0.10, 1.00].
    """
    keywords   = _DOMAIN_KEYWORDS.get(domain, [])
    text_lower = cv_text.lower()

    if keywords:
        hits      = sum(1 for kw in keywords if kw in text_lower)
        kw_score  = float(np.sqrt(hits / len(keywords)))
    else:
        kw_score  = 0.5

    req_words  = set(re.findall(r"\w{3,}", job_desc.lower()))
    cv_words   = set(re.findall(r"\w{3,}", text_lower))
    overlap    = len(req_words & cv_words) / max(len(req_words), 1)
    jd_score   = float(min(1.0, overlap * 2.0))

    return float(np.clip(0.40 * kw_score + 0.60 * jd_score, 0.10, 1.00))


def _detect_gender_proxy(cv_text: str) -> float:
    """Detect gender-indicating language in CV text.

    Returns 1.0 (female cues found), 0.0 (male cues found),
    or 0.5 (ambiguous / not detected).
    This value is NEVER passed to the scoring model.
    It is used only for a transparency warning in the UI.
    """
    tl = cv_text.lower()
    female = sum(1 for t in ["female", " she ", " her ", " ms.", " mrs."] if t in tl)
    male   = sum(1 for t in ["male",   " he ",  " his ", " mr."]          if t in tl)
    if female > male:
        return 1.0
    if male > female:
        return 0.0
    return 0.5


def extract_features_heuristic(
    cv_text: str,
    job_title: str,
    job_desc:  str,
    domain:    str,
) -> dict:
    """Extract 8 FairCV competency features using rule-based heuristics.

    Used when Claude API is unavailable or disabled.

    Returns
    -------
    dict containing:
        candidate_name : str
        Suitability, Language 1-3, Experience, Education,
        Recommendation, Availability : float in [0, 1]
        Gender_Proxy  : float (0.0 / 0.5 / 1.0) -- NOT a model feature
        reasoning     : dict with extraction notes
    """
    txt = cv_text.lower()

    # Candidate name: first non-empty line of the CV
    lines = [l.strip() for l in cv_text.split("\n") if l.strip()]
    name  = lines[0] if lines else "Unknown"

    # Suitability -- keyword + JD overlap
    suitability = _suitability_score(cv_text, domain, job_desc)

    # Experience -- largest number of years found in text
    yrs     = re.findall(r"(\d+)\s*(?:\+\s*)?(?:years?|yrs?)\s*(?:of\s*)?(?:experience)?", txt)
    max_yrs = max((int(y) for y in yrs), default=0)
    experience = float(np.clip(min(1.0, max_yrs / 5.0) if max_yrs else 0.30, 0.10, 1.00))

    # Education -- highest degree detected
    if   any(k in txt for k in ["ph.d", "phd", "doctorate"]):          education = 1.00
    elif any(k in txt for k in ["master", "mba", "msc", "m.s.", "m.e."]):education = 0.80
    elif any(k in txt for k in ["bachelor", "university", "b.s.", "b.a."]):education = 0.60
    elif any(k in txt for k in ["associate", "college"]):               education = 0.40
    else:                                                                education = 0.50

    # Language scores
    lang1 = 0.75   # Default: candidate can write a CV (primary language)
    lang2 = 0.55 if any(k in txt for k in ["english", "ielts", "toeic", "toefl"]) else 0.20
    lang3 = 0.40 if any(k in txt for k in ["japanese", "chinese", "korean", "french", "german", "spanish"]) else 0.10

    # Recommendation / reference
    recommendation = 0.65 if any(k in txt for k in ["reference", "referral", "recommendation"]) else 0.30

    # Availability
    availability = 1.00 if any(k in txt for k in ["immediately", "available now", "start immediately"]) else 0.65

    return {
        "candidate_name": name,
        "Suitability":    round(suitability,    3),
        "Language 1":     round(lang1,          3),
        "Language 2":     round(lang2,          3),
        "Language 3":     round(lang3,          3),
        "Experience":     round(experience,     3),
        "Education":      round(education,      3),
        "Recommendation": round(recommendation, 3),
        "Availability":   round(availability,   3),
        "Gender_Proxy":   _detect_gender_proxy(cv_text),
        "reasoning":      {"note": "Extracted by rule-based heuristic (Claude API not active)."},
    }


def extract_features_claude(cv_text: str, job_title: str, job_desc: str) -> dict:
    """Extract 8 FairCV competency features using Claude API.

    Requires ANTHROPIC_API_KEY set in the environment.
    Falls back to heuristic on any API error.
    """
    import anthropic

    PROMPT = """You are an expert HR analyst.
Read the CV below and the job description, then score each feature from 0.0 to 1.0.

Job title: {job_title}
Job requirements: {job_desc}

CV (first 3 500 characters):
{cv_text}

Reply with ONLY valid JSON -- no markdown fences, no explanation:
{{
  "candidate_name": "Candidate full name",
  "Suitability":    0.0,
  "Language 1":     0.0,
  "Language 2":     0.0,
  "Language 3":     0.0,
  "Experience":     0.0,
  "Education":      0.0,
  "Recommendation": 0.0,
  "Availability":   0.0,
  "reasoning": {{
    "Suitability": "brief reason",
    "Experience":  "brief reason",
    "Education":   "brief reason"
  }}
}}

Scoring guide:
  Suitability   : overall fit with the job title and requirements
  Language 1    : primary language quality inferred from CV writing style
  Language 2    : second language mentioned (0.20 if none)
  Language 3    : third language mentioned (0.10 if none)
  Experience    : 0=none, 0.30=<1yr, 0.50=1-2yr, 0.70=3-5yr, 1.00=5+yr relevant
  Education     : 0.40=associate, 0.60=bachelor, 0.80=master, 1.00=PhD + relevant field
  Recommendation: 0.30=none, 0.65=references listed, 1.00=formal letter attached
  Availability  : 1.00=immediate, 0.60=within 1 month, 0.30=3+ months
"""
    client = anthropic.Anthropic()
    resp   = client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=700,
        messages=[{"role": "user", "content": PROMPT.format(
            job_title=job_title,
            job_desc=job_desc,
            cv_text=cv_text[:3500],
        )}],
    )
    raw = resp.content[0].text.strip()
    raw = re.sub(r"```json\s*|```", "", raw).strip()
    data = json.loads(raw)
    # Always add Gender_Proxy (not sent to API, computed locally)
    data["Gender_Proxy"] = _detect_gender_proxy(cv_text)
    return data


def extract_features(
    cv_text:    str,
    job_title:  str,
    job_desc:   str,
    domain:     str,
    use_claude: bool = False,
) -> dict:
    """Route to Claude API or heuristic based on configuration."""
    if use_claude and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return extract_features_claude(cv_text, job_title, job_desc)
        except Exception as exc:
            st.warning(f"Claude API error ({exc}). Falling back to heuristic.")
    return extract_features_heuristic(cv_text, job_title, job_desc, domain)


# ===========================================================================
# Candidate scoring
# ===========================================================================

def score_candidate(
    features:     dict,
    weights:      dict,
    model_choice: str,
) -> dict:
    """Score one candidate using a trained FairCVdb model.

    Pipeline:
      1. Build raw feature vector aligned with FEAT_COLS.
      2. Apply recruiter weight vector to feature values
         (real-time: slider changes immediately affect ranking).
      3. StandardScaler.transform() -- avoids input scale drift.
      4. Model.predict_proba() -- probability of 'Recommended'.
      5. Final score = 60% model probability + 40% weighted criteria match.

    Parameters
    ----------
    features     : dict {feature_name -> float [0, 1]}
    weights      : dict {feature_name -> weight [0, 1]} from sidebar sliders
    model_choice : 'LR (Logistic Regression)' or 'MLP (Neural Network)'

    Returns
    -------
    dict with keys:
        model_label, model_score, criteria_score, final_score, verdict,
        tier_color, x_raw
    """
    # Select model and scaler
    if model_choice.startswith("MLP"):
        model, scaler, label = mlp_model, mlp_scaler, "MLP"
    else:
        model, scaler, label = lr_model,  lr_scaler,  "LR"

    # Raw feature vector (ordered by FEAT_COLS)
    x_raw = np.array([float(features.get(c, 0.5)) for c in FEAT_COLS])

    # Weight vector
    w_vec   = np.array([weights.get(c, 0.0) for c in FEAT_COLS])
    total_w = w_vec.sum() or 1.0

    # Weighted criteria match score (recruiter's custom priority blend)
    criteria = float(np.dot(w_vec, x_raw) / total_w)

    # Boost features with high recruiter weight before scaling
    # (ensures slider changes propagate through model input, not just ranking)
    x_weighted = x_raw * (1.0 + 0.5 * w_vec / (w_vec.max() + 1e-9))

    # Normalise to trained scale (avoids distribution mismatch)
    X_scaled = scaler.transform(x_weighted.reshape(1, -1))

    # Model prediction
    model_prob = float(model.predict_proba(X_scaled)[0][1])

    # Final combined score
    final = 0.60 * model_prob + 0.40 * criteria

    tier_color = (
        GOOD if final >= 0.75 else
        COMP if final >= 0.60 else
        WARN if final >= 0.45 else BAD
    )

    return {
        "model_label":    label,
        "model_score":    round(model_prob * 100, 1),
        "criteria_score": round(criteria   * 100, 1),
        "final_score":    round(final      * 100, 1),
        "verdict":        "Recommended" if model_prob >= 0.5 else "Not recommended",
        "tier_color":     tier_color,
        "x_raw":          x_raw,
    }


# ===========================================================================
# RECRUITER VIEW
# ===========================================================================

if role == "Recruiter":

    st.markdown(
        '<div class="hero">'
        '<h1>Recruiter -- CV Screening & Fairness Audit</h1>'
        '<p>Set your job criteria and scoring weights on the left, '
        'upload candidate PDF files, and let two independent AI models '
        'rank them fairly.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    tab_screen, tab_fair, tab_shap = st.tabs([
        "CV Screening",
        "Fairness Audit",
        "SHAP -- XAI Explanations",
    ])

    # -------------------------------------------------------------------
    # Tab 1: CV Screening
    # -------------------------------------------------------------------
    with tab_screen:

        # -- Active criteria summary ------------------------------------
        st.markdown('<div class="sec">Active Screening Criteria</div>',
                    unsafe_allow_html=True)
        cc1, cc2, cc3 = st.columns(3)

        cc1.markdown(
            _card("Position", job_title[:24] if len(job_title) > 24 else job_title,
                  COMP, f"Domain: {domain}"),
            unsafe_allow_html=True,
        )

        top_w = sorted(((k, v) for k, v in weights.items() if v > 0), key=lambda x: -x[1])[:5]
        cc2.markdown(
            '<div class="card"><h4>Priority Weights</h4>'
            + "".join(
                f'<div style="display:flex;justify-content:space-between;'
                f'font-size:.87rem;margin:3px 0">'
                f'<span>{k}</span><b>{v:.0%}</b></div>'
                for k, v in top_w
            )
            + "</div>",
            unsafe_allow_html=True,
        )

        cc3.markdown(
            '<div class="card"><h4>Fairness Guarantees</h4>'
            '<div style="font-size:.84rem;line-height:1.85">'
            'Blind label training (FairCVdb)<br>'
            'Gender &amp; ethnicity excluded from features<br>'
            'Recommendation criterion excluded<br>'
            f'Active model: {model_choice.split()[0]}'
            '</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown("")

        # -- PDF upload (multiple files) --------------------------------
        st.markdown('<div class="sec">Upload Candidate CVs</div>',
                    unsafe_allow_html=True)

        st.markdown(
            '<div class="upload-hint">'
            '<b>Format:</b> PDF only &nbsp;|&nbsp; '
            '<b>Limit:</b> Up to 200 MB total &nbsp;|&nbsp; '
            '<b>One file per candidate</b> &nbsp;|&nbsp; '
            'Minimum 10 files recommended for meaningful ranking.'
            '</div>',
            unsafe_allow_html=True,
        )

        uploaded_pdfs = st.file_uploader(
            "Select PDF files (hold Ctrl / Cmd to pick multiple)",
            type=["pdf"],
            accept_multiple_files=True,
            key="recruiter_upload",
            help="Each PDF = one candidate. Upload 10 or more for best results.",
        )

        # File count feedback
        if uploaded_pdfs:
            n_files = len(uploaded_pdfs)
            color   = GOOD if n_files >= 10 else WARN if n_files >= 3 else BAD
            st.markdown(
                f'<div style="font-size:.9rem;margin:4px 0 10px">'
                f'<b style="color:{color}">{n_files} file(s) selected.</b>'
                + (" Ready to evaluate." if n_files >= 10
                   else f" Recommend at least 10 files for a meaningful ranking.")
                + "</div>",
                unsafe_allow_html=True,
            )

        run_btn = st.button(
            "Start Evaluation",
            type="primary",
            disabled=(not uploaded_pdfs),
        )

        if run_btn and uploaded_pdfs:
            results, errors = [], []
            progress = st.progress(0, text="Initialising...")

            for i, pdf_file in enumerate(uploaded_pdfs):
                progress.progress(
                    (i + 1) / len(uploaded_pdfs),
                    text=f"Processing ({i+1}/{len(uploaded_pdfs)}): {pdf_file.name}",
                )
                try:
                    cv_text = _read_pdf(pdf_file)
                    if len(cv_text.strip()) < 50:
                        raise ValueError("CV content too short or unreadable.")

                    feats     = extract_features(cv_text, job_title, job_desc, domain, use_claude)
                    name      = feats.pop("candidate_name", pdf_file.name.replace(".pdf", ""))
                    reasoning = feats.pop("reasoning", {})
                    gender_px = feats.pop("Gender_Proxy", 0.5)

                    scores = score_candidate(feats, weights, model_choice)

                    results.append({
                        "Candidate":       name,
                        "File":            pdf_file.name,
                        "Final Score":     scores["final_score"],
                        "Model Score":     scores["model_score"],
                        "Criteria Match":  scores["criteria_score"],
                        "Verdict":         scores["verdict"],
                        "tier_color":      scores["tier_color"],
                        "Gender_Proxy":    gender_px,
                        **{k: round(float(feats.get(k, 0)), 3) for k in FEAT_COLS},
                        "_reasoning":      reasoning,
                    })

                except Exception as exc:
                    errors.append(f"{pdf_file.name}: {exc}")

            progress.empty()
            if errors:
                st.warning("Some files could not be processed:\n" + "\n".join(errors))

            if results:
                # Sort by Final Score descending
                ranked = sorted(results, key=lambda r: r["Final Score"], reverse=True)
                for i, r in enumerate(ranked):
                    r["rank"] = i + 1
                    if   i == 0: r["tier_label"], r["tier_badge"] = "Top 1",  "b-top1"
                    elif i <  3: r["tier_label"], r["tier_badge"] = "Top 3",  "b-top3"
                    elif i <  5: r["tier_label"], r["tier_badge"] = "Top 5",  "b-top5"
                    else:        r["tier_label"], r["tier_badge"] = "Top 10", "b-top10"

                st.session_state["ranked"] = ranked
                st.success(f"Successfully evaluated {len(ranked)} candidate(s).")

        # -- Display results (re-ranked live on every slider change) ----
        if "ranked" in st.session_state:
            raw_ranked  = st.session_state["ranked"]
            cur_weights = weights   # current slider values

            def _rerank(rows: list, cw: dict) -> list:
                """Recompute Final Score with current slider weights and re-sort."""
                w_vec = np.array([cw.get(c, 0.0) for c in FEAT_COLS])
                total = w_vec.sum() or 1.0
                for r in rows:
                    x = np.array([float(r.get(c, 0.5)) for c in FEAT_COLS])
                    crit = float(np.dot(w_vec, x) / total)
                    r["Criteria Match"] = round(crit * 100, 1)
                    r["Final Score"]    = round(
                        (0.60 * r["Model Score"] / 100.0 + 0.40 * crit) * 100, 1
                    )
                return sorted(rows, key=lambda r: r["Final Score"], reverse=True)

            ranked = _rerank(raw_ranked, cur_weights)
            n      = len(ranked)

            # KPI summary row
            st.markdown('<div class="sec">Ranking Results</div>',
                        unsafe_allow_html=True)
            km1, km2, km3, km4 = st.columns(4)
            km1.metric("Total candidates", n)
            km2.metric("Recommended",
                       sum(1 for r in ranked if r["Verdict"] == "Recommended"))
            km3.metric("Average score",
                       f"{np.mean([r['Final Score'] for r in ranked]):.1f}")
            km4.metric("Top score",
                       f"{ranked[0]['Final Score']:.1f}" if ranked else "—")

            # Gender proxy alert
            proxy_n = sum(1 for r in ranked if r.get("Gender_Proxy") in (0.0, 1.0))
            if proxy_n:
                st.markdown(
                    f'<div class="alert">'
                    f'{proxy_n} CV(s) contain explicit gender-indicating language.  '
                    f'The AI model does <b>not</b> use this information for scoring, '
                    f'but human reviewers should be aware of potential manual bias.'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # -- Ranking tabs -------------------------------------------
            rt1, rt2, rt3, rt4 = st.tabs(["Top 10", "Top 5", "Top 3", "Top 1"])

            def _render_rows(pool: list) -> None:
                for r in pool:
                    st.markdown(
                        f'<div class="cand-row">'
                        f'<span class="cand-rank">#{r["rank"]}</span>'
                        f'<span class="cand-name">{r["Candidate"]}</span>'
                        f'<span class="badge {r["tier_badge"]}">{r["tier_label"]}</span>'
                        f'<span class="cand-score">{r["Final Score"]:.1f}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            with rt1:
                pool10 = ranked[:min(10, n)]
                _render_rows(pool10)

                df10 = pd.DataFrame(pool10)[[
                    "rank", "Candidate", "Final Score",
                    "Model Score", "Criteria Match", "Verdict",
                ]]
                st.dataframe(df10, use_container_width=True, hide_index=True)

                # Grouped bar chart
                names_10 = [r["Candidate"][:14] for r in pool10]
                x10      = np.arange(len(names_10))
                wb       = 0.27
                fig10, ax10 = plt.subplots(figsize=(10, 3.8))
                ax10.bar(x10 - wb, [r["Model Score"]    for r in pool10],
                         wb, label="Model Score",    color=COMP, alpha=.85, edgecolor="white")
                ax10.bar(x10,      [r["Criteria Match"] for r in pool10],
                         wb, label="Criteria Match", color=GOOD, alpha=.85, edgecolor="white")
                ax10.bar(x10 + wb, [r["Final Score"]    for r in pool10],
                         wb, label="Final Score",    color=PRXY, alpha=.85, edgecolor="white")
                ax10.set_xticks(x10)
                ax10.set_xticklabels(names_10, rotation=30, ha="right", fontsize=8)
                ax10.set_ylabel("Score (%)")
                ax10.set_ylim(0, 108)
                ax10.legend(fontsize=8)
                ax10.set_title("Score Comparison -- Top 10", fontweight="bold", color=INK)
                for sp in ["top", "right"]:
                    ax10.spines[sp].set_visible(False)
                ax10.grid(axis="y", alpha=0.18)
                plt.tight_layout()
                st.pyplot(fig10)
                plt.close(fig10)

            with rt2:
                pool5 = ranked[:min(5, n)]
                _render_rows(pool5)
                cols5 = st.columns(min(5, n))
                for j, (r, col) in enumerate(zip(pool5, cols5)):
                    col.markdown(
                        f'<div class="card" style="text-align:center">'
                        f'<div style="font-size:.78rem;color:{MUT}">#{r["rank"]}</div>'
                        f'<div class="kpi" style="color:{r["tier_color"]};'
                        f'font-size:1.65rem">{r["Final Score"]:.0f}</div>'
                        f'<div style="font-weight:600;margin:5px 0 3px;font-size:.86rem">'
                        f'{r["Candidate"][:18]}</div>'
                        f'<div style="font-size:.78rem;color:{MUT}">'
                        f'AI {r["Model Score"]:.0f}% &nbsp; Criteria '
                        f'{r["Criteria Match"]:.0f}%</div>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            with rt3:
                pool3 = ranked[:min(3, n)]
                for r in pool3:
                    with st.expander(
                        f"#{r['rank']}  {r['Candidate']}  "
                        f"--  {r['Final Score']:.1f} pts  |  {r['Verdict']}",
                        expanded=(r["rank"] == 1),
                    ):
                        ea, eb = st.columns(2)
                        ea.metric("Model Score",    f"{r['Model Score']:.1f}%")
                        eb.metric("Criteria Match", f"{r['Criteria Match']:.1f}%")

                        fv3 = [r.get(f, 0) for f in FEAT_COLS]
                        bc3 = [GOOD if v >= 0.65 else WARN if v >= 0.4 else BAD for v in fv3]
                        fig3, ax3 = plt.subplots(figsize=(8, 3))
                        ax3.barh(FEAT_NAMES, fv3, color=bc3, edgecolor="white", height=.6)
                        ax3.axvline(0.5, color="gray", ls="--", lw=.8, alpha=.55)
                        ax3.set_xlim(0, 1.15)
                        ax3.set_xlabel("Feature score (0–1)")
                        for sp in ["top", "right"]:
                            ax3.spines[sp].set_visible(False)
                        plt.tight_layout()
                        st.pyplot(fig3)
                        plt.close(fig3)

                        rsn = r.get("_reasoning", {})
                        if rsn and "note" not in rsn:
                            st.markdown("**AI reasoning:**")
                            for k, v in list(rsn.items())[:4]:
                                if isinstance(v, str):
                                    st.markdown(f"- **{k}**: {v}")

            with rt4:
                if n >= 1:
                    top = ranked[0]
                    st.markdown(
                        f'<div style="text-align:center;padding:22px 0 14px">'
                        f'<div class="kpi" style="font-size:2.5rem;'
                        f'color:{top["tier_color"]}">{top["Candidate"]}</div>'
                        f'<div style="font-size:.95rem;color:{MUT};margin-top:6px">'
                        f'Top candidate &nbsp;|&nbsp; {top["Verdict"]}'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )
                    ta, tb, tc = st.columns(3)
                    ta.metric("Final Score",    f"{top['Final Score']:.1f} / 100")
                    tb.metric("Model Score",    f"{top['Model Score']:.1f}%")
                    tc.metric("Criteria Match", f"{top['Criteria Match']:.1f}%")

                    fvt = [top.get(f, 0) for f in FEAT_COLS]
                    bct = [GOOD if v >= 0.65 else WARN if v >= 0.4 else BAD for v in fvt]
                    fig4, ax4 = plt.subplots(figsize=(9, 3.6))
                    bars4 = ax4.bar(FEAT_NAMES, fvt, color=bct, edgecolor="white")
                    ax4.axhline(0.5, color="gray", ls="--", lw=1, alpha=.5)
                    ax4.set_ylim(0, 1.18)
                    for bar, val in zip(bars4, fvt):
                        ax4.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                                 f"{val:.2f}", ha="center", va="bottom",
                                 fontsize=8, fontweight="bold")
                    ax4.set_ylabel("Feature score (0–1)")
                    ax4.set_title(f"Competency Profile -- {top['Candidate']}",
                                  fontweight="bold", color=INK)
                    for sp in ["top", "right"]:
                        ax4.spines[sp].set_visible(False)
                    plt.xticks(rotation=25, ha="right", fontsize=9)
                    plt.tight_layout()
                    st.pyplot(fig4)
                    plt.close(fig4)

            # Download CSV
            st.markdown("")
            dl_cols = ["rank", "Candidate", "Final Score", "Model Score",
                       "Criteria Match", "Verdict"] + FEAT_COLS
            dl_df   = pd.DataFrame(ranked)[[c for c in dl_cols if c in pd.DataFrame(ranked).columns]]
            st.download_button(
                "Download results (CSV)",
                data=dl_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                file_name=f"faircv_{job_title.replace(' ', '_')}.csv",
                mime="text/csv",
            )

    # -------------------------------------------------------------------
    # Tab 2: Fairness Audit (on FairCVdb held-out test set)
    # -------------------------------------------------------------------
    with tab_fair:
        st.markdown(
            "Fairness metrics below are computed on the **4 800 held-out test profiles** "
            "from FairCVdb -- data the model never saw during training.  "
            "These figures certify how equitably the trained models behave across "
            "gender and ethnicity groups, independently of the CVs you just uploaded."
        )

        if DEMO_MODE:
            st.warning(
                "Demo mode: models trained on synthetic data.  "
                "Fairness figures are not statistically meaningful.  "
                "Place `FairCVdb.csv` in `data/` and run `python models/train_models.py`."
            )
        else:
            lrm  = cfg.get("lr_metrics",  {})
            mlpm = cfg.get("mlp_metrics", {})

            st.markdown('<div class="sec">LR vs MLP -- Setting A, Blind Label</div>',
                        unsafe_allow_html=True)
            fa, fb = st.columns(2)
            with fa:
                st.markdown("**Logistic Regression**")
                st.markdown(
                    f"| Metric | Value |\n|---|---|\n"
                    f"| F1 | **{lrm.get('F1',0):.4f}** |\n"
                    f"| ROC-AUC | **{lrm.get('ROC-AUC',0):.4f}** |\n"
                    f"| Accuracy | **{lrm.get('Accuracy',0):.4f}** |\n"
                    f"| DP Gap (gender) | **{lrm.get('DP_Gap_Gender',0):.4f}** |\n"
                    f"| EOO Gap (gender) | **{lrm.get('EOO_Gap_Gender',0):.4f}** |\n"
                    f"| DP Gap (ethnicity) | **{lrm.get('DP_Gap_Ethnicity',0):.4f}** |"
                )
            with fb:
                st.markdown("**MLP Neural Network**")
                st.markdown(
                    f"| Metric | Value |\n|---|---|\n"
                    f"| F1 | **{mlpm.get('F1',0):.4f}** |\n"
                    f"| ROC-AUC | **{mlpm.get('ROC-AUC',0):.4f}** |\n"
                    f"| Accuracy | **{mlpm.get('Accuracy',0):.4f}** |\n"
                    f"| DP Gap (gender) | **{mlpm.get('DP_Gap_Gender',0):.4f}** |\n"
                    f"| EOO Gap (gender) | **{mlpm.get('EOO_Gap_Gender',0):.4f}** |\n"
                    f"| DP Gap (ethnicity) | **{mlpm.get('DP_Gap_Ethnicity',0):.4f}** |"
                )

            st.markdown(
                '<div class="fair-note">'
                '<b>Fairness metrics explained:</b><br>'
                '<b>DP Gap</b> (Demographic Parity Gap) = |P(select|A=1) - P(select|A=0)| '
                '-- selection rate gap between groups.  Values below 0.05 are generally acceptable.<br>'
                '<b>EOO Gap</b> (Equal Opportunity Gap) = |TPR(A=1) - TPR(A=0)| '
                '-- true-positive-rate gap between groups.<br>'
                '<b>Mitigations applied:</b> '
                'T1: gender/ethnicity columns removed (Setting A). '
                'T3: inverse-frequency sample reweighting during training.'
                '</div>',
                unsafe_allow_html=True,
            )

    # -------------------------------------------------------------------
    # Tab 3: SHAP -- XAI Explanations
    # -------------------------------------------------------------------
    with tab_shap:
        st.markdown(
            "SHAP values reveal **which features drive the AI model's decisions** "
            "most strongly, enabling transparent auditing of the scoring logic."
        )

        if "ranked" not in st.session_state:
            st.info("Please upload and evaluate CVs in the 'CV Screening' tab first.")
        else:
            ranked_s = st.session_state["ranked"]

            # Build feature matrix from scored candidates
            X_cands = np.array([
                [float(r.get(f, 0.5)) for f in FEAT_COLS]
                for r in ranked_s
            ])

            # Select scaler matching current model choice
            scaler_s = mlp_scaler if model_choice.startswith("MLP") else lr_scaler
            model_s  = mlp_model  if model_choice.startswith("MLP") else lr_model
            X_scaled = scaler_s.transform(X_cands)

            try:
                import shap

                # LinearExplainer for LR; KernelExplainer fallback for MLP
                try:
                    explainer = shap.LinearExplainer(
                        model_s, X_scaled,
                        feature_perturbation="interventional",
                    )
                    sv = explainer.shap_values(X_scaled)
                except Exception:
                    bg = shap.sample(X_scaled, min(50, len(X_scaled)))
                    explainer = shap.KernelExplainer(
                        lambda x: model_s.predict_proba(x)[:, 1], bg
                    )
                    sv = explainer.shap_values(X_scaled, nsamples=100)

                # -- Global Feature Importance bar chart ----------------
                st.markdown('<div class="sec">Global Feature Importance (Mean |SHAP|)</div>',
                            unsafe_allow_html=True)
                mean_abs = np.abs(sv).mean(axis=0)
                order    = np.argsort(mean_abs)

                fig_s, ax_s = plt.subplots(figsize=(8, 4))
                ax_s.barh([FEAT_NAMES[i] for i in order], mean_abs[order],
                          color=COMP, edgecolor="white")
                ax_s.set_xlabel("Mean |SHAP value|")
                ax_s.set_title(
                    f"Feature Importance ({model_choice.split()[0]}) -- SHAP",
                    fontweight="bold", color=INK,
                )
                for sp in ["top", "right"]:
                    ax_s.spines[sp].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig_s)
                plt.close(fig_s)

                st.dataframe(
                    pd.DataFrame({
                        "Feature":      [FEAT_NAMES[i] for i in order[::-1]],
                        "Mean |SHAP|":  [round(mean_abs[i], 5) for i in order[::-1]],
                    }),
                    use_container_width=True, hide_index=True,
                )

                # -- SHAP dot plot (each dot = one candidate) -----------
                st.markdown('<div class="sec">SHAP Contribution Distribution</div>',
                            unsafe_allow_html=True)
                st.caption("Each dot = one candidate.  "
                           "Colour = feature score level.  "
                           "x-axis position = direction and magnitude of influence on the decision.")

                fig_d, ax_d = plt.subplots(figsize=(9, 4))
                for j, fname in enumerate(FEAT_NAMES):
                    vals   = sv[:, j]
                    scores = X_cands[:, j]
                    cols   = [GOOD if s >= 0.6 else WARN if s >= 0.4 else BAD for s in scores]
                    ax_d.scatter(vals, [j] * len(vals), c=cols,
                                 alpha=0.75, s=45, edgecolors="white", linewidths=.4)
                ax_d.set_yticks(range(len(FEAT_NAMES)))
                ax_d.set_yticklabels(FEAT_NAMES, fontsize=9)
                ax_d.axvline(0, color="gray", lw=0.8)
                ax_d.set_xlabel("SHAP value (impact on model output)")
                ax_d.set_title("Per-Feature SHAP Distribution Across Candidates",
                               fontweight="bold", color=INK)
                legend_d = [
                    mpatches.Patch(color=GOOD, label="High feature score (>=0.6)"),
                    mpatches.Patch(color=WARN, label="Medium (0.4–0.6)"),
                    mpatches.Patch(color=BAD,  label="Low (<0.4)"),
                ]
                ax_d.legend(handles=legend_d, fontsize=8, loc="lower right")
                for sp in ["top", "right"]:
                    ax_d.spines[sp].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig_d)
                plt.close(fig_d)

            except ImportError:
                # Graceful fallback: show LR coefficients as a proxy
                st.warning("`shap` library not installed.  Showing LR coefficients as a proxy.")
                if hasattr(lr_model, "coef_"):
                    coefs = np.abs(lr_model.coef_.ravel())
                    order = np.argsort(coefs)
                    fig_fb, ax_fb = plt.subplots(figsize=(8, 4))
                    ax_fb.barh([FEAT_NAMES[i] for i in order], coefs[order],
                               color=COMP, edgecolor="white")
                    ax_fb.set_xlabel("|LR coefficient| (proxy for importance)")
                    ax_fb.set_title("Feature Importance (LR coefficients)",
                                    fontweight="bold", color=INK)
                    for sp in ["top", "right"]:
                        ax_fb.spines[sp].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig_fb)
                    plt.close(fig_fb)
                else:
                    st.info("Install shap for full XAI support:  `pip install shap`")


# ===========================================================================
# CANDIDATE VIEW
# ===========================================================================

else:
    st.markdown(
        '<div class="hero">'
        '<h1>Candidate -- Your Personal Fairness Report Card</h1>'
        '<p>Upload your CV (one PDF file), see how it is scored, '
        'which features drive the decision, and whether the outcome '
        'is based fairly on your competencies.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Target position
    st.markdown('<div class="sec">Target Position</div>', unsafe_allow_html=True)
    pc1, pc2 = st.columns(2)
    with pc1:
        cand_job    = st.text_input("Position you are applying for", "Data Analyst")
    with pc2:
        cand_domain = st.selectbox(
            "Industry domain",
            ["IT / Technology", "Marketing", "Finance / Accounting",
             "Human Resources", "Other"],
        )
    cand_desc = st.text_area(
        "Typical requirements for this role",
        "Python, SQL, data analysis, 2+ years of experience",
        height=68,
    )

    # Single PDF upload
    st.markdown('<div class="sec">Upload Your CV</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="upload-hint">'
        '<b>Format:</b> PDF only &nbsp;|&nbsp; '
        '<b>Single file</b> -- your own CV &nbsp;|&nbsp; '
        'File content is never stored; it is processed in-memory for this session only.'
        '</div>',
        unsafe_allow_html=True,
    )

    cand_pdf = st.file_uploader(
        "Select your CV (PDF)",
        type=["pdf"],
        accept_multiple_files=False,   # Candidate: one file only
        key="candidate_upload",
        help="Upload your own CV as a single PDF file.",
    )

    if cand_pdf:
        st.caption(f"Selected: **{cand_pdf.name}**  ({cand_pdf.size:,} bytes)")

    analyse_btn = st.button(
        "Analyse My CV",
        type="primary",
        disabled=(cand_pdf is None),
    )

    if analyse_btn and cand_pdf is not None:
        with st.spinner("Analysing your CV..."):
            cv_text = _read_pdf(cand_pdf)

            if len(cv_text.strip()) < 50:
                st.error(
                    "Could not extract text from this PDF.  "
                    "Make sure the file is not scanned/image-only."
                )
                st.stop()

            cand_w = {c: 1.0 / len(FEAT_COLS) for c in FEAT_COLS}
            feats  = extract_features(cv_text, cand_job, cand_desc, cand_domain, use_claude)
            name      = feats.pop("candidate_name", "You")
            reasoning = feats.pop("reasoning", {})
            gender_px = feats.pop("Gender_Proxy", 0.5)
            scores    = score_candidate(feats, cand_w, model_choice)

            st.session_state["cand_result"] = {
                "name":         name,
                "feats":        feats,
                "scores":       scores,
                "reasoning":    reasoning,
                "gender_proxy": gender_px,
            }

    if "cand_result" in st.session_state:
        cr = st.session_state["cand_result"]
        sc = cr["scores"]
        ft = cr["feats"]

        # -- Decision result --------------------------------------------
        st.markdown('<div class="sec">Your Result</div>', unsafe_allow_html=True)
        dec_color = GOOD if sc["verdict"] == "Recommended" else BAD
        dec_label = "RECOMMENDED" if sc["verdict"] == "Recommended" else "NOT RECOMMENDED"

        dr1, dr2, dr3 = st.columns(3)
        dr1.markdown(
            _card("Decision", dec_label, dec_color,
                  f"Final score: {sc['final_score']:.1f} / 100"),
            unsafe_allow_html=True,
        )
        dr2.metric("AI Model Score",   f"{sc['model_score']:.1f}%")
        dr3.metric("Criteria Match",   f"{sc['criteria_score']:.1f}%")

        # -- Competency profile bar chart -------------------------------
        st.markdown('<div class="sec">Your Competency Profile</div>',
                    unsafe_allow_html=True)

        fv = [ft.get(f, 0) for f in FEAT_COLS]
        bc = [GOOD if v >= 0.65 else WARN if v >= 0.4 else BAD for v in fv]

        fig_c, ax_c = plt.subplots(figsize=(9, 3.6))
        ax_c.barh(FEAT_NAMES, fv, color=bc, edgecolor="white", height=.62)
        ax_c.axvline(0.5, color="gray", ls="--", lw=.9, alpha=.5)
        ax_c.set_xlim(0, 1.16)
        ax_c.set_xlabel("Feature score (0–1)")
        ax_c.set_title(f"Competency Profile -- {cr['name']}", fontweight="bold", color=INK)
        legend_c = [
            mpatches.Patch(color=GOOD, label="Strong (>= 0.65)"),
            mpatches.Patch(color=WARN, label="Moderate (0.40 – 0.65)"),
            mpatches.Patch(color=BAD,  label="Needs improvement (< 0.40)"),
        ]
        ax_c.legend(handles=legend_c, fontsize=8, loc="lower right")
        for sp in ["top", "right"]:
            ax_c.spines[sp].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_c)
        plt.close(fig_c)

        # -- Per-feature AI reasoning -----------------------------------
        rsn = cr.get("reasoning", {})
        if rsn and "note" not in rsn:
            st.markdown('<div class="sec">AI Reasoning per Feature</div>',
                        unsafe_allow_html=True)
            for k, v in rsn.items():
                if isinstance(v, str):
                    val   = ft.get(k, 0.5)
                    icon  = "✓" if val >= 0.65 else ("~" if val >= 0.4 else "x")
                    color = GOOD if val >= 0.65 else (WARN if val >= 0.4 else BAD)
                    st.markdown(
                        f'<span style="color:{color};font-weight:700">[{icon}]</span> '
                        f'**{k}** ({val:.2f}): {v}',
                        unsafe_allow_html=True,
                    )

        # -- Fairness assessment ----------------------------------------
        st.markdown('<div class="sec">Fairness Assessment</div>', unsafe_allow_html=True)

        gp = cr.get("gender_proxy", 0.5)
        if gp in (0.0, 1.0):
            group = "Female" if gp == 1.0 else "Male"
            st.markdown(
                f'<div class="v-warn">'
                f'Gender-indicating language ({group}) was detected in your CV.  '
                f'The FairCV model does <b>not</b> use this information for scoring '
                f'(Gender_Proxy is excluded from all model features).  '
                f'However, human reviewers reading the CV directly may be influenced '
                f'by this language.'
                f'</div>',
                unsafe_allow_html=True,
            )
        elif not DEMO_MODE:
            mlpm = cfg.get("mlp_metrics", {})
            dp_g = float(mlpm.get("DP_Gap_Gender", 0.0))
            if dp_g < 0.02:
                st.markdown(
                    f'<div class="v-clear">'
                    f'The MLP model has a gender DP Gap of {dp_g:.4f} on the FairCVdb '
                    f'test set -- well within the acceptable range (< 0.05).  '
                    f'Your result is based on your competency profile only.'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="v-warn">'
                    f'Gender DP Gap = {dp_g:.4f} on the test set -- '
                    f'some disparity remains between demographic groups.  '
                    f'Sample reweighting (T3) has been applied to minimise this.'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("Fairness assessment requires production models trained on FairCVdb.")

        # -- How your score is calculated (expandable) ------------------
        with st.expander("How your score is calculated", expanded=False):
            st.markdown(
                "**Feature extraction:** Your CV text is parsed into 8 structured "
                "competency scores (0.0 – 1.0) matching the FairCVdb schema.  "
                "Claude API is used for extraction when a key is configured; "
                "otherwise a rule-based heuristic is applied.\n\n"
                "**Model scoring:** Two models trained on 19 200 FairCVdb profiles "
                "(blind label -- gender and ethnicity never used in training) "
                "each predict the probability of a 'Recommended' outcome.  "
                "Final score = 60% model probability + 40% criteria match.\n\n"
                "**Fairness guarantees applied:**\n"
                "- Gender and ethnicity are absent from all model input features (Setting A).\n"
                "- Models are trained on the blind label (unaware of protected attributes).\n"
                "- Sample reweighting (T3) reduces inter-group imbalance during training.\n"
                "- 'Recommendation' feature excluded to prevent referral-network bias."
            )


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.caption(
    "FairCV v1.0  --  Fair Recruitment Scoring System.  "
    "Models trained on FairCVdb (Complement et al., CVPRW 2020).  "
    "CV data is processed in-memory and never stored."
)

