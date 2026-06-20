"""
app/streamlit_app.py
--------------------
FairCV -- Giao dien chinh Streamlit cho he thong sang loc CV cong bang.

Hai vai tro chia se cung mot phien lam viec (session):

  Recruiter (Nha tuyen dung)
      1. Nhap vi tri tuyen dung, mo ta yeu cau cong viec.
      2. Dat trong so uu tien cho tung tieu chi (slider real-time).
      3. Upload goi ho so .zip (nhieu CV text/PDF trong mot file).
      4. Chon mo hinh AI: Logistic Regression hoac MLP.
      5. Xem bang xep hang Top 10/5/3/1, bieu do SHAP, fairness audit.

  Candidate (Ung vien)
      1. Nhap vi tri muon ung tuyen.
      2. Upload CV ca nhan (file PDF).
      3. Xem diem ca nhan, profile nang luc, fairness report card.

Chay ung dung:
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

# -- Them duong dan src vao sys.path de import package faircv -------------
_APP_DIR  = os.path.dirname(os.path.abspath(__file__))
_ROOT_DIR = os.path.dirname(_APP_DIR)
sys.path.insert(0, os.path.join(_ROOT_DIR, "src"))

from faircv.data import (
    COMPETENCY, COMP_NAMES,
    GENDER_LABELS, ETH_LABELS,
    extract_cvs_from_zip,      # Giai nen ZIP trong RAM
    _cached_load_faircv_dataset as load_dataset,  # Co cache Streamlit
    GOOGLE_DRIVE_FILE_ID, LOCAL_CSV_PATH,
)

# ---------------------------------------------------------------------------
# Cau hinh trang
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="FairCV",
    layout="wide",
    page_icon="",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Design tokens (mau sac nhat quan)
# ---------------------------------------------------------------------------

INK  = "#1b2330"   # Mau chu chinh
COMP = "#2E86AB"   # Xanh duong -- tieu chi nang luc
GOOD = "#2E9E5B"   # Xanh la -- tot / dat
WARN = "#E0A100"   # Vang -- can xem xet
BAD  = "#C2384A"   # Do -- khong dat / bias
PRXY = "#E4572E"   # Cam -- proxy / canh bao
MUT  = "#7a869a"   # Xam -- thu yeu

# ---------------------------------------------------------------------------
# CSS noi trang
# ---------------------------------------------------------------------------

st.markdown(f"""
<style>
:root {{
  --ink:{INK}; --comp:{COMP}; --good:{GOOD};
  --bad:{BAD}; --warn:{WARN}; --proxy:{PRXY};
}}
.block-container {{ padding-top:1rem; max-width:1260px; }}

/* Hero banner */
.hero {{
  background: linear-gradient(110deg,#101826 0%,#1f3550 55%,{COMP} 130%);
  color:#fff; padding:20px 26px; border-radius:14px;
  margin-bottom:12px; box-shadow:0 6px 22px rgba(16,24,38,.18);
}}
.hero h1 {{ margin:0; font-size:1.48rem; }}
.hero p  {{ margin:.3rem 0 0; opacity:.9; font-size:.92rem; }}

/* Card thong tin */
.card {{
  background:#fff; border:1px solid #e2e8f0;
  border-radius:12px; padding:14px 16px;
  box-shadow:0 2px 8px rgba(20,30,50,.05); height:100%;
}}
.card h4 {{ margin:0 0 5px; color:var(--ink); font-size:.95rem; }}

/* So lieu lon (KPI) */
.kpi {{ font-size:1.8rem; font-weight:700; line-height:1.1; }}
.sub {{ color:#5a6678; font-size:.82rem; margin-top:2px; }}

/* Nhan bieu hien */
.badge {{
  display:inline-block; padding:3px 10px;
  border-radius:999px; font-size:.77rem; font-weight:700;
  color:#fff; margin:2px 3px 2px 0;
}}
.b-good  {{ background:var(--good);  }}
.b-bad   {{ background:var(--bad);   }}
.b-warn  {{ background:var(--warn);  }}
.b-comp  {{ background:var(--comp);  }}
.b-proxy {{ background:var(--proxy); }}
.b-top1  {{ background:#D97706; }}
.b-top3  {{ background:#7C3AED; }}
.b-top5  {{ background:var(--good); }}
.b-top10 {{ background:var(--comp); }}

/* Dong ung vien trong bang xep hang */
.cand-row {{
  display:flex; align-items:center; gap:12px;
  padding:9px 13px; border:1px solid #e2e8f0;
  border-radius:9px; margin:5px 0; background:#fafbfc;
}}
.cand-rank  {{ font-size:1.2rem; font-weight:700; color:#475569; min-width:30px; }}
.cand-name  {{ font-weight:600; color:var(--ink); flex:1; }}
.cand-score {{ font-size:1.2rem; font-weight:700;
               color:var(--comp); min-width:52px; text-align:right; }}

/* Hop verdict fairness */
.verdict-cause {{ background:#fdece8; border:1px solid #f3b3a4;
                  border-radius:10px; padding:12px 15px; color:#8f2d18; }}
.verdict-clear {{ background:#e7f6ec; border:1px solid #a9dcb9;
                  border-radius:10px; padding:12px 15px; color:#1d6b39; }}
.verdict-warn  {{ background:#fff6e0; border:1px solid #f0d493;
                  border-radius:10px; padding:12px 15px; color:#7a5800; }}

/* Canh bao */
.warn-box {{
  background:#fff7ed; border:1px solid #fed7aa;
  border-radius:9px; padding:9px 13px;
  color:#9a3412; font-size:.86rem; margin:4px 0;
}}

/* Ngan cach section */
.sec {{
  font-size:1.03rem; font-weight:700; color:var(--ink);
  margin:16px 0 7px; padding-bottom:3px;
  border-bottom:2px solid #e2e8f0;
}}

/* Ghi chu cong bang */
.fair-note {{
  background:#eff6ff; border:1px solid #bfdbfe;
  border-radius:9px; padding:10px 14px;
  color:#1d4ed8; font-size:.85rem; margin-top:6px;
}}
</style>
""", unsafe_allow_html=True)


# ===========================================================================
# Helpers dung chung
# ===========================================================================

def _card(title: str, kpi: str, color: str, sub: str) -> str:
    """Tao HTML card thong tin KPI."""
    return (
        f'<div class="card"><h4>{title}</h4>'
        f'<div class="kpi" style="color:{color}">{kpi}</div>'
        f'<div class="sub">{sub}</div></div>'
    )


def _badge(text: str, cls: str) -> str:
    """Tao HTML badge nhan bieu hien."""
    return f'<span class="badge {cls}">{text}</span>'


# ---------------------------------------------------------------------------
# Tai mo hinh da huan luyen (cache resource -- chi tai 1 lan)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Dang tai mo hinh AI...")
def _load_models(out_dir: str = "models/saved"):
    """Nap LR + MLP models va scaler tu thu muc models/saved/.

    Tu dong chay train_models.py neu cac file .pkl chua ton tai.
    """
    paths = {
        "lr":       os.path.join(out_dir, "lr_model.pkl"),
        "lr_sc":    os.path.join(out_dir, "lr_scaler.pkl"),
        "mlp":      os.path.join(out_dir, "mlp_model.pkl"),
        "mlp_sc":   os.path.join(out_dir, "mlp_scaler.pkl"),
        "config":   os.path.join(out_dir, "config.pkl"),
    }
    if not all(os.path.exists(p) for p in paths.values()):
        st.info("Chua tim thay mo hinh -- dang huan luyen tu FairCVdb...")
        import subprocess
        script = os.path.join(_ROOT_DIR, "models", "train_models.py")
        subprocess.run([sys.executable, script], check=True)

    lr_model   = joblib.load(paths["lr"])
    lr_scaler  = joblib.load(paths["lr_sc"])
    mlp_model  = joblib.load(paths["mlp"])
    mlp_scaler = joblib.load(paths["mlp_sc"])
    config     = joblib.load(paths["config"])
    return lr_model, lr_scaler, mlp_model, mlp_scaler, config


lr_model, lr_scaler, mlp_model, mlp_scaler, cfg = _load_models()
FEAT_COLS  = cfg.get("features",      COMPETENCY)
FEAT_NAMES = cfg.get("feature_names", COMP_NAMES)
DEMO_MODE  = cfg.get("demo_mode",     False)

if DEMO_MODE:
    st.warning(
        "**Demo mode:** Mo hinh duoc huan luyen tren du lieu mo phong.  "
        "Dat `FairCVdb.csv` that vao `data/` va chay lai "
        "`python models/train_models.py` de co ket qua chinh xac."
    )


# ===========================================================================
# Sidebar -- Cau hinh chung cho ca 2 vai tro
# ===========================================================================

with st.sidebar:
    st.markdown("## FairCV")
    st.caption("He thong sang loc CV cong bang")

    # -- Chon vai tro --
    role = st.radio("Ban la", ["Recruiter (Nha tuyen dung)", "Candidate (Ung vien)"])

    st.markdown("---")

    if role.startswith("Recruiter"):
        # ---- Thong tin vi tri tuyen dung ----
        st.markdown("### Vi tri tuyen dung")
        job_title = st.text_input("Ten vi tri", "Data Analyst")
        job_desc  = st.text_area(
            "Mo ta yeu cau",
            "Toi thieu 2 nam kinh nghiem phan tich du lieu.  "
            "Thanh thao Python va SQL.  Tieng Anh giao tiep.  "
            "Uu tien co kinh nghiem machine learning.",
            height=105,
        )
        domain = st.selectbox(
            "Linh vuc nganh nghe",
            ["IT / Cong nghe", "Marketing", "Tai chinh / Ke toan",
             "Nhan su / HR", "Khac"],
        )

        st.markdown("---")

        # ---- Trong so tieu chi (real-time) ----
        st.markdown("### Trong so tieu chi")
        st.caption(
            "Keo thanh truot de dieu chinh do uu tien.  "
            "Ket qua xep hang cap nhat ngay lap tuc."
        )
        w_suit  = st.slider("Phu hop vi tri",     0.0, 1.0, 0.30, 0.05)
        w_exp   = st.slider("Kinh nghiem lam viec", 0.0, 1.0, 0.25, 0.05)
        w_edu   = st.slider("Trinh do hoc van",   0.0, 1.0, 0.20, 0.05)
        w_lang1 = st.slider("Ngoai ngu chinh",    0.0, 1.0, 0.10, 0.05)
        w_lang2 = st.slider("Ngoai ngu phu",      0.0, 1.0, 0.05, 0.05)
        w_avail = st.slider("Co the bat dau som", 0.0, 1.0, 0.10, 0.05)

        # Ma tran trong so tuong ung voi FEAT_COLS
        # Recommendation bi loai (de bao dam cong bang -- tranh bias mang luoi)
        weights = {
            "Suitability":    w_suit,
            "Language 1":     w_lang1,
            "Language 2":     w_lang2,
            "Language 3":     0.0,
            "Experience":     w_exp,
            "Education":      w_edu,
            "Recommendation": 0.0,   # Loai bo -- de bao dam cong bang
            "Availability":   w_avail,
        }

        st.markdown("---")

        # ---- Chon mo hinh ----
        st.markdown("### Mo hinh AI")
        model_choice = st.radio(
            "Su dung mo hinh",
            ["LR (Logistic Regression)", "MLP (Neural Network)"],
            help="LR: giai thich duoc, nhanh.\nMLP: phi tuyen, DP Gap thap hon."
        )

        # ---- Cai dat AI extraction ----
        st.markdown("---")
        st.markdown("### Trich xuat CV")
        use_claude = st.toggle(
            "Dung Claude API (chinh xac hon)",
            value=bool(os.environ.get("ANTHROPIC_API_KEY")),
        )
        if use_claude and not os.environ.get("ANTHROPIC_API_KEY"):
            key_in = st.text_input("ANTHROPIC_API_KEY", type="password")
            if key_in:
                os.environ["ANTHROPIC_API_KEY"] = key_in
            use_claude = bool(os.environ.get("ANTHROPIC_API_KEY"))

    else:
        # Candidate -- thong tin toi gian
        job_title    = st.text_input("Vi tri ban ung tuyen", "Data Analyst")
        job_desc     = st.text_area("Yeu cau thuong gap cua vi tri nay",
                                    "Python, SQL, phan tich du lieu, 2+ nam kinh nghiem",
                                    height=80)
        domain       = "Khac"
        model_choice = "MLP (Neural Network)"
        use_claude   = bool(os.environ.get("ANTHROPIC_API_KEY"))
        weights      = {c: 1.0 / len(FEAT_COLS) for c in FEAT_COLS}

    # Hien thi chi so mo hinh
    if not DEMO_MODE:
        st.markdown("---")
        st.markdown("### Hieu nang mo hinh")
        lrm  = cfg.get("lr_metrics",  {})
        mlpm = cfg.get("mlp_metrics", {})
        st.markdown(
            f"| Mo hinh | F1 | AUC | DP Gap |\n|---|---|---|---|\n"
            f"| LR  | {lrm.get('F1',0):.3f} | {lrm.get('ROC-AUC',0):.3f} | "
            f"{lrm.get('DP_Gap_Gender',0):.4f} |\n"
            f"| MLP | {mlpm.get('F1',0):.3f} | {mlpm.get('ROC-AUC',0):.3f} | "
            f"{mlpm.get('DP_Gap_Gender',0):.4f} |"
        )
        st.caption("Huan luyen tren FairCVdb -- blind label (khong biet gioi tinh / dan toc)")


# ===========================================================================
# Ham trich xuat dac trung (Feature Extraction)
# ===========================================================================

# Tu dien tu khoa theo linh vuc nganh nghe
_DOMAIN_KEYWORDS: dict[str, list[str]] = {
    "IT / Cong nghe": [
        "python", "java", "sql", "machine learning", "data", "ai", "software",
        "backend", "frontend", "cloud", "aws", "docker", "kubernetes",
        "tensorflow", "pytorch", "deep learning", "api", "git", "agile",
        "analyst", "engineer", "developer", "code", "programming",
        "lap trinh", "cong nghe", "du lieu", "tri tue nhan tao",
    ],
    "Marketing": [
        "marketing", "brand", "campaign", "social media", "seo", "content",
        "digital", "advertising", "customer", "market research", "crm",
        "analytics", "e-commerce", "conversion", "roi", "kpi",
        "thuong hieu", "quang cao", "khach hang", "thi truong",
    ],
    "Tai chinh / Ke toan": [
        "finance", "accounting", "audit", "tax", "budget", "financial",
        "excel", "reporting", "investment", "risk", "compliance", "ifrs",
        "ke toan", "tai chinh", "thue", "kiem toan", "bao cao tai chinh",
    ],
    "Nhan su / HR": [
        "hr", "human resources", "recruitment", "talent", "training",
        "performance", "payroll", "labor", "onboarding", "kpi",
        "nhan su", "tuyen dung", "nhan luc", "dao tao", "luong",
    ],
    "Khac": [],
}


def parse_suitability_score(cv_text: str, domain: str) -> float:
    """Cham diem 'Suitability' bang mat do tu khoa theo linh vuc nganh nghe.

    Thuat toan:
      1. Lay danh sach tu khoa tuong ung voi `domain`.
      2. Dem so lan xuat hien (khong phan biet hoa/thuong) trong van ban CV.
      3. Tinh ty le: mat_do = so_tu_khoa_tim_thay / tong_so_tu_khoa.
      4. Nhan phi tuyen (sqrt) de trung binh hoa phan pho.
      5. Ep ve khoang [0.1, 1.0].

    Parameters
    ----------
    cv_text : Noi dung text thu cua CV.
    domain  : Ten linh vuc nganh nghe (key trong _DOMAIN_KEYWORDS).

    Returns
    -------
    float trong khoang [0.1, 1.0].
    """
    keywords = _DOMAIN_KEYWORDS.get(domain, [])
    if not keywords:
        return 0.5   # Khong co tu dien -- diem trung binh

    text_lower = cv_text.lower()
    hits = sum(1 for kw in keywords if kw in text_lower)
    ratio = hits / len(keywords)

    # Phi tuyen de tranh diem 0 khi chi khop it tu
    score = float(np.sqrt(ratio))

    # Ep ve [0.1, 1.0] -- khong bao gio cho 0 hoan toan
    return float(np.clip(score, 0.1, 1.0))


def detect_gender_proxy(cv_text: str) -> float:
    """Phat hien bien proxy gioi tinh tu van ban CV.

    Cac tu chi gioi tinh nu: 'nu', 'female', 'chi', 'ba', 'she', 'her'.
    Cac tu chi gioi tinh nam: 'nam', 'male', 'anh', 'ong', 'he', 'his'.

    Returns
    -------
    float: 1.0 neu phat hien proxy nu, 0.0 neu proxy nam, 0.5 neu khong ro.
    """
    text_lower = cv_text.lower()

    female_tokens = ["nu ", " nu,", "female", " chi ", " ba ", " she ", " her "]
    male_tokens   = ["nam ", " nam,", "male", " anh ", " ong ", " he ", " his "]

    female_count = sum(1 for t in female_tokens if t in text_lower)
    male_count   = sum(1 for t in male_tokens   if t in text_lower)

    if female_count > male_count:
        return 1.0
    elif male_count > female_count:
        return 0.0
    return 0.5   # Khong xac dinh


def extract_features_heuristic(
    cv_text: str,
    job_title: str,
    job_desc: str,
    domain: str,
) -> dict:
    """Trich xuat 8 dac trung FairCV bang quy tac (heuristic).

    Su dung khi khong co Claude API hoac de fallback.

    Cac dac trung duoc trich xuat:
      Suitability   -- mat do tu khoa theo linh vuc + yeu cau cong viec
      Language 1    -- tieng Viet / tieng Anh (suy luan tu noi dung CV)
      Language 2    -- ngoai ngu thu 2 duoc de cap
      Language 3    -- ngoai ngu thu 3
      Experience    -- uoc tinh tu so nam kiem duoc trong van ban
      Education     -- bac hoc cao nhat duoc de cap
      Recommendation-- co tuong thuat / gioi thieu khong
      Availability  -- co the bat dau som khong

    Parameters
    ----------
    cv_text   : Noi dung text CV.
    job_title : Ten vi tri tuyen dung.
    job_desc  : Mo ta yeu cau cong viec.
    domain    : Linh vuc nganh nghe.

    Returns
    -------
    dict {feature_name -> float [0,1]} + 'candidate_name' (str).
    """
    txt = cv_text.lower()

    # -- Ten ung vien: dong dau tien khong rong ---------------------
    lines = [l.strip() for l in cv_text.split("\n") if l.strip()]
    name  = lines[0] if lines else "Unknown"

    # -- Suitability: ket hop mat do tu khoa + overlap yeu cau ------
    kw_score = parse_suitability_score(cv_text, domain)
    req_words = set(re.findall(r"\w{3,}", job_desc.lower()))
    cv_words  = set(re.findall(r"\w{3,}", txt))
    overlap   = len(req_words & cv_words) / max(len(req_words), 1)
    suitability = float(np.clip(0.4 * kw_score + 0.6 * min(1.0, overlap * 2), 0.1, 1.0))

    # -- Kinh nghiem: so nam cao nhat tim duoc trong van ban ---------
    yrs = re.findall(
        r"(\d+)\s*(?:\+\s*)?(?:nam|years?|yrs?)\s*"
        r"(?:of\s*)?(?:kinh\s*nghi[eê]m|experience)?",
        txt,
    )
    max_yrs    = max((int(y) for y in yrs), default=0)
    experience = float(np.clip(min(1.0, max_yrs / 5.0) if max_yrs else 0.3, 0.1, 1.0))

    # -- Hoc van: bac hoc cao nhat ----------------------------------
    education = 0.5
    if any(k in txt for k in ["tien si", "ph.d", "phd", "doctorate"]):
        education = 1.0
    elif any(k in txt for k in ["thac si", "master", "mba", "msc", "m.s.", "m.e."]):
        education = 0.8
    elif any(k in txt for k in ["dai hoc", "bachelor", "university",
                                  "b.s.", "b.e.", "b.a.", "cu nhan"]):
        education = 0.6
    elif any(k in txt for k in ["cao dang", "associate"]):
        education = 0.4

    # -- Ngoai ngu --------------------------------------------------
    lang1 = 0.75   # mac dinh: co kha nang viet CV (tieng Viet / Anh)
    lang2 = 0.55 if any(k in txt for k in [
        "english", "tieng anh", "ielts", "toeic", "toefl"
    ]) else 0.2
    lang3 = 0.40 if any(k in txt for k in [
        "japanese", "nhat", "chinese", "trung", "korean", "han",
        "french", "phap", "german", "duc"
    ]) else 0.1

    # -- Gioi thieu / Reference ------------------------------------
    recommendation = 0.65 if any(k in txt for k in [
        "reference", "recommendation", "gioi thieu", "referral"
    ]) else 0.3

    # -- Co the bat dau som ----------------------------------------
    availability = 1.0 if any(k in txt for k in [
        "immediately", "ngay lap tuc", "bat dau ngay", "available now"
    ]) else 0.65

    return {
        "candidate_name": name,
        "Suitability":    round(suitability, 3),
        "Language 1":     round(lang1, 3),
        "Language 2":     round(lang2, 3),
        "Language 3":     round(lang3, 3),
        "Experience":     round(experience, 3),
        "Education":      round(education, 3),
        "Recommendation": round(recommendation, 3),
        "Availability":   round(availability, 3),
        "Gender_Proxy":   detect_gender_proxy(cv_text),
        "reasoning":      {"note": "Heuristic (chua co Claude API)"},
    }


def extract_features_claude(
    cv_text: str,
    job_title: str,
    job_desc: str,
) -> dict:
    """Trich xuat 8 dac trung FairCV bang Claude API (chinh xac hon).

    Yeu cau ANTHROPIC_API_KEY trong bien moi truong.
    """
    import anthropic

    PROMPT = """Ban la chuyen gia phan tich CV tuyen dung.
Doc CV duoi day va yeu cau cong viec, cham diem tung tieu chi tu 0.0 den 1.0.

Vi tri: {job_title}
Yeu cau: {job_desc}

Noi dung CV (3500 ky tu dau):
{cv_text}

Tra ve JSON thuan (KHONG markdown, KHONG giai thich):
{{
  "candidate_name": "Ten day du cua ung vien",
  "Suitability":    0.0,
  "Language 1":     0.0,
  "Language 2":     0.0,
  "Language 3":     0.0,
  "Experience":     0.0,
  "Education":      0.0,
  "Recommendation": 0.0,
  "Availability":   0.0,
  "reasoning": {{
    "Suitability": "ly do ngan gon",
    "Experience":  "ly do ngan gon",
    "Education":   "ly do ngan gon"
  }}
}}

Huong dan cham diem:
- Suitability  : muc do phu hop tong the voi vi tri va yeu cau
- Language 1   : chat luong ngon ngu chinh (suy luan tu cach viet CV)
- Language 2   : ngoai ngu thu 2 duoc de cap (0.2 neu khong co)
- Language 3   : ngoai ngu thu 3 (0.1 neu khong co)
- Experience   : 0=khong, 0.3=<1nam, 0.5=1-2nam, 0.7=3-5nam, 1.0=5+nam
- Education    : 0.4=cao dang, 0.6=dai hoc, 0.8=thac si, 1.0=tien si
- Recommendation: 0.3=khong, 0.65=co reference, 1.0=thu gioi thieu
- Availability : 1.0=ngay, 0.6=1 thang, 0.3=3+ thang
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
    return json.loads(raw)


def extract_features(
    cv_text: str,
    job_title: str,
    job_desc: str,
    domain: str,
    use_claude: bool = False,
) -> dict:
    """Dieu huong sang Claude API hoac heuristic tuy theo cau hinh."""
    if use_claude and os.environ.get("ANTHROPIC_API_KEY"):
        try:
            return extract_features_claude(cv_text, job_title, job_desc)
        except Exception as e:
            st.warning(f"Claude API loi ({e}), chuyen sang heuristic.")
    return extract_features_heuristic(cv_text, job_title, job_desc, domain)


# ===========================================================================
# Ham cham diem ung vien (Scoring)
# ===========================================================================

def score_candidate(
    features:   dict,
    weights:    dict,
    model_choice: str,
) -> dict:
    """Cham diem 1 ung vien bang mo hinh da huan luyen.

    Quy trinh:
      1. Chon mo hinh va scaler tuong ung (LR / MLP).
      2. Nhan trong so tieu chi vao vector dac trung truoc khi scale.
         (Buoc nay dam bao slider real-time anh huong truc tiep den diem.)
      3. Scale bang scaler da fit tren FairCVdb.
      4. Du doan xac suat "Recommended".
      5. Tinh diem cuoi = 60% diem mo hinh + 40% criteria match.

    Parameters
    ----------
    features    : dict {feature_name -> float [0,1]}.
    weights     : dict {feature_name -> weight [0,1]} tu sidebar.
    model_choice: 'LR (Logistic Regression)' hoac 'MLP (Neural Network)'.

    Returns
    -------
    dict voi cac key: model_score, criteria_score, final_score, verdict, ...
    """
    # -- Chon mo hinh va scaler theo lua chon nguoi dung ---------------
    if model_choice.startswith("MLP"):
        model  = mlp_model
        scaler = mlp_scaler
        model_label = "MLP"
    else:
        model  = lr_model
        scaler = lr_scaler
        model_label = "LR"

    # -- Vector dac trung theo dung thu tu FEAT_COLS --------------------
    x_raw = np.array([float(features.get(c, 0.5)) for c in FEAT_COLS])

    # -- Ap trong so tieu chi nha tuyen dung vao vector (REAL-TIME) -----
    # Cach nay dam bao slider thay doi tren sidebar lam thay doi ngay
    # thu tu xep hang ma khong can re-train mo hinh.
    w_vec = np.array([weights.get(c, 0.0) for c in FEAT_COLS])
    total_w = w_vec.sum() or 1.0

    # Criteria score: trung binh co trong so
    criteria = float(np.dot(w_vec, x_raw) / total_w)

    # Vector dac trung co trong so (dua vao mo hinh)
    x_weighted = x_raw * (1.0 + 0.5 * w_vec / (w_vec.max() + 1e-9))

    # -- Chuan hoa (scale) de tranh lech thang diem dau vao AI ---------
    X = x_weighted.reshape(1, -1)
    X_scaled = scaler.transform(X)

    # -- Du doan xac suat boi mo hinh da huan luyen tren FairCVdb ------
    model_prob = float(model.predict_proba(X_scaled)[0][1])

    # -- Diem cuoi hop nhat: 60% mo hinh + 40% criteria match ----------
    final = 0.60 * model_prob + 0.40 * criteria

    tier_color = (
        GOOD if final >= 0.75 else
        COMP if final >= 0.60 else
        WARN if final >= 0.45 else BAD
    )

    return {
        "model_label":   model_label,
        "model_score":   round(model_prob * 100, 1),
        "criteria_score":round(criteria  * 100, 1),
        "final_score":   round(final     * 100, 1),
        "verdict":       "Recommended" if model_prob >= 0.5 else "Not recommended",
        "tier_color":    tier_color,
        "x_raw":         x_raw,
    }


# ===========================================================================
# RECRUITER VIEW
# ===========================================================================

if role.startswith("Recruiter"):

    st.markdown(
        '<div class="hero">'
        '<h1>Recruiter -- Sang loc CV & Kiem toan cong bang</h1>'
        '<p>Dat tieu chi va trong so ben trai, upload goi ho so ZIP, '
        'de 2 mo hinh AI cham diem va xep hang ung vien mot cach cong bang.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    tab_screen, tab_fair, tab_shap = st.tabs([
        "Sang loc CV",
        "Fairness Audit",
        "SHAP -- Giai thich AI",
    ])

    # -------------------------------------------------------------------
    # Tab 1: Sang loc CV (Upload ZIP + Ranking)
    # -------------------------------------------------------------------
    with tab_screen:

        # -- Hien thi tieu chi dang hoat dong --------------------------
        st.markdown('<div class="sec">Tieu chi dang hoat dong</div>',
                    unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.markdown(
            _card("Vi tri",
                  job_title[:22] if len(job_title) > 22 else job_title,
                  COMP, f"Linh vuc: {domain}"),
            unsafe_allow_html=True,
        )
        top3_w = sorted(
            ((k, v) for k, v in weights.items() if v > 0),
            key=lambda x: -x[1],
        )[:4]
        c2.markdown(
            '<div class="card"><h4>Trong so uu tien</h4>'
            + "".join(
                f'<div style="display:flex;justify-content:space-between;'
                f'font-size:.88rem;margin:3px 0">'
                f'<span>{k}</span><b>{v:.0%}</b></div>'
                for k, v in top3_w
            )
            + '</div>',
            unsafe_allow_html=True,
        )
        c3.markdown(
            '<div class="card"><h4>Dam bao cong bang</h4>'
            '<div style="font-size:.84rem;line-height:1.8">'
            'Blind label training<br>'
            'Khong dung gender / ethnicity<br>'
            'Recommendation bi loai tru<br>'
            f'Mo hinh: {model_choice.split()[0]}'
            '</div></div>',
            unsafe_allow_html=True,
        )

        st.markdown("")

        # -- Upload goi ho so ZIP --------------------------------------
        st.markdown('<div class="sec">Upload goi ho so ung vien (.zip)</div>',
                    unsafe_allow_html=True)
        st.caption(
            "File ZIP nen chua cac file .txt hoac .pdf -- "
            "moi file la ho so cua 1 ung vien. "
            "Giai nen hoan toan trong RAM, khong ghi len server."
        )
        uploaded_zip = st.file_uploader(
            "Chon file ZIP",
            type=["zip"],
            help="Toi da ~200MB. Moi file = 1 ung vien.",
        )

        run_btn = st.button(
            "Bat dau cham diem",
            type="primary",
            disabled=(uploaded_zip is None),
        )

        if run_btn and uploaded_zip is not None:
            with st.spinner("Dang giai nen va phan tich ho so..."):

                # -- Buoc 1: Giai nen ZIP trong RAM ---------------------
                try:
                    cv_list = extract_cvs_from_zip(uploaded_zip)
                except ValueError as e:
                    st.error(f"Loi file ZIP: {e}")
                    st.stop()

                if not cv_list:
                    st.warning("Khong tim thay file .txt hoac .pdf trong ZIP.")
                    st.stop()

                st.info(f"Tim thay {len(cv_list)} ho so trong ZIP.")

                # -- Buoc 2: Trich xuat dac trung va cham diem ---------
                results, errors = [], []
                progress = st.progress(0, text="Dang xu ly...")

                for i, cv_item in enumerate(cv_list):
                    progress.progress(
                        (i + 1) / len(cv_list),
                        text=f"Xu ly {i+1}/{len(cv_list)}: {cv_item['filename']}",
                    )
                    try:
                        feats = extract_features(
                            cv_item["text"], job_title, job_desc,
                            domain, use_claude,
                        )
                        name      = feats.pop("candidate_name", cv_item["filename"])
                        reasoning = feats.pop("reasoning", {})
                        gender_px = feats.pop("Gender_Proxy", 0.5)

                        scores = score_candidate(feats, weights, model_choice)

                        results.append({
                            "Ung vien":        name,
                            "File":            cv_item["filename"],
                            "Diem cuoi":       scores["final_score"],
                            "Diem mo hinh":    scores["model_score"],
                            "Phu hop tieu chi":scores["criteria_score"],
                            "Verdict":         scores["verdict"],
                            "tier_color":      scores["tier_color"],
                            "Gender_Proxy":    gender_px,
                            **{k: round(float(feats.get(k, 0)), 3) for k in FEAT_COLS},
                            "_reasoning":      reasoning,
                        })

                    except Exception as e:
                        errors.append(f"{cv_item['filename']}: {e}")

                progress.empty()
                if errors:
                    st.warning("Loi o mot so file:\n" + "\n".join(errors))

            if results:
                # -- Buoc 3: Xep hang theo Diem cuoi (da co trong so) --
                ranked = sorted(results, key=lambda x: x["Diem cuoi"], reverse=True)
                for i, r in enumerate(ranked):
                    r["rank"] = i + 1
                    if   i == 0: r["tier_label"], r["tier_badge"] = "Top 1",  "b-top1"
                    elif i < 3:  r["tier_label"], r["tier_badge"] = "Top 3",  "b-top3"
                    elif i < 5:  r["tier_label"], r["tier_badge"] = "Top 5",  "b-top5"
                    else:        r["tier_label"], r["tier_badge"] = "Top 10", "b-top10"

                st.session_state["ranked"]  = ranked
                st.session_state["weights"] = weights
                st.success(f"Da cham diem {len(ranked)} ung vien.")

        # -- Hien thi ket qua (cap nhat theo within so real-time) ------
        if "ranked" in st.session_state:
            raw_ranked = st.session_state["ranked"]
            cur_weights = weights   # Lay trong so HIEN TAI tu sidebar

            # -- Re-rank theo trong so hien tai (real-time slider) ------
            def _rerank(rows, cur_w):
                """Tinh lai Diem cuoi theo trong so hien tai va sap xep lai."""
                w_vec  = np.array([cur_w.get(c, 0.0) for c in FEAT_COLS])
                total  = w_vec.sum() or 1.0
                for r in rows:
                    x = np.array([float(r.get(c, 0.5)) for c in FEAT_COLS])
                    r["Phu hop tieu chi"] = round(float(np.dot(w_vec, x) / total) * 100, 1)
                    model_s = r["Diem mo hinh"] / 100.0
                    crit_s  = r["Phu hop tieu chi"] / 100.0
                    r["Diem cuoi"] = round((0.60 * model_s + 0.40 * crit_s) * 100, 1)
                return sorted(rows, key=lambda x: x["Diem cuoi"], reverse=True)

            ranked = _rerank(raw_ranked, cur_weights)
            n = len(ranked)

            # -- KPI row ------------------------------------------------
            st.markdown('<div class="sec">Ket qua xep hang</div>',
                        unsafe_allow_html=True)
            k1, k2, k3 = st.columns(3)
            k1.metric("Tong ung vien", n)
            k2.metric("Duoc khuyen nghi",
                      sum(1 for r in ranked if r["Verdict"] == "Recommended"))
            k3.metric("Diem trung binh",
                      f"{np.mean([r['Diem cuoi'] for r in ranked]):.1f}")

            # Canh bao ung vien co gender proxy
            proxy_warn = [r for r in ranked if r.get("Gender_Proxy") in (0.0, 1.0)]
            if proxy_warn:
                st.markdown(
                    f'<div class="warn-box">'
                    f'{len(proxy_warn)} ho so co tu ngu chi gioi tinh ro rang.  '
                    f'Mo hinh KHONG dung thong tin nay de cham diem, nhung HR '
                    f'can luu y ve rui ro bias thu cong khi xem xet.'
                    f'</div>',
                    unsafe_allow_html=True,
                )

            # -- Tabs xep hang ------------------------------------------
            rt1, rt2, rt3, rt4 = st.tabs(["Top 10", "Top 5", "Top 3", "Top 1"])

            def _render_list(pool):
                for r in pool:
                    st.markdown(
                        f'<div class="cand-row">'
                        f'<span class="cand-rank">#{r["rank"]}</span>'
                        f'<span class="cand-name">{r["Ung vien"]}</span>'
                        f'<span class="badge {r["tier_badge"]}">{r["tier_label"]}</span>'
                        f'<span class="cand-score">{r["Diem cuoi"]:.1f}</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )

            with rt1:
                pool10 = ranked[:min(10, n)]
                _render_list(pool10)
                df_show = pd.DataFrame(pool10)[[
                    "rank", "Ung vien", "Diem cuoi",
                    "Diem mo hinh", "Phu hop tieu chi", "Verdict",
                ]]
                st.dataframe(df_show, use_container_width=True, hide_index=True)

                # Bieu do thanh so sanh
                fig, ax = plt.subplots(figsize=(10, 3.8))
                names = [r["Ung vien"][:14] for r in pool10]
                ms    = [r["Diem mo hinh"]    for r in pool10]
                cs    = [r["Phu hop tieu chi"] for r in pool10]
                fs    = [r["Diem cuoi"]        for r in pool10]
                x     = np.arange(len(names))
                w_bar = 0.27
                ax.bar(x - w_bar, ms, w_bar, label="Mo hinh AI",
                       color=COMP, alpha=.85, edgecolor="white")
                ax.bar(x,         cs, w_bar, label="Phu hop tieu chi",
                       color=GOOD, alpha=.85, edgecolor="white")
                ax.bar(x + w_bar, fs, w_bar, label="Diem cuoi",
                       color=PRXY, alpha=.85, edgecolor="white")
                ax.set_xticks(x)
                ax.set_xticklabels(names, rotation=30, ha="right", fontsize=8)
                ax.set_ylabel("Diem (%)")
                ax.set_ylim(0, 105)
                ax.legend(fontsize=8)
                ax.set_title("So sanh diem -- Top 10",
                             fontweight="bold", color=INK)
                for sp in ["top", "right"]:
                    ax.spines[sp].set_visible(False)
                ax.grid(axis="y", alpha=0.18)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig)

            with rt2:
                _render_list(ranked[:min(5, n)])
                cols5 = st.columns(min(5, n))
                for j, (r, col) in enumerate(zip(ranked[:5], cols5)):
                    col.markdown(
                        f'<div class="card" style="text-align:center">'
                        f'<div style="font-size:.78rem;color:{MUT}">#{r["rank"]}</div>'
                        f'<div class="kpi" style="color:{r["tier_color"]};'
                        f'font-size:1.65rem">{r["Diem cuoi"]:.0f}</div>'
                        f'<div style="font-weight:600;margin:5px 0 3px;'
                        f'font-size:.86rem">{r["Ung vien"][:16]}</div>'
                        f'<div style="font-size:.78rem;color:{MUT}">'
                        f'AI {r["Diem mo hinh"]:.0f}% &nbsp; Tieu chi '
                        f'{r["Phu hop tieu chi"]:.0f}%</div></div>',
                        unsafe_allow_html=True,
                    )

            with rt3:
                for r in ranked[:min(3, n)]:
                    with st.expander(
                        f"#{r['rank']}  {r['Ung vien']}  "
                        f"--  {r['Diem cuoi']:.1f} diem  |  {r['Verdict']}",
                        expanded=(r["rank"] == 1),
                    ):
                        ca, cb = st.columns(2)
                        ca.metric("Diem mo hinh",     f"{r['Diem mo hinh']:.1f}%")
                        cb.metric("Phu hop tieu chi", f"{r['Phu hop tieu chi']:.1f}%")

                        fv  = [r.get(f, 0) for f in FEAT_COLS]
                        bc  = [GOOD if v >= 0.65 else WARN if v >= 0.4 else BAD
                               for v in fv]
                        fig3, ax3 = plt.subplots(figsize=(8, 3))
                        ax3.barh(FEAT_NAMES, fv, color=bc,
                                 edgecolor="white", height=.6)
                        ax3.axvline(0.5, color="gray", ls="--", lw=.8, alpha=.6)
                        ax3.set_xlim(0, 1.15)
                        ax3.set_xlabel("Diem dac trung (0-1)")
                        for sp in ["top", "right"]:
                            ax3.spines[sp].set_visible(False)
                        plt.tight_layout()
                        st.pyplot(fig3)
                        plt.close(fig3)

                        # AI reasoning neu co
                        rsn = r.get("_reasoning", {})
                        if rsn and "note" not in rsn:
                            st.markdown("**Phan tich AI:**")
                            for k, v in list(rsn.items())[:4]:
                                if isinstance(v, str):
                                    st.markdown(f"- **{k}**: {v}")

            with rt4:
                if n >= 1:
                    w = ranked[0]
                    st.markdown(
                        f'<div style="text-align:center;padding:18px 0 10px">'
                        f'<div class="kpi" style="font-size:2.4rem;'
                        f'color:{w["tier_color"]}">{w["Ung vien"]}</div>'
                        f'<div style="font-size:.95rem;color:{MUT};margin-top:5px">'
                        f'Ung vien xuat sac nhat  --  {w["Verdict"]}'
                        f'</div></div>',
                        unsafe_allow_html=True,
                    )
                    wa, wb, wc = st.columns(3)
                    wa.metric("Diem cuoi",       f"{w['Diem cuoi']:.1f} / 100")
                    wb.metric("Mo hinh AI",      f"{w['Diem mo hinh']:.1f}%")
                    wc.metric("Phu hop tieu chi",f"{w['Phu hop tieu chi']:.1f}%")

                    fv  = [w.get(f, 0) for f in FEAT_COLS]
                    bc  = [GOOD if v >= 0.65 else WARN if v >= 0.4 else BAD
                           for v in fv]
                    fig4, ax4 = plt.subplots(figsize=(9, 3.6))
                    bars = ax4.bar(FEAT_NAMES, fv, color=bc, edgecolor="white")
                    ax4.axhline(0.5, color="gray", ls="--", lw=1, alpha=.5)
                    ax4.set_ylim(0, 1.15)
                    for bar, val in zip(bars, fv):
                        ax4.text(bar.get_x() + bar.get_width() / 2, val + 0.02,
                                 f"{val:.2f}", ha="center", va="bottom",
                                 fontsize=8, fontweight="bold")
                    ax4.set_ylabel("Diem dac trung (0-1)")
                    ax4.set_title(f"Profile nang luc -- {w['Ung vien']}",
                                  fontweight="bold", color=INK)
                    for sp in ["top", "right"]:
                        ax4.spines[sp].set_visible(False)
                    plt.xticks(rotation=25, ha="right", fontsize=9)
                    plt.tight_layout()
                    st.pyplot(fig4)
                    plt.close(fig4)

            # Download ket qua
            st.markdown("")
            dl_df = pd.DataFrame(ranked)[[
                "rank", "Ung vien", "Diem cuoi", "Diem mo hinh",
                "Phu hop tieu chi", "Verdict",
            ] + FEAT_COLS]
            st.download_button(
                "Tai ket qua (CSV)",
                data=dl_df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig"),
                file_name=f"faircv_{job_title.replace(' ','_')}.csv",
                mime="text/csv",
            )

    # -------------------------------------------------------------------
    # Tab 2: Fairness Audit tren FairCVdb test set
    # -------------------------------------------------------------------
    with tab_fair:
        st.markdown(
            "Chi so fairness duoc tinh tren **4 800 ho so test** cua FairCVdb "
            "(du lieu that, tap test giu lai khi huan luyen).  "
            "Day la bao chung kien minh mo hinh hanh xu cong bang nhu the nao "
            "tren tap du lieu co kiem soat -- doc lap voi cac CV ban vua upload."
        )

        if not DEMO_MODE:
            lrm  = cfg.get("lr_metrics",  {})
            mlpm = cfg.get("mlp_metrics", {})

            st.markdown('<div class="sec">So sanh LR vs MLP (Setting A, blind label)</div>',
                        unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("**Logistic Regression**")
                st.markdown(
                    f"| Chi so | Gia tri |\n|---|---|\n"
                    f"| F1 | **{lrm.get('F1',0):.4f}** |\n"
                    f"| ROC-AUC | **{lrm.get('ROC-AUC',0):.4f}** |\n"
                    f"| DP Gap (gender) | **{lrm.get('DP_Gap_Gender',0):.4f}** |\n"
                    f"| EOO Gap (gender)| **{lrm.get('EOO_Gap_Gender',0):.4f}** |\n"
                    f"| DP Gap (ethnicity)| **{lrm.get('DP_Gap_Ethnicity',0):.4f}** |"
                )
            with c2:
                st.markdown("**MLP (Neural Network)**")
                st.markdown(
                    f"| Chi so | Gia tri |\n|---|---|\n"
                    f"| F1 | **{mlpm.get('F1',0):.4f}** |\n"
                    f"| ROC-AUC | **{mlpm.get('ROC-AUC',0):.4f}** |\n"
                    f"| DP Gap (gender) | **{mlpm.get('DP_Gap_Gender',0):.4f}** |\n"
                    f"| EOO Gap (gender)| **{mlpm.get('EOO_Gap_Gender',0):.4f}** |\n"
                    f"| DP Gap (ethnicity)| **{mlpm.get('DP_Gap_Ethnicity',0):.4f}** |"
                )

            # Giai thich chi so
            st.markdown(
                '<div class="fair-note">'
                '<b>Giai thich chi so cong bang:</b><br>'
                '<b>DP Gap</b> (Demographic Parity Gap): '
                '|P(select|A=1) - P(select|A=0)| -- '
                'chenh lech ty le duoc chon giua 2 nhom. '
                'Gap < 0.05 duoc coi la chap nhan duoc.<br>'
                '<b>EOO Gap</b> (Equal Opportunity Gap): '
                '|TPR(A=1) - TPR(A=0)| -- chenh lech ty le True Positive.<br>'
                '<b>Mo hinh da ap dung:</b> '
                'T1: Loai bo cot gender/ethnicity (Setting A). '
                'T3: Tai trong so mau (inverse-frequency reweighting).'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.warning(
                "Dang o che do demo (mo hinh tren du lieu mo phong). "
                "Chi so fairness tren du lieu nay khong co y nghia thong ke. "
                "Dat FairCVdb.csv that va chay lai train_models.py."
            )

    # -------------------------------------------------------------------
    # Tab 3: SHAP -- Giai thich AI
    # -------------------------------------------------------------------
    with tab_shap:
        st.markdown(
            "Bieu do SHAP cho biet **dac trung nao co anh huong lon nhat** "
            "den quyet dinh cua mo hinh AI, giup HR hieu va kiem soat "
            "tieu chi cham diem mot cach minh bach."
        )

        if "ranked" not in st.session_state:
            st.info("Upload va cham diem ho so o tab 'Sang loc CV' truoc.")
        else:
            ranked_shap = st.session_state["ranked"]

            try:
                import shap

                # Chon mo hinh de giai thich
                if model_choice.startswith("MLP"):
                    model_shap  = mlp_model
                    scaler_shap = mlp_scaler
                else:
                    model_shap  = lr_model
                    scaler_shap = lr_scaler

                # Xay dung ma tran dac trung cua cac ung vien da cham diem
                X_cands = np.array([
                    [float(r.get(f, 0.5)) for f in FEAT_COLS]
                    for r in ranked_shap
                ])
                X_scaled_shap = scaler_shap.transform(X_cands)

                # Tinh SHAP values
                try:
                    explainer  = shap.LinearExplainer(
                        model_shap, X_scaled_shap,
                        feature_perturbation="interventional",
                    )
                    shap_vals  = explainer.shap_values(X_scaled_shap)
                except Exception:
                    # Fallback sang KernelExplainer
                    explainer  = shap.KernelExplainer(
                        lambda x: model_shap.predict_proba(x)[:, 1],
                        shap.sample(X_scaled_shap, min(50, len(X_scaled_shap))),
                    )
                    shap_vals  = explainer.shap_values(
                        X_scaled_shap, nsamples=100,
                    )

                # -- Bieu do Feature Importance tong the ---------------
                mean_abs = np.abs(shap_vals).mean(axis=0)
                order    = np.argsort(mean_abs)

                fig_s, ax_s = plt.subplots(figsize=(8, 4))
                ax_s.barh(
                    [FEAT_NAMES[i] for i in order],
                    mean_abs[order],
                    color=COMP, edgecolor="white",
                )
                ax_s.set_xlabel("Mean |SHAP value|")
                ax_s.set_title(
                    f"Do quan trong dac trung ({model_choice.split()[0]}) "
                    "-- SHAP Feature Importance",
                    fontweight="bold", color=INK,
                )
                for sp in ["top", "right"]:
                    ax_s.spines[sp].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig_s)
                plt.close(fig_s)

                # -- Bieu do SHAP dot plot (ung vien vs dac trung) ------
                st.markdown("**SHAP co dong gop tung dac trung -- moi diem = 1 ung vien**")
                fig_b, ax_b = plt.subplots(figsize=(9, 4))
                for j, fname in enumerate(FEAT_NAMES):
                    vals   = shap_vals[:, j]
                    scores = X_cands[:, j]
                    colors = [GOOD if s >= 0.6 else WARN if s >= 0.4 else BAD
                              for s in scores]
                    ax_b.scatter(vals, [j] * len(vals),
                                 c=colors, alpha=0.7, s=40, edgecolors="white")
                ax_b.set_yticks(range(len(FEAT_NAMES)))
                ax_b.set_yticklabels(FEAT_NAMES, fontsize=9)
                ax_b.axvline(0, color="gray", lw=0.8)
                ax_b.set_xlabel("SHAP value (anh huong den diem)")
                ax_b.set_title("Phan bo dong gop SHAP theo dac trung",
                               fontweight="bold", color=INK)
                legend_p = [
                    mpatches.Patch(color=GOOD, label="Diem cao (>=0.6)"),
                    mpatches.Patch(color=WARN, label="Diem trung (0.4-0.6)"),
                    mpatches.Patch(color=BAD,  label="Diem thap (<0.4)"),
                ]
                ax_b.legend(handles=legend_p, fontsize=8, loc="lower right")
                for sp in ["top", "right"]:
                    ax_b.spines[sp].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig_b)
                plt.close(fig_b)

            except ImportError:
                # Fallback: ve bieu do trong so mo hinh (feature coefficients)
                st.warning("Thu vien `shap` chua duoc cai. Hien thi Feature Importance thay the.")
                if hasattr(lr_model, "coef_"):
                    coefs = np.abs(lr_model.coef_.ravel())
                    order = np.argsort(coefs)
                    fig_fb, ax_fb = plt.subplots(figsize=(8, 4))
                    ax_fb.barh(
                        [FEAT_NAMES[i] for i in order],
                        coefs[order],
                        color=COMP, edgecolor="white",
                    )
                    ax_fb.set_xlabel("|He so LR| (proxy cho do quan trong)")
                    ax_fb.set_title("Feature Importance (LR coefficients)",
                                    fontweight="bold", color=INK)
                    for sp in ["top", "right"]:
                        ax_fb.spines[sp].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig_fb)
                    plt.close(fig_fb)
                else:
                    st.info("Cai shap de xem bieu do: pip install shap")


# ===========================================================================
# CANDIDATE VIEW
# ===========================================================================

else:
    st.markdown(
        '<div class="hero">'
        '<h1>Candidate -- Bao cao fairness ca nhan</h1>'
        '<p>Upload CV cua ban, xem diem duoc cham nhu the nao, '
        'dac trung nao anh huong ket qua va danh gia muc do cong bang.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # Thong tin vi tri ung tuyen
    st.markdown('<div class="sec">Vi tri ban ung tuyen</div>', unsafe_allow_html=True)
    ca, cb = st.columns(2)
    with ca:
        cand_job    = st.text_input("Vi tri", "Data Analyst")
    with cb:
        cand_domain = st.selectbox(
            "Linh vuc", ["IT / Cong nghe", "Marketing",
                         "Tai chinh / Ke toan", "Nhan su / HR", "Khac"]
        )
    cand_desc = st.text_area(
        "Yeu cau thuong gap",
        "Python, SQL, phan tich du lieu, 2+ nam kinh nghiem", height=65,
    )

    # Upload CV
    st.markdown('<div class="sec">Upload CV cua ban (PDF)</div>', unsafe_allow_html=True)
    cand_file = st.file_uploader("Chon file PDF", type=["pdf"])

    if cand_file and st.button("Phan tich CV cua toi", type="primary"):
        with st.spinner("Dang phan tich..."):
            # Doc PDF
            try:
                from pdfminer.high_level import extract_text as _pdf_text
                raw = cand_file.read()
                cv_text = _pdf_text(io.BytesIO(raw))
            except ImportError:
                cand_file.seek(0)
                cv_text = cand_file.read().decode("utf-8", errors="ignore")

            cv_text = re.sub(r"\n{3,}", "\n\n", cv_text).strip()

            if len(cv_text) < 40:
                st.error("Khong doc duoc noi dung CV. Vui long kiem tra file PDF.")
                st.stop()

            cand_w = {c: 1.0 / len(FEAT_COLS) for c in FEAT_COLS}
            feats  = extract_features(
                cv_text, cand_job, cand_desc, cand_domain, use_claude_cand
            )
            cand_name = feats.pop("candidate_name", "Ban")
            reasoning = feats.pop("reasoning", {})
            gender_px = feats.pop("Gender_Proxy", 0.5)
            scores    = score_candidate(feats, cand_w, model_choice)
            st.session_state["cand"] = {
                "name": cand_name, "feats": feats,
                "scores": scores, "reasoning": reasoning,
                "gender_proxy": gender_px,
            }

    if "cand" in st.session_state:
        cr = st.session_state["cand"]
        sc = cr["scores"]
        ft = cr["feats"]

        # -- Ket qua chinh --
        st.markdown('<div class="sec">Ket qua danh gia</div>', unsafe_allow_html=True)
        dec_color = GOOD if sc["verdict"] == "Recommended" else BAD
        dec_label = "DUOC KHUYEN NGHI" if sc["verdict"] == "Recommended" else "CHUA DUOC khuyen nghi"

        d1, d2, d3 = st.columns(3)
        d1.markdown(
            _card("Quyet dinh", dec_label, dec_color,
                  f"Diem: {sc['final_score']:.1f} / 100"),
            unsafe_allow_html=True,
        )
        d2.metric("Diem mo hinh AI",   f"{sc['model_score']:.1f}%")
        d3.metric("Phu hop tieu chi",  f"{sc['criteria_score']:.1f}%")

        # -- Profile dac trung --
        st.markdown('<div class="sec">Profile nang luc cua ban</div>',
                    unsafe_allow_html=True)
        fv = [ft.get(f, 0) for f in FEAT_COLS]
        bc = [GOOD if v >= 0.65 else WARN if v >= 0.4 else BAD for v in fv]

        fig_c, ax_c = plt.subplots(figsize=(9, 3.6))
        bars_c = ax_c.barh(FEAT_NAMES, fv, color=bc, edgecolor="white", height=.6)
        ax_c.axvline(0.5, color="gray", ls="--", lw=.9, alpha=.5, label="Nguong trung binh")
        ax_c.set_xlim(0, 1.15)
        ax_c.set_xlabel("Diem dac trung (0-1)")
        ax_c.set_title(f"Profile nang luc -- {cr['name']}", fontweight="bold", color=INK)
        legend_c = [
            mpatches.Patch(color=GOOD, label="Manh (>=0.65)"),
            mpatches.Patch(color=WARN, label="Trung binh (0.4-0.65)"),
            mpatches.Patch(color=BAD,  label="Can cai thien (<0.4)"),
        ]
        ax_c.legend(handles=legend_c, fontsize=8, loc="lower right")
        for sp in ["top", "right"]:
            ax_c.spines[sp].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_c)
        plt.close(fig_c)

        # -- Phan tich AI --
        rsn = cr.get("reasoning", {})
        if rsn and "note" not in rsn:
            st.markdown('<div class="sec">Phan tich AI theo tung tieu chi</div>',
                        unsafe_allow_html=True)
            for k, v in rsn.items():
                if isinstance(v, str):
                    val   = ft.get(k, 0.5)
                    icon  = "✓" if val >= 0.65 else ("!" if val >= 0.4 else "x")
                    color = GOOD if val >= 0.65 else (WARN if val >= 0.4 else BAD)
                    st.markdown(
                        f'<span style="color:{color};font-weight:700">{icon}</span> '
                        f'**{k}** ({val:.2f}): {v}',
                        unsafe_allow_html=True,
                    )

        # -- Fairness verdict --
        st.markdown('<div class="sec">Danh gia cong bang</div>', unsafe_allow_html=True)

        gp = cr.get("gender_proxy", 0.5)
        if gp in (0.0, 1.0):
            group_name = "Nu" if gp == 1.0 else "Nam"
            st.markdown(
                f'<div class="verdict-warn">'
                f'Ho so co tu ngu chi gioi tinh ({group_name}) duoc phat hien.  '
                f'Mo hinh FairCV <b>khong dung thong tin nay</b> de cham diem '
                f'(Gender_Proxy khong nam trong FEAT_COLS).  '
                f'Tuy nhien, nen luu y rang nha tuyen dung con nguoi co the '
                f'bi anh huong boi thong tin nay khi doc CV thu cong.'
                f'</div>',
                unsafe_allow_html=True,
            )
        elif not DEMO_MODE:
            mlpm = cfg.get("mlp_metrics", {})
            dp_g = mlpm.get("DP_Gap_Gender", 0.0)
            if dp_g < 0.02:
                st.markdown(
                    f'<div class="verdict-clear">'
                    f'Mo hinh MLP co DP Gap gioi tinh = {dp_g:.4f} '
                    f'(rat thap, dat tieu chuan cong bang).  '
                    f'Ket qua danh gia cua ban dua tren nang luc thuan tuy.'
                    f'</div>',
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f'<div class="verdict-warn">'
                    f'DP Gap gioi tinh = {dp_g:.4f} -- con chech lech giua cac nhom.  '
                    f'He thong dang ap dung Technique T3 (reweighting) '
                    f'de giam thieu phan biet doi xu.'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        with st.expander("Hieu them ve cach tinh diem cua ban", expanded=False):
            st.markdown(
                "**Trich xuat dac trung:** Van ban CV duoc phan tich de cho diem "
                "8 tieu chi nang luc (0.0 - 1.0), khop voi cau truc du lieu FairCVdb.  "
                "Claude API (neu co) hoac quy tac heuristic duoc dung cho buoc nay.\n\n"
                "**Cham diem:** Mo hinh da huan luyen tren 19 200 ho so FairCVdb that "
                "(80% tap huan luyen) du doan xac suat 'Recommended'.  "
                "Diem cuoi = 60% diem mo hinh + 40% phu hop tieu chi nha tuyen dung.\n\n"
                "**Bao dam cong bang:**\n"
                "- Gender va ethnicity KHONG co trong dac trung dau vao mo hinh.\n"
                "- Mo hinh duoc huan luyen tren blind_label (khong biet gioi tinh).\n"
                "- Sample reweighting giam chech lech giua cac nhom dan so.\n"
                "- 'Recommendation' bi loai tru de tranh bias mang luoi quan he."
            )

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("---")
st.caption(
    "FairCV v1.0 -- He thong sang loc CV cong bang. "
    "Mo hinh huan luyen tren FairCVdb (Complement et al., CVPRW 2020). "
    "Du lieu CV khong duoc luu tru -- chi dung trong phien lam viec hien tai."
)