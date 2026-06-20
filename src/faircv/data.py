"""
src/faircv/data.py
------------------
Tầng dữ liệu (Data Layer) cho hệ thống FairCV.

Chức năng chính
---------------
1. Tự động tải FairCVdb.csv từ Google Drive nếu chưa có ở máy cục bộ
   (hàm `download_dataset_if_missing`), với cơ chế dự phòng sinh dữ
   liệu mô phỏng khi quá trình tải thất bại.

2. Nạp dữ liệu vào Pandas DataFrame và cache trên Streamlit
   (hàm `load_faircv_dataset`).

3. Giải nén gói hồ sơ ZIP do nhà tuyển dụng upload lên giao diện,
   bóc tách nội dung text từng CV trong RAM mà không ghi đĩa
   (hàm `extract_cvs_from_zip`).

4. Loader truyền thống `load_faircv(csv_path)` dùng cho pipeline và
   CLI, trả về (DataFrame, Dataset).

Hằng số cột
-----------
COMPETENCY   : 8 đặc trưng năng lực đầu vào cho mô hình (Setting A).
COMP_NAMES   : Nhãn hiển thị tương ứng với COMPETENCY.
DEMO_COLS    : Cột nhân khẩu học (Setting B -- gây bias).
LABEL_COLS   : Các cột nhãn mục tiêu có trong FairCVdb.
GENDER_LABELS: {int -> str} ánh xạ nhãn giới tính nhị phân.
ETH_LABELS   : {int -> str} ánh xạ nhãn dân tộc 3 lớp.
"""
from __future__ import annotations

import io
import os
import zipfile
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Hằng số cấu hình Google Drive
# ---------------------------------------------------------------------------

# ID file công khai trên Google Drive chứa FairCVdb.csv.
# Thay bằng ID thực của file của bạn sau khi chia sẻ công khai.
# Ví dụ URL chia sẻ: https://drive.google.com/file/d/<GOOGLE_DRIVE_FILE_ID>/view
GOOGLE_DRIVE_FILE_ID = "YOUR_GOOGLE_DRIVE_FILE_ID_HERE"

# Đường dẫn lưu file CSV sau khi tải về
LOCAL_CSV_PATH = "data/FairCVdb.csv"

# ---------------------------------------------------------------------------
# Hằng số tên cột
# ---------------------------------------------------------------------------

# 8 đặc trưng năng lực chuẩn FairCV (Setting A -- không có thông tin nhạy cảm)
COMPETENCY = [
    "Suitability", "Language 1", "Language 2", "Language 3",
    "Experience", "Education", "Recommendation", "Availability",
]

# Nhãn hiển thị trực quan tương ứng với COMPETENCY
COMP_NAMES = [
    "Suitability", "Language 1", "Language 2", "Language 3",
    "Experience", "Education", "Recommendation", "Availability",
]

# Cột nhân khẩu học -- chỉ dùng trong Setting B (gây bias, chỉ để kiểm toán)
DEMO_COLS = ["gender", "ethnicity"]

# Cột nhúng khuôn mặt (Setting C, 20 chiều)
FACE_COLS = [f"face_emb_{i}" for i in range(20)]

# Cột nhúng ẩn danh (Setting D, 20 chiều)
BLIND_COLS = [f"blind_face_emb_{i}" for i in range(20)]

# Các cột nhãn có trong FairCVdb
LABEL_COLS = ["blind_label", "gender_label", "ethnicity_label"]

# Ánh xạ nhãn giới tính nhị phân
GENDER_LABELS = {0: "Male", 1: "Female"}

# Ánh xạ nhãn dân tộc 3 lớp
ETH_LABELS = {0: "Ethnicity 1", 1: "Ethnicity 2", 2: "Ethnicity 3"}

# Tập đặc trưng theo từng setting
FEATURE_SETS = {
    "A: Competency Only":           COMPETENCY,
    "B: Competency + Demographics": COMPETENCY + DEMO_COLS,
}

# ---------------------------------------------------------------------------
# Dataset container
# ---------------------------------------------------------------------------

@dataclass
class Dataset:
    """Container nhẹ được truyền qua các bước trong pipeline."""

    X: pd.DataFrame           # Ma trận đặc trưng (tất cả cột mô hình nhìn thấy)
    y: np.ndarray             # Nhãn khuyến nghị nhị phân (blind label)
    A_gender: np.ndarray      # Thuộc tính giới tính nhị phân (0=Male, 1=Female)
    A_ethnicity: np.ndarray   # Thuộc tính dân tộc 3 lớp
    feature_names: list       # Tên cột theo thứ tự khớp với X
    name: str = "FairCVdb"
    meta: dict = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.y)

    def split(self, test_frac: float = 0.2, seed: int = 42):
        """Trả về (train_idx, test_idx) arrays."""
        rng = np.random.default_rng(seed)
        idx = rng.permutation(self.n)
        cut = int(self.n * (1 - test_frac))
        return idx[:cut], idx[cut:]


# ---------------------------------------------------------------------------
# Hàm 1: Tự động tải FairCVdb từ Google Drive
# ---------------------------------------------------------------------------

def download_dataset_if_missing(
    file_id: str = GOOGLE_DRIVE_FILE_ID,
    local_path: str = LOCAL_CSV_PATH,
) -> bool:
    """Kiểm tra và tự động tải FairCVdb.csv từ Google Drive nếu chưa có.

    Sử dụng thư viện `gdown` -- cài bằng: pip install gdown

    Cơ chế dự phòng (Fallback)
    --------------------------
    Nếu quá trình tải gặp lỗi (mất mạng, ID sai, hết hạn quyền...), hàm
    tự động gọi `_generate_synthetic_fallback()` để tạo file CSV mô phỏng
    nhỏ (3 000 dòng) tại `local_path`, đảm bảo app không bị crash.

    Parameters
    ----------
    file_id   : ID file công khai Google Drive.
    local_path: Đường dẫn đích lưu file CSV.

    Returns
    -------
    bool
        True nếu file thực sự đã tải về hoặc đã tồn tại từ trước.
        False nếu phải dùng dữ liệu mô phỏng (fallback).
    """
    # Nếu file đã tồn tại, không cần tải lại
    if os.path.exists(local_path):
        return True

    # Tạo thư mục đích nếu chưa có
    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)

    # Kiểm tra xem ID đã được cấu hình thật chưa
    if file_id == "YOUR_GOOGLE_DRIVE_FILE_ID_HERE" or not file_id:
        print(
            "[data.py] GOOGLE_DRIVE_FILE_ID chưa được cấu hình. "
            "Đang tạo dữ liệu mô phỏng (demo mode)..."
        )
        _generate_synthetic_fallback(local_path)
        return False

    # Thử tải từ Google Drive
    try:
        import gdown  # pip install gdown
        url = f"https://drive.google.com/uc?id={file_id}"
        print(f"[data.py] Đang tải FairCVdb.csv từ Google Drive (ID={file_id})...")
        gdown.download(url, local_path, quiet=False)

        # Kiểm tra file tải về có hợp lệ không
        if os.path.exists(local_path) and os.path.getsize(local_path) > 1024:
            print(f"[data.py] Tải thành công: {local_path}")
            return True
        else:
            raise ValueError("File tải về rỗng hoặc quá nhỏ.")

    except Exception as exc:
        print(
            f"[data.py] Tải thất bại: {exc}\n"
            "Đang tạo dữ liệu mô phỏng thay thế (fallback)..."
        )
        _generate_synthetic_fallback(local_path)
        return False


def _generate_synthetic_fallback(
    local_path: str,
    n: int = 3000,
    seed: int = 42,
) -> None:
    """Tạo file CSV mô phỏng nhỏ để app không bị crash khi thiếu dữ liệu thật.

    File này có đúng cấu trúc cột như FairCVdb thật, nhưng chỉ gồm
    dữ liệu ngẫu nhiên được kiểm soát bởi `seed`.  Độ chính xác của
    mô hình sẽ thấp hơn nhiều so với dữ liệu thật.

    Parameters
    ----------
    local_path: Đường dẫn đích ghi file CSV.
    n         : Số dòng mô phỏng.
    seed      : Seed ngẫu nhiên để tái tạo.
    """
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.uniform(0, 1, n) for c in COMPETENCY})
    df["gender"]    = rng.integers(0, 2, n)
    df["ethnicity"] = rng.integers(0, 3, n)

    # Nhãn: trung bình đặc trưng + nhiễu nhỏ (mô phỏng blind label)
    raw = df[COMPETENCY].mean(axis=1).values + rng.normal(0, 0.05, n)
    df["blind_label"]     = np.clip(raw, 0, 1)
    df["gender_label"]    = df["blind_label"]
    df["ethnicity_label"] = df["blind_label"]

    # Phân chia train/test theo tỉ lệ 80/20
    split_col = ["train"] * int(n * 0.8) + ["test"] * (n - int(n * 0.8))
    df["split"] = split_col

    os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
    df.to_csv(local_path, index=False)
    print(f"[data.py] Đã tạo {n} dòng dữ liệu mô phỏng tại '{local_path}'.")


# ---------------------------------------------------------------------------
# Hàm 2: Nạp dataset (có cache Streamlit)
# ---------------------------------------------------------------------------

def load_faircv_dataset(
    csv_path: str = LOCAL_CSV_PATH,
    file_id: str = GOOGLE_DRIVE_FILE_ID,
):
    """Tải về (nếu cần) và nạp FairCVdb vào DataFrame, có cache Streamlit.

    Hàm này tự động:
      1. Gọi `download_dataset_if_missing()` để đảm bảo file CSV tồn tại.
      2. Đọc file CSV thành DataFrame.
      3. Cache kết quả bằng `@st.cache_data` của Streamlit để các lần
         gọi sau không cần đọc lại đĩa.

    Parameters
    ----------
    csv_path : Đường dẫn cục bộ của file CSV.
    file_id  : ID Google Drive (truyền vào để hàm có thể tải nếu thiếu).

    Returns
    -------
    tuple[pd.DataFrame, Dataset]
        DataFrame thô và Dataset container sẵn sàng dùng trong pipeline.
    """
    # Bước 1: Đảm bảo file tồn tại
    download_dataset_if_missing(file_id=file_id, local_path=csv_path)

    # Bước 2: Nạp và trả về (hàm core load_faircv xử lý rename + split)
    return load_faircv(csv_path)


def _cached_load_faircv_dataset(csv_path: str = LOCAL_CSV_PATH,
                                 file_id: str = GOOGLE_DRIVE_FILE_ID):
    """Wrapper có @st.cache_data -- chỉ khả dụng khi chạy trong Streamlit."""
    try:
        import streamlit as st

        @st.cache_data(show_spinner="Đang tải dữ liệu FairCVdb...")
        def _inner(cp, fid):
            return load_faircv_dataset(cp, fid)

        return _inner(csv_path, file_id)
    except Exception:
        # Fallback khi gọi ngoài môi trường Streamlit (CLI, tests)
        return load_faircv_dataset(csv_path, file_id)


# ---------------------------------------------------------------------------
# Hàm 3: Giải nén gói hồ sơ ZIP (In-memory)
# ---------------------------------------------------------------------------

def extract_cvs_from_zip(uploaded_zip_file) -> list[dict]:
    """Giải nén file ZIP chứa nhiều CV text, trả về danh sách dict.

    Hàm này hoàn toàn hoạt động trong RAM (in-memory) -- không ghi
    bất kỳ file nào ra đĩa, phù hợp với môi trường Streamlit Cloud.

    Định dạng hỗ trợ bên trong ZIP: `.txt`, `.pdf` (đọc text thô).
    File không phải text hoặc lỗi decode sẽ bị bỏ qua có thông báo.

    Parameters
    ----------
    uploaded_zip_file : Streamlit UploadedFile object (io.BytesIO-like).

    Returns
    -------
    list[dict]
        Mỗi phần tử là một dict:
        {
            "filename"  : str,   # Tên file gốc bên trong ZIP
            "text"      : str,   # Nội dung text đã decode UTF-8
            "char_count": int,   # Số ký tự (dùng để ước tính độ dài CV)
        }
        Danh sách rỗng nếu ZIP không chứa file hợp lệ nào.
    """
    results = []

    # Đọc bytes từ UploadedFile vào bộ nhớ
    zip_bytes = uploaded_zip_file.read()
    zip_buffer = io.BytesIO(zip_bytes)

    try:
        with zipfile.ZipFile(zip_buffer, "r") as zf:
            # Duyệt qua từng file trong ZIP (bỏ qua thư mục)
            for entry in zf.infolist():
                if entry.is_dir():
                    continue

                filename = entry.filename
                ext = os.path.splitext(filename)[1].lower()

                # Chỉ xử lý file text và PDF
                if ext not in (".txt", ".pdf", ".csv"):
                    continue

                try:
                    raw_bytes = zf.read(entry)

                    if ext == ".txt":
                        # Decode UTF-8, fallback sang latin-1 nếu lỗi
                        try:
                            text = raw_bytes.decode("utf-8")
                        except UnicodeDecodeError:
                            text = raw_bytes.decode("latin-1", errors="replace")

                    elif ext == ".pdf":
                        # Trích xuất text từ PDF bằng pdfminer (nếu có)
                        text = _extract_pdf_text(raw_bytes)

                    else:
                        # CSV -- đọc như text thuần
                        text = raw_bytes.decode("utf-8", errors="replace")

                    # Bỏ qua file rỗng
                    if not text.strip():
                        continue

                    results.append({
                        "filename":   os.path.basename(filename),
                        "text":       text.strip(),
                        "char_count": len(text),
                    })

                except Exception as e:
                    print(f"[data.py] Bỏ qua '{filename}': {e}")

    except zipfile.BadZipFile as e:
        raise ValueError(f"File không phải định dạng ZIP hợp lệ: {e}") from e

    return results


def _extract_pdf_text(raw_bytes: bytes) -> str:
    """Trích xuất text từ PDF bytes, sử dụng pdfminer.six nếu có."""
    try:
        from pdfminer.high_level import extract_text as pdfminer_extract
        return pdfminer_extract(io.BytesIO(raw_bytes))
    except ImportError:
        # Fallback: đọc thô, chỉ lấy ký tự printable
        decoded = raw_bytes.decode("utf-8", errors="ignore")
        printable = "".join(c for c in decoded if c.isprintable() or c in "\n\t ")
        return printable


# ---------------------------------------------------------------------------
# Hàm 4: Loader truyền thống (dùng cho pipeline và CLI)
# ---------------------------------------------------------------------------

def load_faircv(csv_path: str) -> tuple[pd.DataFrame, Dataset]:
    """Nạp FairCVdb.csv và trả về (raw_df, Dataset).

    Đây là hàm core được dùng bởi `train_models.py`, `pipeline.py` và
    CLI -- không phụ thuộc vào Streamlit.

    Parameters
    ----------
    csv_path : Đường dẫn đến FairCVdb.csv (24 000 dòng).

    Returns
    -------
    df : pd.DataFrame
        DataFrame đầy đủ với các cột bổ sung: y_blind, y_gender,
        y_ethnicity, split, bio_anonymized.
    ds : Dataset
        Container sẵn sàng dùng cho pipeline, dùng Setting A và
        blind label.
    """
    df = pd.read_csv(csv_path)

    # -- Đổi tên cột nếu CSV dùng tên snake_case (phiên bản cũ) ----------
    _RENAME = {
        "suitability":     "Suitability",
        "lang_prof_1":     "Language 1",
        "lang_prof_2":     "Language 2",
        "lang_prof_3":     "Language 3",
        "prev_experience": "Experience",
        "educ_attainment": "Education",
        "recommendation":  "Recommendation",
        "availability":    "Availability",
    }
    if not all(c in df.columns for c in COMPETENCY):
        df = df.rename(columns=_RENAME)

    # -- Phân chia train/test (19 200 train / 4 800 test) -----------------
    if "split" not in df.columns or df["split"].isna().all():
        df["split"] = "train"
        df.loc[19200:, "split"] = "test"

    # -- Binarize nhãn tại median (cân bằng lớp ~50/50) -------------------
    g_col = ("gender_label"    if "gender_label"    in df.columns
             else "biased_label_gender")
    e_col = ("ethnicity_label" if "ethnicity_label" in df.columns
             else "biased_label_ethnicity")

    threshold        = df["blind_label"].median()
    df["y_blind"]    = (df["blind_label"] >= threshold).astype(int)
    df["y_gender"]   = (df[g_col] >= df[g_col].median()).astype(int)
    df["y_ethnicity"]= (df[e_col] >= df[e_col].median()).astype(int)

    # -- Sinh bio ẩn danh nếu chưa có cột này ----------------------------
    if "bio_anonymized" not in df.columns:
        df["bio_anonymized"] = df.apply(_build_bio, axis=1)

    # -- Tạo Dataset container (Setting A, blind label) -------------------
    X  = df[COMPETENCY].copy()
    ds = Dataset(
        X=X,
        y=df["y_blind"].values,
        A_gender=df["gender"].values,
        A_ethnicity=df["ethnicity"].values,
        feature_names=COMPETENCY,
        meta={
            "threshold": threshold,
            "n_train":   int((df["split"] == "train").sum()),
            "n_test":    int((df["split"] == "test").sum()),
            "df":        df,
        },
    )
    return df, ds


# ---------------------------------------------------------------------------
# Helper nội bộ
# ---------------------------------------------------------------------------

def _build_bio(row: pd.Series) -> str:
    """Xây dựng tiểu sử văn bản ẩn danh từ điểm số đặc trưng năng lực.

    Được dùng khi FairCVdb không có cột bio thô.  Văn bản kết quả
    trung tính về giới tính, chỉ chứa thông tin chuyên môn.
    """
    return (
        f"Candidate profile. Suitability: {row.get('Suitability', 0):.2f}. "
        f"Experience: {row.get('Experience', 0):.2f}. "
        f"Education: {row.get('Education', 0):.2f}. "
        f"Language (primary): {row.get('Language 1', 0):.2f}. "
        f"Availability: {row.get('Availability', 0):.2f}. "
        f"Recommendation: {row.get('Recommendation', 0):.2f}."
    )