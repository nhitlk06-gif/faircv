"""
models/train_models.py
-----------------------
Script CLI huấn luyện và lưu các mô hình FairCV.

Luồng thực thi
--------------
1. Gọi `load_faircv_dataset()` từ `src.faircv.data` để tự động kéo
   FairCVdb.csv từ Google Drive về thư mục 'data/' nếu chưa có.
2. Bóc tách ma trận 8 đặc trưng năng lực gốc (Setting A, không bias).
3. Fit StandardScaler trên tập train, transform cả train và test.
4. Huấn luyện 2 mô hình tốt nhất từ kết quả thực nghiệm Notebook:
   - Logistic Regression (LR): F1 cao nhất, AUC cao nhất
   - Multi-Layer Perceptron (MLP): DP Gap / EOO Gap thấp nhất (công bằng nhất)
5. Lưu toàn bộ artifact vào 'models/saved/' bằng joblib/pickle.

Artifact xuất ra (thư mục models/saved/)
-----------------------------------------
    lr_model.pkl      sklearn LogisticRegression đã fit
    lr_scaler.pkl     StandardScaler đã fit (dùng cho LR)
    mlp_model.pkl     sklearn MLPClassifier đã fit
    mlp_scaler.pkl    StandardScaler đã fit (dùng cho MLP)
    config.pkl        dict cấu hình: features, metrics, demo_mode, ...

Chạy
----
    python models/train_models.py
    python models/train_models.py --csv data/FairCVdb.csv --out models/saved
    python models/train_models.py --seed 0
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    GridSearchCV, RandomizedSearchCV, StratifiedKFold,
)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# -- Thêm thư mục src vào sys.path để import package faircv ---------------
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_ROOT, "src"))

from faircv.data import (
    load_faircv_dataset,     # Tải từ Drive rồi nạp vào DataFrame
    load_faircv,             # Loader thô (khi CSV đã có sẵn)
    COMPETENCY, COMP_NAMES,
    GENDER_LABELS, ETH_LABELS,
    GOOGLE_DRIVE_FILE_ID, LOCAL_CSV_PATH,
)
from faircv.metrics import compute_performance, compute_group_metrics

# Seed toàn cục để tái tạo kết quả
RANDOM_SEED = 42


def _log(msg: str) -> None:
    """In log có tiền tố [train_models] ra stdout."""
    print(f"[train_models] {msg}")


# ---------------------------------------------------------------------------
# Hàm tính fairness nhanh (không phụ thuộc pipeline đầy đủ)
# ---------------------------------------------------------------------------

def _fairness_gap(y_true: np.ndarray, y_pred: np.ndarray,
                  group: np.ndarray) -> dict:
    """Tính DP Gap và EOO Gap cho một thuộc tính nhóm nhị phân.

    Parameters
    ----------
    y_true : Nhãn thực (0/1).
    y_pred : Dự đoán mô hình (0/1).
    group  : Mảng nhóm nhị phân (0/1).

    Returns
    -------
    dict với keys: DP_Gap, EOO_Gap
    """
    rates, tprs = {}, {}
    for g in [0, 1]:
        mask = group == g
        if not mask.any():
            continue
        rates[g] = float(y_pred[mask].mean())
        pos_mask  = mask & (y_true == 1)
        tprs[g]   = float(y_pred[pos_mask].mean()) if pos_mask.any() else 0.0

    dp_gap  = abs(rates.get(1, 0) - rates.get(0, 0))
    eoo_gap = abs(tprs.get(1, 0)  - tprs.get(0, 0))
    return {"DP_Gap": round(dp_gap, 4), "EOO_Gap": round(eoo_gap, 4)}


# ---------------------------------------------------------------------------
# Quy trình huấn luyện chính
# ---------------------------------------------------------------------------

def train_and_save(
    csv_path: str = LOCAL_CSV_PATH,
    out_dir:  str = "models/saved",
    seed:     int = RANDOM_SEED,
) -> dict:
    """Huấn luyện LR + MLP trên FairCVdb và lưu artifact.

    Tự động tải FairCVdb.csv từ Google Drive nếu chưa có ở `csv_path`.
    Nếu tải thất bại, dùng dữ liệu mô phỏng nhỏ (demo mode).

    Parameters
    ----------
    csv_path : Đường dẫn cục bộ đến FairCVdb.csv.
    out_dir  : Thư mục đích lưu các file .pkl.
    seed     : Seed ngẫu nhiên cho tái tạo kết quả.

    Returns
    -------
    dict config đã lưu vào config.pkl.
    """
    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------------
    # Bước 1: Tải và nạp dữ liệu
    # ------------------------------------------------------------------
    _log(f"Kiểm tra dữ liệu tại '{csv_path}'...")
    # load_faircv_dataset tự động gọi download_dataset_if_missing
    df, ds = load_faircv_dataset(csv_path, GOOGLE_DRIVE_FILE_ID)

    demo_mode = not (os.path.exists(csv_path) and
                     os.path.getsize(csv_path) > 500_000)

    if demo_mode:
        _log("CANH BAO: Dang chay o che do demo (du lieu mo phong). "
             "Dat file FairCVdb.csv that tai 'data/' va chay lai de co mo hinh chinh xac.")
    else:
        _log(f"Da nap {len(df):,} dong tu FairCVdb.csv.")

    # ------------------------------------------------------------------
    # Bước 2: Tách tập train / test
    # ------------------------------------------------------------------
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    test_df  = df[df["split"] == "test"].reset_index(drop=True)

    # 8 đặc trưng năng lực (Setting A -- không có gender/ethnicity)
    X_tr_raw = train_df[COMPETENCY].values.astype(float)
    X_te_raw = test_df[COMPETENCY].values.astype(float)
    y_tr     = train_df["y_blind"].values
    y_te     = test_df["y_blind"].values

    # Thuộc tính nhạy cảm (chỉ dùng để tính fairness metrics -- KHÔNG vào mô hình)
    gender_tr    = train_df["gender"].values
    gender_te    = test_df["gender"].values
    ethnicity_te = test_df["ethnicity"].values

    _log(f"Train: {len(y_tr):,} | Test: {len(y_te):,} | "
         f"Positive rate: {y_tr.mean():.2%}")

    # ------------------------------------------------------------------
    # Bước 3: Chuẩn hóa đặc trưng (StandardScaler fit ONLY trên train)
    # ------------------------------------------------------------------
    _log("Fit StandardScaler tren tap train...")
    scaler = StandardScaler()
    X_tr   = scaler.fit_transform(X_tr_raw)   # fit + transform train
    X_te   = scaler.transform(X_te_raw)       # chỉ transform test (tránh data leakage)

    # ------------------------------------------------------------------
    # Bước 4: Huấn luyện Logistic Regression
    # ------------------------------------------------------------------
    _log("Dieu chinh sieu tham so LR (GridSearchCV, C in [0.001..100])...")
    t0 = time.time()
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    lr_grid = GridSearchCV(
        # class_weight='balanced' giúp xử lý mất cân bằng lớp nếu có
        LogisticRegression(max_iter=1000, solver="lbfgs",
                           class_weight="balanced", random_state=seed),
        param_grid={"C": [0.001, 0.01, 0.1, 1, 10, 100]},
        cv=cv5, scoring="f1", n_jobs=-1,
    )
    lr_grid.fit(X_tr, y_tr)
    best_C = lr_grid.best_params_["C"]
    _log(f"  LR best C={best_C}  (CV F1={lr_grid.best_score_:.4f}, "
         f"{time.time()-t0:.1f}s)")

    # Fit lại mô hình tốt nhất trên toàn bộ tập train
    lr_model = LogisticRegression(
        C=best_C, max_iter=1000, solver="lbfgs",
        class_weight="balanced", random_state=seed,
    )
    lr_model.fit(X_tr, y_tr)

    # Đánh giá LR
    lr_prob = lr_model.predict_proba(X_te)[:, 1]
    lr_pred = (lr_prob >= 0.5).astype(int)
    lr_perf = compute_performance(y_te, lr_pred, lr_prob)
    lr_fair_g = _fairness_gap(y_te, lr_pred, gender_te)
    lr_fair_e = _fairness_gap(y_te, lr_pred, ethnicity_te)
    _log(f"  LR  F1={lr_perf['F1']:.4f}  AUC={lr_perf['ROC-AUC']:.4f}  "
         f"DP_Gap(gender)={lr_fair_g['DP_Gap']:.4f}  "
         f"EOO_Gap(gender)={lr_fair_g['EOO_Gap']:.4f}")

    # ------------------------------------------------------------------
    # Bước 5: Huấn luyện MLP
    # ------------------------------------------------------------------
    _log("Dieu chinh sieu tham so MLP (RandomizedSearchCV, 12 ung vien x 3-fold)...")
    t0 = time.time()
    cv3 = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)

    mlp_param_dist = {
        # Cac ket hop kien truc an
        "hidden_layer_sizes": [(64, 32), (128, 64), (64, 32, 16), (32, 16)],
        # He so phat chinh quy L2 (tranh overfit)
        "alpha":              [0.0001, 0.001, 0.01],
        # Toc do hoc ban dau
        "learning_rate_init": [0.001, 0.0005, 0.0001],
    }

    mlp_search = RandomizedSearchCV(
        MLPClassifier(
            max_iter=200, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=10,
            solver="adam", random_state=seed,
        ),
        param_distributions=mlp_param_dist,
        n_iter=12, cv=cv3, scoring="f1",
        random_state=seed, n_jobs=-1,
    )
    mlp_search.fit(X_tr, y_tr)
    best_mlp = mlp_search.best_params_
    _log(f"  MLP best params={best_mlp}  "
         f"(CV F1={mlp_search.best_score_:.4f}, {time.time()-t0:.1f}s)")

    # Fit lại MLP tốt nhất trên toàn bộ tập train
    mlp_model = MLPClassifier(
        **best_mlp,
        max_iter=300, early_stopping=True,
        validation_fraction=0.1, n_iter_no_change=15,
        solver="adam", random_state=seed,
    )
    mlp_model.fit(X_tr, y_tr)

    # Đánh giá MLP
    mlp_prob = mlp_model.predict_proba(X_te)[:, 1]
    mlp_pred = (mlp_prob >= 0.5).astype(int)
    mlp_perf = compute_performance(y_te, mlp_pred, mlp_prob)
    mlp_fair_g = _fairness_gap(y_te, mlp_pred, gender_te)
    mlp_fair_e = _fairness_gap(y_te, mlp_pred, ethnicity_te)
    _log(f"  MLP F1={mlp_perf['F1']:.4f}  AUC={mlp_perf['ROC-AUC']:.4f}  "
         f"DP_Gap(gender)={mlp_fair_g['DP_Gap']:.4f}  "
         f"EOO_Gap(gender)={mlp_fair_g['EOO_Gap']:.4f}")

    # ------------------------------------------------------------------
    # Bước 6: Lưu artifact bằng joblib
    # ------------------------------------------------------------------

    # Lưu Logistic Regression + scaler
    joblib.dump(lr_model, os.path.join(out_dir, "lr_model.pkl"))
    joblib.dump(scaler,   os.path.join(out_dir, "lr_scaler.pkl"))
    _log(f"Luu: {out_dir}/lr_model.pkl + lr_scaler.pkl")

    # Lưu MLP + scaler (dùng cùng scaler vì cùng đặc trưng đầu vào)
    joblib.dump(mlp_model, os.path.join(out_dir, "mlp_model.pkl"))
    joblib.dump(scaler,    os.path.join(out_dir, "mlp_scaler.pkl"))
    _log(f"Luu: {out_dir}/mlp_model.pkl + mlp_scaler.pkl")

    # Lưu file config tổng hợp
    config = {
        # -- Đặc trưng --
        "features":      COMPETENCY,
        "feature_names": COMP_NAMES,

        # -- Chế độ --
        "demo_mode":  demo_mode,
        "random_seed": seed,

        # -- Siêu tham số tốt nhất --
        "lr_best_C":         best_C,
        "mlp_best_params":   best_mlp,

        # -- Metrics LR --
        "lr_metrics": {
            **lr_perf,
            "DP_Gap_Gender":    lr_fair_g["DP_Gap"],
            "EOO_Gap_Gender":   lr_fair_g["EOO_Gap"],
            "DP_Gap_Ethnicity": lr_fair_e["DP_Gap"],
        },

        # -- Metrics MLP --
        "mlp_metrics": {
            **mlp_perf,
            "DP_Gap_Gender":    mlp_fair_g["DP_Gap"],
            "EOO_Gap_Gender":   mlp_fair_g["EOO_Gap"],
            "DP_Gap_Ethnicity": mlp_fair_e["DP_Gap"],
        },
    }
    joblib.dump(config, os.path.join(out_dir, "config.pkl"))
    _log(f"Luu: {out_dir}/config.pkl")

    return config


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Huan luyen va luu mo hinh FairCV (LR + MLP).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--csv", default=LOCAL_CSV_PATH,
        help="Duong dan den FairCVdb.csv. "
             "Neu chua co, script se tu dong tai tu Google Drive.",
    )
    parser.add_argument(
        "--out", default="models/saved",
        help="Thu muc luu cac file .pkl.",
    )
    parser.add_argument(
        "--seed", type=int, default=RANDOM_SEED,
        help="Seed ngau nhien.",
    )
    args = parser.parse_args()

    config = train_and_save(args.csv, args.out, seed=args.seed)

    # In bảng tổng kết
    print("\n" + "=" * 68)
    print("  KET QUA MO HINH CUOI CUNG (Setting A, blind label)")
    print("=" * 68)
    print(f"{'Model':<6}{'F1':>8}{'AUC':>9}{'DP_Gap(G)':>12}{'EOO_Gap(G)':>12}")
    print("-" * 68)
    for name, m in [("LR", config["lr_metrics"]), ("MLP", config["mlp_metrics"])]:
        print(
            f"{name:<6}{m['F1']:>8.4f}{m['ROC-AUC']:>9.4f}"
            f"{m['DP_Gap_Gender']:>12.4f}{m['EOO_Gap_Gender']:>12.4f}"
        )
    print("=" * 68)
    print(f"\nChe do: {'DEMO (mo phong)' if config['demo_mode'] else 'PRODUCTION (FairCVdb that)'}")
    print(f"Artifact luu tai: {args.out}/")


if __name__ == "__main__":
    main()