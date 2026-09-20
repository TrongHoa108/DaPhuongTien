"""Huấn luyện KNN (chính) + so sánh Logistic / SVM. Xuất accuracy và confusion matrix."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from src.config import FEATURE_COLUMNS, FEATURES_PATH, K_NEIGHBORS, MODELS_DIR, REPORTS_DIR, configure_console
from src.dataset import load_features

configure_console()


def _split(df: pd.DataFrame):
    y = df["word_code"].astype(str)
    groups = df["speaker"].astype(str)
    if groups.nunique() >= 2 and len(df) >= 8:
        gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
        train_idx, test_idx = next(gss.split(df, y, groups))
        return df.iloc[train_idx], df.iloc[test_idx], "group-by-speaker"
    train, test = train_test_split(df, test_size=0.3, random_state=42, stratify=y if y.value_counts().min() >= 2 else None)
    return train, test, "random"


def _pipeline(clf) -> Pipeline:
    return Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )


def _eval(name: str, model: Pipeline, X_te, y_te, labels: list[str]) -> dict:
    pred = model.predict(X_te)
    acc = float(accuracy_score(y_te, pred))
    cm = confusion_matrix(y_te, pred, labels=labels)
    return {"name": name, "accuracy": acc, "pred": pred, "cm": cm}


def _plot_cm(cm: np.ndarray, labels: list[str], title: str, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 0.45), max(5, len(labels) * 0.4)))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_title(title)
    ax.set_xlabel("Dự đoán")
    ax.set_ylabel("Thực tế")
    ax.set_xticks(range(len(labels)), labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main() -> None:
    if not FEATURES_PATH.exists():
        print("Chưa có data/features.csv — chạy: python scripts/extract_features.py")
        sys.exit(1)

    df = load_features()
    train, test, split_kind = _split(df)
    X_tr = train[FEATURE_COLUMNS].to_numpy()
    X_te = test[FEATURE_COLUMNS].to_numpy()
    le = LabelEncoder()
    y_tr = le.fit_transform(train["word_code"].astype(str))
    y_te = le.transform(test["word_code"].astype(str))
    labels = list(le.classes_)

    models = {
        "knn": _pipeline(KNeighborsClassifier(n_neighbors=min(K_NEIGHBORS, max(1, len(train) // 2)), weights="distance")),
        "logreg": _pipeline(LogisticRegression(max_iter=400, class_weight="balanced")),
        "svm": _pipeline(SVC(kernel="rbf", class_weight="balanced")),
    }

    results = []
    preds = {}
    for name, pipe in models.items():
        pipe.fit(X_tr, y_tr)
        ev = _eval(name, pipe, X_te, y_te, list(range(len(labels))))
        results.append({"name": name, "accuracy": ev["accuracy"]})
        preds[name] = ev
        print(f"{name:8}  accuracy={ev['accuracy']:.3f}")

    knn = models["knn"]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": knn,
            "label_encoder": le,
            "feature_columns": FEATURE_COLUMNS,
            "train_distances_mean": _mean_nn_distance(knn, X_tr),
        },
        MODELS_DIR / "knn.joblib",
    )

    y_true_code = test["word_code"].astype(str).to_numpy()
    y_pred_code = le.inverse_transform(preds["knn"]["pred"])
    _plot_cm(
        confusion_matrix(y_true_code, y_pred_code, labels=labels),
        labels,
        "Confusion matrix — KNN",
        REPORTS_DIR / "confusion_knn.png",
    )

    report = classification_report(y_true_code, y_pred_code, zero_division=0)
    (REPORTS_DIR / "classification_report.txt").write_text(report, encoding="utf-8")

    known_speakers = set(train["speaker"].unique())
    mask_known = test["speaker"].isin(known_speakers)
    extra = {}
    if mask_known.any() and (~mask_known).any():
        extra["accuracy_known_speaker"] = float(accuracy_score(y_true_code[mask_known], y_pred_code[mask_known]))
        extra["accuracy_new_speaker"] = float(accuracy_score(y_true_code[~mask_known], y_pred_code[~mask_known]))
        print("Giọng quen:", extra["accuracy_known_speaker"])
        print("Giọng mới :", extra["accuracy_new_speaker"])

    mfcc_cols = [c for c in FEATURE_COLUMNS if c.startswith("mfcc_")]
    k = min(K_NEIGHBORS, max(1, len(train) // 2))
    feature_results = []
    for sname, cols in (
        ("mfcc", mfcc_cols),
        ("mfcc_energy_zcr", mfcc_cols + ["energy_mean", "zcr_mean"]),
        ("all", list(FEATURE_COLUMNS)),
    ):
        p = _pipeline(KNeighborsClassifier(n_neighbors=k, weights="distance"))
        p.fit(train[cols].to_numpy(), y_tr)
        acc = float(accuracy_score(y_te, p.predict(test[cols].to_numpy())))
        feature_results.append({"set": sname, "accuracy": acc})
        print(f"features {sname:18} acc={acc:.3f}")

    payload = {
        "split": split_kind,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "n_classes": len(labels),
        "results": results,
        "feature_sets": feature_results,
        **extra,
    }
    (REPORTS_DIR / "metrics.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Đã lưu model →", MODELS_DIR / "knn.joblib")
    print("Báo cáo     →", REPORTS_DIR)


def _mean_nn_distance(pipe: Pipeline, X: np.ndarray) -> float:
    scaler = pipe.named_steps["scaler"]
    knn: KNeighborsClassifier = pipe.named_steps["clf"]
    Xs = scaler.transform(X)
    dist, _ = knn.kneighbors(Xs, n_neighbors=min(2, Xs.shape[0]))
    # khoảng cách tới hàng xóm khác chính nó
    col = 1 if dist.shape[1] > 1 else 0
    return float(np.mean(dist[:, col]))


if __name__ == "__main__":
    main()
