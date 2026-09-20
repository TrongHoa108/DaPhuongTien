from __future__ import annotations

from pathlib import Path

import numpy as np

from src.audio import load_mono, preprocess
from src.config import MODELS_DIR, REJECT_RATIO
from src.features import extract_vector, vector_to_array
from src.vni import code_to_text


def load_model():
    import joblib

    path = MODELS_DIR / "knn.joblib"
    if not path.exists():
        raise FileNotFoundError("Chưa có model. Chạy: python scripts/train.py")
    return joblib.load(path)


def predict_array(y, model: dict) -> dict:
    feats = extract_vector(preprocess(y))
    x = vector_to_array(feats).reshape(1, -1)
    pipe = model["pipeline"]
    le = model["label_encoder"]
    scaler = pipe.named_steps["scaler"]
    knn = pipe.named_steps["clf"]
    xs = scaler.transform(x)
    n_nb = min(5, knn.n_samples_fit_)
    dist, idx = knn.kneighbors(xs, n_neighbors=n_nb)
    pred = pipe.predict(x)[0]
    code = str(le.inverse_transform([pred])[0])
    mean_d = float(np.mean(dist[0]))
    thresh = float(model.get("train_distances_mean", 1.0)) * REJECT_RATIO
    rejected = mean_d > thresh and mean_d > 0.35
    d0 = float(dist[0][0])
    d1 = float(dist[0][min(1, len(dist[0]) - 1)])
    confidence = float(d1 / (d0 + d1 + 1e-9))
    neighbors = []
    y_train = knn._y
    for d, i in zip(dist[0], idx[0]):
        cls = str(le.inverse_transform([int(y_train[i])])[0])
        neighbors.append({"distance": round(float(d), 4), "word_code": cls, "word_text": code_to_text(cls)})
    return {
        "word_code": None if rejected else code,
        "word_text": None if rejected else code_to_text(code),
        "rejected": rejected,
        "confidence": round(0.0 if rejected else min(1.0, confidence), 3),
        "mean_distance": round(mean_d, 4),
        "neighbors": neighbors,
        "features": {k: round(float(feats[k]), 4) for k in ("energy_mean", "zcr_mean", "pitch_mean")},
    }


def predict_path(path: Path, model: dict | None = None) -> dict:
    model = model or load_model()
    out = predict_array(load_mono(path), model)
    out["file"] = path.name
    return out
