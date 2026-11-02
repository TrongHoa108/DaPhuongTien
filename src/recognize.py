from __future__ import annotations

from pathlib import Path

import numpy as np

from src.audio import load_mono, preprocess, split_word_segments
from src.config import (
    DTW_REJECT_THRESHOLD, DTW_SHORTLIST, MODEL_PATH, REJECT_RATIO, SAMPLE_RATE,
)
from src.dtw import cmvn, load_templates, sequence_from_audio
from src.features import extract_vector, vector_to_array
from src.vni import code_to_text

_MODEL_CACHE: dict[str, dict] = {}


def load_model():
    import joblib

    path = MODEL_PATH
    key = str(path)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    if not path.exists():
        raise FileNotFoundError(f"Chưa có model {path}. Chạy: python scripts/train.py")
    model = joblib.load(path)
    _MODEL_CACHE[key] = model
    return model


def predict_array(y, model: dict) -> dict:
    y_pre = preprocess(y)
    feats = extract_vector(y_pre)
    x = vector_to_array(feats).reshape(1, -1)
    pipe = model["pipeline"]
    le = model["label_encoder"]
    scaler = pipe.named_steps["scaler"]
    knn = pipe.named_steps["clf"]
    xs = scaler.transform(x)

    templates = load_templates()
    n_query = max(5, DTW_SHORTLIST) if templates is not None else 5
    n_query = min(n_query, knn.n_samples_fit_)
    dist, idx = knn.kneighbors(xs, n_neighbors=n_query)

    pred = pipe.predict(x)[0]
    code = str(le.inverse_transform([pred])[0])
    mean_d = float(np.mean(dist[0][:5]))
    # Ngưỡng ưu tiên model đã hiệu chỉnh trên validation; nếu không có thì dùng
    # công thức cũ (mean khoảng cách train × hệ số).
    if "reject_threshold" in model:
        thresh = float(model["reject_threshold"])
    else:
        thresh = float(model.get("train_distances_mean", 1.0)) * REJECT_RATIO
    d0 = float(dist[0][0])
    d1 = float(dist[0][1]) if dist.shape[1] > 1 else d0
    confidence = float(d1 / (d0 + d1 + 1e-9))

    y_train = np.asarray(knn._y)
    neighbors = []
    for d, i in zip(dist[0][:5], idx[0][:5]):
        cls = str(le.inverse_transform([int(y_train[i])])[0])
        neighbors.append({"distance": round(float(d), 4), "word_code": cls, "word_text": code_to_text(cls)})

    # So lại 100 ứng viên gần nhất bằng DTW trên chuỗi MFCC + cao độ
    dtw_distance = None
    if templates is not None:
        candidate_codes = le.inverse_transform(y_train[idx[0][:n_query]]).astype(str)
        query = cmvn(sequence_from_audio(y_pre))
        best_code, best_distance, ranked = templates.rerank(query, candidate_codes)
        if best_code is not None:
            code = best_code
            dtw_distance = round(float(best_distance), 4)
            seen: set[str] = set()
            ranked_neighbors = []
            for d, c in ranked:
                if c in seen:
                    continue
                seen.add(c)
                ranked_neighbors.append({
                    "distance": round(float(d), 4), "word_code": c, "word_text": code_to_text(c),
                })
                if len(ranked_neighbors) >= 5:
                    break
            neighbors = ranked_neighbors
            if len(ranked) >= 2:
                a, b = ranked[0][0], ranked[1][0]
                confidence = float(b / (a + b + 1e-9))

    # Ngưỡng từ chối phải dựa trên chính độ đo dùng để quyết định:
    # - có DTW: so khoảng cách DTW với ngưỡng hiệu chỉnh trên take 03 (p98 ≈ 1,97)
    # - không có DTW: công thức cũ trên khoảng cách KNN
    if templates is not None and dtw_distance is not None:
        rejected = float(dtw_distance) > DTW_REJECT_THRESHOLD
    else:
        rejected = mean_d > thresh and mean_d > 0.35

    return {
        "word_code": None if rejected else code,
        "word_text": None if rejected else code_to_text(code),
        "rejected": rejected,
        "confidence": round(0.0 if rejected else min(1.0, confidence), 3),
        "mean_distance": round(mean_d, 4),
        "dtw_distance": dtw_distance,
        "neighbors": neighbors,
        "features": {k: round(float(feats[k]), 4) for k in ("energy_mean", "zcr_mean", "pitch_mean")},
    }


def predict_path(path: Path, model: dict | None = None) -> dict:
    model = model or load_model()
    out = predict_utterance(load_mono(path), model)
    out["file"] = path.name
    return out


def _co_tieng_noi(y: np.ndarray, sr: int) -> bool:
    """Kiểm tra bản ghi có tiếng nói thật, tránh nhận bừa trên khoảng lặng/nhiễu nền."""
    if y.size < int(0.1 * sr):
        return False
    rms = float(np.sqrt(np.mean(y.astype(np.float64) ** 2)))
    if rms < 0.003:
        return False
    frame = max(1, int(0.02 * sr))
    n = y.size // frame
    if n == 0:
        return False
    frames = y[: n * frame].reshape(n, frame).astype(np.float64)
    return float(np.sqrt(np.mean(frames ** 2, axis=1)).max()) >= 0.01


def predict_utterance(
    y, model: dict | None = None, sr: int = SAMPLE_RATE, allow_multi: bool = False,
) -> dict:
    """Nhận dạng đoạn nói: cắt theo khoảng lặng → KNN+DTW từng từ → ghép câu.

    Mặc định (allow_multi=False) là bài toán ĐƠN TỪ: file nhiều từ bị từ chối.
    allow_multi=True là chế độ "Câu": nhận dạng từng mảnh rồi nối — dùng cho
    mở rộng demo (kết quả từng mảnh bị giới hạn bởi từ vựng và sai số đơn từ).
    """
    model = model or load_model()
    y = np.asarray(y, dtype=np.float32)
    if not _co_tieng_noi(y, sr):
        return {
            "text": "", "word_text": None, "word_code": None, "rejected": True,
            "confidence": 0.0, "mean_distance": None, "n_segments": 0, "n_words": 0,
            "words": [], "neighbors": [], "features": {},
            "mode": "sentence" if allow_multi else "utterance",
            "canh_bao": None, "ly_do": "khong_co_tieng_noi",
        }
    segments = split_word_segments(y, sr=sr)
    if not allow_multi:
        # Bài toán là NHẬN DẠNG ĐƠN TỪ. File dài hoặc có nhiều khoảng lặng là
        # câu/nhiều từ → ngoài phạm vi; từ chối rõ ràng thay vì đoán bừa từng mảnh.
        tong_giay = float(sum(s.size for s in segments)) / sr
        if len(segments) > 1 or tong_giay > 1.5:
            return {
                "text": "", "word_text": None, "word_code": None, "rejected": True,
                "confidence": 0.0, "mean_distance": None, "n_segments": len(segments),
                "n_words": 0, "words": [], "neighbors": [], "features": {},
                "mode": "utterance", "canh_bao": "ban_ghi_nhieu_tu",
                "ly_do": "ban_ghi_nhieu_tu",
            }
    segments = segments[:40]
    words: list[dict] = []
    texts: list[str] = []
    for i, seg in enumerate(segments, 1):
        one = predict_array(seg, model)
        item = {
            "index": i,
            "word_text": one.get("word_text"),
            "word_code": one.get("word_code"),
            "rejected": bool(one.get("rejected")),
            "confidence": one.get("confidence"),
            "mean_distance": one.get("mean_distance"),
            "duration_s": round(float(seg.size) / sr, 3),
        }
        words.append(item)
        if not item["rejected"] and item["word_text"]:
            texts.append(str(item["word_text"]))

    text = " ".join(texts).strip()
    rejected = len(texts) == 0
    n_bo_qua = sum(1 for w in words if w["rejected"])
    canh_bao = f"{n_bo_qua} mảnh không khớp từ vựng" if allow_multi and n_bo_qua else None
    confs = [w["confidence"] for w in words if not w["rejected"] and w.get("confidence") is not None]
    mean_conf = float(np.mean(confs)) if confs else 0.0

    # tương thích UI cũ (một từ) + trường mới cho đoạn
    first_ok = next((w for w in words if not w["rejected"]), None)
    return {
        "text": text,
        "word_text": text or None,
        "word_code": first_ok["word_code"] if first_ok and len(texts) == 1 else None,
        "rejected": rejected,
        "confidence": round(mean_conf, 3),
        "mean_distance": words[0]["mean_distance"] if words else None,
        "n_segments": len(segments),
        "n_words": len(texts),
        "n_rejected": n_bo_qua,
        "words": words,
        "neighbors": [],
        "features": {},
        "mode": "sentence" if allow_multi else "utterance",
        "canh_bao": canh_bao,
    }
