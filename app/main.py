from __future__ import annotations

import io
import sys
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import soundfile as sf

from src.audio import load_mono
from src.config import FEATURES_PATH, MODELS_DIR, REPORTS_DIR, SAMPLE_RATE, UPLOADS_DIR
from src.recognize import load_model, predict_utterance

STATIC = Path(__file__).resolve().parent / "static"
app = FastAPI(title="Nhận dạng đơn từ tiếng Việt")
app.mount("/static", StaticFiles(directory=STATIC), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def health():
    return {
        "model": (MODELS_DIR / "knn.joblib").exists(),
        "features": FEATURES_PATH.exists(),
    }


@app.get("/api/metrics")
def metrics():
    path = REPORTS_DIR / "metrics.json"
    if not path.exists():
        raise HTTPException(404, "Chưa huấn luyện. Chạy python scripts/train.py")
    import json

    return json.loads(path.read_text(encoding="utf-8"))


@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    if not (MODELS_DIR / "knn.joblib").exists():
        raise HTTPException(400, "Chưa có model. Chạy extract + train trước.")
    raw = await file.read()
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = UPLOADS_DIR / (file.filename or "upload.wav")
    tmp.write_bytes(raw)
    try:
        y = load_mono(tmp, sr=SAMPLE_RATE)
    except Exception:
        try:
            y, sr = sf.read(io.BytesIO(raw), always_2d=False)
            if y.ndim > 1:
                y = np.mean(y, axis=1)
        except Exception as exc:
            raise HTTPException(400, f"Không đọc được file âm thanh: {exc}") from exc
    result = predict_utterance(y.astype(np.float32), load_model())
    result["filename"] = file.filename
    return result
