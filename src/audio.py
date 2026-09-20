from __future__ import annotations

import numpy as np
import soundfile as sf
import librosa

from src.config import HOP_LENGTH, SAMPLE_RATE


def load_mono(path, sr: int = SAMPLE_RATE) -> np.ndarray:
    y, file_sr = sf.read(str(path), always_2d=False)
    if y.ndim > 1:
        y = np.mean(y, axis=1)
    y = y.astype(np.float32)
    if file_sr != sr:
        y = librosa.resample(y, orig_sr=file_sr, target_sr=sr)
    return y


def preprocess(y: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    y, _ = librosa.effects.trim(y, top_db=25, frame_length=1024, hop_length=HOP_LENGTH)
    if y.size == 0:
        y = np.zeros(int(0.2 * sr), dtype=np.float32)
    peak = np.max(np.abs(y))
    if peak > 1e-6:
        y = y / peak * 0.9
    # pre-emphasis
    y = np.append(y[0], y[1:] - 0.97 * y[:-1]).astype(np.float32)
    min_len = int(0.12 * sr)
    if y.size < min_len:
        y = np.pad(y, (0, min_len - y.size))
    return y
