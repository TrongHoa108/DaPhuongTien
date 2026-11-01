from __future__ import annotations

import numpy as np
import librosa

from src.config import FEATURE_COLUMNS, HOP_LENGTH, N_FFT, N_MFCC, SAMPLE_RATE


def _pitch_stats(y: np.ndarray, sr: int) -> tuple[float, float]:
    pitches, mags = librosa.piptrack(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP_LENGTH, fmin=60, fmax=400
    )
    values = []
    for t in range(mags.shape[1]):
        idx = mags[:, t].argmax()
        if mags[idx, t] > 0.1:
            p = pitches[idx, t]
            if p > 0:
                values.append(p)
    if not values:
        return 0.0, 0.0
    arr = np.asarray(values, dtype=np.float32)
    return float(arr.mean()), float(arr.std())


def extract_vector(y: np.ndarray, sr: int = SAMPLE_RATE) -> dict[str, float]:
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH
    )
    # Không CMVN theo từng file trước khi lấy mean — nếu trừ mean theo khung
    # thì mfcc_*_mean ≈ 0 với mọi từ, KNN không phân biệt được.
    # Lệch kênh/giọng được StandardScaler xử lý trên toàn tập train.

    feats: dict[str, float] = {}
    for i in range(N_MFCC):
        feats[f"mfcc_{i + 1}_mean"] = float(mfcc[i].mean())
        feats[f"mfcc_{i + 1}_std"] = float(mfcc[i].std())

    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=N_FFT, hop_length=HOP_LENGTH)[0]
    feats["energy_mean"] = float(rms.mean())
    feats["zcr_mean"] = float(zcr.mean())
    p_mean, p_std = _pitch_stats(y, sr)
    feats["pitch_mean"] = p_mean
    feats["pitch_std"] = p_std
    return {k: feats[k] for k in FEATURE_COLUMNS}


def vector_to_array(feats: dict[str, float]) -> np.ndarray:
    return np.array([feats[c] for c in FEATURE_COLUMNS], dtype=np.float32)
