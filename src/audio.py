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


def split_word_segments(
    y: np.ndarray,
    sr: int = SAMPLE_RATE,
    top_db: float = 28,
    min_dur: float = 0.12,
    max_dur: float = 1.6,
) -> list[np.ndarray]:
    """Cắt đoạn nói theo khoảng lặng → mỗi đoạn ứng với một từ (cần nghỉ ngắn giữa các từ)."""
    y = np.asarray(y, dtype=np.float32)
    if y.size == 0:
        return []
    intervals = librosa.effects.split(y, top_db=top_db, frame_length=1024, hop_length=HOP_LENGTH)
    segs: list[np.ndarray] = []
    for start, end in intervals:
        seg = y[start:end]
        dur = seg.size / sr
        if dur < min_dur:
            continue
        if dur > max_dur:
            # đoạn quá dài: cắt thêm lần nữa với ngưỡng nhạy hơn
            sub = librosa.effects.split(seg, top_db=max(18, top_db - 8), frame_length=1024, hop_length=HOP_LENGTH)
            if len(sub) > 1:
                for s2, e2 in sub:
                    piece = seg[s2:e2]
                    d2 = piece.size / sr
                    if min_dur <= d2 <= max_dur:
                        segs.append(piece.astype(np.float32))
                    elif d2 > max_dur:
                        # cắt cửa sổ cố định ~0.7s chồng nhẹ
                        win = int(0.7 * sr)
                        hop = int(0.45 * sr)
                        for i in range(0, max(1, piece.size - win + 1), hop):
                            chunk = piece[i : i + win]
                            if chunk.size / sr >= min_dur:
                                segs.append(chunk.astype(np.float32))
                continue
        segs.append(seg.astype(np.float32))
    if not segs and y.size / sr >= min_dur:
        segs = [y.astype(np.float32)]
    return segs
