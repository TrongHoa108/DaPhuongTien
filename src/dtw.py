"""So khớp chuỗi đặc trưng bằng DTW.

Mỗi bản ghi được biểu diễn bằng chuỗi khung: 13 MFCC + cao độ (nửa cung, so với
trung vị của chính từ). Hai lần đọc cùng một từ dài ngắn khác nhau → DTW căn cho
thẳng thời gian rồi mới đo khoảng cách. Đây vẫn chỉ là một phép ĐO, không có
trọng số nào được học — phân loại vẫn là KNN.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from src.config import (
    DTW_BAND_RADIUS, DTW_PITCH_FRAME, DTW_TEMPLATES_PATH, HOP_LENGTH, N_FFT, N_MFCC,
    PITCH_FMAX, PITCH_FMIN, SAMPLE_RATE,
)


def sequence_from_audio(y: np.ndarray) -> np.ndarray:
    """Chuỗi (số khung, 14): 13 MFCC + cao độ quy về nửa cung so với trung vị."""
    import librosa

    mfcc = librosa.feature.mfcc(
        y=y, sr=SAMPLE_RATE, n_mfcc=N_MFCC, n_fft=N_FFT, hop_length=HOP_LENGTH,
    )
    f0 = librosa.yin(
        y=y, fmin=PITCH_FMIN, fmax=PITCH_FMAX, sr=SAMPLE_RATE,
        frame_length=DTW_PITCH_FRAME, hop_length=HOP_LENGTH,
    )
    n = int(min(mfcc.shape[1], f0.size))
    if n < 3:
        return np.zeros((3, N_MFCC + 1), dtype=np.float32)
    f0 = f0[:n].astype(np.float32)
    f0[~np.isfinite(f0) | (f0 <= 0)] = np.nan
    med = float(np.nanmedian(f0)) if not np.isnan(f0).all() else 120.0
    f0 = np.where(np.isnan(f0), med, f0)
    pitch = 12.0 * np.log2(f0 / med)
    return np.vstack([mfcc[:, :n], pitch[None, :]]).T.astype(np.float32)


def cmvn(seq: np.ndarray) -> np.ndarray:
    """Chuẩn hoá từng hệ số theo chính câu nói — giảm khác biệt micro/giọng."""
    return ((seq - seq.mean(axis=0, keepdims=True))
            / (seq.std(axis=0, keepdims=True) + 1e-6)).astype(np.float32)


def dtw_distance(a: np.ndarray, b: np.ndarray, band_rad: float = DTW_BAND_RADIUS) -> float:
    """Khoảng cách DTW đã chuẩn hoá theo tổng độ dài hai chuỗi."""
    import librosa

    cost = cdist(a, b, metric="euclidean")
    D = librosa.sequence.dtw(
        C=cost, backtrack=False, global_constraints=True, band_rad=band_rad,
    )
    return float(D[-1, -1]) / float(len(a) + len(b))


class DTWTemplates:
    """Bộ tham chiếu chuỗi đã tiền tính, tra cứu theo mã từ."""

    def __init__(self, codes: list[str], by_code: dict[str, list[np.ndarray]]):
        self.codes = codes
        self.by_code = by_code

    def __len__(self) -> int:
        return sum(len(v) for v in self.by_code.values())

    def rerank(self, query: np.ndarray,
               candidate_codes) -> tuple[str | None, float, list[tuple[float, str]]]:
        """So DTW với các từ ứng viên, trả (mã tốt nhất, khoảng cách, danh sách xếp hạng)."""
        dists: list[tuple[float, str]] = []
        for code in dict.fromkeys(candidate_codes):
            for tpl in self.by_code.get(code, []):
                dists.append((dtw_distance(query, tpl), code))
        if not dists:
            return None, float("inf"), []
        dists.sort(key=lambda x: x[0])
        return dists[0][1], dists[0][0], dists

    @classmethod
    def load(cls, path: Path = DTW_TEMPLATES_PATH) -> "DTWTemplates | None":
        if not path.exists():
            return None
        data = np.load(path, allow_pickle=False)
        codes = [str(c) for c in data["codes"]]
        offsets = data["offsets"]
        flat = data["flat"]
        by_code: dict[str, list[np.ndarray]] = defaultdict(list)
        for i, code in enumerate(codes):
            seq = flat[offsets[i]:offsets[i + 1]].astype(np.float32)
            by_code[code].append(cmvn(seq))
        # Chạy nóng numba để lần nhận dạng đầu tiên không bị chậm
        import librosa

        librosa.sequence.dtw(C=np.zeros((3, 3), dtype=np.float32), backtrack=False)
        return cls(codes, dict(by_code))


_TEMPLATES: DTWTemplates | None | bool = False


def load_templates() -> DTWTemplates | None:
    global _TEMPLATES
    if _TEMPLATES is False:
        _TEMPLATES = DTWTemplates.load()
    return _TEMPLATES  # type: ignore[return-value]
