"""Quét data/wav, trích đặc trưng, ghi data/features.csv.

Augmentation (chỉ áp cho mẫu HUẤN LUYỆN, không đụng take 03 là tập kiểm tra):
  --augment-pitch   "3,-3"    dịch cao độ ±3 nửa cung  → chịu được giọng khác
  --augment-noise   "15,8"    thêm nhiễu trắng SNR 15/8 dB → chịu được phòng ồn
  --augment-lowpass "4000"    lọc thông thấp 4 kHz      → chịu được micro kém
Số liệu so sánh trước/sau augmentation nằm ở BAO_CAO.md mục 7.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import librosa
import numpy as np
import pandas as pd

from src.audio import load_mono, preprocess
from src.config import FEATURES_PATH, SAMPLE_RATE, WAV_DIR, configure_console
from src.dataset import wav_records
from src.features import extract_vector

configure_console()


def _add_noise(y: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    """Thêm nhiễu trắng theo tỉ lệ tín hiệu/nhiễu SNR (dB)."""
    rms = float(np.sqrt(np.mean(y ** 2))) + 1e-9
    sigma = rms / (10 ** (snr_db / 20))
    rng = np.random.default_rng(seed)
    return (y + rng.normal(0.0, sigma, y.size)).astype(np.float32)


def _lowpass(y: np.ndarray, cut_hz: float) -> np.ndarray:
    """Lọc thông thấp trong miền tần số (mô phỏng micro kém / đường truyền)."""
    spec = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(y.size, 1 / SAMPLE_RATE)
    spec[freqs > cut_hz] = 0.0
    return np.fft.irfft(spec, n=y.size).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--augment-pitch", default="",
                    help='danh sách nửa cung, ví dụ "3,-3"; để trống = không augmentation')
    ap.add_argument("--augment-noise", default="",
                    help='danh sách SNR dB, ví dụ "15,8"')
    ap.add_argument("--augment-lowpass", default="",
                    help='danh sách tần số cắt Hz, ví dụ "4000"')
    args = ap.parse_args()
    shifts = [float(x) for x in args.augment_pitch.split(",") if x.strip()]
    noise_snrs = [float(x) for x in args.augment_noise.split(",") if x.strip()]
    lowpass_cuts = [float(x) for x in args.augment_lowpass.split(",") if x.strip()]

    rows_meta = wav_records()
    if not rows_meta:
        print(f"Không thấy file .wav trong {WAV_DIR}")
        print("Đặt file theo tên: {nguoi}_{ma_tu}_{lan}.wav  (ví dụ hoa_bac1_02.wav)")
        sys.exit(1)

    rows = []
    for i, meta in enumerate(rows_meta, 1):
        path = Path(meta["path"])
        print(f"[{i}/{len(rows_meta)}] {path.name}")
        raw = load_mono(path)
        feats = extract_vector(preprocess(raw))
        rows.append({**meta, **feats, "file": path.name, "augment": ""})
        if meta["take"] != "03":
            for steps in shifts:
                y_aug = librosa.effects.pitch_shift(raw, sr=SAMPLE_RATE, n_steps=steps)
                feats_aug = extract_vector(preprocess(y_aug))
                rows.append({
                    **meta, **feats_aug,
                    "file": f"{path.stem}@p{steps:+g}.wav",
                    "augment": f"pitch{steps:+g}",
                })
            for snr in noise_snrs:
                y_aug = _add_noise(raw, snr, seed=i)
                feats_aug = extract_vector(preprocess(y_aug))
                rows.append({
                    **meta, **feats_aug,
                    "file": f"{path.stem}@n{snr:g}.wav",
                    "augment": f"noise{snr:g}",
                })
            for cut in lowpass_cuts:
                y_aug = _lowpass(raw, cut)
                feats_aug = extract_vector(preprocess(y_aug))
                rows.append({
                    **meta, **feats_aug,
                    "file": f"{path.stem}@lp{cut:g}.wav",
                    "augment": f"lowpass{cut:g}",
                })

    df = pd.DataFrame(rows)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURES_PATH, index=False, encoding="utf-8-sig")
    print(f"Đã ghi {len(df)} dòng → {FEATURES_PATH}")
    print("Từ:", ", ".join(sorted(df["word_text"].unique())))
    print("Người:", ", ".join(sorted(df["speaker"].unique())))


if __name__ == "__main__":
    main()
