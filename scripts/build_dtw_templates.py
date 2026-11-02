"""Tạo bộ tham chiếu DTW: chuỗi MFCC + cao độ của mọi bản ghi huấn luyện.

Chạy:
    python scripts/build_dtw_templates.py

Kết quả: data/models/dtw_templates.npz (mã từ + các chuỗi đã nén float16).
Mặc định bỏ take 03 ra khỏi tham chiếu để giữ tập kiểm tra tách bạch.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import librosa
import numpy as np

from src.audio import load_mono, preprocess
from src.config import DTW_TEMPLATES_PATH, SAMPLE_RATE, WAV_DIR, configure_console
from src.dtw import sequence_from_audio

configure_console()
NAME_RE = re.compile(r"^(?P<spk>[A-Za-z0-9]+)_(?P<code>.+)_(?P<take>\d{2})\.wav$")


def _noise(y: np.ndarray, snr_db: float, seed: int) -> np.ndarray:
    rms = float(np.sqrt(np.mean(y ** 2))) + 1e-9
    sigma = rms / (10 ** (snr_db / 20))
    return (y + np.random.default_rng(seed).normal(0.0, sigma, y.size)).astype(np.float32)


def _lowpass(y: np.ndarray, cut_hz: float) -> np.ndarray:
    spec = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(y.size, 1 / SAMPLE_RATE)
    spec[freqs > cut_hz] = 0.0
    return np.fft.irfft(spec, n=y.size).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--thu-muc-wav", type=Path, default=WAV_DIR)
    ap.add_argument("--out", type=Path, default=DTW_TEMPLATES_PATH)
    ap.add_argument("--bo-take", default="03", help="bỏ lần đọc này khỏi tham chiếu (tập test)")
    ap.add_argument("--augment-pitch", default="", help='ví dụ "3,-3"')
    ap.add_argument("--augment-noise", default="", help='ví dụ "15,8"')
    ap.add_argument("--augment-lowpass", default="", help='ví dụ "4000"')
    args = ap.parse_args()
    shifts = [float(x) for x in args.augment_pitch.split(",") if x.strip()]
    snrs = [float(x) for x in args.augment_noise.split(",") if x.strip()]
    cuts = [float(x) for x in args.augment_lowpass.split(",") if x.strip()]

    files = []
    for p in sorted(args.thu_muc_wav.glob("*.wav")):
        m = NAME_RE.match(p.name)
        if m and m.group("take") != args.bo_take:
            files.append((p, m.group("code")))
    if not files:
        raise SystemExit(f"Không thấy file phù hợp trong {args.thu_muc_wav}")

    seqs, codes = [], []
    for i, (p, code) in enumerate(files, 1):
        raw = load_mono(p)
        seqs.append(sequence_from_audio(preprocess(raw)).astype(np.float16))
        codes.append(code)
        for steps in shifts:
            y = librosa.effects.pitch_shift(raw, sr=SAMPLE_RATE, n_steps=steps)
            seqs.append(sequence_from_audio(preprocess(y)).astype(np.float16))
            codes.append(code)
        for snr in snrs:
            seqs.append(sequence_from_audio(preprocess(_noise(raw, snr, i))).astype(np.float16))
            codes.append(code)
        for cut in cuts:
            seqs.append(sequence_from_audio(preprocess(_lowpass(raw, cut))).astype(np.float16))
            codes.append(code)
        if i % 500 == 0:
            print(f"  {i}/{len(files)}", flush=True)

    lengths = np.array([len(s) for s in seqs], dtype=np.int64)
    offsets = np.zeros(len(seqs) + 1, dtype=np.int64)
    np.cumsum(lengths, out=offsets[1:])
    flat = np.vstack(seqs)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        codes=np.array(codes),
        offsets=offsets,
        flat=flat,
    )
    print(f"Đã lưu {len(codes)} chuỗi ({flat.shape[0]} khung, {flat.shape[1]} chiều) → {args.out}")
    print(f"Dung lượng: {args.out.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
