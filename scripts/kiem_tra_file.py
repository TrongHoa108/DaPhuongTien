"""Kiểm tra một file âm thanh: vì sao hệ thống nhận ra từ đó.

Chạy:
    python scripts/kiem_tra_file.py duong_dan\file.wav
    python scripts/kiem_tra_file.py file.wav --dung bac1

In ra: thông tin file, số đoạn cắt được, từ nhận dạng, 5 ứng viên gần nhất
kèm khoảng cách — đủ để biết sai vì từ không có trong model, vì nhiễu,
vì cắt đoạn, hay vì các từ gần âm.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf

from src.audio import load_mono, preprocess, split_word_segments
from src.config import SAMPLE_RATE, configure_console
from src.features import extract_vector
from src.recognize import load_model, predict_array

configure_console()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("wav")
    ap.add_argument("--dung", default="", help="mã từ đúng (nếu biết), ví dụ bac1")
    args = ap.parse_args()

    path = Path(args.wav)
    if not path.exists():
        raise SystemExit(f"Không thấy file: {path}")

    info = sf.info(str(path))
    print(f"File      : {path.name}")
    print(f"Tần số    : {info.samplerate} Hz, {info.channels} kênh, "
          f"{info.frames / info.samplerate:.2f} s")
    if info.samplerate != SAMPLE_RATE:
        print(f"  (sẽ resample về {SAMPLE_RATE} Hz)")

    model = load_model()
    classes = set(map(str, model["label_encoder"].classes_))
    if args.dung and args.dung not in classes:
        print(f"CẢNH BÁO: mã '{args.dung}' KHÔNG có trong model ({len(classes)} từ).")

    y = load_mono(path)
    segs = split_word_segments(y)
    print(f"Đoạn cắt  : {len(segs)} "
          f"({', '.join(f'{s.size / SAMPLE_RATE:.2f}s' for s in segs)})")

    for i, seg in enumerate(segs, 1):
        feats = extract_vector(preprocess(seg))
        r = predict_array(np.asarray(seg, dtype=np.float32), model)
        print(f"\nĐoạn {i}:")
        for nb in r["neighbors"]:
            tag = "" if not r.get("dtw_distance") else " (DTW)"
            print(f"  {nb['distance']:8.3f}  {nb['word_code']:12} {nb['word_text']}{tag}")
        print(f"  khoảng cách KNN TB: {r['mean_distance']:.3f}"
              + (f"  |  DTW: {r['dtw_distance']:.3f}" if r.get("dtw_distance") else ""))
        if r["rejected"]:
            print("  => TỪ CHỐI (quá xa dữ liệu train hoặc không có tiếng nói)")
        print(f"  năng lượng TB={feats['energy_mean']:.4f}  "
              f"ZCR={feats['zcr_mean']:.4f}  pitch={feats['pitch_mean']:.1f} Hz")


if __name__ == "__main__":
    main()
