"""Quét data/wav, trích đặc trưng, ghi data/features.csv."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.audio import load_mono, preprocess
from src.config import FEATURES_PATH, WAV_DIR, configure_console
from src.dataset import wav_records
from src.features import extract_vector

configure_console()


def main() -> None:
    rows_meta = wav_records()
    if not rows_meta:
        print(f"Không thấy file .wav trong {WAV_DIR}")
        print("Đặt file theo tên: {nguoi}_{ma_tu}_{lan}.wav  (ví dụ hoa_bac1_02.wav)")
        sys.exit(1)

    rows = []
    for i, meta in enumerate(rows_meta, 1):
        path = Path(meta["path"])
        print(f"[{i}/{len(rows_meta)}] {path.name}")
        y = preprocess(load_mono(path))
        feats = extract_vector(y)
        rows.append({**meta, **feats, "file": path.name})

    df = pd.DataFrame(rows)
    FEATURES_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURES_PATH, index=False, encoding="utf-8-sig")
    print(f"Đã ghi {len(df)} dòng → {FEATURES_PATH}")
    print("Từ:", ", ".join(sorted(df["word_text"].unique())))
    print("Người:", ", ".join(sorted(df["speaker"].unique())))


if __name__ == "__main__":
    main()
