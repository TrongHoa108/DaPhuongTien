from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import FEATURE_COLUMNS, FEATURES_PATH, WAV_DIR, WORDLIST_PATH
from src.vni import parse_wav_name


def load_wordlist() -> dict[str, str]:
    if not WORDLIST_PATH.exists():
        return {}
    # keep_default_na=False: từ "nan" trong danh sách là một từ thật, không phải NaN
    df = pd.read_csv(WORDLIST_PATH, keep_default_na=False)
    return dict(zip(df["code"].astype(str), df["text"].astype(str)))


def list_wavs(wav_dir: Path | None = None) -> list[Path]:
    wav_dir = wav_dir or WAV_DIR
    return sorted(wav_dir.rglob("*.wav"))


def wav_records(wav_dir: Path | None = None) -> list[dict]:
    mapping = load_wordlist()
    rows = []
    for path in list_wavs(wav_dir):
        meta = parse_wav_name(path.name)
        if meta["word_code"] in mapping:
            meta["word_text"] = mapping[meta["word_code"]]
        meta["path"] = str(path)
        rows.append(meta)
    return rows


def load_features(path: Path | None = None) -> pd.DataFrame:
    path = path or FEATURES_PATH
    df = pd.read_csv(path)
    missing = [c for c in FEATURE_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV thiếu cột đặc trưng: {missing}")
    return df
