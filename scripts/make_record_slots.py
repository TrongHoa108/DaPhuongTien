"""Tạo file .wav im lặng đúng tên để thu đè lên. Không ghi đè file đã thu."""

from __future__ import annotations

import re
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.config import WORDLIST_PATH, configure_console

configure_console()

ROOT = Path(__file__).resolve().parents[1]
OUT_ROOT = ROOT / "data" / "thu_am"
TAKES = ("01", "02", "03")
SAMPLE_RATE = 16000
SILENCE_SAMPLES = 160  # 10 ms — chỉ để file là wav hợp lệ, thu thì ghi đè

_BAD = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def _silent_wav() -> bytes:
    data = b"\x00\x00" * SILENCE_SAMPLES
    header = struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        36 + len(data),
        b"WAVE",
        b"fmt ",
        16,
        1,
        1,
        SAMPLE_RATE,
        SAMPLE_RATE * 2,
        2,
        16,
        b"data",
        len(data),
    )
    return header + data


SILENT = _silent_wav()


def _safe_part(name: str) -> str:
    cleaned = _BAD.sub("_", name).strip().rstrip(". ")
    if not cleaned:
        cleaned = "_"
    return cleaned


def make_for(speaker: str) -> tuple[int, int]:
    speaker = speaker.strip().lower()
    if not re.fullmatch(r"[a-z0-9]+", speaker):
        raise SystemExit(
            f"Tên '{speaker}' không hợp lệ. Dùng chữ thường không dấu, không khoảng trắng (vd: hoa)."
        )

    df = pd.read_csv(WORDLIST_PATH, encoding="utf-8-sig")
    created = 0
    skipped = 0
    base = OUT_ROOT / speaker
    for row in df.itertuples(index=False):
        text = str(row.text)
        code = str(row.code)
        letter = _safe_part(str(row.chu_dau))
        folder = base / letter / f"{int(row.stt):04d}_{_safe_part(text)}"
        folder.mkdir(parents=True, exist_ok=True)
        for take in TAKES:
            path = folder / f"{speaker}_{code}_{take}.wav"
            if path.exists() and path.stat().st_size > len(SILENT):
                skipped += 1
                continue
            path.write_bytes(SILENT)
            created += 1
    return created, skipped


def main() -> None:
    speakers = [a for a in sys.argv[1:] if not a.startswith("-")]
    if not speakers:
        speakers = ["ten"]
    total_new = 0
    total_skip = 0
    for speaker in speakers:
        created, skipped = make_for(speaker)
        total_new += created
        total_skip += skipped
        print(f"{speaker}: tạo {created} file, giữ nguyên {skipped} file đã có nội dung")
        print(f"  → {OUT_ROOT / speaker}")
    print(f"Xong. Tổng file mới: {total_new}. Mỗi từ có 3 file _01 _02 _03.")
    print("Mở thư mục theo chữ, đọc đúng tên thư mục (chữ tiếng Việt), thu đè lên file .wav, không đổi tên.")


if __name__ == "__main__":
    main()
