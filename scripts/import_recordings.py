"""Gom file thu thật từ data/thu_am_HT (hoặc thư mục khác) vào data/wav.

Bỏ file im lặng (template). Chuẩn hóa tên người thành một speaker nếu cần.
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import WAV_DIR, configure_console

configure_console()

NAME_RE = re.compile(r"^(?P<spk>[A-Za-z0-9]+)_(?P<code>.+)_(?P<take>\d{2})\.wav$")
DEFAULT_SRC = Path(__file__).resolve().parents[1] / "data" / "thu_am_HT"
MIN_BYTES = 1000


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, default=DEFAULT_SRC)
    ap.add_argument("--speaker", default="caotuankiet", help="Tên người ghi vào file đích")
    ap.add_argument("--clear", action="store_true", help="Xóa toàn bộ .wav cũ trong data/wav")
    ap.add_argument("--min-bytes", type=int, default=MIN_BYTES)
    args = ap.parse_args()

    src = args.src
    if not src.exists():
        raise SystemExit(f"Không thấy thư mục nguồn: {src}")

    WAV_DIR.mkdir(parents=True, exist_ok=True)
    if args.clear:
        for old in WAV_DIR.glob("*.wav"):
            old.unlink()
        print(f"Đã xóa wav cũ trong {WAV_DIR}")

    copied = 0
    skipped_tiny = 0
    skipped_name = 0
    overwritten = 0
    for path in sorted(src.rglob("*.wav")):
        if path.stat().st_size < args.min_bytes:
            skipped_tiny += 1
            continue
        m = NAME_RE.match(path.name)
        if not m:
            skipped_name += 1
            continue
        code = m.group("code")
        take = m.group("take")
        dest_name = f"{args.speaker}_{code}_{take}.wav"
        dest = WAV_DIR / dest_name
        if dest.exists():
            # giữ bản lớn hơn nếu trùng
            if dest.stat().st_size >= path.stat().st_size:
                continue
            overwritten += 1
        shutil.copy2(path, dest)
        copied += 1

    print(f"Nguồn      : {src}")
    print(f"Đích       : {WAV_DIR}")
    print(f"Đã copy    : {copied}")
    print(f"Ghi đè     : {overwritten}")
    print(f"Bỏ im lặng : {skipped_tiny}")
    print(f"Bỏ sai tên : {skipped_name}")
    print(f"Tổng wav   : {len(list(WAV_DIR.glob('*.wav')))}")


if __name__ == "__main__":
    main()
