"""Nhập bản ghi thu qua mic web có gán nhãn vào data/wav để train cùng kênh.

Mục đích: bản ghi mic thực tế lệch kênh so với data train (âm lượng yếu hơn,
nhiều khung lặng, dải tần hẹp hơn) → nhận dạng đơn từ qua mic sai nhiều.
Cách sửa: thu lại một loạt từ bằng chính đường mic web, đặt tên file dạng

    {nguoi}_{ma}_05.wav        ví dụ: caotuankiet_bac1_05.wav

(thu một từ mỗi file, tên "ma" là mã VNI trong data/wordlist.csv) rồi chạy:

    python scripts\\nhap_giong_mic.py --src <thu_muc> --take 05

Script kiểm tra từng file: đúng một đoạn nói, dài ≤ 1,5 s, có tiếng thực,
mã từ có trong wordlist. File đạt yêu cầu được copy vào data/wav. Sau đó:

    python scripts\\extract_features.py --augment-pitch "3,-3" --augment-noise "15,8" --augment-lowpass "4000"
    python scripts\\train.py
    python scripts\\build_dtw_templates.py --augment-pitch "3,-3" --augment-noise "15,8" --augment-lowpass "4000"

LƯU Ý: đừng dùng tên speaker MỚI (ví dụ mic_xxx) — train.py sẽ đổi cách tách
tập test khi thấy ≥2 người (group-by-speaker) và đánh giá sẽ mất so sánh với
take 03. Dùng tên speaker cũ + take mới là đúng "cùng người, kênh khác".
"""

from __future__ import annotations

import argparse
import re
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.config import SAMPLE_RATE, WAV_DIR, WORDLIST_PATH, configure_console
from src.dataset import load_wordlist

configure_console()

NAME_RE = re.compile(r"^(?P<spk>[A-Za-z0-9]+)_(?P<code>.+)_(?P<take>\d{2})\.wav$")
MAX_SECONDS = 1.5
MIN_SEGMENT_SECONDS = 0.12


def file_hop_le(path: Path) -> tuple[bool, str]:
    """(đạt, lí do loại). Đạt = đúng một đoạn nói ngắn và hợp lệ."""
    import librosa
    import soundfile as sf

    try:
        info = sf.info(str(path))
    except Exception as exc:  # noqa: BLE001
        return False, f"đọc được file: {exc}"
    if info.samplerate < 8000:
        return False, f"tần số mẫu quá thấp ({info.samplerate} Hz)"
    if info.frames / info.samplerate > 4.0:
        return False, f"file quá dài ({info.frames / info.samplerate:.2f} s)"

    y, sr = librosa.load(str(path), sr=SAMPLE_RATE, mono=True)
    intervals = librosa.effects.split(y, top_db=28, frame_length=1024, hop_length=160)
    segs = []
    for s, e in intervals:
        dur = (e - s) / sr
        if dur >= MIN_SEGMENT_SECONDS:
            segs.append((s, e))
    if len(segs) == 0:
        return False, "không có tiếng nói thực (im lặng hoặc chỉ nhiễu)"
    if len(segs) > 1:
        return False, f"có {len(segs)} đoạn nói — mỗi file chỉ được một từ"
    dur = (segs[0][1] - segs[0][0]) / sr
    if dur > MAX_SECONDS:
        return False, f"đoạn nói quá dài ({dur:.2f} s > 1,5 s)"
    s, e = segs[0]
    rms = float(np.sqrt(np.mean(y[s:e].astype(np.float64) ** 2)))
    if rms < 0.005:
        return False, f"tín hiệu quá yếu (RMS {rms:.4f})"
    return True, f"OK ({dur:.2f} s, RMS {rms:.3f})"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", type=Path, required=True, help="thư mục chứa bản ghi mic đặt tên {nguoi}_{ma}_{take}.wav")
    ap.add_argument("--dest", type=Path, default=WAV_DIR)
    args = ap.parse_args()

    if not args.src.exists():
        raise SystemExit(f"Không thấy thư mục nguồn: {args.src}")
    mapping = load_wordlist()
    if not mapping:
        raise SystemExit("Không đọc được wordlist — đặt tên file theo mã trong data/wordlist.csv.")

    args.dest.mkdir(parents=True, exist_ok=True)
    ok, bad_ten, bad_noi_dung, ghi_de = 0, 0, 0, 0
    for path in sorted(args.src.rglob("*.wav")):
        m = NAME_RE.match(path.name)
        if not m:
            bad_ten += 1
            continue
        code = m.group("code")
        if code not in mapping:
            bad_ten += 1
            print(f"  bỏ: {path.name} — mã '{code}' không có trong wordlist")
            continue
        dat, li_do = file_hop_le(path)
        if not dat:
            bad_noi_dung += 1
            print(f"  bỏ: {path.name} — {li_do}")
            continue
        dest = args.dest / path.name
        if dest.exists():
            if dest.stat().st_size >= path.stat().st_size:
                continue
            ghi_de += 1
        shutil.copy2(path, dest)
        ok += 1
        print(f"  nhận: {path.name} — {li_do}")

    print(f"\nNguồn   : {args.src}")
    print(f"Đích    : {args.dest}")
    print(f"Nhận    : {ok}")
    print(f"Sai tên : {bad_ten}")
    print(f"Nội dung: {bad_noi_dung}")
    if ok:
        print("\nBước tiếp:")
        print(r'  python scripts\extract_features.py --augment-pitch "3,-3" --augment-noise "15,8" --augment-lowpass "4000"')
        print(r"  python scripts\train.py")
        print(r'  python scripts\build_dtw_templates.py --augment-pitch "3,-3" --augment-noise "15,8" --augment-lowpass "4000"')
        print(r"  python scripts\evaluate_voices.py")


if __name__ == "__main__":
    main()