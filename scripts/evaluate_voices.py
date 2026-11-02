"""Danh gia accuracy theo 2 che do:

1. Giong quen, lan doc moi  : cac file take 03 (nguoi da co trong train).
2. "Giong moi" (MO PHONG)   : cung file do, bien doi cao do / toc do.
   LUU Y: day la mo phong, khong phai nguoi noi that thu hai. Ghi ro trong bao cao.

Chay:
    python scripts/evaluate_voices.py
    python scripts/evaluate_voices.py --so-mau 200
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import librosa
import numpy as np

from src.audio import load_mono
from src.config import REPORTS_DIR, SAMPLE_RATE, WAV_DIR, configure_console
from src.recognize import load_model, predict_array

configure_console()

NAME_RE = re.compile(r"^(?P<spk>[A-Za-z0-9]+)_(?P<code>.+)_(?P<take>\d{2})\.wav$")

def _lowpass(y: np.ndarray, cut_hz: float) -> np.ndarray:
    spec = np.fft.rfft(y)
    freqs = np.fft.rfftfreq(y.size, 1 / SAMPLE_RATE)
    spec[freqs > cut_hz] = 0.0
    return np.fft.irfft(spec, n=y.size).astype(np.float32)


def _add_noise(y: np.ndarray, snr_db: float, seed: int = 7) -> np.ndarray:
    rms = float(np.sqrt(np.mean(y ** 2))) + 1e-9
    sigma = rms / (10 ** (snr_db / 20))
    return (y + np.random.default_rng(seed).normal(0.0, sigma, y.size)).astype(np.float32)


CHUOIEN = {
    "giong_goc": None,
    "cao_do_+4": lambda y: librosa.effects.pitch_shift(y, sr=SAMPLE_RATE, n_steps=4),
    "cao_do_-4": lambda y: librosa.effects.pitch_shift(y, sr=SAMPLE_RATE, n_steps=-4),
    "toc_do_x0.85": lambda y: librosa.effects.time_stretch(y, rate=0.85),
    "toc_do_x1.15": lambda y: librosa.effects.time_stretch(y, rate=1.15),
    "nhieu_15dB": lambda y: _add_noise(y, 15),
    "nhieu_8dB": lambda y: _add_noise(y, 8),
    "loc_thap_4kHz": lambda y: _lowpass(y, 4000),
    "dien_thoai_3.4kHz": lambda y: _lowpass(y, 3400),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--so-mau", type=int, default=0, help="0 = dung toan bo file take 03")
    ap.add_argument("--mau", type=str, default="reports/eval_voices.json")
    args = ap.parse_args()

    model = load_model()
    files = []
    for p in sorted(WAV_DIR.glob("*_03.wav")):
        m = NAME_RE.match(p.name)
        if m:
            files.append((p, m.group("code")))
    if not files:
        raise SystemExit("Khong thay file take 03 trong data/wav")
    if args.so_mau and args.so_mau < len(files):
        rng = np.random.default_rng(7)
        idx = rng.choice(len(files), size=args.so_mau, replace=False)
        files = [files[i] for i in sorted(idx)]

    ket_qua = {}
    for ten, bien_doi in CHUOIEN.items():
        dung = sai = tu_choi = 0
        for p, that in files:
            y = load_mono(p)
            if bien_doi is not None:
                y = bien_doi(y)
            r = predict_array(np.asarray(y, dtype=np.float32), model)
            if r["rejected"]:
                tu_choi += 1
            elif r["word_code"] == that:
                dung += 1
            else:
                sai += 1
        n = len(files)
        ket_qua[ten] = {
            "n": n,
            "dung": dung,
            "sai": sai,
            "tu_choi": tu_choi,
            "accuracy": round(dung / n, 4),
        }
        print(f"{ten:16} n={n} dung={dung} sai={sai} tu_choi={tu_choi} accuracy={dung/n:.1%}", flush=True)

    payload = {
        "mo_ta": (
            "giong_goc = nguoi da co trong tap huan luyen, lan doc moi (take 03). "
            "Cac muc con lai la MO PHONG bien doi giong tren cung file, "
            "KHONG phai nguoi noi that thu hai."
        ),
        "so_file": len(files),
        "ket_qua": ket_qua,
    }
    out = Path(args.mau)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Da ghi", out)


if __name__ == "__main__":
    main()
