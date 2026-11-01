"""Tạo wav giả (hình sin khác tần số) để kiểm tra pipeline — không dùng khi demo với thầy."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import soundfile as sf

from src.config import SAMPLE_RATE, WAV_DIR, configure_console

configure_console()

WORDS = {
    "ba0": 220,
    "bac1": 330,
    "cang3": 440,
    "d-a+t1": 550,
    "co+1": 660,
}
SPEAKERS = {"hoa": 1.00, "minh": 1.08, "an": 0.94}


def tone(freq: float, seconds: float = 0.55, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, seconds, int(SAMPLE_RATE * seconds), endpoint=False)
    y = 0.35 * np.sin(2 * np.pi * freq * t)
    y += 0.12 * np.sin(2 * np.pi * freq * 2 * t)
    y += 0.04 * rng.normal(0, 1, t.shape)
    fade = int(0.02 * SAMPLE_RATE)
    y[:fade] *= np.linspace(0, 1, fade)
    y[-fade:] *= np.linspace(1, 0, fade)
    return y.astype(np.float32)


def main() -> None:
    WAV_DIR.mkdir(parents=True, exist_ok=True)
    n = 0
    for si, (spk, jitter) in enumerate(SPEAKERS.items()):
        for code, freq in WORDS.items():
            for take in (1, 2, 3):
                y = tone(freq * jitter, seed=si * 100 + take)
                name = f"{spk}_{code}_{take:02d}.wav"
                sf.write(WAV_DIR / name, y, SAMPLE_RATE)
                n += 1
    print(f"Đã tạo {n} file giả trong {WAV_DIR}")
    print("Chỉ dùng để thử hệ thống. Demo thật phải thay bằng file thu âm nhóm.")


if __name__ == "__main__":
    main()
