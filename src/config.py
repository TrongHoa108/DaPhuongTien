from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
WAV_DIR = DATA_DIR / "wav"
WORDLIST_PATH = DATA_DIR / "wordlist.csv"
FEATURES_PATH = DATA_DIR / "features.csv"
MODELS_DIR = DATA_DIR / "models"
REPORTS_DIR = ROOT / "reports"
UPLOADS_DIR = ROOT / "uploads"


def configure_console() -> None:
    import sys

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
            sys.stderr.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

SAMPLE_RATE = 16000
N_MFCC = 13
N_FFT = 512
HOP_LENGTH = 160  # 10 ms at 16 kHz
K_NEIGHBORS = 3  # ít mẫu / từ: k nhỏ hơn thường chính xác hơn
REJECT_RATIO = 1.8  # khoảng cách xa hơn mean train * hệ số thì từ chối

FEATURE_COLUMNS = (
    [f"mfcc_{i}_mean" for i in range(1, N_MFCC + 1)]
    + [f"mfcc_{i}_std" for i in range(1, N_MFCC + 1)]
    + ["energy_mean", "zcr_mean", "pitch_mean", "pitch_std"]
)
