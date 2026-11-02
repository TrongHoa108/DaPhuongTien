from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
WAV_DIR = DATA_DIR / "wav"
WORDLIST_PATH = DATA_DIR / "wordlist.csv"
FEATURES_PATH = DATA_DIR / "features.csv"
MODELS_DIR = DATA_DIR / "models"
MODEL_PATH = MODELS_DIR / "knn.joblib"
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
K_NEIGHBORS = 1  # ít mẫu / từ: k=1 tốt hơn k=3 trên tập kiểm tra (xem BAO_CAO.md)
KNN_METRIC = "manhattan"  # L1: cộng chênh lệch tuyệt đối từng đặc trưng
REJECT_RATIO = 1.8  # khoảng cách xa hơn mean train * hệ số thì từ chối

# Ước lượng cao độ (F0) bằng YIN — ổn định hơn piptrack trên bản ghi ngắn
PITCH_FMIN = 60.0
PITCH_FMAX = 400.0
PITCH_FRAME = 1024

# So khớp chuỗi bằng DTW: lọc nhanh 100 từ bằng KNN rồi so chuỗi MFCC + cao độ.
# Trên tập kiểm tra take 03: KNN tóm tắt 68,6% → DTW 78,9%.
DTW_TEMPLATES_PATH = MODELS_DIR / "dtw_templates.npz"
DTW_SHORTLIST = 100      # số từ ứng viên lấy từ KNN trước khi so DTW
DTW_BAND_RADIUS = 0.25   # dải Sakoe-Chiba
DTW_PITCH_FRAME = 1024
# Ngưỡng từ chối hiệu chỉnh trên take 03: khoảng cách tới từ đúng có p98 = 1,97.
DTW_REJECT_THRESHOLD = 2.0

FEATURE_COLUMNS = (
    [f"mfcc_{i}_mean" for i in range(1, N_MFCC + 1)]
    + [f"mfcc_{i}_std" for i in range(1, N_MFCC + 1)]
    + ["energy_mean", "zcr_mean", "pitch_mean", "pitch_std"]
)
