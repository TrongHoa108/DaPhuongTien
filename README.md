# VietWord STT — Nhận dạng đơn từ tiếng Việt

Đề tài 2: chuyển **file âm thanh một từ** thành **văn bản** (isolated word), dùng MFCC + năng lượng + ZCR + pitch và **KNN**. Không bắt buộc học sâu.

Báo cáo bám 7 mục đề: xem [BAO_CAO.md](BAO_CAO.md).

## Cài đặt

```powershell
cd D:\CSDL_DPT
python -m venv .venv
.\.venv\Scripts\pip.exe install -r requirements.txt
```

## Dữ liệu

| File / thư mục | Vai trò |
|---|---|
| `data/wordlist.csv` | Danh sách từ + mã VNI |
| `data/thu_am_HT/` | Bản thu gốc (giải nén từ `thu_am_HT.zip`) |
| `data/wav/` | Wav đã lọc (chỉ file thu thật) để train |
| `data/features.csv` | Đặc trưng theo từ / người |
| `data/models/knn.joblib` | Model KNN |
| `reports/` | Accuracy, confusion matrix |

Tên file: `{nguoi}_{ma}_{lan}.wav`  
Ví dụ: `caotuankiet_bac1_01.wav`

Mã VNI: `0` không dấu; `1–5` sắc huyền hỏi ngã nặng; â/ô/ê → `a+/o+/e+`; ă/ơ/ư → `aa/oo/uu`; đ → `d-`.

## Pipeline

```powershell
# 1) Gom thu âm thật vào data/wav (bỏ file im lặng)
.\.venv\Scripts\python.exe scripts\import_recordings.py --clear --speaker caotuankiet

# 2) Trích đặc trưng
.\.venv\Scripts\python.exe scripts\extract_features.py

# 3) Huấn luyện KNN (+ báo cáo)
.\.venv\Scripts\python.exe scripts\train.py

# 4) Demo web
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

Mở http://127.0.0.1:8000 — tải `.wav` hoặc thu mic.

```powershell
.\.venv\Scripts\python.exe scripts\predict.py data\wav\caotuankiet_ba0_01.wav
```

## Sơ đồ nhanh

```
WAV → 16 kHz, trim, chuẩn hóa
    → MFCC (13) + năng lượng + ZCR + pitch
    → CSV theo từ / người
    → StandardScaler + KNN
    → từ văn bản (hoặc từ chối)
```

## Lưu ý

- Model vẫn là **isolated word (KNN)**. Demo hỗ trợ **đoạn nói**: cắt theo khoảng lặng → nhận từng từ → ghép câu. Nên nghỉ ~0,3–0,5 giây giữa các từ.
- File wav và zip thu âm **không** đẩy lên Git (xem `.gitignore`). Giữ trên Drive / máy local.
