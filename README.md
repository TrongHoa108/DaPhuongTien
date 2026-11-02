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
| `data/wordlist.csv` | 6.199 từ + mã VNI |
| `data/thu_am/` | Bản thu gốc (giải nén từ `thu_am.zip`); 18.599 file, trong đó **7.552 file có tiếng nói thật** |
| `data/wav/` | Wav đã lọc (chỉ file thu thật) để train — 7.552 file, 6.196 từ, 1 người nói |
| `data/features.csv` | Đặc trưng theo từ / người (khi train có augmentation: ~21.300 dòng) |
| `data/models/knn.joblib` | Model KNN (StandardScaler + KNN k=1, Manhattan) — lọc ứng viên |
| `data/models/dtw_templates.npz` | Bộ tham chiếu DTW: chuỗi MFCC + cao độ, 6.874 bản ghi × 6 biến thể (~53 MB) |
| `reports/` | Accuracy, confusion matrix, đánh giá giọng quen / giọng mô phỏng |

Tên file: `{nguoi}_{ma}_{lan}.wav`  
Ví dụ: `caotuankiet_bac1_01.wav`

Mã VNI: `0` không dấu; `1–5` sắc huyền hỏi ngã nặng; â/ô/ê → `a+/o+/e+`; ă/ơ/ư → `aa/oo/uu`; đ → `d-`.

## Pipeline

```powershell
# 1) Gom thu âm thật vào data/wav (bỏ file im lặng)
.\.venv\Scripts\python.exe scripts\import_recordings.py --src data\thu_am --clear --speaker caotuankiet

# 2) Trích đặc trưng + augmentation cho mẫu train (không đụng take 03):
#    cao độ ±3 nửa cung, nhiễu 15/8 dB, lọc thông thấp 4 kHz
.\.venv\Scripts\python.exe scripts\extract_features.py --augment-pitch "3,-3" --augment-noise "15,8" --augment-lowpass "4000"

# 3) Huấn luyện KNN (+ báo cáo accuracy/confusion matrix)
.\.venv\Scripts\python.exe scripts\train.py

# 4) Tạo bộ tham chiếu DTW (chuỗi MFCC + cao độ) — bỏ take 03 để giữ tập test
.\.venv\Scripts\python.exe scripts\build_dtw_templates.py --augment-pitch "3,-3" --augment-noise "15,8" --augment-lowpass "4000"

# 5) Đánh giá giọng quen + giọng mô phỏng (ghi reports/eval_voices.json)
.\.venv\Scripts\python.exe scripts\evaluate_voices.py

# 6) Demo web
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

Mở http://127.0.0.1:8000 — tải `.wav` hoặc thu mic.

```powershell
.\.venv\Scripts\python.exe scripts\predict.py data\wav\caotuankiet_ba0_01.wav
```

## Kết quả

Test = lần đọc `_03` (cùng người, lần đọc mới), 678 file; train = 41.244 mẫu
(augmentation cao độ ±3, nhiễu 15/8 dB, lọc thông thấp 4 kHz).

| Điều kiện thu | KNN tóm tắt | **KNN lọc + DTW (hiện tại)** |
|---|---|---|
| Giọng gốc, lần đọc mới | 67,8% | **78,6%** |
| Nhiễu phòng SNR 15 dB | 52,5% | **68,1%** |
| Nhiễu phòng SNR 8 dB | 37,3% | **65,9%** |
| Micro lọc 4 kHz | 63,0% | **75,7%** |
| Cao độ +4 / −4 nửa cung | 36,9% / 23,7% | **53,8% / 52,2%** |
| Tốc độ ×0,85 / ×1,15 | 40,6% / 40,7% | **55,6% / 60,8%** |
| Dải điện thoại 3,4 kHz | 1,2% | **46,6%** |
| Số lớp trong model | 6.196 | |

Các dòng nhiễu/lọc/cao độ mô phỏng đúng tình huống **thu bằng micro laptop trong phòng bình thường**.
Thời gian nhận dạng: ~0,1 giây/từ sau lần nạp đầu (~2 giây).
Chi tiết: `reports/metrics.json`, `reports/classification_report.txt`, `reports/confusion_knn.png`, `reports/eval_voices.json`.

## Phạm vi và cách demo

Chính của hệ thống là **nhận dạng đơn từ**: mỗi lần thu = **một từ** (khoảng 0,4–1,2 giây).

- Nói một từ, nghỉ một nhịp. Hệ thống **tự dừng sau 0,8 giây im lặng** và tự gửi đi nhận dạng.
- Bật **"Nghe liên tục"** để nói từng từ mà không cần bấm lại: sau mỗi từ, chữ được nối vào văn bản và micro tự nghe tiếp.
- File mẫu một từ để thử ngay:
  - `data\wav\caotuankiet_bom0_03.wav` → `bom`
  - `data\wav\caotuankiet_baap1_03.wav` → `bắp`
  - `data\wav\caotuankiet_ac1_03.wav` → `ác`
- Kiểm tra một file trước khi test: `python scripts\kiem_tra_file.py "file.wav"` — dòng "Đoạn cắt" phải là `1`.

**Chế độ "Đoạn nhiều từ (câu)" (mở rộng):** cắt bản ghi theo khoảng lặng (nghỉ 0,3 giây giữa từ), nhận dạng **từng mảnh** bằng chính hệ KNN+DTW ở mức từ rồi ghép thành câu. Kết quả bị giới hạn bởi: (1) chỉ nhận được các từ có trong danh sách 6.196 từ — mảnh ngoài từ vựng bị bỏ qua (hiện số trong dòng meta), (2) sai số từng mảnh ~20% cộng dồn, (3) từ đọc liền không nghỉ sẽ bị gộp vào một mảnh. Đây vẫn là kỹ thuật nhận dạng **đơn từ** ghép lại, không phải nhận dạng câu liên tục.

**Giới hạn đã biết:** accuracy 78,6% trên lần đọc mới của cùng người nói (KNN + DTW; KNN thuần 68,6%); 89% từ chỉ có 1 lần đọc và dữ liệu chỉ có một người nói, nên khoảng 1/5–1/3 từ sẽ nhận sai, chủ yếu do nhầm thanh điệu hoặc âm đầu. Bản ghi mic thực tế (âm lượng yếu, nhiễu khác thiết bị thu train) có khoảng cách đặc trưng lệch lớn hơn — nếu nhận sai nhiều trên mic, thu thêm dữ liệu qua cùng đường mic (gán nhãn từ tên file) rồi chạy lại `extract_features.py` + `train.py` + `build_dtw_templates.py` sẽ giảm chênh lệch. Đây là giới hạn của dữ liệu/phương pháp, không phải lỗi thao tác.

## Sơ đồ nhanh

```
WAV → 16 kHz, trim, chuẩn hóa
    → MFCC (13) + năng lượng + ZCR + pitch (YIN)
    → CSV theo từ / người
    → StandardScaler + KNN (k=1, Manhattan): lọc 100 từ ứng viên
    → DTW so chuỗi MFCC + cao độ với bộ tham chiếu
    → từ văn bản (hoặc từ chối)
```

## Git & deploy

- Repo: https://github.com/TrongHoa108/DaPhuongTien  
- Nhánh mặc định: **`develop`** (không dùng `main`)  
- Deploy Render (Docker, chọn branch `develop`):  
  https://render.com/deploy?repo=https://github.com/TrongHoa108/DaPhuongTien  

Hoặc trên [Render](https://render.com) → New → Web Service → nối repo → branch **`develop`** → Docker.

## Lưu ý

- Model vẫn là **isolated word (KNN)**. Demo hỗ trợ **đoạn nói**: cắt theo khoảng lặng → nhận từng từ → ghép câu. Nên nghỉ ~0,3–0,5 giây giữa các từ.
- File wav và zip thu âm **không** đẩy lên Git (xem `.gitignore`). Giữ trên Drive / máy local.
- Dữ liệu chỉ có **một người nói** (7.552 file, 6.196 từ; 89% từ chỉ có 1 lần đọc). Vì vậy **chưa đo được accuracy với người nói thật thứ hai**; con số "giọng mới" ở trên là **mô phỏng** (dịch cao độ/đổi tốc độ). Muốn đánh giá thật: thu thêm cùng danh sách với `--speaker <ten_nguoi_moi>` rồi chạy lại `scripts/evaluate_voices.py`.
