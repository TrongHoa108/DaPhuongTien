# Nhận dạng đơn từ tiếng Việt (Speech-to-Text)

Đồ án: chuyển **file âm thanh một từ** thành **văn bản**, dùng đặc trưng âm thanh + KNN. Không dùng mô hình học sâu.

## Cài đặt

```powershell
cd D:\CSDL_DPT
python -m venv .venv
.\.venv\Scripts\pip.exe install -r requirements.txt
```

## Dữ liệu thu âm

Đặt file vào `data/wav/`, tên:

`{nguoi}_{ma_tu}_{lan}.wav`

Ví dụ: `hoa_bac1_02.wav`, `minh_cang3_01.wav`, `an_d-at5_03.wav`

Quy ước mã (đề bài):

- Không dấu: số `0` (`ba` → `ba0`)
- Sắc huyền hỏi ngã nặng: `1 2 3 4 5`
- â ô ê → `a+` `o+` `e+`
- ă ơ ư → `aa` `oo` `uu`
- đ → `d-`

Sửa `data/wordlist.csv` cho khớp danh sách từ của nhóm (cột `code`, `text`).

Mỗi từ × mỗi người nên có **ít nhất 3 lần đọc**.

## Chạy pipeline

```powershell
# (tuỳ chọn) wav giả để thử hệ thống — không dùng khi báo cáo
python scripts/make_dummy_data.py

python scripts/extract_features.py
python scripts/train.py
python -m uvicorn app.main:app --reload --port 8000
```

Mở http://127.0.0.1:8000 — tải `.wav` hoặc thu mic.

Nhận dạng một file:

```powershell
python scripts/predict.py data\wav\hoa_bac1_01.wav
```

## Hệ thống (sơ đồ)

```
WAV → cắt im lặng, 16 kHz, chuẩn hóa
    → MFCC (13, CMVN) + năng lượng + ZCR + pitch
    → vector một dòng trong CSV
    → StandardScaler + KNN (k=5)
    → từ / từ chối nếu quá xa
```

Model: `data/models/knn.joblib`  
Đặc trưng: `data/features.csv`  
Accuracy, confusion matrix: `reports/`

Thuật toán chính: **KNN**. Train còn so sánh Logistic Regression và SVM, và 3 bộ đặc trưng (chỉ MFCC / MFCC+năng lượng+ZCR / đủ).

## Nộp bài

Báo cáo bám 7 mục đề: tìm hiểu isolated word, sơ đồ khối, đặc trưng, CSV theo từ/người, demo + accuracy giọng quen / giọng mới.
