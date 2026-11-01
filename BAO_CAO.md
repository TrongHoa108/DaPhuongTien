# Báo cáo đề tài 2 — Speech-to-Text mức đơn từ (tiếng Việt)

## 1. Bộ dữ liệu huấn luyện / tham chiếu

Hệ thống dùng:

| Thành phần | Vị trí | Mô tả |
|---|---|---|
| Danh sách từ | `data/wordlist.csv` | Đơn từ tiếng Việt, mã VNI (`code`), chữ hiển thị (`text`), nhóm âm |
| File thu âm | `data/wav/` | `{nguoi}_{ma}_{lan}.wav` — một từ / một file |
| Đặc trưng | `data/features.csv` | Vector đặc trưng theo từng file (từ + người) |
| Model | `data/models/knn.joblib` | StandardScaler + KNN |

**Quy ước mã VNI:** không dấu `0`; sắc–huyền–hỏi–ngã–nặng `1–5`; â/ô/ê → `a+/o+/e+`; ă/ơ/ư → `aa/oo/uu`; đ → `d-`.

Bộ thu `thu_am_HT` đã import: các file có nội dung thật (bỏ template im lặng), chuẩn hóa speaker `caotuankiet`. Đây là tập **word-level** (isolated word), không phải câu liên tục.

**Thống kê bộ thu hiện dùng để train (sau lọc file im lặng):**

| Chỉ số | Giá trị |
|---|---|
| File wav thật | 4.569 |
| Số từ (lớp) có ít nhất 1 mẫu | 3.303 / 6.199 trong wordlist |
| Người thu | 1 (`caotuankiet`) |
| Từ đủ 3 lần đọc | ~632 |
| Chất lượng mẫu kỹ thuật | Đạt (16 kHz, độ dài và âm lượng ổn) |

Chưa đủ toàn bộ chữ cái trong wordlist (thiếu nhiều mục ở p–y, đ, nguyên âm có dấu…). Model chỉ nhận đúng các từ **đã thu**.

## 2. Hướng kỹ thuật: isolated word recognition

Khác continuous ASR (câu dài, buộc từ vựng lớn, thường deep learning), bài toán này:

- Đầu vào: **một file = một từ** thuộc danh sách đã có  
- Đặc trưng cổ điển: MFCC, năng lượng, ZCR, pitch  
- Phân loại học máy đơn giản: **KNN** (chính), có thể so sánh Logistic / SVM khi số lớp vừa phải  

Phù hợp quy mô đồ án và dữ liệu nhóm tự thu.

## 3. Hệ thống: âm thanh → văn bản

**Đầu vào:** file `.wav` mới — một từ **hoặc một đoạn** (nhiều từ cách nhau bằng khoảng nghỉ ngắn).  
**Đầu ra:** văn bản ghép các từ nhận dạng được; chi tiết từng đoạn trong `words[]`.

Cách xử lý đoạn: cắt theo khoảng lặng → KNN từng đoạn → ghép câu. Vẫn dựa trên bộ từ đã thu (không phải ASR câu tự do).

Demo web: `http://127.0.0.1:8000` — thu mic / tải file / kéo-thả.

## 4. Sơ đồ khối và quy trình

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────────┐
│ File / mic  │ →  │ Tiền xử lý   │ →  │ Trích đặc trưng     │
│ WAV 1 từ    │    │ 16 kHz, mono │    │ MFCC+E+ZCR+pitch    │
│             │    │ trim, norm   │    │ → vector 1 dòng     │
└─────────────┘    └──────────────┘    └──────────┬──────────┘
                                                   │
                                                   ▼
                       ┌──────────────┐    ┌────────────────┐
                       │ Từ chối nếu  │ ←  │ StandardScaler │
                       │ quá xa KNN   │    │ + KNN (k=5)    │
                       └──────┬───────┘    └────────────────┘
                              │
                              ▼
                       ┌──────────────┐
                       │ Văn bản từ   │
                       │ (word_text)  │
                       └──────────────┘
```

**Quy trình vận hành**

1. Nhận file âm thanh (upload / mic → WAV 16 kHz)  
2. Cắt im lặng, chuẩn hóa biên độ, pre-emphasis  
3. Trích vector đặc trưng  
4. Chuẩn hóa (scaler đã học trên train)  
5. KNN tìm láng giềng gần nhất trong không gian đặc trưng  
6. Nếu khoảng cách trung bình quá lớn so với train → **từ chối**  
7. Ngược lại → giải mã nhãn `word_code` → `word_text`

**Quy trình huấn luyện**

1. Gom wav đúng tên vào `data/wav/`  
2. `python scripts/extract_features.py` → `data/features.csv`  
3. `python scripts/train.py` → model + `reports/metrics.json`  
4. Chạy demo FastAPI  

## 5. Đặc trưng âm thanh

| Đặc trưng | Ý nghĩa | Cách trích |
|---|---|---|
| **MFCC** (13) | Phổ âm thanh theo thính giác người | librosa MFCC, lấy mean + std theo thời gian, CMVN |
| **Năng lượng** | Độ mạnh tín hiệu | trung bình RMS / năng lượng khung |
| **ZCR** | Tần số cắt không (phụ âm / nguyên âm) | zero-crossing rate trung bình |
| **Pitch** | Cao độ (thanh điệu tiếng Việt) | ước lượng F0 (yin), mean + std |

Vector một mẫu = nối các thống kê trên → một dòng CSV (cột `mfcc_*_mean/std`, `energy_mean`, `zcr_mean`, `pitch_mean`, `pitch_std`) kèm `speaker`, `word_code`, `word_text`, `take`, `file`.

## 6. Lưu trữ và so khớp / phân loại

**Lưu trữ**

- `data/features.csv`: mỗi dòng = một lần đọc của một người cho một từ  
- Tra cứu theo `word_code` / `speaker` / `take`  
- Model đóng gói: scaler + KNN + label encoder (`knn.joblib`)

**Phân loại**

1. Chuẩn hóa vector đầu vào bằng `StandardScaler`  
2. **KNN** (`k=5`, `weights=distance`): nhãn = từ của láng giềng gần nhất  
3. Ngưỡng từ chối: khoảng cách trung bình tới láng giềng so với mean khoảng cách trên tập train × hệ số  

Không dùng deep learning (RNN/LSTM/CNN) — đúng ghi chú đề bài.

## 7. Demo và đánh giá

**Demo**

```powershell
cd D:\CSDL_DPT
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

Mở http://127.0.0.1:8000 — tải wav hoặc thu mic → hiện từ nhận dạng.

CLI:

```powershell
.\.venv\Scripts\python.exe scripts\predict.py data\wav\<file>.wav
```

**Đánh giá (lần train hiện tại)**

Tách test = lần đọc `_03` (cùng người, lần nói mới); train = các lần còn lại.

| Chỉ số | Kết quả |
|---|---|
| Split | `holdout-take-03` |
| Train / test | 3.936 / 633 |
| Số lớp trong model | 3.303 |
| **KNN accuracy (k=3)** | **~68,1%** |
| Cùng người, lần đọc mới | ~68,1% |
| Chỉ MFCC | ~63,7% |
| MFCC + năng lượng + ZCR | ~62,9% |
| Đủ đặc trưng (MFCC+E+ZCR+pitch) | ~68,1% |

Chi tiết: `reports/metrics.json`, `reports/confusion_knn.png` (30 lớp phổ biến), `reports/classification_report.txt`.

**Giọng mới (người khác):** hiện mới có 1 người thu → chưa đo `accuracy_new_speaker`. Khi có thêm thành viên, import wav với tên người khác rồi train lại; script sẽ tự tách group-by-speaker.

---

## Cách chạy lại toàn bộ pipeline

```powershell
cd D:\CSDL_DPT
.\.venv\Scripts\python.exe scripts\import_recordings.py --clear --speaker caotuankiet
.\.venv\Scripts\python.exe scripts\extract_features.py
.\.venv\Scripts\python.exe scripts\train.py
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```
