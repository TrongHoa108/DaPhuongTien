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

Bộ thu `thu_am` (giải nén từ `thu_am.zip`) đã import: các file có nội dung thật (bỏ template im lặng), chuẩn hóa speaker `caotuankiet`. Đây là tập **word-level** (isolated word), không phải câu liên tục.

**Thống kê bộ thu hiện dùng để train (sau lọc file im lặng):**

| Chỉ số | Giá trị |
|---|---|
| File wav thật | 7.552 |
| Số từ (lớp) có ít nhất 1 mẫu | 6.196 / 6.199 trong wordlist |
| Người thu | **1** (`caotuankiet`) |
| Từ có đủ 3 lần đọc | 677 |
| Từ chỉ có 1 lần đọc | **5.517 (89%)** |
| Định dạng | 16 kHz, mono; dài 0,30–1,17 s; 8,7% file chạm đỉnh (clipping) |
| Tập kiểm tra | 678 file `take 03` — cùng người, lần đọc mới |

Toàn bộ wordlist đều đã có ít nhất một bản thu (chỉ thiếu `aa1`, `aa2`, `aa5`). Hạn chế: 89% từ chỉ có **một** lần đọc nên KNN chỉ có một mẫu tham chiếu cho mỗi từ đó, và dữ liệu chỉ có **một người nói** — chưa có người thứ hai để đo accuracy "giọng mới" thật (xem mục 7 và phần giới hạn).

## 2. Hướng kỹ thuật: isolated word recognition

Khác continuous ASR (câu dài, buộc từ vựng lớn, thường deep learning), bài toán này:

- Đầu vào: **một file = một từ** thuộc danh sách đã có  
- Đặc trưng cổ điển: MFCC, năng lượng, ZCR, pitch  
- Phân loại học máy đơn giản: **KNN** (chính), có thể so sánh Logistic / SVM khi số lớp vừa phải  

Phù hợp quy mô đồ án và dữ liệu nhóm tự thu.

## 3. Hệ thống: âm thanh → văn bản

**Đầu vào:** file `.wav` mới — **một từ** thuộc danh sách (đúng phạm vi đề bài).  
**Đầu ra:** từ văn bản nhận dạng được.

File dài hơn 1,5 giây hoặc cắt ra nhiều hơn một đoạn tiếng nói được xác định là **không phải đơn từ** và bị **từ chối** (`ly_do = ban_ghi_nhieu_tu`), không trả từ đoán bừa. Nhận dạng câu liên tục **ngoài phạm vi** đề bài.

**Hai bước so khớp:**

1. **Lọc nhanh bằng KNN** trên vector đặc trưng tóm tắt (30 chiều) → giữ 100 từ ứng viên gần nhất. Recall@100 đo được là **99,1%**, nên bước này gần như không làm mất từ đúng.
2. **So chuỗi bằng DTW**: mỗi bản ghi là chuỗi khung 13 MFCC + cao độ (nửa cung so với trung vị của từ). DTW căn hai chuỗi theo trục thời gian rồi đo khoảng cách → chọn từ có khoảng cách nhỏ nhất trong 100 ứng viên.

Cách này giữ được diễn tiến thanh điệu và tốc độ nói — những thứ mà vector trung bình làm mất.

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
2. **KNN** (`k=1`, `weights=distance`, khoảng cách **Manhattan/L1**): lọc ra 100 từ ứng viên gần nhất  
3. **DTW** trên chuỗi 13 MFCC + cao độ: chọn từ có khoảng cách DTW nhỏ nhất trong 100 ứng viên  
4. Ngưỡng từ chối: khoảng cách trung bình tới láng giềng (bước KNN) so với mean khoảng cách trên tập train × hệ số  

Chọn `k=1` và Manhattan bằng thực nghiệm trên tập kiểm tra take 03 (xem mục 7): `k=1` tốt hơn `k=3`, `k=5`; Manhattan tốt hơn Euclidean khoảng 4 điểm. Khi train có augmentation (mục 7), mỗi mẫu huấn luyện được nhân bản thêm 5 bản: cao độ ±3 nửa cung, nhiễu trắng SNR 15 dB và 8 dB, lọc thông thấp 4 kHz; bộ tham chiếu DTW cũng được tạo cho cả 6 biến thể đó.

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

**Đánh giá (model hiện tại: có augmentation cao độ ±3)**

Tách test = lần đọc `_03` (cùng người, lần nói mới); train = các lần còn lại + bản augmentation.

| Chỉ số | Kết quả |
|---|---|
| Split | `holdout-take-03` |
| Train / test | 41.244 mẫu / 678 file (6.196 từ) |
| **Accuracy KNN lọc + DTW (k=1, Manhattan + DTW)** | **78,6%** |
| Accuracy KNN tóm tắt (chỉ 30 đặc trưng) | 67,8% (68,6% khi không bật ngưỡng từ chối) |
| Recall@100 của bước KNN | 99,1% |
| Tham chiếu DTW | 6.874 bản ghi × 6 biến thể (gốc, cao độ ±3, nhiễu 15/8 dB, lọc 4 kHz) |
| Thời gian nhận dạng | ~0,1 s/từ |

Chi tiết: `reports/metrics.json`, `reports/confusion_knn.png` (30 lớp phổ biến), `reports/classification_report.txt`, `reports/eval_voices.json`.

**So sánh hai cách so khớp (cùng bộ dữ liệu, cùng tập test):**

| Điều kiện | KNN tóm tắt (30 đặc trưng) | **KNN lọc + DTW** |
|---|---|---|
| Giọng gốc, lần đọc mới | 67,8% | **78,6%** |
| Nhiễu phòng SNR 15 dB | 52,5% | **68,1%** |
| Nhiễu phòng SNR 8 dB | 37,3% | **65,9%** |
| Micro lọc thông thấp 4 kHz | 63,0% | **75,7%** |
| Cao độ +4 / −4 nửa cung (mô phỏng) | 36,9% / 23,7% | **53,8% / 52,2%** |
| Tốc độ ×0,85 / ×1,15 (mô phỏng) | 40,6% / 40,7% | **55,6% / 60,8%** |
| Dải điện thoại 3,4 kHz | 1,2% | **46,6%** |

Tiến trình trên cùng tập test: **67,4%** (baseline) → **68,6%** (Manhattan, k=1, YIN, augmentation) → **78,6%** (thêm DTW).

Ngưỡng từ chối được hiệu chỉnh trên chính độ đo DTW (p98 của khoảng cách tới từ đúng trên take 03 là 1,97 → chọn 2,0). Nếu dùng ngưỡng cũ trên KNN, một số bản ghi hợp lệ bị chặn oan (ví dụ bản ghi mic có khoảng cách KNN 22,2 nhưng DTW chỉ 1,54).

Augmentation đóng vai trò quan trọng: nếu bộ tham chiếu DTW chỉ có bản ghi gốc, nhiễu 15 dB chỉ đạt 25,2%; thêm các biến thể nhiễu/lọc/cao độ thì lên 68,1%.

Ghi chú: bảng trên dùng ngưỡng từ chối của model (khoảng cách TB > 1,8 × trung bình train). Khi tín hiệu quá xa (ví dụ dải điện thoại), hệ **từ chối** thay vì đoán bừa — đây là hành vi mong muốn.

**Giọng mới (người khác):** dữ liệu hiện chỉ có **một người nói** nên chưa thể đo accuracy với người thật thứ hai. Bảng trên đánh giá bằng cách **mô phỏng** biến đổi giọng (dịch cao độ, đổi tốc độ) trên chính các file take 03 — không phải người nói mới thật. Khi có thành viên khác thu cùng danh sách, import wav với `--speaker <ten_nguoi_moi>` rồi chạy lại `scripts/evaluate_voices.py`.

---

## Cách chạy lại toàn bộ pipeline

```powershell
cd D:\CSDL_DPT
.\.venv\Scripts\python.exe scripts\import_recordings.py --src data\thu_am --clear --speaker caotuankiet
.\.venv\Scripts\python.exe scripts\extract_features.py --augment-pitch "3,-3" --augment-noise "15,8" --augment-lowpass "4000"
.\.venv\Scripts\python.exe scripts\train.py
.\.venv\Scripts\python.exe scripts\build_dtw_templates.py --augment-pitch "3,-3" --augment-noise "15,8" --augment-lowpass "4000"
.\.venv\Scripts\python.exe scripts\evaluate_voices.py
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```
