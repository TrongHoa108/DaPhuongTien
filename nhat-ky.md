# NHẬT KÝ ĐỒ ÁN — Nhận dạng tiếng nói tiếng Việt (Đề tài 2)

**Cập nhật:** 02/11/2026 · **Trạng thái:** tạm dừng, mai làm tiếp.

---

## 1. Vị trí, repo, dữ liệu

| Thành phần | Đường dẫn / ghi chú |
|---|---|
| Code chính | `D:\hockicuoi\daphuongtien\doan` — nhánh `develop` |
| Remote | `https://github.com/TrongHoa108/DaPhuongTien.git` |
| Dữ liệu gốc | `doan\data\thu_am` (18.599 file trong zip, **7.552 file có tiếng nói thật**) |
| Repo tham khảo (khác, không đụng) | `D:\hockicuoi\daphuongtien` — MFCC+DTW+KNN thuần NumPy, 210 test pass, 30/30 E2E |
| Push GitHub | **Không push được**: tài khoản lưu trên máy là `paw1601`, không có quyền ghi vào repo của `TrongHoa108` (403). Toàn bộ thay đổi đang **uncommitted**. |

---

## 2. Yêu cầu thực tế của thầy (mới nhất — quan trọng nhất)

Thầy sẽ **đưa một đoạn văn bản**, người dùng đọc, hệ thống xử lý và **sinh ra từng chữ** — giống tìm kiếm bằng giọng nói của Google/Siri/Gemini.

> Ví dụ: nói “xin chào tôi là brain” → text “xin chào tôi là brain”.

Đây là **nhận dạng câu liên tục (continuous ASR)**. Hệ thống hiện tại (KNN + DTW đơn từ) **không làm được** và không thể “tinh chỉnh” để làm được.

---

## 3. Đã làm được (hệ đơn từ — đủ 7 mục đề)

### Kiến trúc hiện tại

```
WAV 16 kHz → preprocess (trim, chuẩn hoá, pre-emphasis)
→ MFCC 13 + năng lượng + ZCR + pitch (YIN)
→ StandardScaler + KNN (k=1, Manhattan): lọc 100 từ ứng viên   [recall@100 = 99,1%]
→ DTW trên chuỗi MFCC 13 + cao độ (CMVN) với bộ tham chiếu
→ từ văn bản (hoặc từ chối)
```

### Artifacts

| File | Kích thước | Nội dung |
|---|---|---|
| `data/models/knn.joblib` | 9,9 MB | StandardScaler + KNN, 6.196 lớp, 41.244 mẫu (augmentation) |
| `data/models/dtw_templates.npz` | 50,3 MB | 6.874 bản ghi × 6 biến thể (gốc, cao độ ±3, nhiễu 15/8 dB, lọc 4 kHz) |
| `data/features.csv` | 14 MB | Đặc trưng 41.922 dòng (gitignore) |

### Kết quả đo (678 file `take 03` — cùng người, lần đọc mới)

| Điều kiện | Accuracy |
|---|---|
| Giọng gốc | **78,6%** |
| Nhiễu phòng SNR 15 dB | 68,1% |
| Nhiễu phòng SNR 8 dB | 65,9% |
| Micro lọc 4 kHz | 75,7% |
| Cao độ +4 / −4 nửa cung | 53,8% / 52,2% |
| Tốc độ ×0,85 / ×1,15 | 55,6% / 60,8% |
| Dải điện thoại 3,4 kHz | 46,6% |

Tiến trình: 67,4% → 68,6% (Manhattan, k=1, YIN, augmentation) → **78,6%** (thêm DTW).

### Giao diện

- Nhấn mic → nói một từ → **tự dừng sau 0,8 s im lặng** → tự nhận dạng → hiện text.
- Tick **“Nghe liên tục”**: nói từng từ, chữ tự nối vào văn bản, micro tự nghe tiếp.
- Từ chối rõ ràng: im lặng (`khong_co_tieng_noi`), file nhiều từ (`ban_ghi_nhieu_tu`).
- Ngưỡng từ chối hiệu chỉnh trên **DTW** (p98 = 1,97 → chọn 2,0). Trước đó ngưỡng KNN chặn oan bản ghi mic.

### Script

| Script | Việc |
|---|---|
| `scripts/import_recordings.py` | Gom WAV thật vào `data/wav` |
| `scripts/extract_features.py` | Trích đặc trưng + augmentation (`--augment-pitch`, `--augment-noise`, `--augment-lowpass`) |
| `scripts/train.py` | Train KNN + báo cáo metrics/confusion |
| `scripts/build_dtw_templates.py` | Tạo bộ tham chiếu DTW |
| `scripts/evaluate_voices.py` | 9 điều kiện giọng/nhiễu/tốc độ → `reports/eval_voices.json` |
| `scripts/kiem_tra_file.py` | Chẩn đoán một file: đoạn cắt, top-5 DTW, năng lượng/pitch |

---

## 4. Vấn đề đã xác định (đừng chẩn đoán lại)

1. **Dữ liệu:** 1 người nói, 6.196 từ; **89% từ chỉ có 1 lần đọc** → trần ~78%.
2. **`test2.wav` trên Desktop** (trùng `data\thu_am\ten\test\test2.wav`): **12,12 s, 14 đoạn** → không phải file một từ; hệ thống từ chối là đúng. Không thể chuyển thành văn bản bằng hệ đơn từ.
3. **`uploads\mic.wav`** (bản ghi mic gần nhất): 1,54 s, 1 đoạn 0,53 s, năng lượng 0,057, pitch 156 Hz. Hợp lệ. Trước đây bị ngưỡng KNN (22,2 > 20,9) chặn oan; giờ DTW = 1,538 → ra **“xuyền”**.
4. Người dùng nói cụm “xin chào” → hệ đơn từ coi là **một từ**, chọn từ gần nhất (“xuyền”). “xin chào” không phải một mục trong `wordlist.csv`; **“brain” hoàn toàn không có** trong từ vựng.
5. Lỗi “sắc” → “ắc”: vector tóm tắt 30 chiều không phân biệt âm đầu; DTW đỡ hơn nhưng chưa đủ.
6. Nói liền nhiều từ → không xử lý được (đúng phạm vi đề).

---

## 5. Phương án tiếp theo (CHƯA CHỐT — cần quyết định đầu tiên vào ngày mai)

### Đã khảo sát

- **PhoWhisper** (VinAI) có trên HuggingFace: base/small/medium/large.
- Có bản **CTranslate2**: `mad1999/pho-whisper-small-ct2`, `mad1999/pho-whisper-medium-ct2` → dùng trực tiếp với `faster-whisper`, **không cần torch**.
- **Sherpa-onnx**: không tìm thấy model streaming tiếng Việt.
- Máy: **RTX 3050 Laptop 4 GB VRAM**, Python 3.11.9, mạng OK (HF/PyPI trả 200), ổ D: trống 24,5 GB.
- Chưa cài `torch` / `ctranslate2` / `whisper` / `vosk`.

### Kiến trúc đề xuất (nếu được phép)

- **Giữ tab đơn từ KNN+DTW** (đủ 7 mục đề, 78,6%).
- **Thêm tab “Câu”**: faster-whisper + PhoWhisper-small-ct2 (hoặc medium-ct2).
- Backend: WebSocket `/ws/asr` nhận PCM 16 kHz; đệm + VAD; giải mã mỗi cụm khi im lặng 0,5–0,7 s; trả text từng cụm. Offload CPU khỏi event loop (`run_in_threadpool`) — hiện `POST /api/predict` đang chặn event loop.
- Frontend: giữ AudioContext/stream liên tục, gửi chunk 200–250 ms; nối chữ dần. Không ghi đè `uploads/mic.wav` mỗi request (hiện tại có rủi ro tranh chấp).
- Kỳ vọng: PhoWhisper-small đọc rõ → WER ~10–20%; latency 0,5–2 s/cụm (GPU int8) hoặc 1–3 s (CPU).
- Tải model ~500 MB (small) / ~1,5 GB (medium) vào `data/models` (gitignore).

### 4 câu hỏi phải chốt trước khi làm

1. **Thầy có cho phép dùng mô hình ASR có sẵn** (PhoWhisper/Whisper) không? Nếu không → không thể nhận dạng câu; chỉ demo đơn từ và phải nói rõ phạm vi.
2. **Giữ cả hai hay thay thế**: giữ đơn từ KNN+DTW + thêm tab “Câu”, hay chuyển hẳn sang câu?
3. **Chọn model**: PhoWhisper-small (nhanh, ~500 MB) hay medium (chính xác hơn, ~1,5 GB)?
4. **Kiểu hiện chữ**: từng cụm sau khi ngừng nói (~1 s, ổn định) hay từng từ hiện dần (partial, giống Google nhưng nhảy chữ)?

---

## 6. Ngày mai bắt đầu từ đâu (thứ tự)

1. Chốt 4 câu hỏi ở mục 5.
2. Nếu được phép: `pip install faster-whisper`, tải `mad1999/pho-whisper-small-ct2` (hoặc medium-ct2).
3. Viết `src/asr.py` + endpoint `/ws/asr` + tab “Câu” trong UI (tái dùng `renderResult` để nối chữ).
4. Test 10 câu đọc, đo WER + latency (GPU int8 và CPU).
5. Cập nhật `README.md`/`BAO_CAO.md`: ghi rõ phần câu dùng mô hình có sẵn là **mở rộng**, phần đơn từ vẫn KNN+DTW; sửa dòng `README.md:118` đang mâu thuẫn (nói demo hỗ trợ đoạn nói).
6. Commit local (push vẫn 403).

---

## 7. Lệnh chạy hiện tại

```powershell
cd D:\hockicuoi\daphuongtien\doan
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload --port 8000
```

Dựng lại toàn bộ pipeline (thứ tự đúng):

```powershell
.\.venv\Scripts\python.exe scripts\import_recordings.py --src data\thu_am --clear --speaker caotuankiet
.\.venv\Scripts\python.exe scripts\extract_features.py --augment-pitch "3,-3" --augment-noise "15,8" --augment-lowpass "4000"
.\.venv\Scripts\python.exe scripts\train.py
.\.venv\Scripts\python.exe scripts\build_dtw_templates.py --augment-pitch "3,-3" --augment-noise "15,8" --augment-lowpass "4000"
.\.venv\Scripts\python.exe scripts\evaluate_voices.py
```

Kiểm tra một file trước khi test:

```powershell
.\.venv\Scripts\python.exe scripts\kiem_tra_file.py "đường_dẫn\file.wav"
```

Chỉ test khi `Đoạn cắt: 1` và độ dài ≤ 1,5 s.

---

## 8. Trạng thái Git (đang uncommitted)

**Sửa:** `.gitignore`, `BAO_CAO.md`, `README.md`, `app/main.py`, `app/static/app.js`, `app/static/index.html`, `app/static/styles.css`, `data/models/knn.joblib`, `reports/metrics.json`, `scripts/extract_features.py`, `scripts/train.py`, `src/config.py`, `src/dataset.py`, `src/features.py`, `src/recognize.py`

**Mới:** `data/models/dtw_templates.npz`, `scripts/build_dtw_templates.py`, `scripts/evaluate_voices.py`, `scripts/kiem_tra_file.py`, `src/dtw.py`, `nhat-ky.md`

**Backup model cũ:** `%TEMP%\knn_backup.joblib`, `%TEMP%\metrics_backup.json` (khôi phục: `git checkout -- data/models/knn.joblib reports/metrics.json`).

---

## 9. Số đo hiệu năng (để tối ưu nếu cần)

- `load_model()` lần đầu ~1,8 s; `load_templates()` ~2,3 s (51 MB, ~117 MB RAM).
- Mỗi từ: KNN 2,3 ms + DTW rerank ~100 ms (690 so sánh) → ~0,1 s sau khi nạp.
- UI hiện tại: ~1,3–1,8 s/từ (0,8 s chờ im lặng + 256 ms buffer + 300 ms nghỉ + RTT).
- DTW là phần tốn nhất; có thể giảm `DTW_SHORTLIST` (100 → 60) nếu cần nhanh hơn.
