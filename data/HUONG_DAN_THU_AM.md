# Yêu cầu & kế hoạch thu thập dữ liệu
## Đồ án: nhận dạng đơn từ tiếng Việt (Speech-to-Text)

Gửi cả nhóm. Đọc hết rồi thu theo đúng quy ước — sai tên file là hệ thống không nhận.

---

## 1. Nhóm cần thu thập gì

Hai **loại** dữ liệu (không phải chỉ 2 file):

| Loại | Số lượng | Ai làm |
|---|---|---|
| Danh sách từ | **1 file** (Excel/Google Sheet) | 1 bạn chốt list với thầy / lấy list buổi trước |
| File thu âm | **Nhiều file .wav** — mỗi lần đọc 1 từ = 1 file | **Mỗi thành viên** tự thu giọng mình |

Không ghi một file dài chứa nhiều từ. Không đọc cả câu.

### Số lượng tối thiểu

- Mọi người đọc **cùng một danh sách từ**
- Mỗi người × mỗi từ × **3 lần** (lần 01, 02, 03)

Công thức:

`số file = số từ × số thành viên × 3`

Ví dụ 25 từ × 4 người × 3 lần = **300 file wav**.

---

## 2. Việc cần chốt trước khi thu (nhóm trưởng)

Điền vào đây rồi gửi lại nhóm:

- Số thành viên: ___
- Tên viết không dấu dùng trong file (vd: `hoa`, `minh`, `an`, `tuan`): ___
- Số từ trong list thầy/nhóm: ___
- Deadline nộp file lên Drive: ___
- Link folder Drive: ___

Một bạn (gọi là **thủ thư dữ liệu**) tạo Google Sheet 2 tab:

**Tab A – Danh sách từ** (cột bắt buộc):

| stt | chữ | mã_file |
|---|---|---|
| 1 | ba | ba0 |
| 2 | bác | bac1 |
| 3 | cảng | cang3 |

**Tab B – Checklist thu âm**

| người | số từ đã đủ 3 lần | thiếu | đã nộp Drive |
|---|---|---|---|

---

## 3. Cách mã hóa từ (bắt buộc, theo đề thầy)

Số **dấu** kiểu VNI, **đặt sau từ**. Từ không dấu thêm `0`.

| Dấu | Số | Ví dụ chữ | Mã file |
|---|---|---|---|
| không | 0 | ba | `ba0` |
| sắc | 1 | bác | `bac1` |
| huyền | 2 | bà | `ba2` |
| hỏi | 3 | cảng | `cang3` |
| ngã | 4 | mã | `ma4` |
| nặng | 5 | đất | `d-at5` |

Chữ đặc biệt:

| Chữ | Viết trong mã |
|---|---|
| â | `a+` |
| ô | `o+` |
| ê | `e+` |
| ă | `aa` |
| ơ | `oo` |
| ư | `uu` |
| đ | `d-` |

Thêm ví dụ:

- cố → `co+1`
- cân → `ca+n1`
- cơm → `coom2` (ơ → `oo`, huyền → `2`)
- đất → `d-at5` (đ → `d-`, nặng → `5`)

Từ khó (ươ, diphthong…): **đừng tự đoán**. Thủ thư điền cột `mã_file` trên Sheet, cả nhóm **copy đúng mã đó**.

---

## 4. Đặt tên file thu âm

```
{ten}_{ma}_{lan}.wav
```

- `ten`: tên không dấu, viết thường, không dấu cách (`hoa`, `minh`)
- `ma`: mã từ trên Sheet
- `lan`: `01` `02` `03`

Ví dụ Hoa đọc “bác” lần 2:

`hoa_bac1_02.wav`

Minh đọc “cảng” lần 1:

`minh_cang3_01.wav`

An đọc “đất” lần 3:

`an_d-at5_03.wav`

**Cấm:** dấu tiếng Việt trong tên file, khoảng trắng, `Bác 2.m4a`, `hoa bac.wav`.

---

## 5. Cách thu (từng thành viên)

1. Ngồi yên, phòng ít ồn. Tắt quạt sát mic nếu bị ù.
2. Điện thoại / tai nghe mic, cách miệng khoảng 15–20 cm.
3. Mở app thu (Voice Recorder, Ghi âm). Nếu được, chọn **WAV**. Nếu máy chỉ ra **m4a/mp3** thì vẫn thu, nộp lên Drive, thủ thư đổi sang wav sau.
4. Bấm thu → im lặng ~0,3 giây → **nói đúng 1 từ**, rõ, tốc độ bình thường → im lặng ~0,3 giây → dừng.
5. Nghe lại: không cắt cụt, không ho, không nói nhầm từ.
6. Đổi tên file đúng mục 4 rồi mới nộp.

Mỗi từ làm 3 file riêng (không copy 1 file thành 3 tên).

Không: nhạc nền, nói thêm “ừ”, “cái từ là…”, đọc 2 từ trong 1 file.

---

## 6. Nộp file

Mỗi người tạo folder trên Drive:

```
Drive/STT_data/
  hoa/
    hoa_ba0_01.wav
    hoa_ba0_02.wav
    hoa_ba0_03.wav
    hoa_bac1_01.wav
    ...
  minh/
    ...
```

Nộp **đủ số từ × 3**. Thiếu thì ghi vào checklist, không im.

Thủ thư khi gom xong: copy tất cả `.wav` vào một chỗ (không cần giữ folder theo người — tên file đã có tên người).

---

## 7. Chia việc đề xuất

| Vai | Việc | Xong khi |
|---|---|---|
| Nhóm trưởng | Chốt list từ với thầy, deadline, tên không dấu của từng bạn | Sheet tab A có đủ từ + mã |
| Thủ thư dữ liệu | Giữ Sheet, kiểm tra tên file, đổi m4a→wav nếu cần, zip nộp cho bạn code | Đủ file, không trùng tên, không thiếu lần 01–03 |
| Mỗi thành viên | Thu 3 lần / từ, đặt tên, upload folder của mình | Checklist của mình đủ |

Bạn làm code **không thay** nhóm thu âm. Không có wav thật thì không train được.

---

## 8. Checklist trước khi báo “xong”

- [ ] Mọi người dùng **cùng** danh sách từ
- [ ] Mỗi người có đủ 3 lần / từ
- [ ] Tên file đúng `{ten}_{ma}_{lan}.wav`
- [ ] Nghe ngẫu nhiên 10 file: đúng từ, không rè, không cắt
- [ ] Có ít nhất 1 bộ file **để test** (lần đọc thêm, hoặc 1 bạn đọc vài từ lúc khác) — dùng cho mục đánh giá giọng quen / lần mới

---

## 9. Việc tiếp theo (sau khi thu xong)

Đưa thư mục wav cho bạn code. Bạn đó bỏ vào `data/wav/`, sửa `data/wordlist.csv`, chạy trích đặc trưng và huấn luyện.

Không gửi file giả (hình sin) khi demo với thầy.
