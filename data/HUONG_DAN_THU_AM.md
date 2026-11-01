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

Bảng **đơn từ** (tiếng một chữ, không lấy học sinh / Việt Nam / không phải…): mỗi cột một chữ cái, trên xuống 1 âm → 2 âm → 3 âm.

Mở `D:\CSDL_DPT\DON_TU_MOT_CHU.xlsx` (đóng file Excel cũ nếu đang khóa). Khoảng 6.200 tiếng một chữ. Mã file: `data/wordlist.csv`.

Danh sách đủ mã file (377 dòng): `data/wordlist.csv`. Copy **bảng cột** sang Google Sheet để chia việc.

Cấu trúc 3 lớp (đủ bộ):

1. **Âm đơn** — 12 nguyên âm: `a ă â e ê i o ô ơ u ư y`
2. **Âm đơn + dấu** — mỗi nguyên âm đủ 6 thanh (72 mục): a/á/à/ả/ã/ạ, ă/ắ/ằ/ẳ/ẵ/ặ, …
3. **Ghép 1–4 âm, từ có nghĩa** — âm đơn đó + 1, 2, 3 hoặc 4 âm khác (có dấu hoặc không), chỉ lấy từ tiếng Việt có nghĩa (1 tiếng)

| Lớp | Cột `nhom` | Số mục | Ví dụ |
|---|---|---|---|
| Âm đơn | `am_don` | 12 | a, ă, ô, ơ, y |
| Âm đơn + dấu | `am_don_dau` | 60 | á, ạ, ế, ữ, ỵ |
| Ghép 1 âm | `ghep_1` | 137 | ba, ăn, về, đi, cô |
| Ghép 2 âm | `ghep_2` | 83 | bác, đất, một, nhà |
| Ghép 3 âm | `ghep_3` | 51 | phải, sách, trời |
| Ghép 4 âm | `ghep_4` | 33 | không, người, nhanh |
| Ghép 5 (1 tiếng dài) | `ghep_5` | 1 | chuyện |

Mỗi nguyên âm đều có đủ 6 dấu và có từ có nghĩa ở mức ghép 1, 2, 3, 4.

Ví dụ 377 × 4 người × 3 lần = **4524 file wav**. Chia thu theo `nhom` rồi theo `am_goc`.

---

## 2. Việc cần chốt trước khi thu (nhóm trưởng)

Điền vào đây rồi gửi lại nhóm:

- Số thành viên: ___
- Tên viết không dấu dùng trong file (vd: `hoa`, `minh`, `an`, `tuan`): ___
- Số từ trong list thầy/nhóm: ___
- Deadline nộp file lên Drive: ___
- Link folder Drive: ___

Một bạn (gọi là **thủ thư dữ liệu**) tạo Google Sheet 2 tab:

**Tab A – Bảng từ:** import `data/wordlist_bang.csv`.

- Mỗi cột = một nguyên âm (`a ă â e ê i o ô ơ u ư y`)
- Từ trên xuống: 1 âm (đủ 6 dấu) → 2 âm → 3 âm → 4 âm
- Ô trống = nguyên âm đó không có thêm từ ở mức đó
- Mã file tra trong `data/wordlist.csv` (cột `text` / `code`)

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
| nặng | 5 | thật | `tha+t5` |

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
- đất → `d-a+t1` (đ → `d-`, â → `a+`, sắc → `1`)
- cơm → `coom2` (ơ → `oo`, huyền → `2`)
- người → `nguuooi2` (ư → `uu`, ơ → `oo`)
- được → `d-uuooc5` (đ → `d-`, ư → `uu`, ơ → `oo`, nặng → `5`)
- thật → `tha+t5` (â → `a+`, nặng → `5`)

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

`an_d-a+t1_03.wav`

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

Đưa thư mục wav cho bạn code. Bạn đó bỏ vào `data/wav/` (tên file khớp cột `code` trong `data/wordlist.csv`), chạy trích đặc trưng và huấn luyện.

Không gửi file giả (hình sin) khi demo với thầy.
