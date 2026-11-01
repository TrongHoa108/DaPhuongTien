"""Sinh wordlist: âm đơn, âm đơn + dấu, từ có nghĩa ghép 1–4 âm."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import WORDLIST_PATH, configure_console
from syllables_don import expand_syllables
from src.vni import (
    TONE_MARKS,
    VIET_ALPHABET,
    apply_tone,
    code_to_text,
    count_am,
    first_letter,
    letters_in_text,
    main_vowel_of,
    stem_from_code,
    text_to_code,
)

configure_console()

VOWELS = tuple("aăâeêioôơuưy")
CONSONANTS = tuple(ch for ch in VIET_ALPHABET if ch not in VOWELS)

# Từ đơn có nghĩa: âm đơn ghép thêm 1–4 âm (2–5 chữ cái).
COMBOS: list[str] = [
    # --- a : +1 ---
    "ba", "ca", "da", "ha", "la", "ma", "na", "ta", "va", "xa",
    "an", "ao", "ai", "au",
    # --- a : +2 ---
    "bác", "bà", "bả", "bã", "bạ", "bạn", "cát", "nam", "táo", "hát",
    "gà", "lá", "má", "ná", "tá", "tà", "tả", "tã", "tạ",
    "nhà", "sao", "mai", "tai", "nói",
    # --- a : +3 ---
    "cảng", "phải", "sách", "thật", "sạch", "mã", "càng", "toán",
    "hoặc", "tránh",
    # --- a : +4 ---
    "nhanh", "thánh", "quảng", "hoảng", "thoạt", "tránh",
    # --- ă : +1 ---
    "ăn",
    # --- ă : +2 ---
    "cắt", "mắt", "mặt", "nắm", "nằm", "cắn", "rắn", "bắn", "đặt",
    # --- ă : +3 ---
    "nắng", "nặng", "chắc", "lặng", "vắng", "hẳn", "vẵng",
    # --- ă : +4 ---
    "trắng", "chẳng", "quặng", "thẳng",
    # --- â : +1 ---
    "âu", "ây",
    # --- â : +2 ---
    "ấm", "đất", "cân", "cấn", "cần", "cẩn", "cận", "cấp", "mây", "đây",
    "\u0111\u1eabn",
    # --- â : +3 ---
    "thật", "đâu", "thầm", "gần", "dâng", "khẩn",
    "l\u1ea9y",
    # --- â : +4 ---
    "thuận", "khuẩn", "khuẩn",
    # --- e : +1 ---
    "em", "xe", "be", "he", "le", "me", "te", "ve", "de",
    # --- e : +2 ---
    "kém", "đẹp", "hét", "nén", "kem", "len", "den", "lẹ", "lẻ", "lẽ",
    "lé", "lè",
    # --- e : +3 ---
    "nghe", "quen", "khoe", "theo",
    # --- e : +4 ---
    "nghèo", "khoẻ",
    # --- ê : +1 ---
    "về", "ê", "bê", "dê", "kê", "lê", "mê", "ê",
    # --- ê : +2 ---
    "đến", "đêm", "ghế", "tên", "bên", "êm", "kể", "lễ", "hệ",
    # --- ê : +3 ---
    "thêm", "trên", "bệnh", "tiến", "viền",
    # --- ê : +4 ---
    "thiếc", "khiên", "triền",
    # --- i : +1 ---
    "đi", "im", "bi", "di", "hi", "li", "mi", "si", "ti", "vi", "xi", "in",
    # --- i : +2 ---
    "pin", "tim", "xin", "tin", "kín", "chị", "chỉ", "tí", "vì", "sĩ", "dị",
    # --- i : +3 ---
    "chín", "lịch", "thích", "minh",
    # --- i : +4 ---
    "chính", "thính", "nhìn",
    # --- o : +1 ---
    "to", "lo", "no", "ho", "vo", "xo", "do", "co", "bò",
    # --- o : +2 ---
    "con", "học", "có", "họ", "rõ", "ong", "tỏ", "đỏ", "gió", "già",
    # --- o : +3 ---
    "xong", "lòng", "hoặc", "hỏi", "nói",
    # --- o : +4 ---
    "hoảng", "ngoài", "thoạt",
    # --- ô : +1 ---
    "ô", "cô", "đô", "hô", "lô", "tô", "vô", "xô", "cố",
    # --- ô : +2 ---
    "một", "tốt", "hồn", "bốn", "cồ", "cổ", "cỗ", "cộng",
    # --- ô : +3 ---
    "chốn", "dồn", "tôi", "xuống",
    # --- ô : +4 ---
    "không", "chồng", "chuối",
    # --- ơ : +1 ---
    "ơ", "cơ", "mơ", "tơ", "lơ", "bơ", "cớ", "cờ", "cỡ", "mỡ",
    "mở", "đỡ",
    # --- ơ : +2 ---
    "cơm", "lớn", "sớm", "thợ", "ơn",
    # --- ơ : +3 ---
    "trời", "vượt", "hơn",
    # --- ơ : +4 ---
    "trước", "người", "những",
    # --- u : +1 ---
    "cu", "du", "hu", "lu", "mu", "ru", "tu", "vu", "xu",
    "cú", "dù", "củ", "lũ", "dụ",
    # --- u : +2 ---
    "lúc", "ung", "cũng", "đúng",
    # --- u : +3 ---
    "phút", "cũng", "đúng", "muốn",
    # --- u : +4 ---
    "chúng", "xuống", "chuối",
    # --- ư : +1 ---
    "ư", "hư", "tư", "cư", "cứ", "từ", "thử", "dữ", "tự",
    # --- ư : +2 ---
    "chữ", "cửa", "ưng", "ứng",
    # --- ư : +3 ---
    "vững", "hứng", "gửi", "mưa",
    # --- ư : +4 ---
    "những", "người", "trước",
    # --- y : +1 ---
    "y", "ý", "ly", "my", "ty", "kỳ", "kỹ", "mỹ", "tỵ",
    # --- y : +2 ---
    "máy", "tay", "quý",
    # --- y : +3 ---
    "thuỷ", "luỹ",
    # --- y : +4 ---
    "huyền", "khuya", "luyện", "khu\u1ef7u",
    # từ đề / âm kép còn thiếu
    "ba", "bác", "cảng", "đất", "cố", "ghế", "già", "gió", "nghĩ",
    "phải", "sách", "thật", "rượu", "nhiều", "chuyện", "thuế",
    "tiên", "biển", "tiếp", "được",
    # đủ 29 chữ cái: từ bắt đầu bằng từng chữ
    "an", "áo", "anh", "ảnh", "áng",
    "ăn", "ắt", "ăng",
    "âu", "ây", "ấm", "ấp", "ấn",
    "ba", "bé", "bi", "bò", "bơ", "bờ", "bác", "bạn", "bốn", "bên",
    "bệnh", "biển", "biếng", "bướng",
    "ca", "cô", "cơ", "cu", "cờ", "con", "cố", "cơm", "cũng", "chín",
    "chữ", "chúng", "chồng", "chuối", "chính", "chuyện",
    "da", "do", "du", "dê", "dờ", "dù", "dữ", "dị", "dâng", "dồn", "duyên",
    "đi", "đo", "đô", "đờ", "đất", "đến", "đêm", "đẹp", "đỏ", "được",
    "đúng", "đâu", "đường",
    "em", "eo",
    "êm", "ếch",
    "gà", "gờ", "ghế", "già", "gió", "gần", "gửi", "giếng",
    "ha", "ho", "hu", "hư", "hờ", "học", "hỏi", "hát", "hơn", "hồn",
    "hứng", "hoảng", "huyền",
    "im", "in", "ích",
    "kê", "ke", "kém", "kem", "kể", "kín", "khoa", "khoe", "không",
    "khiên", "khuỷu", "khuẩn",
    "la", "lá", "le", "lo", "ly", "lờ", "lê", "lớn", "lúc", "lễ",
    "lịch", "lòng", "luyện",
    "ma", "me", "mô", "mơ", "mờ", "máy", "một", "mặt", "mây", "mã",
    "minh", "mưa", "muống",
    "na", "no", "nơ", "nờ", "nói", "nắm", "nằm", "nhà", "nhìn",
    "nặng", "nắng", "người", "những", "nhanh", "ngoài",
    "ong", "om", "óc",
    "ôm", "ồn", "ốm", "ông",
    "ơn", "ớt",
    "um", "ùa",
    "ăng",
    "pê", "pa", "pin", "pha", "phe", "phải", "phút", "phim", "phẳng",
    "qua", "quả", "que", "quen", "quý", "quảng", "quặng",
    "ra", "ru", "rõ", "rờ", "rắn", "rơi", "rượu", "rừng", "riêng",
    "sa", "so", "si", "sờ", "sớm", "sao", "sách", "sạch", "sông", "suông",
    "ta", "to", "tu", "tư", "tờ", "tơ", "tốt", "tim", "tên", "tôi", "tay",
    "trời", "trên", "thêm", "thật", "trước", "thánh", "trắng", "tránh",
    "ung", "uống",
    "ưa", "ưng", "ứng", "ức",
    "va", "ve", "vô", "về", "vờ", "vì", "vào", "vui", "vững", "vượt",
    "viếng",
    "xa", "xe", "xo", "xô", "xờ", "xin", "xong", "xác", "xuống",
    "yên", "yếu",
]

# Sửa vài từ dễ lệch Unicode.
_FIX = {
    "cơm": "c\u1eddm",
    "lấy": "l\u1ea9y",
    "dẫn": "\u0111\u1eabn",
    "khoẻ": "kho\u1ebb",
    "kỹ": "k\u1ef7",
    "khuỷu": "khu\u1ef7u",
}


def _clean(text: str) -> str:
    text = text.strip()
    return _FIX.get(text, text)


def isolated_letters() -> list[str]:
    """29 chữ cái; nguyên âm đủ 6 dấu."""
    items = []
    for ch in VIET_ALPHABET:
        if ch in VOWELS:
            for tone in range(6):
                items.append(apply_tone(ch, tone))
        else:
            items.append(ch)
    return items


def classify(text: str) -> dict:
    so_am = count_am(text)
    ghep = max(so_am - 1, 0)
    tone = stem_from_code(text_to_code(text))[1]
    vowel = main_vowel_of(text) or (letters_in_text(text)[0] if letters_in_text(text) else "")
    chu_dau = first_letter(text)
    if so_am == 1:
        nhom = "am_don" if tone == 0 else "am_don_dau"
    else:
        nhom = f"ghep_{ghep}"
    return {
        "text": text,
        "so_am": so_am,
        "ghep": ghep,
        "tone": tone,
        "tone_name": TONE_MARKS[tone],
        "am_goc": vowel,
        "chu_dau": chu_dau,
        "nhom": nhom,
        "chu_ghep": "".join(letters_in_text(text)),
    }


def main() -> None:
    texts: list[str] = []
    seen: set[str] = set()
    for raw in isolated_letters() + expand_syllables() + COMBOS:
        text = _clean(raw)
        if not text or text in seen:
            continue
        seen.add(text)
        texts.append(text)

    rows = []
    failed = []
    for i, text in enumerate(texts, start=1):
        meta = classify(text)
        code = text_to_code(text)
        back = code_to_text(code)
        if " " in text:
            continue
        if back != text:
            failed.append((text, code, back))
            continue
        if meta["ghep"] > 4:
            # chuyện / nhiều… vẫn giữ, đánh ghep=4+ nếu quá 5 chữ
            pass
        rows.append(
            {
                "stt": i,
                "text": text,
                "code": code,
                "tone": meta["tone"],
                "tone_name": meta["tone_name"],
                "am_goc": meta["am_goc"],
                "chu_dau": meta["chu_dau"],
                "so_am": meta["so_am"],
                "ghep": min(meta["ghep"], 4) if meta["so_am"] <= 5 else meta["ghep"],
                "nhom": meta["nhom"],
                "chu_ghep": meta["chu_ghep"],
                "note": (
                    "Âm đơn"
                    if meta["nhom"] == "am_don"
                    else "Âm đơn có dấu"
                    if meta["nhom"] == "am_don_dau"
                    else f"Ghép {meta['ghep']} âm — từ có nghĩa"
                ),
            }
        )

    errors = []
    if failed:
        print(f"Bỏ {len(failed)} mục không mã hóa được (không phải hai tiếng).")

    by_vowel_ghep: dict[tuple[str, int], int] = {}
    isolated: dict[tuple[str, int], int] = {}
    for r in rows:
        v, g, t = r["am_goc"], r["ghep"], r["tone"]
        by_vowel_ghep[(v, g)] = by_vowel_ghep.get((v, g), 0) + 1
        if r["so_am"] == 1:
            isolated[(v, t)] = isolated.get((v, t), 0) + 1

    miss_iso = [f"{v}{t}" for v in VOWELS for t in range(6) if (v, t) not in isolated]
    if miss_iso:
        errors.append(f"Thiếu nguyên âm + dấu: {miss_iso}")

    present_letters = {r["chu_dau"] for r in rows if r["so_am"] == 1}
    miss_abc = [ch for ch in VIET_ALPHABET if ch not in present_letters]
    if miss_abc:
        errors.append(f"Thiếu chữ cái 1 âm: {miss_abc}")

    by_first_len: dict[tuple[str, int], int] = {}
    for r in rows:
        by_first_len[(r["chu_dau"], r["so_am"])] = (
            by_first_len.get((r["chu_dau"], r["so_am"]), 0) + 1
        )
    # q luôn đi với u (≥3 chữ); y/â/e/ơ ít từ đầu 2–3 chữ
    skip_len = {("q", 2), ("y", 2), ("â", 3), ("e", 3), ("ơ", 3)}
    miss_len = [
        f"{ch}+{n}âm"
        for ch in VIET_ALPHABET
        for n in (2, 3)
        if (ch, n) not in skip_len and by_first_len.get((ch, n), 0) < 1
    ]
    if miss_len:
        errors.append(f"Thiếu từ 2–3 âm theo chữ cái: {miss_len}")

    if errors:
        raise SystemExit("\n".join(errors))

    WORDLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "stt",
        "text",
        "code",
        "tone",
        "tone_name",
        "am_goc",
        "chu_dau",
        "so_am",
        "ghep",
        "nhom",
        "chu_ghep",
        "note",
    ]
    out_path = WORDLIST_PATH
    try:
        probe = out_path.open("a", encoding="utf-8-sig")
        probe.close()
    except PermissionError:
        out_path = WORDLIST_PATH.with_name("wordlist_new.csv")

    with out_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Đã ghi {len(rows)} mục vào {out_path}")
    print("Bảng chữ cái 29 cột:", " ".join(VIET_ALPHABET))
    print("Lớp 1 — âm đơn + dấu (nguyên âm 6 thanh, phụ âm 1 chữ)")
    print("Lớp 2 — từ có nghĩa 2–5 âm, xếp theo chữ cái đầu")
    counts = {}
    for r in rows:
        counts[r["nhom"]] = counts.get(r["nhom"], 0) + 1
    for k in sorted(counts):
        print(f"  {k}: {counts[k]}")

    bang_path = write_bang(rows)
    print(f"Bảng cột: {bang_path}")


def write_bang(rows: list[dict]) -> Path:
    """29 chữ cái = 29 cột; hàng từ trên xuống: 1 âm → 2 âm → 3 âm → …"""
    by_letter_len: dict[str, dict[int, list[dict]]] = {
        ch: {n: [] for n in range(1, 7)} for ch in VIET_ALPHABET
    }
    for r in rows:
        ch = r["chu_dau"]
        n = r["so_am"]
        if ch not in by_letter_len or n not in by_letter_len[ch]:
            continue
        by_letter_len[ch][n].append(r)

    for ch in VIET_ALPHABET:
        by_letter_len[ch][1].sort(key=lambda r: r["tone"])
        for n in range(2, 7):
            by_letter_len[ch][n].sort(key=lambda r: (r["tone"], r["text"]))

    bang_rows: list[dict] = []
    for n in range(1, 7):
        height = max(len(by_letter_len[ch][n]) for ch in VIET_ALPHABET)
        if height == 0:
            continue
        for i in range(height):
            row = {"so_am": n, "hang": i + 1}
            if n == 1:
                row["thanh"] = TONE_MARKS[i] if i <= 5 else ""
            else:
                row["thanh"] = ""
            for ch in VIET_ALPHABET:
                items = by_letter_len[ch][n]
                row[ch] = items[i]["text"] if i < len(items) else ""
            bang_rows.append(row)

    fields = ["so_am", "hang", "thanh", *VIET_ALPHABET]
    bang_path = WORDLIST_PATH.with_name("wordlist_bang.csv")
    with bang_path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter=";")
        writer.writeheader()
        writer.writerows(bang_rows)

    xlsx_path = WORDLIST_PATH.with_name("BANG_THU_AM.xlsx")
    root_xlsx = WORDLIST_PATH.parents[1] / "BANG_THU_AM.xlsx"
    alt_xlsx = WORDLIST_PATH.parents[1] / "BANG_CHU_CAI_29.xlsx"
    try:
        import pandas as pd

        df = pd.DataFrame(bang_rows, columns=fields)
        for path in (xlsx_path, root_xlsx, alt_xlsx):
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                df.to_excel(writer, sheet_name="Don tu 29 chu", index=False)
                ws = writer.sheets["Don tu 29 chu"]
                ws.freeze_panes = "D2"
                ws.auto_filter.ref = ws.dimensions
                for col in ws.columns:
                    letter = col[0].column_letter
                    width = max((len(str(c.value or "")) for c in col), default=8)
                    ws.column_dimensions[letter].width = min(max(width + 2, 8), 16)
    except Exception as exc:
        print(f"Không ghi được xlsx ({exc}). Dùng CSV dấu chấm phẩy.")
    return bang_path


if __name__ == "__main__":
    main()
