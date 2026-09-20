"""Quy ước mã hóa từ có dấu kiểu VNI (theo đề bài)."""

from __future__ import annotations

TONE_MARKS = {
    1: "sắc",
    2: "huyền",
    3: "hỏi",
    4: "ngã",
    5: "nặng",
    0: "không",
}

# Ánh xạ nguyên âm đã bỏ dấu → (không, sắc, huyền, hỏi, ngã, nặng)
VOWEL_TONES = {
    "a": "aáàảãạ",
    "ă": "ăắằẳẵặ",
    "â": "âấầẩẫậ",
    "e": "eéèẻẽẹ",
    "ê": "êếềểễệ",
    "i": "iíìỉĩị",
    "o": "oóòỏõọ",
    "ô": "ôốồổỗộ",
    "ơ": "ơớờởỡợ",
    "u": "uúùủũụ",
    "ư": "ưứừửữự",
    "y": "yýỳỷỹỵ",
}

# Thứ tự ưu tiên đặt thanh (vần tiếng Việt)
TONE_VOWEL_PRIORITY = ("ê", "ô", "ơ", "â", "ă", "ư", "e", "o", "a", "i", "u", "y")


def stem_from_code(code: str) -> tuple[str, int]:
    """bac1 → ('bac', 1); ba0 → ('ba', 0)."""
    code = code.strip()
    if not code:
        raise ValueError("Mã từ rỗng")
    tone = 0
    body = code
    if code[-1].isdigit():
        tone = int(code[-1])
        body = code[:-1]
    return body, tone


def decode_letters(body: str) -> str:
    """aa, a+, d- → ă, â, đ (chưa gắn thanh)."""
    out = []
    i = 0
    while i < len(body):
        if i + 1 < len(body) and body[i + 1] == "+":
            out.append({"a": "â", "o": "ô", "e": "ê"}.get(body[i], body[i] + "+"))
            i += 2
            continue
        if i + 1 < len(body) and body[i : i + 2] in ("aa", "oo", "uu"):
            out.append({"aa": "ă", "oo": "ơ", "uu": "ư"}[body[i : i + 2]])
            i += 2
            continue
        if i + 1 < len(body) and body[i : i + 2] == "d-":
            out.append("đ")
            i += 2
            continue
        out.append(body[i])
        i += 1
    return "".join(out)


def apply_tone(text: str, tone: int) -> str:
    if tone == 0:
        return text
    lower = text.lower()
    target = None
    for v in TONE_VOWEL_PRIORITY:
        if v in lower:
            target = v
            break
    if target is None:
        return text
    idx = lower.rfind(target)
    marked = VOWEL_TONES[target][tone]
    return text[:idx] + marked + text[idx + len(target) :]


def code_to_text(code: str) -> str:
    body, tone = stem_from_code(code)
    return apply_tone(decode_letters(body), tone)


def parse_wav_name(filename: str) -> dict:
    """
    {nguoi}_{ma_tu}_{lan}.wav
    Ví dụ: hoa_bac1_02.wav, minh_d-at5_01.wav, an_ca+1_03.wav
    """
    stem = filename.rsplit(".", 1)[0]
    parts = stem.split("_")
    if len(parts) < 3:
        raise ValueError(
            f"Tên file không đúng quy ước {{nguoi}}_{{ma_tu}}_{{lan}}.wav: {filename}"
        )
    speaker = parts[0]
    take = parts[-1]
    code = "_".join(parts[1:-1])
    return {
        "speaker": speaker,
        "word_code": code,
        "take": take,
        "word_text": code_to_text(code),
    }
