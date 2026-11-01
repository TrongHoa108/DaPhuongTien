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


def _has_coda(text: str) -> bool:
    return text.endswith(("ng", "nh", "ch", "p", "t", "c", "m", "n"))


def apply_tone(text: str, tone: int) -> str:
    if tone == 0:
        return text
    lower = text.lower()
    if "uyê" in lower:
        target = "ê"
    elif "uya" in lower or "oai" in lower or "oay" in lower:
        target = "a"
    elif "uy" in lower:
        target = "y"
    elif "ua" in lower and not lower.startswith("qu") and "uya" not in lower:
        target = "a" if _has_coda(lower) else "u"
    elif "iê" in lower or "yê" in lower:
        target = "ê"
    elif "uô" in lower:
        target = "ô"
    elif "ươ" in lower:
        target = "ơ"
    elif "oa" in lower:
        target = "a" if _has_coda(lower) else "o"
    elif "oe" in lower:
        target = "e"
    elif "ây" in lower or "âu" in lower:
        target = "â"
    elif any(d in lower for d in ("ai", "ao", "au", "ay")):
        target = "a"
    else:
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


_CHAR_TO_BASE_TONE: dict[str, tuple[str, int]] = {}
for _base, _series in VOWEL_TONES.items():
    for _tone, _ch in enumerate(_series):
        _CHAR_TO_BASE_TONE[_ch] = (_base, _tone)

_LETTER_TO_CODE = {
    "â": "a+",
    "ô": "o+",
    "ê": "e+",
    "ă": "aa",
    "ơ": "oo",
    "ư": "uu",
    "đ": "d-",
}

# 29 chữ cái tiếng Việt (không gồm f, j, w, z — chỉ dùng trong từ mượn)
VIET_ALPHABET = tuple("aăâbcdđeêghiklmnoôơpqrstuưvxy")
VIET_DIGRAPHS = ("ngh", "ch", "gh", "gi", "kh", "ng", "nh", "ph", "qu", "th", "tr")


def letters_in_text(text: str) -> list[str]:
    """Bỏ thanh, giữ chữ cái gốc: Đất → ['đ', 'â', 't']."""
    out: list[str] = []
    for ch in text.strip().lower():
        if ch in _CHAR_TO_BASE_TONE:
            out.append(_CHAR_TO_BASE_TONE[ch][0])
        elif ch in VIET_ALPHABET:
            out.append(ch)
    return out


def count_am(text: str) -> int:
    """Số âm đơn (chữ cái tiếng Việt) sau khi bỏ thanh."""
    return len(letters_in_text(text))


def first_letter(text: str) -> str:
    """Chữ cái đầu (đã bỏ thanh): bác → b; đất → đ; ăn → ă."""
    letters = letters_in_text(text)
    return letters[0] if letters else ""


def digraphs_in_text(text: str) -> list[str]:
    body = "".join(letters_in_text(text))
    found = []
    i = 0
    while i < len(body):
        hit = next((d for d in VIET_DIGRAPHS if body.startswith(d, i)), None)
        if hit:
            found.append(hit)
            i += len(hit)
        else:
            i += 1
    return found


def onset_of(text: str) -> str:
    body = "".join(letters_in_text(text))
    for d in VIET_DIGRAPHS:
        if body.startswith(d):
            return d
    return body[:1]


def coda_of(text: str) -> str:
    body = "".join(letters_in_text(text))
    for ending in ("ng", "nh", "ch", "p", "t", "c", "m", "n"):
        if body.endswith(ending):
            return ending
    return ""


def main_vowel_of(text: str) -> str:
    body = "".join(letters_in_text(text))
    if "uyê" in body or "yê" in body or "iê" in body:
        return "ê"
    if "uya" in body or "oai" in body or "oay" in body:
        return "a"
    if "uy" in body:
        return "y"
    for v in TONE_VOWEL_PRIORITY:
        if v in body:
            return v
    return ""


def text_to_code(text: str) -> str:
    """ba → ba0; đất → d-a+t1; người → nguuooi2."""
    tone = 0
    bases: list[str] = []
    for ch in text.strip().lower():
        if ch in _CHAR_TO_BASE_TONE:
            base, t = _CHAR_TO_BASE_TONE[ch]
            if t:
                tone = t
            bases.append(base)
        else:
            bases.append(ch)
    body = "".join(_LETTER_TO_CODE.get(ch, ch) for ch in bases)
    return f"{body}{tone}"


def parse_wav_name(filename: str) -> dict:
    """
    {nguoi}_{ma_tu}_{lan}.wav
    Ví dụ: hoa_bac1_02.wav, minh_d-a+t1_01.wav, an_co+1_03.wav
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
