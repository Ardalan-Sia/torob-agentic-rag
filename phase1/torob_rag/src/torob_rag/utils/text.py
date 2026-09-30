from __future__ import annotations
import re

_PERSIAN_ARABIC_MAP = str.maketrans({
    "ي": "ی",
    "ك": "ک",
    "ة": "ه",
    "ؤ": "و",
    "إ": "ا",
    "أ": "ا",
    "آ": "ا",
})

_DIGIT_MAP = str.maketrans({
    "۰":"0","۱":"1","۲":"2","۳":"3","۴":"4","۵":"5","۶":"6","۷":"7","۸":"8","۹":"9",
})

def normalize_text(s: str) -> str:
    if s is None:
        return ""
    s = str(s)
    s = s.translate(_PERSIAN_ARABIC_MAP)
    s = s.translate(_DIGIT_MAP)
    s = re.sub(r"\s+", " ", s).strip()
    return s
