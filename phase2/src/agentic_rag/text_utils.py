from __future__ import annotations

import re
import unicodedata


PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
ASCII_DIGITS = "0123456789"
STOPWORDS = {
    "است",
    "هست",
    "این",
    "اون",
    "آن",
    "یک",
    "یه",
    "برای",
    "به",
    "از",
    "در",
    "را",
    "رو",
    "و",
    "یا",
    "من",
    "میخوام",
    "می",
    "خواهم",
    "کمک",
    "کنید",
    "چقدر",
    "چقدره",
    "کدام",
    "کدوم",
    "بهتر",
    "مناسب",
}


def normalize_text(text: object) -> str:
    if text is None:
        return ""
    value = unicodedata.normalize("NFKC", str(text))
    value = value.translate(str.maketrans(PERSIAN_DIGITS + ARABIC_DIGITS, ASCII_DIGITS * 2))
    value = value.replace("ي", "ی").replace("ك", "ک")
    value = re.sub(r"[\u200c\u200f\u202a-\u202e]", " ", value)
    value = re.sub(r"[^\w\s./-]+", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value).strip().lower()
    return value


def query_tokens(text: str, limit: int = 14) -> list[str]:
    tokens = []
    for tok in normalize_text(text).split():
        if len(tok) < 2 or tok in STOPWORDS:
            continue
        if tok not in tokens:
            tokens.append(tok)
        if len(tokens) >= limit:
            break
    return tokens


def contains_any(text: str, terms: list[str]) -> bool:
    norm = normalize_text(text)
    normalized_terms = [normalize_text(term) for term in terms]
    return any(term and term in norm for term in normalized_terms)


def rough_token_count(text: str) -> int:
    return max(1, len(normalize_text(text).split()))


def extract_parenthesized(text: str) -> list[str]:
    return [m.strip() for m in re.findall(r"\(([^)]+)\)", text) if m.strip()]
