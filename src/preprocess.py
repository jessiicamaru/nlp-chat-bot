"""
preprocess.py — Pipeline tiền xử lý tiếng Việt.

Module này là bản đóng gói lại (refactor) của Lab 03. Giữ nguyên tên hàm và
cấu trúc `config` của lab để kết quả có thể đối chiếu trực tiếp:

    normalize_basic -> segment_vi -> lowercase -> remove_numbers
                    -> remove_punctuation -> remove_stopwords

Điểm khác biệt duy nhất so với lab: `segment_vi` có cache, vì chatbot gọi
word_tokenize rất nhiều lần trên cùng một câu (mỗi lượt chat gọi lại pipeline).
"""

from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from pathlib import Path

from underthesea import word_tokenize

from config import DEFAULT_CONFIG, STOPWORDS_PATH


# ---------------------------------------------------------------------------
# 1. Chuẩn hóa cơ bản
# ---------------------------------------------------------------------------
def normalize_basic(text) -> str:
    """Xử lý missing, đưa về NFC, chuẩn hóa whitespace.

    NFC quan trọng với tiếng Việt: "Hòa" có thể được lưu ở dạng dựng sẵn
    (1 codepoint) hoặc tổ hợp (chữ + dấu rời). Không chuẩn hóa thì hai chuỗi
    trông giống hệt nhau trên màn hình nhưng khác nhau với máy -> TF-IDF coi
    chúng là hai term khác nhau.
    """
    if text is None:
        return ""
    # Bắt cả float('nan') của pandas mà không cần import pandas.
    if isinstance(text, float) and text != text:
        return ""

    text = str(text)
    text = unicodedata.normalize("NFC", text)
    text = text.replace("\u00A0", " ")   # non-breaking space
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ---------------------------------------------------------------------------
# 2. Tách từ tiếng Việt
# ---------------------------------------------------------------------------
@lru_cache(maxsize=20000)
def _segment_cached(text: str) -> str:
    return word_tokenize(text, format="text")


def segment_vi(text) -> str:
    """Tách từ ghép tiếng Việt, nối bằng dấu gạch dưới.

    "xử lý ngôn ngữ tự nhiên" -> "xử_lý ngôn_ngữ tự_nhiên"

    Đây là bước bắt buộc trước TF-IDF. Nếu bỏ qua, "học" trong "học máy" và
    "học" trong "đi học" bị gộp thành một term, làm nhiễu vector.
    """
    text = normalize_basic(text)
    if not text:
        return ""
    return _segment_cached(text)


# ---------------------------------------------------------------------------
# 3. Dấu câu
# ---------------------------------------------------------------------------
# Baseline của Lab 03 là r"[^\w\s]" -> quá aggressive.
# Nó phá huỷ chính những token mang thông tin trong tin tức:
#   "TP.HCM" -> "TP HCM"      (mất tên riêng)
#   "3,5%"   -> "3 5"         (mất số liệu)
#   "C++"    -> "C"           (mất tên công nghệ)
# Ở final project ta giữ dấu chấm/phẩy khi nằm GIỮA hai ký tự chữ-số,
# và giữ ký hiệu %, +, # khi dính liền token.
_PUNCT_KEEP_INNER = re.compile(
    r"(?<![\w])[^\w\s%+#]+|[^\w\s%+#]+(?![\w])",
    flags=re.UNICODE,
)


def remove_punctuation(text: str, keep_inner: bool = True) -> str:
    """Bỏ dấu câu.

    keep_inner=True  -> giữ dấu nằm giữa chữ/số (TP.HCM, 3,5%, C++)
    keep_inner=False -> baseline aggressive của Lab 03, dùng để so sánh A/B
    """
    if keep_inner:
        text = _PUNCT_KEEP_INNER.sub(" ", text)
    else:
        text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# 4. Số
# ---------------------------------------------------------------------------
def remove_numbers(text: str) -> str:
    """Bỏ token thuần số.

    Mặc định TẮT trong config retrieval: với corpus tin tức, con số
    ("30 ngày", "100 kỹ sư") thường chính là thứ người dùng đi tìm.
    """
    text = re.sub(r"\b\d+([.,]\d+)*\b", " ", text)
    return re.sub(r"\s+", " ", text).strip()


# ---------------------------------------------------------------------------
# 5. Stopwords
# ---------------------------------------------------------------------------
@lru_cache(maxsize=1)
def load_stopwords(path: str | None = None) -> frozenset[str]:
    """Nạp stopword list tiếng Việt (nguồn: github.com/stopwords/vietnamese-stopwords).

    File gốc ghi cụm từ bằng dấu cách ("bởi vì"). Sau `segment_vi` các cụm này
    thành "bởi_vì", nên ta nạp CẢ HAI dạng.
    """
    p = Path(path) if path else STOPWORDS_PATH
    if not p.exists():
        return frozenset()

    words: set[str] = set()
    for line in p.read_text(encoding="utf-8").splitlines():
        w = normalize_basic(line).lower()
        if not w:
            continue
        words.add(w)
        words.add(w.replace(" ", "_"))
    return frozenset(words)


def remove_stopwords(text: str, stopwords=None) -> str:
    """Lọc stopword ở mức token (token đã được nối bằng '_')."""
    if stopwords is None:
        stopwords = load_stopwords()
    tokens = [t for t in text.split() if t.lower() not in stopwords]
    return " ".join(tokens)


# ---------------------------------------------------------------------------
# 6. Pipeline tổng
# ---------------------------------------------------------------------------
def preprocess_vi(text, config: dict | None = None) -> str:
    """Ghép các building block theo đúng thứ tự của Lab 03.

    Thứ tự KHÔNG tùy tiện:
      - segment TRƯỚC lowercase: underthesea dùng chữ hoa để nhận diện tên riêng.
      - remove_numbers TRƯỚC remove_punctuation: cần "3,5%" còn nguyên để
        regex số bắt được, nếu bỏ dấu trước thì chỉ còn "3 5".
      - remove_stopwords CUỐI CÙNG: stopword list so khớp trên token sạch.
    """
    cfg = DEFAULT_CONFIG.copy()
    if config:
        cfg.update(config)

    text = normalize_basic(text)
    if not text:
        return ""

    if cfg.get("word_segment", True):
        text = segment_vi(text)

    if cfg.get("lowercase", True):
        text = text.lower()

    if cfg.get("remove_numbers", False):
        text = remove_numbers(text)

    if cfg.get("remove_punctuation", False):
        text = remove_punctuation(text, keep_inner=cfg.get("keep_inner_punct", True))

    if cfg.get("remove_stopwords", False):
        text = remove_stopwords(text)

    return re.sub(r"\s+", " ", text).strip()


def tokenize(text, config: dict | None = None) -> list[str]:
    """preprocess_vi + tách thành list token. Đây là input của vectorizer."""
    processed = preprocess_vi(text, config)
    return processed.split() if processed else []
