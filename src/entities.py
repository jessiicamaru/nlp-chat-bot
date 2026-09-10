"""
entities.py — Trích xuất thực thể và slot từ câu người dùng.

Kế thừa Lab 01: dùng `ner()` / `pos_tag()` của underthesea cho thực thể mở
(người, địa điểm, tổ chức) và dùng Regex cho các định dạng cố định
(email, số điện thoại, ngày tháng, URL, số + đơn vị).

Nguyên tắc chọn công cụ — đúng như kết luận của Lab 01:
    Định dạng cố định, viết được luật rõ ràng  -> Regex.
    Thực thể mở, phụ thuộc ngữ cảnh            -> model NER.

Ví dụ cụ thể trong đồ án này:
    "0905123456"      -> Regex thắng: luật rõ ràng, chính xác 100%, không cần model.
    "Mai" trong câu   -> cần NER: "Mai tôi đi Đà Nẵng" (trạng ngữ thời gian)
                        khác "Mai đang học NLP" (tên người).
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

from underthesea import ner

from preprocess import normalize_basic

# ---------------------------------------------------------------------------
# Regex cho các định dạng cố định (Lab 01 — PHẦN B)
# ---------------------------------------------------------------------------
PATTERNS = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "phone": re.compile(r"\b(?:0|\+84)\d{8,10}\b"),
    "url": re.compile(r"https?://[^\s<>\"]+"),
    # Ngày: 12/8/2026, 12-8-2026, 2/9
    "date": re.compile(r"\b\d{1,2}[/-]\d{1,2}(?:[/-]\d{2,4})?\b"),
    # Tiền/số lượng có đơn vị: "10.000 tỷ đồng", "30 ngày", "3,5%"
    # Kết thúc bằng (?!\w) chứ KHÔNG phải \b: sau "%" là khoảng trắng hoặc hết
    # chuỗi, mà "%" cũng là ký tự non-word nên \b không bao giờ khớp ở đó
    # -> "3,5%" bị bỏ sót âm thầm. Đây là một case error analysis của nhóm.
    "quantity": re.compile(
        r"\b\d+(?:[.,]\d+)*\s*(?:%|tỷ|triệu|nghìn|ngàn|đồng|usd|ngày|giờ|tháng|năm|km|kg|người)(?!\w)",
        re.IGNORECASE,
    ),
}


# ---------------------------------------------------------------------------
# Từ khóa chuyên mục
# ---------------------------------------------------------------------------
# Ánh xạ cách người dùng NÓI -> tên chuyên mục trong corpus.
# Người dùng gõ "công nghệ" / "cong nghe" / "tech" đều phải ra "Công nghệ".
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Du lịch": ["du lịch", "du lich", "travel", "đi chơi", "phượt", "tour", "nghỉ dưỡng"],
    "Kinh doanh": ["kinh doanh", "kinh te", "kinh tế", "business", "tài chính", "chứng khoán",
                   "doanh nghiệp", "thị trường"],
    "Công nghệ": ["công nghệ", "cong nghe", "tech", "số hóa", "công nghệ thông tin", "ai",
                  "phần mềm", "điện thoại", "máy tính"],
    "Sức khỏe": ["sức khỏe", "suc khoe", "y tế", "bệnh", "health", "dinh dưỡng", "bác sĩ"],
    "Giáo dục": ["giáo dục", "giao duc", "học sinh", "sinh viên", "tuyển sinh", "thi cử",
                 "đại học", "trường học"],
    "Thể thao": ["thể thao", "the thao", "bóng đá", "sport", "cầu thủ", "giải đấu"],
    "Khoa học": ["khoa học", "khoa hoc", "nghiên cứu", "science", "vũ trụ", "phát minh"],
    "Đời sống": ["đời sống", "doi song", "gia đình", "nhà cửa", "ẩm thực", "món ăn"],
}


def _strip_accents(text: str) -> str:
    """Bỏ dấu tiếng Việt để so khớp cả khi người dùng gõ không dấu."""
    nfd = unicodedata.normalize("NFD", text)
    stripped = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    return stripped.replace("đ", "d").replace("Đ", "D")


@dataclass
class ExtractedInfo:
    """Kết quả trích xuất của một câu người dùng."""

    text: str
    category: str | None = None
    persons: list[str] = field(default_factory=list)
    locations: list[str] = field(default_factory=list)
    organizations: list[str] = field(default_factory=list)
    regex_matches: dict[str, list[str]] = field(default_factory=dict)

    @property
    def all_entities(self) -> list[str]:
        return self.persons + self.locations + self.organizations

    def summary(self) -> str:
        parts = []
        if self.category:
            parts.append(f"chuyên mục={self.category}")
        if self.persons:
            parts.append(f"PER={self.persons}")
        if self.locations:
            parts.append(f"LOC={self.locations}")
        if self.organizations:
            parts.append(f"ORG={self.organizations}")
        for k, v in self.regex_matches.items():
            parts.append(f"{k}={v}")
        return "; ".join(parts) if parts else "(không có thực thể)"


# ---------------------------------------------------------------------------
# Trích xuất
# ---------------------------------------------------------------------------
def detect_category(text: str) -> str | None:
    """Nhận diện chuyên mục người dùng nhắc tới.

    So khớp trên bản KHÔNG DẤU để chịu được "cong nghe", "du lich".
    Ưu tiên từ khóa dài nhất: "công nghệ thông tin" phải thắng "công nghệ",
    nếu không thứ tự duyệt dict sẽ quyết định kết quả một cách tùy tiện.
    """
    if not text:
        return None
    haystack = _strip_accents(normalize_basic(text).lower())

    best: tuple[int, str] | None = None
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            needle = _strip_accents(kw.lower())
            # Bao bởi ranh giới từ để "ai" không khớp trong "hai", "mai".
            if re.search(rf"(?<!\w){re.escape(needle)}(?!\w)", haystack):
                if best is None or len(needle) > best[0]:
                    best = (len(needle), category)
    return best[1] if best else None


def extract_regex(text: str) -> dict[str, list[str]]:
    """Bắt các định dạng cố định. Chỉ giữ nhóm có kết quả."""
    found: dict[str, list[str]] = {}
    for name, pattern in PATTERNS.items():
        matches = pattern.findall(text)
        if matches:
            # findall với group trả tuple -> ép về str.
            found[name] = [m if isinstance(m, str) else m[0] for m in matches]
    return found


def extract_named_entities(text: str) -> tuple[list[str], list[str], list[str]]:
    """Gom token BIO của underthesea thành thực thể hoàn chỉnh.

    `ner()` trả từng token với nhãn B-PER / I-PER / O. Ta phải ghép
    B- + các I- theo sau lại, nếu không "Nguyễn Văn An" sẽ thành ba người.
    """
    persons: list[str] = []
    locations: list[str] = []
    organizations: list[str] = []

    try:
        tagged = ner(text)
    except Exception:
        # NER lỗi không được làm sập chatbot — vẫn còn regex và retrieval.
        return persons, locations, organizations

    buffer: list[str] = []
    buffer_type: str | None = None

    def flush():
        nonlocal buffer, buffer_type
        if buffer and buffer_type:
            entity = " ".join(buffer)
            if buffer_type == "PER":
                persons.append(entity)
            elif buffer_type == "LOC":
                locations.append(entity)
            elif buffer_type == "ORG":
                organizations.append(entity)
        buffer, buffer_type = [], None

    for item in tagged:
        token, tag = item[0], item[-1]
        if tag.startswith("B-"):
            flush()
            buffer, buffer_type = [token], tag[2:]
        elif tag.startswith("I-") and buffer_type == tag[2:]:
            buffer.append(token)
        else:
            flush()
    flush()

    return persons, locations, organizations


def extract(text: str, use_ner: bool = True) -> ExtractedInfo:
    """Trích xuất đầy đủ cho một câu người dùng."""
    text = normalize_basic(text)
    info = ExtractedInfo(text=text)

    info.category = detect_category(text)
    info.regex_matches = extract_regex(text)

    if use_ner and text:
        info.persons, info.locations, info.organizations = extract_named_entities(text)

    return info


def expand_query(text: str, info: ExtractedInfo | None = None) -> str:
    """Nhân đôi thực thể trong query để tăng trọng số khi truy hồi.

    "bạn biết gì về Sa Pa" -> "bạn biết gì về Sa Pa Sa Pa"

    Lặp lại tên riêng làm tăng tf của term đó trong vector query, kéo cosine
    similarity về phía bài báo thực sự nói về địa danh đó thay vì bài chỉ
    trùng các từ chung chung như "biết", "gì".
    """
    info = info or extract(text)
    entities = info.all_entities
    if not entities:
        return text
    return text + " " + " ".join(entities)
