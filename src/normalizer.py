"""
normalizer.py — Chuẩn hóa teencode / viết tắt tiếng Việt trước khi tách từ.

## Vấn đề

Người Việt chat gõ rất khác văn viết chuẩn:

    "bt gì về vụ iphone k b"   ->  "biết gì về vụ iphone không bạn"
    "t thấy đc đấy"            ->  "tôi thấy được đấy"

Nếu đưa thẳng câu teencode vào `word_tokenize`, mọi thứ hỏng theo dây chuyền:
token "k", "đc", "bt" đều là OOV với index đã dựng trên văn bản báo chí, nên
vector query gần như rỗng và bot trả lời trượt.

Đây là lỗi NẶNG HƠN trường hợp không dấu đã xử lý trước đó: không dấu ít nhất
còn giữ nguyên âm tiết, còn teencode thì thay hẳn mặt chữ ("k" và "không"
không có ký tự nào chung).

## Cách tiếp cận: HỌC từ điển từ dữ liệu, không hardcode

Thay vì tự ngồi liệt kê vài trăm từ teencode (vừa thiếu, vừa mang thiên kiến
cá nhân), ta **học** bảng ánh xạ từ corpus ViLexNorm — 10.467 cặp câu
(teencode -> chuẩn) do con người gán nhãn.

Thuật toán:

  1. Với mỗi cặp câu, nếu số token hai bên BẰNG NHAU thì căn theo vị trí
     (79.5% số cặp thỏa điều kiện này — đủ để học).
  2. Đếm mọi ánh xạ `a -> b` tại các vị trí có thay đổi.
  3. Chỉ giữ ánh xạ `a -> b` khi thỏa CẢ BA điều kiện an toàn:
       - `a` xuất hiện đủ nhiều (>= min_count) -> tránh học từ nhiễu;
       - `a` THƯỜNG bị đổi (tỷ lệ đổi >= min_change_rate) -> đây là chốt chặn
         quan trọng nhất, nếu thiếu thì các từ thường như "cả", "mà" sẽ bị
         thay bừa chỉ vì đôi khi chúng xuất hiện ở vị trí có thay đổi;
       - `b` chiếm ưu thế rõ rệt trong các đích của `a` (>= min_dominance)
         -> tránh chọn bừa khi `a` mơ hồ.

Ví dụ về sự mơ hồ có thật trong dữ liệu: `t -> tôi` (889 lần) và
`t -> tao` (132 lần). Ta chọn "tôi" vì chiếm ưu thế, và chấp nhận sai ở
những câu vốn nói "tao" — một đánh đổi được ghi nhận, không giấu.

## Giới hạn

Chuẩn hóa ở mức TỪ ĐƠN LẺ, không xét ngữ cảnh. "m" có thể là "mày" hoặc
"mình" tùy câu, nhưng ta luôn chọn một đích. Muốn tốt hơn phải dùng mô hình
seq2seq có ngữ cảnh — nằm ngoài phạm vi kỹ thuật của đồ án.

Nguồn dữ liệu: ViLexNorm (Nguyen et al., EACL 2024), giấy phép CC BY-NC-SA 4.0.
https://github.com/ngxtnhi/ViLexNorm
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from config import RESOURCES_DIR
from preprocess import has_diacritics, normalize_basic, strip_accents

LEXICON_PATH = RESOURCES_DIR / "teencode_lexicon.json"
VILEXNORM_DIR = RESOURCES_DIR / "vilexnorm"

# Lặp ký tự để nhấn mạnh: "đẹppppp" -> "đẹp", "hayyy" -> "hay".
# Gộp 3 lần trở lên về 1. KHÔNG gộp 2 lần vì tiếng Việt có phụ âm đôi hợp lệ.
# CHỈ gộp CHỮ CÁI: bản đầu dùng (.) nên gộp cả chữ số — "15.000" -> "15.0",
# "2000" -> "20" — làm hỏng mọi câu hỏi có con số (phát hiện khi làm RAG, docs/07).
_ELONGATION = re.compile(r"([^\W\d_])\1{2,}", re.UNICODE)

# Token là dấu câu / emoticon thuần -> giữ nguyên, không tra từ điển.
_PUNCT_ONLY = re.compile(r"^[^\w]+$", re.UNICODE)


# ---------------------------------------------------------------------------
# Học từ điển
# ---------------------------------------------------------------------------
def learn_lexicon(
    pairs: list[tuple[str, str]],
    min_count: int = 4,
    min_change_rate: float = 0.5,
    min_dominance: float = 0.5,
) -> dict[str, str]:
    """Học bảng ánh xạ teencode -> chuẩn từ các cặp câu đã gán nhãn.

    `pairs` là list các cặp (câu gốc, câu đã chuẩn hóa).
    """
    seen: Counter = Counter()                          # số lần thấy token a
    changed: defaultdict[str, Counter] = defaultdict(Counter)   # a -> {b: số lần}

    for original, normalized in pairs:
        src = normalize_basic(original).split()
        tgt = normalize_basic(normalized).split()
        # Chỉ học từ cặp căn được theo vị trí.
        if len(src) != len(tgt):
            continue
        for a, b in zip(src, tgt):
            a_low, b_low = a.lower(), b.lower()
            seen[a_low] += 1
            if a_low != b_low:
                changed[a_low][b_low] += 1

    lexicon: dict[str, str] = {}
    for a, targets in changed.items():
        total_changed = sum(targets.values())
        if total_changed < min_count:
            continue
        # Chốt chặn quan trọng nhất: a phải THƯỜNG bị đổi.
        if total_changed / seen[a] < min_change_rate:
            continue
        best, best_count = targets.most_common(1)[0]
        if best_count / total_changed < min_dominance:
            continue
        # Đích phải khác nguồn và không rỗng.
        if best and best != a:
            lexicon[a] = best

    return lexicon


def load_vilexnorm(split: str = "train") -> list[tuple[str, str]]:
    """Nạp một split của ViLexNorm thành list cặp (gốc, chuẩn)."""
    import pandas as pd

    path = VILEXNORM_DIR / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Không tìm thấy {path}. Tải ViLexNorm về data/resources/vilexnorm/ trước."
        )
    df = pd.read_csv(path).dropna(subset=["original", "normalized"])
    return list(zip(df["original"].astype(str), df["normalized"].astype(str)))


def build_and_save_lexicon(**kwargs) -> dict[str, str]:
    """Học từ điển trên split train rồi lưu ra JSON."""
    lexicon = learn_lexicon(load_vilexnorm("train"), **kwargs)
    LEXICON_PATH.write_text(
        json.dumps(lexicon, ensure_ascii=False, indent=1, sort_keys=True),
        encoding="utf-8",
    )
    return lexicon


# ---------------------------------------------------------------------------
# Áp dụng
# ---------------------------------------------------------------------------
class TeencodeNormalizer:
    """Chuẩn hóa teencode ở mức token, chạy TRƯỚC bước tách từ."""

    def __init__(self, lexicon: dict[str, str] | None = None, collapse_elongation: bool = True):
        if lexicon is None:
            lexicon = self._load_default()
        self.lexicon = lexicon
        self.collapse_elongation = collapse_elongation

    @staticmethod
    def _load_default() -> dict[str, str]:
        if LEXICON_PATH.exists():
            return json.loads(LEXICON_PATH.read_text(encoding="utf-8"))
        return {}

    def normalize_token(self, token: str) -> str:
        """Chuẩn hóa một token. Trả về chính nó nếu không có gì để sửa."""
        if not token or _PUNCT_ONLY.match(token):
            return token

        low = token.lower()

        # 1. Tra thẳng từ điển.
        if low in self.lexicon:
            return self.lexicon[low]

        # 2. Gộp ký tự lặp rồi tra lại: "khummm" -> "khum" -> "không".
        if self.collapse_elongation:
            collapsed = _ELONGATION.sub(r"\1", low)
            if collapsed != low:
                if collapsed in self.lexicon:
                    return self.lexicon[collapsed]
                return collapsed

        return token

    def normalize(self, text: str) -> str:
        """Chuẩn hóa cả câu."""
        text = normalize_basic(text)
        if not text:
            return ""
        return " ".join(self.normalize_token(t) for t in text.split())

    def explain(self, text: str) -> list[tuple[str, str]]:
        """Trả các cặp (gốc, đã sửa) thực sự bị thay đổi — để giải thích/debug."""
        text = normalize_basic(text)
        out = []
        for token in text.split():
            fixed = self.normalize_token(token)
            if fixed != token:
                out.append((token, fixed))
        return out


# ---------------------------------------------------------------------------
# Chuẩn bị câu người dùng — dùng chung cho chatbot VÀ evaluate
# ---------------------------------------------------------------------------
def prepare_user_text(raw: str, normalizer: "TeencodeNormalizer | None"
                      ) -> tuple[str, list[tuple[str, str]]]:
    """Chuẩn hóa teencode rồi giữ nguyên "hệ quy chiếu dấu" của người dùng.

    Tách thành một hàm riêng để chatbot và evaluate.py đi qua ĐÚNG MỘT đường
    xử lý. Trước đây evaluate.py gọi thẳng retriever mà bỏ qua bước chuẩn hóa
    teencode, nên điểm số của truy vấn teencode trong báo cáo không phản ánh
    thứ người dùng thực sự nhận được.

    Bước giữ hệ quy chiếu dấu: từ điển teencode luôn trả từ CÓ DẤU. Nếu câu gốc
    không dấu mà ta để nguyên kết quả, chỉ một token được sửa ("thoi"->"thôi")
    là cả câu bị định tuyến sang index có dấu, nơi các token còn lại đều OOV.
    """
    raw = normalize_basic(raw)
    if normalizer is None or not raw:
        return raw, []
    pairs = normalizer.explain(raw)
    text = normalizer.normalize(raw)
    if not has_diacritics(raw):
        text = strip_accents(text)
    return text, pairs


# ---------------------------------------------------------------------------
# Đánh giá
# ---------------------------------------------------------------------------
def evaluate(normalizer: TeencodeNormalizer, split: str = "test") -> dict:
    """Đo chất lượng chuẩn hóa ở mức token trên split chưa từng thấy khi học.

    Các chỉ số:
      - `acc_before`: tỷ lệ token vốn đã đúng khi KHÔNG làm gì (baseline sao chép).
      - `acc_after` : tỷ lệ token đúng sau khi chuẩn hóa.
      - `ERR`       : Error Reduction Rate — bao nhiêu phần lỗi đã được sửa.
                      ERR = (acc_after - acc_before) / (1 - acc_before)
                      Đây là chỉ số chuẩn của bài toán lexical normalization:
                      accuracy thô bị thổi phồng vì đa số token vốn đã đúng sẵn.
      - precision/recall: chỉ tính trên những token THỰC SỰ cần sửa.
    """
    pairs = load_vilexnorm(split)

    total = correct_before = correct_after = 0
    need_change = 0          # token gold khác input
    changed_by_us = 0        # token ta đã đổi
    changed_correct = 0      # ta đổi và đổi đúng

    for original, gold in pairs:
        src = normalize_basic(original).split()
        tgt = normalize_basic(gold).split()
        if len(src) != len(tgt):
            continue
        for a, b in zip(src, tgt):
            total += 1
            pred = normalizer.normalize_token(a)

            if a.lower() == b.lower():
                correct_before += 1
            else:
                need_change += 1

            if pred.lower() == b.lower():
                correct_after += 1

            if pred.lower() != a.lower():
                changed_by_us += 1
                if pred.lower() == b.lower():
                    changed_correct += 1

    acc_before = correct_before / total if total else 0.0
    acc_after = correct_after / total if total else 0.0
    err = (acc_after - acc_before) / (1 - acc_before) if acc_before < 1 else 0.0

    precision = changed_correct / changed_by_us if changed_by_us else 0.0
    recall = changed_correct / need_change if need_change else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "split": split,
        "n_tokens": total,
        "n_need_change": need_change,
        "acc_before": acc_before,
        "acc_after": acc_after,
        "ERR": err,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def main() -> int:
    print("Đang học từ điển teencode từ ViLexNorm (split train)...")
    lexicon = build_and_save_lexicon()
    print(f"  Học được {len(lexicon)} ánh xạ -> {LEXICON_PATH}")

    print("\n20 ánh xạ mẫu:")
    for k in sorted(lexicon)[:20]:
        print(f"    {k:<12} -> {lexicon[k]}")

    normalizer = TeencodeNormalizer(lexicon)

    print("\nVí dụ chuẩn hóa:")
    demos = [
        "bt gì về vụ iphone k b",
        "t thấy đc đấy",
        "cho t hỏi vs",
        "ko bt là gì lun",
        "đẹppppp quá trờiii",
    ]
    for d in demos:
        print(f"    {d!r}\n      -> {normalizer.normalize(d)!r}")

    for split in ("dev", "test"):
        m = evaluate(normalizer, split)
        print(f"\n--- Đánh giá trên split {split} ---")
        print(f"  Số token           : {m['n_tokens']:,} (cần sửa: {m['n_need_change']:,})")
        print(f"  Accuracy trước     : {m['acc_before']:.2%}  (baseline: không làm gì)")
        print(f"  Accuracy sau       : {m['acc_after']:.2%}")
        print(f"  ERR (giảm lỗi)     : {m['ERR']:.2%}")
        print(f"  Precision / Recall : {m['precision']:.2%} / {m['recall']:.2%}  (F1 {m['f1']:.2%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
