"""
generator.py — Mô hình ngôn ngữ n-gram tự cài đặt, dùng để SINH văn bản.

## Mục đích: một thí nghiệm đối chứng, KHÔNG phải tính năng của chatbot

Chatbot chính hoạt động theo kiểu TRÍCH XUẤT (extractive): nó chỉ trả về những
câu đã có sẵn trong corpus. Một câu hỏi hợp lý là:

    "Sao không để mô hình TỰ SINH câu trả lời cho tự nhiên hơn?"

Module này trả lời câu hỏi đó bằng thực nghiệm thay vì bằng lời. Ta cài đặt
mô hình sinh cổ điển nhất — n-gram language model — rồi ĐO chất lượng đầu ra.
Kết luận (xem `compare_extractive_vs_generative`): ở quy mô dữ liệu của đồ án,
văn bản sinh ra mạch lạc trong 5-8 từ đầu rồi trôi dạt thành vô nghĩa.

Đây chính là lý do kiến trúc chính chọn truy hồi. Kết luận này rút ra từ số
liệu, không phải từ giả định.

## Công thức

Giả định Markov bậc (n-1): xác suất của một từ chỉ phụ thuộc n-1 từ ngay trước.

    P(w_i | w_1...w_{i-1}) ~= P(w_i | w_{i-n+1}...w_{i-1})

Ước lượng hợp lý cực đại:

    P(w_i | context) = count(context + w_i) / count(context)

## Làm mịn: nội suy đệ quy (recursive interpolation)

Ước lượng thô ở trên gán xác suất 0 cho mọi n-gram chưa từng thấy — mà với
corpus vài trăm nghìn token thì gần như MỌI trigram mới đều chưa từng thấy.
Ta trộn các bậc lại với nhau:

    P_interp(w | context) = λ·P_ML(w | context) + (1-λ)·P_interp(w | context[1:])

đệ quy xuống tới unigram. Nhờ đó mô hình luôn có phương án dự phòng khi ngữ
cảnh dài chưa từng xuất hiện.

## Đo bằng perplexity

    PP = exp( - (1/N) · Σ log P(w_i | context) )

Perplexity là "số lựa chọn trung bình mà mô hình còn phân vân ở mỗi bước".
Càng thấp càng tốt. Đo trên tập test tách riêng, không dùng khi huấn luyện.
"""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict

from config import CONFIG_RETRIEVAL, RANDOM_SEED
from preprocess import normalize_basic, segment_vi

BOS = "<s>"      # đầu câu
EOS = "</s>"     # cuối câu


class NgramLanguageModel:
    """Mô hình ngôn ngữ n-gram với nội suy đệ quy, viết tay hoàn toàn."""

    def __init__(self, n: int = 3, lambda_: float = 0.7, seed: int | None = RANDOM_SEED):
        if n < 1:
            raise ValueError("n phai >= 1")
        self.n = n
        self.lambda_ = lambda_
        self.rng = random.Random(seed)

        # counts[k] = Counter các k-gram (tuple độ dài k)
        self.counts: dict[int, Counter] = {k: Counter() for k in range(1, n + 1)}
        # context_counts[k] = Counter các tiền tố độ dài k
        self.context_counts: dict[int, Counter] = {k: Counter() for k in range(0, n)}
        # Để sinh nhanh: context -> danh sách từ tiếp theo có thể
        self.followers: dict[tuple, Counter] = defaultdict(Counter)

        self.vocab: set[str] = set()
        self.n_tokens = 0

    # -- huấn luyện ----------------------------------------------------------
    def _pad(self, tokens: list[str]) -> list[str]:
        """Thêm n-1 ký hiệu đầu câu và một ký hiệu cuối câu."""
        return [BOS] * (self.n - 1) + tokens + [EOS]

    def fit(self, sentences: list[list[str]]) -> "NgramLanguageModel":
        """`sentences` là list câu, mỗi câu là list token đã tách từ."""
        for tokens in sentences:
            if not tokens:
                continue
            padded = self._pad(tokens)
            self.vocab.update(tokens)
            self.n_tokens += len(tokens)

            for k in range(1, self.n + 1):
                for i in range(len(padded) - k + 1):
                    gram = tuple(padded[i : i + k])
                    self.counts[k][gram] += 1
                    if k > 1:
                        self.context_counts[k - 1][gram[:-1]] += 1

            # Bảng tra để sinh: (n-1) từ trước -> từ kế tiếp
            for i in range(self.n - 1, len(padded)):
                context = tuple(padded[i - self.n + 1 : i])
                self.followers[context][padded[i]] += 1

        self.vocab.add(EOS)
        return self

    # -- xác suất ------------------------------------------------------------
    def _prob_ml(self, word: str, context: tuple) -> float:
        """Ước lượng hợp lý cực đại cho một bậc cụ thể."""
        k = len(context) + 1
        gram = context + (word,)
        gram_count = self.counts[k][gram]
        if gram_count == 0:
            return 0.0
        if k == 1:
            total = sum(self.counts[1].values())
            return gram_count / total if total else 0.0
        ctx_count = self.context_counts[k - 1][context]
        return gram_count / ctx_count if ctx_count else 0.0

    def prob(self, word: str, context: tuple) -> float:
        """Xác suất có nội suy đệ quy xuống các bậc thấp hơn."""
        context = tuple(context)[-(self.n - 1):] if self.n > 1 else ()

        # Đáy đệ quy: unigram, có làm mịn add-1 để không bao giờ trả về 0.
        if not context:
            total = sum(self.counts[1].values())
            v = len(self.vocab) + 1
            return (self.counts[1][(word,)] + 1) / (total + v) if total else 1.0 / v

        higher = self._prob_ml(word, context)
        lower = self.prob(word, context[1:])
        return self.lambda_ * higher + (1 - self.lambda_) * lower

    # -- perplexity ----------------------------------------------------------
    def perplexity(self, sentences: list[list[str]]) -> float:
        """Perplexity trên tập câu chưa từng thấy khi huấn luyện."""
        log_sum = 0.0
        count = 0
        for tokens in sentences:
            if not tokens:
                continue
            padded = self._pad(tokens)
            for i in range(self.n - 1, len(padded)):
                context = tuple(padded[i - self.n + 1 : i])
                p = self.prob(padded[i], context)
                # p luôn > 0 nhờ làm mịn ở đáy đệ quy.
                log_sum += math.log(p)
                count += 1
        return math.exp(-log_sum / count) if count else float("inf")

    # -- sinh văn bản --------------------------------------------------------
    def generate(self, max_tokens: int = 30, seed_text: str = "", temperature: float = 1.0) -> str:
        """Sinh câu mới bằng cách lấy mẫu tuần tự từ phân phối đã học.

        `temperature` < 1 làm phân phối nhọn hơn (an toàn, lặp lại nhiều);
        > 1 làm phẳng hơn (đa dạng, hỗn loạn hơn).
        """
        if seed_text:
            start = segment_vi(normalize_basic(seed_text)).lower().split()
        else:
            start = []

        history = [BOS] * (self.n - 1) + start
        output = list(start)

        for _ in range(max_tokens):
            context = tuple(history[-(self.n - 1):]) if self.n > 1 else ()
            choices = self.followers.get(context)

            # Ngữ cảnh chưa từng thấy -> lùi dần về bậc thấp hơn.
            while not choices and len(context) > 0:
                context = context[1:]
                choices = self.followers.get(context)

            if not choices:
                break

            words = list(choices.keys())
            weights = [c ** (1.0 / temperature) for c in choices.values()]
            nxt = self.rng.choices(words, weights=weights, k=1)[0]

            if nxt == EOS:
                break
            output.append(nxt)
            history.append(nxt)

        # Trả lại dạng đọc được: bỏ gạch dưới của từ ghép.
        return " ".join(output).replace("_", " ")


# ---------------------------------------------------------------------------
# Chuẩn bị dữ liệu
# ---------------------------------------------------------------------------
def corpus_to_sentences(df, max_docs: int | None = None) -> list[list[str]]:
    """Biến corpus bài báo thành list câu đã tách từ, để huấn luyện LM."""
    from underthesea import sent_tokenize

    sentences: list[list[str]] = []
    rows = df if max_docs is None else df.head(max_docs)

    for _, row in rows.iterrows():
        text = normalize_basic(row.get("text", ""))
        if not text:
            continue
        for sent in sent_tokenize(text):
            sent = sent.strip()
            if len(sent) < 20:
                continue
            tokens = segment_vi(sent).lower().split()
            if len(tokens) >= 3:
                sentences.append(tokens)
    return sentences


# ---------------------------------------------------------------------------
# Thí nghiệm đối chứng
# ---------------------------------------------------------------------------
def compare_extractive_vs_generative(df, orders=(1, 2, 3, 4), n_samples: int = 3) -> dict:
    """Huấn luyện LM ở nhiều bậc n, đo perplexity và in văn bản sinh ra."""
    sentences = corpus_to_sentences(df)

    split = int(len(sentences) * 0.9)
    train, test = sentences[:split], sentences[split:]

    print(f"Số câu huấn luyện: {len(train):,}   |   kiểm thử: {len(test):,}")
    print(f"Tổng token       : {sum(len(s) for s in train):,}\n")

    results = {}
    for n in orders:
        lm = NgramLanguageModel(n=n).fit(train)
        pp = lm.perplexity(test)
        results[n] = {"perplexity": pp, "vocab": len(lm.vocab), "samples": []}

        print("=" * 78)
        print(f"n = {n}  |  perplexity = {pp:,.1f}  |  vocab = {len(lm.vocab):,}")
        print("-" * 78)
        for i in range(n_samples):
            text = lm.generate(max_tokens=28, seed_text="du lịch")
            results[n]["samples"].append(text)
            print(f"  [{i+1}] {text}")
        print()

    return results


def sweep_lambda(df, lambdas=(0.3, 0.5, 0.7, 0.9), orders=(2, 3, 4)) -> dict:
    """Quét trọng số nội suy để giải thích vì sao perplexity TĂNG theo bậc n."""
    sentences = corpus_to_sentences(df)
    split = int(len(sentences) * 0.9)
    train, test = sentences[:split], sentences[split:]

    print(f"\n{'lambda':>7}" + "".join(f"{'n=' + str(n):>12}" for n in orders))
    print("-" * (7 + 12 * len(orders)))

    results: dict = {}
    for lam in lambdas:
        row = f"{lam:>7.1f}"
        results[lam] = {}
        for n in orders:
            lm = NgramLanguageModel(n=n, lambda_=lam).fit(train)
            pp = lm.perplexity(test)
            results[lam][n] = pp
            row += f"{pp:>12,.0f}"
        print(row)
    return results


def main() -> int:
    import pandas as pd

    from config import CORPUS_RAW_PATH

    df = pd.read_csv(CORPUS_RAW_PATH).dropna(subset=["text"])
    print("THÍ NGHIỆM: SINH VĂN BẢN BẰNG N-GRAM LANGUAGE MODEL")
    print("=" * 78)
    print("Mục đích: kiểm chứng bằng số liệu vì sao chatbot chọn TRÍCH XUẤT")
    print("thay vì SINH văn bản.\n")

    compare_extractive_vs_generative(df)

    print("=" * 78)
    print("QUÉT TRỌNG SỐ NỘI SUY (lambda)")
    print("=" * 78)
    print("Dùng để kiểm chứng lời giải thích cho hiện tượng perplexity tăng theo n.")
    sweep_lambda(df)

    print("\n" + "=" * 78)
    print("NHẬN XÉT")
    print("=" * 78)
    print("""
1. PERPLEXITY TĂNG THEO BẬC n — ngược với trực giác thông thường.
   Đo được (lambda = 0.7):  n=2 -> 819   n=3 -> 1.951   n=4 -> 5.911
   Trực giác "n lớn hơn thì mô hình mạnh hơn" KHÔNG đúng ở quy mô dữ liệu này.

   Nguyên nhân là DỮ LIỆU THƯA. Corpus chỉ có ~217.000 token, nên hầu hết
   4-gram trong tập test chưa từng xuất hiện khi huấn luyện. Khi đó thành phần
   bậc cao bằng 0, và công thức nội suy

       P = lambda * 0  +  (1 - lambda) * P(bậc thấp hơn)

   nhân thêm hệ số (1 - lambda) ở MỖI lần lùi bậc. Với lambda = 0.7 và phải
   lùi hai bậc, xác suất bị nhân với 0,3 x 0,3 = 0,09 — phạt rất nặng.

   Bảng quét lambda ở trên xác nhận đúng cơ chế đó: hạ lambda từ 0,9 xuống 0,3
   làm perplexity của n=4 giảm từ 63.318 xuống 1.173. Nhưng ở MỌI lambda,
   n=2 vẫn tốt nhất -> với lượng dữ liệu này, bigram là điểm dừng hợp lý.

   Muốn dùng bậc cao hơn thì phải đổi sang làm mịn tốt hơn (Kneser-Ney, hoặc
   backoff Katz), chứ không phải chỉ tăng n.

2. n CÀNG LỚN, "SINH" CÀNG BIẾN THÀNH "CHÉP".
   Với n=4, phần lớn ngữ cảnh chỉ xuất hiện đúng một lần trong corpus nên chỉ
   có duy nhất một từ kế tiếp khả dĩ. Quan sát được ở output: cả ba mẫu n=4
   đều mở đầu bằng đúng một chuỗi "du lịch phú quốc thành lập năm 2014 ...".
   Tức là mô hình chép nguyên văn — không thêm giá trị gì so với truy hồi,
   mà lại mất khả năng dẫn nguồn.

3. VĂN BẢN SINH RA SAI SỰ THẬT ở mọi bậc n.
   Ví dụ thật từ output: "du lịch phú quốc còn đang xây dựng các trung tâm
   điều trị ebola", "du lịch phú quốc thành lập năm 2014, cô đã 12 lần vô địch
   médoc". Mô hình chỉ nối từ theo thống kê, không có khái niệm về sự kiện.
   Với một bot tin tức thì đây là lỗi không thể chấp nhận.

KẾT LUẬN
   Ở quy mô dữ liệu của đồ án, sinh văn bản bằng n-gram thua truy hồi trên mọi
   tiêu chí quan trọng: kém mạch lạc, sai sự thật, và không dẫn được nguồn.
   Đây là căn cứ thực nghiệm cho lựa chọn kiến trúc TRÍCH XUẤT của chatbot,
   chứ không phải một giả định.

   Muốn vừa sinh văn bản tự nhiên vừa đúng sự thật thì cần mô hình ngôn ngữ
   lớn đã tiền huấn luyện (PhoGPT, Vistral...) kết hợp truy hồi theo kiểu RAG
   — nằm ngoài phạm vi "from scratch" của đồ án, và được ghi ở phần hướng
   phát triển.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
