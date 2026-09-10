"""
vectorizer.py — Bag-of-Words / n-gram / TF-IDF **tự cài đặt** bằng NumPy + SciPy.

Đây là phần lõi "from scratch" của đồ án. Toàn bộ công thức được viết tay,
không gọi `sklearn.feature_extraction`. Bản sklearn chỉ được dùng trong
`tests/test_vectorizer.py` để CHỨNG MINH bản tự viết cho ra cùng kết quả.

Công thức sử dụng (khớp với sklearn.TfidfVectorizer mặc định):

    tf(t, d)   = số lần term t xuất hiện trong document d
                 (hoặc 1 + log(tf) nếu sublinear_tf=True)

    idf(t)     = ln( (1 + n) / (1 + df(t)) ) + 1        # smooth_idf=True

    tfidf(t,d) = tf(t,d) * idf(t)

    v(d)       = tfidf(:,d) / L2_norm(tfidf(:,d))       # chuẩn hóa L2

Vì sao có "+1" trong log và "+1" ở cuối:
  - (1 + df) tránh chia cho 0 với term chỉ xuất hiện ở query, chưa từng thấy
    lúc fit.
  - "+1" ở cuối đảm bảo idf > 0, nên term xuất hiện ở MỌI document vẫn còn
    trọng số nhỏ thay vì bị triệt tiêu hoàn toàn về 0.

Sau khi chuẩn hóa L2, cosine similarity rút gọn thành phép nhân ma trận:
    cos(a, b) = dot(a, b) / (norm(a) * norm(b)) = dot(a_norm, b_norm)
"""

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy import sparse


# ---------------------------------------------------------------------------
# Sinh n-gram
# ---------------------------------------------------------------------------
def make_ngrams(tokens: list[str], ngram_range: tuple[int, int] = (1, 1)) -> list[str]:
    """Sinh n-gram từ danh sách token đã tách từ.

    ["xử_lý", "ngôn_ngữ", "tự_nhiên"] với ngram_range=(1,2) cho:
      unigram: xử_lý | ngôn_ngữ | tự_nhiên
      bigram : xử_lý ngôn_ngữ | ngôn_ngữ tự_nhiên

    Bigram giúp phân biệt "không tốt" với "tốt" — điều mà BoW unigram không
    làm được vì nó bỏ hoàn toàn thứ tự từ.
    """
    lo, hi = ngram_range
    grams: list[str] = []
    n_tokens = len(tokens)
    for n in range(lo, hi + 1):
        if n <= 0 or n > n_tokens:
            continue
        for i in range(n_tokens - n + 1):
            grams.append(" ".join(tokens[i : i + n]))
    return grams


# ---------------------------------------------------------------------------
# Count / Bag-of-Words
# ---------------------------------------------------------------------------
class CountVectorizer:
    """Bag-of-Words: mỗi document -> vector đếm tần suất term.

    Ma trận kết quả lưu ở dạng sparse CSR vì ma trận term-document rất thưa:
    một bài báo dùng vài trăm term trong vocabulary hàng chục nghìn term.
    """

    def __init__(
        self,
        ngram_range: tuple[int, int] = (1, 1),
        min_df: int = 1,
        max_df: float = 1.0,
        max_features: int | None = None,
    ):
        self.ngram_range = ngram_range
        self.min_df = min_df
        self.max_df = max_df
        self.max_features = max_features

        self.vocabulary_: dict[str, int] = {}
        self.feature_names_: list[str] = []
        self.document_frequency_: np.ndarray | None = None
        self.n_docs_: int = 0

    # -- fit ----------------------------------------------------------------
    def fit(self, documents: list[list[str]]) -> "CountVectorizer":
        """Xây vocabulary từ corpus. `documents` là list các list token."""
        n_docs = len(documents)
        self.n_docs_ = n_docs

        # Document frequency: mỗi term chỉ đếm 1 lần cho mỗi document.
        df_counter: Counter = Counter()
        for tokens in documents:
            df_counter.update(set(make_ngrams(tokens, self.ngram_range)))

        # Lọc theo min_df / max_df.
        if isinstance(self.max_df, float):
            max_df_abs = self.max_df * n_docs
        else:
            max_df_abs = self.max_df

        kept = [
            (term, df)
            for term, df in df_counter.items()
            if df >= self.min_df and df <= max_df_abs
        ]

        # Giữ top-N term phổ biến nhất nếu có giới hạn max_features.
        if self.max_features is not None and len(kept) > self.max_features:
            kept.sort(key=lambda x: (-x[1], x[0]))
            kept = kept[: self.max_features]

        # Sắp xếp alphabet để index ổn định giữa các lần chạy (tái lập được).
        kept.sort(key=lambda x: x[0])

        self.feature_names_ = [t for t, _ in kept]
        self.vocabulary_ = {t: i for i, t in enumerate(self.feature_names_)}
        self.document_frequency_ = np.array([df for _, df in kept], dtype=np.float64)
        return self

    # -- transform ----------------------------------------------------------
    def transform(self, documents: list[list[str]]) -> sparse.csr_matrix:
        """Chiếu document lên vocabulary đã học. Term lạ (OOV) bị bỏ qua."""
        indptr = [0]
        indices: list[int] = []
        values: list[float] = []

        for tokens in documents:
            counts: Counter = Counter()
            for gram in make_ngrams(tokens, self.ngram_range):
                idx = self.vocabulary_.get(gram)
                if idx is not None:
                    counts[idx] += 1
            for idx, cnt in sorted(counts.items()):
                indices.append(idx)
                values.append(float(cnt))
            indptr.append(len(indices))

        return sparse.csr_matrix(
            (values, indices, indptr),
            shape=(len(documents), len(self.vocabulary_)),
            dtype=np.float64,
        )

    def fit_transform(self, documents: list[list[str]]) -> sparse.csr_matrix:
        return self.fit(documents).transform(documents)


# ---------------------------------------------------------------------------
# TF-IDF
# ---------------------------------------------------------------------------
class TfidfVectorizer:
    """TF-IDF tự cài đặt, API tương thích sklearn nhưng viết tay toàn bộ."""

    def __init__(
        self,
        ngram_range: tuple[int, int] = (1, 1),
        min_df: int = 1,
        max_df: float = 1.0,
        max_features: int | None = None,
        sublinear_tf: bool = False,
        smooth_idf: bool = True,
        norm: str | None = "l2",
    ):
        self.counter = CountVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            max_features=max_features,
        )
        self.sublinear_tf = sublinear_tf
        self.smooth_idf = smooth_idf
        self.norm = norm
        self.idf_: np.ndarray | None = None

    # -- thuộc tính tiện dụng ------------------------------------------------
    @property
    def vocabulary_(self) -> dict[str, int]:
        return self.counter.vocabulary_

    @property
    def feature_names_(self) -> list[str]:
        return self.counter.feature_names_

    # -- fit -----------------------------------------------------------------
    def fit(self, documents: list[list[str]]) -> "TfidfVectorizer":
        self.counter.fit(documents)
        n = self.counter.n_docs_
        df = self.counter.document_frequency_

        if self.smooth_idf:
            # Coi như có thêm 1 document ảo chứa mọi term -> không bao giờ chia 0.
            self.idf_ = np.log((1.0 + n) / (1.0 + df)) + 1.0
        else:
            self.idf_ = np.log(n / df) + 1.0
        return self

    # -- transform -----------------------------------------------------------
    def transform(self, documents: list[list[str]]) -> sparse.csr_matrix:
        if self.idf_ is None:
            raise RuntimeError("Phai goi fit() truoc khi goi transform().")

        X = self.counter.transform(documents).tocsr()

        if self.sublinear_tf:
            # 1 + log(tf): giảm ảnh hưởng của term lặp lại quá nhiều lần.
            X.data = 1.0 + np.log(X.data)

        # Nhân mỗi cột với idf tương ứng. X.indices chính là chỉ số cột của
        # từng phần tử khác 0 -> nhân theo phần tử, không cần dựng ma trận chéo.
        X.data = X.data * self.idf_[X.indices]

        if self.norm == "l2":
            X = normalize_l2(X)
        return X

    def fit_transform(self, documents: list[list[str]]) -> sparse.csr_matrix:
        return self.fit(documents).transform(documents)

    # -- giải thích ----------------------------------------------------------
    def top_terms(self, vector, k: int = 10) -> list[tuple[str, float]]:
        """Trả về k term có trọng số cao nhất trong một vector 1 hàng.

        Dùng để GIẢI THÍCH vì sao bot chọn một document — phần quan trọng khi
        báo cáo, vì nó cho thấy mô hình không phải hộp đen.
        """
        row = sparse.csr_matrix(vector)
        if row.shape[0] != 1:
            raise ValueError("top_terms() chi nhan vector 1 hang.")
        pairs = sorted(zip(row.indices, row.data), key=lambda x: -x[1])
        names = self.feature_names_
        return [(names[i], float(v)) for i, v in pairs[:k]]


# ---------------------------------------------------------------------------
# Chuẩn hóa & độ tương đồng
# ---------------------------------------------------------------------------
def normalize_l2(X: sparse.csr_matrix) -> sparse.csr_matrix:
    """Chia mỗi hàng cho chuẩn L2 của nó. Hàng toàn 0 được giữ nguyên."""
    X = sparse.csr_matrix(X, copy=True)
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    norms[norms == 0.0] = 1.0            # tránh chia 0 với document rỗng
    row_lengths = np.diff(X.indptr)
    X.data = X.data / np.repeat(norms, row_lengths)
    return X


def cosine_similarity(A, B) -> np.ndarray:
    """Cosine similarity giữa mọi hàng của A và mọi hàng của B -> (n_A, n_B).

    Vector đã chuẩn hóa L2 nên cosine chỉ còn là tích vô hướng. Ta vẫn chuẩn
    hóa lại cho an toàn, phòng khi caller truyền ma trận chưa chuẩn hóa.
    """
    A = normalize_l2(sparse.csr_matrix(A))
    B = normalize_l2(sparse.csr_matrix(B))
    return np.asarray((A @ B.T).todense())
