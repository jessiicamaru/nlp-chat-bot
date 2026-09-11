"""
retriever.py — Truy hồi thông tin bằng TF-IDF + cosine similarity.

Kiến trúc hai tầng:

    Tầng 1 (article-level) — tìm BÀI BÁO liên quan nhất
        Index: title + description + text, đã qua preprocess_vi(CONFIG_RETRIEVAL).
        Title và description được lặp lại nhiều lần khi index vì chúng cô đọng
        chủ đề bài báo tốt hơn phần thân — đây là cách tăng trọng số trường
        quan trọng mà không cần sửa công thức TF-IDF.

    Tầng 2 (sentence-level) — trong bài đã chọn, tìm CÂU trả lời sát nhất
        Dùng lại `sent_tokenize` của Lab 01 để tách câu, rồi so cosine giữa
        query và từng câu. Nhờ vậy bot trả lời một đoạn ngắn đúng trọng tâm
        thay vì ném cả bài báo 3000 ký tự vào mặt người dùng.

Vì sao không dùng embedding/LLM: đồ án yêu cầu làm from scratch bằng kỹ thuật
đã học ở lab. Đổi lại, TF-IDF có ưu điểm thật: chạy tức thì, không cần GPU,
và mọi kết quả đều truy vết được về term cụ thể (xem `explain`).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from underthesea import sent_tokenize

from config import (
    CONFIG_RETRIEVAL,
    FRESHNESS_ALPHA,
    INDEX_CACHE_PATH,
    RETRIEVAL_THRESHOLD,
    TFIDF_MAX_DF,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
    TOP_K,
)
from preprocess import (
    fold_query,
    fold_tokens,
    has_diacritics,
    normalize_basic,
    strip_accents,
    strip_frame_words,
    tokenize,
)
from dates import compute_recency, format_vn_date, parse_vn_date
from vectorizer import TfidfVectorizer, cosine_similarity

# Số lần lặp lại title/description khi dựng document index.
TITLE_WEIGHT = 3
DESC_WEIGHT = 2


@dataclass
class RetrievalResult:
    """Một kết quả truy hồi, kèm đủ thông tin để giải thích vì sao nó được chọn."""

    doc_id: int
    score: float
    title: str
    category: str
    url: str
    snippet: str
    description: str = ""
    matched_terms: list[tuple[str, float]] = field(default_factory=list)
    published_date: object = None        # datetime.date | None
    base_score: float = 0.0              # điểm cosine TRƯỚC khi nhân độ mới

    @property
    def published_str(self) -> str:
        return format_vn_date(self.published_date)

    def __repr__(self) -> str:
        return f"<{self.score:.3f} | {self.category} | {self.title[:60]}>"


class NewsRetriever:
    """Truy hồi bài báo từ corpus bằng TF-IDF + cosine similarity."""

    def __init__(
        self,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        threshold=RETRIEVAL_THRESHOLD,
        freshness_alpha: float = FRESHNESS_ALPHA,
    ):
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True,   # bài báo dài, cần hãm term lặp nhiều lần
        )
        # Index thứ hai, dựng trên bản ĐÃ BỎ DẤU của cùng corpus. Chỉ dùng khi
        # câu hỏi của người dùng không có dấu. Giữ hai index tách biệt (thay vì
        # trộn chung) để index chính không bị pha loãng và các ngưỡng đã dò
        # được trên nó vẫn còn hiệu lực.
        self.folded_vectorizer = TfidfVectorizer(
            ngram_range=ngram_range, min_df=min_df, max_df=max_df, sublinear_tf=True
        )
        self.folded_matrix = None

        self.threshold = threshold
        self.freshness_alpha = freshness_alpha
        self.recency = None              # điểm độ mới (0, 1] cho từng bài
        self.published_dates: list = []
        self.df: pd.DataFrame | None = None
        self.doc_matrix = None
        self._sentence_cache: dict[int, list[str]] = {}

    # -- xây index -----------------------------------------------------------
    @staticmethod
    def _build_index_text(row: pd.Series) -> str:
        """Ghép các trường thành một document, có đánh trọng số bằng lặp lại."""
        title = normalize_basic(row.get("title", ""))
        desc = normalize_basic(row.get("description", ""))
        body = normalize_basic(row.get("text", ""))
        parts = [title] * TITLE_WEIGHT + [desc] * DESC_WEIGHT + [body]
        return " ".join(p for p in parts if p)

    def fit(self, df: pd.DataFrame) -> "NewsRetriever":
        self.df = df.reset_index(drop=True)

        raw_docs = [self._build_index_text(row) for _, row in self.df.iterrows()]
        tokenized = [tokenize(doc, CONFIG_RETRIEVAL) for doc in raw_docs]

        self.doc_matrix = self.vectorizer.fit_transform(tokenized)

        # Index phụ ở mức ÂM TIẾT không dấu. Tách từ vẫn chạy trên bản có dấu
        # (chính xác), rồi mới hạ xuống âm tiết — xem preprocess.fold_tokens.
        folded = [fold_tokens(doc) for doc in tokenized]
        self.folded_matrix = self.folded_vectorizer.fit_transform(folded)

        # Độ mới: dùng để phá thế hòa khi hai bài liên quan xấp xỉ nhau.
        self.published_dates = [parse_vn_date(v) for v in self.df.get("published_at", [])]
        self.recency = compute_recency(self.published_dates)

        self._sentence_cache.clear()
        return self


    # -- cache ra đĩa --------------------------------------------------------
    def fingerprint(self, df: pd.DataFrame) -> str:
        """Chữ ký của (dữ liệu + tham số ảnh hưởng tới index).

        Đổi một bài báo, hoặc đổi ngram_range/min_df/trọng số trường, đều phải
        làm cache hết hiệu lực. Ngược lại, đổi FRESHNESS_ALPHA thì KHÔNG — hệ số
        độ mới chỉ áp dụng lúc tìm kiếm, không nằm trong index.
        """
        h = hashlib.sha256()
        for col in ("url", "title", "description", "text", "published_at"):
            if col in df.columns:
                h.update(col.encode())
                h.update("".join(df[col].astype(str)).encode("utf-8", "ignore"))
        params = {
            "ngram_range": list(self.vectorizer.counter.ngram_range),
            "min_df": self.vectorizer.counter.min_df,
            "max_df": self.vectorizer.counter.max_df,
            "sublinear_tf": self.vectorizer.sublinear_tf,
            "title_weight": TITLE_WEIGHT,
            "desc_weight": DESC_WEIGHT,
            "config": CONFIG_RETRIEVAL,
        }
        h.update(json.dumps(params, sort_keys=True, ensure_ascii=False).encode())
        return h.hexdigest()

    def fit_cached(self, df: pd.DataFrame, cache_path: Path | None = None,
                   verbose: bool = False) -> "NewsRetriever":
        """Như fit(), nhưng nạp lại index đã lưu nếu dữ liệu chưa đổi.

        Dựng index tốn ~20 giây cho 381 bài, gần như toàn bộ là thời gian tách
        từ. `lru_cache` của segment_vi chỉ sống trong một tiến trình, nên khởi
        động lại là mất trắng. Cache ra đĩa giúp lần chạy sau gần như tức thì.
        """
        import joblib

        cache_path = Path(cache_path) if cache_path else INDEX_CACHE_PATH
        expected = self.fingerprint(df)

        if cache_path.exists():
            try:
                blob = joblib.load(cache_path)
                if blob.get("fingerprint") == expected:
                    self.vectorizer = blob["vectorizer"]
                    self.folded_vectorizer = blob["folded_vectorizer"]
                    self.doc_matrix = blob["doc_matrix"]
                    self.folded_matrix = blob["folded_matrix"]
                    self.published_dates = blob["published_dates"]
                    self.recency = blob["recency"]
                    self.df = df.reset_index(drop=True)
                    self._sentence_cache = blob.get("sentence_cache", {})
                    if verbose:
                        print(f"Nạp index từ cache: {cache_path}")
                    return self
                if verbose:
                    print("Cache có nhưng dữ liệu đã đổi -> dựng lại index.")
            except Exception as exc:
                # Cache hỏng/lệch phiên bản thư viện: bỏ qua, dựng lại từ đầu.
                if verbose:
                    print(f"Không đọc được cache ({exc}) -> dựng lại index.")

        self.fit(df)
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            joblib.dump(
                {
                    "fingerprint": expected,
                    "vectorizer": self.vectorizer,
                    "folded_vectorizer": self.folded_vectorizer,
                    "doc_matrix": self.doc_matrix,
                    "folded_matrix": self.folded_matrix,
                    "published_dates": self.published_dates,
                    "recency": self.recency,
                    "sentence_cache": self._sentence_cache,
                },
                cache_path,
                compress=3,
            )
            if verbose:
                print(f"Đã lưu index vào {cache_path}")
        except Exception as exc:
            # Không ghi được cache thì vẫn chạy bình thường, chỉ chậm hơn.
            if verbose:
                print(f"Không lưu được cache: {exc}")
        return self

    # -- truy hồi ------------------------------------------------------------
    def _query_vector(self, query: str):
        tokens = strip_frame_words(tokenize(query, CONFIG_RETRIEVAL))
        return self.vectorizer.transform([tokens])

    def _folded_query_vector(self, query: str):
        return self.folded_vectorizer.transform([strip_frame_words(fold_query(query))])

    def _select_index(self, query: str):
        """Chọn index phù hợp với câu hỏi -> (vector query, ma trận document).

        Câu có dấu -> index chính. Câu không dấu -> index đã bỏ dấu.
        """
        if has_diacritics(query):
            return self._query_vector(query), self.doc_matrix
        return self._folded_query_vector(query), self.folded_matrix

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        category: str | None = None,
        min_score: float | None = None,
    ) -> list[RetrievalResult]:
        """Trả top-k bài báo vượt ngưỡng similarity.

        `category` cho phép thu hẹp phạm vi khi người dùng đã nói rõ chuyên mục.
        """
        if self.df is None or self.doc_matrix is None:
            raise RuntimeError("Phai goi fit() truoc khi search().")

        threshold = self.threshold if min_score is None else min_score

        folded = not has_diacritics(query)
        q_vec, doc_matrix = self._select_index(query)
        if q_vec.nnz == 0:
            # Không term nào của query có trong vocabulary -> không có căn cứ.
            return []

        base_scores = cosine_similarity(q_vec, doc_matrix)[0]

        # Thưởng độ mới: score' = cosine * (1 + alpha * recency).
        # Nhân chứ không cộng, để bài không liên quan (cosine ~ 0) vẫn ở lại 0
        # dù mới tinh — nếu cộng thì tin mới nhất sẽ nổi lên với mọi câu hỏi.
        if self.freshness_alpha and self.recency is not None:
            scores = base_scores * (1.0 + self.freshness_alpha * self.recency)
        else:
            scores = base_scores

        # Lọc theo chuyên mục bằng cách triệt tiêu điểm của bài ngoài mục.
        if category:
            mask = (self.df["category"].astype(str).str.lower() == category.lower()).to_numpy()
            scores = np.where(mask, scores, 0.0)

        order = np.argsort(-scores)[:top_k]

        results: list[RetrievalResult] = []
        for doc_id in order:
            score = float(scores[doc_id])
            if score < threshold:
                continue
            row = self.df.iloc[doc_id]
            results.append(
                RetrievalResult(
                    doc_id=int(doc_id),
                    score=score,
                    title=str(row.get("title", "")),
                    category=str(row.get("category", "")),
                    url=str(row.get("url", "")),
                    description=normalize_basic(row.get("description", "")),
                    snippet=self.best_sentences(int(doc_id), query),
                    matched_terms=self._matched_terms(q_vec, doc_id, folded=folded),
                    published_date=(
                        self.published_dates[doc_id] if self.published_dates else None
                    ),
                    base_score=float(base_scores[doc_id]),
                )
            )
        return results

    # -- tầng 2: chọn câu ----------------------------------------------------
    def _sentences(self, doc_id: int) -> list[str]:
        """Tách câu của một bài (có cache vì sent_tokenize khá chậm)."""
        if doc_id not in self._sentence_cache:
            text = normalize_basic(self.df.iloc[doc_id].get("text", ""))
            sents = [s.strip() for s in sent_tokenize(text)] if text else []
            # Bỏ câu quá ngắn (chú thích ảnh, tên tác giả) và quá dài.
            kept = [s for s in sents if 30 <= len(s) <= 400]

            # Khử trùng lặp, giữ nguyên thứ tự xuất hiện. Nhiều bài VnExpress
            # lặp lại nguyên câu sapo trong phần thân; nếu không lọc thì hai
            # câu điểm cao nhất có thể là cùng một câu, và snippet đọc như bị lỗi.
            seen: set[str] = set()
            unique: list[str] = []
            for s in kept:
                key = s.lower()
                if key not in seen:
                    seen.add(key)
                    unique.append(s)
            self._sentence_cache[doc_id] = unique
        return self._sentence_cache[doc_id]

    def best_sentences(self, doc_id: int, query: str, n: int = 2) -> str:
        """Chọn n câu trong bài có cosine similarity cao nhất với query.

        Giữ nguyên THỨ TỰ XUẤT HIỆN trong bài khi ghép lại, nếu không hai câu
        rời rạc ghép ngược thứ tự sẽ đọc rất khó hiểu.
        """
        sentences = self._sentences(doc_id)
        if not sentences:
            return normalize_basic(self.df.iloc[doc_id].get("description", ""))

        fold = not has_diacritics(query)
        sent_tokens = [tokenize(s, CONFIG_RETRIEVAL) for s in sentences]
        if fold:
            # Câu hỏi không dấu -> hạ luôn phía câu trong bài về mức âm tiết.
            sent_tokens = [fold_tokens(toks) for toks in sent_tokens]
        # Vectorizer riêng cho từng bài: IDF tính trong phạm vi bài đó, nên
        # term đặc trưng của bài này được đánh giá đúng mức.
        local_vec = TfidfVectorizer(ngram_range=(1, 1), min_df=1, sublinear_tf=True)
        try:
            S = local_vec.fit_transform(sent_tokens)
            q_tokens = fold_query(query) if fold else tokenize(query, CONFIG_RETRIEVAL)
            q = local_vec.transform([q_tokens])
        except ValueError:
            return sentences[0]

        if q.nnz == 0:
            return " ".join(sentences[:n])

        sims = cosine_similarity(q, S)[0]
        if not np.any(sims > 0):
            return " ".join(sentences[:n])

        chosen = sorted(np.argsort(-sims)[:n])
        return " ".join(sentences[i] for i in chosen)

    # -- giải thích ----------------------------------------------------------
    def _matched_terms(self, q_vec, doc_id: int, k: int = 6, folded: bool = False
                       ) -> list[tuple[str, float]]:
        """Term nào của query thực sự khớp với bài, và đóng góp bao nhiêu điểm."""
        matrix = self.folded_matrix if folded else self.doc_matrix
        vectorizer = self.folded_vectorizer if folded else self.vectorizer
        doc_vec = matrix.getrow(doc_id)
        doc_map = dict(zip(doc_vec.indices, doc_vec.data))
        names = vectorizer.feature_names_

        contributions = [
            (names[idx], float(val * doc_map[idx]))
            for idx, val in zip(q_vec.indices, q_vec.data)
            if idx in doc_map
        ]
        contributions.sort(key=lambda x: -x[1])
        return contributions[:k]

    def explain(self, query: str, top_k: int = 3) -> dict:
        """Bảng chẩn đoán đầy đủ cho một query — dùng ở phần error analysis."""
        folded = not has_diacritics(query)
        tokens = strip_frame_words(
            fold_query(query) if folded else tokenize(query, CONFIG_RETRIEVAL)
        )
        q_vec, _ = self._select_index(query)
        vectorizer = self.folded_vectorizer if folded else self.vectorizer
        known = {vectorizer.feature_names_[i] for i in q_vec.indices}
        results = self.search(query, top_k=top_k, min_score=0.0)

        return {
            "query": query,
            "index": "đã bỏ dấu" if folded else "có dấu",
            "tokens": tokens,
            "terms_in_vocab": sorted(known),
            "oov_terms": [t for t in tokens if t not in known],
            "results": [
                {
                    "score": round(r.score, 4),
                    "title": r.title,
                    "category": r.category,
                    "matched_terms": [(t, round(v, 4)) for t, v in r.matched_terms],
                }
                for r in results
            ],
        }

    # -- duyệt theo chuyên mục ----------------------------------------------
    def list_categories(self) -> dict[str, int]:
        return self.df["category"].value_counts().to_dict()

    def browse(self, category: str, n: int = 5) -> list[RetrievalResult]:
        """Liệt kê bài mới nhất trong một chuyên mục (không cần query)."""
        mask = self.df["category"].astype(str).str.lower() == category.lower()
        subset = self.df[mask]

        # Sắp theo ngày đăng giảm dần. Trước đây dùng .head(n) tức là theo thứ tự
        # chèn vào corpus — người dùng hỏi "tin mới nhất" mà lại nhận bài cũ.
        if self.published_dates:
            import datetime as _dt

            oldest = _dt.date.min
            subset = subset.assign(
                _pub=[self.published_dates[i] or oldest for i in subset.index]
            ).sort_values("_pub", ascending=False).drop(columns=["_pub"])
        subset = subset.head(n)

        return [
            RetrievalResult(
                doc_id=int(idx),
                score=1.0,
                published_date=(self.published_dates[idx] if self.published_dates else None),
                title=str(row.get("title", "")),
                category=str(row.get("category", "")),
                url=str(row.get("url", "")),
                description=normalize_basic(row.get("description", "")),
                snippet=normalize_basic(row.get("description", "")),
            )
            for idx, row in subset.iterrows()
        ]

    def stats(self) -> dict:
        return {
            "n_documents": len(self.df),
            "n_categories": self.df["category"].nunique(),
            "categories": self.list_categories(),
            "vocabulary_size": len(self.vectorizer.vocabulary_),
            "sources": self.df["source"].unique().tolist(),
            "avg_text_length": int(self.df["text"].astype(str).str.len().mean()),
        }
