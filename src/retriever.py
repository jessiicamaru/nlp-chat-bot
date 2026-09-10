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

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from underthesea import sent_tokenize

from config import (
    CONFIG_RETRIEVAL,
    RETRIEVAL_THRESHOLD,
    TFIDF_MAX_DF,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
    TOP_K,
)
from preprocess import normalize_basic, tokenize
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
    ):
        self.vectorizer = TfidfVectorizer(
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            sublinear_tf=True,   # bài báo dài, cần hãm term lặp nhiều lần
        )
        self.threshold = threshold
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
        self._sentence_cache.clear()
        return self

    # -- truy hồi ------------------------------------------------------------
    def _query_vector(self, query: str):
        return self.vectorizer.transform([tokenize(query, CONFIG_RETRIEVAL)])

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

        q_vec = self._query_vector(query)
        if q_vec.nnz == 0:
            # Không term nào của query có trong vocabulary -> không có căn cứ.
            return []

        scores = cosine_similarity(q_vec, self.doc_matrix)[0]

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
                    matched_terms=self._matched_terms(q_vec, doc_id),
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
            self._sentence_cache[doc_id] = [s for s in sents if 30 <= len(s) <= 400]
        return self._sentence_cache[doc_id]

    def best_sentences(self, doc_id: int, query: str, n: int = 2) -> str:
        """Chọn n câu trong bài có cosine similarity cao nhất với query.

        Giữ nguyên THỨ TỰ XUẤT HIỆN trong bài khi ghép lại, nếu không hai câu
        rời rạc ghép ngược thứ tự sẽ đọc rất khó hiểu.
        """
        sentences = self._sentences(doc_id)
        if not sentences:
            return normalize_basic(self.df.iloc[doc_id].get("description", ""))

        sent_tokens = [tokenize(s, CONFIG_RETRIEVAL) for s in sentences]
        # Vectorizer riêng cho từng bài: IDF tính trong phạm vi bài đó, nên
        # term đặc trưng của bài này được đánh giá đúng mức.
        local_vec = TfidfVectorizer(ngram_range=(1, 1), min_df=1, sublinear_tf=True)
        try:
            S = local_vec.fit_transform(sent_tokens)
            q = local_vec.transform([tokenize(query, CONFIG_RETRIEVAL)])
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
    def _matched_terms(self, q_vec, doc_id: int, k: int = 6) -> list[tuple[str, float]]:
        """Term nào của query thực sự khớp với bài, và đóng góp bao nhiêu điểm."""
        doc_vec = self.doc_matrix.getrow(doc_id)
        doc_map = dict(zip(doc_vec.indices, doc_vec.data))
        names = self.vectorizer.feature_names_

        contributions = [
            (names[idx], float(val * doc_map[idx]))
            for idx, val in zip(q_vec.indices, q_vec.data)
            if idx in doc_map
        ]
        contributions.sort(key=lambda x: -x[1])
        return contributions[:k]

    def explain(self, query: str, top_k: int = 3) -> dict:
        """Bảng chẩn đoán đầy đủ cho một query — dùng ở phần error analysis."""
        tokens = tokenize(query, CONFIG_RETRIEVAL)
        q_vec = self._query_vector(query)
        known = {self.vectorizer.feature_names_[i] for i in q_vec.indices}
        results = self.search(query, top_k=top_k, min_score=0.0)

        return {
            "query": query,
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
        subset = self.df[mask].head(n)
        return [
            RetrievalResult(
                doc_id=int(idx),
                score=1.0,
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
