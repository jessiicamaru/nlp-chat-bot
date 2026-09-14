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

    Đường dự phòng chống gõ sai (docs/09)
        Khi tầng 1 không có bài nào đạt ngưỡng, thử lại trên chỉ mục n-gram
        KÝ TỰ của tiêu đề đã bỏ dấu: "son dong" vẫn gần "son doong". Câu gõ
        đúng không bao giờ đi vào nhánh này, nên hành vi cũ giữ nguyên.

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
    BM25_B,
    BM25_K1,
    CONFIG_RETRIEVAL,
    FRESHNESS_ALPHA,
    FRESHNESS_HALFLIFE_DAYS,
    FRESHNESS_REFERENCE,
    FUZZY_CHAR_N,
    FUZZY_ENABLED,
    FUZZY_THRESHOLD,
    INDEX_CACHE_PATH,
    RANKING_METHOD,
    RETRIEVAL_THRESHOLD,
    TFIDF_MAX_DF,
    TFIDF_MIN_DF,
    TFIDF_NGRAM_RANGE,
    TOP_K,
)
from preprocess import (
    fold_for_chars,
    fold_query,
    fold_tokens,
    has_diacritics,
    normalize_basic,
    strip_accents,
    strip_frame_words,
    tokenize,
)
from dates import compute_recency, format_vn_date, parse_vn_date
from vectorizer import (
    TfidfVectorizer,
    bm25_idf,
    bm25_scores,
    bm25_weights,
    cosine_similarity,
    make_char_ngrams,
)

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
    match: str = "exact"                 # "exact" (mức từ) | "fuzzy" (dự phòng gõ sai)
    fuzzy_score: float = 0.0             # cosine từ + cosine n-gram ký tự, chỉ khi fuzzy

    @property
    def published_str(self) -> str:
        return format_vn_date(self.published_date)

    def __repr__(self) -> str:
        return f"<{self.score:.3f} | {self.category} | {self.title[:60]}>"


@dataclass
class _Scored:
    """Điểm của một câu hỏi trên toàn corpus (nội bộ, dùng chung search/rank)."""

    q_vec: object                        # vector TF-IDF của câu hỏi
    folded: bool                         # có dùng index bỏ dấu không
    mask: object                         # mặt nạ chuyên mục hoặc None
    base: np.ndarray                     # cosine thuần — thước đo CHẤP NHẬN
    scores: np.ndarray                   # điểm XẾP HẠNG (đã nhân độ mới)


class NewsRetriever:
    """Truy hồi bài báo từ corpus bằng TF-IDF + cosine similarity."""

    def __init__(
        self,
        ngram_range=TFIDF_NGRAM_RANGE,
        min_df=TFIDF_MIN_DF,
        max_df=TFIDF_MAX_DF,
        threshold=RETRIEVAL_THRESHOLD,
        freshness_alpha: float = FRESHNESS_ALPHA,
        freshness_halflife: float = FRESHNESS_HALFLIFE_DAYS,
        ranking: str = RANKING_METHOD,
        bm25_k1: float = BM25_K1,
        bm25_b: float = BM25_B,
        freshness_reference: str = FRESHNESS_REFERENCE,
        fuzzy_threshold: float | None = FUZZY_THRESHOLD if FUZZY_ENABLED else None,
    ):
        if ranking not in ("tfidf", "bm25"):
            raise ValueError(f"ranking phai la 'tfidf' hoac 'bm25', nhan duoc {ranking!r}")
        if freshness_reference not in ("corpus", "candidates"):
            raise ValueError("freshness_reference phai la 'corpus' hoac 'candidates', "
                             f"nhan duoc {freshness_reference!r}")
        self.freshness_reference = freshness_reference
        # None = tắt đường dự phòng chống gõ sai.
        self.fuzzy_threshold = fuzzy_threshold

        # Chỉ mục thứ ba: n-gram ký tự của TIÊU ĐỀ đã bỏ dấu (xem config.FUZZY_*).
        self.char_vectorizer = TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True)
        self.char_matrix = None
        self.ranking = ranking
        self.bm25_k1 = bm25_k1
        self.bm25_b = bm25_b
        # Ma trận ĐẾM term (đầu vào của BM25) và ma trận trọng số BM25 tính sẵn.
        self.doc_counts = None
        self.folded_counts = None
        self._bm25_main = None
        self._bm25_folded = None

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
        self.freshness_halflife = freshness_halflife
        self.recency = None              # điểm độ mới (0, 1] theo mốc CẢ corpus
        self.published_dates: list = []
        self._date_ordinals = None       # ngày đăng dạng số (NaN nếu không rõ)
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

        # Ma trận đếm dùng chung vocabulary với TF-IDF -> BM25 không cần tách
        # từ lại, và đổi k1/b chỉ cần tính lại trọng số (set_bm25).
        self.doc_counts = self.vectorizer.counter.transform(tokenized)
        self.folded_counts = self.folded_vectorizer.counter.transform(folded)
        self._rebuild_bm25()

        self.char_matrix = self.char_vectorizer.fit_transform(
            [self._char_grams(row.get("title", "")) for _, row in self.df.iterrows()])

        # Độ mới: dùng để phá thế hòa khi hai bài liên quan xấp xỉ nhau.
        self.published_dates = [parse_vn_date(v) for v in self.df.get("published_at", [])]
        self._set_dates(self.published_dates)

        self._sentence_cache.clear()
        return self

    @staticmethod
    def _char_grams(text) -> list[str]:
        return make_char_ngrams(fold_for_chars(text), FUZZY_CHAR_N)

    def _set_dates(self, dates: list) -> None:
        self.published_dates = dates
        self._date_ordinals = np.array(
            [d.toordinal() if d else np.nan for d in dates], dtype=np.float64)
        self.recency = compute_recency(dates, half_life_days=self.freshness_halflife)


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
            "fuzzy_index": {"field": "title", "char_n": FUZZY_CHAR_N},
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
                    self.doc_counts = blob["doc_counts"]
                    self.folded_counts = blob["folded_counts"]
                    self.char_vectorizer = blob["char_vectorizer"]
                    self.char_matrix = blob["char_matrix"]
                    self._rebuild_bm25()
                    # KHÔNG nạp recency từ cache mà tính lại (_set_dates). Nửa chu
                    # kỳ độ mới không nằm trong vân tay cache (nó không ảnh hưởng
                    # index), nên nếu nạp recency đã lưu thì đổi
                    # FRESHNESS_HALFLIFE_DAYS sẽ bị bỏ qua âm thầm.
                    self._set_dates(blob["published_dates"])
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
                    "doc_counts": self.doc_counts,
                    "folded_counts": self.folded_counts,
                    "char_vectorizer": self.char_vectorizer,
                    "char_matrix": self.char_matrix,
                    "published_dates": self.published_dates,
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

    # -- BM25 ----------------------------------------------------------------
    def _rebuild_bm25(self) -> None:
        if self.doc_counts is None:
            return
        c = self.vectorizer.counter
        fc = self.folded_vectorizer.counter
        self._bm25_main = bm25_weights(
            self.doc_counts, bm25_idf(c.document_frequency_, c.n_docs_),
            k1=self.bm25_k1, b=self.bm25_b)
        self._bm25_folded = bm25_weights(
            self.folded_counts, bm25_idf(fc.document_frequency_, fc.n_docs_),
            k1=self.bm25_k1, b=self.bm25_b)

    def set_bm25(self, k1: float, b: float, ranking: str | None = None) -> None:
        """Đổi tham số BM25 mà không phải dựng lại index (dùng khi dò trên dev)."""
        self.bm25_k1, self.bm25_b = k1, b
        if ranking is not None:
            self.ranking = ranking
        self._rebuild_bm25()

    def _query_tokens(self, query: str) -> tuple[list[str], bool]:
        """(token của câu hỏi, có dùng index bỏ dấu không)."""
        if has_diacritics(query):
            return strip_frame_words(tokenize(query, CONFIG_RETRIEVAL)), False
        return strip_frame_words(fold_query(query)), True

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

    def _category_mask(self, category: str | None):
        if not category:
            return None
        return (self.df["category"].astype(str).str.lower() == category.lower()).to_numpy()

    def _score(self, query: str, category: str | None = None) -> _Scored:
        """Toàn bộ phần TÍNH ĐIỂM của một câu hỏi, chưa dựng kết quả.

        Tách riêng để search() và evaluate.py (qua rank()) dùng CHUNG một cách
        tính — nếu evaluate tự tính lại điểm thì hai bên sớm muộn sẽ lệch nhau.
        """
        if self.df is None or self.doc_matrix is None:
            raise RuntimeError("Phai goi fit() truoc khi search().")

        folded = not has_diacritics(query)
        q_vec, doc_matrix = self._select_index(query)
        mask = self._category_mask(category)
        if q_vec.nnz == 0:
            # Không term nào của query có trong vocabulary -> đường chính không
            # có căn cứ. Câu gõ sai ("ipone") hay rơi vào đây; dự phòng vẫn thử.
            zeros = np.zeros(doc_matrix.shape[0])
            return _Scored(q_vec, folded, mask, zeros, zeros)

        # Cosine TF-IDF: luôn tính, vì đây là thước đo CHẤP NHẬN câu trả lời.
        base_scores = cosine_similarity(q_vec, doc_matrix)[0]

        # Điểm XẾP HẠNG: cosine hoặc BM25 tùy cấu hình.
        if self.ranking == "bm25" and self._bm25_main is not None:
            tokens, use_folded = self._query_tokens(query)
            counter = (self.folded_vectorizer if use_folded else self.vectorizer).counter
            W = self._bm25_folded if use_folded else self._bm25_main
            rank_scores = bm25_scores(W, counter.transform([tokens]))
        else:
            rank_scores = base_scores

        # Lọc theo chuyên mục bằng cách triệt tiêu điểm của bài ngoài mục. Lọc
        # TRƯỚC khi tính độ mới, để mốc "bài cạnh tranh" chỉ gồm bài trong mục.
        if mask is not None:
            rank_scores = np.where(mask, rank_scores, 0.0)
            base_scores = np.where(mask, base_scores, 0.0)

        # Thưởng độ mới: score' = rank_score * (1 + alpha * recency).
        # Nhân chứ không cộng, để bài không liên quan (điểm ~ 0) vẫn ở lại 0
        # dù mới tinh — nếu cộng thì tin mới nhất sẽ nổi lên với mọi câu hỏi.
        # Phép nhân cũng không phụ thuộc thang đo, nên dùng được cho cả BM25.
        return _Scored(q_vec, folded, mask, base_scores, self._apply_freshness(rank_scores))

    def _score_fuzzy(self, query: str, scored: _Scored) -> tuple[np.ndarray, np.ndarray]:
        """Đường dự phòng: (điểm chấp nhận, điểm xếp hạng).

        Điểm chấp nhận = cosine mức từ + cosine n-gram ký tự của tiêu đề. Cộng
        hai tín hiệu thay vì chỉ dùng n-gram ký tự: bài đúng thường vẫn còn chút
        điểm mức từ (các từ gõ đúng trong câu), còn bài chỉ "na ná cách viết"
        thì không. Trên dev, luật cộng cứu được nhiều câu nhất mà 0 câu trả sai
        (config.FUZZY_*). Độ mới vẫn chỉ dùng để xếp hạng như đường chính.
        """
        combined = scored.base + self.fuzzy_scores(query)
        if scored.mask is not None:
            combined = np.where(scored.mask, combined, 0.0)
        return combined, self._apply_freshness(combined)

    def rank(self, query: str, top_k: int = TOP_K, category: str | None = None,
             fuzzy: bool = False) -> list[tuple[int, float, float]]:
        """Top-k (doc_id, điểm xếp hạng, điểm chấp nhận), KHÔNG lọc ngưỡng.

        Nhẹ hơn search() vì không tách câu / dựng snippet — evaluate.py gọi hàm
        này hàng chục nghìn lần khi quét tham số. `fuzzy=True` xếp theo đường
        dự phòng gõ sai (điểm chấp nhận khi đó là cosine từ + cosine ký tự).
        """
        scored = self._score(query, category)
        accept, scores = (self._score_fuzzy(query, scored) if fuzzy
                          else (scored.base, scored.scores))
        order = np.argsort(-scores, kind="stable")[:top_k]
        return [(int(i), float(scores[i]), float(accept[i])) for i in order]

    def search(
        self,
        query: str,
        top_k: int = TOP_K,
        category: str | None = None,
        min_score: float | None = None,
    ) -> list[RetrievalResult]:
        """Trả top-k bài báo vượt ngưỡng similarity.

        `category` cho phép thu hẹp phạm vi khi người dùng đã nói rõ chuyên mục.
        `min_score=0.0` tắt ngưỡng (dùng khi chẩn đoán) — khi đó không có
        "từ chối" nên cũng không bao giờ đi vào đường dự phòng gõ sai.
        """
        threshold = self.threshold if min_score is None else min_score
        scored = self._score(query, category)

        # XẾP HẠNG theo điểm đã nhân độ mới, nhưng CHẤP NHẬN theo cosine thuần.
        #
        # Hai câu hỏi khác nhau thì phải dùng hai thước đo khác nhau:
        #   - "bài nào nên đứng trước?"      -> độ liên quan + độ mới
        #   - "có đủ căn cứ để trả lời không?" -> CHỈ độ liên quan
        #
        # Lỗi cũ: ngưỡng áp lên điểm đã nhân độ mới. Với nửa chu kỳ 3 ngày, bài
        # 16 ngày tuổi gần như không được thưởng, nên phải liên quan hơn hẳn mới
        # vượt được ngưỡng — độ mới đã âm thầm biến thành bộ lọc loại bài cũ.
        # "tin ve dao hai nam" từ trả lời đúng chuyển sang "không tìm thấy".
        # Kiểm thử hồi quy (tests/test_chatbot.py) bắt được lỗi này.
        results: list[RetrievalResult] = []
        if scored.q_vec.nnz:
            order = np.argsort(-scored.scores, kind="stable")[:top_k]
            results = [self._make_result(int(i), query, scored.scores[i], scored.base[i], scored)
                       for i in order if float(scored.base[i]) >= threshold]
        if results or not threshold or self.fuzzy_threshold is None:
            return results

        # Đường chính từ chối -> thử dự phòng gõ sai (docs/09).
        accept, scores = self._score_fuzzy(query, scored)
        order = np.argsort(-scores, kind="stable")[:top_k]
        for i in order:
            if float(accept[i]) < self.fuzzy_threshold:
                continue
            res = self._make_result(int(i), query, scores[i], scored.base[i], scored)
            res.match, res.fuzzy_score = "fuzzy", float(accept[i])
            results.append(res)
        return results

    # -- độ mới ----------------------------------------------------------------
    def _apply_freshness(self, rank_scores: np.ndarray) -> np.ndarray:
        if not self.freshness_alpha or self.recency is None:
            return rank_scores
        return rank_scores * (1.0 + self.freshness_alpha * self._recency_for(rank_scores))

    def _recency_for(self, rank_scores: np.ndarray) -> np.ndarray:
        """Điểm độ mới cho MỘT câu hỏi, theo mốc tham chiếu đã cấu hình.

        "candidates": mốc là ngày mới nhất trong các bài còn cơ hội lên hạng 1,
        tức điểm x (1 + alpha) >= điểm cao nhất. Bài kém hơn mức đó thì dù được
        thưởng tối đa cũng không vượt được bài đầu, nên không có lý do để ngày
        đăng của nó quyết định "thế nào là mới". Hệ quả quan trọng: crawl thêm
        bài không liên quan (dù mới tới đâu) không làm đổi thứ hạng câu hỏi này.
        """
        ordinals = self._date_ordinals
        if self.freshness_reference == "corpus" or ordinals is None:
            return self.recency
        known = ~np.isnan(ordinals)
        top = float(rank_scores.max()) if rank_scores.size else 0.0
        if top <= 0 or not known.any():
            return self.recency
        pool = known & (rank_scores > 0) & (rank_scores * (1.0 + self.freshness_alpha) >= top)
        reference = ordinals[pool].max() if pool.any() else np.nanmax(ordinals)

        age = np.clip(reference - np.where(known, ordinals, reference), 0.0, None)
        values = 0.5 ** (age / self.freshness_halflife)
        # Bài không rõ ngày: trung vị của phần còn lại — không thưởng, không phạt
        # (cùng quy ước với dates.compute_recency).
        return np.where(known, values, float(np.median(values[known])))

    # -- dự phòng gõ sai ------------------------------------------------------
    def fuzzy_query_text(self, query: str) -> str:
        """Phần mang nội dung của câu hỏi, đã bỏ dấu, để sinh n-gram ký tự."""
        tokens, _ = self._query_tokens(query)
        return fold_for_chars(" ".join(tokens))

    def fuzzy_scores(self, query: str) -> np.ndarray:
        """Cosine giữa n-gram ký tự của câu hỏi và của tiêu đề mọi bài."""
        q = self.char_vectorizer.transform(
            [make_char_ngrams(self.fuzzy_query_text(query), FUZZY_CHAR_N)])
        if q.nnz == 0:
            return np.zeros(self.char_matrix.shape[0])
        return cosine_similarity(q, self.char_matrix)[0]

    def _make_result(self, doc_id: int, query: str, score, base_score,
                     scored: _Scored) -> RetrievalResult:
        row = self.df.iloc[doc_id]
        return RetrievalResult(
            doc_id=doc_id,
            score=float(score),
            title=str(row.get("title", "")),
            category=str(row.get("category", "")),
            url=str(row.get("url", "")),
            description=normalize_basic(row.get("description", "")),
            snippet=self.best_sentences(doc_id, query),
            matched_terms=self._matched_terms(scored.q_vec, doc_id, folded=scored.folded),
            published_date=(self.published_dates[doc_id] if self.published_dates else None),
            base_score=float(base_score),
        )

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
