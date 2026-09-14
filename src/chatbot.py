"""
chatbot.py — Bộ điều phối chính của chatbot.

Luồng xử lý một lượt chat:

    câu người dùng
        |
        v
    [0] prepare_user_text      chuẩn hóa teencode (học từ ViLexNorm), giữ nguyên
        |                      "hệ quy chiếu dấu" nếu câu gốc không dấu
        v
    [1] entities.extract       NER + Regex + nhận diện chuyên mục (Lab 01)
        |
        v
    [2] IntentClassifier       (TF-IDF + Naive Bayes tự cài đặt, Lab 03 + 04)
        |
        +-- conf >= INTENT_THRESHOLD và action != retrieve
        |       --> thực thi action (reply / help / list_categories / stats /
        |           browse_category / summarize / source)
        |
        +-- ngược lại --> [3] NewsRetriever
                              chọn index có dấu / không dấu (âm tiết)
                              XẾP HẠNG: cosine (hoặc BM25) x (1 + alpha*recency)
                                        recency tính theo bài MỚI NHẤT trong các
                                        bài đang cạnh tranh, không theo cả corpus
                              CHẤP NHẬN: cosine thuần >= RETRIEVAL_THRESHOLD
                              (x CATEGORY_SCOPED_THRESHOLD_FACTOR nếu đã nêu mục)
                              |
                              +-- đạt  --> 2 câu sát nhất của bài + ngày đăng + nguồn
                              +-- không --> [3b] dự phòng gõ sai: cosine từ + cosine
                                              n-gram ký tự tiêu đề >= FUZZY_THRESHOLD
                                              |
                                              +-- đạt  --> trả lời, kèm lời nhắc
                                              |            "có thể bạn gõ nhầm"
                                              +-- không --> fallback

Lớp RAG tùy chọn (rag.py) bọc bên ngoài lớp này và chỉ chạy khi [3] đã có bằng chứng.

Thiết kế then chốt: **luôn có ngưỡng tin cậy**. Chatbot chỉ trả lời khi có
căn cứ định lượng; không đủ căn cứ thì nói thẳng là không biết. Điều này tránh
được lỗi tệ nhất của chatbot retrieval — trả về một bài báo ngẫu nhiên nhưng
nói bằng giọng chắc chắn.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from config import (
    CATEGORY_SCOPED_THRESHOLD_FACTOR,
    CONFIG_RETRIEVAL,
    QUERY_FRAME_WORDS,
    CORPUS_RAW_PATH,
    INTENT_THRESHOLD,
    INTENTS_PATH,
    RANDOM_SEED,
    RETRIEVAL_THRESHOLD,
    TOP_K,
)
from dialogue import DialogueState, Turn
from entities import extract, expand_query
from preprocess import tokenize
from intent_classifier import IntentClassifier
from normalizer import TeencodeNormalizer, prepare_user_text
from retriever import NewsRetriever, RetrievalResult


@dataclass
class BotReply:
    """Câu trả lời của bot kèm toàn bộ dấu vết để giải thích và debug."""

    text: str
    intent: str = ""
    confidence: float = 0.0
    route: str = "fallback"                 # intent | retrieval | fallback
    results: list[RetrievalResult] = field(default_factory=list)
    entities: dict = field(default_factory=dict)
    normalized_input: str = ""              # câu sau khi chuẩn hóa teencode
    normalizations: list = field(default_factory=list)  # các cặp (gốc, đã sửa)

    @property
    def sources(self) -> list[dict]:
        return [
            {
                "title": r.title,
                "url": r.url,
                "category": r.category,
                "score": round(r.score, 4),
                "cosine": round(r.base_score, 4),
                "published": r.published_str,
                "match": r.match,
            }
            for r in self.results
        ]


class NewsChatbot:
    """Chatbot tin tức tiếng Việt."""

    def __init__(
        self,
        intent_threshold: float = INTENT_THRESHOLD,
        retrieval_threshold: float = RETRIEVAL_THRESHOLD,
        use_ner: bool = True,
        use_normalizer: bool = True,
        seed: int | None = RANDOM_SEED,
        intent_w_nb: float | None = None,
        freshness_alpha: float | None = None,
        freshness_halflife: float | None = None,
        ranking: str | None = None,
        bm25_k1: float | None = None,
        bm25_b: float | None = None,
        freshness_reference: str | None = None,
        fuzzy_threshold: float | None = None,
        use_fuzzy: bool = True,
    ):
        """Các tham số None -> dùng giá trị mặc định trong config.py.

        Cho phép truyền tham số trực tiếp để evaluate.py dựng bot với đúng bộ
        tham số đã dò trên DEV rồi đo trên TEST.
        """
        self.intent_threshold = intent_threshold
        self.use_ner = use_ner
        self.rng = random.Random(seed)

        # Chuẩn hóa teencode chạy TRƯỚC mọi bước khác. Chỉ áp dụng cho câu
        # người dùng gõ — corpus báo chí vốn đã là văn viết chuẩn.
        self.normalizer = TeencodeNormalizer() if use_normalizer else None

        self.classifier = (IntentClassifier(w_nb=intent_w_nb)
                           if intent_w_nb is not None else IntentClassifier())
        retriever_kwargs = {"threshold": retrieval_threshold}
        if freshness_alpha is not None:
            retriever_kwargs["freshness_alpha"] = freshness_alpha
        if freshness_halflife is not None:
            retriever_kwargs["freshness_halflife"] = freshness_halflife
        for key, val in (("ranking", ranking), ("bm25_k1", bm25_k1), ("bm25_b", bm25_b),
                         ("freshness_reference", freshness_reference),
                         ("fuzzy_threshold", fuzzy_threshold)):
            if val is not None:
                retriever_kwargs[key] = val
        if not use_fuzzy:
            retriever_kwargs["fuzzy_threshold"] = None
        self.retriever = NewsRetriever(**retriever_kwargs)
        self.state = DialogueState()
        self._ready = False

    # -- khởi tạo ------------------------------------------------------------
    def train(
        self,
        corpus_path: Path | None = None,
        intents_path: Path | None = None,
        df: pd.DataFrame | None = None,
        use_cache: bool = True,
        verbose: bool = False,
    ) -> "NewsChatbot":
        """Huấn luyện classifier và dựng index truy hồi.

        `use_cache=True` nạp lại index đã lưu nếu corpus chưa đổi — giảm thời
        gian khởi động từ ~21 giây xuống dưới 1 giây.
        """
        self.classifier.train_from_file(intents_path or INTENTS_PATH)

        if df is None:
            df = pd.read_csv(corpus_path or CORPUS_RAW_PATH)
        df = df.dropna(subset=["title", "text"]).reset_index(drop=True)

        if use_cache:
            self.retriever.fit_cached(df, verbose=verbose)
        else:
            self.retriever.fit(df)

        self._ready = True
        return self

    # -- vòng xử lý chính ----------------------------------------------------
    def respond(self, user_text: str) -> BotReply:
        if not self._ready:
            raise RuntimeError("Phai goi train() truoc khi chat.")

        raw_text = (user_text or "").strip()
        if not raw_text:
            return BotReply(text="Bạn chưa nhập gì cả. Hãy hỏi mình điều gì đó nhé!")

        # [0] Chuẩn hóa teencode: "bt gì về vụ iphone k b"
        #                      -> "biết gì về vụ iphone không bạn"
        # Phải chạy trước tách từ, vì word_tokenize không biết "bt", "k", "đc".
        # Dùng chung với evaluate.py — xem normalizer.prepare_user_text để biết
        # vì sao phải giữ nguyên "hệ quy chiếu dấu" của người dùng.
        user_text, normalizations = prepare_user_text(raw_text, self.normalizer)

        # [1] Trích xuất thực thể + chuyên mục.
        info = extract(user_text, use_ner=self.use_ner)
        self.state.remember_category(info.category)

        # [2] Phân loại intent.
        tag, confidence = self.classifier.predict(user_text)
        action = self.classifier.get_action(tag) if tag else "retrieve"

        if tag and confidence >= self.intent_threshold and action != "retrieve":
            reply = self._handle_action(action, tag, user_text, info)
            reply.intent, reply.confidence = tag, confidence
        else:
            # [3] Không đủ tự tin về intent -> để retrieval quyết định.
            reply = self._handle_retrieval(user_text, info)
            reply.intent = tag or ""
            reply.confidence = confidence

        reply.normalized_input = user_text
        reply.normalizations = normalizations
        reply.entities = {
            "category": info.category,
            "persons": info.persons,
            "locations": info.locations,
            "organizations": info.organizations,
            "regex": info.regex_matches,
        }

        # Gợi ý chủ động khi người dùng bế tắc nhiều lượt liên tiếp.
        if self.state.needs_help_nudge() and reply.route == "fallback":
            reply.text += "\n\n" + self._help_text()

        self.state.add_turn(
            Turn(
                user_text=raw_text,
                bot_text=reply.text,
                intent=reply.intent,
                confidence=reply.confidence,
                route=reply.route,
            )
        )
        return reply

    # -- các action ----------------------------------------------------------
    def _handle_action(self, action: str, tag: str, user_text: str, info) -> BotReply:
        handlers = {
            "reply": self._act_reply,
            "help": self._act_help,
            "list_categories": self._act_list_categories,
            "stats": self._act_stats,
            "browse_category": self._act_browse,
            "summarize": self._act_summarize,
            "source": self._act_source,
        }
        handler = handlers.get(action, self._act_reply)
        return handler(tag, user_text, info)

    def _act_reply(self, tag: str, user_text: str, info) -> BotReply:
        return BotReply(text=self.classifier.get_response(tag, self.rng), route="intent")

    def _help_text(self) -> str:
        cats = ", ".join(self.retriever.list_categories())
        return (
            "Mình có thể giúp bạn:\n"
            "  • Tìm tin theo từ khóa — ví dụ: \"cho tôi biết về đảo Hải Nam\"\n"
            "  • Xem tin theo chuyên mục — ví dụ: \"tin công nghệ mới nhất\"\n"
            "  • Tóm tắt bài vừa xem — \"tóm tắt bài đó\"\n"
            "  • Lấy link bài gốc — \"cho mình link\"\n"
            "  • Xem thống kê kho dữ liệu — \"có bao nhiêu bài báo\"\n"
            f"\nChuyên mục hiện có: {cats}"
        )

    def _act_help(self, tag: str, user_text: str, info) -> BotReply:
        return BotReply(text=self._help_text(), route="intent")

    def _act_list_categories(self, tag: str, user_text: str, info) -> BotReply:
        cats = self.retriever.list_categories()
        lines = [f"  • {name} ({count} bài)" for name, count in cats.items()]
        return BotReply(
            text="Kho tin của mình gồm các chuyên mục sau:\n" + "\n".join(lines),
            route="intent",
        )

    def _act_stats(self, tag: str, user_text: str, info) -> BotReply:
        s = self.retriever.stats()
        text = (
            "Thống kê kho dữ liệu:\n"
            f"  • Số bài báo      : {s['n_documents']}\n"
            f"  • Số chuyên mục   : {s['n_categories']}\n"
            f"  • Kích thước từ vựng: {s['vocabulary_size']:,} term\n"
            f"  • Độ dài trung bình : {s['avg_text_length']:,} ký tự/bài\n"
            f"  • Nguồn            : {', '.join(s['sources'])}"
        )
        return BotReply(text=text, route="intent")

    def _act_browse(self, tag: str, user_text: str, info) -> BotReply:
        category = info.category or self.state.last_category
        if not category:
            # Classifier đoán là "duyệt chuyên mục" nhưng không có chuyên mục nào
            # được nêu. Nếu câu vẫn có từ khóa nội dung ("tin ve dao hai nam"),
            # thì người dùng đang hỏi một chủ đề cụ thể — đi tìm kiếm sẽ hữu ích
            # hơn nhiều so với việc hỏi ngược lại "bạn muốn xem mục nào?".
            if self._has_topic_beyond_category(user_text, ""):
                retrieved = self._handle_retrieval(user_text, info)
                if retrieved.route != "fallback":
                    return retrieved

            cats = ", ".join(self.retriever.list_categories())
            return BotReply(
                text=f"Bạn muốn xem chuyên mục nào? Hiện có: {cats}",
                route="intent",
            )

        # "tin du lịch ninh bình" vừa nêu chuyên mục VỪA nêu chủ đề cụ thể.
        # Liệt kê cả mục sẽ bỏ mất "ninh bình" — thứ người dùng thực sự hỏi.
        # Nếu còn từ khóa nội dung ngoài tên chuyên mục thì tìm kiếm trong mục
        # đó thay vì liệt kê.
        if self._has_topic_beyond_category(user_text, category):
            # Dùng ngưỡng THẤP HƠN khi đã lọc theo chuyên mục.
            # Lý do: ngưỡng 0.12 được dò trên toàn corpus 381 bài, nơi rủi ro
            # trả nhầm là cao. Khi người dùng đã tự nêu chuyên mục, tập ứng
            # viên co lại còn vài chục bài cùng chủ đề nên một điểm thấp hơn
            # vẫn là bằng chứng đủ mạnh. Ví dụ "tin du lịch ninh bình": bài
            # đúng đạt 0.097 — dưới ngưỡng toàn cục nhưng cao gấp 2.1 lần bài
            # đứng thứ hai trong cùng chuyên mục.
            scoped = self.retriever.search(
                expand_query(user_text, info),
                top_k=TOP_K,
                category=category,
                min_score=self.retriever.threshold * CATEGORY_SCOPED_THRESHOLD_FACTOR,
            )
            if scoped:
                self.state.remember_results(scoped, user_text)
                return self._format_retrieval(scoped, route="retrieval")

        results = self.retriever.browse(category, n=5)
        if not results:
            return BotReply(
                text=f"Kho dữ liệu chưa có bài nào trong chuyên mục \"{category}\".",
                route="intent",
            )

        self.state.remember_results(results, user_text)
        lines = [f"Các bài mới nhất trong chuyên mục **{category}**:"]
        for i, r in enumerate(results, 1):
            lines.append(f"\n{i}. {r.title}")
            if r.description:
                lines.append(f"   {r.description[:130]}...")
        return BotReply(text="\n".join(lines), route="intent", results=results)

    def _act_summarize(self, tag: str, user_text: str, info) -> BotReply:
        doc = self.state.current_doc
        if doc is None:
            return BotReply(
                text="Mình chưa biết bạn muốn tóm tắt bài nào. Bạn tìm một bài trước nhé, "
                     "ví dụ: \"cho tôi biết về đảo Hải Nam\".",
                route="intent",
            )

        # Tóm tắt = chọn câu tiêu biểu nhất theo TF-IDF, lấy chính tiêu đề làm
        # "query" vì tiêu đề là bản cô đọng chủ đề bài báo.
        summary = self.retriever.best_sentences(doc.doc_id, doc.title, n=3)
        text = (
            f"**{doc.title}**\n\n"
            f"{summary}\n\n"
            f"Nguồn: {doc.url}"
        )
        return BotReply(text=text, route="intent", results=[doc])

    def _act_source(self, tag: str, user_text: str, info) -> BotReply:
        if not self.state.last_results:
            return BotReply(
                text="Mình chưa nhắc tới bài nào trong phiên này. Bạn thử hỏi một chủ đề trước nhé.",
                route="intent",
            )
        lines = ["Nguồn các bài mình vừa nhắc tới:"]
        for i, r in enumerate(self.state.last_results, 1):
            lines.append(f"{i}. {r.title}\n   {r.url}")
        return BotReply(text="\n".join(lines), route="intent", results=self.state.last_results)

    # -- retrieval -----------------------------------------------------------
    @staticmethod
    def _has_topic_beyond_category(user_text: str, category: str) -> bool:
        """Câu hỏi còn từ khóa nội dung nào ngoài tên chuyên mục không?

        Bỏ đi các token của chính tên chuyên mục và các từ khung câu ("tin",
        "xem", "mới nhất"...). Nếu vẫn còn token -> người dùng đang hỏi một
        chủ đề cụ thể chứ không chỉ muốn duyệt mục.
        """
        # Dùng chung QUERY_FRAME_WORDS với retriever, để "câu này còn nội dung
        # gì không?" và "vector query còn term gì không?" luôn trả lời giống
        # nhau. Hai nơi từng giữ hai danh sách riêng và đã lệch nhau.
        #
        # So khớp ở mức ÂM TIẾT, không so nguyên token. Lý do: `word_tokenize`
        # tách CÙNG một cụm khác nhau tùy ngữ cảnh —
        #     word_tokenize("Sức khỏe")                    -> ["sức", "khỏe"]
        #     word_tokenize("... tin sức khỏe gì luôn")    -> ["sức_khỏe"]
        # nên so theo token thì "sức_khỏe" không khớp {"sức","khỏe"}, và câu
        # chỉ nêu đúng tên chuyên mục lại bị coi là có chủ đề riêng.
        def syllables(tokens: list[str]) -> set[str]:
            out: set[str] = set()
            for t in tokens:
                out.update(t.lower().split("_"))
            return out

        frame_syllables = syllables(list(QUERY_FRAME_WORDS))
        category_syllables = syllables(tokenize(category, CONFIG_RETRIEVAL))
        skip = category_syllables | frame_syllables

        for token in tokenize(user_text, CONFIG_RETRIEVAL):
            if token.lower() in QUERY_FRAME_WORDS:
                continue
            # Token còn mang nội dung nếu có ÍT NHẤT một âm tiết không thuộc
            # tên chuyên mục và cũng không phải từ khung.
            if any(syl not in skip for syl in token.lower().split("_")):
                return True
        return False

    def _format_retrieval(self, results: list[RetrievalResult], route: str) -> BotReply:
        """Định dạng kết quả truy hồi — dùng chung cho cả hai đường vào."""
        top = results[0]
        lines = []
        if top.match == "fuzzy":
            # Đường dự phòng gõ sai: không khớp chính xác từ nào đủ mạnh, bài được
            # chọn vì TIÊU ĐỀ gần cách viết của câu hỏi. Nói rõ để người dùng
            # tự kiểm tra, thay vì trả lời bằng giọng chắc chắn như khớp thật.
            lines += ["_(Không tìm thấy từ khóa khớp chính xác — có thể bạn gõ nhầm. "
                      "Bài có tiêu đề gần nhất với câu hỏi:)_", ""]
        lines += [
            f"**{top.title}**",
            "",
            top.snippet,
            "",
            f"Nguồn: {top.url}",
            f"_(chuyên mục: {top.category} · độ tương đồng: {top.score:.2f}"
            + (" · khớp gần đúng" if top.match == "fuzzy" else "") + ")_",
        ]

        if len(results) > 1:
            lines.append("\nCác bài liên quan khác:")
            for r in results[1:]:
                lines.append(f"  • {r.title} ({r.score:.2f})")

        return BotReply(text="\n".join(lines), route=route, results=results)

    def _handle_retrieval(self, user_text: str, info) -> BotReply:
        # Câu chỉ nêu ĐÚNG tên chuyên mục, không có chủ đề cụ thể nào khác
        # ("không biết tin sức khỏe gì luôn") — đây thực chất là yêu cầu duyệt
        # mục. Đem đi tìm kiếm sẽ trượt, vì tên chuyên mục có idf rất thấp
        # (xuất hiện ở mọi bài trong mục đó) nên điểm cosine không bao giờ
        # đạt ngưỡng.
        if info.category and not self._has_topic_beyond_category(user_text, info.category):
            return self._act_browse("", user_text, info)

        # Nhân đôi thực thể để tăng trọng số tên riêng trong vector query.
        query = expand_query(user_text, info)

        # Nếu người dùng nêu rõ chuyên mục, thu hẹp phạm vi tìm kiếm trước — và
        # dùng CÙNG ngưỡng nới lỏng như nhánh duyệt mục (_act_browse). Trước đây
        # chỉ nhánh duyệt mục có ngưỡng nới, nên cùng một câu "tin du lịch ninh
        # bình" cho kết quả khác nhau tùy độ tin cậy intent rơi trên hay dưới
        # ngưỡng 0.25 một chút (0.246 -> nhánh này -> không tìm thấy).
        scoped_min = (self.retriever.threshold * CATEGORY_SCOPED_THRESHOLD_FACTOR
                      if info.category else None)
        results = self.retriever.search(query, top_k=TOP_K, category=info.category,
                                        min_score=scoped_min)
        if not results and info.category:
            # Không có gì trong chuyên mục đó -> nới ra toàn corpus.
            results = self.retriever.search(query, top_k=TOP_K)

        if not results:
            return BotReply(text=self.classifier.get_fallback(self.rng), route="fallback")

        self.state.remember_results(results, user_text)
        return self._format_retrieval(results, route="retrieval")

    # -- tiện ích ------------------------------------------------------------
    def explain(self, user_text: str) -> dict:
        """Gộp chẩn đoán của cả intent classifier và retriever cho một câu."""
        raw = user_text
        user_text, norm_pairs = prepare_user_text(raw, self.normalizer)
        info = extract(user_text, use_ner=self.use_ner)
        return {
            "raw_input": raw,
            "normalized_input": user_text,
            "normalizations": norm_pairs,
            "intent": self.classifier.explain(user_text),
            "entities": info.summary(),
            "retrieval": self.retriever.explain(expand_query(user_text, info)),
        }

    def reset(self) -> None:
        self.state.reset()


def build_default_bot(corpus_path: Path | None = None) -> NewsChatbot:
    """Tạo và huấn luyện bot với cấu hình mặc định — dùng cho CLI, API, notebook."""
    return NewsChatbot().train(corpus_path=corpus_path)
