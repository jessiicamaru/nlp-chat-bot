"""
intent_classifier.py — Phân loại ý định (intent) bằng Multinomial Naive Bayes tự cài đặt.

Vì sao chọn Naive Bayes cho bài toán này:
  - Tập train rất nhỏ (164 câu, 14 intent). Mô hình phức tạp sẽ overfit ngay.
  - Naive Bayes hội tụ nhanh với ít dữ liệu và cho ra XÁC SUẤT, nhờ đó ta đặt
    được ngưỡng tin cậy: dưới ngưỡng thì chuyển sang retrieval thay vì đoán bừa.
  - Toàn bộ mô hình chỉ là hai bảng đếm -> giải thích được từng dự đoán.

Công thức (Multinomial NB với làm mịn Laplace):

    P(c)      = số document lớp c / tổng số document

    P(t | c)  = (count(t, c) + alpha) / (sum_t' count(t', c) + alpha * |V|)

    c_hat     = argmax_c [ log P(c) + sum_t tf(t, d) * log P(t | c) ]

Tính trên miền log để tránh tràn số (nhân hàng trăm xác suất nhỏ sẽ về 0).
Làm mịn Laplace (alpha) cần thiết vì nếu một term chưa từng xuất hiện trong
lớp c thì P(t|c) = 0 và tích toàn bộ bị triệt tiêu, dù các term khác khớp rất tốt.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
from scipy import sparse

from config import CONFIG_INTENT, INTENT_W_NB, INTENTS_PATH, TFIDF_NGRAM_RANGE
from preprocess import fold_query, fold_tokens, has_diacritics, tokenize
from vectorizer import TfidfVectorizer, cosine_similarity


# ---------------------------------------------------------------------------
# Multinomial Naive Bayes
# ---------------------------------------------------------------------------
class MultinomialNaiveBayes:
    """Naive Bayes đa thức, viết tay bằng NumPy."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.classes_: list[str] = []
        self.class_log_prior_: np.ndarray | None = None
        self.feature_log_prob_: np.ndarray | None = None   # (n_classes, n_features)

    def fit(self, X: sparse.csr_matrix, y: list[str]) -> "MultinomialNaiveBayes":
        X = sparse.csr_matrix(X)
        self.classes_ = sorted(set(y))
        n_classes = len(self.classes_)
        n_features = X.shape[1]
        class_index = {c: i for i, c in enumerate(self.classes_)}

        # Đếm tổng trọng số term theo từng lớp.
        counts = np.zeros((n_classes, n_features), dtype=np.float64)
        class_doc_counts = np.zeros(n_classes, dtype=np.float64)

        for row_idx, label in enumerate(y):
            ci = class_index[label]
            class_doc_counts[ci] += 1
            row = X.getrow(row_idx)
            counts[ci, row.indices] += row.data

        # Tiên nghiệm P(c) trên miền log.
        self.class_log_prior_ = np.log(class_doc_counts / class_doc_counts.sum())

        # Likelihood P(t|c) có làm mịn Laplace.
        smoothed = counts + self.alpha
        self.feature_log_prob_ = np.log(
            smoothed / smoothed.sum(axis=1, keepdims=True)
        )
        return self

    def joint_log_likelihood(self, X: sparse.csr_matrix) -> np.ndarray:
        """log P(c) + sum_t tf(t,d) * log P(t|c) cho từng lớp."""
        X = sparse.csr_matrix(X)
        return np.asarray(X @ self.feature_log_prob_.T) + self.class_log_prior_

    def predict_proba(self, X: sparse.csr_matrix) -> np.ndarray:
        """Chuyển log-likelihood về xác suất bằng softmax ổn định số học."""
        jll = self.joint_log_likelihood(X)
        # Trừ max từng hàng trước khi exp -> tránh tràn số.
        jll = jll - jll.max(axis=1, keepdims=True)
        probs = np.exp(jll)
        return probs / probs.sum(axis=1, keepdims=True)

    def predict(self, X: sparse.csr_matrix) -> list[str]:
        idx = self.joint_log_likelihood(X).argmax(axis=1)
        return [self.classes_[i] for i in idx]


# ---------------------------------------------------------------------------
# Bộ phân loại intent hoàn chỉnh (preprocessing + TF-IDF + NB)
# ---------------------------------------------------------------------------
class IntentClassifier:
    """Gói toàn bộ: câu thô -> tiền xử lý -> TF-IDF -> ensemble -> (intent, độ tin cậy).

    ENSEMBLE HAI TÍN HIỆU — đây là kết quả rút ra từ thực nghiệm, không phải
    lựa chọn tùy tiện:

    Tín hiệu 1 — Naive Bayes posterior P(c | câu).
        Ưu: có cơ sở xác suất, tổng hợp bằng chứng từ MỌI term trong câu.
        Nhược: với câu ngắn (2-3 token) phân phối rất PHẲNG. Đo thực tế:
        "cảm ơn nhé" chỉ đạt 0.189 dù intent đúng hiển nhiên, vì softmax trên
        14 lớp với vector đã chuẩn hóa L2 cho chênh lệch log-likelihood rất nhỏ.

    Tín hiệu 2 — cosine similarity tới pattern gần nhất của từng lớp.
        Ưu: trả lời trực tiếp câu hỏi "ta đã từng thấy câu nào giống thế này
        chưa?", rất nhạy với câu ngắn. "bye" -> 1.000 vì trùng khớp pattern.
        Nhược: chỉ nhìn MỘT pattern gần nhất, dễ bị đánh lừa bởi từ chung
        chung ("hôm nay" kéo "thời tiết sao hỏa hôm nay" lên 0.478).

    Hai nhược điểm này bù trừ cho nhau, nên ta lấy trung bình có trọng số.
    Trọng số `w_nb` và ngưỡng chấp nhận được dò bằng `evaluate.py` trên tập
    DEV, chứ không chọn bằng cảm tính.

    Kết quả dò hiện hành là w_nb = 1.0, tức NAIVE BAYES THUẦN: trên dev 55 câu,
    ensemble không còn thắng (docs/06, mục 4.2). Tín hiệu cosine được giữ trong
    mã để thí nghiệm tái lập được. Hệ quả ghi nhận: câu rất ngắn như "thanks
    nhé" (0.18) lại rơi xuống dưới ngưỡng 0.25 — đúng loại lỗi ensemble từng sửa.
    """

    def __init__(self, alpha: float = 0.3, ngram_range=TFIDF_NGRAM_RANGE, w_nb: float = INTENT_W_NB):
        # alpha nhỏ hơn 1.0 vì tập train nhỏ và các intent tách biệt khá rõ;
        # làm mịn quá mạnh sẽ kéo mọi lớp về gần nhau.
        self.vectorizer = TfidfVectorizer(ngram_range=ngram_range, min_df=1, sublinear_tf=True)
        self.model = MultinomialNaiveBayes(alpha=alpha)
        self.w_nb = w_nb
        self.intents: dict[str, dict] = {}
        self.fallback_responses: list[str] = []

        # Lưu lại ma trận pattern để tính tín hiệu cosine lúc dự đoán.
        self.pattern_matrix = None
        self.pattern_labels: np.ndarray | None = None

        # Bộ thứ hai huấn luyện trên pattern ĐÃ BỎ DẤU, phục vụ người dùng gõ
        # không dấu ("cam on nhe"). Xem giải thích ở preprocess.strip_accents.
        self.folded_vectorizer = TfidfVectorizer(
            ngram_range=ngram_range, min_df=1, sublinear_tf=True
        )
        self.folded_model = MultinomialNaiveBayes(alpha=alpha)
        self.folded_pattern_matrix = None

        self._trained = False

    # -- nạp dữ liệu ---------------------------------------------------------
    def load_intents(self, path: Path | None = None) -> tuple[list[str], list[str]]:
        path = Path(path) if path else INTENTS_PATH
        data = json.loads(path.read_text(encoding="utf-8"))

        texts: list[str] = []
        labels: list[str] = []
        for intent in data["intents"]:
            tag = intent["tag"]
            self.intents[tag] = intent
            for pattern in intent["patterns"]:
                texts.append(pattern)
                labels.append(tag)

        self.fallback_responses = data.get("fallback", {}).get("responses", [])
        return texts, labels

    # -- huấn luyện ----------------------------------------------------------
    def fit(self, texts: list[str], labels: list[str]) -> "IntentClassifier":
        docs = [tokenize(t, CONFIG_INTENT) for t in texts]
        X = self.vectorizer.fit_transform(docs)
        self.model.fit(X, labels)

        # Giữ nguyên ma trận pattern cho tín hiệu cosine.
        self.pattern_matrix = X
        self.pattern_labels = np.array(labels)

        # Bản bỏ dấu, dùng cùng nhãn.
        folded_docs = [fold_tokens(doc) for doc in docs]
        Xf = self.folded_vectorizer.fit_transform(folded_docs)
        self.folded_model.fit(Xf, labels)
        self.folded_pattern_matrix = Xf

        self._trained = True
        return self

    # -- tính điểm -----------------------------------------------------------
    def _class_scores(self, text: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Trả (điểm ensemble, xác suất NB, cosine theo lớp), hoặc None nếu không có căn cứ."""
        tokens = tokenize(text, CONFIG_INTENT)
        if not tokens:
            return None

        # Câu không dấu -> dùng bộ mô hình đã bỏ dấu.
        folded = not has_diacritics(text)
        if folded:
            tokens = fold_query(text)
            vectorizer, model = self.folded_vectorizer, self.folded_model
            patterns = self.folded_pattern_matrix
        else:
            vectorizer, model = self.vectorizer, self.model
            patterns = self.pattern_matrix

        X = vectorizer.transform([tokens])
        # Không term nào khớp vocabulary -> mô hình không có bằng chứng gì.
        if X.nnz == 0:
            return None

        nb_probs = model.predict_proba(X)[0]

        # Cosine tới pattern gần nhất CỦA TỪNG LỚP.
        sims = cosine_similarity(X, patterns)[0]
        cos_per_class = np.zeros(len(model.classes_), dtype=np.float64)
        for i, cls in enumerate(model.classes_):
            mask = self.pattern_labels == cls
            if mask.any():
                cos_per_class[i] = sims[mask].max()

        ensemble = self.w_nb * nb_probs + (1.0 - self.w_nb) * cos_per_class
        return ensemble, nb_probs, cos_per_class

    def train_from_file(self, path: Path | None = None) -> "IntentClassifier":
        texts, labels = self.load_intents(path)
        return self.fit(texts, labels)

    # -- dự đoán -------------------------------------------------------------
    def predict(self, text: str) -> tuple[str, float]:
        """Trả (tag, độ tin cậy ensemble). Không có căn cứ -> ('', 0.0)."""
        if not self._trained:
            raise RuntimeError("Phai train truoc khi predict.")

        scores = self._class_scores(text)
        if scores is None:
            return "", 0.0

        ensemble = scores[0]
        best = int(ensemble.argmax())
        return self.model.classes_[best], float(ensemble[best])

    def predict_top_k(self, text: str, k: int = 3) -> list[tuple[str, float]]:
        """Xem k intent có điểm cao nhất — dùng khi phân tích lỗi."""
        scores = self._class_scores(text)
        if scores is None:
            return []
        ensemble = scores[0]
        order = np.argsort(-ensemble)[:k]
        return [(self.model.classes_[i], float(ensemble[i])) for i in order]

    # -- phản hồi ------------------------------------------------------------
    def get_response(self, tag: str, rng: random.Random | None = None) -> str:
        rng = rng or random
        intent = self.intents.get(tag)
        if not intent or not intent.get("responses"):
            return self.get_fallback(rng)
        return rng.choice(intent["responses"])

    def get_fallback(self, rng: random.Random | None = None) -> str:
        rng = rng or random
        if not self.fallback_responses:
            return "Xin lỗi, mình chưa hiểu ý bạn."
        return rng.choice(self.fallback_responses)

    def get_action(self, tag: str) -> str:
        return self.intents.get(tag, {}).get("action", "reply")

    # -- giải thích ----------------------------------------------------------
    def explain(self, text: str, k: int = 8) -> dict:
        """Cho biết term nào đẩy câu về intent nào — phục vụ error analysis."""
        tokens = tokenize(text, CONFIG_INTENT)
        X = self.vectorizer.transform([tokens])
        tag, conf = self.predict(text)

        contributions: list[tuple[str, float]] = []
        if tag and X.nnz:
            ci = self.model.classes_.index(tag)
            names = self.vectorizer.feature_names_
            for idx, val in zip(X.indices, X.data):
                # Đóng góp của term vào log-likelihood của lớp thắng cuộc.
                contributions.append((names[idx], float(val * self.model.feature_log_prob_[ci, idx])))
            contributions.sort(key=lambda x: x[1], reverse=True)

        return {
            "input": text,
            "tokens": tokens,
            "intent": tag,
            "confidence": conf,
            "top_intents": self.predict_top_k(text, 3),
            "term_contributions": contributions[:k],
            "oov_ratio": 1.0 - (X.nnz / len(tokens)) if tokens else 1.0,
        }
