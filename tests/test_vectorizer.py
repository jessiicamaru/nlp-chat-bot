"""
test_vectorizer.py — Đối chiếu TF-IDF tự cài đặt với scikit-learn.

Mục đích: chứng minh phần "from scratch" là ĐÚNG chứ không chỉ "chạy được".
sklearn chỉ đóng vai trò tham chiếu, không được dùng trong runtime của chatbot.

Chạy:  .venv/Scripts/python.exe tests/test_vectorizer.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sklearn.feature_extraction.text import TfidfVectorizer as SkTfidf
from sklearn.metrics.pairwise import cosine_similarity as sk_cosine

from vectorizer import TfidfVectorizer, cosine_similarity, make_char_ngrams, make_ngrams

DOCS = [
    "xử_lý ngôn_ngữ tự_nhiên là một nhánh của trí_tuệ nhân_tạo",
    "ngôn_ngữ lập_trình python rất phổ_biến trong trí_tuệ nhân_tạo",
    "python được dùng nhiều trong xử_lý dữ_liệu và học_máy",
    "học_máy và học_sâu là hai nhánh của trí_tuệ nhân_tạo",
    "báo_chí việt_nam đưa tin về du_lịch và kinh_tế",
]
QUERIES = [
    "python trí_tuệ nhân_tạo",
    "du_lịch việt_nam",
    "không_có_term_nào_khớp_cả",   # kiểm tra vector rỗng, tránh chia 0
]

TOKENIZED = [d.split() for d in DOCS]
Q_TOKENIZED = [q.split() for q in QUERIES]

PASS, FAIL = 0, 0


def check(name, ours, theirs, atol=1e-9):
    global PASS, FAIL
    ok = np.allclose(ours, theirs, atol=atol)
    max_diff = float(np.max(np.abs(np.asarray(ours) - np.asarray(theirs))))
    status = "PASS" if ok else "FAIL"
    print(f"  [{status}] {name:<48} max_diff={max_diff:.3e}")
    if ok:
        PASS += 1
    else:
        FAIL += 1


def run_case(ngram_range, sublinear_tf, min_df=1, max_df=1.0):
    label = f"ngram={ngram_range} sublinear={sublinear_tf} min_df={min_df} max_df={max_df}"
    print(f"\n--- {label} ---")

    ours = TfidfVectorizer(
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        min_df=min_df,
        max_df=max_df,
    )
    X_ours = ours.fit_transform(TOKENIZED)

    # sklearn: text đã tách từ sẵn nên tokenizer chỉ cần split theo khoảng trắng.
    sk = SkTfidf(
        ngram_range=ngram_range,
        sublinear_tf=sublinear_tf,
        min_df=min_df,
        max_df=max_df,
        analyzer=lambda doc: make_ngrams(doc.split(), ngram_range),
        norm="l2",
        smooth_idf=True,
    )
    X_sk = sk.fit_transform(DOCS)

    # 1. Vocabulary phải trùng khớp hoàn toàn.
    v_ours = sorted(ours.vocabulary_)
    v_sk = sorted(sk.vocabulary_)
    global PASS, FAIL
    if v_ours == v_sk:
        print(f"  [PASS] vocabulary khớp ({len(v_ours)} term)")
        PASS += 1
    else:
        only_ours = set(v_ours) - set(v_sk)
        only_sk = set(v_sk) - set(v_ours)
        print(f"  [FAIL] vocabulary lệch | chi-co-ta={list(only_ours)[:5]} "
              f"chi-co-sklearn={list(only_sk)[:5]}")
        FAIL += 1
        return

    # sklearn sắp xếp vocabulary theo alphabet giống ta, nhưng ta map lại cho chắc.
    order = [sk.vocabulary_[t] for t in ours.feature_names_]

    check("idf_", ours.idf_, sk.idf_[order])
    check("ma trận TF-IDF của corpus", X_ours.toarray(), X_sk.toarray()[:, order])

    Q_ours = ours.transform(Q_TOKENIZED)
    Q_sk = sk.transform(QUERIES)
    check("vector TF-IDF của query", Q_ours.toarray(), Q_sk.toarray()[:, order])

    check(
        "cosine similarity (query x corpus)",
        cosine_similarity(Q_ours, X_ours),
        sk_cosine(Q_sk, X_sk),
    )


def test_edge_cases():
    print("\n--- edge case ---")
    global PASS, FAIL

    v = TfidfVectorizer(ngram_range=(1, 2))
    v.fit(TOKENIZED)

    # Query rỗng không được làm chương trình chết vì chia cho 0.
    empty = v.transform([[]])
    sims = cosine_similarity(empty, v.transform(TOKENIZED))
    if np.all(sims == 0.0) and not np.any(np.isnan(sims)):
        print("  [PASS] query rỗng -> similarity toàn 0, không NaN")
        PASS += 1
    else:
        print(f"  [FAIL] query rỗng -> {sims}")
        FAIL += 1

    # Mọi hàng khác 0 phải có chuẩn L2 = 1.
    X = v.transform(TOKENIZED)
    norms = np.sqrt(np.asarray(X.multiply(X).sum(axis=1)).ravel())
    if np.allclose(norms, 1.0):
        print("  [PASS] mọi document đều có chuẩn L2 = 1")
        PASS += 1
    else:
        print(f"  [FAIL] chuẩn L2 = {norms}")
        FAIL += 1

    # n-gram sinh đúng số lượng: unigram n + bigram (n-1).
    grams = make_ngrams(["a", "b", "c", "d"], (1, 2))
    if len(grams) == 4 + 3 and "a b" in grams:
        print("  [PASS] make_ngrams sinh đúng unigram + bigram")
        PASS += 1
    else:
        print(f"  [FAIL] make_ngrams -> {grams}")
        FAIL += 1


def test_bm25():
    """Đối chiếu BM25 vector hóa với cài đặt NGÂY THƠ viết thẳng từ công thức.

    sklearn không có BM25, nên tham chiếu ở đây là một bản vòng lặp viết theo
    đúng định nghĩa, dễ đọc tới mức kiểm được bằng mắt. Nếu bản vector hóa
    (dùng ma trận thưa) khớp bản này trên nhiều (k1, b) thì phép biến đổi sang
    ma trận là đúng.
    """
    import math
    from collections import Counter as _C

    from vectorizer import CountVectorizer, bm25_idf, bm25_scores, bm25_weights

    print("\n--- BM25: vector hóa vs cài đặt ngây thơ theo công thức ---")
    global PASS, FAIL

    cv = CountVectorizer(ngram_range=(1, 1)).fit(TOKENIZED)
    counts = cv.transform(TOKENIZED)
    idf = bm25_idf(cv.document_frequency_, cv.n_docs_)

    n = len(TOKENIZED)
    df = _C(t for doc in TOKENIZED for t in set(doc))
    avgdl = sum(len(d) for d in TOKENIZED) / n

    def naive(query, doc, k1, b):
        tf = _C(doc)
        s = 0.0
        for t in query:
            if t not in df:
                continue
            idf_t = math.log((n - df[t] + 0.5) / (df[t] + 0.5) + 1)
            f = tf[t]
            s += idf_t * f * (k1 + 1) / (f + k1 * (1 - b + b * len(doc) / avgdl))
        return s

    for k1, b in [(1.2, 0.75), (0.5, 0.0), (2.0, 1.0), (1.5, 0.3)]:
        W = bm25_weights(counts, idf, k1=k1, b=b)
        for q in Q_TOKENIZED + [["trí_tuệ", "trí_tuệ", "nhân_tạo"]]:
            ours = bm25_scores(W, cv.transform([q]))
            ref = np.array([naive(q, d, k1, b) for d in TOKENIZED])
            check(f"k1={k1} b={b} q={' '.join(q)[:22]}", ours, ref, atol=1e-12)

    # Tính chất: tf bão hòa — lặp một term nhiều lần không tăng điểm mãi mãi.
    W = bm25_weights(counts, idf, k1=1.2, b=0.75)
    long_doc = [["python"] * k for k in (1, 5, 50)]
    cv2 = CountVectorizer().fit(long_doc)
    W2 = bm25_weights(cv2.transform(long_doc), bm25_idf(cv2.document_frequency_, 3), k1=1.2, b=0.0)
    s = bm25_scores(W2, cv2.transform([["python"]]))
    idf_py = bm25_idf(cv2.document_frequency_, 3)[0]
    if s[0] < s[1] < s[2] < idf_py * 2.2 + 1e-9:
        print(f"  [PASS] tf bão hòa: {s.round(4)} < trần idf*(k1+1) = {idf_py * 2.2:.4f}")
        PASS += 1
    else:
        print(f"  [FAIL] tf không bão hòa: {s}")
        FAIL += 1


def test_char_ngrams():
    """N-gram ký tự (chỉ mục chống gõ sai) phải khớp analyzer="char_wb" của sklearn."""
    global PASS, FAIL
    print("\n--- n-gram ký tự (char_wb) ---")
    texts = ["tham hiem son doong", "son dong", "a bo ca", "iphone 17 pro max", "  x  "]
    for n in (2, 3, 4):
        analyzer = SkTfidf(analyzer="char_wb", ngram_range=(n, n), lowercase=False).build_analyzer()
        for t in texts:
            ours, theirs = make_char_ngrams(t, n), analyzer(t)
            if ours == theirs:
                PASS += 1
            else:
                print(f"  [FAIL] n={n} {t!r}: {ours} != {theirs}")
                FAIL += 1
    print(f"  [PASS] make_char_ngrams khớp sklearn char_wb trên {len(texts) * 3} ca")

    # TF-IDF trên n-gram ký tự dùng lại đúng lớp TfidfVectorizer, chỉ khác đầu vào.
    docs = ["tham hiem son doong", "dong song cuu long", "gia ve tau cat linh"]
    grams = [make_char_ngrams(d, 3) for d in docs]
    ours = TfidfVectorizer(sublinear_tf=True).fit_transform(grams).toarray()
    sk = SkTfidf(analyzer="char_wb", ngram_range=(3, 3), lowercase=False, sublinear_tf=True)
    ref = sk.fit_transform(docs)
    order = [sk.vocabulary_[g] for g in sorted(sk.vocabulary_)]
    check("tfidf trên n-gram ký tự", ours, ref.toarray()[:, order])


if __name__ == "__main__":
    print("=" * 74)
    print("ĐỐI CHIẾU TF-IDF TỰ CÀI ĐẶT  vs  scikit-learn")
    print("=" * 74)

    run_case((1, 1), False)
    run_case((1, 2), False)
    run_case((1, 2), True)
    run_case((1, 2), False, min_df=2)
    run_case((1, 1), False, max_df=0.6)

    test_edge_cases()
    test_bm25()
    test_char_ngrams()

    print("\n" + "=" * 74)
    print(f"KẾT QUẢ: {PASS} pass / {FAIL} fail")
    print("=" * 74)
    sys.exit(1 if FAIL else 0)
