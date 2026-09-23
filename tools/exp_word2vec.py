"""
exp_word2vec.py — Thí nghiệm Lab 05: Word2Vec có nên vào chatbot không?

Câu hỏi thí nghiệm (docs/10):
  1. Vector trung bình Word2Vec (đúng cách Lab 05, Phần 4) xếp hạng bài báo tốt
     hơn hay kém TF-IDF, trên câu hỏi THẬT (dev/test) và trên câu DIỄN ĐẠT LẠI
     (dev_paraphrase/test_paraphrase) — loại câu mà embedding sinh ra để giải?
  2. Ghép hai tín hiệu (điểm = cosine TF-IDF + λ · cosine Word2Vec) có hơn không?
  3. Cosine Word2Vec có dùng làm NGƯỠNG CHẤP NHẬN được không (tách câu trong /
     ngoài phạm vi), như cosine TF-IDF đang làm?

Quy trình giống PHA 1.2 của evaluate.py (TF-IDF vs BM25):
  - Độ mới TẮT, để đo đúng chất lượng xếp hạng thuần.
  - Mọi lựa chọn (tham số Word2Vec, cách lấy trung bình, λ) chốt trên DEV.
  - TEST chỉ chạy MỘT lần, ở cuối, với cấu hình đã chốt.
  - Kiểm định dấu có cặp: chỉ coi là "hơn" khi p < 0.05.

Word2Vec ở đây là bản TỰ CÀI ĐẶT (src/word2vec.py) — không cần gensim.

Chạy:  .venv/Scripts/python.exe tools/exp_word2vec.py
       -> data/eval/word2vec_report.txt
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import evaluate as ev  # noqa: E402
from config import CONFIG_RETRIEVAL, CORPUS_RAW_PATH, MODELS_DIR, RETRIEVAL_THRESHOLD  # noqa: E402
from preprocess import fold_query, has_diacritics, normalize_basic, strip_frame_words, tokenize  # noqa: E402
from retriever import NewsRetriever  # noqa: E402
from word2vec import Word2Vec  # noqa: E402

TOKENS_CACHE = MODELS_DIR / "w2v_corpus_tokens.joblib"
CONFIG_GRID = [  # (vector_size, min_count) — window=5, sg, epochs=30 như BASE_CONFIG của Lab 05
    (50, 1), (50, 3), (50, 5), (100, 1), (100, 3), (100, 5),
]
LAMBDA_GRID = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
PROBE_PAIRS = [("máy_bay", "phi_cơ"), ("học_sinh", "học_trò"), ("bác_sĩ", "thầy_thuốc"),
               ("điện_thoại", "smartphone"), ("tên_lửa", "hỏa_tiễn"), ("giá", "tiền"),
               ("mỹ", "hoa_kỳ"), ("ô_tô", "xe_hơi")]
PROBE_WORDS = ["máy_bay", "bác_sĩ", "học_sinh", "điện_thoại", "iphone", "dầu", "chuối", "robot"]


def _train(cfg, toks):
    """Huấn luyện MỘT cấu hình — hàm cấp module để chạy song song nhiều tiến trình."""
    vs, mc = cfg
    t0 = time.time()
    m = Word2Vec(vector_size=vs, window=5, min_count=mc, epochs=30, seed=42).fit(toks)
    return cfg, m, time.time() - t0


def log(*args):
    print(*args, flush=True)
    REPORT.append(" ".join(str(a) for a in args))


REPORT: list[str] = []


# ---------------------------------------------------------------------------
def corpus_tokens(df: pd.DataFrame) -> list[list[str]]:
    """Token của mỗi bài (tiêu đề + mô tả + thân), cùng bộ tách từ với index TF-IDF."""
    fp = str(len(df)) + "|" + "".join(df["url"])
    if TOKENS_CACHE.exists():
        blob = joblib.load(TOKENS_CACHE)
        if blob.get("fp") == fp:
            return blob["tokens"]
    t0 = time.time()
    toks = [tokenize(" ".join(normalize_basic(r.get(c, "")) for c in ("title", "description", "text")),
                     CONFIG_RETRIEVAL) for _, r in df.iterrows()]
    joblib.dump({"fp": fp, "tokens": toks}, TOKENS_CACHE, compress=3)
    log(f"(tách từ {len(toks)} bài: {time.time() - t0:.0f}s, đã cache)")
    return toks


def query_tokens(q: str, retriever: NewsRetriever) -> list[str]:
    """Token câu hỏi theo ĐÚNG đường của bot (chuẩn hóa teencode, từ khung, ghép từ)."""
    text = ev.prepared(q)
    if not has_diacritics(text):
        return strip_frame_words(fold_query(text))        # âm tiết không dấu
    toks = strip_frame_words(tokenize(text, CONFIG_RETRIEVAL))
    return retriever._glue_known_compounds(toks, retriever.vectorizer)


def unit_rows(M: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(M, axis=1, keepdims=True)
    return M / np.where(n == 0, 1.0, n)


def rank_of(scores: np.ndarray, gold: set[int], k: int = 10) -> int | None:
    order = np.argsort(-scores, kind="stable")[:k]
    return next((i for i, d in enumerate(order, 1) if d in gold), None)


def metrics(ranks: list[int | None]) -> tuple[int, int, float]:
    return (sum(r == 1 for r in ranks), sum(bool(r) and r <= 3 for r in ranks), ev.mrr(ranks))


class Scorer:
    """Điểm TF-IDF và Word2Vec của một bộ câu hỏi trên toàn kho (tính sẵn)."""

    def __init__(self, cases, key, retriever, url_to_idx):
        self.cases = cases
        self.gold = [{url_to_idx[u] for u in c["gold_urls"]} for c in cases]
        self.qtok = [query_tokens(c[key], retriever) for c in cases]
        self.tfidf = np.array([retriever._score(ev.prepared(c[key])).base for c in cases])
        self.styles = [c.get("style", "-") for c in cases]

    def w2v(self, model: Word2Vec, D: np.ndarray, weights=None) -> np.ndarray:
        Q = np.array([model.document_vector(t, weights) for t in self.qtok])
        return unit_rows(Q) @ D.T

    def ranks(self, S: np.ndarray) -> list[int | None]:
        return [rank_of(S[i], g) for i, g in enumerate(self.gold)]


def sign_test(ranks_a, ranks_b) -> tuple[int, int, float]:
    ra = [1 / r if r else 0 for r in ranks_a]
    rb = [1 / r if r else 0 for r in ranks_b]
    w = sum(a > b for a, b in zip(ra, rb))
    l = sum(a < b for a, b in zip(ra, rb))
    return w, l, ev.sign_test_p(w, l)


def fmt(ranks) -> str:
    r1, r3, m = metrics(ranks)
    n = len(ranks)
    return f"R@1 {r1:>3}/{n} ({r1 / n:5.1%})  R@3 {r3:>3}/{n}  MRR {m:.3f}"


def separability(in_scores: list[float], oos_scores: list[float]) -> tuple[float, float]:
    """Ngưỡng tốt nhất và điểm TB(trả lời được, chặn đúng) — như PHA 1.4."""
    cands = sorted(set(in_scores) | set(oos_scores))
    best = (0.0, 0.0)
    for th in cands:
        ans = np.mean([s >= th for s in in_scores])
        blk = np.mean([s < th for s in oos_scores])
        best = max(best, ((ans + blk) / 2, th))
    return best


# ---------------------------------------------------------------------------
def main() -> int:
    df = pd.read_csv(CORPUS_RAW_PATH).dropna(subset=["title", "text"]).reset_index(drop=True)
    url_to_idx = {u: i for i, u in enumerate(df["url"])}
    retriever = NewsRetriever(freshness_alpha=0.0, fuzzy_threshold=None).fit_cached(df)
    idf = dict(zip(retriever.vectorizer.feature_names_, retriever.vectorizer.idf_))

    sets = {name: ev.load_split(name) if "para" not in name else
            __import__("json").loads((ev.EVAL_DIR / f"{name}.json").read_text(encoding="utf-8"))
            for name in ("dev", "test", "dev_paraphrase", "test_paraphrase")}
    sc = {name: Scorer(s["retrieval"], "query", retriever, url_to_idx) for name, s in sets.items()}
    oos_dev = Scorer([{"query": x["text"], "gold_urls": []} for x in sets["dev"]["out_of_scope"]],
                     "query", retriever, url_to_idx)

    toks = corpus_tokens(df)
    n_tok = sum(len(t) for t in toks)
    log("=" * 78)
    log("THÍ NGHIỆM LAB 05 — WORD2VEC (tự cài đặt) vs TF-IDF")
    log("=" * 78)
    log(f"Kho: {len(df)} bài, {n_tok:,} token sau tách từ + bỏ stopword, "
        f"{len(set(t for d in toks for t in d)):,} từ khác nhau")
    log(f"Câu hỏi: dev {len(sc['dev'].cases)}, test {len(sc['test'].cases)}, "
        f"dev_paraphrase {len(sc['dev_paraphrase'].cases)}, "
        f"test_paraphrase {len(sc['test_paraphrase'].cases)}, ngoài phạm vi (dev) {len(oos_dev.cases)}")
    log("Độ mới TẮT ở mọi phương pháp — so chất lượng xếp hạng thuần.\n")

    base = {name: s.ranks(s.tfidf) for name, s in sc.items()}

    # ------------------------------------------------ PHẦN 1: chọn cấu hình trên DEV
    log("-" * 78)
    log("PHẦN 1 — CHỌN CẤU HÌNH WORD2VEC TRÊN DEV (window=5, skip-gram, epochs=30)")
    log("-" * 78)
    log(f"{'TF-IDF (tham chiếu)':<34} dev: {fmt(base['dev'])}")
    log(f"{'':<34} para: {fmt(base['dev_paraphrase'])}\n")
    # Các cấu hình độc lập nhau -> huấn luyện song song (mỗi cấu hình vẫn cùng
    # seed, nên kết quả y hệt chạy tuần tự; chỉ nhanh hơn).
    from concurrent.futures import ProcessPoolExecutor
    t_all = time.time()
    with ProcessPoolExecutor(max_workers=len(CONFIG_GRID)) as pool:
        models = {cfg: (m, took) for cfg, m, took in
                  pool.map(_train, CONFIG_GRID, [toks] * len(CONFIG_GRID))}
    log(f"(huấn luyện song song {len(CONFIG_GRID)} cấu hình: {time.time() - t_all:.0f}s)\n")

    trained = {}
    best = None
    for vs, mc in CONFIG_GRID:
        m, took = models[(vs, mc)]
        for wname, weights in (("TB thường (Lab 05)", None), ("TB trọng số idf", idf)):
            D = unit_rows(np.array([m.document_vector(t, weights) for t in toks]))
            trained[(vs, mc, wname)] = (m, D)
            r_dev = sc["dev"].ranks(sc["dev"].w2v(m, D, weights))
            r_par = sc["dev_paraphrase"].ranks(sc["dev_paraphrase"].w2v(m, D, weights))
            obj = (ev.mrr(r_dev) + ev.mrr(r_par)) / 2
            label = f"d={vs:<3} min_count={mc} {wname}"
            log(f"{label:<34} dev: {fmt(r_dev)}")
            log(f"{'':<34} para: {fmt(r_par)}   ({len(m):,} từ, {took:.0f}s)")
            cand = (round(obj, 6), -vs, -mc, (vs, mc, wname))
            if best is None or cand > best:
                best = cand
    cfg = best[3]
    model, D = trained[cfg]
    weights = idf if cfg[2] == "TB trọng số idf" else None
    log(f"\n-> chốt Word2Vec: d={cfg[0]}, min_count={cfg[1]}, {cfg[2]} "
        f"(mục tiêu: TB MRR của dev và dev_paraphrase)")

    # ------------------------------------------------ PHẦN 2: chất lượng vector
    log("\n" + "-" * 78)
    log("PHẦN 2 — VECTOR HỌC ĐƯỢC TRÔNG THẾ NÀO? (mô hình đã chốt)")
    log("-" * 78)
    log("Cặp từ đồng nghĩa / gần nghĩa (cosine; '-' = một trong hai từ KHÔNG có vector):")
    for a, b in PROBE_PAIRS:
        s = f"{model.similarity(a, b):+.3f}" if a in model and b in model else "   -  "
        cnt = lambda w: int(model.counts[model.vocab[w]]) if w in model else 0
        log(f"   {a:>12} ~ {b:<12} {s}   (số lần xuất hiện: {cnt(a)} / {cnt(b)})")
    log("\nHàng xóm gần nhất (Lab 05: 'Similar ≠ Synonym'):")
    for w in PROBE_WORDS:
        if w in model:
            nb = ", ".join(f"{x}" for x, _ in model.most_similar(w, topn=6))
            log(f"   {w:>11}: {nb}")

    # ------------------------------------------------ PHẦN 3: độ phủ từ vựng
    log("\n" + "-" * 78)
    log("PHẦN 3 — ĐỘ PHỦ (Lab 05, Phần 5): câu hỏi có bao nhiêu token có vector?")
    log("-" * 78)
    for name in ("dev", "dev_paraphrase"):
        s = sc[name]
        by = {}
        for st, t in zip(s.styles, s.qtok):
            by.setdefault(st, []).append(model.coverage(t))
        for st, cov in sorted(by.items()):
            zero = sum(c == 0 for c in cov)
            log(f"   {name:<15} {st:<14} n={len(cov):>3}  độ phủ TB {np.mean(cov):5.1%}  "
                f"câu KHÔNG có token nào: {zero}")

    # ------------------------------------------------ PHẦN 4: lai, chọn λ trên DEV
    log("\n" + "-" * 78)
    log("PHẦN 4 — LAI: điểm = cosine TF-IDF + λ · cosine Word2Vec (chọn λ trên DEV)")
    log("-" * 78)
    w_dev = sc["dev"].w2v(model, D, weights)
    w_par = sc["dev_paraphrase"].w2v(model, D, weights)
    log(f"{'λ':>5}  {'dev R@1':>8} {'dev MRR':>8}  {'para R@1':>9} {'para MRR':>9}")
    best_l = None
    for lam in LAMBDA_GRID:
        r_dev = sc["dev"].ranks(sc["dev"].tfidf + lam * w_dev)
        r_par = sc["dev_paraphrase"].ranks(sc["dev_paraphrase"].tfidf + lam * w_par)
        a, b = metrics(r_dev), metrics(r_par)
        log(f"{lam:>5.2f}  {a[0]:>5}/{len(r_dev)} {a[2]:>8.3f}  {b[0]:>6}/{len(r_par)} {b[2]:>9.3f}")
        cand = (round((a[2] + b[2]) / 2, 6), -lam, lam)
        if best_l is None or cand > best_l:
            best_l = cand
    lam = best_l[2]
    log(f"\n-> chốt λ = {lam}")

    # ------------------------------------------------ PHẦN 5: ngưỡng chấp nhận
    log("\n" + "-" * 78)
    log("PHẦN 5 — DÙNG LÀM NGƯỠNG CHẤP NHẬN ĐƯỢC KHÔNG? (DEV: trong phạm vi vs ngoài phạm vi)")
    log("-" * 78)
    w_oos = oos_dev.w2v(model, D, weights)
    for name, S_in, S_oos in (("TF-IDF", sc["dev"].tfidf, oos_dev.tfidf), ("Word2Vec", w_dev, w_oos)):
        ins = [float(S_in[i].max()) for i in range(len(S_in))]
        oos = [float(S_oos[i].max()) for i in range(len(S_oos))]
        score, th = separability(ins, oos)
        log(f"   {name:<9} điểm top-1 câu trong phạm vi: trung vị {np.median(ins):.3f} | "
            f"ngoài phạm vi: trung vị {np.median(oos):.3f}, max {max(oos):.3f}")
        log(f"   {'':<9} ngưỡng tốt nhất {th:.3f} -> TB(trả lời được, chặn đúng) = {score:.1%}")

    # ------------------------------------------------ PHẦN 6: TEST, một lần
    log("\n" + "-" * 78)
    log(f"PHẦN 6 — BÁO CÁO TRÊN TEST (MỘT lần; d={cfg[0]}, min_count={cfg[1]}, {cfg[2]}, λ={lam})")
    log("-" * 78)
    results = {}
    for name in ("dev", "test", "dev_paraphrase", "test_paraphrase"):
        s = sc[name]
        W = s.w2v(model, D, weights)
        results[name] = {"TF-IDF": base[name], "Word2Vec": s.ranks(W),
                         "Lai": s.ranks(s.tfidf + lam * W)}
    for name in ("test", "test_paraphrase"):
        log(f"\n[{name}]")
        for meth, rk in results[name].items():
            log(f"   {meth:<9} {fmt(rk)}")
        for meth in ("Word2Vec", "Lai"):
            w, l, p = sign_test(results[name][meth], results[name]["TF-IDF"])
            log(f"   {meth} so với TF-IDF: tốt hơn {w} câu, kém hơn {l} câu -> p = {p:.3f}"
                f"{'  (CÓ ý nghĩa)' if p < 0.05 else '  (không có ý nghĩa thống kê)'}")

    # theo kiểu câu trên test
    log("\n[test — theo kiểu câu, Recall@1]")
    for st in sorted(set(sc["test"].styles)):
        idx = [i for i, x in enumerate(sc["test"].styles) if x == st]
        row = "   " + f"{st:<10}"
        for meth in ("TF-IDF", "Word2Vec", "Lai"):
            rk = [results["test"][meth][i] for i in idx]
            row += f"  {meth} {sum(r == 1 for r in rk):>2}/{len(idx)}"
        log(row)

    # Chấp nhận thực tế: lai chỉ đổi thứ hạng, cổng chấp nhận vẫn là cosine TF-IDF
    log("\n[test_paraphrase — nếu đưa bản Lai vào bot, cổng chấp nhận vẫn là cosine TF-IDF >= "
        f"{RETRIEVAL_THRESHOLD}]")
    s = sc["test_paraphrase"]
    W = s.w2v(model, D, weights)
    for meth, S in (("TF-IDF", s.tfidf), ("Lai", s.tfidf + lam * W)):
        ok = 0
        for i, g in enumerate(s.gold):
            top = int(np.argmax(S[i]))
            ok += (top in g) and s.tfidf[i][top] >= RETRIEVAL_THRESHOLD
        log(f"   {meth:<9} trả lời ĐÚNG bài và vượt ngưỡng: {ok}/{len(s.gold)}")

    out = PROJ / "data" / "eval" / "word2vec_report.txt"
    out.write_text("\n".join(REPORT) + "\n", encoding="utf-8")
    print(f"\nĐã ghi {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
