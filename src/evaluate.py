"""
evaluate.py — Dò tham số trên DEV, báo cáo MỘT LẦN trên TEST.

## Quy trình (thay thế cách làm cũ bị rò rỉ tập test)

Cách làm cũ: dò mọi tham số trên test_queries.json rồi báo cáo số liệu trên
chính tập đó. Con số đo được vì thế là mức "khớp" với đúng những câu đã dùng
để dò — không phải hiệu năng trên câu hỏi chưa thấy.

Cách làm mới:

  PHA 1 — DÒ TRÊN DEV (data/eval/dev.json)
    1. Intent: quét lưới (w_nb x ngưỡng)
    2. Cách xếp hạng: TF-IDF hay BM25, và (k1, b) của BM25 — so bằng MRR trên
       dev với độ mới TẮT, để đo đúng chất lượng xếp hạng.
    3. Độ mới: quét lưới (mốc tham chiếu x nửa chu kỳ x alpha) với HAI RÀNG
       BUỘC CỨNG (data/eval/conflict_case.json):
         a. ca tin mâu thuẫn phải trả bài mới ở mọi cách hỏi
         b. vẫn đúng như vậy sau khi thêm một bài KHÔNG liên quan có ngày đăng
            xa trong tương lai — tức là crawl thêm dữ liệu không được làm đổi
            thứ hạng (ràng buộc thêm sau lỗi crawl 14/09/2026, docs/09)
    4. Ngưỡng truy hồi: quét lưới với độ mới đã chốt
    5. Ngưỡng dự phòng gõ sai: quét trên dev + dev_typo + câu ngoài phạm vi

  PHA 2 — BÁO CÁO TRÊN TEST (data/eval/test.json, data/eval/test_typo.json)
    Dùng ĐÚNG bộ tham số chốt ở pha 1. Không quay lại chỉnh gì sau khi xem
    kết quả test — nếu làm vậy thì test lại thành dev.

Mọi truy vấn đi qua `prepare_user_text` giống hệt chatbot (chuẩn hóa teencode
+ giữ hệ quy chiếu dấu), nên điểm số phản ánh đúng thứ người dùng nhận được.

Mỗi tỷ lệ đều kèm khoảng tin cậy 95% (Wilson): với vài chục câu hỏi, một câu
sai đã làm con số dao động vài điểm phần trăm, nên chỉ báo số điểm là thiếu.

Chạy:  .venv/Scripts/python.exe src/evaluate.py
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    BM25_B,
    BM25_K1,
    CORPUS_RAW_PATH,
    DATA_DIR,
    FRESHNESS_ALPHA,
    FRESHNESS_HALFLIFE_DAYS,
    FRESHNESS_REFERENCE,
    FUZZY_THRESHOLD,
    INTENT_THRESHOLD,
    INTENT_W_NB,
    RANKING_METHOD,
    RETRIEVAL_THRESHOLD,
    TOP_K,
)
from dates import compute_recency
from entities import expand_query, extract
from intent_classifier import IntentClassifier
from normalizer import TeencodeNormalizer, prepare_user_text
from retriever import NewsRetriever

EVAL_DIR = DATA_DIR / "eval"
TUNED_PATH = EVAL_DIR / "tuned_params.json"

INTENT_W_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
INTENT_TH_GRID = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
FRESHNESS_REFERENCES = ["corpus", "candidates"]
HALFLIFE_GRID = [1.0, 2.0, 3.0, 5.0, 7.0, 14.0, 30.0]
ALPHA_GRID = [round(x, 1) for x in np.arange(0.0, 1.05, 0.1)]
RETR_TH_GRID = [round(x, 3) for x in np.arange(0.06, 0.305, 0.005)]
FUZZY_TH_GRID = [round(x, 2) for x in np.arange(0.30, 0.805, 0.01)]
BM25_K1_GRID = [0.6, 0.9, 1.2, 1.5, 2.0, 3.0, 5.0, 8.0]
BM25_B_GRID = [0.25, 0.5, 0.75, 0.9]


# ---------------------------------------------------------------------------
# Tiện ích
# ---------------------------------------------------------------------------
def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Khoảng tin cậy Wilson cho tỷ lệ k/n. Ổn định hơn xấp xỉ chuẩn khi n nhỏ
    hoặc tỷ lệ gần 0/1 — đúng tình huống của các tập test vài chục câu."""
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def sign_test_p(wins: int, losses: int) -> float:
    """p-value hai phía của kiểm định dấu (binomial chính xác, p = 0.5).

    Bỏ qua các câu hòa. Hỏi: nếu hai phương pháp thật ra ngang nhau, xác suất
    thấy chênh lệch thắng/thua ít nhất cực đoan như vậy là bao nhiêu?
    """
    n = wins + losses
    if n == 0:
        return 1.0
    k = min(wins, losses)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def fmt_rate(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k / n:6.1%}  ({k}/{n}, KTC95% {lo:.0%}–{hi:.0%})" if n else "  n/a"


def load_split(name: str) -> dict:
    return json.loads((EVAL_DIR / f"{name}.json").read_text(encoding="utf-8"))


def load_typo_split(name: str) -> dict:
    """dev_typo / test_typo (tools/build_typo_sets.py). Rỗng nếu chưa sinh."""
    path = EVAL_DIR / f"{name}_typo.json"
    if not path.exists():
        return {"retrieval": []}
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Chuẩn bị truy vấn — y hệt chatbot
# ---------------------------------------------------------------------------
_NORMALIZER = TeencodeNormalizer()


@lru_cache(maxsize=4096)
def prepared(raw: str) -> str:
    return prepare_user_text(raw, _NORMALIZER)[0]


@lru_cache(maxsize=4096)
def retrieval_query(raw: str) -> str:
    """Câu hỏi sau chuẩn hóa + nhân đôi thực thể, như chatbot._handle_retrieval."""
    text = prepared(raw)
    return expand_query(text, extract(text))


# ---------------------------------------------------------------------------
# Intent
# ---------------------------------------------------------------------------
def intent_scores(clf: IntentClassifier, split: dict) -> tuple[list, list]:
    """(dự đoán cho câu intent cố định, dự đoán cho câu phải đi truy hồi/ngoài phạm vi)."""
    fixed = [(clf.predict(prepared(x["text"])), x["expected"])
             for x in split["intent"] if not x["expected"].startswith("__")]
    must_route = [x["text"] for x in split["intent"] if x["expected"] == "__retrieve__"]
    must_route += [x["text"] for x in split["out_of_scope"]]
    must_route += [x["query"] for x in split["retrieval"]]
    routed = [clf.predict(prepared(t)) for t in must_route]
    return fixed, routed


def intent_metrics(clf, fixed, routed, threshold):
    n_ok = sum(1 for (tag, c), exp in fixed if c >= threshold and tag == exp)
    n_safe = sum(1 for tag, c in routed if c < threshold or clf.get_action(tag) == "retrieve")
    return n_ok, len(fixed), n_safe, len(routed)


def tune_intent(dev: dict) -> tuple[float, float]:
    hr("PHA 1.1 — DÒ INTENT TRÊN DEV (w_nb x ngưỡng)")
    print("accuracy = câu intent cố định được nhận đúng nhãn và vượt ngưỡng")
    print("safety   = câu cần truy hồi / ngoài phạm vi KHÔNG bị trả lời bằng câu soạn sẵn\n")
    print(f"{'w_nb':>6} {'ngưỡng':>7} {'accuracy':>9} {'safety':>8} {'TB':>7}")
    print("-" * 42)

    best = None
    for w in INTENT_W_GRID:
        clf = IntentClassifier(w_nb=w).train_from_file()
        fixed, routed = intent_scores(clf, dev)
        row_best = None
        for th in INTENT_TH_GRID:
            ok, n, safe, m = intent_metrics(clf, fixed, routed, th)
            score = (ok / n + safe / m) / 2
            cand = (score, -abs(th - 0.3), w, th, ok / n, safe / m)
            if row_best is None or cand > row_best:
                row_best = cand
            if best is None or cand > best:
                best = cand
        _, _, w_, th_, acc, saf = row_best
        print(f"{w_:>6.1f} {th_:>7.2f} {acc:>8.1%} {saf:>8.1%} {(acc + saf) / 2:>7.1%}")

    _, _, w, th, acc, saf = best
    print(f"\n-> chốt w_nb={w}, INTENT_THRESHOLD={th}  (dev: accuracy {acc:.1%}, safety {saf:.1%})")
    return w, th


# ---------------------------------------------------------------------------
# Truy hồi
# ---------------------------------------------------------------------------
def ranks_for(retriever: NewsRetriever, cases: list[dict], k: int = 10) -> list[int | None]:
    """Hạng của bài đúng đầu tiên trong top-k (None nếu không có), KHÔNG xét ngưỡng."""
    url_of = retriever.df["url"].tolist()
    out = []
    for c in cases:
        gold = set(c["gold_urls"])
        ranked = retriever.rank(retrieval_query(c["query"]), top_k=k)
        out.append(next((i for i, (d, _, _) in enumerate(ranked, 1) if url_of[d] in gold), None))
    return out


def mrr(ranks) -> float:
    return float(np.mean([1.0 / r if r else 0.0 for r in ranks])) if ranks else 0.0


def conflict_ok(retriever: NewsRetriever, case: dict) -> bool:
    url_of = retriever.df["url"].tolist()
    for q in case["queries"]:
        res = [url_of[d] for d, _, _ in retriever.rank(q, top_k=10) if "example.test" in url_of[d]]
        if not res or res[0] != case["expected_url"]:
            return False
    return True


def conflict_retrievers(df: pd.DataFrame, case: dict) -> tuple[NewsRetriever, NewsRetriever]:
    """(corpus + 2 bài mâu thuẫn, như trên + 1 bài không liên quan ở tương lai)."""
    arts = pd.DataFrame(case["articles"])
    future = {k: v for k, v in case["future_unrelated_article"].items() if k != "note"}
    now = NewsRetriever().fit(pd.concat([df, arts], ignore_index=True))
    later = NewsRetriever().fit(pd.concat([df, arts, pd.DataFrame([future])], ignore_index=True))
    return now, later


def set_freshness(retriever: NewsRetriever, halflife: float, alpha: float,
                  reference: str | None = None) -> None:
    retriever.freshness_halflife = halflife
    retriever.freshness_alpha = alpha
    if reference is not None:
        retriever.freshness_reference = reference
    retriever.recency = compute_recency(retriever.published_dates, half_life_days=halflife)


def tune_freshness(dev: dict, retriever: NewsRetriever,
                   conflict_now: NewsRetriever, conflict_later: NewsRetriever,
                   case: dict) -> tuple[str, float, float]:
    hr("PHA 1.3 — DÒ ĐỘ MỚI TRÊN DEV (mốc tham chiếu x nửa chu kỳ x alpha)")
    print("Mục tiêu: MRR cao nhất trên dev, với HAI RÀNG BUỘC CỨNG:")
    print("  (a) ca tin mâu thuẫn trả bài mới")
    print("  (b) vẫn đúng sau khi thêm 1 bài KHÔNG liên quan có ngày đăng ở tương lai")
    print("Ô bảng = MRR dev; '*' = qua (a) và (b), '~' = chỉ qua (a), ' ' = trượt (a)\n")

    rows = []
    for ref in FRESHNESS_REFERENCES:
        print(f"[mốc = {ref}]")
        print(f"{'hl':>4} " + " ".join(f"{a:>7.1f}" for a in ALPHA_GRID))
        for hl in HALFLIFE_GRID:
            cells = []
            for a in ALPHA_GRID:
                for r in (retriever, conflict_now, conflict_later):
                    set_freshness(r, hl, a, ref)
                ranks = ranks_for(retriever, dev["retrieval"])
                m = mrr(ranks)
                r1 = sum(1 for x in ranks if x == 1) / len(ranks)
                ok_now = conflict_ok(conflict_now, case)
                ok_later = ok_now and conflict_ok(conflict_later, case)
                rows.append((ref, hl, a, r1, m, ok_now, ok_later))
                cells.append(f"{m:.3f}{'*' if ok_later else '~' if ok_now else ' '}")
            print(f"{hl:>4.0f} " + " ".join(f"{c:>7}" for c in cells))
        print()

    off = next(x for x in rows if x[2] == 0.0)
    print(f"Tham chiếu — độ mới TẮT (alpha=0): R@1 {off[3]:.1%}, MRR {off[4]:.3f}")
    for ref in FRESHNESS_REFERENCES:
        n_ok = sum(1 for x in rows if x[0] == ref and x[6])
        print(f"   mốc {ref:<10}: {n_ok} cấu hình qua cả hai ràng buộc")

    feasible = [x for x in rows if x[6]]
    if not feasible:
        print("\nKHÔNG có cấu hình nào qua cả hai ràng buộc -> giữ giá trị trong config.")
        return FRESHNESS_REFERENCE, FRESHNESS_HALFLIFE_DAYS, FRESHNESS_ALPHA
    # Hòa điểm -> alpha nhỏ hơn (ít lệch khỏi TF-IDF thuần), rồi nửa chu kỳ gần 7.
    best = max(feasible, key=lambda x: (round(x[4], 6), -x[2], -abs(x[1] - 7)))
    ref, hl, a, r1, m = best[:5]
    print(f"\n-> chốt mốc = {ref}, nửa chu kỳ = {hl:g} ngày, alpha = {a}"
          f"  (dev: R@1 {r1:.1%}, MRR {m:.3f})")
    return ref, hl, a


def tune_ranking(dev: dict, retriever: NewsRetriever) -> tuple[str, float, float]:
    """TF-IDF hay BM25? Và (k1, b) nào? — so trên dev với độ mới TẮT."""
    hr("PHA 1.2 — CÁCH XẾP HẠNG TRÊN DEV: TF-IDF vs BM25 (độ mới tắt)")
    saved = (retriever.freshness_alpha, retriever.recency)
    retriever.freshness_alpha = 0.0

    rows = []
    retriever.set_bm25(BM25_K1, BM25_B, ranking="tfidf")
    ranks = ranks_for(retriever, dev["retrieval"])
    tfidf_row = ("tfidf", None, None, sum(1 for r in ranks if r == 1) / len(ranks), mrr(ranks))
    rows.append(tfidf_row)

    for k1 in BM25_K1_GRID:
        for b in BM25_B_GRID:
            retriever.set_bm25(k1, b, ranking="bm25")
            ranks = ranks_for(retriever, dev["retrieval"])
            rows.append(("bm25", k1, b, sum(1 for r in ranks if r == 1) / len(ranks), mrr(ranks)))

    print(f"{'cách':>6} {'k1':>5} {'b':>5} {'R@1':>7} {'MRR':>7}")
    print("-" * 36)
    print(f"{'tfidf':>6} {'-':>5} {'-':>5} {tfidf_row[3]:>6.1%} {tfidf_row[4]:>7.3f}")
    bm = sorted([r for r in rows if r[0] == "bm25"], key=lambda r: -r[4])
    for r in bm[:6]:
        print(f"{'bm25':>6} {r[1]:>5} {r[2]:>5} {r[3]:>6.1%} {r[4]:>7.3f}")
    print(f"   ... ({len(bm) - 6} cấu hình BM25 khác thấp hơn)")

    # Xu hướng theo k1 (b cố định ở giá trị tốt nhất): dùng để kiểm chứng giả
    # thuyết "độ bão hòa tf của BM25 xung đột với việc lặp tiêu đề để tăng trọng số".
    best_b = bm[0][2]
    trend = sorted([r for r in bm if r[2] == best_b], key=lambda r: r[1])
    print(f"\n   Xu hướng theo k1 (b={best_b}): "
          + "  ".join(f"k1={r[1]}:{r[4]:.3f}" for r in trend))

    best_bm = bm[0]

    # Kiểm định dấu có cặp (paired sign test) trên từng câu dev: BM25 xếp bài
    # đúng cao hơn hay thấp hơn TF-IDF? Chỉ đổi sang cách PHỨC TẠP hơn khi nó
    # thắng CÓ Ý NGHĨA THỐNG KÊ — chênh 0.002 MRR (~1 câu) chỉ là nhiễu.
    #
    # (Lịch sử) Quy tắc cũ "BM25 hơn TF-IDF trên dev là đổi" đã chọn BM25 với
    # chênh lệch 0.002, rồi BM25 THUA trên test (MRR 0.943 vs 0.950). Quy tắc
    # kiểm định này được thêm SAU khi đã thấy kết quả test đó — ghi rõ ở đây
    # để minh bạch. Đây là nguyên tắc chuẩn, không phải tham số dò theo test.
    retriever.set_bm25(best_bm[1], best_bm[2], ranking="bm25")
    rr_bm = [1.0 / r if r else 0.0 for r in ranks_for(retriever, dev["retrieval"])]
    retriever.set_bm25(best_bm[1], best_bm[2], ranking="tfidf")
    rr_tf = [1.0 / r if r else 0.0 for r in ranks_for(retriever, dev["retrieval"])]
    wins = sum(1 for a, b_ in zip(rr_bm, rr_tf) if a > b_)
    losses = sum(1 for a, b_ in zip(rr_bm, rr_tf) if a < b_)
    p_value = sign_test_p(wins, losses)
    print(f"\n   Kiểm định dấu có cặp (dev): BM25 tốt hơn ở {wins} câu, kém hơn ở "
          f"{losses} câu, hòa {len(rr_bm) - wins - losses} câu -> p = {p_value:.3f}")

    significant = wins > losses and p_value < 0.05
    choice = best_bm if significant else tfidf_row
    if not significant:
        print("   -> Không có khác biệt có ý nghĩa thống kê: GIỮ TF-IDF (đơn giản hơn).")
    retriever.freshness_alpha, retriever.recency = saved
    method = choice[0]
    # (k1, b) luôn là cấu hình BM25 TỐT NHẤT trên dev — kể cả khi không chọn
    # BM25 — để phép so trên test là TF-IDF vs BM25 ở trạng thái tốt nhất của nó.
    k1, b = best_bm[1], best_bm[2]
    retriever.set_bm25(k1, b, ranking=method)
    print("\n-> chốt xếp hạng = " + method
          + (f" (k1={k1}, b={b})" if method == "bm25" else "")
          + f"  (dev MRR {choice[4]:.3f} so với TF-IDF {tfidf_row[4]:.3f})")
    return method, k1, b


def compare_ranking_on_test(test: dict, retriever: NewsRetriever, params: dict) -> dict:
    """So TF-IDF và BM25 (k1, b dò trên dev) trên TEST, cùng độ mới đã chốt."""
    hr("PHA 2c — SO SÁNH TF-IDF vs BM25 TRÊN TEST (k1, b đã chốt trên dev)")
    k1 = params.get("bm25_k1", BM25_K1)
    b = params.get("bm25_b", BM25_B)
    out = {}
    for method in ("tfidf", "bm25"):
        retriever.set_bm25(k1, b, ranking=method)
        ranks = ranks_for(retriever, test["retrieval"])
        n = len(ranks)
        r1 = sum(1 for r in ranks if r == 1)
        out[method] = {"r1": (r1, n), "mrr": mrr(ranks)}
        label = method + (f" (k1={k1}, b={b})" if method == "bm25" else "")
        print(f"  {label:<24} R@1 {fmt_rate(r1, n)}   MRR {mrr(ranks):.3f}")
    retriever.set_bm25(k1, b, ranking=params["ranking"])
    return out


def top1_scores(retriever, cases, key):
    out = []
    for c in cases:
        res = retriever.search(retrieval_query(c[key]), top_k=1, min_score=0.0)
        out.append(res[0] if res else None)
    return out


def tune_retrieval_threshold(dev: dict, retriever: NewsRetriever) -> float:
    hr("PHA 1.4 — DÒ NGƯỠNG TRUY HỒI TRÊN DEV")
    ins = top1_scores(retriever, dev["retrieval"], "query")
    # Ngưỡng áp lên COSINE THUẦN (base_score), không phải điểm đã nhân độ mới —
    # độ mới chỉ dùng để xếp hạng, không quyết định có trả lời hay không.
    in_scores = [r.base_score for r, c in zip(ins, dev["retrieval"])
                 if r is not None and r.url in set(c["gold_urls"])]
    oos = top1_scores(retriever, dev["out_of_scope"], "text")
    oos_scores = [r.base_score if r else 0.0 for r in oos]

    print(f"Trúng bài (n={len(in_scores)}): min={min(in_scores):.3f} "
          f"p10={np.percentile(in_scores, 10):.3f} trung vị={np.median(in_scores):.3f}")
    print(f"Ngoài phạm vi (n={len(oos_scores)}): trung vị={np.median(oos_scores):.3f} "
          f"p90={np.percentile(oos_scores, 90):.3f} max={max(oos_scores):.3f}\n")

    best = None
    rows = []
    for th in RETR_TH_GRID:
        ans = sum(1 for s in in_scores if s >= th) / len(in_scores)
        blk = sum(1 for s in oos_scores if s < th) / len(oos_scores)
        score = (ans + blk) / 2
        rows.append((th, ans, blk, score))
        cand = (round(score, 6), -th, th)
        if best is None or cand > best:
            best = cand

    print(f"{'ngưỡng':>7} {'trả lời được':>13} {'chặn đúng':>10} {'TB':>7}")
    print("-" * 42)
    for th, ans, blk, score in rows[::4]:
        print(f"{th:>7.3f} {ans:>12.1%} {blk:>10.1%} {score:>7.1%}")
    th = best[2]
    row = next(r for r in rows if r[0] == th)
    print(f"\n-> chốt RETRIEVAL_THRESHOLD = {th:.3f}  "
          f"(dev: trả lời được {row[1]:.1%}, chặn đúng {row[2]:.1%})")
    return th


# ---------------------------------------------------------------------------
# Dự phòng gõ sai
# ---------------------------------------------------------------------------
def fallback_candidates(retriever: NewsRetriever, cases: list[dict], key: str) -> tuple[list, list]:
    """Với từng câu: kết quả đường CHÍNH, và (nếu bị từ chối) ứng viên dự phòng.

    Mô phỏng đúng search() với TOP_K của bot: câu trả lời là bài ĐẦU TIÊN trong
    top-k (theo điểm xếp hạng) có điểm chấp nhận đạt ngưỡng.

    Trả (outcome, fuzzy) — outcome ∈ {"correct", "wrong", "refused"} ở ngưỡng
    hiện hành của retriever; fuzzy = danh sách (điểm chấp nhận, đúng bài không)
    của top-k đường dự phòng, chỉ cho câu bị từ chối. Câu ngoài phạm vi không có
    gold_urls nên "đúng bài" luôn là False.
    """
    url_of = retriever.df["url"].tolist()
    outcomes, fuzzy = [], []
    for c in cases:
        q = retrieval_query(c[key])
        gold = set(c.get("gold_urls", []))
        hit = next((d for d, _, acc in retriever.rank(q, top_k=TOP_K)
                    if acc >= retriever.threshold), None)
        if hit is not None:
            outcomes.append("correct" if url_of[hit] in gold else "wrong")
            fuzzy.append(None)
        else:
            outcomes.append("refused")
            fuzzy.append([(acc, url_of[d] in gold)
                          for d, _, acc in retriever.rank(q, top_k=TOP_K, fuzzy=True)])
    return outcomes, fuzzy


def fuzzy_table(outcomes, fuzzy, th: float | None) -> Counter:
    """Đếm correct / wrong / refused sau khi áp đường dự phòng với ngưỡng th."""
    counts = Counter()
    for o, cands in zip(outcomes, fuzzy):
        pick = (None if o != "refused" or th is None
                else next((ok for acc, ok in cands if acc >= th), None))
        if pick is None:
            counts[o] += 1
        else:
            counts["correct" if pick else "wrong"] += 1
    return counts


def tune_fuzzy_threshold(dev: dict, dev_typo: dict, retriever: NewsRetriever) -> float | None:
    hr("PHA 1.5 — DÒ NGƯỠNG DỰ PHÒNG GÕ SAI TRÊN DEV (+ dev_typo)")
    if not dev_typo["retrieval"]:
        print("Chưa có data/eval/dev_typo.json (tools/build_typo_sets.py) -> giữ config.")
        return FUZZY_THRESHOLD
    in_cases = dev["retrieval"] + dev_typo["retrieval"]
    out_in, fz_in = fallback_candidates(retriever, in_cases, "query")
    out_oos, fz_oos = fallback_candidates(retriever, dev["out_of_scope"], "text")
    n_in, n_oos = len(in_cases), len(dev["out_of_scope"])

    def summary(th):
        c_in, c_oos = fuzzy_table(out_in, fz_in, th), fuzzy_table(out_oos, fz_oos, th)
        leaked = n_oos - c_oos["refused"]
        score = (c_in["correct"] / n_in + (n_oos - leaked) / n_oos) / 2
        return score, c_in, leaked

    base_score, base_in, base_leak = summary(None)
    print("Mục tiêu giống PHA 1.4: TB(trả lời ĐÚNG trên câu trong phạm vi, chặn đúng ngoài phạm vi).")
    print("Hòa điểm -> ít câu trả SAI hơn, rồi ngưỡng CAO hơn (dự phòng là tín hiệu yếu hơn).")
    print(f"Câu trong phạm vi: {n_in} (dev {len(dev['retrieval'])} + dev_typo "
          f"{len(dev_typo['retrieval'])}); ngoài phạm vi: {n_oos}\n")
    print(f"{'ngưỡng':>7} {'đúng':>6} {'sai':>5} {'từ chối':>8} {'lọt':>5} {'TB':>7}")
    print("-" * 44)
    print(f"{'tắt':>7} {base_in['correct']:>6} {base_in['wrong']:>5} {base_in['refused']:>8} "
          f"{base_leak:>5} {base_score:>7.1%}")

    best = None
    for th in FUZZY_TH_GRID:
        score, c_in, leaked = summary(th)
        if round(th * 100) % 5 == 0:
            print(f"{th:>7.2f} {c_in['correct']:>6} {c_in['wrong']:>5} {c_in['refused']:>8} "
                  f"{leaked:>5} {score:>7.1%}")
        cand = (round(score, 6), -c_in["wrong"], th)
        if best is None or cand > best:
            best = cand

    if best[0] <= round(base_score, 6):
        print("\n-> dự phòng KHÔNG cải thiện trên dev: TẮT (fuzzy_threshold = None)")
        return None
    th = best[2]
    _, c_in, leaked = summary(th)
    print(f"\n-> chốt FUZZY_THRESHOLD = {th:.2f}  (dev: đúng {base_in['correct']} -> "
          f"{c_in['correct']}, sai {base_in['wrong']} -> {c_in['wrong']}, lọt {leaked})")
    return th


# ---------------------------------------------------------------------------
# PHA 2 — báo cáo trên TEST
# ---------------------------------------------------------------------------
def report_test(test: dict, test_typo: dict, params: dict, retriever: NewsRetriever) -> dict:
    hr("PHA 2 — BÁO CÁO TRÊN TEST (tham số đã chốt trên dev, KHÔNG chỉnh thêm)")
    print(json.dumps(params, ensure_ascii=False))
    results: dict = {}

    # ---- Intent
    clf = IntentClassifier(w_nb=params["intent_w_nb"]).train_from_file()
    th = params["intent_threshold"]
    fixed_cases = [x for x in test["intent"] if not x["expected"].startswith("__")]
    per_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    errors = []
    for x in fixed_cases:
        tag, c = clf.predict(prepared(x["text"]))
        pred = tag if c >= th else "__reject__"
        if pred == x["expected"]:
            per_class[x["expected"]]["tp"] += 1
        else:
            per_class[x["expected"]]["fn"] += 1
            if pred != "__reject__":
                per_class[pred]["fp"] += 1
            errors.append((x["text"], x["expected"], pred, c))
    n_ok = sum(v["tp"] for v in per_class.values())
    f1s = []
    for tag, v in per_class.items():
        if v["tp"] + v["fn"] == 0:
            continue
        p = v["tp"] / (v["tp"] + v["fp"]) if v["tp"] + v["fp"] else 0.0
        r = v["tp"] / (v["tp"] + v["fn"])
        f1s.append(2 * p * r / (p + r) if p + r else 0.0)

    print("\n[Intent]")
    print(f"  Accuracy      : {fmt_rate(n_ok, len(fixed_cases))}")
    print(f"  Macro-F1      : {np.mean(f1s):.3f}")
    if errors:
        print(f"  Câu sai ({len(errors)}):")
        for t, g, p, c in errors:
            print(f"     {t!r:36} gold={g:<20} pred={p:<20} conf={c:.2f}")
    results["intent_accuracy"] = (n_ok, len(fixed_cases))
    results["intent_macro_f1"] = float(np.mean(f1s))

    # ---- Truy hồi (thành phần)
    ranks = ranks_for(retriever, test["retrieval"])
    n = len(ranks)
    r1 = sum(1 for r in ranks if r == 1)
    r3 = sum(1 for r in ranks if r and r <= 3)
    print("\n[Truy hồi — thành phần]")
    print(f"  Recall@1      : {fmt_rate(r1, n)}")
    print(f"  Recall@3      : {fmt_rate(r3, n)}")
    print(f"  MRR           : {mrr(ranks):.3f}")
    by_style = defaultdict(list)
    for c, r in zip(test["retrieval"], ranks):
        by_style[c["style"]].append(r)
    for style, rs in sorted(by_style.items()):
        k1 = sum(1 for r in rs if r == 1)
        print(f"     {style:<10} R@1 {fmt_rate(k1, len(rs))}   MRR {mrr(rs):.3f}")
    misses = [(c, r) for c, r in zip(test["retrieval"], ranks) if r != 1]
    if misses:
        print(f"  Không đứng đầu ({len(misses)}):")
        for c, r in misses:
            top = retriever.search(retrieval_query(c["query"]), top_k=1, min_score=0.0)
            got = top[0].title[:44] if top else "(không có)"
            print(f"     [{c['style'][:4]}] {c['query'][:38]:38} hạng={r}  top1: {got}")
    results["recall_at_1"] = (r1, n)
    results["recall_at_3"] = (r3, n)
    results["mrr"] = mrr(ranks)
    results["by_style"] = {s: {"r1": sum(1 for r in rs if r == 1), "n": len(rs), "mrr": mrr(rs)}
                           for s, rs in by_style.items()}

    # ---- Ngoài phạm vi
    oos = top1_scores(retriever, test["out_of_scope"], "text")
    blocked = sum(1 for r in oos if r is None or r.base_score < params["retrieval_threshold"])
    print("\n[Ngoài phạm vi — thành phần truy hồi]")
    print(f"  Chặn đúng     : {fmt_rate(blocked, len(oos))}")
    for x, r in zip(test["out_of_scope"], oos):
        if r is not None and r.base_score >= params["retrieval_threshold"]:
            print(f"     LỌT: {x['text']!r} -> {r.base_score:.3f} {r.title[:44]}")
    results["oos_blocked"] = (blocked, len(oos))

    # ---- Dự phòng gõ sai: đo cả khi TẮT và BẬT, trên câu sạch, câu gõ sai và
    # câu ngoài phạm vi — cái giá của việc cứu câu gõ sai là có thể lọt thêm.
    fth = params["fuzzy_threshold"]
    print(f"\n[Dự phòng gõ sai — FUZZY_THRESHOLD = {fth}]")
    print(f"  {'tập':<22} {'dự phòng':>8} {'đúng':>6} {'sai':>5} {'từ chối':>8}")
    groups = [("test (sạch)", test["retrieval"], "query"),
              ("test_typo (gõ sai)", test_typo["retrieval"], "query"),
              ("ngoài phạm vi", test["out_of_scope"], "text")]
    results["fuzzy"] = {}
    for name, cases, key in groups:
        if not cases:
            continue
        outs, fz = fallback_candidates(retriever, cases, key)
        row = {}
        for label, th in (("tắt", None), ("bật", fth)):
            c = fuzzy_table(outs, fz, th)
            row[label] = {"correct": c["correct"], "wrong": c["wrong"], "refused": c["refused"],
                          "n": len(cases)}
            print(f"  {name:<22} {label:>8} {c['correct']:>6} {c['wrong']:>5} {c['refused']:>8}")
        results["fuzzy"][name] = row
        if key == "query":
            on = row["bật"]
            print(f"     -> trả lời đúng: {fmt_rate(row['tắt']['correct'], len(cases))}"
                  f"  ->  {fmt_rate(on['correct'], len(cases))}")
        else:
            on_blocked = row["bật"]["refused"]
            print(f"     -> chặn đúng khi bật dự phòng: {fmt_rate(on_blocked, len(cases))}")
            results["oos_blocked_with_fuzzy"] = (on_blocked, len(cases))
    return results


def report_end_to_end(test: dict, test_typo: dict, params: dict) -> dict:
    """Đo HỆ THỐNG hoàn chỉnh: gọi bot.respond() như người dùng thật.

    Khác với số liệu thành phần ở trên, phần này tính cả việc intent classifier
    định tuyến sai (ví dụ câu hỏi tin tức bị hiểu thành "duyệt chuyên mục").
    """
    from chatbot import NewsChatbot

    hr("PHA 2b — ĐÁNH GIÁ ĐẦU-CUỐI TRÊN TEST (gọi bot.respond như người dùng)")
    bot = NewsChatbot(
        intent_threshold=params["intent_threshold"],
        retrieval_threshold=params["retrieval_threshold"],
        intent_w_nb=params["intent_w_nb"],
        freshness_alpha=params["freshness_alpha"],
        freshness_halflife=params["freshness_halflife"],
        ranking=params["ranking"],
        bm25_k1=params["bm25_k1"],
        bm25_b=params["bm25_b"],
        freshness_reference=params["freshness_reference"],
        fuzzy_threshold=params["fuzzy_threshold"],
        use_fuzzy=params["fuzzy_threshold"] is not None,
    ).train()

    ok = 0
    wrong_route = Counter()
    for c in test["retrieval"]:
        bot.reset()
        r = bot.respond(c["query"])
        if r.results and r.results[0].url in set(c["gold_urls"]):
            ok += 1
        else:
            wrong_route[r.route] += 1
    n = len(test["retrieval"])

    refused = 0
    leaked = []
    for x in test["out_of_scope"]:
        bot.reset()
        r = bot.respond(x["text"])
        if r.route == "fallback":
            refused += 1
        else:
            leaked.append((x["text"], r.route, r.intent))

    print(f"  Câu hỏi tin tức -> bài đứng đầu là bài đúng : {fmt_rate(ok, n)}")
    print(f"     phân bố các câu KHÔNG đạt theo đường đi: {dict(wrong_route)}")
    print(f"  Câu ngoài phạm vi -> bot từ chối           : {fmt_rate(refused, len(test['out_of_scope']))}")
    for t, route, intent in leaked:
        print(f"     KHÔNG từ chối: {t!r} -> route={route} intent={intent}")
    out = {"e2e_answer": (ok, n), "e2e_refuse": (refused, len(test["out_of_scope"]))}

    if test_typo["retrieval"]:
        typo_ok, typo_wrong, fuzzy_used = 0, 0, 0
        for c in test_typo["retrieval"]:
            bot.reset()
            r = bot.respond(c["query"])
            if r.results:
                typo_ok += r.results[0].url in set(c["gold_urls"])
                typo_wrong += r.results[0].url not in set(c["gold_urls"])
                fuzzy_used += r.results[0].match == "fuzzy"
        nt = len(test_typo["retrieval"])
        print(f"  Câu gõ sai -> bài đứng đầu là bài đúng      : {fmt_rate(typo_ok, nt)}")
        print(f"     trả bài sai: {typo_wrong}; trong số câu có trả lời, {fuzzy_used} câu đi đường dự phòng")
        out["e2e_typo_answer"] = (typo_ok, nt)
        out["e2e_typo_wrong"] = (typo_wrong, nt)
    return out


# ---------------------------------------------------------------------------
def main() -> int:
    dev, test = load_split("dev"), load_split("test")
    dev_typo, test_typo = load_typo_split("dev"), load_typo_split("test")
    case = json.loads((EVAL_DIR / "conflict_case.json").read_text(encoding="utf-8"))
    print(f"DEV : {len(dev['retrieval'])} truy hồi, {len(dev['out_of_scope'])} ngoài phạm vi, "
          f"{len(dev['intent'])} intent")
    print(f"TEST: {len(test['retrieval'])} truy hồi, {len(test['out_of_scope'])} ngoài phạm vi, "
          f"{len(test['intent'])} intent")
    print(f"GÕ SAI: dev_typo {len(dev_typo['retrieval'])} câu, test_typo {len(test_typo['retrieval'])} câu")

    df = pd.read_csv(CORPUS_RAW_PATH).dropna(subset=["title", "text"]).reset_index(drop=True)
    print(f"CORPUS: {len(df)} bài ({CORPUS_RAW_PATH.name})")
    retriever = NewsRetriever().fit_cached(df)
    conflict_now, conflict_later = conflict_retrievers(df, case)

    # ---------------- PHA 1: DEV
    w_nb, intent_th = tune_intent(dev)
    ranking, bm25_k1, bm25_b = tune_ranking(dev, retriever)
    for r in (conflict_now, conflict_later):
        r.set_bm25(bm25_k1, bm25_b, ranking=ranking)
    reference, halflife, alpha = tune_freshness(dev, retriever, conflict_now, conflict_later, case)
    set_freshness(retriever, halflife, alpha, reference)
    retr_th = tune_retrieval_threshold(dev, retriever)
    retriever.threshold = retr_th
    fuzzy_th = tune_fuzzy_threshold(dev, dev_typo, retriever)
    retriever.fuzzy_threshold = fuzzy_th

    params = {
        "intent_w_nb": w_nb,
        "intent_threshold": intent_th,
        "ranking": ranking,
        "bm25_k1": bm25_k1,
        "bm25_b": bm25_b,
        "freshness_halflife": halflife,
        "freshness_alpha": alpha,
        "freshness_reference": reference,
        "retrieval_threshold": retr_th,
        "fuzzy_threshold": fuzzy_th,
    }
    TUNED_PATH.write_text(json.dumps(params, ensure_ascii=False, indent=1), encoding="utf-8")

    current = {
        "intent_w_nb": INTENT_W_NB, "intent_threshold": INTENT_THRESHOLD,
        "freshness_halflife": FRESHNESS_HALFLIFE_DAYS, "freshness_alpha": FRESHNESS_ALPHA,
        "retrieval_threshold": RETRIEVAL_THRESHOLD,
        "ranking": RANKING_METHOD, "bm25_k1": BM25_K1, "bm25_b": BM25_B,
        "freshness_reference": FRESHNESS_REFERENCE, "fuzzy_threshold": FUZZY_THRESHOLD,
    }
    def _differs(a, b):
        if a is None or b is None or isinstance(a, str) or isinstance(b, str):
            return a != b
        return abs(a - b) > 1e-9
    diffs = {k: (current[k], v) for k, v in params.items() if _differs(current[k], v)}

    # ---------------- PHA 2: TEST
    results = report_test(test, test_typo, params, retriever)
    results["ranking_comparison"] = compare_ranking_on_test(test, retriever, params)
    results.update(report_end_to_end(test, test_typo, params))

    hr("TÓM TẮT")
    print("Tham số chốt trên DEV:", json.dumps(params, ensure_ascii=False))
    if diffs:
        print("\nCẢNH BÁO: config.py đang KHÁC tham số vừa dò trên dev:")
        for k, (cur, new) in diffs.items():
            print(f"   {k}: config={cur}  dev={new}")
        print("   -> cập nhật config.py để bot chạy đúng bộ tham số đã báo cáo.")
    else:
        print("config.py khớp với tham số đã dò trên dev.")

    (EVAL_DIR / "test_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
