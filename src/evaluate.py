"""
evaluate.py — Dò tham số trên DEV, báo cáo MỘT LẦN trên TEST.

## Quy trình (thay thế cách làm cũ bị rò rỉ tập test)

Cách làm cũ: dò mọi tham số trên test_queries.json rồi báo cáo số liệu trên
chính tập đó. Con số đo được vì thế là mức "khớp" với đúng những câu đã dùng
để dò — không phải hiệu năng trên câu hỏi chưa thấy.

Cách làm mới:

  PHA 1 — DÒ TRÊN DEV (data/eval/dev.json)
    1. Intent: quét lưới (w_nb x ngưỡng)
    2. Độ mới: quét lưới (nửa chu kỳ x alpha), RÀNG BUỘC CỨNG là ca tin mâu
       thuẫn (data/eval/conflict_case.json) phải trả bài mới ở mọi cách hỏi
    3. Ngưỡng truy hồi: quét lưới với độ mới đã chốt

  PHA 2 — BÁO CÁO TRÊN TEST (data/eval/test.json)
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
    CORPUS_RAW_PATH,
    DATA_DIR,
    FRESHNESS_ALPHA,
    FRESHNESS_HALFLIFE_DAYS,
    INTENT_THRESHOLD,
    INTENT_W_NB,
    RETRIEVAL_THRESHOLD,
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
HALFLIFE_GRID = [3.0, 7.0, 14.0, 30.0]
ALPHA_GRID = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
RETR_TH_GRID = [round(x, 3) for x in np.arange(0.06, 0.305, 0.005)]


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


def fmt_rate(k: int, n: int) -> str:
    lo, hi = wilson(k, n)
    return f"{k / n:6.1%}  ({k}/{n}, KTC95% {lo:.0%}–{hi:.0%})" if n else "  n/a"


def load_split(name: str) -> dict:
    return json.loads((EVAL_DIR / f"{name}.json").read_text(encoding="utf-8"))


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
    out = []
    for c in cases:
        gold = set(c["gold_urls"])
        res = retriever.search(retrieval_query(c["query"]), top_k=k, min_score=0.0)
        out.append(next((i for i, r in enumerate(res, 1) if r.url in gold), None))
    return out


def mrr(ranks) -> float:
    return float(np.mean([1.0 / r if r else 0.0 for r in ranks])) if ranks else 0.0


def conflict_ok(retriever: NewsRetriever, case: dict) -> bool:
    for q in case["queries"]:
        res = [r for r in retriever.search(q, top_k=10, min_score=0.0)
               if "example.test" in r.url]
        if not res or res[0].url != case["expected_url"]:
            return False
    return True


def set_freshness(retriever: NewsRetriever, halflife: float, alpha: float) -> None:
    retriever.freshness_halflife = halflife
    retriever.freshness_alpha = alpha
    retriever.recency = compute_recency(retriever.published_dates, half_life_days=halflife)


def tune_freshness(dev: dict, retriever: NewsRetriever,
                   conflict_retriever: NewsRetriever, case: dict) -> tuple[float, float]:
    hr("PHA 1.2 — DÒ ĐỘ MỚI TRÊN DEV (nửa chu kỳ x alpha)")
    print("Mục tiêu: MRR cao nhất trên dev. RÀNG BUỘC CỨNG: ca tin mâu thuẫn phải đúng.\n")
    print(f"{'half-life':>9} {'alpha':>6} {'R@1':>7} {'MRR':>7}  ca mâu thuẫn")
    print("-" * 46)

    best = None
    for hl in HALFLIFE_GRID:
        for a in ALPHA_GRID:
            set_freshness(retriever, hl, a)
            set_freshness(conflict_retriever, hl, a)
            ranks = ranks_for(retriever, dev["retrieval"])
            r1 = sum(1 for r in ranks if r == 1) / len(ranks)
            m = mrr(ranks)
            ok = conflict_ok(conflict_retriever, case)
            print(f"{hl:>9.0f} {a:>6.1f} {r1:>6.1%} {m:>7.3f}  {'ĐÚNG' if ok else 'sai'}")
            if ok:
                # Hòa điểm -> chọn alpha nhỏ hơn: thay đổi ít hơn so với TF-IDF thuần.
                cand = (round(m, 6), -a, -abs(hl - 7), hl, a)
                if best is None or cand > best:
                    best = cand

    if best is None:
        print("\nKHÔNG có cấu hình nào vừa đúng ca mâu thuẫn -> giữ giá trị trong config.")
        return FRESHNESS_HALFLIFE_DAYS, FRESHNESS_ALPHA
    _, _, _, hl, a = best
    print(f"\n-> chốt nửa chu kỳ = {hl:g} ngày, alpha = {a}")
    return hl, a


def top1_scores(retriever, cases, key):
    out = []
    for c in cases:
        res = retriever.search(retrieval_query(c[key]), top_k=1, min_score=0.0)
        out.append(res[0] if res else None)
    return out


def tune_retrieval_threshold(dev: dict, retriever: NewsRetriever) -> float:
    hr("PHA 1.3 — DÒ NGƯỠNG TRUY HỒI TRÊN DEV")
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
# PHA 2 — báo cáo trên TEST
# ---------------------------------------------------------------------------
def report_test(test: dict, params: dict, retriever: NewsRetriever) -> dict:
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
    return results


def report_end_to_end(test: dict, params: dict) -> dict:
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
    return {"e2e_answer": (ok, n), "e2e_refuse": (refused, len(test["out_of_scope"]))}


# ---------------------------------------------------------------------------
def main() -> int:
    dev, test = load_split("dev"), load_split("test")
    case = json.loads((EVAL_DIR / "conflict_case.json").read_text(encoding="utf-8"))
    print(f"DEV : {len(dev['retrieval'])} truy hồi, {len(dev['out_of_scope'])} ngoài phạm vi, "
          f"{len(dev['intent'])} intent")
    print(f"TEST: {len(test['retrieval'])} truy hồi, {len(test['out_of_scope'])} ngoài phạm vi, "
          f"{len(test['intent'])} intent")

    df = pd.read_csv(CORPUS_RAW_PATH).dropna(subset=["title", "text"]).reset_index(drop=True)
    retriever = NewsRetriever().fit_cached(df)
    df_c = pd.concat([df, pd.DataFrame(case["articles"])], ignore_index=True)
    conflict_retriever = NewsRetriever().fit(df_c)

    # ---------------- PHA 1: DEV
    w_nb, intent_th = tune_intent(dev)
    halflife, alpha = tune_freshness(dev, retriever, conflict_retriever, case)
    set_freshness(retriever, halflife, alpha)
    retr_th = tune_retrieval_threshold(dev, retriever)
    retriever.threshold = retr_th

    params = {
        "intent_w_nb": w_nb,
        "intent_threshold": intent_th,
        "freshness_halflife": halflife,
        "freshness_alpha": alpha,
        "retrieval_threshold": retr_th,
    }
    TUNED_PATH.write_text(json.dumps(params, ensure_ascii=False, indent=1), encoding="utf-8")

    current = {
        "intent_w_nb": INTENT_W_NB, "intent_threshold": INTENT_THRESHOLD,
        "freshness_halflife": FRESHNESS_HALFLIFE_DAYS, "freshness_alpha": FRESHNESS_ALPHA,
        "retrieval_threshold": RETRIEVAL_THRESHOLD,
    }
    diffs = {k: (current[k], v) for k, v in params.items() if abs(current[k] - v) > 1e-9}

    # ---------------- PHA 2: TEST
    results = report_test(test, params, retriever)
    results.update(report_end_to_end(test, params))

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
