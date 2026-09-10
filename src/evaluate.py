"""
evaluate.py — Đánh giá định lượng và dò siêu tham số.

Trả lời bốn câu hỏi bắt buộc phải có số liệu trong báo cáo:

  1. Intent classifier chính xác bao nhiêu trên tập TEST viết riêng?
  2. Trọng số ensemble `w_nb` và ngưỡng `INTENT_THRESHOLD` nên đặt bao nhiêu?
  3. Retriever tìm đúng bài ở top-k với tỷ lệ nào (Recall@k, MRR)?
  4. Ngưỡng `RETRIEVAL_THRESHOLD` nào cân bằng tốt nhất giữa "trả lời được câu
     trong phạm vi" và "từ chối câu ngoài phạm vi"?

Chạy:  .venv/Scripts/python.exe src/evaluate.py
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

from config import CORPUS_RAW_PATH, INTENTS_DIR
from entities import expand_query
from intent_classifier import IntentClassifier
from retriever import NewsRetriever

TEST_PATH = INTENTS_DIR / "test_queries.json"


def load_tests() -> dict:
    return json.loads(TEST_PATH.read_text(encoding="utf-8"))


def hr(title: str) -> None:
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


# ---------------------------------------------------------------------------
# 1. Dò siêu tham số cho intent classifier
# ---------------------------------------------------------------------------
def tune_intent(tests: dict) -> tuple[float, float]:
    """Quét lưới (w_nb, threshold) và chọn cấu hình tốt nhất.

    Ta tối ưu ĐỒNG THỜI hai mục tiêu ngược chiều nhau:

      - accuracy: câu thuộc intent CỐ ĐỊNH phải được nhận đúng nhãn và vượt
        ngưỡng, nếu không bot sẽ đi truy hồi thay vì trả lời trực tiếp.

      - safety: câu cần TRUY HỒI hoặc NGOÀI PHẠM VI không được gán nhầm thành
        một intent cố định. Lưu ý: gán vào intent có action="retrieve", hoặc
        rơi xuống dưới ngưỡng, ĐỀU dẫn tới module truy hồi nên đều được tính
        là đúng — chỉ có việc trả lời bằng một câu soạn sẵn sai chỗ mới là lỗi.

    Điểm chọn = trung bình cộng hai tỷ lệ.
    """
    intent_tests = tests["intent_tests"]
    oos = tests["out_of_scope"]

    fixed = [t for t in intent_tests if not t["expected"].startswith("__")]
    to_retrieve = [t["text"] for t in intent_tests if t["expected"] == "__retrieve__"]
    must_not_answer = to_retrieve + oos

    hr("1. DÒ SIÊU THAM SỐ CHO INTENT CLASSIFIER")
    print(f"Tập test: {len(fixed)} câu intent cố định, "
          f"{len(to_retrieve)} câu cần truy hồi, {len(oos)} câu ngoài phạm vi\n")

    print(f"{'w_nb':>6} {'ngưỡng':>8} {'accuracy':>10} {'safety':>9} {'điểm TB':>9}")
    print("-" * 48)

    best = (0.0, 0.0, -1.0)
    rows = []

    for w_nb in [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0]:
        clf = IntentClassifier(w_nb=w_nb).train_from_file()

        # Tính trước dự đoán để không phải train lại cho từng ngưỡng.
        fixed_preds = [(clf.predict(t["text"]), t["expected"]) for t in fixed]
        safe_preds = [clf.predict(q) for q in must_not_answer]

        for threshold in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
            n_correct = sum(
                1 for (tag, conf), exp in fixed_preds
                if conf >= threshold and tag == exp
            )
            accuracy = n_correct / len(fixed)

            # An toàn khi: dưới ngưỡng, HOẶC intent được gán vốn đã đi truy hồi.
            n_safe = sum(
                1 for tag, conf in safe_preds
                if conf < threshold or clf.get_action(tag) == "retrieve"
            )
            safety = n_safe / len(safe_preds)

            score = (accuracy + safety) / 2
            rows.append((w_nb, threshold, accuracy, safety, score))
            if score > best[2]:
                best = (w_nb, threshold, score)

    # In gọn: với mỗi w_nb chỉ hiện ngưỡng tốt nhất.
    by_w = defaultdict(list)
    for r in rows:
        by_w[r[0]].append(r)
    for w_nb in sorted(by_w):
        r = max(by_w[w_nb], key=lambda x: x[4])
        mark = "  <-- tốt nhất" if (r[0], r[1]) == (best[0], best[1]) else ""
        print(f"{r[0]:>6.1f} {r[1]:>8.2f} {r[2]:>9.1%} {r[3]:>13.1%} {r[4]:>8.1%}{mark}")

    print(f"\nChọn: w_nb={best[0]}, INTENT_THRESHOLD={best[1]}")
    print("  w_nb=1.0 là Naive Bayes thuần, w_nb=0.0 là cosine thuần.")
    return best[0], best[1]


# ---------------------------------------------------------------------------
# 2. Báo cáo chi tiết intent classifier
# ---------------------------------------------------------------------------
def report_intent(tests: dict, w_nb: float, threshold: float) -> None:
    hr("2. BÁO CÁO CHI TIẾT INTENT CLASSIFIER")

    clf = IntentClassifier(w_nb=w_nb).train_from_file()
    intent_tests = tests["intent_tests"]
    fixed = [t for t in intent_tests if not t["expected"].startswith("__")]

    per_class = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    errors = []

    for t in fixed:
        tag, conf = clf.predict(t["text"])
        pred = tag if conf >= threshold else "__reject__"
        gold = t["expected"]

        if pred == gold:
            per_class[gold]["tp"] += 1
        else:
            per_class[gold]["fn"] += 1
            if pred != "__reject__":
                per_class[pred]["fp"] += 1
            errors.append((t["text"], gold, pred, conf))

    print(f"{'intent':<22} {'P':>7} {'R':>7} {'F1':>7} {'n':>4}")
    print("-" * 50)
    f1s = []
    for tag in sorted(per_class):
        m = per_class[tag]
        p = m["tp"] / (m["tp"] + m["fp"]) if (m["tp"] + m["fp"]) else 0.0
        r = m["tp"] / (m["tp"] + m["fn"]) if (m["tp"] + m["fn"]) else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) else 0.0
        n = m["tp"] + m["fn"]
        if n:
            f1s.append(f1)
        print(f"{tag:<22} {p:>7.2f} {r:>7.2f} {f1:>7.2f} {n:>4}")

    acc = sum(m["tp"] for m in per_class.values()) / len(fixed)
    print("-" * 50)
    print(f"{'Accuracy':<22} {acc:>7.1%}   (macro-F1 = {np.mean(f1s):.2f})")

    if errors:
        print(f"\nCác câu dự đoán sai ({len(errors)}):")
        for text, gold, pred, conf in errors:
            print(f"  \"{text}\"\n     gold={gold}  pred={pred}  conf={conf:.3f}")

    # Kiểm tra riêng khả năng không trả lời bừa.
    print("\nCâu NGOÀI PHẠM VI (đúng = chuyển sang retrieval, không trả lời soạn sẵn):")
    for q in tests["out_of_scope"]:
        tag, conf = clf.predict(q)
        safe = conf < threshold or clf.get_action(tag) == "retrieve"
        print(f"  [{'OK  ' if safe else 'SAI '}] {conf:.3f} {tag or '(không có)':<20} <- {q}")


# ---------------------------------------------------------------------------
# 3. Đánh giá retriever
# ---------------------------------------------------------------------------
def evaluate_retrieval(tests: dict, retriever: NewsRetriever) -> None:
    hr("3. ĐÁNH GIÁ RETRIEVER (Recall@k, MRR)")

    cases = tests["retrieval_tests"]
    ranks: list[int | None] = []

    for case in cases:
        results = retriever.search(case["query"], top_k=10, min_score=0.0)
        needle = case["expect_title_contains"].lower()
        rank = None
        for i, r in enumerate(results, 1):
            if needle in r.title.lower():
                rank = i
                break
        ranks.append(rank)

    def recall_at(k: int) -> float:
        return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)

    mrr = np.mean([1.0 / r if r else 0.0 for r in ranks])

    print(f"Số truy vấn test: {len(cases)}")
    print(f"  Recall@1  : {recall_at(1):.1%}")
    print(f"  Recall@3  : {recall_at(3):.1%}")
    print(f"  Recall@5  : {recall_at(5):.1%}")
    print(f"  Recall@10 : {recall_at(10):.1%}")
    print(f"  MRR       : {mrr:.3f}")

    misses = [(c, r) for c, r in zip(cases, ranks) if r is None or r > 3]
    if misses:
        print(f"\nTruy vấn KHÔNG vào được top-3 ({len(misses)}):")
        for c, r in misses:
            got = retriever.search(c["query"], top_k=1, min_score=0.0)
            got_title = got[0].title[:55] if got else "(không có kết quả)"
            print(f"  \"{c['query']}\"")
            print(f"     cần: {c['expect_title_contains']}  | hạng: {r} | top-1: {got_title}")


# ---------------------------------------------------------------------------
# 4. Dò ngưỡng cho retriever
# ---------------------------------------------------------------------------
def tune_retrieval(tests: dict, retriever: NewsRetriever) -> float:
    hr("4. DÒ NGƯỠNG CHẤP NHẬN CHO RETRIEVER")

    in_scope = tests["retrieval_tests"]
    oos = tests["out_of_scope"]

    # Điểm top-1 của truy vấn TRONG phạm vi (chỉ tính khi bài đúng đứng đầu).
    in_scores = []
    for case in in_scope:
        res = retriever.search(case["query"], top_k=1, min_score=0.0)
        if res and case["expect_title_contains"].lower() in res[0].title.lower():
            in_scores.append(res[0].score)

    # Điểm top-1 của truy vấn NGOÀI phạm vi — đây là nhiễu cần chặn.
    oos_scores = []
    for q in oos:
        res = retriever.search(expand_query(q), top_k=1, min_score=0.0)
        oos_scores.append(res[0].score if res else 0.0)

    print(f"Truy vấn trong phạm vi (n={len(in_scores)}): "
          f"min={min(in_scores):.3f} p25={np.percentile(in_scores, 25):.3f} "
          f"trung vị={np.median(in_scores):.3f}")
    print(f"Truy vấn ngoài phạm vi (n={len(oos_scores)}): "
          f"trung vị={np.median(oos_scores):.3f} p75={np.percentile(oos_scores, 75):.3f} "
          f"max={max(oos_scores):.3f}")

    print(f"\n{'ngưỡng':>8} {'trả lời được':>14} {'chặn đúng':>12} {'điểm TB':>9}")
    print("-" * 46)

    best = (0.0, -1.0)
    for th in [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.18, 0.22, 0.26, 0.30]:
        answered = sum(1 for s in in_scores if s >= th) / len(in_scores)
        blocked = sum(1 for s in oos_scores if s < th) / len(oos_scores)
        score = (answered + blocked) / 2
        mark = ""
        if score > best[1]:
            best = (th, score)
        print(f"{th:>8.2f} {answered:>13.1%} {blocked:>11.1%} {score:>8.1%}{mark}")

    print(f"\nChọn: RETRIEVAL_THRESHOLD = {best[0]:.2f}")
    print("  Ngưỡng cao -> bot im lặng nhiều (bỏ sót câu trả lời đúng).")
    print("  Ngưỡng thấp -> bot trả lời bừa cho cả câu ngoài phạm vi.")
    return best[0]


# ---------------------------------------------------------------------------
def main() -> int:
    tests = load_tests()

    w_nb, intent_th = tune_intent(tests)
    report_intent(tests, w_nb, intent_th)

    print("\nĐang dựng index truy hồi...")
    df = pd.read_csv(CORPUS_RAW_PATH).dropna(subset=["title", "text"]).reset_index(drop=True)
    retriever = NewsRetriever().fit(df)
    print(f"  {len(df)} bài, {len(retriever.vectorizer.vocabulary_):,} term")

    evaluate_retrieval(tests, retriever)
    retr_th = tune_retrieval(tests, retriever)

    hr("TÓM TẮT — CẬP NHẬT VÀO config.py")
    print(f"  IntentClassifier(w_nb={w_nb})")
    print(f"  INTENT_THRESHOLD    = {intent_th}")
    print(f"  RETRIEVAL_THRESHOLD = {retr_th:.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
