"""
compare_improvements.py — Số liệu cho docs/09: chọn thiết kế chống gõ sai, và
so sánh TRƯỚC / SAU hai bản sửa (độ mới theo bài cạnh tranh, dự phòng gõ sai).

Hai phần, chạy độc lập với evaluate.py:

  PHẦN A — CHỌN THIẾT KẾ (chỉ DEV)
    Quét: trường đưa vào chỉ mục n-gram ký tự (tiêu đề / tiêu đề+mô tả / cả bài)
          x n (3, 4) x luật chấp nhận (chỉ ký tự / cộng với cosine mức từ).
    Với mỗi thiết kế, dò ngưỡng tốt nhất bằng cùng mục tiêu với evaluate.py
    PHA 1.5, rồi in số câu cứu được / trả sai / lọt ngoài phạm vi.
    Thiết kế này chỉ xét bài ĐỨNG ĐẦU của mỗi đường (sàng lọc nhanh); ngưỡng
    cuối cùng vẫn do evaluate.py dò trên đúng cài đặt thật trong retriever.

  PHẦN B — TRƯỚC / SAU (TEST, tham số đã chốt; không dò gì ở đây)
    TRƯỚC = cấu hình lúc nộp bài: mốc độ mới "corpus", alpha 0.6, không dự phòng.
    SAU   = cấu hình hiện hành trong config.py.
    Cả hai chạy trên CÙNG corpus hiện tại qua bot.respond() như người dùng thật.

Chạy:  .venv/Scripts/python.exe tools/compare_improvements.py [--skip-a] [--skip-b]
"""

import json
import sys
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "src"))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import evaluate as ev  # noqa: E402
from chatbot import NewsChatbot  # noqa: E402
from config import CORPUS_RAW_PATH, TOP_K  # noqa: E402
from preprocess import fold_for_chars  # noqa: E402
from retriever import NewsRetriever  # noqa: E402
from vectorizer import TfidfVectorizer, cosine_similarity, make_char_ngrams  # noqa: E402

FIELDS = {
    "tiêu đề": lambda r: str(r.title),
    "tiêu đề + mô tả": lambda r: f"{r.title} {r.description}",
    "cả bài (tiêu đề x3, mô tả x2)": lambda r: " ".join(
        [str(r.title)] * 3 + [str(r.description)] * 2 + [str(r.text)]),
}
RULES = {"chỉ n-gram ký tự": False, "cộng cosine mức từ": True}


# ---------------------------------------------------------------------------
def part_a(df: pd.DataFrame) -> None:
    ev.hr("PHẦN A — CHỌN THIẾT KẾ CHỈ MỤC GÕ SAI (DEV + dev_typo + ngoài phạm vi)")
    dev, dev_typo = ev.load_split("dev"), ev.load_typo_split("dev")
    retriever = NewsRetriever(fuzzy_threshold=None).fit_cached(df)
    url_of = df["url"].tolist()

    # Đường chính: câu nào đã được trả lời (đúng/sai), câu nào bị từ chối.
    in_cases = dev["retrieval"] + dev_typo["retrieval"]
    refused, base_correct = [], 0
    for c in in_cases:
        q = ev.retrieval_query(c["query"])
        d, _, acc = retriever.rank(q, top_k=1)[0]
        if acc >= retriever.threshold:
            base_correct += url_of[d] in set(c["gold_urls"])
        else:
            refused.append((q, set(c["gold_urls"])))
    oos = []
    for x in dev["out_of_scope"]:
        q = ev.retrieval_query(x["text"])
        if retriever.rank(q, top_k=1)[0][2] < retriever.threshold:
            oos.append((q, set()))
    n_in, n_oos = len(in_cases), len(dev["out_of_scope"])
    base_leak = n_oos - len(oos)
    print(f"Đường chính: {base_correct}/{n_in} câu trong phạm vi trả lời đúng, "
          f"{len(refused)} câu bị từ chối; ngoài phạm vi lọt {base_leak}/{n_oos}\n")

    def word_scores(q):
        q_vec, matrix = retriever._select_index(q)
        return cosine_similarity(q_vec, matrix)[0] if q_vec.nnz else np.zeros(len(df))

    def query_chars(q, n):
        tokens, _ = retriever._query_tokens(q)
        return make_char_ngrams(fold_for_chars(" ".join(tokens)), n)

    print(f"{'trường':<30} {'n':>2} {'luật':<20} {'ngưỡng':>6} {'cứu':>4} {'sai':>4} {'lọt':>4}")
    print("-" * 78)
    for fname, field in FIELDS.items():
        texts = [fold_for_chars(field(r)) for r in df.itertuples()]
        for n in (3, 4):
            vec = TfidfVectorizer(sublinear_tf=True)
            matrix = vec.fit_transform([make_char_ngrams(t, n) for t in texts])
            scored = []
            for kind, items in (("in", refused), ("oos", oos)):
                for q, gold in items:
                    qv = vec.transform([query_chars(q, n)])
                    chars = cosine_similarity(qv, matrix)[0] if qv.nnz else np.zeros(len(df))
                    scored.append((kind, gold, word_scores(q), chars))
            for rname, add_word in RULES.items():
                best = None
                for th in ev.FUZZY_TH_GRID + [round(x, 2) for x in np.arange(0.15, 0.30, 0.01)]:
                    saved = wrong = leak = 0
                    for kind, gold, words, chars in scored:
                        s = words + chars if add_word else chars
                        i = int(np.argmax(s))
                        if s[i] < th:
                            continue
                        if kind == "oos":
                            leak += 1
                        elif url_of[i] in gold:
                            saved += 1
                        else:
                            wrong += 1
                    obj = ((base_correct + saved) / n_in + (n_oos - base_leak - leak) / n_oos) / 2
                    cand = (round(obj, 6), -wrong, th, saved, wrong, leak)
                    if best is None or cand > best:
                        best = cand
                _, _, th, saved, wrong, leak = best
                print(f"{fname:<30} {n:>2} {rname:<20} {th:>6.2f} {saved:>4} {wrong:>4} {leak:>4}")


# ---------------------------------------------------------------------------
def part_b() -> None:
    ev.hr("PHẦN B — TRƯỚC / SAU TRÊN TEST (cùng corpus hiện tại, qua bot.respond)")
    test, test_typo = ev.load_split("test"), ev.load_typo_split("test")
    case = json.loads((ev.EVAL_DIR / "conflict_case.json").read_text(encoding="utf-8"))
    # "TRƯỚC" phải dựng lại ĐẦY ĐỦ hành vi lúc nộp bài: mốc độ mới theo cả kho,
    # alpha 0.6, không có đường dự phòng gõ sai, KHÔNG ghép từ ghép ở câu hỏi, và
    # vẫn nhân đôi thực thể theo kiểu cũ. Thiếu một trong số đó là so bản mới với
    # chính nó.
    configs = {
        "TRƯỚC (nộp bài)": dict(freshness_reference="corpus", freshness_alpha=0.6,
                                use_fuzzy=False, glue_compounds=False, legacy_expand=True),
        "SAU (hiện hành)": dict(),
    }
    rows = {}
    for name, kwargs in configs.items():
        bot = NewsChatbot(**kwargs).train()
        r = bot.retriever
        legacy = bool(kwargs.get("legacy_expand"))       # nhánh TRƯỚC
        if legacy:
            # ranks_for đi qua ev.retrieval_query (expand_query hiện hành = đồng
            # nhất); với nhánh TRƯỚC phải dùng lại bản nhân đôi cũ.
            from entities import expand_query_legacy
            url_of = r.df["url"].tolist()
            ranks = []
            for c in test["retrieval"]:
                gold = set(c["gold_urls"])
                q = expand_query_legacy(ev.prepared(c["query"]))
                ranked = r.rank(q, top_k=10)
                ranks.append(next((i for i, (d, _, _) in enumerate(ranked, 1)
                                   if url_of[d] in gold), None))
        else:
            ranks = ev.ranks_for(r, test["retrieval"])

        def answer_rate(cases):
            ok = wrong = 0
            for c in cases:
                bot.reset()
                res = bot.respond(c["query"]).results
                if res:
                    ok += res[0].url in set(c["gold_urls"])
                    wrong += res[0].url not in set(c["gold_urls"])
            return ok, wrong

        ok, wrong = answer_rate(test["retrieval"])
        typo_ok, typo_wrong = answer_rate(test_typo["retrieval"])
        refused = 0
        for x in test["out_of_scope"]:
            bot.reset()
            refused += bot.respond(x["text"]).route == "fallback"

        now, later = ev.conflict_retrievers(r.df, case)
        for cr in (now, later):
            cr.freshness_reference, cr.freshness_alpha = r.freshness_reference, r.freshness_alpha
            cr.fuzzy_threshold = r.fuzzy_threshold
        rows[name] = {
            "R@1": f"{sum(1 for x in ranks if x == 1)}/{len(ranks)}",
            "MRR": f"{ev.mrr(ranks):.3f}",
            "ca Cát Linh (kho hiện tại)": "đúng" if ev.conflict_ok(now, case) else "SAI",
            "ca Cát Linh (+ bài tương lai)": "đúng" if ev.conflict_ok(later, case) else "SAI",
            "câu sạch: đúng / sai": f"{ok} / {wrong} (trên {len(test['retrieval'])})",
            "câu gõ sai: đúng / sai": f"{typo_ok} / {typo_wrong} (trên {len(test_typo['retrieval'])})",
            "ngoài phạm vi bị từ chối": f"{refused}/{len(test['out_of_scope'])}",
        }
    names = list(rows)
    print(f"\n{'chỉ số':<32}" + "".join(f"{n:>26}" for n in names))
    print("-" * (32 + 26 * len(names)))
    for key in rows[names[0]]:
        print(f"{key:<32}" + "".join(f"{rows[n][key]:>26}" for n in names))


def main() -> int:
    df = pd.read_csv(CORPUS_RAW_PATH).dropna(subset=["title", "text"]).reset_index(drop=True)
    print(f"CORPUS: {len(df)} bài; TOP_K = {TOP_K}")
    if "--skip-a" not in sys.argv:
        part_a(df)
    if "--skip-b" not in sys.argv:
        part_b()
    return 0


if __name__ == "__main__":
    sys.exit(main())
