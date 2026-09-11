"""Chấm lại một lần chạy RAG (rag_results.json) bằng mã hiện tại — không cần GPU.

Dùng câu PhoGPT đã sinh (lưu trong file) làm "backend cố định", rồi:
  1. chấm lại bằng bộ kiểm tra đã sửa (v1 bị đếm nhầm số thứ tự "1. 2. 3." là số bịa);
  2. mô phỏng "v1 + hậu xử lý v2 + chốt chặn": áp bước làm sạch và hai chốt chặn
     của v2 lên đúng những câu v1 đã sinh — đo riêng phần cải thiện đến từ hậu
     xử lý, tách khỏi phần đến từ prompt mới (phần đó chỉ đo được trên Colab).

Hợp lệ vì định tuyến cục bộ và trên Colab trùng 36/36 câu (đã kiểm tra), nên
nguồn dựng lại ở đây đúng là nguồn PhoGPT đã thấy.

Chạy:  .venv/Scripts/python.exe tools/rescore_rag_run.py data/eval/rag/run1_results.json
"""
import json
import sys
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "src"))

from chatbot import NewsChatbot  # noqa: E402
from rag import LLMBackend, RagChatbot  # noqa: E402


class Replay(LLMBackend):
    name = "replay"

    def __init__(self):
        self.text = ""

    def generate(self, prompt, max_new_tokens=160):
        return self.text


def main(path: str) -> None:
    run = json.loads(Path(path).read_text(encoding="utf-8"))
    bot = NewsChatbot().train()
    case = json.loads((PROJ / "data/eval/conflict_case.json").read_text(encoding="utf-8"))
    df_c = pd.concat([pd.read_csv(PROJ / "data/raw/corpus_raw.csv"), pd.DataFrame(case["articles"])],
                     ignore_index=True)
    conflict_bot = NewsChatbot().train(df=df_c, use_cache=False)

    replay = Replay()
    out = []
    for row in run["rows"]:
        b = conflict_bot if row["nhóm"] == "mâu thuẫn" else bot
        replay.text = row["rag"]
        rec = {"nhóm": row["nhóm"], "câu hỏi": row["câu hỏi"], "route_v1": row["route"]}
        for tag, version in (("v1_rescored", "v1"), ("v1_post_v2", "v2")):
            b.reset()
            r = RagChatbot(b, replay, prompt_version=version, generate_when_guarded=True).respond(row["câu hỏi"])
            rec[f"{tag}_route"] = r.route
            rec[f"{tag}_guard"] = r.guard
            rec[f"{tag}_detail"] = r.guard_detail
            rec[f"{tag}_numbers"] = r.checks.get("unsupported_numbers")
            rec[f"{tag}_support"] = r.checks.get("support_ratio")
            rec[f"{tag}_refused"] = r.checks.get("refused")
            rec[f"{tag}_text"] = r.text if r.route.startswith("rag") else ""
        out.append(rec)

    res = pd.DataFrame(out)
    news = res[(res["nhóm"] == "tin tức") & (res["route_v1"] == "rag")]
    print(f"Câu tin tức đi qua RAG trong lần chạy: {len(news)}")
    old = run["summary"]["news_with_unsupported_numbers"]
    new = int(news["v1_rescored_numbers"].map(bool).sum())
    print(f"  câu có 'số bịa' — bộ kiểm tra cũ: {old} | bộ kiểm tra đã sửa: {new}")
    for _, r in news[news["v1_rescored_numbers"].map(bool)].iterrows():
        print(f"     {r['câu hỏi'][:45]:45} {r['v1_rescored_numbers']}")
    print(f"  support ratio TB (chấm lại): {news['v1_rescored_support'].mean():.3f}")
    g = news["v1_post_v2_guard"].value_counts(dropna=False).to_dict()
    print(f"  mô phỏng hậu xử lý + chốt chặn v2 trên câu v1 — chốt chặn kích hoạt: {g}")
    shown = news[news["v1_post_v2_guard"].isna()]
    print(f"  câu sinh vẫn được hiển thị: {len(shown)}, trong đó còn số bịa: "
          f"{int(shown['v1_post_v2_numbers'].map(bool).sum())}")
    print()
    for grp in ("mâu thuẫn", "bẫy"):
        print(f"--- {grp}")
        for _, r in res[res["nhóm"] == grp].iterrows():
            print(f"  {r['câu hỏi'][:50]:50} | v1: {r['route_v1']:8} | +v2 post: "
                  f"{r['v1_post_v2_route']} {r['v1_post_v2_guard'] or ''} {r['v1_post_v2_detail'] or ''}")

    dest = Path(path).with_name(Path(path).stem + "_rescored.csv")
    res.to_csv(dest, index=False, encoding="utf-8-sig")
    print("\nĐã lưu", dest)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(PROJ / "data" / "eval" / "rag" / "run1_results.json"))
