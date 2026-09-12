"""Chấm lại một lần chạy RAG trên Colab bằng mã hiện tại — không cần GPU.

Dùng câu PhoGPT đã sinh (lưu trong file kết quả) làm "backend phát lại", cho đi
qua đúng pipeline hiện tại (làm sạch, chốt chặn, bộ kiểm tra), rồi:
  - chấm lại bằng bộ kiểm tra hiện tại (lần 1 bị đếm nhầm số thứ tự "1. 2. 3.");
  - với câu v1: mô phỏng "v1 + hậu xử lý v2 + chốt chặn" — tách phần cải thiện do
    hậu xử lý khỏi phần do prompt mới (phần đó chỉ đo được bằng cách chạy lại).

Hợp lệ vì định tuyến cục bộ và trên Colab trùng nhau (đã kiểm tra 36/36 câu ở lần
1), nên nguồn dựng lại ở đây đúng là nguồn PhoGPT đã thấy.

Chạy:  .venv/Scripts/python.exe tools/rescore_rag_run.py data/eval/rag/run2_results.json
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
        version = row.get("prompt", "v1")                 # lần 1 chỉ có v1
        # Lần 2 lưu nguyên văn PhoGPT; lần 1 chỉ lưu câu đã qua bước làm sạch v1.
        replay.text = row.get("câu LLM nguyên văn") or row["rag"]
        rec = {"prompt": version, "nhóm": row["nhóm"], "câu hỏi": row["câu hỏi"],
               "route_lúc_chạy": row["route"], "chốt_chặn_lúc_chạy": row.get("chốt chặn", "")}
        tags = [("now", version)] + ([("v1_post_v2", "v2")] if version == "v1" else [])
        for tag, v in tags:
            b.reset()
            r = RagChatbot(b, replay, prompt_version=v, generate_when_guarded=True).respond(row["câu hỏi"])
            rec[f"{tag}_route"] = r.route
            rec[f"{tag}_guard"] = r.guard or ""
            rec[f"{tag}_detail"] = ", ".join(r.guard_detail)
            rec[f"{tag}_numbers"] = ", ".join(r.checks.get("unsupported_numbers", []))
            rec[f"{tag}_support"] = r.checks.get("support_ratio")
            rec[f"{tag}_refused"] = r.checks.get("refused")
            rec[f"{tag}_llm_text"] = r.llm_text
            rec[f"{tag}_shown"] = r.text if r.route.startswith("rag") else ""
        out.append(rec)

    res = pd.DataFrame(out)
    for version in res["prompt"].unique():
        d = res[(res["prompt"] == version) & res["route_lúc_chạy"].str.startswith("rag")]
        news = d[d["nhóm"] == "tin tức"]
        print(f"=== prompt {version} — {len(news)} câu tin tức đi qua RAG")
        print(f"  câu LLM có số không có trong nguồn (bộ kiểm tra hiện tại): "
              f"{int((news['now_numbers'] != '').sum())}")
        for _, r in news[news["now_numbers"] != ""].iterrows():
            print(f"     {r['câu hỏi'][:45]:45} [{r['now_numbers']}]")
        print(f"  support ratio TB: {news['now_support'].mean():.3f}")
        print(f"  chốt chặn (mã hiện tại): {news['now_guard'].replace('', 'không').value_counts().to_dict()}")
        if "v1_post_v2_guard" in news:
            print(f"  mô phỏng hậu xử lý + chốt chặn v2 trên câu v1: "
                  f"{news['v1_post_v2_guard'].replace('', 'không').value_counts().to_dict()}")
        for grp in ("mâu thuẫn", "bẫy"):
            for _, r in d[d["nhóm"] == grp].iterrows():
                print(f"  [{grp}] {r['câu hỏi'][:48]:48} | {r['now_route']} {r['now_guard']} {r['now_detail']}")
        print()

    dest = Path(path).with_name(Path(path).stem + "_rescored.csv")
    res.to_csv(dest, index=False, encoding="utf-8-sig")
    print("Đã lưu", dest)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else str(PROJ / "data" / "eval" / "rag" / "run2_results.json"))
