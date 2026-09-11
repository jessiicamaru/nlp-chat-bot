"""Sinh notebooks/RAG_PhoGPT_Colab.ipynb — notebook chạy RAG với PhoGPT trên Google Colab.

Notebook tự nhận biết môi trường:
  - Trên Colab  : tải PhoGPT, chạy thật trên GPU T4.
  - Chạy cục bộ : DRY-RUN với EchoBackend (giả lập) — để kiểm tra mọi ô chạy
                  không lỗi trước khi mang lên Colab.
"""
import json
from pathlib import Path

cells = []


def _cell(kind, text, **extra):
    # id cố định: nbformat 4.5 yêu cầu mỗi ô có id; thiếu thì VS Code/Jupyter tự
    # thêm khi mở file, làm file "bị sửa" dù không ai sửa gì.
    cells.append({"cell_type": kind, "id": f"cell-{len(cells):02d}", "metadata": {},
                  "source": text.strip("\n").splitlines(keepends=True), **extra})


def md(text):
    _cell("markdown", text)


def code(text):
    _cell("code", text, execution_count=None, outputs=[])


md(r"""
# RAG với PhoGPT-4B-Chat — Chatbot tin tức tiếng Việt (lần chạy 2: so sánh prompt v1 / v2)

Notebook này thêm bước **sinh câu trả lời tự nhiên** lên trên chatbot trích xuất
đã xây from scratch:

```text
câu hỏi ─> [truy hồi TF-IDF tự cài đặt] ─> 1–3 bài báo ─> [chốt chặn giả định] ─> [PhoGPT] ─> [chốt chặn số bịa]
```

**Nguyên tắc an toàn:** PhoGPT chỉ được gọi khi phần truy hồi đã tìm được bài báo
vượt ngưỡng. Câu ngoài phạm vi bị từ chối **trước** khi tới PhoGPT.

**Lần chạy này** chạy CÙNG một bộ câu hỏi (giống lần 1) qua hai phiên bản prompt
trên CÙNG một mô hình, để so sánh công bằng:

- **v1** — prompt của lần chạy đầu (6 quy tắc đánh số), không chốt chặn.
- **v2** — prompt ngắn, câu từ khóa được chuyển thành câu hỏi, làm sạch phần mô
  hình chép lại prompt, thêm hai chốt chặn tất định. Lý do từng thay đổi: `docs/07`.

## Cách chạy (khoảng 10–15 phút)

1. **Runtime → Change runtime type → T4 GPU** → Save.
2. **Runtime → Run all**.
3. Khi ô số 2 hỏi, **chọn file `rag_bundle.zip`** (bản MỚI) để tải lên.
4. Chờ chạy xong. Ô cuối cùng sẽ **tải về** 2 file kết quả:
   `rag_results_run2.json` và `rag_samples_run2.csv`.
5. Chép 2 file đó vào thư mục `final-project/local/`.

> Lần này dùng bản GGUF **Q8_0** (~3.9 GB, gần như không mất chất lượng) thay cho
> Q4_K_M (4 bit) của lần 1. Tải về máy Colab, không tốn dung lượng máy bạn.
""")

md(r"""
## 0. Cấu hình
""")

code(r"""
# ----- Có thể chỉnh -----
BACKEND = "llamacpp"    # "llamacpp" | "transformers" | "auto" (lần 1: transformers lỗi triton_pre_mlir)
GGUF_FILE = "PhoGPT-4B-Chat-Q8_0.gguf"     # lần 1 dùng "PhoGPT-4B-Chat-Q4_K_M.gguf"
PROMPT_VERSIONS = ["v1", "v2"]
MAX_NEW_TOKENS = {"v1": 256, "v2": 160}  # v1 giữ đúng như lần 1; v2 chỉ cần 2-3 câu
EVAL_SPLIT = "dev"      # "dev" khi còn đang chỉnh prompt; đổi sang "test" CHỈ MỘT LẦN khi đã chốt
N_EVAL = 25             # số câu hỏi tin tức lấy mẫu (cùng SEED -> cùng 25 câu như lần 1)
SEED = 2026
RUN_TAG = "run2"        # hậu tố tên file kết quả, để không đè file lần trước
# ------------------------

import os, sys, time, json, random, zipfile, subprocess
from pathlib import Path

try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

DRY_RUN = not IN_COLAB
print("Môi trường:", "Google Colab" if IN_COLAB else "CỤC BỘ — chế độ DRY-RUN (backend giả lập)")
print("Python    :", sys.version.split()[0])
""")

md(r"""
## 1. Kiểm tra GPU
""")

code(r"""
if IN_COLAB:
    r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,memory.used",
                        "--format=csv"], capture_output=True, text=True)
    print(r.stdout or r.stderr)
    if r.returncode != 0:
        print("⚠️ KHÔNG THẤY GPU. Vào Runtime → Change runtime type → chọn T4 GPU rồi chạy lại.")
else:
    print("(bỏ qua — đang chạy cục bộ)")
""")

md(r"""
## 2. Nạp mã nguồn dự án

Trên Colab: chọn file `rag_bundle.zip` khi được hỏi. Chạy cục bộ: dùng thẳng thư mục dự án.
""")

code(r"""
if IN_COLAB:
    PROJECT = Path("/content/final-project")
    if not (PROJECT / "src" / "chatbot.py").exists():
        from google.colab import files
        print("Chọn file rag_bundle.zip ...")
        uploaded = files.upload()
        zip_name = next(iter(uploaded))
        with zipfile.ZipFile(zip_name) as z:
            z.extractall("/content")
else:
    # Tìm ngược lên từ thư mục hiện tại cho tới khi gặp thư mục chứa src/chatbot.py.
    here = Path.cwd()
    PROJECT = next((p for p in [here, *here.parents] if (p / "src" / "chatbot.py").exists()), here)

assert (PROJECT / "src" / "chatbot.py").exists(), f"Không tìm thấy mã nguồn trong {PROJECT}"
sys.path.insert(0, str(PROJECT / "src"))
OUT_DIR = PROJECT / "rag_out"
OUT_DIR.mkdir(exist_ok=True)
print("Dự án:", PROJECT)
""")

md(r"""
## 3. Cài thư viện
""")

code(r"""
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "underthesea"], check=False)
import underthesea
print("underthesea:", underthesea.__version__)
""")

md(r"""
## 4. Dựng chatbot trích xuất (phần tự cài đặt)
""")

code(r"""
from chatbot import NewsChatbot

t0 = time.time()
bot = NewsChatbot().train(verbose=True)
print(f"Dựng xong trong {time.time() - t0:.1f}s — {bot.retriever.stats()['n_documents']} bài báo")
print()
print(bot.respond("tin về đảo hải nam").text[:300])
""")

md(r"""
## 5. Nạp PhoGPT

Mặc định dùng bản GGUF chạy bằng `llama.cpp` trên GPU. Lần chạy 1, `transformers`
(bản 5.x trên Colab) không nạp được mã tùy biến của PhoGPT (thiếu gói
`triton_pre_mlir`); `BACKEND = "auto"` vẫn thử nó trước nếu muốn.
""")

code(r"""
from rag import EchoBackend, LlamaCppBackend, RagChatbot, TransformersBackend

backend, load_errors = None, []
t0 = time.time()

if DRY_RUN:
    backend = EchoBackend()
else:
    if BACKEND in ("auto", "transformers"):
        try:
            backend = TransformersBackend()
        except Exception as exc:
            load_errors.append(("transformers", repr(exc)[:400]))
    if backend is None and BACKEND in ("auto", "llamacpp"):
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-q", "llama-cpp-python",
                            "--extra-index-url",
                            "https://abetlen.github.io/llama-cpp-python/whl/cu124"], check=False)
            backend = LlamaCppBackend(filename=GGUF_FILE)
        except Exception as exc:
            load_errors.append(("llama.cpp", repr(exc)[:400]))

for name, err in load_errors:
    print(f"⚠️ {name} lỗi: {err}")
assert backend is not None, "Không nạp được backend nào — xem lỗi ở trên và gửi lại."
print(f"Backend: {backend.name} {getattr(backend, 'filename', '')}  (nạp trong {time.time() - t0:.0f}s)")


def make_rag(chatbot, version):
    # generate_when_guarded: khi chốt chặn giả định đã chặn, VẪN gọi PhoGPT để ghi
    # lại nó định trả lời gì (cột "câu LLM") — đo được chốt chặn đã ngăn được gì.
    return RagChatbot(chatbot, backend, prompt_version=version,
                      max_new_tokens=MAX_NEW_TOKENS[version], generate_when_guarded=True)


rag = make_rag(bot, "v2")
_ = backend.generate("### Câu hỏi: Xin chào\n### Trả lời:", max_new_tokens=8)   # khởi động
""")

md(r"""
## 6. Thử nhanh — so sánh trích xuất và RAG
""")

code(r"""
def show(question):
    bot.reset()
    r = rag.respond(question)
    print("=" * 90)
    print("HỎI :", question)
    print(f"[route={r.route}  guard={r.guard}  |  {r.latency_s:.1f}s]")
    print("\nTRÍCH XUẤT (bot gốc):\n", r.extractive_text[:350])
    if r.route.startswith("rag"):
        print("\nHIỂN THỊ (RAG v2):\n", r.text[:600])
        if r.guard:
            print(f"\n[chốt chặn '{r.guard}' {r.guard_detail}] — PhoGPT định trả lời:\n", r.llm_text)
        print("\nNguồn:", [f"[{s['id']}] {s['title'][:50]} ({s['published']})" for s in r.sources])
        print("Kiểm tra:", r.checks)

for q in ["cho tôi biết về đảo hải nam", "bt gì về vụ giá xăng k",
          "tin sức khỏe về ăn chuối", "đảo Hải Nam miễn visa từ năm 2015 phải không",
          "thời tiết sao hỏa hôm nay"]:
    show(q)
""")

md(r"""
## 7. Đánh giá

Bốn nhóm câu hỏi:

| Nhóm | Mục đích | Tự động chấm thế nào |
|---|---|---|
| **tin tức** | câu hỏi thật lấy mẫu từ tập đánh giá | độ trung thành (số bịa, tỷ lệ từ có trong nguồn, trích dẫn) |
| **mâu thuẫn** | 2 bài mâu thuẫn (giá vé Cát Linh) | câu trả lời phải theo bài MỚI (8.000 đồng, đã hoãn tăng giá) |
| **bẫy** | hỏi chi tiết mà bài báo KHÔNG có | phải từ chối, không được bịa |
| **ngoài phạm vi** | câu không liên quan kho dữ liệu | PhoGPT không được gọi |

Mỗi câu chạy qua **cả v1 và v2** (cột `prompt`). Chấm tự động chỉ là **tín hiệu
cảnh báo**. File CSV xuất ra có sẵn cột để chấm tay (tự nhiên 1–5, trung thành 1–5).
""")

code(r"""
import pandas as pd

split = json.loads((PROJECT / "data" / "eval" / f"{EVAL_SPLIT}.json").read_text(encoding="utf-8"))
rng = random.Random(SEED)
news = rng.sample(split["retrieval"], min(N_EVAL, len(split["retrieval"])))
oos = rng.sample(split["out_of_scope"], min(5, len(split["out_of_scope"])))

# Câu BẪY: đã kiểm tra thủ công rằng chi tiết được hỏi KHÔNG có trong bài báo tương ứng.
# Cách diễn đạt được chọn để câu hỏi ĐI TỚI được PhoGPT. Hai cách hỏi tự nhiên hơn
# ("... lãi bao nhiêu trong năm 2020", "... 30 ngày đúng không") bị intent
# classifier hiểu nhầm thành "thống kê kho dữ liệu" và "đồng ý" — ghi nhận là
# điểm yếu của intent classifier (xem báo cáo), không phải của RAG.
traps = [
    ("tai nghe AirPods 5 có chống nước chuẩn IP68 không", "bài nói chống nước được cải thiện, KHÔNG nêu chuẩn IP68"),
    ("Messi mua CLB Eldense với giá bao nhiêu tiền", "bài không nêu giá mua"),
    ("vì sao kem Tràng Tiền phải đóng cửa", "bài không nêu lý do đóng cửa"),
    ("lợi nhuận năm 2020 của metro Bến Thành Suối Tiên", "bài chỉ nói mục tiêu 2026-2030"),
    ("giữ cơm nguội trong tủ lạnh 30 ngày có sao không", "GIẢ ĐỊNH SAI: bài khuyên không giữ 5-7 ngày"),
    ("đảo Hải Nam miễn visa cho khách Việt từ năm 2015 phải không", "GIẢ ĐỊNH SAI: bài không nói năm 2015"),
]

case = json.loads((PROJECT / "data" / "eval" / "conflict_case.json").read_text(encoding="utf-8"))
print(f"tin tức: {len(news)} | mâu thuẫn: {len(case['queries'])} | bẫy: {len(traps)} | "
      f"ngoài phạm vi: {len(oos)}   (tập {EVAL_SPLIT})")
""")

code(r"""
# Bot riêng cho ca mâu thuẫn: corpus + 2 bài giá vé tàu Cát Linh (không ghi đè cache chính)
df_conflict = pd.concat([pd.read_csv(PROJECT / "data" / "raw" / "corpus_raw.csv"),
                         pd.DataFrame(case["articles"])], ignore_index=True)
conflict_bot = NewsChatbot().train(df=df_conflict, use_cache=False)
print("Đã dựng bot cho ca mâu thuẫn.")
""")

code(r"""
def conflict_verdict(answer):
    a = answer.lower()
    new = "8.000" in a or "8000" in a or "hoãn" in a or "giữ nguyên" in a
    old = "15.000" in a or "15000" in a
    if new:
        return "ĐÚNG (theo tin mới)"
    if old:
        return "SAI (theo tin cũ)"
    return "CẦN ĐỌC"


def trap_verdict(r):
    # Chấm trên câu HIỂN THỊ cho người dùng.
    if not r.route.startswith("rag"):
        return "không qua RAG (truy hồi không tìm thấy)"
    if r.guard == "premise":
        return "ĐÚNG (chốt chặn giả định)"
    if r.guard in ("numbers", "empty"):
        return "AN TOÀN (hiển thị trích xuất)"
    if r.checks.get("refused"):
        return "ĐÚNG (từ chối)"
    if r.checks.get("unsupported_numbers"):
        return "SAI (có số bịa)"
    return "CẦN ĐỌC"


rows = []

def run(version, group, question, chatbot, gold=None, note="", style=""):
    chatbot.bot.reset()
    r = chatbot.respond(question)
    is_rag = r.route.startswith("rag")
    row = {
        "prompt": version,
        "nhóm": group, "câu hỏi": question, "kiểu gõ": style, "ghi chú": note,
        "route": r.route, "rag": r.text if is_rag else "",
        "chốt chặn": r.guard or "", "chi tiết chốt chặn": ", ".join(r.guard_detail),
        "câu LLM": r.llm_text, "câu LLM nguyên văn": r.raw_generation,
        "trích xuất": r.extractive_text, "độ trễ (s)": round(r.latency_s, 2),
        "nguồn": " | ".join(f"[{s['id']}] {s['title']} ({s['published']})" for s in r.sources),
        "số bịa": ", ".join(r.checks.get("unsupported_numbers", [])),
        "tỷ lệ từ có trong nguồn": r.checks.get("support_ratio"),
        "trích dẫn hợp lệ": r.checks.get("citations_valid"),
        "trích dẫn sai": r.checks.get("citations_invalid"),
        "từ chối": r.checks.get("refused"),
        "truy hồi đúng bài": (bool(r.sources) and r.sources[0]["url"] in set(gold)) if gold else None,
        "đánh giá tự động": "",
        "điểm tự nhiên (1-5)": "", "điểm trung thành (1-5)": "", "nhận xét": "",
    }
    if group == "mâu thuẫn":
        row["đánh giá tự động"] = conflict_verdict(r.text)
    elif group == "bẫy":
        row["đánh giá tự động"] = trap_verdict(r)
    elif group == "ngoài phạm vi":
        row["đánh giá tự động"] = "ĐÚNG (không gọi LLM)" if not is_rag else "SAI (đã gọi LLM)"
    rows.append(row)
    return r


t_all = time.time()
for version in PROMPT_VERSIONS:
    news_rag, conflict_rag = make_rag(bot, version), make_rag(conflict_bot, version)
    for i, c in enumerate(news, 1):
        run(version, "tin tức", c["query"], news_rag, gold=c["gold_urls"], style=c.get("style", ""))
        if i % 5 == 0:
            print(f"  [{version}] tin tức {i}/{len(news)}  ({time.time() - t_all:.0f}s)")
    for q in case["queries"]:
        run(version, "mâu thuẫn", q, conflict_rag)
    for q, note in traps:
        run(version, "bẫy", q, news_rag, note=note)
    for x in oos:
        run(version, "ngoài phạm vi", x["text"], news_rag, style=x.get("style", ""))
print(f"Xong {len(rows)} lượt trong {time.time() - t_all:.0f}s")
""")

md(r"""
## 8. Tổng hợp
""")

code(r"""
df_res = pd.DataFrame(rows)


def summarize(d):
    rag_rows = d[d["route"].str.startswith("rag")]
    news_rag = rag_rows[rag_rows["nhóm"] == "tin tức"]
    called = rag_rows[rag_rows["độ trễ (s)"] > 0]
    return {
        "n_news": int((d["nhóm"] == "tin tức").sum()),
        "n_news_answered_by_rag": int(len(news_rag)),
        "news_guard": news_rag["chốt chặn"].replace("", "không").value_counts().to_dict(),
        "latency_mean_s": round(float(called["độ trễ (s)"].mean()), 2) if len(called) else None,
        "latency_p90_s": round(float(called["độ trễ (s)"].quantile(0.9)), 2) if len(called) else None,
        # Chấm trên câu PhoGPT sinh (cột "câu LLM"), kể cả khi bị chặn không hiển thị.
        "news_llm_with_unsupported_numbers": int((news_rag["số bịa"] != "").sum()),
        "news_llm_support_ratio_mean": round(float(news_rag["tỷ lệ từ có trong nguồn"].dropna().mean()), 3)
                                       if len(news_rag) else None,
        "news_llm_refused": int(news_rag["từ chối"].fillna(False).astype(bool).sum()),
        "news_with_valid_citation": int((news_rag["trích dẫn hợp lệ"].fillna(0) > 0).sum()),
        "news_with_invalid_citation": int((news_rag["trích dẫn sai"].fillna(0) > 0).sum()),
        "news_retrieval_correct": int(news_rag["truy hồi đúng bài"].fillna(False).astype(bool).sum()),
        "conflict": d[d["nhóm"] == "mâu thuẫn"]["đánh giá tự động"].value_counts().to_dict(),
        "traps": d[d["nhóm"] == "bẫy"]["đánh giá tự động"].value_counts().to_dict(),
        "out_of_scope": d[d["nhóm"] == "ngoài phạm vi"]["đánh giá tự động"].value_counts().to_dict(),
    }


summary = {
    "backend": backend.name,
    "gguf_file": getattr(backend, "filename", None),
    "eval_split": EVAL_SPLIT,
    "max_new_tokens": MAX_NEW_TOKENS,
    "by_prompt": {v: summarize(df_res[df_res["prompt"] == v]) for v in PROMPT_VERSIONS},
}
if IN_COLAB:
    import torch, transformers
    summary["env"] = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                      "torch": torch.__version__, "transformers": transformers.__version__,
                      "underthesea": underthesea.__version__, "python": sys.version.split()[0]}
summary["load_errors"] = load_errors
summary["dry_run"] = DRY_RUN

for k, v in summary.items():
    if k != "by_prompt":
        print(f"{k:16}: {v}")
print()
print(pd.DataFrame(summary["by_prompt"]).to_string())
""")

code(r"""
pd.set_option("display.max_colwidth", 160)
cols = ["prompt", "nhóm", "câu hỏi", "rag", "chốt chặn", "số bịa", "đánh giá tự động"]
df_res[df_res["nhóm"] != "tin tức"][cols]
""")

md(r"""
## 9. Lưu và tải kết quả
""")

code(r"""
res_path = OUT_DIR / f"rag_results_{RUN_TAG}.json"
csv_path = OUT_DIR / f"rag_samples_{RUN_TAG}.csv"
res_path.write_text(json.dumps({"summary": summary, "rows": rows}, ensure_ascii=False, indent=1,
                               default=str), encoding="utf-8")
df_res.to_csv(csv_path, index=False, encoding="utf-8-sig")
print("Đã lưu:", res_path, "và", csv_path)

if IN_COLAB and not DRY_RUN:
    from google.colab import files
    files.download(str(res_path))
    files.download(str(csv_path))
""")

md(r"""
## 10. Chat thử (tùy chọn)

Gõ câu hỏi, gõ `q` để thoát. Chỉ chạy được trên Colab/Jupyter có ô nhập liệu.
""")

code(r"""
if IN_COLAB:
    while True:
        q = input("Bạn > ").strip()
        if q.lower() in ("q", "quit", "exit", ""):
            break
        show(q)
else:
    print("(bỏ qua ô chat khi chạy cục bộ tự động)")
""")

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"},
                   "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}},
      "nbformat": 4, "nbformat_minor": 5}

out = Path(__file__).resolve().parent.parent / "notebooks" / "RAG_PhoGPT_Colab.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Wrote", out, f"({len(cells)} cells)")
