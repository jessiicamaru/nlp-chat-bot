"""
rag.py — Retrieval-Augmented Generation: truy hồi (tự cài đặt) + sinh câu (PhoGPT).

## Vì sao có module này

Chatbot gốc TRÍCH XUẤT 100%: nó trả về câu có sẵn trong bài báo. Thí nghiệm
n-gram (docs/04) đã chứng minh tự sinh văn bản bằng mô hình nhỏ tự cài đặt thì
bịa sự kiện. Muốn câu trả lời TỰ NHIÊN mà vẫn ĐÚNG SỰ THẬT, cách chuẩn là RAG:

    câu hỏi ──> [truy hồi TF-IDF tự cài đặt] ──> 1-3 bài báo liên quan
                                                    │
                          PhoGPT diễn đạt lại  <────┘  (chỉ được dùng các bài này)

Module này là một LỚP THÊM, không thay thế gì: tắt nó đi thì chatbot vẫn chạy
nguyên như cũ. Đây là phần DUY NHẤT của đồ án dùng mô hình tiền huấn luyện, và
nằm ngoài phạm vi "from scratch" — ghi rõ trong báo cáo.

## Nguyên tắc an toàn

1. LLM CHỈ được gọi khi phần truy hồi đã tìm được bằng chứng vượt ngưỡng. Câu
   ngoài phạm vi bị từ chối TRƯỚC khi tới LLM.
2. (v2) Chốt chặn GIẢ ĐỊNH: câu hỏi nêu con số / mã hiệu (2015, IP68, 30 ngày...)
   mà bài báo KHÔNG hề nhắc tới -> không gọi LLM, trả lời rằng bài báo không
   nhắc chi tiết đó. Lần chạy v1 cho thấy PhoGPT-4B gần như luôn "đồng ý" với
   giả định sai ("Đúng.", "lợi nhuận năm 2020 là 27 tỷ đồng").
3. (v2) Chốt chặn SỐ BỊA: câu trả lời chứa con số không có trong nguồn -> không
   hiển thị câu sinh ra, hiển thị câu trích xuất gốc thay thế.

Hai chốt chặn là quy tắc tất định (regex + so khớp), kiểm thử được mà không cần
GPU — lấy tinh thần Lab01 (regex, trích xuất thực thể) làm hàng rào cho LLM.

## Hai phiên bản prompt

  - v1: prompt đầu tiên (6 quy tắc đánh số, yêu cầu trích dẫn [i]). Kết quả chạy
        thật trên Colab: xem docs/07 — mô hình chép lại định dạng prompt, bịa
        ngày tháng, trả lời bằng danh sách đánh số, đồng ý với giả định sai.
  - v2: prompt ngắn, không đánh số, câu hỏi đặt cuối; câu dạng từ khóa được
        chuyển thành câu hỏi; hậu xử lý cắt phần chép prompt; hai chốt chặn.
  Giữ cả hai để so sánh A/B trong CÙNG một phiên Colab.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from config import CONFIG_RETRIEVAL
from preprocess import load_stopwords, normalize_basic, tokenize

# Mẫu prompt chính thức của PhoGPT-4B-Chat (github.com/VinAIResearch/PhoGPT,
# khớp với phogpt_4b_chat_preset.json trong repo GGUF của VinAI).
PHOGPT_TEMPLATE = "### Câu hỏi: {instruction}\n### Trả lời:"

REFUSAL_PHRASE = "Các bài báo hiện có không đề cập đến điều này."

PROMPT_VERSIONS = ("v1", "v2")
DEFAULT_PROMPT_VERSION = "v2"

INSTRUCTION_V1 = """Bạn là trợ lý tin tức. Hãy trả lời câu hỏi CHỈ dựa trên các bài báo được cung cấp dưới đây.

Quy tắc:
1. Trả lời ngắn gọn bằng tiếng Việt tự nhiên, từ 2 đến 4 câu.
2. Chỉ dùng thông tin có trong các bài báo. Không thêm kiến thức bên ngoài, không đoán.
3. Ghi nguồn bằng số trong ngoặc vuông ngay sau thông tin, ví dụ [1] hoặc [2].
4. Nếu các bài báo mâu thuẫn nhau, hãy ưu tiên bài có ngày đăng MỚI HƠN và nói rõ điều đó.
5. Nếu các bài báo không có thông tin để trả lời, hãy trả lời đúng một câu: "{refusal}"
6. Nếu câu hỏi chứa giả định sai so với bài báo, hãy chỉ ra điều đó.

Các bài báo:
{context}

Câu hỏi: {question}"""

# v2: một đoạn văn xuôi thay cho danh sách quy tắc (danh sách đánh số khiến mô
# hình trả lời bằng danh sách đánh số, và chép lại chính các quy tắc); bỏ yêu
# cầu trích dẫn (v1: 0/21 câu trích dẫn đúng — nguồn được gắn bằng code); câu
# hỏi đặt ở CUỐI, sát chỗ mô hình bắt đầu viết.
INSTRUCTION_V2 = """Đọc các đoạn tin dưới đây rồi trả lời câu hỏi ở cuối bằng 2 đến 3 câu tiếng Việt tự nhiên, chỉ dùng thông tin có trong các đoạn tin. Nếu các đoạn tin không có câu trả lời, chỉ viết: "{refusal}" Nếu hai tin mâu thuẫn nhau, dùng tin có ngày đăng mới hơn.

{context}

Câu hỏi: {question}"""


# ---------------------------------------------------------------------------
# Ghép ngữ cảnh & prompt
# ---------------------------------------------------------------------------
def build_context(results, retriever, question: str, sentences_per_doc: int = 4,
                  max_chars_per_doc: int = 900, version: str = DEFAULT_PROMPT_VERSION
                  ) -> tuple[str, list[dict]]:
    """Ghép các bài truy hồi được thành khối ngữ cảnh đánh số.

    Mỗi bài chỉ lấy vài câu LIÊN QUAN NHẤT tới câu hỏi (dùng lại tầng chọn câu
    của retriever), không nhét cả bài: vừa tiết kiệm độ dài ngữ cảnh, vừa giảm
    khả năng mô hình bị cuốn theo chi tiết không liên quan.
    """
    blocks: list[str] = []
    sources: list[dict] = []
    for i, r in enumerate(results, 1):
        body = retriever.best_sentences(r.doc_id, question, n=sentences_per_doc)
        body = normalize_basic(body)[:max_chars_per_doc]
        if version == "v1":
            blocks.append(f"[{i}] (đăng {r.published_str}, chuyên mục {r.category})\n"
                          f"Tiêu đề: {r.title}\nNội dung: {body}")
        else:
            blocks.append(f"Tin {i} (ngày {r.published_str}): {r.title}\n{body}")
        sources.append({"id": i, "title": r.title, "url": r.url,
                        "published": r.published_str, "text": f"{r.title} {body}"})
    return "\n\n".join(blocks), sources


# Dấu hiệu câu hỏi. Nhóm "cuối câu" chỉ tính khi đứng cuối ("hàng không",
# "không khí" không phải câu hỏi); "sao" đứng một mình bị bỏ vì "sao hỏa",
# "ngôi sao".
_Q_ANYWHERE = ("gì", "nào", "ai", "đâu", "mấy", "bao nhiêu", "bao giờ", "vì sao", "tại sao",
               "thế nào", "ra sao", "làm sao", "phải không", "đúng không", "có phải",
               "gi", "bao nhieu", "bao gio", "vi sao", "tai sao", "the nao", "phai khong",
               "dung khong")
_Q_FINAL = ("không", "chưa", "à", "hả", "nhỉ", "chăng", "sao", "khong", "ko", "k", "chua")


def is_question(text: str) -> bool:
    t = normalize_basic(text).lower().strip()
    if t.endswith("?"):
        return True
    words = re.findall(r"\w+", t)
    if not words:
        return False
    if words[-1] in _Q_FINAL:
        return True
    padded = " " + " ".join(words) + " "
    return any(f" {m} " in padded for m in _Q_ANYWHERE)


def frame_question(text: str) -> str:
    """Câu dạng từ khóa ("Huawei Mate XT2 gập ba") -> câu hỏi thật.

    v1 đưa nguyên cụm từ khóa cho mô hình: nó hoặc lặp lại cụm đó ("Huawei Mate
    XT2 gập ba."), hoặc tự đặt ra 8 câu hỏi mới. Tập đánh giá có nhiều truy vấn
    kiểu từ khóa vì người dùng gõ thật như vậy.
    """
    t = normalize_basic(text).strip().rstrip(".")
    if is_question(t):
        return t if t.endswith("?") else t + "?"
    return f"Các tin trên cho biết gì về {t}?"


def build_prompt(question: str, context: str, version: str = DEFAULT_PROMPT_VERSION) -> str:
    if version == "v1":
        instruction = INSTRUCTION_V1.format(refusal=REFUSAL_PHRASE, context=context,
                                            question=normalize_basic(question))
    else:
        instruction = INSTRUCTION_V2.format(refusal=REFUSAL_PHRASE, context=context,
                                            question=frame_question(question))
    return PHOGPT_TEMPLATE.format(instruction=instruction)


# ---------------------------------------------------------------------------
# Hậu xử lý câu sinh ra
# ---------------------------------------------------------------------------
# Chuỗi dừng: mô hình bắt đầu viết tiếp một lượt hỏi mới.
STOP_SEQUENCES = ("### Câu hỏi", "###", "\nCâu hỏi", "Câu hỏi 1:")

# Dòng là mô hình CHÉP LẠI prompt chứ không phải câu trả lời: đầu mục nguồn
# ("[2] (đăng 06/30/2021, ...)" — kèm ngày BỊA), tiêu đề khối, hoặc lặp quy tắc.
_ECHO_LINE = re.compile(
    r"^\s*(\[\d+\]\s*\(đăng|tin \d+\s*\(ngày|\[\d+\]\s*$|tiêu đề\s*:|nội dung\s*:|"
    r"các bài báo( hiện có)?\s*:|các đoạn tin\s*:|quy tắc\s*:?|trả lời\s*:|"
    r"nếu (câu hỏi|các bài báo|các đoạn tin|không có|có nhiều|thông tin)|"
    r"ghi nguồn|chỉ dùng thông tin|trả lời ngắn gọn)",
    re.I)
_LIST_MARKER = re.compile(r"^\s*(\d+[.)]|[-*•])\s+")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"\w+")

# Từ vựng của phần hướng dẫn. Một dòng gần như toàn từ trong đây ("Trả lời
# ngắn gọn bằng tiếng Việt tự nhiên.") là mô hình lặp lại quy tắc, bắt được cả
# khi nó diễn đạt hơi khác nên không khớp mẫu _ECHO_LINE.
_INSTRUCTION_VOCAB = set(_WORD.findall(
    re.sub(r"\{\w+\}", " ", INSTRUCTION_V1 + " " + INSTRUCTION_V2).lower()))


def _is_instruction_echo(line: str) -> bool:
    low = line.lower()
    if "không đề cập đến điều này" in low:          # câu từ chối chuẩn là câu trả lời hợp lệ
        return False
    words = _WORD.findall(low)
    return len(words) >= 5 and sum(w in _INSTRUCTION_VOCAB for w in words) / len(words) >= 0.8


def clean_generation(text: str, version: str = DEFAULT_PROMPT_VERSION,
                     max_sentences: int = 4) -> str:
    """Làm sạch câu sinh ra.

    v1: chỉ cắt phần mô hình viết tiếp sang lượt hỏi mới (hành vi lần chạy đầu).
    v2: thêm — bỏ dòng chép prompt, bỏ ký hiệu danh sách, lấy đoạn đầu, bỏ câu
        lặp, giữ tối đa `max_sentences` câu.
    """
    for stop in STOP_SEQUENCES[:2] if version == "v1" else STOP_SEQUENCES:
        if stop in text:
            text = text.split(stop, 1)[0]
    if version == "v1":
        return text.strip()

    paragraphs: list[list[str]] = [[]]
    for line in text.strip().splitlines():
        if not line.strip():
            if paragraphs[-1]:
                paragraphs.append([])
            continue
        line = _LIST_MARKER.sub("", line).strip()
        if _ECHO_LINE.match(line) or _is_instruction_echo(line):
            continue
        paragraphs[-1].append(line)
    first = next((p for p in paragraphs if p), [])

    out, seen = [], set()
    for sent in _SENT_SPLIT.split(" ".join(first)):
        key = re.sub(r"\W+", " ", sent.lower()).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(sent.strip())
        if len(out) == max_sentences:
            break
    return " ".join(out).strip()


# ---------------------------------------------------------------------------
# So khớp con số / mã hiệu (dùng cho kiểm tra độ trung thành và chốt chặn)
# ---------------------------------------------------------------------------
_NUMBER = re.compile(r"(?<![\w])(\d+(?:[.,]\d+)*)(?:\s*(nghìn|ngàn|triệu|tỷ|tỉ|k)(?!\w))?", re.I)
_CODE = re.compile(r"\b(?=\w*\d)(?=\w*[^\W\d_])\w+\b")      # có cả chữ và số: IP68, XT2, M20
_CLOCK = re.compile(r"\b(\d{1,2})h(\d{2})?\b")               # "9h40" là giờ, không phải mã hiệu
_CITATION = re.compile(r"\[(\d+)\]")
_MULTIPLIER = {"nghìn": 1e3, "ngàn": 1e3, "k": 1e3, "triệu": 1e6, "tỷ": 1e9, "tỉ": 1e9}


def _norm_number(s: str) -> str:
    # "8.000" và "8000" là một số; "3,5" và "3.5" cũng vậy.
    return s.replace(".", "").replace(",", "")


def _fmt(v: float) -> str:
    return str(int(round(v))) if abs(v - round(v)) < 1e-9 else f"{v:g}"


def _number_forms(num: str, unit: str | None) -> set[str]:
    """Mọi cách viết tương đương của một số: "100 nghìn" ~ "100.000" ~ "100000"."""
    forms = {_norm_number(num)}
    try:
        value = float(num.replace(".", "").replace(",", "."))   # quy ước Việt: , là thập phân
    except ValueError:
        return forms
    forms.add(_fmt(value))
    if unit:
        forms.add(_fmt(value * _MULTIPLIER[unit.lower()]))
    return forms


def _codes(text: str) -> set[str]:
    return {c.lower() for c in _CODE.findall(_CLOCK.sub(r"\1 giờ \2", text))}


def _numbers(text: str) -> list[tuple[str, set[str]]]:
    """[(cách viết gốc, các dạng tương đương)] — bỏ số nằm trong mã hiệu (IP68 -> không lấy 68)."""
    text = _CODE.sub(" ", _CLOCK.sub(r"\1 giờ \2", text))
    return [(m.group(0).strip(), _number_forms(m.group(1), m.group(2)))
            for m in _NUMBER.finditer(text)]


def _number_pool(text: str) -> set[str]:
    return set().union(*(forms for _, forms in _numbers(text))) if text else set()


def unsupported_premises(question: str, source_texts: list[str]) -> list[str]:
    """Con số / mã hiệu trong câu hỏi mà KHÔNG bài nào nhắc tới.

    So với TOÀN BỘ bài báo (không chỉ vài câu đưa vào prompt) để hạn chế báo
    nhầm: chi tiết có ở bất kỳ đâu trong bài thì không coi là giả định sai.
    Chỉ bắt được giả định sai có con số/mã hiệu — "Messi mua với giá bao nhiêu"
    hay "vì sao đóng cửa" thì quy tắc này không thấy (ghi nhận trong docs/07).
    """
    corpus = " ".join(source_texts)
    corpus_codes = _codes(corpus)
    corpus_numbers = _number_pool(corpus)
    missing = [c for c in sorted(_codes(question)) if c not in corpus_codes]
    for raw, forms in _numbers(question):
        if not (forms & corpus_numbers) and raw not in missing:
            missing.append(raw)
    return missing


# ---------------------------------------------------------------------------
# Backend sinh câu
# ---------------------------------------------------------------------------
class LLMBackend:
    name = "base"

    def generate(self, prompt: str, max_new_tokens: int = 160) -> str:  # pragma: no cover
        raise NotImplementedError


class EchoBackend(LLMBackend):
    """Giả lập: trả câu đầu của bài thứ nhất trong ngữ cảnh. Dùng để kiểm thử toàn
    bộ pipeline (prompt, định tuyến, chốt chặn, kiểm tra) mà không cần GPU."""
    name = "echo"

    _V1 = re.compile(r"\[1\].*?Nội dung: (.+?)(?:\n\n\[2\]|\n\nCâu hỏi:)", re.S)
    _V2 = re.compile(r"Tin 1 \(ngày [^)]*\): [^\n]*\n(.+?)(?:\n\nTin 2|\n\nCâu hỏi:)", re.S)

    def generate(self, prompt: str, max_new_tokens: int = 160) -> str:
        m = self._V2.search(prompt) or self._V1.search(prompt)
        if not m:
            return REFUSAL_PHRASE
        return _SENT_SPLIT.split(m.group(1).strip())[0]


class TransformersBackend(LLMBackend):
    """PhoGPT-4B-Chat qua transformers, float16 trên GPU (T4 không hỗ trợ bfloat16).

    Lần chạy đầu trên Colab (transformers 5.x) lỗi khi nạp: mã tùy biến của
    PhoGPT (kiến trúc MPT) khai báo import `triton_pre_mlir`, chỉ dùng cho kiểu
    attention "triton". Vì vậy mặc định notebook dùng LlamaCppBackend; backend
    này giữ lại để thử khi có môi trường transformers 4.x.

    Giải mã THAM LAM (greedy): với RAG ta cần câu trả lời bám sát nguồn và TÁI
    LẬP ĐƯỢC giữa các lần chạy, không cần sáng tạo.
    """
    name = "transformers"

    def __init__(self, model_id: str = "vinai/PhoGPT-4B-Chat", device: str = "cuda"):
        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
        config.init_device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, config=config, torch_dtype=torch.float16, trust_remote_code=True)
        self.model.eval()
        self.device = device
        self._torch = torch

    def generate(self, prompt: str, max_new_tokens: int = 160) -> str:
        enc = self.tokenizer(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(self.device)
        with self._torch.no_grad():
            out = self.model.generate(
                inputs=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                repetition_penalty=1.1,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            )
        new_tokens = out[0][input_ids.shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)


class LlamaCppBackend(LLMBackend):
    """PhoGPT-4B-Chat bản GGUF qua llama.cpp.

    Mặc định Q8_0 (~3.9 GB, gần như không mất chất lượng so với float16) thay
    cho Q4_K_M của lần chạy đầu — T4 có 15 GB VRAM, không cần nén tới 4 bit.
    Repo vinai/PhoGPT-4B-Chat-gguf có: Q4_K_M, Q8_0, và bản đầy đủ (f16).
    """
    name = "llama.cpp"

    def __init__(self, repo_id: str = "vinai/PhoGPT-4B-Chat-gguf",
                 filename: str = "PhoGPT-4B-Chat-Q8_0.gguf", n_ctx: int = 4096,
                 n_gpu_layers: int = -1):
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama

        path = hf_hub_download(repo_id, filename)
        self.filename = filename
        self.llm = Llama(model_path=path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False)

    def generate(self, prompt: str, max_new_tokens: int = 160) -> str:
        out = self.llm(prompt, max_tokens=max_new_tokens, temperature=0.0,
                       repeat_penalty=1.1, stop=list(STOP_SEQUENCES[:2]))
        return out["choices"][0]["text"]


# ---------------------------------------------------------------------------
# Kiểm tra độ trung thành với nguồn (tự động)
# ---------------------------------------------------------------------------
_REFUSAL_MARKERS = ("không đề cập", "không có thông tin", "không nhắc", "không nêu", "không nói")


def faithfulness_report(answer: str, sources: list[dict], question: str = "") -> dict:
    """Các chỉ số tự động — là TÍN HIỆU CẢNH BÁO, không thay được người đọc.

    - unsupported_numbers: số trong câu trả lời KHÔNG có trong nguồn, câu hỏi,
      hay ngày đăng của nguồn. Số trong câu hỏi được loại trừ vì câu trả lời có
      thể nhắc lại để bác bỏ giả định sai. Số thứ tự đầu dòng ("1. ...") không
      tính — lần chạy v1 bị đếm nhầm 7/12 câu vì lỗi này.
    - support_ratio: tỷ lệ từ nội dung của câu trả lời có mặt trong nguồn.
    - citations_valid / citations_invalid: trích dẫn [i] có trỏ tới bài có thật.
    - refused: câu trả lời là lời từ chối (câu chuẩn, hoặc câu NGẮN có ý "không
      đề cập"). Câu dài liệt kê nội dung rồi thêm "không có thông tin về X" ở
      cuối không tính là từ chối.
    """
    answer_wo_cite = _CITATION.sub(" ", answer)
    answer_wo_list = "\n".join(_LIST_MARKER.sub("", ln) for ln in answer_wo_cite.splitlines())
    context = " ".join(s["text"] for s in sources)
    dates = " ".join(s.get("published", "") for s in sources)

    allowed = _number_pool(context) | _number_pool(question) | _number_pool(dates)
    unsupported = sorted({raw for raw, forms in _numbers(answer_wo_list) if not (forms & allowed)})

    stop = load_stopwords()
    ctx_tokens = set(tokenize(context, CONFIG_RETRIEVAL))
    ans_tokens = [t for t in tokenize(answer_wo_list, CONFIG_RETRIEVAL) if t not in stop]
    support = (sum(1 for t in ans_tokens if t in ctx_tokens) / len(ans_tokens)
               if ans_tokens else 1.0)

    cited = [int(c) for c in _CITATION.findall(answer)]
    valid_ids = {s["id"] for s in sources}
    low = answer.lower()
    refused = ("không đề cập đến điều này" in low
               or (len(low.split()) <= 30 and any(m in low for m in _REFUSAL_MARKERS)))

    return {
        "unsupported_numbers": unsupported,
        "support_ratio": round(support, 3),
        "citations_valid": sum(1 for c in cited if c in valid_ids),
        "citations_invalid": sum(1 for c in cited if c not in valid_ids),
        "refused": refused,
    }


# ---------------------------------------------------------------------------
# Chatbot có RAG
# ---------------------------------------------------------------------------
@dataclass
class RagReply:
    text: str                       # câu HIỂN THỊ cho người dùng
    route: str                      # "rag" | "rag_guard" | route gốc của chatbot
    extractive_text: str            # câu trả lời của bot trích xuất, để so sánh
    sources: list[dict] = field(default_factory=list)
    prompt: str = ""
    latency_s: float = 0.0
    checks: dict = field(default_factory=dict)   # chấm trên llm_text, không phải text
    llm_text: str = ""              # câu PhoGPT sinh (đã làm sạch), kể cả khi bị chặn
    raw_generation: str = ""        # nguyên văn PhoGPT sinh, chưa làm sạch
    guard: str | None = None        # None | "premise" | "numbers" | "empty"
    guard_detail: list[str] = field(default_factory=list)


class RagChatbot:
    """Bọc NewsChatbot: dùng nguyên pipeline cũ, chỉ thay bước cuối bằng sinh câu.

    guards=True bật hai chốt chặn (mặc định: bật với v2, tắt với v1 để v1 tái
    hiện đúng lần chạy đầu). generate_when_guarded=True: khi chốt chặn giả định
    đã chặn, VẪN gọi LLM để ghi lại nó sẽ trả lời gì — chỉ dùng khi đánh giá,
    để đo được chốt chặn đã ngăn được gì.
    """

    def __init__(self, bot, backend: LLMBackend, top_k: int = 3, max_new_tokens: int = 160,
                 prompt_version: str = DEFAULT_PROMPT_VERSION, guards: bool | None = None,
                 generate_when_guarded: bool = False):
        if prompt_version not in PROMPT_VERSIONS:
            raise ValueError(f"prompt_version phải là một trong {PROMPT_VERSIONS}")
        self.bot = bot
        self.backend = backend
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.prompt_version = prompt_version
        self.guards = (prompt_version == "v2") if guards is None else guards
        self.generate_when_guarded = generate_when_guarded

    def _full_texts(self, results) -> list[str]:
        df = self.bot.retriever.df
        return [" ".join(str(df.iloc[r.doc_id].get(c, "")) for c in ("title", "description", "text"))
                for r in results]

    def respond(self, question: str) -> RagReply:
        reply = self.bot.respond(question)

        # Chỉ sinh câu khi phần truy hồi đã có bằng chứng. Chào hỏi, thống kê,
        # từ chối... giữ nguyên câu trả lời của bot gốc, KHÔNG gọi LLM.
        if reply.route != "retrieval" or not reply.results:
            return RagReply(text=reply.text, route=reply.route, extractive_text=reply.text)

        v = self.prompt_version
        results = reply.results[: self.top_k]
        understood = reply.normalized_input or question
        context, sources = build_context(results, self.bot.retriever, understood, version=v)
        prompt = build_prompt(question if v == "v1" else understood, context, version=v)

        premises = unsupported_premises(understood, self._full_texts(results)) if self.guards else []

        raw, latency = "", 0.0
        if not premises or self.generate_when_guarded:
            t0 = time.perf_counter()
            raw = self.backend.generate(prompt, max_new_tokens=self.max_new_tokens)
            latency = time.perf_counter() - t0
        llm_text = clean_generation(raw, version=v)
        checks = faithfulness_report(llm_text, sources, understood) if llm_text else {}

        guard, detail, text = None, [], llm_text
        if premises:
            guard, detail = "premise", premises
            listed = ", ".join(f"«{p}»" for p in premises)
            text = (f"Các bài báo liên quan không nhắc tới {listed}, nên mình không xác nhận "
                    f"được chi tiết này. Thông tin gần nhất tìm được:\n\n{reply.text}")
        elif self.guards and not llm_text:
            guard, text = "empty", reply.text
        elif self.guards and checks.get("unsupported_numbers"):
            guard, detail = "numbers", checks["unsupported_numbers"]
            text = ("(Câu diễn đạt lại có con số không kiểm chứng được trong nguồn — "
                    f"hiển thị trích dẫn gốc.)\n\n{reply.text}")
        elif not llm_text:
            text = REFUSAL_PHRASE

        return RagReply(
            text=text, route="rag" if guard is None else "rag_guard",
            extractive_text=reply.text, sources=sources, prompt=prompt, latency_s=latency,
            checks=checks, llm_text=llm_text, raw_generation=raw, guard=guard,
            guard_detail=detail,
        )
