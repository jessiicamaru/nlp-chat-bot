"""
rag.py — Retrieval-Augmented Generation: truy hồi (tự cài đặt) + sinh câu (PhoGPT).

## Vì sao có module này

Chatbot gốc TRÍCH XUẤT 100%: nó trả về câu có sẵn trong bài báo. Thí nghiệm
n-gram (docs/04) đã chứng minh tự sinh văn bản bằng mô hình nhỏ tự cài đặt thì
bịa sự kiện. Muốn câu trả lời TỰ NHIÊN mà vẫn ĐÚNG SỰ THẬT, cách chuẩn là RAG:

    câu hỏi ──> [truy hồi TF-IDF tự cài đặt] ──> 2-3 bài báo liên quan
                                                    │
                          PhoGPT diễn đạt lại  <────┘  (chỉ được dùng các bài này)

Module này là một LỚP THÊM, không thay thế gì: tắt nó đi thì chatbot vẫn chạy
nguyên như cũ. Đây là phần DUY NHẤT của đồ án dùng mô hình tiền huấn luyện, và
nằm ngoài phạm vi "from scratch" — ghi rõ trong báo cáo.

## Nguyên tắc an toàn quan trọng nhất

LLM CHỈ được gọi khi phần truy hồi đã tìm được bằng chứng vượt ngưỡng. Câu ngoài
phạm vi bị từ chối TRƯỚC khi tới LLM, nên LLM không có cơ hội bịa ra câu trả
lời cho câu hỏi mà kho dữ liệu không có.

## Thành phần

  - build_context / build_prompt : ghép bài báo + quy tắc thành prompt
  - LLMBackend                   : giao diện chung; hai cài đặt cho Colab
      TransformersBackend        : vinai/PhoGPT-4B-Chat, float16 trên GPU
      LlamaCppBackend            : vinai/PhoGPT-4B-Chat-gguf (Q4_K_M), dự phòng
      EchoBackend                : giả lập, để kiểm thử không cần GPU
  - RagChatbot                   : bọc NewsChatbot, thêm bước sinh câu
  - faithfulness_report          : kiểm tra tự động độ trung thành với nguồn
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from config import CONFIG_RETRIEVAL
from preprocess import load_stopwords, normalize_basic, tokenize

# Mẫu prompt chính thức của PhoGPT-4B-Chat (github.com/VinAIResearch/PhoGPT).
PHOGPT_TEMPLATE = "### Câu hỏi: {instruction}\n### Trả lời:"

REFUSAL_PHRASE = "Các bài báo hiện có không đề cập đến điều này."

INSTRUCTION = """Bạn là trợ lý tin tức. Hãy trả lời câu hỏi CHỈ dựa trên các bài báo được cung cấp dưới đây.

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


# ---------------------------------------------------------------------------
# Ghép ngữ cảnh & prompt
# ---------------------------------------------------------------------------
def build_context(results, retriever, question: str, sentences_per_doc: int = 4,
                  max_chars_per_doc: int = 900) -> tuple[str, list[dict]]:
    """Ghép các bài truy hồi được thành khối ngữ cảnh đánh số [1], [2], ...

    Mỗi bài chỉ lấy vài câu LIÊN QUAN NHẤT tới câu hỏi (dùng lại tầng chọn câu
    của retriever), không nhét cả bài: vừa tiết kiệm độ dài ngữ cảnh, vừa giảm
    khả năng mô hình bị cuốn theo chi tiết không liên quan.
    """
    blocks: list[str] = []
    sources: list[dict] = []
    for i, r in enumerate(results, 1):
        body = retriever.best_sentences(r.doc_id, question, n=sentences_per_doc)
        body = normalize_basic(body)[:max_chars_per_doc]
        blocks.append(f"[{i}] (đăng {r.published_str}, chuyên mục {r.category})\n"
                      f"Tiêu đề: {r.title}\nNội dung: {body}")
        sources.append({"id": i, "title": r.title, "url": r.url,
                        "published": r.published_str, "text": f"{r.title} {body}"})
    return "\n\n".join(blocks), sources


def build_prompt(question: str, context: str) -> str:
    instruction = INSTRUCTION.format(refusal=REFUSAL_PHRASE, context=context,
                                     question=normalize_basic(question))
    return PHOGPT_TEMPLATE.format(instruction=instruction)


def clean_generation(text: str) -> str:
    """Cắt phần mô hình tự viết tiếp sang một câu hỏi mới, bỏ khoảng trắng thừa."""
    for stop in ("### Câu hỏi", "###"):
        if stop in text:
            text = text.split(stop, 1)[0]
    return text.strip()


# ---------------------------------------------------------------------------
# Backend sinh câu
# ---------------------------------------------------------------------------
class LLMBackend:
    name = "base"

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:  # pragma: no cover
        raise NotImplementedError


class EchoBackend(LLMBackend):
    """Giả lập: trả câu đầu của bài [1] kèm trích dẫn. Dùng để kiểm thử toàn bộ
    pipeline (prompt, định tuyến, kiểm tra độ trung thành) mà không cần GPU."""
    name = "echo"

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        m = re.search(r"\[1\].*?Nội dung: (.+?)(?:\n\n\[2\]|\n\nCâu hỏi:)", prompt, flags=re.S)
        if not m:
            return REFUSAL_PHRASE
        first = re.split(r"(?<=[.!?])\s+", m.group(1).strip())[0]
        return f"{first} [1]"


class TransformersBackend(LLMBackend):
    """PhoGPT-4B-Chat qua transformers, float16 trên GPU (T4 không hỗ trợ bfloat16).

    Giải mã THAM LAM (greedy, do_sample=False) thay vì lấy mẫu như khuyến nghị
    mặc định của PhoGPT: với RAG ta cần câu trả lời bám sát nguồn và TÁI LẬP
    ĐƯỢC giữa các lần chạy, không cần sáng tạo.
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

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
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
        return clean_generation(self.tokenizer.decode(new_tokens, skip_special_tokens=True))


class LlamaCppBackend(LLMBackend):
    """PhoGPT-4B-Chat bản GGUF lượng tử hóa (Q4_K_M, 2.36 GB) qua llama.cpp.

    Dự phòng khi TransformersBackend lỗi (mã tùy biến của PhoGPT có thể không
    tương thích với phiên bản transformers mới trên Colab).
    """
    name = "llama.cpp"

    def __init__(self, repo_id: str = "vinai/PhoGPT-4B-Chat-gguf",
                 filename: str = "PhoGPT-4B-Chat-Q4_K_M.gguf", n_ctx: int = 4096,
                 n_gpu_layers: int = -1):
        from huggingface_hub import hf_hub_download
        from llama_cpp import Llama

        path = hf_hub_download(repo_id, filename)
        self.llm = Llama(model_path=path, n_ctx=n_ctx, n_gpu_layers=n_gpu_layers, verbose=False)

    def generate(self, prompt: str, max_new_tokens: int = 256) -> str:
        out = self.llm(prompt, max_tokens=max_new_tokens, temperature=0.0,
                       repeat_penalty=1.1, stop=["### Câu hỏi"])
        return clean_generation(out["choices"][0]["text"])


# ---------------------------------------------------------------------------
# Kiểm tra độ trung thành với nguồn (tự động)
# ---------------------------------------------------------------------------
_NUMBER = re.compile(r"\d+(?:[.,]\d+)*")
_CITATION = re.compile(r"\[(\d+)\]")


def _norm_number(s: str) -> str:
    # "8.000" và "8000" là một số; "3,5" và "3.5" cũng vậy.
    return s.replace(".", "").replace(",", "")


def faithfulness_report(answer: str, sources: list[dict], question: str = "") -> dict:
    """Các chỉ số tự động — là TÍN HIỆU CẢNH BÁO, không thay được người đọc.

    - unsupported_numbers: số xuất hiện trong câu trả lời nhưng KHÔNG có trong
      nguồn lẫn câu hỏi. Tin tức đầy con số, và bịa số là lỗi ảo giác phổ biến
      nhất; số trong câu hỏi được loại trừ vì câu trả lời có thể nhắc lại để
      bác bỏ một giả định sai.
    - support_ratio: tỷ lệ từ nội dung của câu trả lời có mặt trong nguồn.
      Thấp nghĩa là mô hình đưa nhiều từ "ngoài nguồn" vào.
    - citations_valid / citations_invalid: trích dẫn [i] có trỏ tới bài có thật.
    - refused: mô hình dùng câu từ chối chuẩn.
    """
    answer_wo_cite = _CITATION.sub(" ", answer)
    context = " ".join(s["text"] for s in sources)

    ctx_numbers = {_norm_number(n) for n in _NUMBER.findall(context)}
    q_numbers = {_norm_number(n) for n in _NUMBER.findall(question)}
    ans_numbers = [_norm_number(n) for n in _NUMBER.findall(answer_wo_cite)]
    unsupported = sorted({n for n in ans_numbers if n not in ctx_numbers | q_numbers})

    stop = load_stopwords()
    ctx_tokens = set(tokenize(context, CONFIG_RETRIEVAL))
    ans_tokens = [t for t in tokenize(answer_wo_cite, CONFIG_RETRIEVAL) if t not in stop]
    support = (sum(1 for t in ans_tokens if t in ctx_tokens) / len(ans_tokens)
               if ans_tokens else 1.0)

    cited = [int(c) for c in _CITATION.findall(answer)]
    valid_ids = {s["id"] for s in sources}
    refused = "không đề cập" in answer.lower() or "không có thông tin" in answer.lower()

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
    text: str
    route: str                      # "rag" | route gốc của chatbot (intent/fallback)
    extractive_text: str            # câu trả lời của bot trích xuất, để so sánh
    sources: list[dict] = field(default_factory=list)
    prompt: str = ""
    latency_s: float = 0.0
    checks: dict = field(default_factory=dict)


class RagChatbot:
    """Bọc NewsChatbot: dùng nguyên pipeline cũ, chỉ thay bước cuối bằng sinh câu."""

    def __init__(self, bot, backend: LLMBackend, top_k: int = 3, max_new_tokens: int = 256):
        self.bot = bot
        self.backend = backend
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens

    def respond(self, question: str) -> RagReply:
        reply = self.bot.respond(question)

        # Chỉ sinh câu khi phần truy hồi đã có bằng chứng. Chào hỏi, thống kê,
        # từ chối... giữ nguyên câu trả lời của bot gốc, KHÔNG gọi LLM.
        if reply.route != "retrieval" or not reply.results:
            return RagReply(text=reply.text, route=reply.route, extractive_text=reply.text)

        question_for_ctx = reply.normalized_input or question
        context, sources = build_context(reply.results[: self.top_k], self.bot.retriever,
                                         question_for_ctx)
        prompt = build_prompt(question, context)

        t0 = time.perf_counter()
        answer = self.backend.generate(prompt, max_new_tokens=self.max_new_tokens)
        latency = time.perf_counter() - t0

        if not answer:
            answer = REFUSAL_PHRASE
        return RagReply(
            text=answer, route="rag", extractive_text=reply.text, sources=sources,
            prompt=prompt, latency_s=latency,
            checks=faithfulness_report(answer, sources, question),
        )
