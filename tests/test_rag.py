"""
test_rag.py — Kiểm thử phần RAG KHÔNG cần GPU (dùng backend giả lập).

Mục đích: khi chạy thật trên Colab, mọi thứ trừ bản thân PhoGPT đều đã được
kiểm chứng — định tuyến, ghép prompt, kiểm tra độ trung thành. Nếu có lỗi trên
Colab thì chỉ còn có thể nằm ở phần tải/chạy mô hình.

Chạy:  .venv/Scripts/python.exe tests/test_rag.py
"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chatbot import NewsChatbot
from rag import (PHOGPT_TEMPLATE, REFUSAL_PHRASE, EchoBackend, LLMBackend, RagChatbot,
                 build_prompt, clean_generation, faithfulness_report)


class MustNotBeCalled(LLMBackend):
    name = "guard"

    def generate(self, prompt, max_new_tokens=256):
        raise AssertionError("LLM bị gọi dù không có bằng chứng truy hồi!")


_BOT = NewsChatbot().train()
TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


@test
def cau_ngoai_pham_vi_khong_goi_llm():
    """Nguyên tắc an toàn: không có bằng chứng thì LLM không được chạy."""
    rag = RagChatbot(_BOT, MustNotBeCalled())
    for q in ["asdfgh qwerty zxcvb", "thời tiết sao hỏa hôm nay thế nào", "xin chào"]:
        _BOT.reset()
        r = rag.respond(q)
        assert r.route != "rag", (q, r.route)


@test
def cau_tin_tuc_di_qua_llm_voi_nguon():
    rag = RagChatbot(_BOT, EchoBackend())
    _BOT.reset()
    r = rag.respond("tin về đảo hải nam")
    assert r.route == "rag", r.route
    assert r.sources and "Hải Nam" in r.sources[0]["title"]
    assert r.prompt.startswith("### Câu hỏi:") and r.prompt.endswith("### Trả lời:")
    assert REFUSAL_PHRASE in r.prompt
    assert "(đăng " in r.prompt, "prompt phải kèm ngày đăng để mô hình ưu tiên tin mới"
    assert r.checks["citations_valid"] >= 1 and r.checks["citations_invalid"] == 0


@test
def cau_teencode_van_qua_rag():
    rag = RagChatbot(_BOT, EchoBackend())
    _BOT.reset()
    r = rag.respond("cho t hỏi vụ hải nam vs")
    assert r.route == "rag" and "Hải Nam" in r.sources[0]["title"]


@test
def prompt_dung_mau_phogpt():
    p = build_prompt("câu hỏi?", "[1] ngữ cảnh")
    assert p.startswith("### Câu hỏi:") and p.endswith("\n### Trả lời:")
    assert PHOGPT_TEMPLATE.count("{instruction}") == 1


@test
def cat_phan_mo_hinh_viet_lan_sang():
    assert clean_generation("Giá vé là 8.000 đồng [1].\n### Câu hỏi: tiếp") == "Giá vé là 8.000 đồng [1]."


@test
def phat_hien_so_bia():
    src = [{"id": 1, "text": "Giá vé giữ nguyên 8.000 đồng mỗi lượt."}]
    bad = faithfulness_report("Giá vé là 20.000 đồng [1].", src)
    assert bad["unsupported_numbers"] == ["20000"], bad
    good = faithfulness_report("Giá vé là 8000 đồng [1].", src)
    assert good["unsupported_numbers"] == [], good


@test
def so_trong_cau_hoi_khong_tinh_la_bia():
    """Câu trả lời được nhắc lại số trong câu hỏi để bác bỏ giả định sai."""
    src = [{"id": 1, "text": "Không nên giữ cơm 5-7 ngày trong ngăn mát."}]
    r = faithfulness_report("Không đúng, bài báo khuyên không giữ quá 5-7 ngày, không phải 30 ngày [1].",
                            src, question="cơm nguội để được 30 ngày đúng không")
    assert r["unsupported_numbers"] == [], r


@test
def trich_dan_khong_ton_tai():
    src = [{"id": 1, "text": "abc"}, {"id": 2, "text": "def"}]
    r = faithfulness_report("abc [1] def [5]", src)
    assert r["citations_valid"] == 1 and r["citations_invalid"] == 1, r


@test
def nhan_dien_tu_choi():
    r = faithfulness_report(REFUSAL_PHRASE, [{"id": 1, "text": "x"}])
    assert r["refused"] is True


def main() -> int:
    passed = failed = 0
    print("=" * 70)
    print("KIỂM THỬ RAG (backend giả lập, không cần GPU)")
    print("=" * 70)
    for fn in TESTS:
        try:
            fn()
            print(f"  [PASS] {fn.__name__}")
            passed += 1
        except Exception as exc:
            print(f"  [FAIL] {fn.__name__}: {exc!r}")
            traceback.print_exc(limit=1)
            failed += 1
    print("=" * 70)
    print(f"KẾT QUẢ: {passed} pass / {failed} fail")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
