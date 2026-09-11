"""
test_rag.py — Kiểm thử phần RAG KHÔNG cần GPU (dùng backend giả lập).

Mục đích: khi chạy thật trên Colab, mọi thứ trừ bản thân PhoGPT đều đã được
kiểm chứng — định tuyến, ghép prompt, làm sạch câu sinh, chốt chặn, kiểm tra độ
trung thành. Nếu có lỗi trên Colab thì chỉ còn có thể nằm ở phần tải/chạy mô hình.

Các chuỗi "đầu ra PhoGPT" dùng trong test được CHÉP NGUYÊN từ lần chạy v1 thật
trên Colab (local/rag_results.json) — test giữ cho các lỗi đó không quay lại.

Chạy:  .venv/Scripts/python.exe tests/test_rag.py
"""

import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from chatbot import NewsChatbot
from normalizer import TeencodeNormalizer, prepare_user_text
from rag import (PHOGPT_TEMPLATE, REFUSAL_PHRASE, EchoBackend, LLMBackend, RagChatbot,
                 build_prompt, clean_generation, faithfulness_report, frame_question,
                 is_question, unsupported_premises)


class MustNotBeCalled(LLMBackend):
    name = "guard"

    def generate(self, prompt, max_new_tokens=160):
        raise AssertionError("LLM bị gọi dù không được phép!")


class Fixed(LLMBackend):
    """Backend trả một câu cố định — giả lập một câu trả lời có lỗi cụ thể."""
    name = "fixed"

    def __init__(self, text):
        self.text = text

    def generate(self, prompt, max_new_tokens=160):
        return self.text


_BOT = NewsChatbot().train()
TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


def ask(rag, q):
    _BOT.reset()
    return rag.respond(q)


# --- Định tuyến ------------------------------------------------------------
@test
def cau_ngoai_pham_vi_khong_goi_llm():
    """Nguyên tắc an toàn: không có bằng chứng thì LLM không được chạy."""
    for version in ("v1", "v2"):
        rag = RagChatbot(_BOT, MustNotBeCalled(), prompt_version=version)
        for q in ["asdfgh qwerty zxcvb", "thời tiết sao hỏa hôm nay thế nào", "xin chào"]:
            r = ask(rag, q)
            assert not r.route.startswith("rag"), (version, q, r.route)


@test
def cau_tin_tuc_di_qua_llm_voi_nguon():
    r = ask(RagChatbot(_BOT, EchoBackend()), "tin về đảo hải nam")
    assert r.route == "rag", (r.route, r.guard)
    assert r.sources and "Hải Nam" in r.sources[0]["title"]
    assert r.prompt.startswith("### Câu hỏi:") and r.prompt.endswith("### Trả lời:")
    assert REFUSAL_PHRASE in r.prompt
    assert f"(ngày {r.sources[0]['published']})" in r.prompt, \
        "prompt phải kèm ngày đăng để mô hình ưu tiên tin mới"
    assert r.text and r.text == r.llm_text


@test
def prompt_v1_van_tai_hien_duoc():
    """v1 phải giữ nguyên để so sánh A/B với lần chạy đầu."""
    r = ask(RagChatbot(_BOT, EchoBackend(), prompt_version="v1"), "tin về đảo hải nam")
    assert r.route == "rag" and "(đăng " in r.prompt and "Quy tắc:" in r.prompt
    assert r.guard is None


@test
def cau_teencode_van_qua_rag():
    r = ask(RagChatbot(_BOT, EchoBackend()), "cho t hỏi vụ hải nam vs")
    assert r.route == "rag" and "Hải Nam" in r.sources[0]["title"]


@test
def prompt_dung_mau_phogpt():
    for v in ("v1", "v2"):
        p = build_prompt("câu hỏi?", "Tin 1 (ngày 01/01/2026): x\nngữ cảnh", version=v)
        assert p.startswith("### Câu hỏi:") and p.endswith("\n### Trả lời:")
    assert PHOGPT_TEMPLATE.count("{instruction}") == 1


# --- Câu hỏi dạng từ khóa --------------------------------------------------
@test
def nhan_dien_cau_hoi():
    yes = ["giá vé tàu cát linh bao nhiêu", "vé tàu có tăng giá không", "vì sao kem Tràng Tiền đóng cửa",
           "ng việt tốn nhiều tiền cho trà sữa ko", "ai vô địch", "gia ve the nao"]
    no = ["Huawei Mate XT2 gập ba", "hạn hán kênh đào panama", "thời tiết sao hỏa",
          "hàng không giá rẻ", "robot chó dẫn đường trung quốc"]
    for q in yes:
        assert is_question(q), q
    for q in no:
        assert not is_question(q), q


@test
def cum_tu_khoa_duoc_chuyen_thanh_cau_hoi():
    assert frame_question("Huawei Mate XT2 gập ba") == "Các tin trên cho biết gì về Huawei Mate XT2 gập ba?"
    assert frame_question("giá vé bao nhiêu") == "giá vé bao nhiêu?"


# --- Làm sạch câu sinh (đầu vào chép từ lần chạy v1 thật) -----------------
@test
def cat_phan_mo_hinh_viet_lan_sang():
    assert clean_generation("Giá vé là 8.000 đồng.\n### Câu hỏi: tiếp") == "Giá vé là 8.000 đồng."
    assert clean_generation("Giá vé là 8.000 đồng [1].\n### Câu hỏi: tiếp", version="v1") \
        == "Giá vé là 8.000 đồng [1]."


@test
def bo_dau_muc_nguon_bi_chep_lai_kem_ngay_bia():
    raw = ("Vụ nổ tên lửa Blue Origin tạo sóng âm lan khắp nước Mỹ.\n\n"
           "[2]  (đăng 06/30/2021, chuyên mục Khoa học)")
    assert clean_generation(raw) == "Vụ nổ tên lửa Blue Origin tạo sóng âm lan khắp nước Mỹ."
    assert clean_generation("[1] (đăng 10/09/2016, chuyên mục Công nghệ)") == ""


@test
def bo_danh_sach_danh_so_va_cau_lap():
    raw = ("1. Người dân Estonia biến vườn nhà thành quán.\n"
           "2. Truyền thống này phản ánh lịch sử đón khách lạ.\n"
           "3. Truyền thống này phản ánh lịch sử đón khách lạ.\n"
           "4. Câu thứ tư.\n5. Câu thứ năm.\n6. Câu thứ sáu.")
    out = clean_generation(raw)
    assert not out.startswith("1."), out
    assert out.count("Truyền thống này") == 1, out
    assert out.endswith("Câu thứ năm."), out          # tối đa 4 câu


@test
def bo_dong_chep_lai_quy_tac():
    raw = ("1. Quy tắc:\n\nTrả lời ngắn gọn bằng tiếng Việt tự nhiên.\n"
           "Nếu câu hỏi chứa giả định sai so với bài báo, hãy chỉ ra.")
    assert clean_generation(raw) == "", clean_generation(raw)
    # Câu trả lời thật lẫn với dòng lặp quy tắc (v1 hàng 10): giữ câu thật.
    raw3 = ("1. Nhiều người đã về đất liền bằng cáp treo trong ngày.\n"
            "2. Nếu không có đủ thông tin để trả lời, vui lòng trả lời đúng một câu.")
    assert clean_generation(raw3) == "Nhiều người đã về đất liền bằng cáp treo trong ngày.", \
        clean_generation(raw3)
    assert clean_generation(REFUSAL_PHRASE) == REFUSAL_PHRASE
    raw2 = "Vụ đắm tàu xảy ra năm 2012.\n\nCâu hỏi 1: Tại sao thuyền trưởng không quay lại?"
    assert clean_generation(raw2) == "Vụ đắm tàu xảy ra năm 2012."


# --- Kiểm tra độ trung thành ----------------------------------------------
@test
def phat_hien_so_bia():
    src = [{"id": 1, "text": "Giá vé giữ nguyên 8.000 đồng mỗi lượt.", "published": "10/09/2026"}]
    bad = faithfulness_report("Giá vé là 20.000 đồng [1].", src)
    assert bad["unsupported_numbers"] == ["20.000"], bad
    good = faithfulness_report("Giá vé là 8000 đồng [1].", src)
    assert good["unsupported_numbers"] == [], good


@test
def so_thu_tu_dau_dong_va_ngay_dang_khong_tinh_la_bia():
    src = [{"id": 1, "text": "Robot chó chạy trên sa mạc.", "published": "08/09/2026"}]
    r = faithfulness_report("1. Robot chó chạy trên sa mạc.\n2. Bài đăng ngày 08/09/2026.", src)
    assert r["unsupported_numbers"] == [], r
    r = faithfulness_report("Robot ra mắt năm 2016.", src)
    assert r["unsupported_numbers"] == ["2016"], r
    # v1 hàng 14: nguồn viết "9h40", câu trả lời viết "9 giờ 40 phút"
    r = faithfulness_report("Sinh lúc 9 giờ 40 phút.", [{"id": 1, "text": "Sản phụ sinh lúc 9h40."}])
    assert r["unsupported_numbers"] == [], r


@test
def so_trong_cau_hoi_khong_tinh_la_bia():
    """Câu trả lời được nhắc lại số trong câu hỏi để bác bỏ giả định sai."""
    src = [{"id": 1, "text": "Không nên giữ cơm 5-7 ngày trong ngăn mát."}]
    r = faithfulness_report("Không đúng, bài báo khuyên không giữ quá 5-7 ngày, không phải 30 ngày [1].",
                            src, question="cơm nguội để được 30 ngày đúng không")
    assert r["unsupported_numbers"] == [], r


@test
def so_co_don_vi_nghin_trieu():
    src = [{"id": 1, "text": "Hàng nhập khẩu trên 100.000 đồng, tai nghe giá 3,8 triệu đồng."}]
    r = faithfulness_report("Hàng dưới 100 nghìn đồng, tai nghe 3.800.000 đồng.", src)
    assert r["unsupported_numbers"] == [], r


@test
def trich_dan_khong_ton_tai():
    src = [{"id": 1, "text": "abc"}, {"id": 2, "text": "def"}]
    r = faithfulness_report("abc [1] def [5]", src)
    assert r["citations_valid"] == 1 and r["citations_invalid"] == 1, r


@test
def nhan_dien_tu_choi():
    assert faithfulness_report(REFUSAL_PHRASE, [{"id": 1, "text": "x"}])["refused"] is True
    # v1 hàng 4: liệt kê 6 ý rồi thêm câu "không đề cập" -> KHÔNG phải từ chối
    long = ("Người dân Estonia biến vườn nhà thành quán ăn để giới thiệu văn hóa địa phương. " * 3
            + "Lưu ý: Các bài báo không đề cập đến giả định sai.")
    assert faithfulness_report(long, [{"id": 1, "text": "x"}])["refused"] is False


# --- Chốt chặn giả định ---------------------------------------------------
TRAPS_WITH_NUMBER = {
    "tai nghe AirPods 5 có chống nước chuẩn IP68 không": "ip68",
    "lợi nhuận năm 2020 của metro Bến Thành Suối Tiên": "2020",
    "giữ cơm nguội trong tủ lạnh 30 ngày có sao không": "30",
    "đảo Hải Nam miễn visa cho khách Việt từ năm 2015 phải không": "2015",
}


@test
def chot_chan_gia_dinh_bat_bay_co_so_va_khong_goi_llm():
    rag = RagChatbot(_BOT, MustNotBeCalled())
    for q, detail in TRAPS_WITH_NUMBER.items():
        r = ask(rag, q)
        assert r.route == "rag_guard" and r.guard == "premise", (q, r.route, r.guard)
        assert r.guard_detail == [detail], (q, r.guard_detail)
        assert "không nhắc tới" in r.text


@test
def chot_chan_gia_dinh_khong_bao_nham_cau_hop_le():
    rag = RagChatbot(_BOT, EchoBackend())
    for q in ["đảo hải nam miễn visa 30 ngày", "hàng nhập dưới 100 nghìn có được miễn thuế",
              "Huawei Mate XT2 gập ba", "du khách mắc kẹt trên đảo Cát Bà vì bão số 4",
              "nhan vien openai tieu 7000 do tien token moi ngay"]:
        r = ask(rag, q)
        assert r.route == "rag" and r.guard is None, (q, r.route, r.guard, r.guard_detail)


@test
def chot_chan_gia_dinh_ham_thuan():
    assert unsupported_premises("năm 2015 có gì", ["Bài viết về năm 2016"]) == ["2015"]
    assert unsupported_premises("giá 15.000 đồng", ["giá 15000 đồng"]) == []
    assert unsupported_premises("chuẩn IP68", ["chuẩn ip68 chống nước"]) == []


@test
def che_do_danh_gia_van_ghi_lai_cau_llm_khi_bi_chan():
    rag = RagChatbot(_BOT, Fixed("Đúng."), generate_when_guarded=True)
    r = ask(rag, "đảo Hải Nam miễn visa cho khách Việt từ năm 2015 phải không")
    assert r.guard == "premise" and r.llm_text == "Đúng." and "không nhắc tới" in r.text


# --- Chốt chặn số bịa & câu rỗng ------------------------------------------
@test
def chot_chan_so_bia_thay_bang_trich_xuat():
    rag = RagChatbot(_BOT, Fixed("Robot chó dẫn đường ra mắt vào năm 2016."))
    r = ask(rag, "robot chó dẫn đường trung quốc")
    assert r.route == "rag_guard" and r.guard == "numbers" and r.guard_detail == ["2016"], r.guard
    assert r.extractive_text in r.text


@test
def chot_chan_cau_rong_thay_bang_trich_xuat():
    rag = RagChatbot(_BOT, Fixed("[1] (đăng 10/09/2016, chuyên mục Công nghệ)"))
    r = ask(rag, "robot chó dẫn đường trung quốc")
    assert r.guard == "empty" and r.text == r.extractive_text


@test
def v1_khong_co_chot_chan():
    rag = RagChatbot(_BOT, Fixed("Robot chó dẫn đường ra mắt vào năm 2016."), prompt_version="v1")
    r = ask(rag, "robot chó dẫn đường trung quốc")
    assert r.route == "rag" and r.guard is None and r.checks["unsupported_numbers"] == ["2016"]


# --- Hồi quy: lỗi gộp chữ số của bộ chuẩn hóa teencode --------------------
@test
def chuan_hoa_teencode_khong_lam_hong_con_so():
    n = TeencodeNormalizer()
    assert prepare_user_text("tàu cát linh 15.000 đồng", n)[0] == "tàu cát linh 15.000 đồng"
    assert prepare_user_text("năm 2000 có 1000 người", n)[0] == "năm 2000 có 1000 người"
    assert prepare_user_text("đẹppppp quá", n)[0].startswith("đẹp ")


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
