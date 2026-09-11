"""
test_chatbot.py — Kiểm thử hồi quy cho các lỗi THẬT đã gặp trong quá trình làm.

Mỗi test tương ứng một lỗi từng làm bot trả lời sai mà KHÔNG báo lỗi gì — loại
lỗi nguy hiểm nhất, vì chỉ lộ ra khi có người đọc kỹ câu trả lời. Test ở đây để
khi dò lại tham số hay sửa code sau này, các lỗi đó không âm thầm quay lại.

Chạy:  .venv/Scripts/python.exe tests/test_chatbot.py
"""

import json
import sys
import traceback
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "src"))

import pandas as pd

import chatbot as chatbot_mod
import config
import preprocess
from chatbot import NewsChatbot
from dates import compute_recency, parse_vn_date
from entities import extract_regex
from normalizer import TeencodeNormalizer, prepare_user_text
from preprocess import fold_query, has_diacritics, strip_frame_words, tokenize
from retriever import NewsRetriever

CORPUS = pd.read_csv(config.CORPUS_RAW_PATH).dropna(subset=["title", "text"]).reset_index(drop=True)
URL_HAI_NAM = CORPUS.loc[CORPUS["title"].str.contains("Hải Nam 'chưa dễ hút"), "url"].iloc[0]

_BOT = None


def bot() -> NewsChatbot:
    global _BOT
    if _BOT is None:
        _BOT = NewsChatbot().train()
    _BOT.reset()
    return _BOT


TESTS = []


def test(fn):
    TESTS.append(fn)
    return fn


# ---------------------------------------------------------------------------
# Tiền xử lý & regex
# ---------------------------------------------------------------------------
@test
def regex_bat_duoc_phan_tram():
    """Lỗi: pattern kết thúc bằng \\b nên '3,5%' bị bỏ sót ('%' là ký tự non-word)."""
    found = extract_regex("Doanh thu tăng 3,5% so với cùng kỳ")
    assert "3,5%" in found.get("quantity", []), found


@test
def nhan_dien_cau_khong_dau():
    assert has_diacritics("tin về đảo") is True
    assert has_diacritics("tin ve dao") is False


@test
def cau_khong_dau_khong_qua_word_tokenize():
    """Lỗi: word_tokenize tách sai câu không dấu thành ['ve_dao','hai','nam']."""
    assert fold_query("tin ve dao hai nam") == ["tin", "ve", "dao", "hai", "nam"]


@test
def loc_tu_khung_khong_lam_rong_query():
    """Câu toàn từ khung thì phải giữ nguyên, không được trả về vector rỗng."""
    toks = ["có", "tin", "gì", "mới"]
    assert strip_frame_words(toks) == toks
    assert strip_frame_words(["biết", "vụ", "iphone", "bạn"]) == ["iphone"]


@test
def tu_khung_chi_co_mot_nguon_duy_nhat():
    """Lỗi: chatbot.py và retriever.py giữ hai danh sách riêng và đã lệch nhau."""
    assert chatbot_mod.QUERY_FRAME_WORDS is config.QUERY_FRAME_WORDS
    assert preprocess.QUERY_FRAME_WORDS is config.QUERY_FRAME_WORDS


# ---------------------------------------------------------------------------
# Chuẩn hóa teencode
# ---------------------------------------------------------------------------
@test
def teencode_duoc_chuan_hoa():
    text, pairs = prepare_user_text("bt gì về vụ iphone k b", TeencodeNormalizer())
    assert text == "biết gì về vụ iphone không bạn", text
    assert ("k", "không") in pairs


@test
def chuan_hoa_giu_he_quy_chieu_khong_dau():
    """Lỗi: 'thoi'->'thôi' làm cả câu không dấu bị định tuyến sang index có dấu."""
    text, _ = prepare_user_text("thoi tiet sao hoa hom nay", TeencodeNormalizer())
    assert not has_diacritics(text), text


@test
def hoi_thoi_tiet_khong_bi_chao_tam_biet():
    """Lỗi: bot trả lời 'Tạm biệt bạn!' cho câu hỏi thời tiết không dấu."""
    r = bot().respond("thoi tiet sao hoa hom nay")
    assert r.intent != "tam_biet" or r.route != "intent", (r.route, r.intent)
    assert r.route == "fallback", (r.route, r.intent, r.text[:80])


# ---------------------------------------------------------------------------
# Định tuyến & chuyên mục
# ---------------------------------------------------------------------------
@test
def so_khop_ten_chuyen_muc_theo_am_tiet():
    """Lỗi: word_tokenize('Sức khỏe') -> ['sức','khỏe'] nhưng trong câu lại ra
    ['sức_khỏe'] -> câu chỉ nêu tên chuyên mục bị coi là có chủ đề riêng."""
    f = NewsChatbot._has_topic_beyond_category
    assert f("không biết tin sức khỏe gì luôn", "Sức khỏe") is False
    assert f("tin sức khỏe về ăn chuối", "Sức khỏe") is True


@test
def cau_chi_neu_chuyen_muc_thi_duyet_muc():
    r = bot().respond("ko bt tin sức khỏe gì lun")
    assert r.route == "intent" and r.results, (r.route, r.text[:80])
    assert all(x.category == "Sức khỏe" for x in r.results)


@test
def chuyen_muc_kem_chu_de_thi_tim_trong_muc():
    r = bot().respond("tin du lịch ninh bình")
    assert r.results and "Ninh Bình" in r.results[0].title, r.text[:100]


# ---------------------------------------------------------------------------
# Truy hồi
# ---------------------------------------------------------------------------
@test
def truy_hoi_cau_khong_dau():
    r = bot().respond("tin ve dao hai nam")
    assert r.results and r.results[0].url == URL_HAI_NAM, r.text[:100]


@test
def truy_hoi_cau_teencode():
    r = bot().respond("cho t hỏi vụ hải nam vs")
    assert r.results and r.results[0].url == URL_HAI_NAM, r.text[:100]


@test
def cau_ngoai_pham_vi_bi_tu_choi():
    for q in ["asdfgh qwerty zxcvb", "thời tiết sao hỏa hôm nay thế nào"]:
        r = bot().respond(q)
        assert r.route == "fallback", (q, r.route, r.intent)


@test
def snippet_khong_lap_cau():
    """Lỗi: bài VnExpress lặp câu sapo trong thân bài -> snippet in một câu hai lần."""
    b = bot()
    for q in ["cho tôi biết về giá iphone", "tin về đảo hải nam", "robot chó dẫn đường"]:
        r = b.respond(q)
        if not r.results:
            continue
        sents = b.retriever._sentences(r.results[0].doc_id)
        assert len(sents) == len({s.lower() for s in sents}), q


@test
def tham_chieu_bai_do_qua_nhieu_luot():
    b = bot()
    first = b.respond("tin về đảo hải nam")
    summary = b.respond("tóm tắt bài đó")
    link = b.respond("cho mình link")
    assert first.results and summary.results and link.results
    assert summary.results[0].url == first.results[0].url == link.results[0].url


@test
def dau_vao_rong_khong_lam_sap_bot():
    r = bot().respond("   ")
    assert r.text


# ---------------------------------------------------------------------------
# Độ mới & cache
# ---------------------------------------------------------------------------
@test
def doc_duoc_ngay_vnexpress():
    d = parse_vn_date("Thứ ba, 25/8/2026, 10:59 (GMT+7)")
    assert (d.year, d.month, d.day) == (2026, 8, 25)
    assert parse_vn_date("31/2/2026") is None


@test
def duyet_muc_sap_theo_ngay_moi_nhat():
    """Lỗi: browse() dùng thứ tự chèn vào corpus, 'tin mới nhất' lại ra bài cũ."""
    res = bot().retriever.browse("Du lịch", n=10)
    dates = [x.published_date for x in res if x.published_date]
    assert dates == sorted(dates, reverse=True)


@test
def tin_moi_phu_dinh_tin_cu():
    """Lỗi: TF-IDF thuần trả bài CŨ đã sai (0.5269 vs 0.4302) cho tin mâu thuẫn."""
    case = json.loads((config.DATA_DIR / "eval" / "conflict_case.json").read_text(encoding="utf-8"))
    df = pd.concat([CORPUS, pd.DataFrame(case["articles"])], ignore_index=True)
    r = NewsRetriever().fit(df)
    for q in case["queries"]:
        res = [x for x in r.search(q, top_k=10, min_score=0.0) if "example.test" in x.url]
        assert res and res[0].url == case["expected_url"], (q, [x.url for x in res])


@test
def cache_tinh_lai_do_moi_theo_nua_chu_ky():
    """Lỗi: cache lưu sẵn recency; đổi nửa chu kỳ bị bỏ qua âm thầm."""
    r = NewsRetriever(freshness_halflife=1.0).fit_cached(CORPUS)
    expected = compute_recency(r.published_dates, half_life_days=1.0)
    assert (abs(r.recency - expected) < 1e-12).all()
    r2 = NewsRetriever(freshness_halflife=30.0).fit_cached(CORPUS)
    assert not (abs(r2.recency - r.recency) < 1e-12).all()


# ---------------------------------------------------------------------------
def main() -> int:
    passed, failed = 0, 0
    print("=" * 70)
    print("KIỂM THỬ HỒI QUY CHATBOT")
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
