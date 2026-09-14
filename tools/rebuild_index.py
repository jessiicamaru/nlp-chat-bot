"""Dựng lại (hoặc nạp) index truy hồi sau khi corpus thay đổi, rồi kiểm tra nhanh.

Chatbot không có bước "train" riêng: intent classifier học lại từ
data/intents/intents_vi.json mỗi lần khởi động (dưới 1 giây), còn index truy hồi
được cache ra models/retriever_index.joblib kèm vân tay SHA-256 của corpus. Khi
corpus đổi, vân tay đổi, index tự dựng lại. Script này chủ động làm việc đó ngay
sau khi crawl — để lần khởi động bot/web sau không phải chờ — và kiểm tra rằng mô
hình mới vẫn trả lời được.

Mã thoát: 0 = ổn; 1 = lỗi (corpus rỗng, không trả lời được câu kiểm tra).

Chạy:  .venv/Scripts/python.exe tools/rebuild_index.py
"""
import sys
import time
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "src"))

from chatbot import NewsChatbot  # noqa: E402
from config import INDEX_CACHE_PATH  # noqa: E402

# Câu kiểm tra khói: chỉ đòi hỏi bot TRẢ LỜI được bằng truy hồi / duyệt mục,
# không đòi đúng một bài cụ thể — corpus mới có thể có bài liên quan hơn.
SMOKE = [
    ("tin công nghệ mới nhất", {"intent", "retrieval"}),
    ("có bao nhiêu bài báo", {"intent"}),
    ("thời tiết sao hỏa hôm nay", {"fallback"}),
]


def main() -> int:
    before = INDEX_CACHE_PATH.stat().st_mtime if INDEX_CACHE_PATH.exists() else None

    t0 = time.perf_counter()
    bot = NewsChatbot().train(verbose=True)
    elapsed = time.perf_counter() - t0

    after = INDEX_CACHE_PATH.stat().st_mtime if INDEX_CACHE_PATH.exists() else None
    rebuilt = before is None or after != before

    stats = bot.retriever.stats()
    dates = [d for d in bot.retriever.published_dates if d]
    newest = max(dates).strftime("%d/%m/%Y") if dates else "không đọc được ngày"

    print()
    print(f"Index        : {'DỰNG LẠI (corpus đã đổi)' if rebuilt else 'nạp từ cache (corpus không đổi)'}"
          f" trong {elapsed:.1f}s")
    print(f"Số bài       : {stats['n_documents']}")
    print(f"Chuyên mục   : {stats['categories']}")
    print(f"Từ vựng      : {stats['vocabulary_size']:,} term")
    print(f"Bài mới nhất : {newest}")

    if stats["n_documents"] == 0:
        print("LỖI: corpus rỗng.")
        return 1

    failed = 0
    print("\nKiểm tra khói:")
    for question, ok_routes in SMOKE:
        bot.reset()
        route = bot.respond(question).route
        ok = route in ok_routes
        failed += not ok
        print(f"  [{'OK ' if ok else 'LỖI'}] {question!r} -> route={route} (mong đợi {sorted(ok_routes)})")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
