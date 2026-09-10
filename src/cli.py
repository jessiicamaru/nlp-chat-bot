"""
cli.py — Giao diện dòng lệnh để chat và debug.

Chạy:
    .venv/Scripts/python.exe src/cli.py
    .venv/Scripts/python.exe src/cli.py --debug        # hiện intent, điểm, thực thể
    .venv/Scripts/python.exe src/cli.py --ask "tin về đảo hải nam"

Lệnh trong phiên chat:
    /debug     bật/tắt chế độ debug
    /explain <câu>   xem chi tiết vì sao bot chọn câu trả lời đó
    /reset     xóa lịch sử hội thoại
    /stats     thống kê kho dữ liệu
    /quit      thoát
"""

from __future__ import annotations

import argparse
import sys

from chatbot import build_default_bot

BANNER = r"""
+--------------------------------------------------------------+
|   CHATBOT TIN TUC TIENG VIET                                 |
|   TF-IDF + Cosine Similarity + Naive Bayes (tu cai dat)      |
+--------------------------------------------------------------+
Go '/help' de xem lenh, '/quit' de thoat.
"""


def print_debug(reply) -> None:
    print(f"    [route={reply.route} intent={reply.intent or '-'} conf={reply.confidence:.3f}]")
    ent = reply.entities
    bits = []
    if ent.get("category"):
        bits.append(f"chuyên mục={ent['category']}")
    for key in ("persons", "locations", "organizations"):
        if ent.get(key):
            bits.append(f"{key}={ent[key]}")
    if ent.get("regex"):
        bits.append(f"regex={ent['regex']}")
    if bits:
        print(f"    [thực thể: {'; '.join(bits)}]")
    if reply.results:
        print("    [nguồn:]")
        for s in reply.sources:
            print(f"      - {s['score']:.3f} {s['title'][:62]}")


def print_explain(bot, text: str) -> None:
    info = bot.explain(text)
    intent = info["intent"]

    print("\n--- GIẢI THÍCH ---")
    print(f"Câu vào     : {text}")
    print(f"Token       : {intent['tokens']}")
    print(f"Thực thể    : {info['entities']}")
    print(f"\nIntent dự đoán: {intent['intent'] or '(không có)'} "
          f"(độ tin cậy {intent['confidence']:.3f})")
    print("  Top intent:")
    for tag, score in intent["top_intents"]:
        print(f"    {score:.3f}  {tag}")
    if intent["term_contributions"]:
        print("  Term đóng góp nhiều nhất:")
        for term, val in intent["term_contributions"][:5]:
            print(f"    {val:+.4f}  {term}")

    retr = info["retrieval"]
    print(f"\nTruy hồi:")
    print(f"  Term có trong từ vựng: {retr['terms_in_vocab'][:8]}")
    if retr["oov_terms"]:
        print(f"  Term KHÔNG có (OOV)  : {retr['oov_terms']}")
    for r in retr["results"]:
        print(f"    {r['score']:.4f} [{r['category']}] {r['title'][:52]}")
        print(f"           khớp: {[t for t, _ in r['matched_terms'][:4]]}")
    print("--- HẾT ---\n")


def chat_loop(bot, debug: bool = False) -> None:
    print(BANNER)
    stats = bot.retriever.stats()
    print(f"Kho du lieu: {stats['n_documents']} bai / "
          f"{stats['n_categories']} chuyen muc / "
          f"{stats['vocabulary_size']:,} term\n")

    while True:
        try:
            user = input("Bạn  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nTạm biệt!")
            return

        if not user:
            continue

        low = user.lower()
        if low in ("/quit", "/exit", "/q"):
            print("Tạm biệt!")
            return
        if low == "/help":
            print(__doc__)
            continue
        if low == "/debug":
            debug = not debug
            print(f"    (debug = {debug})")
            continue
        if low == "/reset":
            bot.reset()
            print("    (đã xóa lịch sử hội thoại)")
            continue
        if low == "/stats":
            print(bot.respond("có bao nhiêu bài báo").text)
            continue
        if low.startswith("/explain"):
            target = user[len("/explain"):].strip()
            if not target:
                print("    Cách dùng: /explain <câu cần giải thích>")
            else:
                print_explain(bot, target)
            continue

        reply = bot.respond(user)
        print(f"\nBot  > {reply.text}\n")
        if debug:
            print_debug(reply)


def main() -> int:
    ap = argparse.ArgumentParser(description="Chatbot tin tức tiếng Việt")
    ap.add_argument("--debug", action="store_true", help="hiện intent, điểm số, thực thể")
    ap.add_argument("--ask", type=str, default=None, help="hỏi một câu rồi thoát")
    ap.add_argument("--explain", type=str, default=None, help="giải thích một câu rồi thoát")
    ap.add_argument("--corpus", type=str, default=None, help="đường dẫn corpus CSV khác")
    args = ap.parse_args()

    print("Đang nạp dữ liệu và huấn luyện mô hình...", file=sys.stderr)
    bot = build_default_bot(corpus_path=args.corpus)

    if args.explain:
        print_explain(bot, args.explain)
        return 0

    if args.ask:
        reply = bot.respond(args.ask)
        print(reply.text)
        if args.debug:
            print_debug(reply)
        return 0

    chat_loop(bot, debug=args.debug)
    return 0


if __name__ == "__main__":
    sys.exit(main())
