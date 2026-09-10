"""
api.py — Web API + trang chat tĩnh cho phần demo.

Chạy:
    .venv/Scripts/python.exe src/api.py
    # rồi mở http://127.0.0.1:8000

Endpoint:
    GET  /              trang chat
    POST /chat          {"message": "...", "session_id": "..."} -> câu trả lời
    POST /explain       {"message": "..."} -> chẩn đoán chi tiết
    GET  /stats         thống kê kho dữ liệu
    POST /reset         xóa lịch sử của một phiên

Mỗi `session_id` có một `DialogueState` riêng, nhờ vậy hai người dùng mở hai
tab khác nhau sẽ không giẫm lên ngữ cảnh của nhau — "tóm tắt bài đó" của
người này không lấy nhầm bài của người kia.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from chatbot import NewsChatbot, build_default_bot
from dialogue import DialogueState

WEB_DIR = Path(__file__).resolve().parent / "web"

app = FastAPI(title="Chatbot tin tức tiếng Việt", version="1.0")

# Mô hình được nạp MỘT lần lúc khởi động (mất vài giây vì phải dựng index),
# sau đó mọi request dùng chung.
bot: NewsChatbot | None = None

# Trạng thái hội thoại tách riêng theo phiên.
sessions: dict[str, DialogueState] = {}


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ExplainRequest(BaseModel):
    message: str


@app.on_event("startup")
def load_model() -> None:
    global bot
    print("Đang nạp corpus và huấn luyện mô hình...", file=sys.stderr)
    bot = build_default_bot()
    s = bot.retriever.stats()
    print(f"Sẵn sàng: {s['n_documents']} bài, {s['vocabulary_size']:,} term", file=sys.stderr)


@app.get("/")
def index():
    return FileResponse(WEB_DIR / "index.html")


@app.post("/chat")
def chat(req: ChatRequest):
    if bot is None:
        return JSONResponse({"error": "Mô hình chưa sẵn sàng."}, status_code=503)

    session_id = req.session_id or str(uuid.uuid4())
    # Gắn state của đúng phiên này vào bot trước khi xử lý.
    bot.state = sessions.setdefault(session_id, DialogueState())

    reply = bot.respond(req.message)

    return {
        "session_id": session_id,
        "reply": reply.text,
        "intent": reply.intent,
        "confidence": round(reply.confidence, 4),
        "route": reply.route,
        "sources": reply.sources,
        "entities": reply.entities,
    }


@app.post("/explain")
def explain(req: ExplainRequest):
    if bot is None:
        return JSONResponse({"error": "Mô hình chưa sẵn sàng."}, status_code=503)
    return bot.explain(req.message)


@app.get("/stats")
def stats():
    if bot is None:
        return JSONResponse({"error": "Mô hình chưa sẵn sàng."}, status_code=503)
    return bot.retriever.stats()


@app.post("/reset")
def reset(req: ChatRequest):
    if req.session_id and req.session_id in sessions:
        sessions[req.session_id].reset()
    return {"ok": True}


def main() -> int:
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
    return 0


if __name__ == "__main__":
    sys.exit(main())
