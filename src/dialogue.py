"""
dialogue.py — Quản lý trạng thái hội thoại (dialogue state tracking).

Đây là thứ tách biệt "chatbot" khỏi "công cụ tìm kiếm". Nếu không có state,
mỗi câu bị xử lý độc lập và đoạn hội thoại sau sẽ hỏng:

    User: cho tôi biết về đảo Hải Nam
    Bot : [bài báo A]
    User: tóm tắt bài đó          <- "bài đó" là bài nào?
    User: cho mình link           <- link của bài nào?

`DialogueState` giữ lại bài báo vừa nhắc tới (last_results) và chuyên mục
đang quan tâm, nhờ đó các intent tom_tat_bai / hoi_nguon giải được tham chiếu.

Đây là dạng đơn giản nhất của **anaphora resolution** (giải tham chiếu):
không phân tích cú pháp, chỉ neo vào lượt hội thoại gần nhất. Đủ dùng cho
phạm vi đồ án và giới hạn của nó được ghi rõ ở phần Error Analysis.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime

from config import HISTORY_MAXLEN


@dataclass
class Turn:
    """Một lượt hội thoại đã hoàn tất."""

    user_text: str
    bot_text: str
    intent: str
    confidence: float
    route: str                      # đường đi đã chọn: intent / retrieval / fallback
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))


class DialogueState:
    """Bộ nhớ ngắn hạn của một phiên chat."""

    def __init__(self, maxlen: int = HISTORY_MAXLEN):
        self.history: deque[Turn] = deque(maxlen=maxlen)
        self.last_results: list = []       # list[RetrievalResult] của lượt gần nhất
        self.last_category: str | None = None
        self.last_query: str | None = None
        self.fallback_streak: int = 0      # số lần fallback liên tiếp

    # -- cập nhật ------------------------------------------------------------
    def add_turn(self, turn: Turn) -> None:
        self.history.append(turn)
        if turn.route == "fallback":
            self.fallback_streak += 1
        else:
            self.fallback_streak = 0

    def remember_results(self, results: list, query: str | None = None) -> None:
        """Chỉ ghi đè khi thực sự có kết quả.

        Nếu lượt này không tìm được gì, ta GIỮ kết quả cũ để người dùng vẫn
        hỏi tiếp được "tóm tắt bài đó" về bài đã nhắc trước đó.
        """
        if results:
            self.last_results = results
            if query:
                self.last_query = query

    def remember_category(self, category: str | None) -> None:
        if category:
            self.last_category = category

    # -- truy vấn ------------------------------------------------------------
    @property
    def current_doc(self):
        """Bài báo đang được nhắc tới, hoặc None."""
        return self.last_results[0] if self.last_results else None

    def needs_help_nudge(self) -> bool:
        """Fallback 2 lần liên tiếp -> chủ động gợi ý cách dùng."""
        return self.fallback_streak >= 2

    def transcript(self) -> str:
        lines = []
        for t in self.history:
            lines.append(f"User: {t.user_text}")
            lines.append(f"Bot : {t.bot_text}")
        return "\n".join(lines)

    def reset(self) -> None:
        self.history.clear()
        self.last_results = []
        self.last_category = None
        self.last_query = None
        self.fallback_streak = 0
