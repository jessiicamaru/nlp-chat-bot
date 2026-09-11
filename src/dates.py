"""
dates.py — Phân tích ngày đăng và tính điểm "độ mới" cho bài báo.

## Vì sao cần

Chatbot tin tức có một lỗi nguy hiểm hơn hẳn việc trả lời sai chủ đề: trả lời
bằng thông tin **đã lỗi thời**, kèm dẫn nguồn thật, bằng giọng chắc chắn.

Thí nghiệm đã đo được trên chính dự án này. Hai bài cùng chủ đề, mâu thuẫn nhau:

    Ngày 1  : "Giá vé tàu Cát Linh tăng lên 15.000 đồng từ tháng 10"
    Ngày 10 : "Hoãn tăng giá vé tàu Cát Linh, giữ nguyên 8.000 đồng"

Hỏi "giá vé tàu cát linh bao nhiêu", TF-IDF thuần xếp hạng:

    0.4956  bài NGÀY 1  (cũ, nay đã SAI)   <- bot trả lời bài này
    0.4098  bài NGÀY 10 (mới, ĐÚNG)

Bài cũ thắng chỉ vì tiêu đề của nó chứa đúng các từ trong câu hỏi
("giá vé tàu Cát Linh"), còn bài mới mở đầu bằng "Hoãn tăng giá".

TF-IDF chỉ đo độ trùng lặp từ ngữ. Nó **không biết** bài nào mới hơn, không biết
bài B phủ định bài A, và không có khái niệm "thông tin bị thay thế".

## Cách xử lý

Nhân điểm cosine với một hệ số độ mới:

    score' = cosine * (1 + alpha * recency)

    recency = 0.5 ^ (số ngày tuổi / nửa chu kỳ)   -> giảm dần theo hàm mũ, thuộc (0, 1]

`alpha` cố tình để NHỎ. Mục đích không phải là luôn ưu tiên bài mới, mà chỉ
**phá thế hòa**: khi hai bài có độ liên quan xấp xỉ nhau thì bài mới thắng.
Bài cũ nhưng liên quan hơn hẳn vẫn phải thắng bài mới ít liên quan — nếu không
bot sẽ chỉ trả về tin mới nhất bất kể người dùng hỏi gì.

`alpha` được dò bằng thực nghiệm trong `evaluate.py`, chọn giá trị lớn nhất mà
KHÔNG làm giảm Recall@1 và MRR trên tập test.

## Mốc thời gian tham chiếu

Dùng ngày đăng MỚI NHẤT trong corpus làm mốc, không dùng `datetime.now()`.
Lý do: kết quả phải tái lập được. Nếu lấy thời gian thực thì cùng một câu hỏi
sẽ cho thứ hạng khác nhau tùy hôm nào chạy, và mọi con số trong báo cáo sẽ
không kiểm chứng lại được. Với hệ thống chạy thật thì nên đổi sang thời gian
thực — ghi chú ở phần hướng phát triển.
"""

from __future__ import annotations

import datetime as dt
import re

import numpy as np

from config import FRESHNESS_HALFLIFE_DAYS

# VnExpress ghi ngày dạng: "Thứ ba, 25/8/2026, 10:59 (GMT+7)"
# Ta chỉ cần phần d/m/Y; giờ phút không ảnh hưởng tới xếp hạng theo ngày.
_DATE_PATTERN = re.compile(r"(\d{1,2})/(\d{1,2})/(\d{4})")

# Dạng ISO "2026-09-10" cũng được hỗ trợ, để nạp được corpus từ nguồn khác.
_ISO_PATTERN = re.compile(r"(\d{4})-(\d{1,2})-(\d{1,2})")


def parse_vn_date(value) -> dt.date | None:
    """Trích ngày từ chuỗi ngày đăng. Trả None nếu không đọc được."""
    if value is None:
        return None
    text = str(value)

    m = _ISO_PATTERN.search(text)
    if m:
        year, month, day = int(m.group(1)), int(m.group(2)), int(m.group(3))
    else:
        m = _DATE_PATTERN.search(text)
        if not m:
            return None
        day, month, year = int(m.group(1)), int(m.group(2)), int(m.group(3))

    try:
        return dt.date(year, month, day)
    except ValueError:
        # Ngày không hợp lệ (31/2, tháng 13...) -> coi như không đọc được.
        return None


def compute_recency(
    published: list[dt.date | None],
    reference: dt.date | None = None,
    half_life_days: float = FRESHNESS_HALFLIFE_DAYS,
) -> np.ndarray:
    """Tính điểm độ mới trong khoảng (0, 1] cho từng bài.

    Bài không đọc được ngày nhận điểm trung vị của phần còn lại — không thưởng
    mà cũng không phạt. Gán 0 sẽ đẩy chúng xuống đáy một cách oan uổng, gán 1
    lại thưởng cho dữ liệu thiếu.
    """
    valid = [d for d in published if d is not None]
    if not valid:
        return np.ones(len(published), dtype=np.float64)

    reference = reference or max(valid)

    scores = np.empty(len(published), dtype=np.float64)
    known: list[float] = []

    for i, d in enumerate(published):
        if d is None:
            scores[i] = np.nan
            continue
        age = max((reference - d).days, 0)
        value = 0.5 ** (age / half_life_days)
        scores[i] = value
        known.append(value)

    if known:
        scores = np.where(np.isnan(scores), float(np.median(known)), scores)
    else:
        scores = np.nan_to_num(scores, nan=1.0)

    return scores


def age_in_days(published: dt.date | None, reference: dt.date | None = None) -> int | None:
    if published is None:
        return None
    reference = reference or dt.date.today()
    return max((reference - published).days, 0)


def format_vn_date(d: dt.date | None) -> str:
    """Hiển thị ngày cho người đọc: 10/09/2026."""
    return d.strftime("%d/%m/%Y") if d else "không rõ ngày"
