"""
build_paraphrase_set.py — Tập truy vấn DIỄN ĐẠT LẠI (paraphrase) cho thí nghiệm Lab 05.

## Vì sao cần

Word Embedding (Lab 05) sinh ra để giải đúng một điểm yếu của TF-IDF: hai từ
khác mặt chữ nhưng cùng nghĩa ("ô_tô" vs "xe_hơi") là hai chiều vocabulary tách
biệt, nên cosine = 0 dù người đọc thấy chúng giống nhau.

Tập dev/test hiện có gần như toàn câu hỏi kiểu TỪ KHÓA ("giá iphone 18 pro",
"scb trương mỹ lan") — đúng thế mạnh của TF-IDF. Đo embedding trên đó là đo sai
chỗ: nó có thể giúp người dùng thật mà vẫn thua trên tập đánh giá. Tập này bù
đúng lỗ hổng đó: mỗi câu MÔ TẢ LẠI một bài báo bằng lời khác, cố tránh các từ
nội dung trong tiêu đề, dùng từ đồng nghĩa / cách nói vùng miền:

    "Vì sao nem rán thường cháy đen hai đầu?"  ->  "chả giò chiên bị khét hai đầu"
    "5 điều cần biết khi máy bay hạ cánh khẩn" ->  "phi cơ phải đáp khẩn cấp cần biết gì"

## Quy tắc

- Nhãn đúng theo URL (qua chỉ số bài trong kho; 381 bài đầu giữ nguyên thứ tự
  từ bản nộp, bài crawl thêm nối vào sau).
- Chia dev/test ngẫu nhiên có hạt giống cố định, 40% / 60%, như build_eval_sets.
- DEV dùng để dò (trọng số lai, tham số Word2Vec), TEST chỉ báo cáo MỘT lần.
- Người viết truy vấn cũng là người làm thí nghiệm, nên truy vấn có thể vô thức
  "chiều" một phương pháp. Để giảm, câu được viết TRƯỚC khi huấn luyện Word2Vec
  và không sửa lại sau khi thấy kết quả.

Chạy:  .venv/Scripts/python.exe tools/build_paraphrase_set.py
"""

import json
import random
import sys
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
CORPUS = PROJ / "data" / "raw" / "corpus_raw.csv"
OUT_DIR = PROJ / "data" / "eval"
SEED = 2026
DEV_RATIO = 0.4

# (câu hỏi diễn đạt lại, [chỉ số bài đúng], kiểu)
#   đồng nghĩa : thay từ nội dung bằng từ đồng nghĩa / cách nói khác
#   mô tả      : kể lại sự việc, gần như không dùng lại từ nào của tiêu đề
PARAPHRASES = [
    ("khu phố ở thủ đô Nhật hạn chế homestay vì khách thiếu văn minh", [38], "mô tả"),
    ("cô gái đi chơi xa một mình để quên người cũ", [40], "mô tả"),
    ("người Việt ngại tới hòn đảo Hàn Quốc sau vụ án mạng", [47], "mô tả"),
    ("giới trẻ cầm sách học đi chụp ảnh ở điểm địa chất", [49], "đồng nghĩa"),
    ("mẹo giữ an toàn khi ra nước ngoài chơi", [60], "đồng nghĩa"),
    ("nghe lời trí tuệ nhân tạo leo núi rồi bị lạc trong đêm", [61], "đồng nghĩa"),
    ("fanpage giả mạo lừa tiền đặt phòng dịp lễ", [63], "mô tả"),
    ("nơi nghỉ không có internet, muốn đặt chỗ phải gửi thư", [64], "mô tả"),
    ("phi cơ phải đáp khẩn cấp cần biết gì", [70], "đồng nghĩa"),
    ("hãng hàng không bán vé miễn phí ngày 9/9", [71], "đồng nghĩa"),
    ("lợi tức trái phiếu kho bạc Hoa Kỳ cao nhất ba năm", [79], "đồng nghĩa"),
    ("số người mở tài khoản mua bán cổ phiếu tăng mạnh", [83], "đồng nghĩa"),
    ("dầu thô đắt hơn 100 đô một thùng", [92], "đồng nghĩa"),
    ("thịt gia cầm ngoại nhập còn rẻ hơn rau xanh", [95], "đồng nghĩa"),
    ("dân thu nhập thấp ở Hoa Kỳ khó kiếm chỗ ở", [96], "đồng nghĩa"),
    ("đề xuất đánh thuế các món hàng giá trị nhỏ mua từ nước ngoài", [99], "mô tả"),
    ("linh kiện điện tử nhập nhiều hơn xuất tới 60 tỷ đô", [105], "đồng nghĩa"),
    ("smartphone màn hình gấp đầu tiên của Apple", [120, 121, 132], "đồng nghĩa"),
    ("nhà khoa học trưởng OpenAI nói thế giới chưa sẵn sàng với trí tuệ nhân tạo", [136], "đồng nghĩa"),
    ("Samsung làm máy gấp giống điện thoại thường", [147], "đồng nghĩa"),
    ("vì sao người ta khoái xem người máy té", [155], "đồng nghĩa"),
    ("tưởng đau bụng khó tiêu hóa ra khối u ở gan", [184], "đồng nghĩa"),
    ("sưng chân vì tĩnh mạch bị đứt đoạn", [185], "đồng nghĩa"),
    ("bé gái nhìn gần bị mờ nặng", [192], "mô tả"),
    ("một phần năm học trò cấp hai bị trêu chọc thường xuyên", [210], "đồng nghĩa"),
    ("nhiều trường ở Sài Gòn chưa cho học trò ăn trưa tại trường", [219], "đồng nghĩa"),
    ("thầy hiệu trưởng xoay sở khi gộp nhiều trường thành một", [226], "đồng nghĩa"),
    ("học bổng đại học Hồng Kông", [228], "đồng nghĩa"),
    ("ngày tựu trường ở ngôi trường có 112 lớp", [246], "mô tả"),
    ("lịch đá giải bóng đá Đông Nam Á năm 2026", [276], "đồng nghĩa"),
    ("chó máy dắt người khiếm thị đi đường", [294], "đồng nghĩa"),
    ("nhà máy thủy điện có con đập cao nhất hành tinh", [306], "đồng nghĩa"),
    ("robot lau nhà rồi sẽ phổ biến như máy giặt", [307], "đồng nghĩa"),
    ("Trung Quốc phóng thành công hỏa tiễn dùng lại được", [315], "đồng nghĩa"),
    ("iPhone mới bỏ tùy chọn màu tối", [321], "đồng nghĩa"),
    ("mô hình OpenAI tự tìm lỗ hổng an ninh mạng", [331], "đồng nghĩa"),
    ("cách nấu nước màu caramel cho món kho", [351], "đồng nghĩa"),
    ("trẻ em vùng núi Sơn La có nhà ăn mới", [358], "đồng nghĩa"),
    ("chả giò chiên bị khét hai đầu", [374], "đồng nghĩa"),
    ("Ả Rập Xê Út khóa ống dẫn, giá dầu leo thang", [404], "đồng nghĩa"),
    ("sếp Anthropic muốn các hãng làm chậm trí tuệ nhân tạo", [429], "đồng nghĩa"),
    ("Elon Musk tặng thiết bị Internet vệ tinh cho vùng mất sóng", [434, 500], "đồng nghĩa"),
    ("thói quen hằng ngày giúp giảm lượng đường trong máu", [444], "đồng nghĩa"),
    ("tỉnh thưởng tiền cho thầy cô có trò đoạt giải quốc tế", [464], "đồng nghĩa"),
    ("trường đại học dùng trí tuệ nhân tạo chấm thi nói", [467], "đồng nghĩa"),
    ("bé trai lạc trên núi được tìm thấy sau 17 giờ", [519], "đồng nghĩa"),
    ("những chỗ muỗi đẻ trứng trong nhà", [524], "đồng nghĩa"),
]


def main() -> int:
    df = pd.read_csv(CORPUS).dropna(subset=["title", "text"]).reset_index(drop=True)
    urls, titles = df["url"].tolist(), df["title"].tolist()

    items = []
    for query, ids, style in PARAPHRASES:
        for i in ids:
            if not 0 <= i < len(urls):
                raise IndexError(f"chỉ số bài {i} ngoài kho ({len(urls)} bài)")
        items.append({"query": query, "gold_urls": [urls[i] for i in ids],
                      "gold_titles": [titles[i] for i in ids], "style": style})

    rng = random.Random(f"{SEED}-paraphrase")
    rng.shuffle(items)
    k = round(len(items) * DEV_RATIO)
    for name, part, purpose in (
        ("dev", items[:k], "DÒ tham số thí nghiệm Word2Vec / trọng số lai."),
        ("test", items[k:], "CHỈ BÁO CÁO, một lần, với tham số đã chốt trên dev_paraphrase."),
    ):
        out = {"meta": {"split": f"{name}_paraphrase", "purpose": purpose, "seed": SEED},
               "retrieval": part}
        path = OUT_DIR / f"{name}_paraphrase.json"
        path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"{path.name}: {len(part)} câu")
    return 0


if __name__ == "__main__":
    sys.exit(main())
