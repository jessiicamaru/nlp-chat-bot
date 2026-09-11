# Chatbot tin tức tiếng Việt — Đồ án cuối kỳ NLP

Chatbot tiếng Việt xây dựng **from scratch** bằng TF-IDF, cosine similarity và
Naive Bayes tự cài đặt — **không dùng mô hình ngôn ngữ lớn**, không gọi
`sklearn` trong runtime.

Bot trả lời câu hỏi về tin tức dựa trên kho **381 bài báo VnExpress** thuộc
8 chuyên mục, do chính dự án thu thập.

**Sinh viên:** Hoàng Công Dũng — 23IT036

---

## Kết quả đo được

| Hạng mục | Chỉ số | Kết quả |
|---|---|---|
| TF-IDF tự cài đặt | khớp với scikit-learn | **28/28 test pass**, sai số ~1e-16 |
| Intent classification | Accuracy (tập test riêng) | **88.5%** |
| Intent classification | Macro-F1 | **0.91** |
| Retrieval | Recall@1 | **90.5%** |
| Retrieval | Recall@3 | **100%** |
| Retrieval | MRR | **0.952** |
| Retrieval | chặn câu ngoài phạm vi | **100%** |
| Chuẩn hóa teencode | ERR trên ViLexNorm test | **67.5%** |
| Chuẩn hóa teencode | Accuracy 83.9% → | **94.8%** |

Toàn bộ số liệu tái lập được bằng `python src/evaluate.py`.

---

## Cài đặt

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> Yêu cầu Python 3.10+. Dự án được phát triển và kiểm thử trên Python 3.14.

## Chạy

```powershell
# Chat trong terminal
python src/cli.py
python src/cli.py --debug                       # hiện intent, điểm số, thực thể
python src/cli.py --ask "tin về đảo hải nam"
python src/cli.py --explain "tin về đảo hải nam"  # giải thích vì sao chọn bài đó

# Web demo -> http://127.0.0.1:8000
python src/api.py

# Kiểm thử và đánh giá
python tests/test_vectorizer.py     # đối chiếu TF-IDF tự viết với sklearn
python src/evaluate.py              # Recall@k, MRR, F1, dò ngưỡng
python src/normalizer.py            # học + đánh giá từ điển teencode (ERR)
python src/generator.py             # thí nghiệm sinh văn bản n-gram

# Thu thập thêm dữ liệu
python src/crawler.py --per-category 45
python src/crawler.py --per-category 20 --only "Công nghệ" "Thể thao"
```

## Báo cáo

`notebooks/FinalProject_Chatbot_23IT036.ipynb` — notebook đã chạy và lưu output,
gồm đầy đủ lý thuyết, thực nghiệm, đánh giá và phân tích lỗi.

---

## Bot làm được gì

| Chức năng | Ví dụ câu hỏi |
|---|---|
| Tìm tin theo từ khóa | `cho tôi biết về đảo Hải Nam` |
| Xem tin theo chuyên mục | `tin công nghệ mới nhất` |
| Tìm trong một chuyên mục | `tin sức khỏe về ăn chuối` |
| Tóm tắt bài vừa xem | `tóm tắt bài đó` |
| Lấy link bài gốc | `cho mình link` |
| Thống kê kho dữ liệu | `có bao nhiêu bài báo` |
| Liệt kê chuyên mục | `có những chuyên mục nào` |
| Chào hỏi, cảm ơn, hỏi về bot | `bạn là ai` |
| **Hiểu câu gõ không dấu** | `tin ve dao hai nam` |
| **Hiểu teencode** | `bt gì về vụ iphone k b` → `biết gì về vụ iphone không bạn` |
| **Từ chối khi không biết** | `thời tiết sao hỏa hôm nay` → nói thẳng là không có dữ liệu |

---

## Cấu trúc dự án

```text
final-project/
├── src/
│   ├── config.py            # đường dẫn + siêu tham số (đã dò thực nghiệm)
│   ├── preprocess.py        # pipeline tiếng Việt (Lab 03) + xử lý không dấu
│   ├── vectorizer.py        # BoW / n-gram / TF-IDF / cosine — TỰ CÀI ĐẶT
│   ├── intent_classifier.py # Naive Bayes TỰ CÀI ĐẶT + ensemble cosine
│   ├── retriever.py         # truy hồi 2 tầng: bài báo -> câu
│   ├── normalizer.py        # chuẩn hóa teencode học từ ViLexNorm
│   ├── generator.py         # n-gram LM — thí nghiệm đối chứng sinh văn bản
│   ├── entities.py          # NER + Regex (Lab 01)
│   ├── dialogue.py          # trạng thái hội thoại, giải tham chiếu
│   ├── chatbot.py           # bộ điều phối
│   ├── crawler.py           # thu thập VnExpress (Lab 02)
│   ├── evaluate.py          # đánh giá + dò ngưỡng
│   ├── cli.py               # giao diện dòng lệnh
│   ├── api.py               # FastAPI
│   └── web/index.html       # giao diện chat
├── data/
│   ├── raw/corpus_raw.csv           # 381 bài báo
│   ├── intents/intents_vi.json      # 14 intent, ~150 pattern
│   ├── intents/test_queries.json    # tập test viết riêng
│   ├── resources/vietnamese-stopwords.txt
│   ├── resources/teencode_lexicon.json  # học được từ ViLexNorm
│   └── resources/vilexnorm/         # corpus chuẩn hóa (CC BY-NC-SA 4.0)
├── notebooks/FinalProject_Chatbot_23IT036.ipynb   # BÁO CÁO
├── docs/
│   ├── 01-nghien-cuu-du-an-tham-khao.md
│   ├── 02-kien-truc.md
│   ├── 03-chuan-hoa-teencode.md
│   └── 04-thi-nghiem-sinh-van-ban.md
├── tests/test_vectorizer.py
└── requirements.txt
```

---

## Kiến thức từ các lab được dùng ở đâu

| Lab | Nội dung | Dùng ở đâu |
|---|---|---|
| **Lab 01** | `sent_tokenize`, `word_tokenize`, `pos_tag`, `ner`, Regex | Tách câu chọn snippet; trích thực thể (`entities.py`) |
| **Lab 02** | requests + BeautifulSoup, validation | `crawler.py` — mở rộng corpus lên 381 bài |
| **Lab 03** | `preprocess_vi(text, config)`, stopwords | `preprocess.py` — pipeline dùng chung |
| **Lab 04** | BoW, n-gram, TF-IDF, cosine similarity | `vectorizer.py`, `retriever.py`, `intent_classifier.py` |

---

## Điểm nổi bật kỹ thuật

**1. TF-IDF viết tay và được kiểm chứng.** Toàn bộ công thức cài đặt bằng
NumPy + SciPy. `tests/test_vectorizer.py` đối chiếu với scikit-learn trên 5 cấu
hình khác nhau — khớp tới sai số ~1e-16. sklearn **chỉ** xuất hiện trong test.

**2. Hiểu được câu gõ không dấu.** `word_tokenize` tách sai hoàn toàn trên text
không dấu (`"tin ve dao hai nam"` → `['ve_dao','hai','nam']`). Giải pháp: dựng
index phụ ở mức **âm tiết**, chỉ dùng khi câu hỏi không có dấu. Thay đổi này
còn làm Recall@3 tăng từ 95.2% lên **100%**.

**3. Ngưỡng dò bằng thực nghiệm, không chọn cảm tính.** `evaluate.py` quét lưới
trên tập test viết riêng, tối ưu đồng thời độ chính xác và khả năng từ chối.

**4. Giải thích được từng câu trả lời.** `--explain` cho biết term nào khớp,
đóng góp bao nhiêu điểm, term nào là OOV.

**5. Biết nói "mình không biết".** Mọi đường đi đều có ngưỡng tin cậy.

**6. Hiểu teencode, và bảng ánh xạ được HỌC chứ không hardcode.** Học từ
[ViLexNorm](https://github.com/ngxtnhi/ViLexNorm) (10.467 cặp câu do người gán
nhãn) với ba điều kiện an toàn. ERR 67.5% trên split test chưa từng thấy.

**7. Đã kiểm chứng vì sao KHÔNG sinh văn bản.** `src/generator.py` cài đặt
n-gram LM hoàn chỉnh và đo: ở quy mô dữ liệu này, sinh văn bản thua truy hồi
trên mọi tiêu chí (bịa sự kiện, không dẫn được nguồn, n cao thì suy biến thành
chép nguyên văn). Lựa chọn kiến trúc dựa trên số liệu, không phải giả định.

---

## Hạn chế đã biết

1. **Không hiểu từ đồng nghĩa** — TF-IDF so khớp trên mặt chữ; "xe hơi" không
   tìm ra bài dùng "ô tô". Đây là hạn chế cốt lõi của mô hình túi từ.
2. **Không suy luận, không sinh văn bản, không diễn đạt lại** — bot trích xuất
   100%, chỉ trả về câu đã có sẵn trong corpus. Không thành phần nào có khả năng
   tạo ra từ chưa có trong dữ liệu. Đây là giới hạn **kiến trúc**, đã kiểm chứng
   bằng thực nghiệm ở [docs/04](docs/04-thi-nghiem-sinh-van-ban.md).
3. **Kho tri thức tĩnh** — muốn cập nhật phải chạy lại crawler.
4. **Hai intent chồng lấn** (`huong_dan` / `liet_ke_chuyen_muc`) vẫn nhầm lẫn.
5. **Tham chiếu chỉ neo vào lượt gần nhất** — "bài thứ hai ấy" chưa giải được.

Phân tích chi tiết 12 lỗi thật kèm nguyên nhân: xem **Phần J** của notebook.

Muốn bot **diễn đạt lại thay vì chép nguyên văn** thì cần ghép mô hình ngôn ngữ
lớn tiếng Việt (PhoGPT, Vistral) theo kiểu RAG — `NewsRetriever` hiện tại chính
là thành phần "R". Xem [docs/04](docs/04-thi-nghiem-sinh-van-ban.md).

---

## Nguồn tham khảo

- Danh sách stopword: [stopwords/vietnamese-stopwords](https://github.com/stopwords/vietnamese-stopwords)
- Thư viện NLP tiếng Việt: [underthesea](https://github.com/undertheseanlp/underthesea)
- Khảo sát các dự án chatbot tham khảo: [`docs/01-nghien-cuu-du-an-tham-khao.md`](docs/01-nghien-cuu-du-an-tham-khao.md)
- Corpus chuẩn hóa teencode: [ViLexNorm](https://github.com/ngxtnhi/ViLexNorm) —
  Nguyen et al., EACL 2024. Giấy phép **CC BY-NC-SA 4.0**, chỉ dùng cho mục đích
  nghiên cứu/học tập.
- Dữ liệu: bài báo công khai từ VnExpress, thu thập cho mục đích học tập.
