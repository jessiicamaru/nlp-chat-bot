# Kiến trúc hệ thống

> Tài liệu mô tả **trạng thái hiện hành** của hệ thống. Lịch sử các thay đổi và lý
> do đằng sau từng thay đổi nằm ở docs/03 → docs/07. Mọi hằng số nhắc tới ở đây
> đều lấy từ `src/config.py` và đã được dò trên tập DEV (docs/06).

## Sơ đồ luồng xử lý một lượt chat

```text
                               câu người dùng
                                     |
                                     v
             +------------------------------------------------+
        [0]  |  normalizer.prepare_user_text()                |  học từ ViLexNorm
             |  - chuẩn hóa teencode: "bt"->"biết", "k"->"không"|
             |  - gộp ký tự lặp (chỉ chữ cái): "đẹppppp"->"đẹp" |
             |  - câu gốc KHÔNG dấu -> bỏ dấu lại sau chuẩn hóa |
             +------------------------------------------------+
                                     |
                                     v
             +------------------------------------------------+
        [1]  |  entities.extract()            Lab 01          |
             |  - chuyên mục (khớp từ khóa, không phân biệt dấu)|
             |  - NER underthesea: PER / LOC / ORG (ghép BIO)  |
             |  - Regex: email, phone, url, date, quantity     |
             +------------------------------------------------+
                                     |
                                     v
             +------------------------------------------------+
        [2]  |  IntentClassifier.predict()    Lab 03 + 04     |
             |  preprocess_vi(CONFIG_INTENT) -> TF-IDF (1,2)   |
             |  score = w_nb·P_NB(c|x) + (1-w_nb)·max cos(x,p_c)|
             |  w_nb = 1.0 (Naive Bayes thuần, dò trên dev)    |
             |  câu không dấu -> mô hình phụ ở mức âm tiết     |
             +------------------------------------------------+
                  |                                   |
   conf >= 0.25   |                                   |  conf < 0.25
   và action      |                                   |  hoặc action == retrieve
   != retrieve    v                                   v
   +----------------------------+     +------------------------------------------+
   | thực thi action            | [3] |  NewsRetriever.search()       Lab 04     |
   | reply / help /             |     |  - chọn index: có dấu | âm tiết không dấu |
   | list_categories / stats /  |     |  - lọc từ khung câu hỏi (QUERY_FRAME_WORDS)|
   | browse_category /          |     |  - XẾP HẠNG: cosine TF-IDF (hoặc BM25)    |
   | summarize / source         |     |              x (1 + 0.6·recency)          |
   +----------------------------+     |  - CHẤP NHẬN: cosine thuần >= 0.13        |
                  |                   |    (x 0.6 nếu người dùng đã nêu chuyên mục)|
                  |                   |  - tầng 2: 2 câu sát nhất trong bài       |
                  |                   +------------------------------------------+
                  |                        |                         |
                  |                    đạt ngưỡng              dưới ngưỡng
                  v                        v                         v
            câu soạn sẵn           trả lời + ngày đăng        fallback
            hoặc dữ liệu           + chuyên mục + nguồn     "mình chưa biết"
                  |                        |                         |
                  +------------------------+-------------------------+
                                           |
                                           v
                               +-----------------------+
                               |  DialogueState        |
                               |  bài vừa nhắc, mục    |
                               |  đang xem, số lần     |
                               |  fallback liên tiếp   |
                               +-----------------------+

  ---- Lớp TÙY CHỌN, mặc định tắt (chạy trên Colab, docs/07) ----
  rag.RagChatbot bọc toàn bộ luồng trên. Chỉ khi [3] đã có bài vượt ngưỡng:
    chốt chặn giả định -> PhoGPT-4B-Chat -> làm sạch -> chốt chặn rỗng / lặp lại / số bịa
  Mọi đường khác (intent, fallback) giữ nguyên câu trả lời trích xuất.
```

## Vai trò từng module

| File | Trách nhiệm | Kế thừa lab |
|---|---|---|
| `config.py` | Đường dẫn, siêu tham số (dò trên dev), từ khung câu hỏi | — |
| `preprocess.py` | Chuẩn hóa NFC, tách từ (có cache), stopwords, bỏ dấu, hạ về âm tiết | Lab 03 |
| `normalizer.py` | Học từ điển teencode từ ViLexNorm; `prepare_user_text` dùng chung cho bot và đánh giá | — |
| `vectorizer.py` | BoW, n-gram, TF-IDF, chuẩn hóa L2, cosine, BM25 — **tự cài đặt** | Lab 04 |
| `intent_classifier.py` | Multinomial Naive Bayes **tự cài đặt** + tín hiệu cosine tới pattern | Lab 04 |
| `retriever.py` | Truy hồi hai tầng, hai index, độ mới, cache đĩa, giải thích | Lab 01 + 04 |
| `dates.py` | Đọc ngày đăng VnExpress, tính điểm độ mới | — |
| `entities.py` | NER + Regex, nhận diện chuyên mục | Lab 01 |
| `dialogue.py` | Trạng thái hội thoại, giải tham chiếu "bài đó" | — |
| `chatbot.py` | Điều phối toàn bộ | — |
| `crawler.py` | Thu thập bài báo VnExpress (8 chuyên mục) | Lab 02 |
| `generator.py` | n-gram LM — thí nghiệm đối chứng, **không** dùng trong bot | — |
| `evaluate.py` | Dò tham số trên DEV, báo cáo trên TEST, Wilson CI, kiểm định dấu | — |
| `rag.py` | Lớp RAG tùy chọn: ghép ngữ cảnh, prompt, backend, chốt chặn | — |
| `cli.py` / `api.py` / `web/` | Giao diện dòng lệnh / FastAPI + trang chat | — |

## Các quyết định thiết kế quan trọng

### 1. Mọi đường đi đều có ngưỡng tin cậy

Lỗi tệ nhất của một chatbot truy hồi không phải là im lặng, mà là **trả về một
bài báo ngẫu nhiên bằng giọng chắc chắn**. Cả intent classifier (0.25) lẫn
retriever (0.13) đều có ngưỡng; dưới ngưỡng thì bot nói thẳng là không biết.

### 2. Xếp hạng và chấp nhận dùng HAI thước đo khác nhau

| Câu hỏi | Thước đo |
|---|---|
| Bài nào nên đứng trước? | cosine (hoặc BM25) **×** (1 + α·recency) |
| Có đủ căn cứ để trả lời không? | **chỉ** cosine TF-IDF, trong [0, 1] |

Trước đây ngưỡng áp lên điểm đã nhân độ mới, và độ mới âm thầm biến thành bộ lọc
loại bài cũ (docs/06, mục 5.1). BM25 cũng chỉ được dùng để xếp hạng vì điểm của
nó không bị chặn và không so được giữa các câu hỏi.

### 3. Hai cấu hình tiền xử lý cho hai nhiệm vụ

| | `CONFIG_INTENT` | `CONFIG_RETRIEVAL` |
|---|---|---|
| `remove_stopwords` | False | True |
| Lý do | câu hỏi 5–10 token, stopword **chính là** tín hiệu phân biệt intent | bài báo trung bình 3.358 ký tự, stopword làm loãng vector |

### 4. Hai index tách biệt cho câu có dấu và không dấu

Trộn chung sẽ làm gấp đôi vocabulary, pha loãng điểm số và làm mất hiệu lực các
ngưỡng đã dò. Giữ tách biệt và chỉ chuyển sang index âm tiết khi câu hỏi **không
có dấu nào**. Vì chuẩn hóa teencode luôn trả từ có dấu, `prepare_user_text` bỏ
dấu lại khi câu gốc không dấu — nếu không, một token được sửa sẽ làm cả câu bị
định tuyến sang index có dấu (docs/05, lỗi 2).

### 5. Một quy tắc nghiệp vụ chỉ nằm ở một chỗ

Danh sách từ khung câu hỏi (`QUERY_FRAME_WORDS`) và hệ số nới ngưỡng khi đã nêu
chuyên mục (`CATEGORY_SCOPED_THRESHOLD_FACTOR`) đều từng tồn tại ở hai nơi và
lệch nhau, gây ra lỗi âm thầm. Nay mỗi quy tắc chỉ có một định nghĩa trong
`config.py`.

### 6. LLM là lớp tùy chọn, không phải lõi

Thí nghiệm RAG (docs/07) cho thấy PhoGPT làm câu trả lời tự nhiên hơn nhưng không
trung thực hơn bản trích xuất. Vì vậy lớp này bọc bên ngoài, mặc định tắt, và
không mô-đun lõi nào phụ thuộc vào nó.

## Kích thước và tốc độ (381 bài, đo trên CPU laptop)

| Thành phần | Giá trị |
|---|---|
| Vocabulary index chính (unigram + bigram) | 105.020 term |
| Vocabulary index âm tiết không dấu | 87.294 term |
| Vocabulary intent classifier | 493 term (164 pattern, 14 intent) |
| Từ điển teencode học được | 342 ánh xạ |
| Dựng index, không có cache | ~24 s (gần như toàn bộ là `word_tokenize`) |
| Khởi động bot, có cache joblib | ~1.1 s |
| Trả lời một câu hỏi | ~21–24 ms |

| Thao tác | Độ phức tạp |
|---|---|
| Dựng index (một lần) | `O(n · độ dài trung bình)` |
| Một truy vấn | `O(q · n)` trên ma trận thưa CSR |
| Bộ nhớ | `O(số phần tử khác 0)` |

Chatbot lõi chạy tức thì trên CPU, không cần GPU — ưu điểm thật của phương pháp
cổ điển so với mô hình ngôn ngữ lớn. Chỉ lớp RAG tùy chọn mới cần GPU.
