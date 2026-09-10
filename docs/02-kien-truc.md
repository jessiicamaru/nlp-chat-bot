# Kiến trúc hệ thống

## Sơ đồ luồng xử lý một lượt chat

```text
                         câu người dùng
                               |
                               v
                  +-------------------------+
                  |  entities.extract()     |  Lab 01: NER + Regex
                  |  - chuyên mục           |
                  |  - PER / LOC / ORG      |
                  |  - email/phone/ngày/số  |
                  +-------------------------+
                               |
                               v
                  +-------------------------+
                  |  preprocess_vi()        |  Lab 03
                  |  NFC -> tách từ ->      |
                  |  lowercase -> dấu câu   |
                  +-------------------------+
                               |
                               v
                  +-------------------------+
                  |  IntentClassifier       |  Lab 04 + NB tự cài đặt
                  |  ensemble:              |
                  |   0.8 * P_NB(c)         |
                  | + 0.2 * max cos(q, p_c) |
                  +-------------------------+
                       |               |
        conf >= 0.25   |               |  conf < 0.25
        và action      |               |  hoặc action == retrieve
        != retrieve    |               |
                       v               v
        +----------------------+   +--------------------------+
        |  thực thi action     |   |  NewsRetriever.search()  |  Lab 04
        |  reply / help /      |   |  TF-IDF + cosine         |
        |  stats / browse /    |   |  tầng 1: chọn bài        |
        |  summarize / source  |   |  tầng 2: chọn câu        |
        +----------------------+   +--------------------------+
                       |                  |             |
                       |          score>=0.12      score<0.12
                       |                  |             |
                       v                  v             v
                 câu soạn sẵn      trả lời có       fallback
                 hoặc dữ liệu       dẫn nguồn    "mình chưa biết"
                       |                  |             |
                       +------------------+-------------+
                                          |
                                          v
                               +---------------------+
                               |  DialogueState      |
                               |  nhớ bài vừa nhắc   |
                               |  nhớ chuyên mục     |
                               +---------------------+
```

## Vai trò từng module

| File | Trách nhiệm | Kế thừa lab |
|---|---|---|
| `config.py` | Đường dẫn, siêu tham số (đã dò bằng thực nghiệm) | — |
| `preprocess.py` | Chuẩn hóa, tách từ, stopwords, xử lý không dấu | Lab 03 |
| `vectorizer.py` | BoW, n-gram, TF-IDF, cosine — **tự cài đặt** | Lab 04 |
| `intent_classifier.py` | Naive Bayes **tự cài đặt** + ensemble cosine | Lab 04 |
| `retriever.py` | Truy hồi hai tầng, giải thích kết quả | Lab 01 + 04 |
| `entities.py` | NER + Regex, nhận diện chuyên mục | Lab 01 |
| `dialogue.py` | Trạng thái hội thoại, giải tham chiếu | — |
| `chatbot.py` | Điều phối toàn bộ | — |
| `crawler.py` | Thu thập bài báo VnExpress | Lab 02 |
| `evaluate.py` | Đo Recall@k, MRR, F1; dò ngưỡng | — |
| `cli.py` / `api.py` | Giao diện dòng lệnh / web | — |

## Ba quyết định thiết kế quan trọng

### 1. Mọi đường đi đều có ngưỡng tin cậy

Lỗi tệ nhất của một chatbot truy hồi không phải là im lặng, mà là **trả về một
bài báo ngẫu nhiên bằng giọng chắc chắn**. Vì vậy cả intent classifier lẫn
retriever đều có ngưỡng; dưới ngưỡng thì bot nói thẳng là không biết.

### 2. Hai cấu hình tiền xử lý cho hai nhiệm vụ

| | `CONFIG_INTENT` | `CONFIG_RETRIEVAL` |
|---|---|---|
| `remove_stopwords` | False | True |
| Lý do | câu hỏi 5–10 token, stopword **chính là** tín hiệu phân biệt intent | bài báo 3000+ ký tự, stopword làm loãng vector |

### 3. Hai index tách biệt cho câu có dấu và không dấu

Trộn chung sẽ làm gấp đôi vocabulary, pha loãng điểm số và làm mất hiệu lực các
ngưỡng đã dò. Giữ tách biệt và chỉ chuyển sang index phụ khi câu hỏi **không có
dấu nào** — một tín hiệu rất rõ ràng, vì tiếng Việt thật gần như luôn có dấu.

## Độ phức tạp

Với `n` document, `|V|` term trong vocabulary, `q` token trong câu hỏi:

| Thao tác | Độ phức tạp | Thực đo (381 bài) |
|---|---|---|
| Dựng index (một lần) | `O(n · độ dài trung bình)` | ~4 giây |
| Một truy vấn | `O(q · n)` trên ma trận thưa | < 100 ms |
| Bộ nhớ | `O(số phần tử khác 0)` | ma trận thưa CSR |

Chatbot chạy tức thì trên CPU, không cần GPU — đây là ưu điểm thật của phương
pháp cổ điển so với mô hình ngôn ngữ lớn.
