# Khảo sát dự án mã nguồn mở tham khảo

Trước khi viết code, nhóm khảo sát các dự án chatbot tiếng Việt và chatbot
TF-IDF có sẵn trên GitHub để quyết định **kế thừa gì** và **tự viết gì**.

## Các dự án đã đánh giá

| Dự án | Kỹ thuật | Đánh giá | Quyết định |
|---|---|---|---|
| [Dec1mo/Vietnamese-Chatbot-From-Scratch](https://github.com/Dec1mo/Vietnamese-Chatbot-From-Scratch) | Keras NN phân loại intent + context handler + fuzzy matching địa chỉ, dùng underthesea | Kiến trúc rất tốt: tách bạch `intent → context → slot filling`. Nhưng mô hình Keras vượt phạm vi các lab đã học | **Kế thừa kiến trúc**, thay mô hình bằng Naive Bayes + TF-IDF tự cài đặt |
| [undertheseanlp/chatbot](https://github.com/undertheseanlp/chatbot) | ChatScript (luật) + Django 1.11 | Chatbot chit-chat "Hoài An". Django 1.11 đã quá cũ; thuần luật nên không có thành phần học máy nào. Giấy phép GPL-3.0 sẽ ràng buộc toàn bộ mã nguồn | **Không dùng** |
| [heraclex12/vietnamese-chat-with-rasa](https://github.com/heraclex12/vietnamese-chat-with-rasa) | Framework Rasa | Rasa lo hết phần NLU — trái với yêu cầu "from scratch" của đồ án | **Chỉ tham khảo** cách tổ chức tập intent |
| [sushant097/Chatbot-using-Python-NLTK](https://github.com/sushant097/Chatbot-using-Python-NLTK-) | TF-IDF + cosine similarity, NLTK | Đúng kỹ thuật của Lab 04, nhưng là tiếng Anh và gọi thẳng sklearn | **Kế thừa ý tưởng** truy hồi, viết lại cho tiếng Việt và tự cài đặt TF-IDF |
| [YUSANITY/TF-IDF-DOCUMENT-RETRIEVAL-CHATBOT](https://github.com/YUSANITY/TF-IDF-DOCUMENT-RETRIEVAL-CHATBOT) | TF-IDF chấm điểm tài liệu | Cùng hướng với dự án trên | **Tham khảo** cách chấm điểm tài liệu |
| [stopwords/vietnamese-stopwords](https://github.com/stopwords/vietnamese-stopwords) | Danh sách 1.942 stopword tiếng Việt | Tài nguyên chất lượng, dùng được ngay | **Dùng trực tiếp** (`data/resources/`) |

## Kết luận: kế thừa gì, tự viết gì

### Kế thừa
- **Kiến trúc 3 tầng** từ `Dec1mo`: intent classification → xử lý ngữ cảnh → sinh câu trả lời.
- **Định dạng `intents.json`** (`tag` / `patterns` / `responses`) — quy ước phổ biến trong hầu hết dự án chatbot.
- **Ý tưởng truy hồi bằng cosine similarity** từ các dự án NLTK.
- **Danh sách stopword tiếng Việt** từ `stopwords/vietnamese-stopwords`.

### Tự viết hoàn toàn
- `vectorizer.py` — BoW, n-gram, TF-IDF, chuẩn hóa L2, cosine similarity, BM25 (NumPy + SciPy).
- `intent_classifier.py` — Multinomial Naive Bayes + tín hiệu cosine tới pattern gần nhất.
- `retriever.py` — truy hồi hai tầng (bài báo → câu), hai index, xếp hạng theo độ mới.
- `preprocess.py` — pipeline tiếng Việt, kể cả xử lý câu không dấu.
- `normalizer.py` — thuật toán học từ điển teencode (dữ liệu học lấy từ ViLexNorm).
- `generator.py` — n-gram language model cho thí nghiệm đối chứng.
- `entities.py`, `dialogue.py`, `dates.py`, `chatbot.py`, `crawler.py`, `evaluate.py`.
- Toàn bộ tập intent tiếng Việt (14 intent, 164 pattern) và tập đánh giá dev/test.

### Dùng mô hình có sẵn (ngoài phạm vi "from scratch", ghi rõ)
- **underthesea** — `word_tokenize`, `sent_tokenize`, `ner` (đúng như các lab).
- **PhoGPT-4B-Chat** (VinAI) — chỉ trong lớp RAG **tùy chọn, mặc định tắt**
  (`rag.py`, docs/07). Không mô-đun lõi nào phụ thuộc vào nó.

### Khác biệt chính so với mọi dự án tham khảo

1. **TF-IDF tự cài đặt và được kiểm chứng** — không dự án nào ở trên tự viết
   công thức; tất cả đều gọi sklearn/NLTK. Đồ án này viết tay và chứng minh
   khớp với sklearn tới sai số ~1e-16.
2. **Xử lý câu gõ không dấu** — không dự án tham khảo nào giải quyết, dù đây là
   cách gõ rất phổ biến của người Việt.
3. **Ngưỡng dò bằng thực nghiệm** — các dự án tham khảo đặt ngưỡng bằng số
   "đẹp" tùy chọn; đồ án này quét lưới trên tập **DEV** và báo cáo một lần trên
   tập **TEST** tách riêng (docs/06). Bản đầu của đồ án từng dò và báo cáo trên
   cùng một tập — lỗi đó đã được sửa và ghi lại.
4. **Có đánh giá định lượng** — Recall@k, MRR, macro-F1, khoảng tin cậy Wilson
   95%, kiểm định dấu có cặp. Các dự án tham khảo chỉ demo định tính.
