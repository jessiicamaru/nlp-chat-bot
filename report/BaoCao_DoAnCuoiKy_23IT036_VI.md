# BÁO CÁO ĐỒ ÁN CUỐI KỲ — XỬ LÝ NGÔN NGỮ TỰ NHIÊN

# Chatbot hỏi đáp tin tức tiếng Việt xây dựng từ đầu: truy hồi TF-IDF tự cài đặt, xử lý văn bản phi chuẩn, và đánh giá thực nghiệm một lớp sinh câu RAG

| | |
|---|---|
| **Sinh viên** | Hoàng Công Dũng |
| **Mã số sinh viên** | 23IT036 |
| **Học phần** | Xử lý ngôn ngữ tự nhiên |
| **Mã nguồn** | <https://github.com/jessiicamaru/nlp-chat-bot> |
| **Thời gian thực hiện** | 10/09/2026 – 13/09/2026 (19 commit) |
| **Sản phẩm đi kèm** | mã nguồn, notebook báo cáo đã chạy (86 ô, 0 lỗi), 7 tài liệu kỹ thuật, 95 kiểm thử tự động, notebook Colab cho RAG |

---

## Tóm tắt

Báo cáo trình bày quá trình xây dựng và đánh giá một chatbot hỏi đáp tin tức tiếng
Việt trên kho 381 bài báo VnExpress do chính dự án thu thập. Phần lõi được cài đặt
**từ đầu** bằng các kỹ thuật của Lab 01–04: tiền xử lý tiếng Việt, túi từ, n-gram,
TF-IDF và độ tương đồng cosine viết tay bằng NumPy/SciPy (khớp scikit-learn tới sai
số cỡ $10^{-16}$), phân loại ý định bằng Multinomial Naive Bayes tự cài đặt, và truy
hồi hai tầng (bài báo → câu). Hệ thống xử lý ba dạng văn bản phi chuẩn thường gặp khi
chat: câu **không dấu** (bằng một index phụ ở mức âm tiết), **teencode** (bằng một từ
điển ánh xạ *học* từ kho ViLexNorm, ERR 67,84% trên split test), và **tin lỗi thời**
(bằng xếp hạng có thưởng độ mới, tách bạch với ngưỡng chấp nhận).

Điểm nhấn phương pháp của đồ án là **đánh giá trung thực**. Các con số ban đầu (Recall@1
96,8%, độ chính xác intent 88,5%) được phát hiện là bị **rò rỉ tập test** và đã được đo
lại theo quy trình tách DEV/TEST, gán nhãn theo URL bài báo, khoảng tin cậy Wilson 95%
và đánh giá đầu-cuối. Kết quả thật trên tập TEST: Recall@1 truy hồi **93,4%** (KTC 88–97%),
MRR **0,950**, độ chính xác intent **61,5%** (43–78%), và **77,0%** câu hỏi tin tức nhận
đúng bài qua toàn bộ hệ thống. Ba thí nghiệm cho **kết quả âm** được ghi lại đầy đủ: mô
hình sinh n-gram thua truy hồi trên mọi tiêu chí; BM25 tự cài đặt không hơn TF-IDF (kiểm
định dấu có cặp, p = 1,0); và một lớp RAG với PhoGPT-4B-Chat — sau ba lần chạy thật trên
Google Colab — làm câu trả lời **tự nhiên hơn hẳn** (câu dùng được 9/33 → 19/33, p = 0,006)
nhưng **không trung thực hơn** (câu chứa thông tin sai 3/33 → 4/33), và còn tệ hơn bản
trích xuất với giả định sai không chứa con số. Do đó chatbot nộp bài là bản trích xuất;
RAG được giữ như một lớp mở rộng tùy chọn, mặc định tắt.

**Từ khóa:** chatbot tiếng Việt, TF-IDF, BM25, Naive Bayes, chuẩn hóa từ vựng, teencode,
truy hồi thông tin, RAG, PhoGPT, đánh giá tách tập, kết quả âm.

---

## Mục lục

1. [Giới thiệu](#1-giới-thiệu)
2. [Cơ sở lý thuyết và công trình liên quan](#2-cơ-sở-lý-thuyết-và-công-trình-liên-quan)
3. [Dữ liệu](#3-dữ-liệu)
4. [Kiến trúc hệ thống](#4-kiến-trúc-hệ-thống)
5. [Phương pháp chi tiết](#5-phương-pháp-chi-tiết)
6. [Phương pháp đánh giá](#6-phương-pháp-đánh-giá)
7. [Tiến trình phát triển theo giai đoạn](#7-tiến-trình-phát-triển-theo-giai-đoạn)
8. [Kết quả thực nghiệm](#8-kết-quả-thực-nghiệm)
9. [Thảo luận và phân tích lỗi](#9-thảo-luận-và-phân-tích-lỗi)
10. [Hạn chế và hướng phát triển](#10-hạn-chế-và-hướng-phát-triển)
11. [Kết luận](#11-kết-luận)
12. [Tuyên bố về việc sử dụng công cụ AI](#12-tuyên-bố-về-việc-sử-dụng-công-cụ-ai)
13. [Tài liệu tham khảo](#13-tài-liệu-tham-khảo)
14. [Phụ lục](#14-phụ-lục)

---

## 1. Giới thiệu

### 1.1. Bối cảnh

Chatbot hỏi đáp trên một kho văn bản là bài toán kinh điển của xử lý ngôn ngữ tự nhiên
và truy hồi thông tin. Với tiếng Việt, bài toán có thêm ba khó khăn đặc thù. Thứ nhất,
tiếng Việt là ngôn ngữ đơn lập: đơn vị có nghĩa là **từ** (có thể gồm nhiều âm tiết,
như `hải_nam`, `sức_khỏe`), nên bắt buộc phải **tách từ** trước khi biểu diễn văn bản.
Thứ hai, người dùng chat thường gõ **không dấu** (`tin ve dao hai nam`) hoặc dùng
**teencode** (`bt gì về vụ iphone k b`), là những dạng văn bản mà bộ tách từ huấn luyện
trên văn bản chuẩn xử lý sai hoàn toàn. Thứ ba, với **tin tức**, thông tin thay đổi theo
thời gian: một bài báo hôm nay có thể phủ định bài báo tuần trước.

### 1.2. Mục tiêu và ràng buộc

Đồ án đặt ra các mục tiêu sau:

1. Xây dựng một chatbot tiếng Việt trả lời câu hỏi về tin tức trên kho bài báo tự thu thập.
2. **Tự cài đặt** phần lõi (biểu diễn văn bản, phân loại ý định, truy hồi) bằng đúng các
   kỹ thuật đã học ở Lab 01–04, và **chứng minh** bản tự cài đặt đúng.
3. Xử lý được câu không dấu, teencode và tin lỗi thời.
4. Mọi siêu tham số phải được **dò bằng thực nghiệm**, và mọi kết quả phải được **báo cáo
   trung thực** — kể cả kết quả âm.
5. (Mở rộng) Kiểm tra bằng thực nghiệm câu hỏi "có nên dùng mô hình sinh để câu trả lời
   tự nhiên hơn không?".

Ràng buộc: không dùng mô hình ngôn ngữ lớn trong phần lõi; `scikit-learn` chỉ được dùng để
**đối chiếu** trong kiểm thử, không xuất hiện trong mã chạy; phần lõi chạy trên CPU laptop.

### 1.3. Đóng góp chính

- **Cài đặt từ đầu có kiểm chứng:** TF-IDF, cosine, BM25, Multinomial Naive Bayes, n-gram LM
  viết tay; 45 kiểm thử đối chiếu với scikit-learn và với cài đặt tham chiếu.
- **Xử lý văn bản phi chuẩn:** index âm tiết cho câu không dấu; từ điển teencode *học* từ
  10.467 cặp câu có gán nhãn với ba điều kiện an toàn; cơ chế "giữ hệ quy chiếu dấu".
- **Xử lý tin lỗi thời:** xếp hạng nhân hệ số độ mới, **tách bạch "xếp hạng" và "chấp nhận"**.
- **Quy trình đánh giá trung thực:** phát hiện và sửa rò rỉ tập test, tách DEV/TEST, nhãn
  theo URL, KTC Wilson, kiểm định dấu, đánh giá đầu-cuối, nhật ký mọi lần xem tập test.
- **Ba kết quả âm được ghi lại đầy đủ:** sinh n-gram, BM25, và RAG với PhoGPT.
- **21 kiểm thử hồi quy**, mỗi kiểm thử ứng với một lỗi thật đã gặp; chúng bắt được hai lỗi
  thiết kế ngay lần chạy đầu.

### 1.4. Cấu trúc báo cáo

Mục 2 tóm tắt cơ sở lý thuyết. Mục 3 mô tả dữ liệu. Mục 4–5 trình bày kiến trúc và phương
pháp. Mục 6 mô tả quy trình đánh giá. Mục 7 kể lại **tiến trình phát triển theo từng giai
đoạn**, kèm số liệu tại mỗi giai đoạn và những lần phải sửa notebook. Mục 8 trình bày kết
quả. Mục 9–11 thảo luận, nêu hạn chế và kết luận.

---

## 2. Cơ sở lý thuyết và công trình liên quan

### 2.1. Tiền xử lý tiếng Việt

Pipeline tiền xử lý kế thừa Lab 03 gồm: chuẩn hóa Unicode về dạng **NFC** (hai chuỗi hiển
thị giống nhau có thể có code point khác nhau ở dạng NFD), tách từ bằng `word_tokenize` của
underthesea, chuyển chữ thường, bỏ dấu câu, và (tùy cấu hình) bỏ stopword.

### 2.2. Túi từ, n-gram và TF-IDF

Mô hình túi từ biểu diễn văn bản $d$ bằng vector tần suất $\mathrm{tf}(t, d)$. Bigram bổ
sung một phần thông tin thứ tự (phân biệt "không tốt" với "tốt"). TF-IDF giảm trọng số của
term phổ biến:

$$\mathrm{idf}(t) = \ln\frac{1 + N}{1 + \mathrm{df}(t)} + 1 \qquad \mathrm{tfidf}(t,d) = \mathrm{tf}'(t,d)\cdot\mathrm{idf}(t)$$

với $\mathrm{tf}'(t,d) = 1 + \ln \mathrm{tf}(t,d)$ khi dùng *sublinear tf*. Vector được chuẩn
hóa L2, nên độ tương đồng cosine rút gọn thành tích vô hướng:
$\cos(\vec a, \vec b) = \vec a \cdot \vec b / (\lVert\vec a\rVert\lVert\vec b\rVert)$.

### 2.3. Okapi BM25

BM25 [Robertson & Zaragoza, 2009] sửa hai điểm yếu của TF-IDF: tần suất **bão hòa** và chuẩn
hóa độ dài **có tham số**:

$$\mathrm{score}(q,d) = \sum_{t\in q}\mathrm{idf}(t)\cdot\frac{\mathrm{tf}(t,d)\,(k_1+1)}{\mathrm{tf}(t,d) + k_1\left(1 - b + b\,\frac{|d|}{\mathrm{avgdl}}\right)}, \qquad \mathrm{idf}(t) = \ln\!\left(\frac{N-\mathrm{df}(t)+0{,}5}{\mathrm{df}(t)+0{,}5}+1\right)$$

Dạng idf có "+1" (biến thể của Lucene) luôn dương, tránh trường hợp khớp một term rất phổ
biến lại làm giảm điểm.

### 2.4. Multinomial Naive Bayes

Với lớp $c$ và văn bản $x$ biểu diễn bằng trọng số term, có làm mịn Laplace hệ số $\alpha$:

$$P(t\mid c) = \frac{\mathrm{count}(t,c) + \alpha}{\sum_{t'}\mathrm{count}(t',c) + \alpha|V|}, \qquad \hat c = \arg\max_c\Big[\log P(c) + \sum_t x_t\log P(t\mid c)\Big]$$

Hậu nghiệm $P(c\mid x)$ thu được bằng softmax ổn định số học trên log-likelihood.

### 2.5. Chuẩn hóa từ vựng (lexical normalization)

Chuẩn hóa từ vựng đưa token phi chuẩn về dạng chuẩn (`k` → `không`). ViLexNorm [Nguyen và
cộng sự, EACL 2024] là kho 10.467 cặp câu bình luận mạng xã hội tiếng Việt kèm bản chuẩn hóa do
người gán nhãn. Chỉ số chuẩn của bài toán là **ERR** (Error Reduction Rate):

$$\mathrm{ERR} = \frac{\mathrm{Acc}_{\text{sau}} - \mathrm{Acc}_{\text{trước}}}{1 - \mathrm{Acc}_{\text{trước}}}$$

ERR được dùng thay cho accuracy thô vì phần lớn token (~84%) vốn đã đúng sẵn.

### 2.6. Mô hình ngôn ngữ n-gram

Mô hình n-gram giả định Markov bậc $n-1$. Để tránh xác suất 0, dùng nội suy đệ quy:

$$P_{\text{interp}}(w\mid h) = \lambda\,P_{\text{ML}}(w\mid h) + (1-\lambda)\,P_{\text{interp}}(w\mid h_{2:})$$

và đo bằng perplexity $\mathrm{PP} = \exp\!\big(-\tfrac1N\sum_i\log P(w_i\mid h_i)\big)$.

### 2.7. Retrieval-Augmented Generation

RAG [Lewis và cộng sự, 2020] ghép một bộ truy hồi với một mô hình sinh: bộ truy hồi lấy tài
liệu liên quan, mô hình sinh viết câu trả lời dựa trên các tài liệu đó. Đồ án dùng
**PhoGPT-4B-Chat** [Nguyen và cộng sự, 2023] — mô hình sinh tiếng Việt khoảng 3,7 tỷ tham số
của VinAI — với mẫu prompt chính thức `### Câu hỏi: {instruction}\n### Trả lời:`.

### 2.8. Khảo sát dự án tham khảo

Trước khi viết mã, sáu dự án mã nguồn mở đã được khảo sát (`docs/01`):

| Dự án | Kỹ thuật | Quyết định |
|---|---|---|
| Dec1mo/Vietnamese-Chatbot-From-Scratch | Keras NN + context handler + underthesea | Kế thừa **kiến trúc** intent → ngữ cảnh → trả lời; thay mô hình bằng NB + TF-IDF tự cài đặt |
| undertheseanlp/chatbot | ChatScript (luật) + Django 1.11, GPL-3.0 | Không dùng |
| heraclex12/vietnamese-chat-with-rasa | Framework Rasa | Chỉ tham khảo cách tổ chức intent (Rasa lo hết NLU, trái yêu cầu from scratch) |
| sushant097/Chatbot-using-Python-NLTK | TF-IDF + cosine (tiếng Anh, gọi sklearn) | Kế thừa **ý tưởng** truy hồi, viết lại cho tiếng Việt |
| YUSANITY/TF-IDF-DOCUMENT-RETRIEVAL-CHATBOT | Chấm điểm tài liệu bằng TF-IDF | Tham khảo |
| stopwords/vietnamese-stopwords | Danh sách 1.942 stopword | Dùng trực tiếp |

Khác biệt chính của đồ án so với các dự án trên: TF-IDF tự viết và có kiểm chứng; xử lý câu
không dấu và teencode; ngưỡng dò bằng thực nghiệm trên DEV; có đánh giá định lượng kèm khoảng
tin cậy.

---

## 3. Dữ liệu

### 3.1. Kho bài báo

Kho ban đầu từ Lab 02 chỉ có **28 bài**, cùng một chuyên mục (Du lịch). `src/crawler.py` mở
rộng lên **381 bài / 8 chuyên mục** từ VnExpress, kế thừa quy trình Lab 02 (requests +
BeautifulSoup + kiểm tra hợp lệ) và bổ sung: nghỉ ngẫu nhiên 0,8–1,6 giây giữa hai request,
cache theo URL để chạy lại không tải trùng, loại bài có phần thân dưới 300 ký tự, ghi log lỗi
ra file riêng, và **gộp** vào kho cũ theo URL (dữ liệu tích lũy, không ghi đè).

| Chuyên mục | Số bài |
|---|---|
| Du lịch | 72 |
| Công nghệ | 48 |
| Sức khỏe | 45 |
| Đời sống | 45 |
| Kinh doanh | 44 |
| Thể thao | 44 |
| Khoa học | 43 |
| Giáo dục | 40 |
| **Tổng** | **381** |

Mỗi bài có các trường `url, source, category, title, description, text, published_at,
crawled_at`. Độ dài phần thân trung bình **3.358 ký tự** (trung vị 2.857). Ngày đăng trải từ
17/04/2025 đến 10/09/2026.

### 3.2. Tập intent

`data/intents/intents_vi.json` gồm **14 intent, 164 pattern** viết tay, mỗi intent gắn một
*action*:

| Action | Intent |
|---|---|
| `reply` (câu soạn sẵn) | `chao_hoi`, `tam_biet`, `cam_on`, `hoi_ve_bot`, `dong_y`, `tu_choi`, `che_bai` |
| `help` | `huong_dan` |
| `list_categories` | `liet_ke_chuyen_muc` |
| `stats` | `thong_ke_corpus` |
| `browse_category` | `tin_theo_chuyen_muc` |
| `summarize` | `tom_tat_bai` |
| `source` | `hoi_nguon` |
| `retrieve` | `tim_tin` |

### 3.3. Tài nguyên ngôn ngữ

- **Stopword:** 1.942 mục từ `stopwords/vietnamese-stopwords` (nạp cả dạng có khoảng trắng và
  dạng nối `_` để khớp với đầu ra tách từ).
- **ViLexNorm** [Nguyen và cộng sự, 2024]: 10.467 cặp câu, chia sẵn train/dev/test; giấy phép
  CC BY-NC-SA 4.0 (chỉ dùng cho nghiên cứu/học tập, có ghi nguồn).

### 3.4. Tập đánh giá chatbot (DEV/TEST)

Tập đánh giá được sinh bởi `tools/build_eval_sets.py` (hạt giống 2026) theo quy tắc: **mọi
truy vấn cũ** — những câu từng được dùng để dò tham số trước khi phát hiện rò rỉ — bắt buộc vào
DEV; truy vấn **mới** viết thêm được chia ngẫu nhiên 40% DEV / 60% TEST. Mỗi truy vấn truy hồi
được gán **URL của (các) bài đúng**, thay vì tiêu chí dễ dãi "tiêu đề chứa từ X".

| | DEV | TEST |
|---|---|---|
| Mục đích | dò tham số, xem nhiều lần | **chỉ báo cáo** |
| Truy vấn truy hồi | 112 (có dấu 78 · không dấu 23 · teencode 11) | 122 (có dấu 86 · không dấu 22 · teencode 14) |
| Câu ngoài phạm vi | 28 | 24 |
| Câu intent | 55 | 32 (26 câu thuộc intent cố định dùng tính accuracy) |

**Ca tin mâu thuẫn** (`data/eval/conflict_case.json`): hai bài giả lập cách nhau 9 ngày về giá
vé tàu Cát Linh — bài 01/09 "tăng lên 15.000 đồng", bài 10/09 "hoãn tăng giá, giữ nguyên 8.000
đồng" — cùng 4 cách hỏi. Đây là **ràng buộc cứng** khi dò tham số độ mới.

### 3.5. Tập đánh giá lớp RAG

- **Bẫy DEV** (`data/eval/rag/dev_traps.json`): 6 câu, trong đó 5 bẫy thật và 1 câu hóa ra hợp
  lệ (xem mục 7.14).
- **Bẫy TEST** (`data/eval/rag/test_traps.json`): **11 câu mới**, viết và **chốt trước khi chạy**,
  mỗi chi tiết đối chiếu với toàn văn bài báo. Phân loại: 2 giả định sai có con số, 1 con số sai
  nhưng con số đó có ở chỗ khác trong bài (điểm mù đã biết), 6 giả định sai không có con số, 2
  câu hỏi chi tiết bài không có.

---

## 4. Kiến trúc hệ thống

### 4.1. Luồng xử lý một lượt chat

```text
                               câu người dùng
                                     |
       [0] normalizer.prepare_user_text   chuẩn hóa teencode (từ điển học từ ViLexNorm),
           |                              gộp ký tự lặp; câu gốc không dấu -> bỏ dấu lại
       [1] entities.extract               chuyên mục + NER (underthesea) + Regex
           |
       [2] IntentClassifier.predict       TF-IDF (1,2) + Naive Bayes; w_nb = 1.0
           |
           +-- conf >= 0.25 và action != retrieve  -->  thực thi action
           |       (reply / help / list_categories / stats / browse_category / summarize / source)
           |
           +-- ngược lại --> [3] NewsRetriever.search
                              - câu có dấu -> index chính; không dấu -> index âm tiết
                              - lọc từ khung câu hỏi (QUERY_FRAME_WORDS)
                              - XẾP HẠNG:  cosine TF-IDF x (1 + 0,6 · recency)
                              - CHẤP NHẬN: cosine thuần >= 0,13
                                           (x 0,6 nếu người dùng đã nêu chuyên mục)
                              - tầng 2: chọn 2 câu sát nhất trong bài
                                   |                     |
                               đạt ngưỡng           dưới ngưỡng
                                   v                     v
                       trả lời + ngày đăng + nguồn    fallback
                                   |
                              DialogueState (bài vừa nhắc, chuyên mục, số lần fallback)

   Lớp tùy chọn, mặc định tắt: rag.RagChatbot bọc toàn bộ luồng trên và chỉ chạy khi [3]
   đã có bài vượt ngưỡng: chốt chặn giả định -> PhoGPT -> làm sạch -> chốt chặn rỗng/lặp lại/số bịa.
```

### 4.2. Các mô-đun

| Tệp | Trách nhiệm | Kế thừa lab |
|---|---|---|
| `config.py` | Đường dẫn, siêu tham số (dò trên DEV), từ khung câu hỏi | — |
| `preprocess.py` | NFC, tách từ (có cache), stopword, bỏ dấu, hạ về âm tiết | Lab 03 |
| `normalizer.py` | Học từ điển teencode; `prepare_user_text` dùng chung cho bot và đánh giá | — |
| `vectorizer.py` | BoW, n-gram, TF-IDF, L2, cosine, BM25 — tự cài đặt | Lab 04 |
| `intent_classifier.py` | Multinomial NB tự cài đặt + tín hiệu cosine tới pattern | TF-IDF + cosine: Lab 04; **NB: ngoài phạm vi lab** |
| `retriever.py` | Truy hồi hai tầng, hai index, độ mới, cache đĩa, giải thích | Lab 01 + 04 |
| `dates.py` | Đọc ngày đăng VnExpress, tính điểm độ mới | — |
| `entities.py` | NER + Regex, nhận diện chuyên mục | Lab 01 |
| `dialogue.py` | Trạng thái hội thoại, giải tham chiếu "bài đó" | — |
| `chatbot.py` | Điều phối | — |
| `crawler.py` | Thu thập VnExpress | Lab 02 |
| `generator.py` | n-gram LM — thí nghiệm đối chứng, không dùng trong bot | — |
| `evaluate.py` | Dò trên DEV, báo cáo trên TEST, Wilson CI, kiểm định dấu | — |
| `rag.py` | Lớp RAG tùy chọn | — |
| `cli.py`, `api.py`, `web/` | Giao diện dòng lệnh, FastAPI + trang chat | — |

### 4.3. Sáu quyết định thiết kế

1. **Mọi đường đi đều có ngưỡng tin cậy.** Lỗi tệ nhất của chatbot truy hồi là trả về một bài
   ngẫu nhiên bằng giọng chắc chắn. Dưới ngưỡng thì bot nói thẳng là không biết.
2. **Xếp hạng và chấp nhận dùng hai thước đo khác nhau.** "Bài nào đứng trước?" dùng
   cosine × độ mới; "có đủ căn cứ trả lời không?" chỉ dùng cosine thuần (mục 7.10).
3. **Hai cấu hình tiền xử lý:** intent **giữ** stopword (câu 5–10 token, "là gì", "có không"
   chính là tín hiệu), truy hồi **bỏ** stopword (bài dài, stopword làm loãng vector).
4. **Hai index tách biệt** cho câu có dấu và không dấu, để không pha loãng index chính và không
   làm mất hiệu lực các ngưỡng đã dò.
5. **Một quy tắc nghiệp vụ chỉ nằm ở một chỗ** (bài học từ hai lỗi thật, mục 7.6 và 7.10).
6. **LLM là lớp tùy chọn**, không mô-đun lõi nào phụ thuộc vào nó.

---

## 5. Phương pháp chi tiết

### 5.1. Tiền xử lý

`preprocess_vi(text, config)` giữ nguyên chữ ký hàm của Lab 03, gồm
`normalize_basic → segment_vi → lowercase → remove_numbers → remove_punctuation →
remove_stopwords`. `segment_vi` được bọc `lru_cache` vì chatbot gọi tách từ nhiều lần trên cùng
một câu. Hai cấu hình được dùng:

| | `CONFIG_INTENT` | `CONFIG_RETRIEVAL` |
|---|---|---|
| `word_segment`, `lowercase`, `remove_punctuation` | True | True |
| `remove_stopwords` | **False** | **True** |
| `remove_numbers` | False | False |

`remove_punctuation` mặc định giữ dấu nằm **giữa** chữ/số (`keep_inner=True`) để không phá các
cụm như `TP.HCM`, `3,5%`, `C++`; chế độ bỏ hết dấu câu của Lab 03 được giữ lại để so sánh.

### 5.2. TF-IDF tự cài đặt

`vectorizer.py` cài `CountVectorizer` và `TfidfVectorizer` bằng NumPy + SciPy (ma trận thưa CSR):

- Vocabulary sắp theo thứ tự chữ cái để chỉ số ổn định giữa các lần chạy; lọc `min_df`, `max_df`.
- `idf = ln((1+N)/(1+df)) + 1` (smooth), tf tùy chọn sublinear `1 + ln tf`.
- Nhân idf theo phần tử của `X.data` qua `X.indices`, không dựng ma trận chéo.
- Chuẩn hóa L2 theo hàng; hàng toàn 0 giữ nguyên để tránh chia cho 0.
- `top_terms` và `_matched_terms` phục vụ **giải thích**: term nào khớp, đóng góp bao nhiêu điểm.

**Kiểm chứng:** `tests/test_vectorizer.py` đối chiếu với `sklearn.feature_extraction.text` trên
5 cấu hình (unigram/bigram, sublinear, smooth, min_df/max_df...) và các trường hợp biên — 28
kiểm tra đều khớp với sai số tuyệt đối cỡ $10^{-16}$.

### 5.3. Truy hồi hai tầng

**Tầng 1 — chọn bài.** Mỗi bài được ghép thành một văn bản index gồm tiêu đề lặp 3 lần, mô tả lặp
2 lần và phần thân (`TITLE_WEIGHT = 3`, `DESC_WEIGHT = 2`) — tăng trọng số trường quan trọng mà
không sửa công thức. Index dùng unigram + bigram, `min_df = 1`, `max_df = 0,85`, sublinear tf;
vocabulary chính có **105.020 term**.

**Tầng 2 — chọn câu.** Trong bài được chọn, tách câu bằng `sent_tokenize` (Lab 01), bỏ câu ngắn
hơn 30 hoặc dài hơn 400 ký tự (chú thích ảnh, tên tác giả), **khử trùng lặp** (VnExpress thường
lặp câu sapo trong thân bài), rồi dựng một TF-IDF **cục bộ cho riêng bài đó** và chọn 2 câu có
cosine cao nhất với câu hỏi, ghép lại **theo thứ tự xuất hiện**.

**Giải thích.** `--explain` (CLI) và `/explain` (API) trả về term khớp và điểm đóng góp của từng
term, term OOV, và các ứng viên bị loại.

### 5.4. Câu không dấu: index âm tiết

`word_tokenize` được huấn luyện trên văn bản có dấu nên tách sai văn bản không dấu:
`word_tokenize("tin ve dao hai nam") → ['ve_dao', 'hai', 'nam']`. Thay vì khôi phục dấu (bài toán
seq2seq riêng), hệ thống hạ **cả hai phía** về cùng mặt phẳng âm tiết không dấu:

- Phía tài liệu: tách từ trên bản **có dấu** (chính xác), rồi mới bỏ dấu và tách từ ghép ra âm tiết
  (`["đảo","hải_nam"] → ["dao","hai","nam"]`).
- Phía câu hỏi: bỏ qua tách từ, chỉ cắt theo khoảng trắng.
- Bigram khôi phục phần lớn thông tin từ ghép (`"hai nam"` xuất hiện như một bigram).

Index âm tiết (87.294 term) được giữ **tách biệt** với index chính và chỉ được chọn khi câu hỏi
**không có dấu nào** (`has_diacritics`). Intent classifier cũng có một mô hình phụ tương tự.

### 5.5. Chuẩn hóa teencode

**Học từ điển** (`normalizer.learn_lexicon`) trên split train của ViLexNorm:

1. Chỉ dùng các cặp câu có **cùng số token** (79,5% số cặp) để căn theo vị trí.
2. Đếm mọi ánh xạ `a → b` tại vị trí có thay đổi.
3. Giữ `a → b` khi thỏa **cả ba** điều kiện:

| Điều kiện | Ngưỡng | Lý do |
|---|---|---|
| `a` xuất hiện đủ nhiều | `count ≥ 4` | tránh học từ nhiễu gán nhãn |
| `a` **thường** bị đổi | tỷ lệ đổi ≥ 0,5 | chốt chặn quan trọng nhất: thiếu nó, từ thường như "cả", "mà" bị thay bừa |
| `b` chiếm ưu thế | ≥ 0,5 | tránh chọn bừa khi `a` mơ hồ |

Kết quả là **342 ánh xạ** (ví dụ `bt/bik/bít → biết`, `b/bn → bạn`, `k → không`). Nhập nhằng thật
được ghi nhận: `t → tôi` (889 lần) và `t → tao` (132 lần) — chọn `tôi`, chấp nhận sai với câu vốn
nói "tao".

**Áp dụng.** `TeencodeNormalizer` chạy **trước** tách từ: tra từ điển, rồi gộp ký tự lặp từ 3 lần
trở lên (`"khummm" → "khum" → "không"`). Gộp ký tự **chỉ áp cho chữ cái** — bản đầu gộp cả chữ số
và làm hỏng con số (mục 7.14).

**Giữ hệ quy chiếu dấu** (`prepare_user_text`). Từ điển luôn trả từ **có dấu**; nếu câu gốc không
dấu mà giữ nguyên kết quả, chỉ một token được sửa cũng làm cả câu bị định tuyến sang index có dấu,
nơi các token còn lại đều là OOV. Vì vậy: nếu câu gốc không có dấu, bỏ dấu lại sau khi chuẩn hóa.
Hàm này được dùng **chung** cho chatbot và `evaluate.py`, để số liệu đánh giá phản ánh đúng thứ
người dùng nhận được.

**Lọc từ khung câu hỏi.** Chuẩn hóa *bung* từ viết tắt thành từ đầy đủ, vô tình thêm token khung
("biết", "không", "bạn") làm loãng vector câu hỏi đã chuẩn hóa L2: `"giá iphone"` đạt 0,167 nhưng
`"biết gì về vụ iphone không bạn"` chỉ 0,100. `QUERY_FRAME_WORDS` (52 từ, định nghĩa **duy nhất**
trong `config.py`) được lọc khỏi câu hỏi trước khi truy hồi; nếu lọc hết thì giữ nguyên bản.

### 5.6. Phân loại ý định

Đặc trưng: TF-IDF unigram + bigram, sublinear tf, không bỏ stopword (vocabulary 493 term). Mô hình:
Multinomial NB với $\alpha = 0{,}3$ (nhỏ hơn 1 vì tập train nhỏ và các intent tách biệt rõ).

Thiết kế ban đầu là **ensemble** hai tín hiệu bù trừ:

$$\mathrm{score}(c) = w_{nb}\cdot P_{NB}(c\mid x) + (1-w_{nb})\cdot\max_{p\in c}\cos(\vec x, \vec p)$$

vì NB cho phân phối **phẳng** với câu 2–3 token ("cảm ơn nhé" chỉ đạt 0,189 dù ý định hiển nhiên),
còn cosine tới pattern gần nhất rất nhạy với câu ngắn ("bye" → 1,000). Khi dò lại trên DEV, trọng
số tốt nhất là $w_{nb} = 1{,}0$ — tức **NB thuần**; tín hiệu cosine được giữ trong mã để thí nghiệm
tái lập được. Ngưỡng chấp nhận intent: 0,25.

### 5.7. Thực thể và trạng thái hội thoại

`entities.py` áp nguyên tắc của Lab 01: **định dạng cố định → Regex**, **thực thể mở → NER**.

- Regex: `email`, `phone` (`(0|+84)` + 8–10 chữ số), `url`, `date` (`12/8/2026`, `2/9`), `quantity`
  (số + đơn vị: %, tỷ, triệu, nghìn, đồng, USD, ngày, giờ, tháng, năm, km, kg, người). Pattern
  `quantity` kết thúc bằng `(?!\w)` thay vì `\b` — vì sau `%` không tồn tại ranh giới từ, `\b` bỏ
  sót `3,5%` (một lỗi thật, có kiểm thử).
- NER: `underthesea.ner` với nhãn BIO, ghép các token B-/I- thành thực thể PER/LOC/ORG.
- Chuyên mục: khớp từ khóa **không phân biệt dấu**, ưu tiên cụm dài nhất.

`DialogueState` giữ bài vừa nhắc (`last_results`), chuyên mục đang xem (`last_category`) và số
lần fallback liên tiếp (gợi ý trợ giúp khi người dùng bế tắc). Đây là dạng đơn giản nhất của giải
tham chiếu: "tóm tắt bài đó", "cho mình link" neo vào lượt gần nhất. API tách `DialogueState`
theo `session_id`.

Khi người dùng chỉ nêu tên chuyên mục ("tin sức khỏe"), bot duyệt mục (bài mới nhất trước); nếu
kèm chủ đề ("tin sức khỏe về ăn chuối"), bot tìm trong mục với ngưỡng nới
$0{,}13 \times 0{,}6$ — tập ứng viên đã co lại nên cosine thấp hơn vẫn là bằng chứng đủ mạnh. Việc
phát hiện "có chủ đề ngoài tên chuyên mục" so khớp ở mức **âm tiết** (mục 7.6).

### 5.8. Xếp hạng theo độ mới

$$\mathrm{recency}(d) = 0{,}5^{\,\mathrm{age}(d)/h}, \qquad \mathrm{score}'(q,d) = \mathrm{rank}(q,d)\cdot\big(1 + \alpha\cdot\mathrm{recency}(d)\big)$$

với $h$ là nửa chu kỳ (ngày) và $\alpha$ nhỏ. Các lựa chọn và lý do:

- **Nhân chứ không cộng:** bài không liên quan (điểm ≈ 0) vẫn ≈ 0 dù mới tinh; phép nhân không phụ
  thuộc thang đo nên dùng được cho cả BM25.
- **$\alpha$ nhỏ:** mục tiêu là **phá thế hòa**, không phải luôn ưu tiên tin mới.
- **Mốc tham chiếu là ngày đăng mới nhất trong kho**, không phải `datetime.now()`, để kết quả tái
  lập được. Bài thiếu ngày nhận recency trung vị.
- **Ngưỡng chấp nhận áp lên cosine thuần**, không lên điểm đã nhân độ mới.
- Mọi câu trả lời hiển thị **ngày đăng**, và duyệt chuyên mục sắp theo ngày giảm dần.

Giá trị hiện hành (dò trên DEV): $h = 3$ ngày, $\alpha = 0{,}6$.

### 5.9. Cache index ra đĩa

Dựng index tốn khoảng 24 giây, gần như toàn bộ là `word_tokenize`. Index được lưu bằng `joblib`
kèm **vân tay SHA-256** của nội dung kho và mọi tham số ảnh hưởng tới index (`ngram_range`,
`min_df`, `max_df`, `sublinear_tf`, trọng số trường, cấu hình tiền xử lý). Khởi động có cache:
**~1,1 giây**. Điểm độ mới luôn được **tính lại** sau khi nạp cache, vì nửa chu kỳ không nằm trong
vân tay (mục 7.9). Ma trận đếm cũng được cache nên đổi $(k_1, b)$ của BM25 không cần tách từ lại.

### 5.10. BM25

`bm25_idf`, `bm25_weights`, `bm25_scores` tính sẵn ma trận trọng số $W$ cho cả kho sao cho
$\mathrm{score}(q,d) = W_d\cdot\vec q_{\text{đếm}}$; mỗi truy vấn chỉ là một phép nhân ma trận thưa.
Vì sklearn không có BM25, bản vector hóa được đối chiếu với **một bản cài đặt ngây thơ viết thẳng
từ định nghĩa** (vòng lặp) trên 4 cặp $(k_1, b)$, cộng một kiểm thử tính chất (lặp một term thì
điểm tăng nhưng tiến tới trần $\mathrm{idf}\cdot(k_1+1)$) — 17 kiểm tra. BM25 chỉ dùng để **xếp
hạng**; chấp nhận vẫn dựa trên cosine TF-IDF vì điểm BM25 không bị chặn và không so được giữa hai
câu hỏi.

### 5.11. Thí nghiệm đối chứng: mô hình sinh n-gram

`generator.py` cài n-gram LM với nội suy đệ quy, huấn luyện trên câu tách từ kho bài báo (9.567 câu
huấn luyện, 1.063 câu kiểm thử, 216.976 token), đo perplexity với $n = 1..4$ và quét $\lambda$. Mục
đích: trả lời bằng số liệu câu hỏi "sao không để mô hình tự sinh câu trả lời?".

### 5.12. Lớp RAG với PhoGPT-4B-Chat

**Ghép ngữ cảnh.** Lấy tối đa 3 bài đã vượt ngưỡng; mỗi bài lấy **4 câu** liên quan nhất (tái dùng
tầng 2), cắt tối đa 900 ký tự, kèm ngày đăng.

**Hai phiên bản prompt** (toàn văn ở Phụ lục B):

| | v1 | v2 |
|---|---|---|
| Hướng dẫn | 6 quy tắc **đánh số** | **một đoạn văn xuôi** |
| Trích dẫn | yêu cầu `[i]` | bỏ; nguồn gắn bằng code |
| Ngữ cảnh | `[i] (đăng dd/mm/yyyy, chuyên mục X)` | `Tin i (ngày dd/mm/yyyy): tiêu đề` |
| Câu từ khóa | đưa nguyên | chuyển thành "Các tin trên cho biết gì về X?" |
| Hậu xử lý | cắt ở `###` | bỏ dòng chép prompt, bỏ ký hiệu danh sách, bỏ nhãn "Tin N:", lấy đoạn đầu, bỏ câu lặp, tối đa 4 câu, đổi mở đầu thành "Theo các bài báo," |
| Chốt chặn | không | có (dưới đây) |
| Số token tối đa | 256 | 160 |

**Chốt chặn tất định** (chỉ ở v2, đều là quy tắc regex/so khớp nên kiểm thử được không cần GPU):

| Chốt chặn | Thời điểm | Quy tắc | Hành động |
|---|---|---|---|
| Giả định | trước LLM | con số hoặc mã hiệu (chữ + số, như `IP68`) trong câu hỏi **không xuất hiện ở bất kỳ đâu** trong toàn văn các bài; số có đơn vị ("5 triệu", "100 nghìn") so theo **giá trị** | không hiển thị câu sinh; trả lời "các bài báo không nhắc tới «…»" + đoạn trích |
| Rỗng | sau LLM | câu sinh rỗng sau làm sạch | hiển thị đoạn trích |
| Lặp lại | sau LLM | chỉ với câu dạng từ khóa: câu sinh không có âm tiết nội dung nào ngoài câu hỏi | hiển thị đoạn trích |
| Số bịa | sau LLM | câu sinh chứa số không có trong nguồn, câu hỏi hay ngày đăng (bỏ qua số thứ tự đầu dòng; "9h40" ≡ "9 giờ 40") | hiển thị đoạn trích |

**Backend.** `LlamaCppBackend` (GGUF qua llama.cpp trên GPU; Q4_K_M ở lần 1, Q8_0 ở lần 2–3),
`TransformersBackend` (float16; không nạp được trên Colab do mã tùy biến của PhoGPT khai báo gói
`triton_pre_mlir`), và `EchoBackend` (giả lập, dùng cho kiểm thử). Giải mã tham lam (temperature 0),
`repeat_penalty = 1,1`.

**Nguyên tắc an toàn:** LLM chỉ được gọi khi truy hồi đã có bằng chứng vượt ngưỡng; mọi đường khác
(intent, fallback) giữ nguyên câu trả lời trích xuất.

---

## 6. Phương pháp đánh giá

### 6.1. Chỉ số

| Thành phần | Chỉ số |
|---|---|
| Truy hồi | Recall@1, Recall@3, MRR (bài đúng theo URL) |
| Intent | Accuracy, macro-F1 (câu phải đúng nhãn **và** vượt ngưỡng) |
| Ngoài phạm vi | tỷ lệ bị chặn (thành phần) và bị từ chối (đầu-cuối) |
| Đầu-cuối | gọi `bot.respond()`: câu tin tức → bài đứng đầu đúng; câu ngoài phạm vi → bot từ chối |
| Teencode | Accuracy trước/sau, ERR, precision, recall, F1 (token) |
| Sinh văn bản | perplexity |
| RAG | chấm tay thang A–E; an toàn trên bẫy; tự động: số không có trong nguồn, tỷ lệ từ có trong nguồn, trích dẫn hợp lệ |

### 6.2. Khoảng tin cậy Wilson 95%

Với tập vài chục câu, một câu sai đã làm tỷ lệ dao động vài điểm phần trăm. Mọi tỷ lệ kèm KTC Wilson:

$$\frac{\hat p + \frac{z^2}{2n}}{1 + \frac{z^2}{n}} \pm \frac{z}{1+\frac{z^2}{n}}\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}, \qquad z = 1{,}96$$

### 6.3. Kiểm định dấu có cặp

Để quyết định có dùng phương pháp phức tạp hơn không, so từng câu hỏi, bỏ câu hòa, và tính p-value
hai phía của phân phối nhị thức chính xác với $p = 0{,}5$:

$$p = \min\!\Big(1,\; 2\sum_{i=0}^{\min(W,L)}\binom{W+L}{i}\,2^{-(W+L)}\Big)$$

**Quy tắc:** phương pháp phức tạp hơn phải thắng **có ý nghĩa thống kê** trên DEV.

### 6.4. Quy trình DEV/TEST

- **Pha 1 — dò trên DEV** (`evaluate.py`): intent (lưới $w_{nb}$ × ngưỡng) → cách xếp hạng (TF-IDF
  hay BM25, $k_1$, $b$; độ mới **tắt**) → độ mới (lưới $h$ × $\alpha$, **ràng buộc cứng**: ca tin
  mâu thuẫn đúng ở cả 4 cách hỏi, rồi chọn MRR cao nhất) → ngưỡng truy hồi (lưới bước 0,005).
- **Pha 2 — báo cáo trên TEST** với đúng bộ tham số đã chốt, không chỉnh gì sau khi xem.
- Mọi truy vấn đi qua `prepare_user_text` giống hệt chatbot.
- Mọi lần xem tập TEST được **ghi nhật ký** (Phụ lục C).

### 6.5. Thang chấm tay cho câu trả lời RAG

| Nhãn | Nghĩa |
|---|---|
| **A** | đúng nguồn và trả lời được câu hỏi |
| **B** | đúng nguồn nhưng trình bày hỏng (danh sách, lặp câu, đuôi chép prompt) |
| **C** | không trả lời (lặp lại câu hỏi / tự đặt câu hỏi) |
| **D** | có thông tin **sai** so với bài gốc |
| **E** | từ chối dù bài có câu trả lời |

Mỗi khẳng định được đối chiếu với **toàn văn** bài gốc. Chấm trên câu PhoGPT sinh, **trước** chốt
chặn. Nhãn lần chạy 3 lưu ở `data/eval/rag/run3_annotation.csv` và `run3_trap_annotation.csv`.

---

## 7. Tiến trình phát triển theo giai đoạn

Mục này kể lại quá trình theo thứ tự thời gian, dựa trên lịch sử commit. Điều đáng chú ý là
**nhiều con số đã thay đổi** giữa các giai đoạn — có lúc tăng nhờ cải tiến, có lúc giảm vì phương
pháp đo được sửa cho đúng. Cả hai loại thay đổi đều được ghi lại.

### Tổng quan

| Giai đoạn | Commit | Nội dung | Notebook báo cáo |
|---|---|---|---|
| 1 | `190f608` | Lõi NLP: tiền xử lý, TF-IDF, NB, truy hồi, crawler, đánh giá | — |
| 2 | `16ff385` | CLI, web, câu không dấu | — |
| 3 | `f45118f`, `bdd5050`, `ad9bd81` | Notebook đầu tiên, docs 01–02, khử trùng lặp snippet | **45 ô**, 7 case lỗi |
| 4 | `2f4d3de` | Chuẩn hóa teencode + lọc từ khung | — |
| 5 | `1b7f8a2` | Thí nghiệm sinh n-gram, docs 03–04 | **61 ô**, 12 case |
| 6 | `5b90008`, `72f3f1a` | Độ mới + cache, docs 05, sửa một so sánh sai | **72 ô**, 15 case |
| 7 | `c3da7c2` | **Tách DEV/TEST** — sửa rò rỉ tập test | — |
| 8 | `c4a6a23` | 21 kiểm thử hồi quy + sửa 2 lỗi thiết kế | — |
| 9 | `6587696` | BM25 (kết quả âm) | — |
| 10 | `5320448` | docs 06, notebook viết lại phần đánh giá | **79 ô**, 19 case |
| 11 | `98fb144` | Lớp RAG + notebook Colab | — |
| 12 | `1743132` | Phân tích RAG lần 1, prompt v2, sửa lỗi gộp chữ số | — |
| 13 | `fe10190` | Phân tích RAG lần 2, chốt bẫy test, **đóng băng mã** | — |
| 14 | `be306ce`, `40c1792` | Kết quả RAG lần 3 trên test, Phần K của notebook | **86 ô** |
| 15 | (commit kiểm tra tài liệu) | Rà soát toàn bộ tài liệu, viết báo cáo này | 86 ô |

### 7.1. Giai đoạn 1 — Lõi NLP (`190f608`)

Cài đặt `preprocess`, `vectorizer`, `intent_classifier`, `retriever`, `entities`, `dialogue`,
`chatbot`, `crawler`, `evaluate`. Kho được mở rộng từ 28 lên 381 bài. Số liệu **lúc đó** (đo trên
một tập test viết tay, về sau phát hiện bị dùng để dò tham số):

| | Giá trị báo cáo lúc đó |
|---|---|
| TF-IDF khớp sklearn | 28/28, sai số ~1e-16 |
| Intent accuracy / macro-F1 | 88,5% / 0,91 |
| Truy hồi Recall@1 / Recall@3 / MRR | 85,7% / 95,2% / 0,905 |

### 7.2. Giai đoạn 2 — Giao diện và câu không dấu (`16ff385`)

Thêm `cli.py` (có `/debug`, `/explain`), `api.py` (FastAPI, tách trạng thái theo phiên), trang chat.
Phát hiện `word_tokenize` tách sai câu không dấu → cài index âm tiết (mục 5.4). Trên cùng tập 21 câu:
Recall@1 85,7% → **90,5%**, Recall@3 95,2% → **100%**, MRR 0,905 → **0,952** — do các truy vấn tên
riêng nước ngoài ("champions league man utd") vốn không dấu nay khớp đúng.

### 7.3. Giai đoạn 3 — Notebook đầu tiên (`f45118f`, `bdd5050`, `ad9bd81`)

Notebook báo cáo **45 ô, 0 lỗi, 2 biểu đồ, 7 case phân tích lỗi**; tài liệu docs/01 (khảo sát) và
docs/02 (kiến trúc); `build_notebook.py` chuyển vào `tools/` để được quản lý phiên bản. Sửa lỗi
**snippet lặp câu**: VnExpress lặp câu sapo trong thân bài nên hai câu điểm cao nhất có thể là một →
khử trùng lặp khi tách câu. Không ảnh hưởng chỉ số.

### 7.4. Giai đoạn 4 — Teencode (`2f4d3de`)

Cài từ điển teencode học từ ViLexNorm (mục 5.5). Kết quả lúc đó trên split test: accuracy 83,88% →
94,77%, **ERR 67,54%**, precision/recall 90,71% / 70,16%. Phát hiện chuẩn hóa làm **loãng** vector
câu hỏi → thêm `QUERY_FRAME_WORDS`. Hai lỗi phát hiện khi kiểm thử:

1. Hai danh sách từ khung ở `chatbot.py` và `retriever.py` **lệch nhau** ("biết" chỉ có ở một bên) →
   gộp về một nguồn trong `config.py`.
2. `word_tokenize("Sức khỏe") → ["sức","khỏe"]` nhưng `word_tokenize("... tin sức khỏe gì luôn") →
   ["sức_khỏe"]` → so khớp tên chuyên mục chuyển sang mức âm tiết.

### 7.5. Giai đoạn 5 — Thí nghiệm sinh văn bản (`1b7f8a2`)

Cài n-gram LM (mục 5.11). Ba phát hiện (chi tiết mục 8.7): perplexity **tăng** theo $n$; $n$ lớn thì
"sinh" thành "chép"; văn bản sinh sai sự thật ở mọi $n$. Một nhận xét viết sẵn trong mã ("n tăng →
perplexity giảm") bị chính số liệu bác bỏ và được sửa lại. Notebook: **61 ô**, thêm Phần G2 (teencode)
và G3 (sinh văn bản); phân tích lỗi 7 → **12 case**; docs/03 và docs/04.

### 7.6. Giai đoạn 6 — Tin lỗi thời và cache (`5b90008`, `72f3f1a`)

Dựng ca tin mâu thuẫn: TF-IDF thuần xếp bài **cũ** lên đầu (0,4956 so với 0,4098) và kết quả **đảo
lộn tùy cách hỏi** (2/4 cách hỏi trả tin cũ). Nguyên nhân: `published_at` được thu thập nhưng chưa
từng được dùng. Thêm xếp hạng độ mới (mục 5.8), luôn hiển thị ngày đăng, sắp duyệt mục theo ngày. Dò
lưới lúc đó chọn nửa chu kỳ **7 ngày**, $\alpha = 0{,}6$.

Hai lỗi phát hiện khi kiểm thử:

1. **Tập dò ngưỡng không đại diện:** chỉ có câu viết chuẩn, trong khi bot đã xử lý không dấu và
   teencode → bổ sung 10 truy vấn + 4 câu ngoài phạm vi, dò lại ngưỡng (lúc đó 0,12 → 0,155, trên
   điểm đã nhân độ mới).
2. **Chuẩn hóa phá định tuyến index:** `"thoi tiet sao hoa hom nay"` → `"thôi tiet sao hoa hom nay"`
   (giờ "có dấu") → intent `tam_biet` (0,280) → bot trả lời **"Tạm biệt bạn!"** cho câu hỏi thời tiết
   → thêm cơ chế giữ hệ quy chiếu dấu.

Cache index: khởi động 24,2 s → **1,1 s**. Số liệu báo cáo lúc đó: Recall@1 90,5% → 96,8%, MRR 0,984.

**Sửa một so sánh sai ngay trong giai đoạn** (`72f3f1a`): con số "90,5% → 96,8%" so hai tập test
**khác nhau** (21 và 31 câu). So đúng trên cùng 31 câu, chỉ bật/tắt độ mới: **93,5% → 96,8%**. Notebook
**72 ô**, thêm Phần G4; phân tích lỗi 12 → **15 case**; docs/05.

### 7.7. Giai đoạn 7 — Phát hiện và sửa rò rỉ tập test (`c3da7c2`)

**Lỗi phương pháp:** mọi siêu tham số ($w_{nb}$, ngưỡng intent, ngưỡng truy hồi, $\alpha$, nửa chu kỳ)
đều được chọn bằng cách quét trên `test_queries.json`, rồi con số báo cáo lại đo trên **chính tập đó**.
Hơn nữa tập quá nhỏ (31 truy vấn, 26 câu intent): một câu sai làm Recall@1 dao động 3,2 điểm.

Sửa bằng quy trình mục 6.4 và bộ dữ liệu mục 3.4. **Báo cáo test lần 1:**

| | Trước (rò rỉ) | Thật trên TEST |
|---|---|---|
| Truy hồi Recall@1 | 96,8% | **93,4%** (114/122) |
| Truy hồi MRR | 0,984 | **0,950** |
| Intent accuracy | 88,5% | **61,5%** (16/26) |
| Đầu-cuối: tin tức → đúng bài | — | 77,9% (95/122) |
| Đầu-cuối: ngoài phạm vi → từ chối | — | 83,3% (20/24) |

Hai kết luận cũ **bị bác bỏ**: (1) "độ mới làm tốt lên truy hồi" — trên DEV 112 câu, độ mới không
cải thiện nhất quán, mức tăng trước đó là nhiễu của tập nhỏ; (2) ensemble NB + cosine không còn
thắng — DEV chọn $w_{nb} = 1{,}0$. Một lỗi tự gây ra cũng được sửa: cache lưu sẵn điểm độ mới nhưng
nửa chu kỳ không nằm trong vân tay → đổi nửa chu kỳ sẽ bị bỏ qua âm thầm; nay luôn tính lại recency.

### 7.8. Giai đoạn 8 — Kiểm thử hồi quy bắt được hai lỗi thiết kế (`c4a6a23`)

21 kiểm thử (danh sách ở Phụ lục F), mỗi cái ứng với một lỗi thật từng làm bot trả lời sai mà không
báo lỗi. Ngay lần chạy đầu sau khi áp tham số mới dò trên DEV, chúng bắt được:

1. **Độ mới biến thành bộ lọc loại bài cũ.** Ngưỡng áp lên điểm **đã nhân** độ mới; với nửa chu kỳ
   3 ngày, bài 16 ngày tuổi gần như không được thưởng nên khó vượt ngưỡng: `"tin ve dao hai nam"`
   và `"cho t hỏi vụ hải nam vs"` từ đúng thành "không tìm thấy". **Sửa:** xếp hạng theo điểm có độ
   mới, **chấp nhận theo cosine thuần**; dò lại ngưỡng trên thang cosine: **0,13** (DEV: trả lời được
   88,7%, chặn đúng 100%).
2. **Cùng một câu, kết quả tùy nhánh.** Ngưỡng nới khi đã nêu chuyên mục chỉ có ở nhánh duyệt mục;
   `"tin du lịch ninh bình"` có độ tin cậy intent 0,246 — dưới 0,25 một chút — nên đi nhánh truy hồi
   (không có ngưỡng nới) và trượt. **Sửa:** một hằng số duy nhất `CATEGORY_SCOPED_THRESHOLD_FACTOR`
   áp ở cả hai nhánh.

**Báo cáo test lần 2:** Recall@1 93,4% (không đổi), MRR 0,950; tỷ lệ chặn câu ngoài phạm vi (thành
phần) 95,8% → **87,5%** vì ngưỡng trên cosine thuần để lọt 3 câu "khó"; đầu-cuối **77,0%** / **75,0%**.
Hai sửa đổi xuất phát từ kiểm thử hồi quy, không từ kết quả test.

### 7.9. Giai đoạn 9 — BM25, một kết quả âm (`6587696`)

Cài và kiểm chứng BM25 (mục 5.10); `test_vectorizer.py` 28 → **45** kiểm tra. Trên DEV (độ mới tắt),
BM25 tốt nhất ($k_1 = 8$, $b = 0{,}9$) đạt MRR 0,967 so với TF-IDF 0,965 — chênh khoảng một câu.
Quy tắc ban đầu "hơn trên DEV là đổi" đã chọn BM25, rồi BM25 **thua trên TEST** (MRR 0,943 so với
0,950). Quy tắc được sửa thành kiểm định dấu có cặp: BM25 tốt hơn 3 câu, kém hơn 3 câu, hòa 106 câu,
**p = 1,0** → giữ TF-IDF. **Minh bạch:** quy tắc kiểm định được thêm **sau khi** đã thấy kết quả test
lần 3; dưới cả hai quy tắc, trên test hai phương pháp không khác nhau có ý nghĩa.

### 7.10. Giai đoạn 10 — Tài liệu đánh giá trung thực (`5320448`)

docs/06 ghi toàn bộ phương pháp mới; docs/03, docs/05 thêm khung **đính chính** (giữ nội dung cũ để
thấy lịch sử); README thay bằng số liệu TEST có KTC. Notebook viết lại thành **79 ô**: Phần I "Đánh
giá trung thực", Phần I2 (BM25), G4 dùng tập DEV, phân tích lỗi 15 → **19 case**. Sửa thêm một khẳng
định sai chính mình viết trong đợt này ("tắt độ mới cho MRR cao nhất trên dev"): output notebook cho
thấy cấu hình 30 ngày / $\alpha$ = 0,2 đạt 0,969 > 0,965 — phát biểu đúng là độ mới không cải thiện
nhất quán, mọi chênh lệch trong phạm vi nhiễu.

### 7.11. Giai đoạn 11 — Lớp RAG (`98fb144`)

Cài `rag.py` (prompt v1, ba backend, kiểm tra độ trung thành tự động), `tests/test_rag.py` (9 kiểm thử,
gồm một backend **ném lỗi nếu bị gọi** để chứng minh LLM không chạy khi không có bằng chứng), notebook
Colab 26 ô (tự nhận biết Colab/cục bộ, chạy thử cục bộ bằng backend giả lập) và công cụ đóng gói
(`rag_bundle.zip`, không kèm cache vì pickle phụ thuộc phiên bản). Máy cá nhân không có GPU NVIDIA
nên PhoGPT chạy trên Google Colab (T4), người thực hiện tự chạy notebook và gửi lại kết quả.

### 7.12. Giai đoạn 12 — RAG lần chạy 1 (DEV) và prompt v2 (`1743132`)

**Cấu hình:** Colab T4; `transformers` lỗi nạp → llama.cpp GGUF **Q4_K_M**; prompt v1; 25 câu tin
tức DEV, 4 câu mâu thuẫn, 6 bẫy, 5 câu ngoài phạm vi. **Kiểm tra tái lập:** định tuyến và bài top-1
cục bộ trùng **36/36** với Colab.

**Kết quả** (21 câu tin tức tới RAG): chỉ **3** câu A, **8** câu D, 0/21 trích dẫn đúng, **0/5** bẫy
xử lý đúng (mô hình trả lời "Đúng." cho giả định sai), mâu thuẫn 3/3 theo tin mới, ngoài phạm vi 5/5
không gọi LLM, độ trễ 1,84 s (p90 4,9 s). Bộ kiểm tra số bịa có lỗi của chính nó (đếm số thứ tự "1.",
"2." là số bịa): 12 → **6** câu sau khi sửa, cả 6 là bịa thật (ngày bịa `06/30/2021`, năm 2016, năm
2022). Chỉ 3/8 câu D có lỗi ở con số; 5/8 là lỗi ngữ nghĩa (đảo nhân quả, phủ định bịa, trộn hai bài,
bịa cho khớp câu hỏi khi truy hồi sai).

**Thiết kế v2** từ các lỗi quan sát được (mục 5.12). Hai đo đạc **không cần GPU**:

- Chốt chặn giả định chặn nhầm câu hợp lệ: DEV **0/93**, TEST **1/97** — lần duy nhất là câu truy hồi
  đã sai bài, tức **0** lần chặn nhầm câu truy hồi đúng.
- Áp hậu xử lý + chốt chặn v2 lên chính các câu v1 đã sinh: bẫy an toàn 0/5 → 5/5 (4 do thiết kế, 1
  do may), câu hiển thị còn số bịa 6/21 → 0/18, câu D còn hiển thị 8 → 5, câu C không đổi (3).

**Lỗi tìm ra ngoài lề:** khi đo chốt chặn, câu DEV "nhan vien openai tieu **7000** do..." bị báo nhầm
với «70». Truy ra bước gộp ký tự lặp dùng `(.)\1{2,}` — gộp **cả chữ số**: `15.000 → 15.0`,
`2000 → 20`, `1000 → 10`. Sửa thành chỉ gộp chữ cái `([^\W\d_])\1{2,}`. ERR teencode test 67,54% →
**67,84%**. Lỗi chạm **1/195** câu DEV và **0/178** câu TEST; dò lại trên DEV cho **đúng bộ tham số
cũ**, nên các số liệu test không đổi.

### 7.13. Giai đoạn 13 — RAG lần chạy 2 (DEV, v1 vs v2) (`fe10190`)

**Cấu hình:** Q8_0; cùng 40 câu DEV; mỗi câu chạy qua v1 và v2; khi chốt chặn giả định chặn, PhoGPT
**vẫn được gọi** để ghi lại nó định trả lời gì.

| Nhãn (21 câu tin tức) | v1 · Q4_K_M (lần 1) | v1 · Q8_0 | **v2 · Q8_0** |
|---|---|---|---|
| A | 3 | 4 | **10** |
| B | 5 | 3 | 3 |
| C | 3 | 4 | 3 |
| D | 8 | 7 | **5** |
| E | 2 | 3 | **0** |

- **Lượng tử hóa không phải nguyên nhân:** v1 trên Q4_K_M và Q8_0 cho phân bố lỗi gần như nhau.
- **v2 giúp thật:** 6 câu v2 đạt A mà v1 không, 0 câu ngược lại (p ≈ 0,03) — nhưng **lạc quan** vì v2
  được thiết kế trên chính các câu này.
- PhoGPT v2 **tự nó** chỉ xử lý đúng 1/5 bẫy; có chốt chặn: 5/5.
- **Đính chính:** câu "vì sao kem Tràng Tiền phải đóng cửa" từng bị ghi là bẫy, nhưng bài **có** nêu lý
  do ("hoàn thành sứ mệnh lịch sử" sau 68 năm). Lỗi do lúc kiểm tra chỉ tìm chuỗi thay vì đọc cả bài;
  bẫy DEV còn 5 câu thật. Bộ bẫy TEST vì vậy được đối chiếu với **toàn văn**.

**Sửa sau lần 2 (chỉ hậu xử lý, không đổi prompt):** bỏ nhãn "Tin N:" (làm chốt chặn số bịa bắt nhầm
câu đúng), bỏ dấu ```, đổi mở đầu thành "Theo các bài báo," (không viết thường danh từ riêng — lỗi
"tP HCM" đã gặp), thêm chốt chặn **lặp lại** chỉ cho câu dạng từ khóa (bản đầu áp cho mọi câu đã chặn
nhầm câu trả lời có/không "Vé tàu Cát Linh không tăng giá."), và so số có đơn vị theo giá trị. Chấm lại
đầu ra v2: 17/21 câu hiển thị câu sinh, còn 3 câu D.

**Chuẩn bị lần 3:** soạn 11 bẫy test mới, đối chiếu toàn văn; **đóng băng mã** ở commit `fe10190`. Khi
soạn bẫy phát hiện **4/11 cách hỏi tự nhiên không tới được RAG** ("...đúng không", "...phải không" bị
intent classifier xếp nhầm); 2 câu thử lại theo thứ tự biến thể cố định thì tới được, 2 câu không biến
thể nào tới được và được giữ nguyên với ghi chú. `test_rag.py`: 9 → 25 → **29** kiểm thử.

### 7.14. Giai đoạn 14 — RAG lần chạy 3 trên TEST và hoàn thiện notebook (`be306ce`, `40c1792`)

Chạy **một lần** trên TEST với mã đóng băng (kết quả ở mục 8.8.3). Chạy lại `rescore_rag_run.py` tái lập
**đúng** mọi quyết định chốt chặn của Colab, xác nhận mã cục bộ khớp với mã đã chạy. Notebook báo cáo
thêm **Phần K** (bảng A–E, bảng bẫy, ba ví dụ trích xuất vs RAG, đọc thẳng từ dữ liệu đã lưu), ghi rõ
hạn chế 2 **không** được giải quyết bởi RAG; mỗi ô có id cố định; **86 ô, 0 lỗi**.

### 7.15. Giai đoạn 15 — Rà soát tài liệu

Trước khi viết báo cáo này, toàn bộ tài liệu và chú thích mã được đối chiếu với mã nguồn, dữ liệu và
lịch sử commit. Các sai lệch được sửa gồm: docs/02 còn mô tả ensemble 0,8/0,2 và ngưỡng 0,12; chú thích
`config.py` mô tả nửa chu kỳ 7 ngày và khẳng định "độ mới làm tốt lên truy hồi" đã bị bác bỏ; README
còn ghi "Recall@3 100%" (số của tập rò rỉ) và "~150 pattern" (thực tế 164); docs/05 ghi một điểm số
cũ; case 5 của phần phân tích lỗi ghi "đã sửa" dù dò trên DEV đã tắt ensemble; `requirements.txt`
thiếu `scipy` và `joblib`. Rà soát cũng phát hiện một **khẳng định sai** trong docs/07: câu "tàu cát linh
15.000 đồng" không tới được RAG không phải do intent classifier (intent rỗng, độ tin cậy 0) mà do
**ngưỡng truy hồi** (cosine 0,096 < 0,13) — đã đính chính.

---

## 8. Kết quả thực nghiệm

### 8.1. Kiểm chứng cài đặt

| Bộ kiểm thử | Số kiểm tra | Kết quả |
|---|---|---|
| `test_vectorizer.py` — TF-IDF vs scikit-learn (5 cấu hình + trường hợp biên) | 28 | pass, sai số ~1e-16 |
| `test_vectorizer.py` — BM25 vs cài đặt ngây thơ (4 cặp $k_1,b$) + tính chất bão hòa | 17 | pass |
| `test_chatbot.py` — kiểm thử hồi quy | 21 | pass |
| `test_rag.py` — lớp RAG, backend giả lập | 29 | pass |
| **Tổng** | **95** | **pass** |

### 8.2. Dò tham số trên DEV

**Intent** (accuracy = câu intent cố định đúng nhãn và vượt ngưỡng; safety = câu cần truy hồi/ngoài
phạm vi không bị trả lời bằng câu soạn sẵn; mỗi dòng là ngưỡng tốt nhất cho $w_{nb}$ đó):

| $w_{nb}$ | ngưỡng | accuracy | safety | trung bình |
|---|---|---|---|---|
| 0,0 | 0,35 | 91,1% | 61,3% | 76,2% |
| 0,2 | 0,40 | 80,0% | 76,0% | 78,0% |
| 0,4 | 0,35 | 84,4% | 80,7% | 82,6% |
| 0,6 | 0,30 | 86,7% | 82,0% | 84,3% |
| 0,8 | 0,30 | 82,2% | 91,3% | 86,8% |
| **1,0** | **0,25** | **82,2%** | **92,7%** | **87,4%** ← chọn |

**Cách xếp hạng** (độ mới tắt): TF-IDF R@1 94,6%, MRR 0,965; BM25 tốt nhất ($k_1$=8, $b$=0,9) R@1
95,5%, MRR 0,967. MRR của BM25 theo $k_1$ ($b$ = 0,9): 0,942 (0,6) · 0,942 (0,9) · 0,944 (1,2) ·
0,946 (1,5) · 0,954 (2,0) · 0,959 (3,0) · 0,960 (5,0) · 0,967 (8,0). Kiểm định dấu: 3 thắng / 3 thua
/ 106 hòa, p = 1,000 → **TF-IDF**.

**Độ mới** (ràng buộc cứng: ca mâu thuẫn đúng):

| $h$ (ngày) | $\alpha$ | R@1 | MRR | mâu thuẫn |
|---|---|---|---|---|
| 3 | 0,0 | 94,6% | 0,965 | sai |
| 3 | 0,4 | 93,8% | 0,960 | đúng |
| **3** | **0,6** | **94,6%** | **0,964** | **đúng** ← chọn |
| 3 | 0,8 | 94,6% | 0,963 | đúng |
| 7 | 0,6 | 93,8% | 0,960 | đúng |
| 14 | 0,6 | 93,8% | 0,958 | sai |
| 14 | 1,0 | 93,8% | 0,958 | đúng |
| 30 | 0,2 | 95,5% | 0,969 | sai |
| 30 | 1,0 | 93,8% | 0,958 | sai |

(Bảng đầy đủ 24 cấu hình: `data/eval/test_report_v3.txt`.) Cấu hình MRR cao nhất (30 ngày, $\alpha$ =
0,2) **không** xử lý được ca mâu thuẫn; mọi cấu hình xử lý được đều thấp hơn nhẹ (0,957–0,964). Độ mới
vì vậy là một **đánh đổi** khoảng 0,001 MRR.

**Ngưỡng truy hồi.** Phân bố cosine top-1 trên DEV: câu trúng bài (n = 106) min 0,064, p10 0,128, trung
vị 0,218; câu ngoài phạm vi (n = 28) trung vị 0,072, p90 0,109, **max 0,125**.

| ngưỡng | trả lời được | chặn đúng | TB |
|---|---|---|---|
| 0,100 | 96,2% | 82,1% | 89,2% |
| 0,120 | 92,5% | 92,9% | 92,7% |
| **0,130** | **88,7%** | **100,0%** | ← chọn |
| 0,140 | 85,8% | 100,0% | 92,9% |
| 0,160 | 76,4% | 100,0% | 88,2% |

Ngưỡng 0,13 là giá trị thấp nhất (trên lưới bước 0,005) chặn được 100% câu ngoài phạm vi DEV.

**Bộ tham số chốt** (`data/eval/tuned_params.json`): $w_{nb}$ = 1,0; ngưỡng intent 0,25; xếp hạng
TF-IDF; $h$ = 3 ngày; $\alpha$ = 0,6; ngưỡng truy hồi 0,13.

### 8.3. Kết quả trên TEST

| Hạng mục | Chỉ số | Kết quả | KTC Wilson 95% |
|---|---|---|---|
| Truy hồi | Recall@1 | **93,4%** (114/122) | 88–97% |
| Truy hồi | Recall@3 | **95,9%** (117/122) | 91–98% |
| Truy hồi | MRR | **0,950** | — |
| — câu có dấu | Recall@1 / MRR | 95,3% (82/86) / 0,967 | 89–98% |
| — câu không dấu | Recall@1 / MRR | 95,5% (21/22) / 0,960 | 78–99% |
| — câu teencode | Recall@1 / MRR | **78,6%** (11/14) / 0,832 | 52–92% |
| Ngoài phạm vi (thành phần) | bị chặn | 87,5% (21/24) | 69–96% |
| Intent | Accuracy | **61,5%** (16/26) | 43–78% |
| Intent | Macro-F1 | 0,712 | — |
| **Đầu-cuối** | tin tức → bài đứng đầu đúng | **77,0%** (94/122) | 69–84% |
| **Đầu-cuối** | ngoài phạm vi → bot từ chối | **75,0%** (18/24) | 55–88% |

**Phân tích.**

- *Truy hồi* tốt và ổn định giữa các kiểu gõ có dấu/không dấu; teencode là chỗ yếu (các viết tắt như
  `vc`, `ntn` chưa có trong từ điển), nhưng chỉ 14 câu nên KTC rất rộng.
- *Intent* là điểm yếu lớn nhất. Cả 10 câu sai đều bị **từ chối vì dưới ngưỡng** (độ tin cậy
  0,00–0,25), đều là câu ngắn, cách nói đời thường: "ừm", "đúng vậy", "thanks nhé", "ngu thế", "bot
  làm được những gì".
- *Khoảng cách thành phần → đầu-cuối* 16 điểm (93,4% → 77,0%). Trong 28 câu đầu-cuối không đạt: **18**
  câu cosine dưới ngưỡng (fallback), **7** câu bị intent classifier định tuyến nhầm, **3** câu ra bài sai.
  Đây là đánh đổi có chủ đích của ngưỡng 0,13 (chặn đúng 100% trên DEV, từ chối ~11% câu trả lời được).
- *3 câu ngoài phạm vi lọt ngưỡng truy hồi:* "kết quả xổ số miền bắc" (0,142, khớp bài săn mây miền
  Bắc), "top 10 truyện tranh hay nhất" (0,148, khớp "Internet ... top 10"), "bài văn tả con mèo lớp 3"
  (0,137). Thêm 3 câu bị intent classifier trả lời nhầm bằng câu soạn sẵn (ví dụ "địa chỉ tiệm sửa
  laptop gần đây" → `tam_biet`).

### 8.4. BM25 so với TF-IDF trên TEST

| | Recall@1 | MRR |
|---|---|---|
| TF-IDF | 93,4% (114/122, 88–97%) | 0,950 |
| BM25 ($k_1$=8, $b$=0,9) | 91,8% (112/122, 86–95%) | 0,943 |

**Giải thích (giả thuyết được dữ liệu ủng hộ):** index tăng trọng số tiêu đề bằng cách **lặp** tiêu
đề 3 lần — mẹo này chỉ hiệu quả khi tf còn tăng theo số lần lặp. Độ bão hòa tf của BM25 triệt tiêu
đúng hiệu ứng đó; khi giảm bão hòa (tăng $k_1$), MRR của BM25 tăng đều và tiến về hành vi của TF-IDF.
Cách làm đúng là BM25F (bão hòa riêng từng trường rồi cộng có trọng số).

### 8.5. Chuẩn hóa teencode

| Split ViLexNorm | Số token (cần sửa) | Acc trước | Acc sau | **ERR** | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| dev | 10.003 (1.547) | 84,53% | 95,24% | 69,23% | 91,18% | 71,49% | 80,14% |
| **test** | 10.186 (1.642) | 83,88% | 94,82% | **67,84%** | 91,36% | 70,22% | 79,41% |

Precision cao hơn hẳn recall là **đánh đổi có chủ đích**: ba điều kiện an toàn khiến từ điển thận trọng
— bỏ sót từ hiếm nhưng gần như không sửa hỏng từ đã đúng. **Giới hạn của con số:** ERR chỉ tính trên
các cặp căn được theo vị trí (79,5%); các cặp đổi số token bị loại khỏi cả học lẫn đánh giá, nên con số
trên toàn tập sẽ thấp hơn.

### 8.6. Tin lỗi thời

Với `conflict_case.json` và cấu hình hiện hành:

| Câu hỏi | TF-IDF thuần (cũ / mới) | Có độ mới (mới / cũ) | Qua `bot.respond()` |
|---|---|---|---|
| giá vé tàu cát linh bao nhiêu | **0,4956 cũ** / 0,4098 | **0,6556 mới** / 0,5328 | trả bài mới |
| vé tàu cát linh có tăng giá không | 0,4126 / **0,4577 mới** | **0,7323 mới** / 0,4435 | trả bài mới |
| tàu cát linh 15.000 đồng | **0,1087 cũ** / 0,0959 | **0,1534 mới** / 0,1169 | "không tìm thấy" (cosine 0,096 < 0,13) |
| hoãn tăng giá vé tàu | 0,1823 / **0,4582 mới** | **0,7332 mới** / 0,1960 | trả bài mới |

Xếp hạng theo độ mới đưa bài mới lên đầu ở **cả 4** cách hỏi (TF-IDF thuần: 2/4). Ở mức đầu-cuối, 3/4
cách hỏi nhận bài mới; cách hỏi thứ tư bị ngưỡng chấp nhận chặn — an toàn, nhưng không trả lời được.
(Notebook Phần G4 dựng lại hai bài trong ô lệnh nên cho bộ số riêng: 0,5269/0,4302 khi tắt và
0,6883/0,5664 khi bật, cùng kết luận.)

### 8.7. Mô hình sinh n-gram

Dữ liệu: 9.567 câu huấn luyện / 1.063 câu kiểm thử / 216.976 token.

| $n$ | Perplexity ($\lambda$ = 0,7) |
|---|---|
| 1 | 1.285,4 |
| **2** | **818,9** |
| 3 | 1.950,7 |
| 4 | 5.911,2 |

| $\lambda$ | $n$=2 | $n$=3 | $n$=4 |
|---|---|---|---|
| 0,3 | 715 | 844 | 1.173 |
| 0,5 | 716 | 1.113 | 2.109 |
| 0,7 | 819 | 1.951 | 5.911 |
| 0,9 | 1.282 | 7.624 | 63.318 |

1. **Perplexity tăng theo $n$** — ngược trực giác. Nguyên nhân: dữ liệu thưa nên thành phần bậc cao
   gần như luôn bằng 0, và công thức nội suy nhân $(1-\lambda)$ ở **mỗi** lần lùi bậc (lùi hai bậc với
   $\lambda$ = 0,7 là nhân 0,09). Kiểm chứng: hạ $\lambda$ từ 0,9 xuống 0,3 làm perplexity $n$ = 4 giảm
   54 lần. Ở mọi $\lambda$, $n$ = 2 vẫn tốt nhất.
2. **$n$ lớn thì "sinh" thành "chép":** với $n$ = 4, các mẫu sinh ra mở đầu bằng cùng một chuỗi dài chép
   từ kho ("du lịch phú quốc thành lập năm 2014 ở hàng châu, tập trung phát triển ôtô điện...") vì phần
   lớn ngữ cảnh chỉ xuất hiện một lần.
3. **Sai sự thật ở mọi $n$:** "du lịch phú quốc còn đang xây dựng các trung tâm điều trị ebola".

Kết luận: ở quy mô dữ liệu này, sinh bằng n-gram **thua truy hồi trên mọi tiêu chí** (mạch lạc, đúng sự
thật, dẫn nguồn) — căn cứ thực nghiệm cho kiến trúc trích xuất.

### 8.8. Lớp RAG với PhoGPT-4B-Chat

#### 8.8.1. Lần chạy 1 (DEV, Q4_K_M, prompt v1)

| Nhãn (21 câu tin tức) | Số câu | Ví dụ |
|---|---|---|
| A | 3 | "ten lua spectrum cua duc phong ve tinh" → trả lời đủ, đúng giờ phóng |
| B | 5 | Blue Origin: nội dung đúng + dòng `[2] (đăng 06/30/2021...)` chép từ định dạng ngữ cảnh |
| C | 3 | "Huawei Mate XT2 gập ba" → "Huawei Mate XT2 gập ba." |
| D | 8 | "Hạn hán ... gây ra sự đình trệ ở eo biển Hormuz" (bài: **xung đột** gây đình trệ) |
| E | 2 | "bán kết Mỹ Mở rộng nữ có bốn hạt giống" → "không đề cập" |

Bẫy: 0/5 — "đảo Hải Nam miễn visa từ năm 2015 phải không" → "Đúng."; "lợi nhuận năm 2020 của metro
Bến Thành Suối Tiên" → "là 27 tỷ đồng" (số của mục tiêu 2026–2030).

#### 8.8.2. Lần chạy 2 (DEV, Q8_0, v1 vs v2)

Xem bảng mục 7.13. Các lỗi D còn lại của v2 cho thấy lỗi **không cần con số**: "hàng nhập **dưới** 100
nghìn đồng có thể không còn được miễn thuế" (bài: hạ ngưỡng xuống 100.000 → hàng **trên** 100.000 mất
miễn thuế; mô hình lấy chữ của câu hỏi đảo nghĩa câu trả lời); "... Wimbledon 2009 và đây cũng chính là
năm Sabalenka lọt vào vòng đấu này" (nửa sau bịa). Chốt chặn giả định đúng 4/4 bẫy có con số; chốt chặn
số bịa bắt được "Chornobyl năm 196" (1986) và "khoảng 1,5 triệu euro" (bịa hoàn toàn).

#### 8.8.3. Lần chạy 3 — TEST, một lần, mã đóng băng

**Cấu hình:** Q8_0; 40 câu tin tức TEST (33 câu tới RAG, **cả 33 truy hồi đúng bài**), 4 câu mâu thuẫn,
11 bẫy mới (9 câu tới RAG), 24 câu ngoài phạm vi; mỗi câu qua v1 và v2.

**Chất lượng câu trả lời (33 câu tin tức, trước chốt chặn):**

| Nhãn | v1 | **v2** |
|---|---|---|
| A — đúng, trả lời được | 9 | **19** |
| B — đúng, trình bày hỏng | 10 | **2** |
| C — không trả lời | 8 | 8 |
| **D — có thông tin sai** | **3** | **4** |
| E — từ chối sai | 3 | **0** |

So có cặp trên nhãn A: **11** câu v2 thắng, **1** câu v1 thắng → kiểm định dấu **p = 0,0063**. Đây là
câu hỏi chưa từng thấy khi thiết kế prompt, nên "v2 hữu ích hơn v1" đứng vững. **Nhưng thông tin sai
không giảm (3 → 4).** Các lỗi D của v2:

| Câu hỏi | v2 viết | Bài gốc |
|---|---|---|
| Messi được đề cử Quả bóng vàng 2026 | "được đề cử **vì** anh không có tên trong danh sách 2024 và 2025" | đảo nhân quả |
| chuẩn bị cho tuổi già từ năm 40 tuổi | "nhóm người dân có **độ tuổi trung bình là 40**" | "người **từ** 40 tuổi" |
| khách Pháp chết ở Thung lũng Chết | gán kỷ lục "nhiệt độ cao nhất từng ghi nhận" cho ca này | hôm đó 46,7 °C; kỷ lục vùng là 56,7 °C |
| iPhone 18 Pro Max có mấy màu | "xanh dương sáng", "đỏ sẫm" | "bạc", "đỏ burgundy" |

Ca iPhone: **cả v1 và v2** bịa hai màu vì 4 câu đưa vào ngữ cảnh **không chứa** câu liệt kê màu — lỗi
bắt đầu từ tầng chọn câu tự cài đặt, LLM chỉ lấp chỗ trống.

**Câu bẫy (11 câu mới):** v1 và v2 hòa **6 an toàn / 5 sai**, nhưng vì lý do trái ngược.

| Loại bẫy | Số câu | v1 | v2 |
|---|---|---|---|
| giả định sai có con số | 2 | 0 an toàn ("Đúng.", bịa "năm 2031") | **2 an toàn** (chốt chặn giả định) |
| con số sai có ở chỗ khác trong bài | 1 | sai | **sai** (điểm mù dự đoán trước) |
| giả định sai **không có số**, tới RAG | 4 | 2 an toàn (tình cờ từ chối/né) | **0 an toàn** |
| thiếu chi tiết | 2 | 2 an toàn | 2 an toàn (1 nhờ chốt chặn số bịa — LLM bịa "3,5 tỷ USD"; 1 vì câu rỗng) |
| không tới RAG (Harvard, Tim Cook) | 2 | an toàn | an toàn |

v2 khẳng định thẳng điều bài báo **phủ định**: "robot Optimus của Tesla tự bước ra khỏi dây chuyền" (bài
nói Tesla **chưa** sản xuất hàng loạt được Optimus; robot tự bước ra là Iron của Xpeng), "chị em song
sinh" (hai anh em), "Google mua Hugging Face" (Nvidia mua). Prompt v2 làm mô hình **quả quyết hơn**: trả
lời tốt hơn nhiều với câu hỏi thật, nhưng cũng dễ gật đầu với giả định sai hơn.

**Các nhóm khác:** mâu thuẫn — v2 **3/3** theo tin mới, v1 2/3 (một câu tự mâu thuẫn); ngoài phạm vi —
21/24 không tới LLM, 3 câu lọt ngưỡng truy hồi (v2: 1 từ chối đúng, 1 bịa "top 10 truyện tranh hay nhất
là những tác phẩm được đánh giá cao...", 1 bị chốt chặn giả định chặn vì số "3" trong "lớp 3"); chốt
chặn — giả định **0 lần báo nhầm** trên 33 câu tin tức, lặp lại 4 lần đều đúng, số bịa 1 lần do làm tròn
("2 km" so với "2,1 km"); trích dẫn — v1 có 3 câu trích dẫn hợp lệ nhưng 7 câu sai số hiệu, v2 không yêu
cầu trích dẫn; độ trễ — v2 1,18 s (p90 3,07 s), v1 1,49 s (p90 4,00 s).

**Tổng hợp ba lần chạy:**

| | Lần 1 (DEV) | Lần 2 (DEV) | **Lần 3 (TEST)** |
|---|---|---|---|
| Mô hình | Q4_K_M | Q8_0 | Q8_0 |
| Câu tin tức tới RAG | 21 | 21 | 33 |
| A: v1 → v2 | 3 → — | 4 → 10 | **9 → 19** |
| D: v1 → v2 | 8 → — | 7 → 5 | **3 → 4** |
| E: v1 → v2 | 2 → — | 3 → 0 | **3 → 0** |
| Bẫy an toàn (v2, có chốt chặn) | — | 5/5 | **6/11** |
| Mâu thuẫn theo tin mới | 3/3 (v1) | 3/3 (v2) | **3/3 (v2)** |

### 8.9. Hiệu năng

| Thành phần | Giá trị (CPU laptop, 381 bài) |
|---|---|
| Dựng index, không cache | ~24 s |
| Khởi động bot, có cache | ~1,1 s |
| Trả lời một câu hỏi (lõi) | ~21–24 ms |
| Vocabulary: index chính / index âm tiết / intent | 105.020 / 87.294 / 493 |
| RAG v2 trên Colab T4 (Q8_0) | 1,18 s trung bình, 3,07 s p90 |

---

## 9. Thảo luận và phân tích lỗi

### 9.1. Phân tích 19 case lỗi (Phần J của notebook)

| # | Tầng | Lỗi | Trạng thái |
|---|---|---|---|
| 1 | Regex | `\b` sau `%` bỏ sót "3,5%" | đã sửa |
| 2 | Tách từ | câu không dấu bị tách sai | đã sửa (index âm tiết) |
| 3 | NER | "Công ty ABC" gán LOC; số điện thoại gán LOC | giảm nhẹ (Regex bắt song song) |
| 4 | Stopword | "AI" (viết tắt tiếng Anh) bị xóa vì "ai" là stopword | tồn tại |
| 5 | Intent | câu ngắn "cảm ơn nhé" dưới ngưỡng; ensemble từng sửa, nhưng dò trên DEV chọn NB thuần | **tồn tại** |
| 6 | Truy hồi | "tin du lịch ninh bình" dưới ngưỡng toàn cục | đã sửa (ngưỡng nới khi nêu mục) |
| 7 | Intent | `huong_dan` / `liet_ke_chuyen_muc` chồng lấn | tồn tại |
| 8 | Truy hồi / teencode | chuẩn hóa đúng nhưng vector bị loãng | đã sửa (lọc từ khung) |
| 9 | Kiến trúc | hai danh sách từ khung lệch nhau | đã sửa (một nguồn) |
| 10 | Tách từ | "Sức khỏe" tách khác nhau theo ngữ cảnh | đã sửa (so khớp âm tiết) |
| 11 | Sinh n-gram | perplexity tăng theo $n$ | đã giải thích |
| 12 | Teencode | "t" luôn thành "tôi" kể cả khi là "tao" | tồn tại |
| 13 | Xếp hạng | trả bài cũ lỗi thời | đã sửa (độ mới) |
| 14 | Phương pháp đánh giá | tập dò ngưỡng thiếu câu không dấu/teencode | đã sửa |
| 15 | Chuẩn hóa / định tuyến | bot "tạm biệt" khi hỏi thời tiết | đã sửa (giữ hệ quy chiếu dấu) |
| 16 | Phương pháp đánh giá | rò rỉ tập test | đã sửa (DEV/TEST) |
| 17 | Xếp hạng / ngưỡng | độ mới loại bài cũ | đã sửa (xếp hạng ≠ chấp nhận) |
| 18 | Định tuyến | kết quả tùy nhánh khi intent 0,246 | đã sửa (một hằng số) |
| 19 | Quy tắc chọn mô hình | chọn BM25 vì hơn 0,002 MRR | đã sửa (kiểm định dấu) |

### 9.2. Bài học rút ra

1. **Đo trước khi sửa.** Giả thuyết "dùng ngưỡng biên top-1/top-2 thay ngưỡng tuyệt đối" nghe hợp lý
   nhưng bị bác bỏ khi đo: tỷ lệ top-1/top-2 của câu trong phạm vi (1,03×–8,62×) và ngoài phạm vi
   (1,04×–1,59×) chồng lấn nặng.
2. **Tập đánh giá phải đúng phân bố đầu vào thật** — mở rộng khả năng hệ thống mà không mở rộng tập
   đánh giá thì mọi tham số dò sau đó đều lệch (case 14).
3. **Hai câu hỏi khác nhau cần hai thước đo khác nhau** — "bài nào đứng trước" và "có đủ căn cứ trả lời"
   (case 17).
4. **Một quy tắc nghiệp vụ chỉ nằm ở một chỗ** (case 9, 18).
5. **Bước tiền xử lý không được âm thầm thay đổi tín hiệu dùng để chọn nhánh** (case 15).
6. **Phương pháp phức tạp hơn phải thắng có ý nghĩa thống kê** (case 19).
7. **Truy hồi đúng bài chưa đủ** — phải đưa đúng câu; và khi truy hồi sai, RAG **che** lỗi bằng một câu
   trả lời trôi chảy, còn bot trích xuất **lộ** lỗi ra để người đọc tự nhận ra.
8. **Kết quả âm có giá trị** — ba thí nghiệm âm (n-gram, BM25, RAG) đều trả lời được một câu hỏi thiết
   kế bằng số liệu thay vì bằng giả định.

### 9.3. Các mối đe dọa tới tính hợp lệ

- **Người viết truy vấn cũng là người xây bot**, nên truy vấn có thể mang cùng "điểm mù" với bot. Đã giảm
  thiểu bằng cách viết từ phần mô tả bài và diễn đạt lại, nhưng tốt nhất là nhờ người khác viết.
- **Cỡ mẫu nhỏ:** 26 câu intent (KTC 43–78%), 14 câu teencode, 33 câu RAG trên test, 11 bẫy. Các con số
  RAG nên đọc như **dạng lỗi**, không như một độ chính xác có ý nghĩa thống kê cao.
- **Một người chấm, và là công cụ AI** cho câu trả lời RAG (mục 12); người chấm biết câu nào là v1/v2.
- **Tập TEST đã được dùng nhiều lần** (Phụ lục C). Tham số chatbot không đổi sau lần 3; các lần sau chỉ
  báo cáo hoặc tính lại với tham số đã chốt, nhưng mọi cải tiến tiếp theo cần một tập test mới.
- **Mẫu làm sạch câu sinh** rút ra từ đầu ra DEV, nên đo trên các câu DEV đó là lạc quan; lần 3 trên TEST
  là phép đo trên đầu ra chưa thấy.
- **Một nguồn tin (VnExpress), 381 bài:** không nên suy rộng sang văn phong hay miền dữ liệu khác.

---

## 10. Hạn chế và hướng phát triển

### 10.1. Hạn chế

1. **Không hiểu từ đồng nghĩa** — TF-IDF so khớp mặt chữ ("xe hơi" không tìm ra "ô tô").
2. **Lõi trích xuất 100%**, không suy luận hay diễn đạt lại; lớp RAG diễn đạt lại được nhưng không trung
   thực hơn.
3. **Kho tri thức tĩnh**, cập nhật bằng cách crawl lại và dựng lại index (IDF của mọi term đổi theo).
4. **Intent classifier yếu nhất** (61,5%): câu ngắn đời thường dưới ngưỡng; hai intent chồng lấn; câu hỏi
   xác nhận "...đúng không" bị định tuyến nhầm.
5. **Tham chiếu chỉ neo lượt gần nhất** ("bài thứ hai ấy" chưa giải được).
6. **Không phát hiện mâu thuẫn giữa các bài**; độ mới chỉ phá thế hòa, bài cũ liên quan vượt trội vẫn thắng.
7. **Khoảng cách thành phần → đầu-cuối** 16 điểm do ngưỡng chấp nhận.
8. **Tầng chọn câu** có thể bỏ sót câu chứa đáp án.
9. **Chuẩn hóa teencode không xét ngữ cảnh** (recall 70,2%; "t" mơ hồ).
10. **Chốt chặn RAG chỉ bắt lỗi có con số/mã hiệu**; giả định sai không có số lọt hoàn toàn.

### 10.2. Hướng phát triển

| Hướng | Kỹ thuật | Kỳ vọng |
|---|---|---|
| Intent classifier | Thêm pattern cho câu ngắn và câu xác nhận; hiệu chỉnh xác suất thay ngưỡng cứng; đo trên tập test mới | Sửa điểm yếu lớn nhất |
| Tầng chọn câu | Tăng số câu / ưu tiên câu chứa thực thể và con số của câu hỏi | Sửa ca "màu iPhone" cho cả bot và RAG |
| Từ đồng nghĩa | Embedding (Word2Vec/PhoBERT) lai với TF-IDF | Sửa hạn chế 1 |
| BM25F | Bão hòa tf riêng từng trường | Cho BM25 cơ hội thật, thay mẹo lặp tiêu đề |
| Phát hiện cùng-một-sự-việc | Gom cụm bài tương đồng, cảnh báo "có bài mới hơn" | Sửa hạn chế 6 |
| Kiểm tra suy diễn cho RAG | NLI giữa câu sinh và nguồn | Bắt giả định sai không có số |
| Chuẩn hóa có ngữ cảnh | seq2seq (BARTpho) | Tăng recall, giải "t" mơ hồ |
| Đánh giá | Tập truy vấn do người khác viết; nhiều người chấm RAG | Giảm thiên kiến |

---

## 11. Kết luận

Đồ án đã xây dựng một chatbot hỏi đáp tin tức tiếng Việt với phần lõi **tự cài đặt và có kiểm chứng**,
xử lý được câu không dấu, teencode và tin lỗi thời, chạy tức thì trên CPU. Trên tập TEST tách riêng, truy
hồi đạt Recall@1 **93,4%** và MRR **0,950**; qua toàn bộ hệ thống, **77,0%** câu hỏi tin tức nhận đúng
bài. Intent classifier (61,5%) là điểm yếu lớn nhất và được xác định rõ nguyên nhân.

Giá trị lớn nhất của đồ án không nằm ở một con số cao, mà ở **cách các con số được tạo ra**: phát hiện và
sửa rò rỉ tập test, chấp nhận các con số thấp hơn nhưng trung thực, dùng khoảng tin cậy và kiểm định thống
kê cho mọi quyết định chọn phương pháp, ghi nhật ký mọi lần xem tập test, và báo cáo đầy đủ ba kết quả âm.
Thí nghiệm RAG — ba lần chạy thật, một bộ bẫy chốt trước, mã đóng băng trước lần đo cuối — cho một kết luận
rõ ràng và có số liệu: mô hình sinh làm câu trả lời **tự nhiên hơn** nhưng **không trung thực hơn** bản
trích xuất. Vì một chatbot tin tức phải đặt độ trung thực lên trước độ trơn tru, sản phẩm nộp là bản trích
xuất; RAG là một lớp mở rộng tùy chọn, mặc định tắt.

---

## 12. Tuyên bố về việc sử dụng công cụ AI

Đồ án được thực hiện với sự hỗ trợ của trợ lý lập trình AI (Claude, Anthropic) trong vai trò viết và sửa
mã, chạy thí nghiệm, phân tích kết quả và soạn tài liệu, dưới sự chỉ đạo, kiểm tra và ra quyết định của
sinh viên (chọn đề tài, chọn hướng tiếp cận, yêu cầu môi trường ảo cục bộ, quyết định chạy PhoGPT thủ công
trên Colab thay vì tự động hóa, chạy các notebook Colab, duyệt và đẩy mã). Riêng **nhãn chấm tay cho câu
trả lời RAG** (mục 6.5, 8.8) do trợ lý AI thực hiện bằng cách đối chiếu từng khẳng định với toàn văn bài
gốc; các tệp nhãn được lưu công khai trong kho mã để có thể kiểm tra lại, và nên được người chấm độc lập
xác minh trước khi coi là kết luận chắc chắn.

---

## 13. Tài liệu tham khảo

1. Robertson, S., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*.
   Foundations and Trends in Information Retrieval, 3(4), 333–389.
2. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge
   University Press.
3. Jurafsky, D., & Martin, J. H. *Speech and Language Processing* (bản thảo 3rd ed.) — các chương về mô
   hình ngôn ngữ n-gram, Naive Bayes và perplexity.
4. Nguyen, T.-N., Le, T.-P., & Nguyen, K. V. (2024). *ViLexNorm: A Lexical Normalization Corpus for
   Vietnamese Social Media Text*. EACL 2024. <https://github.com/ngxtnhi/ViLexNorm>
5. Nguyen, D. Q., và cộng sự (2023). *PhoGPT: Generative Pre-training for Vietnamese*. arXiv:2311.02945.
   <https://github.com/VinAIResearch/PhoGPT>
6. Lewis, P., và cộng sự (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*.
   NeurIPS 2020.
7. Wilson, E. B. (1927). *Probable Inference, the Law of Succession, and Statistical Inference*. Journal of
   the American Statistical Association, 22(158), 209–212.
8. Pedregosa, F., và cộng sự (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12, 2825–2830.
9. underthesea — Vietnamese NLP Toolkit. <https://github.com/undertheseanlp/underthesea>
10. stopwords/vietnamese-stopwords. <https://github.com/stopwords/vietnamese-stopwords>
11. llama.cpp / llama-cpp-python và bản GGUF `vinai/PhoGPT-4B-Chat-gguf` trên Hugging Face.
12. Các dự án tham khảo ở mục 2.8 (Dec1mo, undertheseanlp/chatbot, heraclex12, sushant097, YUSANITY).
13. VnExpress (<https://vnexpress.net>) — nguồn bài báo, thu thập cho mục đích học tập.
14. Tài liệu Lab 01 (pipeline NLP đầu tiên), Lab 02 (thu thập dữ liệu), Lab 03 (tiền xử lý tiếng Việt),
    Lab 04 (BoW, n-gram, TF-IDF, cosine) của học phần.

---

## 14. Phụ lục

### Phụ lục A — Bảng siêu tham số hiện hành

| Hằng số | Giá trị | Nguồn |
|---|---|---|
| `RETRIEVAL_THRESHOLD` | 0,13 (cosine thuần) | dò trên DEV |
| `INTENT_THRESHOLD` | 0,25 | dò trên DEV |
| `INTENT_W_NB` | 1,0 (NB thuần) | dò trên DEV |
| `CATEGORY_SCOPED_THRESHOLD_FACTOR` | 0,6 | đặt tay (DEV quá ít câu nêu mục) |
| `FRESHNESS_ALPHA` | 0,6 | dò trên DEV |
| `FRESHNESS_HALFLIFE_DAYS` | 3,0 | dò trên DEV |
| `RANKING_METHOD` | `tfidf` | kiểm định dấu trên DEV |
| `BM25_K1`, `BM25_B` | 8,0; 0,9 | cấu hình BM25 tốt nhất trên DEV |
| `TOP_K` | 3 | — |
| `TFIDF_NGRAM_RANGE`, `MIN_DF`, `MAX_DF` | (1,2); 1; 0,85 | — |
| `TITLE_WEIGHT`, `DESC_WEIGHT` | 3; 2 | — |
| NB $\alpha$ | 0,3 | — |
| Teencode: `min_count`, `min_change_rate`, `min_dominance` | 4; 0,5; 0,5 | — |
| RAG: bài / câu mỗi bài / ký tự mỗi bài | 3; 4; 900 | — |
| RAG: token tối đa v1 / v2 | 256; 160 | — |

### Phụ lục B — Prompt RAG

Mẫu chung (PhoGPT-4B-Chat): `### Câu hỏi: {instruction}\n### Trả lời:`

**v1:**

```text
Bạn là trợ lý tin tức. Hãy trả lời câu hỏi CHỈ dựa trên các bài báo được cung cấp dưới đây.

Quy tắc:
1. Trả lời ngắn gọn bằng tiếng Việt tự nhiên, từ 2 đến 4 câu.
2. Chỉ dùng thông tin có trong các bài báo. Không thêm kiến thức bên ngoài, không đoán.
3. Ghi nguồn bằng số trong ngoặc vuông ngay sau thông tin, ví dụ [1] hoặc [2].
4. Nếu các bài báo mâu thuẫn nhau, hãy ưu tiên bài có ngày đăng MỚI HƠN và nói rõ điều đó.
5. Nếu các bài báo không có thông tin để trả lời, hãy trả lời đúng một câu: "Các bài báo hiện có không đề cập đến điều này."
6. Nếu câu hỏi chứa giả định sai so với bài báo, hãy chỉ ra điều đó.

Các bài báo:
{context}

Câu hỏi: {question}
```

**v2:**

```text
Đọc các đoạn tin dưới đây rồi trả lời câu hỏi ở cuối bằng 2 đến 3 câu tiếng Việt tự nhiên, chỉ dùng thông tin có trong các đoạn tin. Nếu các đoạn tin không có câu trả lời, chỉ viết: "Các bài báo hiện có không đề cập đến điều này." Nếu hai tin mâu thuẫn nhau, dùng tin có ngày đăng mới hơn.

{context}

Câu hỏi: {question}
```

### Phụ lục C — Nhật ký các lần dùng tập TEST

| Lần | Việc | Có thay đổi gì dựa trên kết quả test? |
|---|---|---|
| 1 | Báo cáo đầu tiên sau khi tách DEV/TEST (`test_report_v1.txt`) | Không |
| 2 | Sau hai sửa lỗi do kiểm thử hồi quy phát hiện (`_v2.txt`) | Không |
| 3 | Sau khi thêm BM25 (`_v3.txt`) | **Có** — quy tắc chọn phương pháp đổi sang kiểm định dấu |
| 4 | Đo chốt chặn giả định có chặn nhầm câu hợp lệ (không LLM) | Không |
| 5 | Đo lại sau khi sửa quy tắc so số có đơn vị | Không |
| 6 | RAG lần chạy 3 (mã đóng băng `fe10190`) | Không — chỉ báo cáo |

Ngoài ra, mỗi lần thực thi lại notebook báo cáo, Phần E chạy lại `evaluate.py` với đúng bộ tham số đã chốt;
các lần tính lại cho `test_results.json` giống hệt và không dẫn tới quyết định nào.

### Phụ lục D — Hướng dẫn tái lập

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python tests/test_vectorizer.py     # 45 kiểm tra TF-IDF/BM25
python tests/test_chatbot.py        # 21 kiểm thử hồi quy
python tests/test_rag.py            # 29 kiểm thử RAG (không GPU)
python src/evaluate.py              # dò trên DEV, báo cáo trên TEST
python src/normalizer.py            # học + đánh giá từ điển teencode
python src/generator.py             # thí nghiệm n-gram
python tools/rescore_rag_run.py data/eval/rag/run3_test_results.json   # chấm lại RAG lần 3

python src/cli.py                   # chat trong terminal
python src/api.py                   # web demo http://127.0.0.1:8000
```

RAG: `python tools/build_rag_notebook.py` và `python tools/make_colab_bundle.py`, rồi mở
`dist/RAG_PhoGPT_Colab.ipynb` trên Google Colab (GPU T4) → Run all → tải lên `dist/rag_bundle.zip`.

### Phụ lục E — Bản đồ tệp kết quả

| Tệp | Nội dung |
|---|---|
| `data/eval/tuned_params.json` | tham số dò trên DEV |
| `data/eval/test_results.json` | số liệu TEST |
| `data/eval/test_report_v1..v3.txt` | báo cáo đầy đủ từng lần xem test |
| `data/eval/rag/run1_results.json`, `run1_samples.csv` | RAG lần 1 |
| `data/eval/rag/run2_results.json`, `run2_samples.csv` | RAG lần 2 |
| `data/eval/rag/run3_test_results.json`, `run3_test_samples.csv` | RAG lần 3 (TEST) |
| `data/eval/rag/run*_rescored.csv` | chấm lại bằng mã hiện tại |
| `data/eval/rag/run3_annotation.csv`, `run3_trap_annotation.csv` | nhãn chấm tay lần 3 |
| `data/eval/rag/dev_traps.json`, `test_traps.json` | bộ câu bẫy |
| `notebooks/FinalProject_Chatbot_23IT036.ipynb` | notebook báo cáo (86 ô, đã chạy) |
| `notebooks/RAG_PhoGPT_Colab.ipynb` | notebook Colab cho RAG |
| `docs/01..07` | tài liệu kỹ thuật theo từng chủ đề |

### Phụ lục F — Danh sách 21 kiểm thử hồi quy

`regex_bat_duoc_phan_tram`, `nhan_dien_cau_khong_dau`, `cau_khong_dau_khong_qua_word_tokenize`,
`loc_tu_khung_khong_lam_rong_query`, `tu_khung_chi_co_mot_nguon_duy_nhat`, `teencode_duoc_chuan_hoa`,
`chuan_hoa_giu_he_quy_chieu_khong_dau`, `hoi_thoi_tiet_khong_bi_chao_tam_biet`,
`so_khop_ten_chuyen_muc_theo_am_tiet`, `cau_chi_neu_chuyen_muc_thi_duyet_muc`,
`chuyen_muc_kem_chu_de_thi_tim_trong_muc`, `truy_hoi_cau_khong_dau`, `truy_hoi_cau_teencode`,
`cau_ngoai_pham_vi_bi_tu_choi`, `snippet_khong_lap_cau`, `tham_chieu_bai_do_qua_nhieu_luot`,
`dau_vao_rong_khong_lam_sap_bot`, `doc_duoc_ngay_vnexpress`, `duyet_muc_sap_theo_ngay_moi_nhat`,
`tin_moi_phu_dinh_tin_cu`, `cache_tinh_lai_do_moi_theo_nua_chu_ky`.

---

### Phụ lục G — Cải tiến sau khi nộp bài (nhánh `cap-nhat-du-lieu`)

Bản nộp được đóng băng ở tag `nop-bai` (kho 381 bài). Sau đó kho được crawl bổ
sung hằng ngày theo quy trình ở `docs/08`, và **lần crawl thật đầu tiên
(14/09/2026, +151 bài, tổng 532)** làm lộ ra hai lỗi mà bộ dữ liệu tĩnh không thể
phát hiện. Cả hai đã được sửa trên nhánh `cap-nhat-du-lieu`; toàn bộ lập luận và
số liệu ở `docs/09-cai-thien-mo-hinh.md`.

**G.1. Mốc tính độ mới trôi theo kho.** Điểm độ mới được tính theo ngày đăng mới
nhất **của cả kho**. Khi crawl thêm 151 bài không liên quan, mốc nhảy từ 10/09
lên 14/09, hai bài trong ca kiểm thử "tin mới phủ định tin cũ" cùng già đi, và
bot quay lại trả bài cũ **đã sai** (0,5104 so với 0,5086) — đúng lỗi mà mục 5.5
của báo cáo sinh ra để chống. Đây là lỗi thiết kế chứ không phải lỗi tham số:
quét toàn bộ lưới (7 nửa chu kỳ × 11 giá trị α) cho thấy **không cấu hình nào**
của mốc "toàn kho" vừa xử lý đúng ca mâu thuẫn vừa còn đúng sau khi thêm một bài
không liên quan ở ngày tương lai. Cách sửa: lấy mốc là ngày mới nhất trong **các
bài đang cạnh tranh** cho chính câu hỏi đó (những bài thỏa `điểm × (1 + α) ≥ điểm
cao nhất`). Tính chất thu được là một bất biến: thêm bài không liên quan không
làm đổi thứ hạng. Dò lại trên DEV: α giảm 0,6 → 0,3.

**G.2. Không nhận ra câu gõ sai chính tả.** Người dùng gõ "thám hiểm Sơn Dòng"
(thiếu một chữ *o*, quên gạch của *Đ*) thì bot từ chối, trong khi "Sơn Đoòng" trả
lời đúng — cosine chỉ 0,086 < ngưỡng 0,13. Hạ ngưỡng không phải cách sửa: ở 0,08
tỷ lệ chặn đúng câu ngoài phạm vi rơi từ 100% xuống 58%. Cách sửa là thêm **chỉ
mục thứ ba gồm n-gram ký tự (n = 3) của tiêu đề đã bỏ dấu**, dùng như **đường dự
phòng** chỉ chạy khi đường chính từ chối, với điểm chấp nhận = cosine mức từ +
cosine n-gram ký tự. Thiết kế được chọn bằng thực nghiệm trên DEV (12 phương án:
trường đưa vào chỉ mục × n × luật chấp nhận); phương án "chỉ tiêu đề, n = 3, điểm
cộng" cứu được nhiều câu nhất mà không trả sai câu nào.

**G.3. Một bài học về dữ liệu đánh giá.** Sau khi bật đường dự phòng, kiểm thử
hồi quy bắt được một câu **ngoài phạm vi** viết không dấu bị trả lời, trong khi
việc dò trên DEV báo 0 câu lọt. Nguyên nhân: tập câu ngoài phạm vi của DEV khi đó
toàn câu **có dấu**, mà chỉ mục n-gram ký tự lại luôn làm việc trên bản đã bỏ dấu
— bỏ dấu làm hai câu khác nhau trông giống nhau hơn. Đã sửa bằng cách **bổ sung
dữ liệu đánh giá** (sinh thêm biến thể không dấu cho câu ngoài phạm vi của DEV,
28 → 36 câu) chứ không phải bằng cách vặn ngưỡng cho vừa ca đó. Các biến thể chỉ
được sinh từ câu của DEV, không bao giờ từ TEST, để không làm hỏng phép đo.

**G.4. Kết quả.** Đo trên cùng kho 532 bài, cùng bộ câu hỏi TEST, qua
`bot.respond()` (`tools/compare_improvements.py`):

| Chỉ số trên TEST | Trước (cấu hình lúc nộp) | Sau |
|---|---|---|
| Recall@1 (thành phần) | 109/122 | **112/122** |
| MRR | 0,927 | **0,940** |
| Câu hỏi thường → đúng bài | 90/122 | **101/122** |
| Câu gõ sai chính tả → đúng bài | 74/122 | **86/122** |
| Ca tin mới phủ định tin cũ | sai | **đúng** |
| Ca trên sau khi thêm bài "tương lai" | sai | **đúng** |
| Câu ngoài phạm vi bị từ chối | 18/24 | 18/24 |

Cái giá phải nói rõ: đường dự phòng biến một phần câu "từ chối" thành câu "trả
lời", nên số câu **trả lời sai** tăng (trên tập gõ sai 7 → 8, trên tập sạch 3 →
5). Bù lại nó không làm lọt thêm câu ngoài phạm vi nào, và mỗi câu trả lời theo
đường này đều kèm lời nhắc "có thể bạn gõ nhầm". Bộ kiểm thử hồi quy tăng từ 21
lên 26 ca, trong đó có hai ca canh đúng hai bất biến vừa nêu.
