# Chatbot tin tức tiếng Việt — Đồ án cuối kỳ NLP

Chatbot tiếng Việt xây dựng **from scratch** bằng TF-IDF, cosine similarity và
Naive Bayes tự cài đặt — **không dùng mô hình ngôn ngữ lớn**, không gọi
`sklearn` trong runtime.

Bot trả lời câu hỏi về tin tức dựa trên kho **381 bài báo VnExpress** thuộc
8 chuyên mục, do chính dự án thu thập.

**Sinh viên:** Hoàng Công Dũng — 23IT036

---

## Kết quả đo được

Tham số được dò trên tập **DEV**, rồi báo cáo trên tập **TEST** tách riêng —
câu trong TEST chưa từng được dùng để dò. Mọi tỷ lệ kèm khoảng tin cậy 95%.
Chi tiết phương pháp và lịch sử sửa đổi: [docs/06](docs/06-danh-gia-trung-thuc-va-bm25.md).

| Hạng mục | Chỉ số | Kết quả trên TEST | KTC 95% |
|---|---|---|---|
| TF-IDF tự cài đặt | khớp với scikit-learn | **28/28 test pass**, sai số ~1e-16 | — |
| BM25 tự cài đặt | khớp cài đặt tham chiếu | **17/17 test pass** | — |
| Truy hồi | Recall@1 | **93.4%** (114/122) | 88–97% |
| Truy hồi | Recall@3 | **95.9%** (117/122) | 91–98% |
| Truy hồi | MRR | **0.950** | — |
| Truy hồi — teencode | Recall@1 | 78.6% (11/14) | 52–92% |
| Intent classification | Accuracy | **61.5%** (16/26) | 43–78% |
| **Đầu-cuối** | câu tin tức → đúng bài | **77.0%** (94/122) | 69–84% |
| **Đầu-cuối** | câu ngoài phạm vi → bot từ chối | **75.0%** (18/24) | 55–88% |
| Chuẩn hóa teencode | ERR trên ViLexNorm test | **67.8%** | — |
| Xếp hạng độ mới | tin mới phủ định tin cũ | **đúng cả 4 cách hỏi** | — |
| Tốc độ | khởi động (có cache) / một câu hỏi | **1.1s / 24ms** | — |

> **Đính chính:** các phiên bản trước của README ghi Recall@1 96.8%, MRR 0.984,
> intent 88.5%. Các con số đó được đo trên chính tập đã dùng để dò tham số (rò
> rỉ tập test) và tập chỉ có 21–31 câu. Con số ở bảng trên là con số trung thực.

Toàn bộ số liệu tái lập được bằng `python src/evaluate.py` (dò trên dev, báo cáo trên test).

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
python tests/test_vectorizer.py     # đối chiếu TF-IDF với sklearn + BM25 với cài đặt tham chiếu
python tests/test_chatbot.py        # 21 kiểm thử hồi quy, mỗi cái ứng với một lỗi thật
python src/evaluate.py              # dò trên DEV, báo cáo trên TEST (Recall@k, MRR, F1, KTC 95%)
python tools/build_eval_sets.py     # tái tạo tập dev/test (seed cố định)
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

## RAG với PhoGPT (tùy chọn, chạy trên Google Colab)

Lớp **thêm** lên trên chatbot trích xuất: phần truy hồi tự cài đặt tìm 1–3 bài
báo, rồi **PhoGPT-4B-Chat** (VinAI) diễn đạt lại thành câu trả lời tự nhiên. Tắt
nó đi thì chatbot vẫn chạy nguyên như cũ. Đây là phần **duy nhất** dùng mô hình
tiền huấn luyện — nằm ngoài phạm vi from scratch.

**Nguyên tắc an toàn:** PhoGPT chỉ được gọi khi truy hồi đã tìm được bằng chứng
vượt ngưỡng; câu ngoài phạm vi bị từ chối **trước** khi tới mô hình. Thêm hai
**chốt chặn tất định**: câu hỏi nêu con số/mã hiệu mà bài báo không hề nhắc
(«năm 2015», «IP68») thì không cho mô hình trả lời; câu sinh ra có số không có
trong nguồn thì hiển thị câu trích xuất thay thế.

**Kết quả lần chạy thật đầu tiên (Colab T4, 21 câu tin tức dev) — tệ, và được ghi
lại đầy đủ** ở [docs/07](docs/07-rag-phogpt.md): chỉ 3/21 câu vừa đúng vừa trả lời
được, 8/21 có thông tin sai (đảo nhân quả, gắn sai năm, bịa cho khớp câu hỏi),
0/6 câu bẫy được xử lý đúng — mô hình 4B đồng ý với giả định sai ("Đúng."). Điểm
tốt: ca tin mâu thuẫn trả lời theo tin mới 3/3, câu ngoài phạm vi không bao giờ tới
mô hình. Prompt v2 + hai chốt chặn được thiết kế từ các lỗi đó; chốt chặn giả định
chặn nhầm **0** câu hợp lệ trên cả dev (93 câu) lẫn test (97 câu). Lần chạy 2
(so sánh v1/v2) đang chờ.

```powershell
python tools/make_colab_bundle.py   # tạo dist/rag_bundle.zip (mã nguồn + dữ liệu)
python tests/test_rag.py            # 25 kiểm thử RAG, không cần GPU (backend giả lập)
python tools/rescore_rag_run.py     # chấm lại một lần chạy Colab bằng mã hiện tại
```

Chạy trên Colab: mở `dist/RAG_PhoGPT_Colab.ipynb` trong Colab → Runtime → T4 GPU
→ Run all → tải lên `dist/rag_bundle.zip` khi được hỏi. Notebook chạy cùng bộ câu
hỏi qua prompt v1 và v2, tự chấm độ trung thành (số bịa), ca tin mâu thuẫn, câu
bẫy và câu ngoài phạm vi, rồi tải về `rag_results_run2.json` + `rag_samples_run2.csv`.

---

## Cấu trúc dự án

```text
final-project/
├── src/
│   ├── config.py            # đường dẫn + siêu tham số (đã dò thực nghiệm)
│   ├── preprocess.py        # pipeline tiếng Việt (Lab 03) + xử lý không dấu
│   ├── vectorizer.py        # BoW / n-gram / TF-IDF / cosine / BM25 — TỰ CÀI ĐẶT
│   ├── intent_classifier.py # Naive Bayes TỰ CÀI ĐẶT + ensemble cosine
│   ├── retriever.py         # truy hồi 2 tầng: bài báo -> câu
│   ├── normalizer.py        # chuẩn hóa teencode học từ ViLexNorm
│   ├── generator.py         # n-gram LM — thí nghiệm đối chứng sinh văn bản
│   ├── dates.py             # phân tích ngày đăng + điểm độ mới
│   ├── rag.py               # RAG: truy hồi + PhoGPT (tùy chọn, chạy trên Colab)
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
│   ├── eval/dev.json                # tập DEV — để dò tham số
│   ├── eval/test.json               # tập TEST — chỉ để báo cáo
│   ├── eval/conflict_case.json      # ca tin mâu thuẫn (ràng buộc cứng)
│   ├── eval/test_report_v*.txt      # báo cáo từng lần xem tập test
│   ├── eval/rag/run1_*.json|csv     # kết quả thật lần chạy PhoGPT 1 (Colab)
│   ├── resources/vietnamese-stopwords.txt
│   ├── resources/teencode_lexicon.json  # học được từ ViLexNorm
│   └── resources/vilexnorm/         # corpus chuẩn hóa (CC BY-NC-SA 4.0)
├── notebooks/FinalProject_Chatbot_23IT036.ipynb   # BÁO CÁO
├── notebooks/RAG_PhoGPT_Colab.ipynb              # RAG trên Colab
├── docs/
│   ├── 01-nghien-cuu-du-an-tham-khao.md
│   ├── 02-kien-truc.md
│   ├── 03-chuan-hoa-teencode.md
│   ├── 04-thi-nghiem-sinh-van-ban.md
│   ├── 05-do-moi-va-thong-tin-loi-thoi.md
│   ├── 06-danh-gia-trung-thuc-va-bm25.md
│   └── 07-rag-phogpt.md
├── tests/test_vectorizer.py         # TF-IDF vs sklearn, BM25 vs tham chiếu
├── tests/test_chatbot.py            # 21 kiểm thử hồi quy
├── tests/test_rag.py                # 25 kiểm thử RAG (không cần GPU)
├── tools/build_eval_sets.py         # sinh tập dev/test
├── tools/build_notebook.py          # sinh notebook báo cáo
├── tools/build_rag_notebook.py      # sinh notebook RAG cho Colab
├── tools/make_colab_bundle.py       # đóng gói mã + dữ liệu cho Colab
├── tools/rescore_rag_run.py         # chấm lại kết quả Colab bằng mã hiện tại
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
còn làm Recall@3 tăng lên **100%**.

**3. Đánh giá trung thực.** Tham số dò trên tập DEV, báo cáo một lần trên tập
TEST tách riêng; nhãn đúng theo URL bài báo; mọi tỷ lệ kèm khoảng tin cậy 95%;
có cả đánh giá đầu-cuối gọi `bot.respond()` như người dùng thật. Xem
[docs/06](docs/06-danh-gia-trung-thuc-va-bm25.md).

**4. Giải thích được từng câu trả lời.** `--explain` cho biết term nào khớp,
đóng góp bao nhiêu điểm, term nào là OOV.

**5. Biết nói "mình không biết".** Mọi đường đi đều có ngưỡng tin cậy.

**6. Hiểu teencode, và bảng ánh xạ được HỌC chứ không hardcode.** Học từ
[ViLexNorm](https://github.com/ngxtnhi/ViLexNorm) (10.467 cặp câu do người gán
nhãn) với ba điều kiện an toàn. ERR 67.8% trên split test chưa từng thấy.

**7. Không trả lời bằng thông tin lỗi thời.** Với hai bài mâu thuẫn cách nhau
9 ngày, TF-IDF thuần trả về bài **cũ đã sai** (0.4956 vs 0.4098) kèm dẫn nguồn
thật. Đã thêm xếp hạng theo độ mới `score = cosine × (1 + α·recency)` — sửa
được ca này ở **cả 4 cách hỏi**. Độ mới chỉ dùng để **xếp hạng**; việc **chấp
nhận** trả lời dựa trên cosine thuần. Trên tập dev lớn, độ mới là một **đánh
đổi** nhỏ (mất ~0.001 MRR), không phải cải tiến miễn phí như từng tưởng. Mọi
câu trả lời đều kèm **ngày đăng**. Xem [docs/05](docs/05-do-moi-va-thong-tin-loi-thoi.md).

**8. BM25 tự cài đặt — và một kết quả âm được ghi lại đầy đủ.** BM25 không hơn
TF-IDF (kiểm định dấu có cặp trên dev: 3 thắng / 3 thua / 106 hòa, p = 1.0).
Nguyên nhân có kiểm chứng: độ bão hòa tf của BM25 triệt tiêu mẹo lặp tiêu đề để
tăng trọng số — MRR của BM25 tăng đều khi giảm bão hòa (tăng k1).

**9. Kiểm thử hồi quy.** 21 test, mỗi cái ứng với một lỗi thật từng làm bot trả
lời sai mà không báo lỗi. Chúng đã bắt được hai lỗi thiết kế ngay lần chạy đầu.

**10. Đã kiểm chứng vì sao KHÔNG sinh văn bản.** `src/generator.py` cài đặt
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
6. **Không phát hiện mâu thuẫn giữa các bài** — xếp hạng theo độ mới chỉ *giảm
   nhẹ* vấn đề: nếu bài cũ liên quan **vượt trội** thì nó vẫn thắng. Bot không
   hiểu bài B phủ định bài A.

7. **Intent classifier là điểm yếu lớn nhất** — 61.5% trên test. Câu ngắn,
   cách diễn đạt đời thường ("ừm", "thanks nhé") có độ tin cậy dưới ngưỡng.
8. **Khoảng cách thành phần → đầu-cuối** — truy hồi đúng 93.4% nhưng đầu-cuối
   chỉ 77.0%, chủ yếu do ngưỡng chấp nhận (đánh đổi để không trả lời bừa).

Phân tích chi tiết các lỗi thật kèm nguyên nhân: xem **Phần J** của notebook.

Muốn bot **diễn đạt lại thay vì chép nguyên văn** thì cần ghép mô hình ngôn ngữ
lớn tiếng Việt theo kiểu RAG — đã thử với PhoGPT-4B-Chat (mục RAG ở trên,
[docs/07](docs/07-rag-phogpt.md)): câu trả lời tự nhiên hơn nhưng **kém trung
thành hơn** bot trích xuất, nhất là khi truy hồi sai.

---

## Nguồn tham khảo

- Danh sách stopword: [stopwords/vietnamese-stopwords](https://github.com/stopwords/vietnamese-stopwords)
- Thư viện NLP tiếng Việt: [underthesea](https://github.com/undertheseanlp/underthesea)
- Khảo sát các dự án chatbot tham khảo: [`docs/01-nghien-cuu-du-an-tham-khao.md`](docs/01-nghien-cuu-du-an-tham-khao.md)
- Corpus chuẩn hóa teencode: [ViLexNorm](https://github.com/ngxtnhi/ViLexNorm) —
  Nguyen et al., EACL 2024. Giấy phép **CC BY-NC-SA 4.0**, chỉ dùng cho mục đích
  nghiên cứu/học tập.
- Dữ liệu: bài báo công khai từ VnExpress, thu thập cho mục đích học tập.
