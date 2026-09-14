# Chatbot tin tức tiếng Việt — Đồ án cuối kỳ NLP

Chatbot tiếng Việt xây dựng **from scratch** bằng TF-IDF, cosine similarity và
Naive Bayes tự cài đặt — lõi **không dùng mô hình ngôn ngữ lớn**, không gọi
`sklearn` trong runtime. Một lớp RAG với PhoGPT được thử nghiệm riêng như phần
mở rộng **tùy chọn, mặc định tắt** (xem mục RAG bên dưới).

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
| Xếp hạng độ mới | tin mới phủ định tin cũ | **bài mới xếp đầu ở cả 4 cách hỏi** (1 cách hỏi dưới ngưỡng chấp nhận) | — |
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
python tests/test_rag.py            # 29 kiểm thử lớp RAG, không cần GPU
python src/evaluate.py              # dò trên DEV, báo cáo trên TEST (Recall@k, MRR, F1, KTC 95%)
python tools/build_eval_sets.py     # tái tạo tập dev/test (seed cố định)
python src/normalizer.py            # học + đánh giá từ điển teencode (ERR)
python src/generator.py             # thí nghiệm sinh văn bản n-gram

# Thu thập thêm dữ liệu + cập nhật mô hình (xem runbook docs/08 TRƯỚC khi chạy)
powershell -ExecutionPolicy Bypass -File tools\daily_update.ps1 -RestartApi
python src/crawler.py --per-category 20 --only "Công nghệ" "Thể thao"
python tools/rebuild_index.py       # dựng lại index + kiểm tra khói sau khi kho đổi
```

## Báo cáo

- **Báo cáo đồ án:** [`report/BaoCao_DoAnCuoiKy_23IT036_VI.md`](report/BaoCao_DoAnCuoiKy_23IT036_VI.md)
  (tiếng Việt) và [`report/Report_FinalProject_23IT036_EN.md`](report/Report_FinalProject_23IT036_EN.md)
  (tiếng Anh) — phương pháp, tiến trình từng giai đoạn kèm số liệu, kết quả, phân tích lỗi.
- `notebooks/FinalProject_Chatbot_23IT036.ipynb` — notebook đã chạy và lưu output,
  gồm đầy đủ lý thuyết, thực nghiệm, đánh giá, phân tích lỗi, và **Phần K** —
  thí nghiệm RAG với PhoGPT cùng kết luận rút ra từ ba lần chạy thật.

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
vượt ngưỡng; câu ngoài phạm vi bị từ chối **trước** khi tới mô hình. Thêm các
**chốt chặn tất định**: câu hỏi nêu con số/mã hiệu mà bài báo không hề nhắc
(«năm 2015», «IP68») thì không cho mô hình trả lời; câu sinh ra có số không có
trong nguồn, hoặc chỉ lặp lại câu hỏi, thì hiển thị câu trích xuất thay thế.

**Ba lần chạy thật trên Colab T4, ghi lại đầy đủ** ở [docs/07](docs/07-rag-phogpt.md).
Bảng dưới là kết quả trên tập **TEST** (33 câu tin tức chưa từng dùng để chỉnh
prompt), chấm tay từng câu đối chiếu bài gốc:

| Chất lượng câu trả lời | prompt v1 | **prompt v2** |
|---|---|---|
| đúng, trả lời được | 9 | **19** |
| đúng nhưng trình bày hỏng | 10 | **2** |
| không trả lời (lặp lại câu hỏi) | 8 | 8 |
| **có thông tin sai** | **3** | **4** |
| từ chối sai | 3 | **0** |

Prompt v2 hơn v1 rõ rệt về mức dùng được (11 câu thắng / 1 thua, kiểm định dấu
p = 0.006) — nhưng **thông tin sai không giảm**. RAG chỉ đổi kiểu sai: v1 sai lộ
liễu (bịa ngày, từ chối nhầm), v2 sai trôi chảy và khó phát hiện hơn.

Trên **11 câu bẫy mới** (hỏi chi tiết bài không có / giả định sai, chốt trước khi
chạy): v1 và v2 hòa 6 an toàn / 5 sai, nhưng vì lý do trái ngược. v2 an toàn **nhờ
chốt chặn** ở các bẫy có con số; với giả định sai **không có số** ("robot Optimus
của Tesla tự bước ra khỏi dây chuyền" — bài nói Tesla chưa làm được) thì v2 khẳng
định luôn, còn v1 lại tình cờ từ chối. Ca tin mâu thuẫn: v2 **3/3** theo tin mới.
Câu ngoài phạm vi: 21/24 không bao giờ tới mô hình.

**Kết luận:** RAG làm câu trả lời tự nhiên hơn hẳn nhưng **không an toàn hơn** bot
trích xuất — bot trích xuất không bao giờ khẳng định điều bài báo không nói. Vì vậy
RAG là lớp **tùy chọn, mặc định tắt**; chatbot chính vẫn là bản trích xuất.

```powershell
python tools/make_colab_bundle.py   # tạo dist/rag_bundle.zip (mã nguồn + dữ liệu)
python tests/test_rag.py            # 29 kiểm thử RAG, không cần GPU (backend giả lập)
python tools/rescore_rag_run.py     # chấm lại một lần chạy Colab bằng mã hiện tại
```

Chạy trên Colab: mở `dist/RAG_PhoGPT_Colab.ipynb` trong Colab → Runtime → T4 GPU
→ Run all → tải lên `dist/rag_bundle.zip` khi được hỏi. Notebook chạy cùng bộ câu
hỏi qua prompt v1 và v2, tự chấm độ trung thành (số bịa), ca tin mâu thuẫn, câu
bẫy (bộ riêng cho từng tập, `data/eval/rag/*_traps.json`) và câu ngoài phạm vi,
rồi tải file kết quả về máy.

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
│   ├── intents/intents_vi.json      # 14 intent, 164 pattern
│   ├── eval/dev.json                # tập DEV — để dò tham số
│   ├── eval/test.json               # tập TEST — chỉ để báo cáo
│   ├── eval/conflict_case.json      # ca tin mâu thuẫn (ràng buộc cứng)
│   ├── eval/test_report_v*.txt      # báo cáo từng lần xem tập test
│   ├── eval/tuned_params.json       # tham số dò trên dev
│   ├── eval/test_results.json       # số liệu báo cáo trên test
│   ├── eval/rag/run{1,2,3}_*        # kết quả thật 3 lần chạy PhoGPT trên Colab
│   ├── eval/rag/*_traps.json        # câu bẫy dev / test (bộ test chốt trước khi chạy)
│   ├── eval/rag/run3_*annotation.csv  # nhãn chấm tay lần chạy 3 (tập test)
│   ├── resources/vietnamese-stopwords.txt
│   ├── resources/teencode_lexicon.json  # học được từ ViLexNorm
│   └── resources/vilexnorm/         # corpus chuẩn hóa (CC BY-NC-SA 4.0)
├── report/                                        # BÁO CÁO đồ án (VI + EN)
├── notebooks/FinalProject_Chatbot_23IT036.ipynb   # notebook báo cáo (đã chạy)
├── notebooks/RAG_PhoGPT_Colab.ipynb              # RAG trên Colab
├── docs/
│   ├── 01-nghien-cuu-du-an-tham-khao.md
│   ├── 02-kien-truc.md
│   ├── 03-chuan-hoa-teencode.md
│   ├── 04-thi-nghiem-sinh-van-ban.md
│   ├── 05-do-moi-va-thong-tin-loi-thoi.md
│   ├── 06-danh-gia-trung-thuc-va-bm25.md
│   ├── 07-rag-phogpt.md
│   └── 08-runbook-cap-nhat-du-lieu.md   # crawl hằng ngày + cập nhật mô hình
├── tests/test_vectorizer.py         # TF-IDF vs sklearn, BM25 vs tham chiếu
├── tests/test_chatbot.py            # 21 kiểm thử hồi quy
├── tests/test_rag.py                # 29 kiểm thử RAG (không cần GPU)
├── tools/build_eval_sets.py         # sinh tập dev/test
├── tools/build_notebook.py          # sinh notebook báo cáo
├── tools/build_rag_notebook.py      # sinh notebook RAG cho Colab
├── tools/make_colab_bundle.py       # đóng gói mã + dữ liệu cho Colab
├── tools/rescore_rag_run.py         # chấm lại kết quả Colab bằng mã hiện tại
├── tools/daily_update.ps1           # cập nhật hằng ngày: sao lưu, crawl, dựng index, kiểm thử
├── tools/rebuild_index.py           # dựng lại index + kiểm tra khói
└── requirements.txt
```

---

## Kiến thức từ các lab được dùng ở đâu

| Lab | Nội dung | Dùng ở đâu |
|---|---|---|
| **Lab 01** | `sent_tokenize`, `word_tokenize`, `ner`, Regex | Tách câu chọn snippet (`retriever.py`), tách từ (`preprocess.py`), trích thực thể (`entities.py`) |
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
index phụ ở mức **âm tiết**, chỉ dùng khi câu hỏi không có dấu. Trên tập test,
câu không dấu đạt Recall@1 **95.5%** (21/22) — ngang câu có dấu (95.3%).

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
thật. Đã thêm xếp hạng theo độ mới `score = cosine × (1 + α·recency)` — bài mới
được xếp đầu ở **cả 4 cách hỏi** (một cách hỏi, "tàu cát linh 15.000 đồng", có
cosine dưới ngưỡng chấp nhận nên bot trả lời "không tìm thấy" thay vì trả bài). Độ mới chỉ dùng để **xếp hạng**; việc **chấp
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
2. **Không suy luận, không sinh văn bản, không diễn đạt lại** — bot lõi trích
   xuất 100%, chỉ trả về câu đã có sẵn trong corpus. Đây là giới hạn **kiến
   trúc**, đã kiểm chứng bằng thực nghiệm ở [docs/04](docs/04-thi-nghiem-sinh-van-ban.md).
   Lớp RAG tùy chọn diễn đạt lại được, nhưng không trung thực hơn (docs/07).
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
9. **Tầng chọn câu có thể bỏ sót câu trả lời** — truy hồi đúng bài nhưng chỉ vài
   câu có cosine cao nhất được đưa ra; thí nghiệm RAG gặp ca câu liệt kê màu
   iPhone không nằm trong các câu được chọn (docs/07, mục 11).
10. **Câu hỏi xác nhận bị định tuyến sai** — "...đúng không", "...phải không" hay
   bị intent classifier xếp nhầm, nên không tới được nhánh truy hồi.

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
