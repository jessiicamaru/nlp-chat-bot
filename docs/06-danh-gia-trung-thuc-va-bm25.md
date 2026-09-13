# Đánh giá trung thực: tách DEV/TEST, kiểm thử hồi quy, và BM25

Tài liệu này ghi lại một đợt sửa **phương pháp đánh giá**, và những gì thay đổi
khi các con số được đo đúng cách. Nhiều kết luận ở các tài liệu trước (docs/03,
docs/05) bị chính đợt này **bác bỏ** — giữ nguyên các tài liệu đó để thấy lịch
sử, nhưng số liệu hiện hành là số liệu ở đây.

---

## 1. Lỗi phương pháp: dò tham số và báo cáo trên cùng một tập

Trước đợt này, mọi siêu tham số — `w_nb`, `INTENT_THRESHOLD`,
`RETRIEVAL_THRESHOLD`, `alpha` và nửa chu kỳ độ mới — đều được chọn bằng cách
quét lưới trên `test_queries.json`. Sau đó các con số trong báo cáo (88.5%,
96.8%, MRR 0.984) lại được đo trên **chính tập đó**.

Đó là **rò rỉ tập test**: con số phản ánh mức "khớp" với đúng những câu đã dùng
để dò, chứ không phải hiệu năng trên câu hỏi chưa thấy bao giờ. Người chấm cẩn
thận sẽ nhận ra.

Thêm vào đó, các tập quá nhỏ: 31 truy vấn truy hồi, 26 câu intent. Một câu sai
đã làm Recall@1 dao động 3,2 điểm phần trăm.

## 2. Cách làm mới

| | DEV | TEST |
|---|---|---|
| Dùng để | dò tham số, xem bao nhiêu lần cũng được | **chỉ báo cáo** |
| Truy hồi | 112 | 122 |
| Ngoài phạm vi | 28 | 24 |
| Intent | 55 | 32 |

- **Toàn bộ truy vấn cũ** (đã từng dùng để dò) bắt buộc vào DEV — chúng đã
  "nhiễm", không còn là câu chưa thấy.
- Truy vấn **mới** viết thêm được chia ngẫu nhiên có hạt giống cố định
  (`tools/build_eval_sets.py`) để tái lập được.
- **Nhãn đúng theo URL bài báo**, không theo "tiêu đề chứa từ X". Tập cũ chấm
  đúng khi tiêu đề chứa `iPhone` — có hàng chục bài như vậy.
- Truy vấn đi qua **đúng đường xử lý của chatbot** (`prepare_user_text`).
  Trước đây `evaluate.py` gọi thẳng retriever và bỏ qua chuẩn hóa teencode.
- Mọi tỷ lệ kèm **khoảng tin cậy Wilson 95%**.
- Thêm **đánh giá đầu-cuối**: gọi `bot.respond()` như người dùng thật, tính
  cả việc intent classifier định tuyến sai.

### Giới hạn cần nói rõ

Người viết truy vấn cũng là người xây bot, nên truy vấn có thể mang cùng "điểm
mù" với bot. Để giảm thiểu, truy vấn được viết từ phần **mô tả** bài báo và
diễn đạt lại, không chép tiêu đề. Cách tốt nhất vẫn là nhờ người khác viết thêm.

---

## 3. Kết quả thật trên TEST

| Hạng mục | Chỉ số | Kết quả | KTC 95% | Trước đây báo cáo |
|---|---|---|---|---|
| Truy hồi | Recall@1 | **93.4%** (114/122) | 88–97% | 96.8% |
| Truy hồi | Recall@3 | **95.9%** (117/122) | 91–98% | 100% |
| Truy hồi | MRR | **0.950** | — | 0.984 |
| Truy hồi — có dấu | Recall@1 | 95.3% (82/86) | 89–98% | — |
| Truy hồi — không dấu | Recall@1 | 95.5% (21/22) | 78–99% | — |
| Truy hồi — teencode | Recall@1 | **78.6%** (11/14) | 52–92% | — |
| Chặn câu ngoài phạm vi | tỷ lệ | 87.5% (21/24) | 69–96% | 100% |
| Intent | Accuracy | **61.5%** (16/26) | 43–78% | 88.5% |
| Intent | Macro-F1 | 0.712 | — | 0.91 |
| **Đầu-cuối** | câu tin tức → bài đứng đầu đúng | **77.0%** (94/122) | 69–84% | — |
| **Đầu-cuối** | câu ngoài phạm vi → bot từ chối | **75.0%** (18/24) | 55–88% | — |

### Đọc bảng này thế nào

**Truy hồi vẫn tốt** (93%), giảm vừa phải so với con số rò rỉ.

**Intent classifier là điểm yếu thật sự**, không phải truy hồi. Cả 10 câu sai
đều cùng một kiểu: độ tin cậy **thấp hơn ngưỡng** (0.14–0.25) nên bị từ chối,
dù nhãn dự đoán có thể đúng. Các câu này ngắn và cách diễn đạt không có trong
164 pattern huấn luyện: `"ừm"`, `"đúng vậy"`, `"thanks nhé"`, `"ngu thế"`.

**Teencode là chỗ yếu của truy hồi** — 78.6%, nhưng chỉ có 14 câu nên khoảng
tin cậy rất rộng (52–92%). Các từ như `vc` (việc), `ntn` (như thế nào) không có
trong từ điển học từ ViLexNorm.

**Khoảng cách thành phần → đầu-cuối là 16 điểm** (93.4% → 77.0%). Trong 28 câu
đầu-cuối không đạt: 18 câu bot trả lời "không tìm thấy" vì cosine dưới ngưỡng,
7 câu bị intent classifier hiểu nhầm (ví dụ thành "duyệt chuyên mục"), 3 câu ra
bài sai. Ngưỡng 0.13 là một **đánh đổi có chủ đích**: dò trên dev nó chặn đúng
100% câu ngoài phạm vi, đổi lại từ chối ~11% câu trả lời được.

---

## 4. Hai kết luận cũ bị bác bỏ

### 4.1. "Độ mới làm tốt lên chất lượng truy hồi" — SAI

docs/05 từng ghi: thêm độ mới làm Recall@1 tăng 93.5% → 96.8%. Trên dev 112 câu,
độ mới **không cải thiện một cách nhất quán**: tắt độ mới cho MRR 0.965; vài cấu
hình tăng nhẹ (tốt nhất là nửa chu kỳ 30 ngày, alpha = 0.2: 0.969 — chênh khoảng
một câu), phần lớn giảm nhẹ, và **mọi cấu hình xử lý đúng ca tin mâu thuẫn đều
thấp hơn một chút** (0.957–0.964). Tất cả chênh lệch đều nằm trong phạm vi
nhiễu. Mức tăng 3 điểm đo trước đó là **nhiễu của tập test nhỏ**.

Độ mới vẫn được giữ, vì nó là cách **duy nhất** xử lý đúng ca tin mâu thuẫn —
nhưng nay được ghi đúng bản chất: một **đánh đổi** (mất ~0.001 MRR để không trả
lời bằng tin đã lỗi thời), không phải một cải tiến miễn phí.

### 4.2. "Ensemble NB + cosine tốt hơn NB thuần" — không còn đúng

Dò lại trên dev, `w_nb = 1.0` (Naive Bayes thuần) cho điểm cân bằng cao nhất.

---

## 5. Kiểm thử hồi quy bắt được hai lỗi thiết kế

`tests/test_chatbot.py` có 21 kiểm thử, mỗi cái ứng với một lỗi thật từng làm
bot trả lời sai mà **không báo lỗi gì**. Ngay lần chạy đầu tiên sau khi áp tham
số mới dò trên dev, chúng bắt được hai lỗi.

### 5.1. Độ mới âm thầm biến thành bộ lọc loại bài cũ

Ngưỡng chấp nhận áp lên điểm **đã nhân** hệ số độ mới. Với nửa chu kỳ 3 ngày
vừa dò được, bài 16 ngày tuổi gần như không được thưởng, nên phải liên quan hơn
hẳn mới vượt ngưỡng. `"tin ve dao hai nam"` và `"cho t hỏi vụ hải nam vs"` từ
trả lời đúng thành "không tìm thấy".

**Sửa:** tách hai câu hỏi khác nhau ra hai thước đo khác nhau.

| Câu hỏi | Thước đo |
|---|---|
| Bài nào nên đứng trước? | độ liên quan **×** độ mới |
| Có đủ căn cứ để trả lời không? | **chỉ** độ liên quan (cosine) |

Dò lại ngưỡng trên thang cosine: **0.13** (dev: trả lời được 88.7%, chặn đúng
100% câu ngoài phạm vi).

### 5.2. Cùng một câu, kết quả tùy nhánh

Ngưỡng nới lỏng khi người dùng đã nêu chuyên mục (tập ứng viên co lại nên một
cosine thấp hơn vẫn đủ căn cứ) chỉ tồn tại ở nhánh "duyệt chuyên mục". Với
`w_nb = 1.0`, câu `"tin du lịch ninh bình"` có độ tin cậy intent **0.246** —
dưới ngưỡng 0.25 một chút — nên đi sang nhánh truy hồi, nơi không có ngưỡng
nới, và trả lời "không tìm thấy".

**Sửa:** áp cùng quy tắc ở cả hai nhánh qua một hằng số duy nhất
`CATEGORY_SCOPED_THRESHOLD_FACTOR`.

**Bài học:** một quy tắc nghiệp vụ phải nằm ở một chỗ. Nếu nó chỉ tồn tại ở một
nhánh, kết quả sẽ phụ thuộc vào việc một con số khác tình cờ rơi bên nào ngưỡng.

---

## 6. BM25 — một kết quả âm

### Cài đặt

BM25 (Okapi) viết tay trong `vectorizer.py`, idf biến thể Lucene (luôn dương):

```
score(q, d) = Σ idf(t) · tf(t,d)·(k1+1) / (tf(t,d) + k1·(1 − b + b·|d|/avgdl))
idf(t)      = ln( (N − df + 0.5) / (df + 0.5) + 1 )
```

sklearn không có BM25, nên để kiểm chứng, bản vector hóa (ma trận thưa) được
đối chiếu với một bản **cài đặt ngây thơ viết thẳng từ định nghĩa** bằng vòng
lặp, trên 4 cặp (k1, b) — khớp tuyệt đối. Thêm một kiểm thử tính chất: lặp một
term nhiều lần thì điểm tăng nhưng tiến tới trần `idf·(k1+1)`.

Điểm BM25 không bị chặn trên và không so được giữa hai câu hỏi, nên BM25 chỉ
dùng để **xếp hạng**; việc **chấp nhận** vẫn dựa trên cosine TF-IDF.

### Kết quả trên DEV (độ mới tắt)

| Cách | k1 | b | Recall@1 | MRR |
|---|---|---|---|---|
| TF-IDF | — | — | 94.6% | 0.965 |
| BM25 tốt nhất | 8.0 | 0.9 | 95.5% | 0.967 |

MRR của BM25 theo k1 (b = 0.9):

```
k1 = 0.6   0.9   1.2   1.5   2.0   3.0   5.0   8.0
MRR  0.942 0.942 0.944 0.946 0.954 0.959 0.960 0.967
```

### Vì sao BM25 không hơn — giả thuyết được dữ liệu ủng hộ

Index tăng trọng số tiêu đề bằng cách **lặp tiêu đề 3 lần** (và mô tả 2 lần).
Mẹo này chỉ hiệu quả khi tf còn tăng theo số lần lặp. **Độ bão hòa tf** — đặc
điểm cốt lõi của BM25 — triệt tiêu đúng hiệu ứng đó.

Nếu giả thuyết đúng, BM25 phải tốt dần khi **giảm bão hòa** (tăng k1). Dữ liệu
cho thấy đúng như vậy: MRR tăng đều từ 0.942 lên 0.967. Ở k1 lớn, BM25 gần như
quay lại hành vi của TF-IDF.

Cách làm đúng là **BM25F**: bão hòa riêng từng trường (tiêu đề, mô tả, thân bài)
rồi mới cộng có trọng số — ghi vào hướng phát triển.

### Quyết định: giữ TF-IDF — và một quy tắc quyết định bị sửa

Quy tắc ban đầu là "BM25 hơn TF-IDF trên dev thì đổi". Nó chọn BM25 với chênh
lệch **0.002 MRR** (khoảng một câu). Trên test, BM25 lại **thua**:

| Trên TEST | Recall@1 | MRR |
|---|---|---|
| TF-IDF | 93.4% (88–97%) | 0.950 |
| BM25 (k1=8, b=0.9) | 91.8% (86–95%) | 0.943 |

Chênh lệch 0.002 với khoảng tin cậy rộng cả chục điểm là nhiễu. Quy tắc được
sửa thành: **phương pháp phức tạp hơn phải thắng CÓ Ý NGHĨA THỐNG KÊ trên dev**
— kiểm định dấu có cặp trên từng câu:

```
BM25 tốt hơn ở 3 câu, kém hơn ở 3 câu, hòa 106 câu  ->  p = 1.000
```

Không có khác biệt → giữ TF-IDF vì đơn giản hơn.

**Minh bạch:** quy tắc kiểm định được thêm **sau khi** đã thấy kết quả test
BM25. Đó là nguyên tắc chuẩn chứ không phải tham số dò theo test, và dưới cả
hai quy tắc thì kết luận vẫn như nhau: trên test, hai phương pháp không khác
nhau có ý nghĩa (khoảng tin cậy chồng lấn gần như hoàn toàn).

---

## 7. Số lần đã xem tập TEST

Để minh bạch, liệt kê mọi lần kết quả test được xem:

| Lần | Lý do | Có đổi gì dựa trên test không? |
|---|---|---|
| 1 | Báo cáo đầu tiên sau khi tách dev/test | Không |
| 2 | Sau hai sửa lỗi do **kiểm thử hồi quy** phát hiện | Không — lỗi đến từ test hồi quy, tham số vẫn dò trên dev |
| 3 | Sau khi thêm BM25 | **Có** — quy tắc chọn phương pháp đổi sang kiểm định thống kê sau khi thấy BM25 thua trên test |

Báo cáo chi tiết từng lần: `data/eval/test_report_v1.txt`, `_v2.txt`, `_v3.txt`.

### Những lần tập TEST được dùng về sau, cho lớp RAG (docs/07)

Sau lần 3, tham số của chatbot **không đổi nữa** (sửa lỗi gộp chữ số của bộ chuẩn
hóa teencode chỉ chạm 0/178 câu test, và dò lại trên dev cho đúng bộ tham số cũ —
docs/07, mục 6). Tập test chỉ được dùng thêm để đánh giá lớp RAG:

| Lần | Việc | Có đổi gì dựa trên test không? |
|---|---|---|
| 4 | Đo chốt chặn giả định có chặn nhầm câu hợp lệ không (không cần LLM) | Không |
| 5 | Đo lại chỉ số trên sau khi sửa quy tắc so số có đơn vị (quy tắc sửa từ suy luận, không từ dữ liệu test) | Không |
| 6 | Lần chạy RAG 3 trên Colab: 40 câu tin tức + 24 câu ngoài phạm vi, mã đóng băng ở commit `fe10190`, bẫy mới chốt trước | Không — kết quả chỉ được báo cáo |

Ngoài ra, notebook báo cáo (Phần E) chạy lại toàn bộ `evaluate.py` mỗi khi notebook
được thực thi lại, nên báo cáo test được **tính lại** với đúng bộ tham số đã chốt.
Các lần tính lại cho ra `test_results.json` giống hệt từng byte và không dẫn tới
quyết định nào.

## 8. Việc nên làm tiếp (theo mức ưu tiên do số liệu chỉ ra)

1. **Intent classifier** — điểm yếu lớn nhất (61.5%). Thêm pattern cho câu ngắn
   và cách diễn đạt đời thường; cân nhắc hiệu chỉnh xác suất (calibration)
   thay vì một ngưỡng cứng. Phải dò trên dev và đo trên một tập test **mới**,
   vì tập test hiện tại đã bị xem 3 lần.
2. **Teencode trong truy hồi** (78.6%) — bổ sung các viết tắt còn thiếu
   (`vc`, `ntn`, `tk`, `ks`...).
3. **BM25F** — thay mẹo lặp tiêu đề bằng trọng số trường thật sự.
4. **Tập đánh giá do người khác viết** — giảm thiên kiến người-xây-bot-tự-chấm.

Hai việc được thêm vào sau thí nghiệm RAG (docs/07):

5. **Tầng chọn câu** — hiện lấy cố định vài câu có cosine cao nhất. Câu hỏi "iPhone
   18 Pro Max có mấy màu" truy hồi đúng bài nhưng câu liệt kê màu không nằm trong
   các câu được chọn. Lỗi này ảnh hưởng cả bot trích xuất lẫn RAG.
6. **Câu hỏi xác nhận** ("...đúng không", "...phải không") — 4/11 câu bẫy viết tự
   nhiên không tới được nhánh truy hồi vì bị intent classifier xếp nhầm. Đây là
   thêm một bằng chứng cho mục 1.
