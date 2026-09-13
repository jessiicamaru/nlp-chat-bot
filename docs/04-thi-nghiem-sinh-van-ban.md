# Thí nghiệm đối chứng: TRÍCH XUẤT hay SINH văn bản?

## Câu hỏi đặt ra

Chatbot hiện tại hoạt động theo kiểu **trích xuất** (extractive): nó chỉ trả về
những câu đã có sẵn trong corpus. Một câu hỏi hoàn toàn hợp lý:

> Sao không để mô hình **tự sinh** câu trả lời cho tự nhiên hơn, thay vì chép
> nguyên văn từ bài báo?

Tài liệu này trả lời bằng **thực nghiệm**, không bằng lời khẳng định.

## Điều cần nói rõ trước

TF-IDF + cosine similarity là một hàm **đo độ giống nhau**. Nó không có bất kỳ
cơ chế nào để tạo ra từ mới. Không có tham số nào chỉnh được để nó biết sinh
văn bản — đây là giới hạn **kiến trúc**, không phải giới hạn cấu hình.

Vậy nên để trả lời câu hỏi trên, ta phải cài đặt một mô hình **sinh** thật sự.
Chọn mô hình sinh cổ điển nhất, và vẫn nằm trong phạm vi "from scratch":
**n-gram language model** (`src/generator.py`).

## Mô hình

Giả định Markov bậc `n-1` — xác suất một từ chỉ phụ thuộc `n-1` từ ngay trước:

```
P(w_i | w_1...w_{i-1})  ≈  P(w_i | w_{i-n+1}...w_{i-1})
                         =  count(context + w_i) / count(context)
```

**Làm mịn bằng nội suy đệ quy**, vì ước lượng thô gán xác suất 0 cho mọi n-gram
chưa từng thấy:

```
P_interp(w | context) = λ·P_ML(w | context) + (1-λ)·P_interp(w | context[1:])
```

Đo bằng **perplexity** — "số lựa chọn trung bình mà mô hình còn phân vân ở mỗi
bước", càng thấp càng tốt:

```
PP = exp( -(1/N) · Σ log P(w_i | context) )
```

Dữ liệu: 9.567 câu huấn luyện / 1.063 câu kiểm thử, tổng ~217.000 token.

---

## Kết quả 1 — Perplexity TĂNG theo bậc n (ngược trực giác)

| n | Perplexity (λ=0.7) |
|---|---|
| 1 | 1.285 |
| 2 | **819** ← tốt nhất |
| 3 | 1.951 |
| 4 | 5.911 |

Trực giác thông thường "n lớn hơn thì mô hình mạnh hơn" **không đúng** ở đây.

### Đã kiểm chứng nguyên nhân, không đoán

Giả thuyết: dữ liệu quá thưa, nên thành phần bậc cao gần như luôn bằng 0 và
công thức nội suy nhân thêm hệ số `(1-λ)` ở **mỗi** lần lùi bậc. Với λ=0.7 và
phải lùi hai bậc, xác suất bị nhân với `0,3 × 0,3 = 0,09` — phạt rất nặng.

Kiểm chứng bằng cách quét λ:

| λ | n=2 | n=3 | n=4 |
|---|---|---|---|
| 0.3 | 715 | 844 | **1.173** |
| 0.5 | 716 | 1.113 | 2.109 |
| 0.7 | 819 | 1.951 | 5.911 |
| 0.9 | 1.282 | 7.624 | **63.318** |

Hạ λ từ 0,9 xuống 0,3 làm perplexity của n=4 giảm **54 lần** (63.318 → 1.173).
Điều này xác nhận đúng cơ chế đã nêu.

Nhưng ở **mọi** λ, `n=2` vẫn tốt nhất → với lượng dữ liệu này, bigram là điểm
dừng hợp lý. Muốn dùng bậc cao hơn thì phải đổi sang làm mịn tốt hơn
(Kneser-Ney, backoff Katz), chứ **không phải** chỉ tăng n.

---

## Kết quả 2 — n càng lớn, "sinh" càng biến thành "chép"

Ba mẫu sinh ra ở `n=4` (cùng seed "du lịch"):

```text
[1] du lịch phú quốc thành lập năm 2014 ở hàng châu , tập trung phát triển
    ôtô điện tích hợp công nghệ phần mềm , trí tuệ nhân tạo , robot ...
[2] du lịch phú quốc thành lập năm 2014 ở hàng châu , tập trung phát triển
    ôtô điện tích hợp công nghệ phần mềm , trí tuệ nhân tạo , khoa học dữ liệu ...
[3] du lịch phú quốc thành lập năm 2014 , cô đã 12 lần vô địch médoc ...
```

Cả ba đều mở đầu bằng **đúng một chuỗi giống hệt nhau**. Nguyên nhân: với n=4,
phần lớn ngữ cảnh chỉ xuất hiện **đúng một lần** trong corpus, nên chỉ có duy
nhất một từ kế tiếp khả dĩ.

Tức là ở bậc cao, mô hình chép nguyên văn — **không thêm giá trị gì** so với
truy hồi, mà lại **mất khả năng dẫn nguồn**.

---

## Kết quả 3 — Văn bản sinh ra sai sự thật ở mọi bậc n

Trích nguyên văn từ output:

> "du lịch phú quốc còn đang xây dựng các **trung tâm điều trị ebola**"
>
> "du lịch phú quốc thành lập năm 2014, **cô đã 12 lần vô địch médoc**"

Mô hình nối từ theo thống kê, **không có khái niệm về sự kiện**. Nó ghép "du
lịch Phú Quốc" với "trung tâm điều trị Ebola" chỉ vì các cặp từ đó từng đứng
cạnh nhau ở đâu đó trong corpus.

Với một bot **tin tức**, đây là lỗi không thể chấp nhận: người dùng hỏi tin
thật, bot bịa ra sự kiện không có.

---

## Kết luận

| Tiêu chí | Trích xuất (đang dùng) | Sinh bằng n-gram |
|---|---|---|
| Mạch lạc | Hoàn hảo (câu do người viết) | Trôi dạt sau 5–8 từ |
| Đúng sự thật | Luôn đúng (chép từ nguồn) | **Bịa sự kiện** |
| Dẫn nguồn được | Có | **Không** |
| Nghe tự nhiên | Trung bình (văn báo chí) | Kém |
| Ở n cao | — | Suy biến thành chép |

Ở quy mô dữ liệu của đồ án, sinh văn bản bằng n-gram **thua truy hồi trên mọi
tiêu chí quan trọng**. Đây là căn cứ thực nghiệm cho lựa chọn kiến trúc trích
xuất, chứ không phải một giả định ban đầu.

## Muốn vừa sinh tự nhiên vừa đúng sự thật thì cần gì

Cần mô hình ngôn ngữ lớn đã **tiền huấn luyện** (PhoGPT, Vistral, GPT...) kết
hợp truy hồi theo kiểu **RAG**: dùng chính retriever hiện tại để lấy bài báo
liên quan, rồi đưa vào LLM làm ngữ cảnh để nó diễn đạt lại.

Kiến trúc hiện tại đã sẵn sàng cho hướng này — `NewsRetriever` chính là thành
phần "R" của RAG. Phần còn thiếu là "G", và nó nằm ngoài phạm vi "from scratch"
của đồ án vì đòi hỏi mô hình tiền huấn luyện cùng tài nguyên tính toán lớn.

> **Cập nhật sau:** hướng này về sau đã được **thử thật** như một lớp tùy chọn,
> dùng PhoGPT-4B-Chat trên Google Colab — xem [docs/07](07-rag-phogpt.md). Kết
> quả: câu trả lời tự nhiên hơn hẳn nhưng **không trung thực hơn** bản trích xuất
> (trên tập test, số câu chứa thông tin sai không giảm), nên kết luận của tài
> liệu này — chatbot lõi dùng truy hồi — vẫn giữ nguyên.

## Cách chạy lại

```powershell
python src/generator.py
```
