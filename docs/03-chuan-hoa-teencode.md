# Xử lý tiếng Việt phi chuẩn: không dấu và teencode

Tài liệu này ghi lại cách tiếp cận và **lý do** đằng sau từng quyết định, kể cả
những hướng đã thử rồi bỏ.

## Bối cảnh

Chatbot ban đầu chỉ hiểu tiếng Việt viết chuẩn. Nhưng người dùng thật gõ chat
theo ba kiểu lệch chuẩn, theo thứ tự mức độ khó tăng dần:

| Kiểu | Ví dụ | Mức độ phá vỡ pipeline |
|---|---|---|
| Bỏ dấu | `tin ve dao hai nam` | Nặng — tách từ sai, mọi term thành OOV |
| Teencode / viết tắt | `bt gì về vụ iphone k b` | **Rất nặng** — thay hẳn mặt chữ |
| Lặp ký tự | `đẹppppp quá trờiii` | Nhẹ — chỉ cần gộp ký tự lặp |

---

## Vấn đề 1 — Câu không dấu

### Vì sao không thể chỉ "bỏ dấu rồi tách từ như thường"

`word_tokenize` của underthesea được huấn luyện trên tiếng Việt **có dấu**.
Đưa câu không dấu vào, nó tách sai hoàn toàn. Đo trên chính dự án này:

```text
word_tokenize("tin ve dao hai nam")   ->  ['ve_dao', 'hai', 'nam']
```

Nó dính `ve dao` thành một từ và cắt rời `hai nam`. Mọi bước sau đó đều hỏng.

### Giải pháp: index phụ ở mức ÂM TIẾT

Không cố **khôi phục** dấu (bài toán khó, cần mô hình seq2seq riêng) mà hạ
**cả hai phía** xuống cùng một mặt phẳng âm tiết không dấu:

- **Phía document:** vẫn tách từ trên bản có dấu (chính xác), rồi mới bỏ dấu
  và tách từ ghép ra âm tiết — `["đảo","hải_nam"]` → `["dao","hai","nam"]`.
- **Phía query:** bỏ hẳn bước tách từ, chỉ cắt theo khoảng trắng.
- Bigram trong TF-IDF khôi phục phần lớn thông tin từ ghép đã mất
  (`"hai nam"` xuất hiện như một bigram).

### Vì sao giữ HAI index tách biệt thay vì trộn chung

Đã cân nhắc phương án trộn cả dạng có dấu và không dấu vào một index. Bỏ vì:

1. Vocabulary gấp đôi (105k → ~200k term).
2. Điểm cosine bị pha loãng → **mọi ngưỡng đã dò công phu trước đó mất hiệu lực**.
3. Sinh nhập nhằng: `hai` khớp cả `hải`, `hai`, `hài`.

Giữ tách biệt, chỉ chuyển sang index phụ khi câu **không có dấu nào** — một
tín hiệu rất rõ ràng, vì tiếng Việt thật gần như luôn có ít nhất một dấu.

### Kết quả ngoài dự kiến

Thay đổi này làm retrieval **tốt lên**, không chỉ thêm tính năng:

| | Trước | Sau |
|---|---|---|
| Recall@1 | 85.7% | **90.5%** |
| Recall@3 | 95.2% | **100%** |
| MRR | 0.905 | **0.952** |

Lý do: các truy vấn chứa tên riêng nước ngoài (`"champions league man utd"`)
vốn **không có dấu nào**, nên trước đây bị tách từ sai. Nay chúng đi qua index
âm tiết và khớp đúng.

---

## Vấn đề 2 — Teencode

Nặng hơn hẳn trường hợp không dấu: `k` và `không` **không có ký tự nào chung**,
nên không mẹo so khớp bề mặt nào cứu được. Bắt buộc phải có **bảng ánh xạ**.

### Vì sao HỌC từ dữ liệu thay vì tự liệt kê

Tự ngồi liệt kê vài trăm từ teencode thì vừa thiếu, vừa mang thiên kiến cá nhân
của người viết, và **không đo được**. Thay vào đó ta học bảng ánh xạ từ
**ViLexNorm** — 10.467 cặp câu (teencode → chuẩn) do con người gán nhãn, lấy
từ bình luận mạng xã hội Việt Nam thật.

### Thuật toán

1. Với mỗi cặp câu, nếu hai bên **bằng nhau về số token** thì căn theo vị trí.
   79,5% số cặp thỏa điều kiện này — đủ để học.
2. Đếm mọi ánh xạ `a → b` tại các vị trí có thay đổi.
3. Chỉ giữ `a → b` khi thỏa **cả ba** điều kiện an toàn:

| Điều kiện | Ngưỡng | Vì sao cần |
|---|---|---|
| `a` xuất hiện đủ nhiều | `count >= 4` | Tránh học từ nhiễu gán nhãn |
| `a` **thường** bị đổi | `tỷ lệ đổi >= 0.5` | **Chốt chặn quan trọng nhất** |
| `b` chiếm ưu thế | `>= 0.5` | Tránh chọn bừa khi `a` mơ hồ |

**Vì sao điều kiện thứ hai là quan trọng nhất:** thiếu nó, những từ thường như
`cả`, `mà` sẽ bị thay bừa chỉ vì đôi khi chúng tình cờ đứng ở vị trí có thay
đổi trong câu. Đây là dạng lỗi âm thầm và rất khó phát hiện — nó không làm
chương trình chết, chỉ làm chất lượng tụt dần.

### Nhập nhằng được ghi nhận, không giấu

Dữ liệu có mâu thuẫn thật: `t → tôi` (889 lần) và `t → tao` (132 lần). Ta chọn
`tôi` vì chiếm ưu thế, và **chấp nhận sai** ở những câu vốn nói "tao".

Đây là giới hạn cố hữu của chuẩn hóa ở mức **từ đơn lẻ, không xét ngữ cảnh**.
Muốn tốt hơn phải dùng mô hình seq2seq có ngữ cảnh — ngoài phạm vi kỹ thuật
của đồ án.

### Kết quả (split test, chưa từng thấy khi học)

| Chỉ số | Giá trị |
|---|---|
| Accuracy trước (không làm gì) | 83.88% |
| Accuracy sau | **94.77%** |
| **ERR** (tỷ lệ giảm lỗi) | **67.54%** |
| Precision / Recall | 90.71% / 70.16% |

> **Vì sao báo cáo ERR chứ không chỉ accuracy:** ~84% token vốn đã đúng sẵn,
> nên accuracy thô bị thổi phồng. ERR chỉ đo phần lỗi thực sự được sửa, và là
> chỉ số chuẩn của bài toán lexical normalization.

**Precision (90.7%) cao hơn hẳn Recall (70.2%)** — đây là đánh đổi có chủ đích.
Ba điều kiện an toàn ở trên khiến từ điển thận trọng: nó bỏ sót một số từ
teencode hiếm, nhưng gần như không sửa sai từ đã đúng. Với chatbot, sửa hỏng
một từ vốn đúng gây hại nhiều hơn là bỏ sót một từ lạ.

### Giới hạn của con số này

ERR chỉ tính trên các cặp **căn được theo vị trí** (79,5%). Các cặp mà chuẩn
hóa làm thay đổi số token (tách/gộp từ) bị loại khỏi cả huấn luyện lẫn đánh
giá. Con số thật trên toàn bộ dữ liệu sẽ **thấp hơn**. Ghi nhận rõ ở đây để
không so sánh khập khiễng với các kết quả công bố trên toàn tập.

---

## Hệ quả không lường trước: chuẩn hóa làm LOÃNG vector truy hồi

Sau khi bật chuẩn hóa teencode, một số câu **vẫn** trượt — thậm chí câu đã được
chuẩn hóa đúng hoàn toàn:

```text
"bt gì về vụ iphone k b"  ->  "biết gì về vụ iphone không bạn"  ->  fallback (!)
```

### Nguyên nhân

Chuẩn hóa **bung** từ viết tắt thành từ đầy đủ, tức là **thêm token khung** vào
câu. Vector query được chuẩn hóa L2, nên mỗi token thừa đều chia bớt trọng số
của token thực sự quan trọng:

| Câu | Điểm top-1 |
|---|---|
| `giá iphone` | **0.167** (đạt ngưỡng 0.12) |
| `biết gì về vụ iphone không bạn` | **0.100** (trượt) |

Cùng một ý định, chỉ khác cách diễn đạt hội thoại.

### Giải pháp: `QUERY_FRAME_WORDS`

Lọc các từ chỉ đóng vai trò **khung câu hỏi** (`tin`, `biết`, `vụ`, `xem`,
`bạn`, `mình`...) trước khi dựng vector truy hồi.

Các từ này **không** nằm trong stopword list chuẩn, vì trong văn bản thường
chúng vẫn là từ nội dung (`tin` trong "bản tin", `biết` trong "hiểu biết"). Nên
đây là một danh sách riêng, chỉ áp dụng cho **query**, không áp dụng khi index
document.

**Chốt an toàn:** nếu lọc hết sạch thì trả lại nguyên bản. Câu như
`"có tin gì mới không"` toàn từ khung — bỏ hết sẽ thành vector rỗng và bot mất
luôn khả năng trả lời, tệ hơn là cứ để nguyên.

---

## Hai lỗi phát hiện khi kiểm thử

### Lỗi 1 — Hai danh sách từ khung lệch nhau

`chatbot.py` và `retriever.py` từng giữ hai danh sách riêng. Chúng lệch nhau:
`biết` có ở danh sách của retriever nhưng thiếu ở chatbot. Hậu quả: câu
`"không biết tin sức khỏe gì luôn"` bị coi là **có chủ đề cụ thể**, đem đi tìm
kiếm rồi trượt.

→ Gộp về một nguồn duy nhất trong `config.py`.

**Bài học:** cùng một khái niệm không được có hai định nghĩa ở hai nơi.

### Lỗi 2 — Tách từ không ổn định theo ngữ cảnh

Đây là lỗi tinh vi hơn nhiều:

```text
word_tokenize("Sức khỏe")                    ->  ["sức", "khỏe"]
word_tokenize("... tin sức khỏe gì luôn")    ->  ["sức_khỏe"]
```

Cùng một cụm, tách khác nhau tùy ngữ cảnh xung quanh. Nên phép so
`"sức_khỏe" ∈ {"sức","khỏe"}` luôn sai, và câu chỉ nêu đúng tên chuyên mục lại
bị hiểu nhầm là có chủ đề riêng.

→ Chuyển sang so khớp ở mức **âm tiết** (tách theo `_`) thay vì so nguyên token.

**Bài học:** không được giả định bộ tách từ cho ra kết quả giống nhau cho cùng
một cụm ở hai ngữ cảnh khác nhau.

---

## Thứ tự pipeline sau khi bổ sung

```text
câu người dùng
    |
    v
[0] TeencodeNormalizer      <- MỚI: "bt" -> "biết", "k" -> "không"
    |                          phải chạy TRƯỚC tách từ
    v
[1] entities.extract()         NER + Regex + nhận diện chuyên mục
    |
    v
[2] preprocess_vi()            NFC -> tách từ -> lowercase -> dấu câu
    |
    v
[3] strip_frame_words()     <- MỚI: bỏ từ khung, chỉ cho nhánh truy hồi
    |
    v
[4] IntentClassifier / NewsRetriever
```

## Nguồn dữ liệu

**ViLexNorm** — Nguyen et al., *ViLexNorm: A Lexical Normalization Corpus for
Vietnamese Social Media Text*, EACL 2024.
<https://github.com/ngxtnhi/ViLexNorm>
Giấy phép **CC BY-NC-SA 4.0** — chỉ dùng cho mục đích nghiên cứu/học tập,
phải ghi nguồn, và không được đổi sang giấy phép khác.
