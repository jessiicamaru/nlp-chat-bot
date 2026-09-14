# Thông tin lỗi thời và xếp hạng theo độ mới

> ⚠️ **ĐÍNH CHÍNH (xem [docs/06](06-danh-gia-trung-thuc-va-bm25.md)).** Các số
> liệu trong tài liệu này được đo khi tham số còn được dò và báo cáo trên cùng
> một tập (rò rỉ tập test), trên tập chỉ 21–31 câu. Khi đo lại đúng cách trên
> tập DEV 112 câu:
>
> - Kết luận **"độ mới làm tốt lên truy hồi" là SAI** — độ mới không cải thiện
>   nhất quán (chênh lệch nằm trong nhiễu); các cấu hình xử lý đúng tin mâu thuẫn
>   đều thấp hơn một chút. Độ mới là một **đánh đổi**, không phải cải tiến.
> - Ngưỡng chấp nhận nay áp lên **cosine thuần**, không lên điểm đã nhân độ mới
>   (áp lên điểm có độ mới đã âm thầm loại bài cũ). Giá trị hiện hành: 0.13.
> - Nửa chu kỳ hiện hành: 3 ngày (dò lại trên dev).
>
> Phần phân tích ca tin mâu thuẫn bên dưới vẫn đúng.


## Câu hỏi đặt ra

> Nếu hôm nay tin nói "vấn đề X là A", hôm sau tin nói "vấn đề X là B",
> bot có biết và trả về đúng thông tin mới không?

Đây là câu hỏi nguy hiểm nhất với một chatbot **tin tức**. Trả lời sai chủ đề
thì người dùng nhận ra ngay. Trả lời bằng thông tin **đã lỗi thời**, kèm dẫn
nguồn thật, bằng giọng chắc chắn — thì không ai nhận ra.

## Trả lời ban đầu: KHÔNG, và nó trả về đúng thông tin sai

Dựng lại đúng tình huống đó bằng hai bài báo mâu thuẫn nhau:

| Ngày | Tiêu đề |
|---|---|
| 01/09 | Giá vé tàu Cát Linh **tăng lên 15.000 đồng** từ tháng 10 |
| 10/09 | **Hoãn tăng giá** vé tàu Cát Linh, giữ nguyên 8.000 đồng |

Hỏi `"giá vé tàu cát linh bao nhiêu"` với TF-IDF thuần:

```text
0.5269  [NGÀY 1 — cũ, nay đã SAI]   <- bot trả lời bài này
0.4302  [NGÀY 10 — mới, ĐÚNG]
```

*(Các con số trong tài liệu này tái lập được bằng Phần G4 của notebook.)*

**Vì sao bài cũ thắng:** tiêu đề của nó chứa đúng các từ trong câu hỏi
(`giá vé tàu Cát Linh`), còn bài mới mở đầu bằng `Hoãn tăng giá`. TF-IDF chỉ
đo độ trùng lặp từ ngữ — nó **không biết** bài nào mới hơn, không biết bài B
phủ định bài A, và không có khái niệm "thông tin bị thay thế".

Nguyên nhân gốc: `published_at` được crawler thu thập và lưu vào corpus, nhưng
**chưa từng được dùng** ở bất kỳ đâu trong xếp hạng.

### Tệ hơn: kết quả không ổn định theo cách diễn đạt

| Câu hỏi | Trả về |
|---|---|
| `giá vé tàu cát linh bao nhiêu` | ❌ bài cũ |
| `vé tàu cát linh có tăng giá không` | ✅ bài mới |
| `tàu cát linh 15.000 đồng` | ❌ bài cũ |
| `hoãn tăng giá vé tàu` | ✅ bài mới |

Không phải "thường đúng" — mà là **tùy may rủi** theo cách người dùng gõ.

---

## Giải pháp: thưởng độ mới khi xếp hạng

```
score' = cosine × (1 + alpha × recency)

recency = 0.5 ^ (số ngày tuổi / nửa chu kỳ)      thuộc (0, 1]
```

### Vì sao NHÂN chứ không CỘNG

Nếu cộng (`cosine + alpha × recency`) thì một bài hoàn toàn không liên quan
(`cosine ≈ 0`) nhưng vừa đăng hôm nay vẫn được cộng một lượng lớn, và sẽ nổi
lên đầu với **mọi** câu hỏi. Nhân giữ nguyên tính chất: không liên quan thì
vẫn bằng 0 dù mới tinh.

### Vì sao alpha phải NHỎ

Mục đích không phải luôn ưu tiên tin mới, mà chỉ **phá thế hòa**: khi hai bài
có độ liên quan xấp xỉ nhau thì bài mới thắng. Bài cũ nhưng liên quan hơn hẳn
vẫn phải thắng bài mới ít liên quan.

### Mốc thời gian tham chiếu

Dùng ngày đăng **mới nhất trong corpus**, không dùng `datetime.now()`. Lý do:
kết quả phải tái lập được. Nếu lấy thời gian thực thì cùng một câu hỏi sẽ cho
thứ hạng khác nhau tùy hôm nào chạy, và mọi con số trong báo cáo sẽ không kiểm
chứng lại được. Hệ thống chạy thật thì nên đổi sang thời gian thực.

> **Cách này đã bị THAY khi kho bắt đầu được cập nhật hằng ngày.** Mốc "mới nhất
> trong corpus" làm thứ hạng của một câu hỏi phụ thuộc vào những bài chẳng liên
> quan gì tới nó — crawl thêm 151 bài là ca Cát Linh hỏng lại. Mốc hiện hành là
> **ngày mới nhất trong các bài đang cạnh tranh** cho chính câu hỏi đó. Vẫn tái
> lập được (không dùng `datetime.now()`), nhưng không còn trôi theo kho. Chi tiết
> ở mục "Mốc độ mới trượt theo kho" bên dưới và [docs/09](09-cai-thien-mo-hinh.md).

---

## Dò tham số: tối ưu ĐỒNG THỜI hai mục tiêu

Không thể chỉ chọn tham số nào làm ca Cát Linh đúng — phải đảm bảo không phá
chất lượng truy hồi chung. Nên quét lưới và đo **cả hai**:

| half-life | alpha | Recall@1 | MRR | Ca tin mâu thuẫn |
|---|---|---|---|---|
| 30 | 0.35 | 93.5% | 0.968 | ❌ trả tin cũ |
| 14 | 0.60 | 96.8% | 0.984 | ❌ trả tin cũ |
| 3 | 0.60 | 93.5% | 0.968 | ✅ |
| **7** | **0.60** | **96.8%** | **0.984** | **✅** ← chọn |

*(đo trên tập test 31 truy vấn đã bổ sung câu không dấu và teencode)*

### Vì sao half-life 30 ngày KHÔNG đủ

Hai bài cách nhau 9 ngày, với nửa chu kỳ 30 ngày:

```
recency(bài cũ)  = 0.5^(9/30)  = 0.81   ->  hệ số 1 + 0.6×0.81 = 1.49
recency(bài mới) = 0.5^(0/30)  = 1.00   ->  hệ số 1 + 0.6×1.00 = 1.60
```

Chênh lệch hệ số chỉ 7%, trong khi khoảng cách cosine là ~22% → không đủ lật.
Với nửa chu kỳ 7 ngày, `recency(bài cũ) = 0.5^(9/7) = 0.41` → hệ số 1.25 so
với 1.60, chênh 28% → đủ.

### Kết quả ngoài mong đợi

Thêm độ mới **làm tốt lên** chất lượng truy hồi chứ không chỉ là đánh đổi:

| | Tắt độ mới | Bật độ mới |
|---|---|---|
| Recall@1 | 93.5% | **96.8%** |
| Recall@3 | 100% | 100% |
| MRR | 0.968 | **0.984** |

> Đo trên **cùng một tập test 31 truy vấn**, chỉ bật/tắt yếu tố độ mới
> (`alpha=0` so với `alpha=0.6`).
>
> Các phần trước của báo cáo có ghi Recall@1 = 90.5%. Con số đó đo trên tập
> test **cũ, chỉ 21 truy vấn viết chuẩn**, nên không so trực tiếp với bảng này
> được. Mọi số trong phần G4 đều dùng tập test 31 truy vấn đã mở rộng.

Lý do: corpus có nhiều bài cùng chủ đề đăng cách nhau vài ngày (báo chí thường
đưa tin nhiều lần về một sự việc). Trước đây bài nào trúng từ khóa hơn thì
thắng, kể cả khi đó là bài cũ hơn và ít đầy đủ hơn.

### Sau khi sửa: cả 4 cách hỏi đều xếp bài mới lên đầu

```text
0.6883  [NGÀY 10 — mới, ĐÚNG]   <- bot trả lời bài này
0.5664  [NGÀY 1  — cũ, SAI]
```

*(Notebook Phần G4 dựng lại hai bài ngay trong ô lệnh. Cùng thí nghiệm nhưng dùng
`data/eval/conflict_case.json` — bộ dùng trong `evaluate.py` và kiểm thử hồi quy —
cho 0.4956 / 0.4098 khi tắt độ mới và 0.6556 / 0.5328 khi bật, với cấu hình hiện
hành nửa chu kỳ 3 ngày, α = 0.6. Hai bộ số khác nhau vì hai bản dựng khác nhau,
kết luận như nhau.)*

Kể cả câu `"tàu cát linh 15.000 đồng"` — **trích đúng con số của tin cũ** — nay
cũng xếp bài nói con số đó đã bị hoãn lên đầu.

> **Lưu ý (kiểm lại khi viết báo cáo):** đây là kết quả ở mức **xếp hạng**. Qua
> `bot.respond()`, câu này có cosine chỉ 0.096 — dưới ngưỡng chấp nhận 0.13 —
> nên bot trả lời "không tìm thấy" thay vì trả bài. Ba cách hỏi còn lại đều được
> trả lời bằng bài mới.

---

## Luôn hiện ngày đăng

Xếp hạng theo độ mới **giảm nhẹ** vấn đề chứ không giải quyết triệt để. Bot vẫn
không hiểu bài B phủ định bài A; nó chỉ ưu tiên bài mới khi hai bài gần ngang
nhau. Nếu bài cũ liên quan **vượt trội**, nó vẫn thắng.

Nên mọi câu trả lời nay đều kèm ngày đăng:

```text
_(đăng 10/09/2026 · chuyên mục: Công nghệ · độ tương đồng: 0.41)_
```

Bot không biết bài nào đã lỗi thời, nên ít nhất phải cho người đọc đủ dữ kiện
để tự đánh giá. `browse()` cũng đã đổi sang sắp xếp theo ngày giảm dần — trước
đây nó dùng thứ tự chèn vào corpus, nên người dùng hỏi "tin mới nhất" lại nhận
bài cũ.

---

## Hai lỗi phát hiện khi kiểm thử

### Lỗi 1 — Tập test dò ngưỡng không đại diện

Tập `retrieval_tests` chỉ có câu **viết chuẩn có dấu**, trong khi bot đã được
bổ sung khả năng xử lý câu **không dấu** và **teencode**. Ngưỡng vì thế được dò
trên một phân bố truy vấn khác với thực tế.

Hậu quả cụ thể: `"tin ve dao hai nam"` đạt 0.168, trượt ngưỡng 0.18 vừa dò
được, dù trước đó vẫn trả lời tốt.

→ Bổ sung 10 truy vấn không dấu/teencode + 4 câu ngoài phạm vi không dấu, rồi
dò lại. Cũng làm mịn lưới quét quanh vùng ranh giới (bước 0.005 thay vì 0.03),
vì điểm cao nhất của truy vấn ngoài phạm vi nằm ở 0.151 — ngưỡng tối ưu 0.155
nằm lọt giữa hai mốc của lưới thô cũ.

**Bài học:** tập đánh giá phải phản ánh đúng phân bố đầu vào thực tế. Mở rộng
khả năng của hệ thống mà không mở rộng tập test thì mọi tham số dò được sau đó
đều lệch.

### Lỗi 2 — Chuẩn hóa teencode phá vỡ định tuyến index

Từ điển teencode luôn trả về từ **có dấu**. Chỉ cần **một** token được sửa là
cả câu bỗng "có dấu", và `has_diacritics()` định tuyến sang index có dấu — nơi
những token còn lại (vẫn không dấu) đều là OOV. Kết quả: cả câu bị phán đoán
dựa trên đúng một từ.

Đo được:

```text
"thoi tiet sao hoa hom nay"          (hỏi thời tiết)
  -> chuẩn hóa: thoi -> thôi
  -> "thôi tiet sao hoa hom nay"     (giờ đã "có dấu")
  -> intent tạm biệt, 0.280 > ngưỡng 0.25
  -> Bot: "Tạm biệt bạn! Hẹn gặp lại."
```

Bot chào tạm biệt khi được hỏi về thời tiết, chỉ vì `"thôi"` nằm trong pattern
`"thôi nhé"`.

→ **Sửa:** nếu câu gốc không có dấu nào thì bỏ dấu lại sau khi chuẩn hóa. Vẫn
giữ được lợi ích của chuẩn hóa (viết tắt đã bung ra: `k` → `không` → `khong`,
khớp được với index âm tiết), chỉ là giữ nguyên "hệ quy chiếu dấu" mà người
dùng đang gõ.

**Bài học:** khi một pipeline có nhiều nhánh xử lý, bước tiền xử lý không được
phép âm thầm làm thay đổi tín hiệu dùng để chọn nhánh.

---

## Cache index ra đĩa

Không liên quan tới độ mới, nhưng cùng đợt sửa.

Dựng index tốn ~21 giây cho 381 bài, gần như toàn bộ là thời gian `word_tokenize`.
`lru_cache` của `segment_vi` chỉ sống trong một tiến trình, nên khởi động lại
là mất trắng.

Giải pháp: `joblib` + **vân tay SHA-256** của (nội dung corpus + các tham số
ảnh hưởng tới index: `ngram_range`, `min_df`, `max_df`, trọng số trường,
`CONFIG_RETRIEVAL`).

| | Thời gian |
|---|---|
| Lần đầu (dựng + lưu) | 24.2s |
| Lần sau (nạp cache) | **1.1s** |
| Corpus đổi | tự dựng lại |

Lưu ý: `FRESHNESS_ALPHA` **không** nằm trong vân tay, vì hệ số độ mới chỉ áp
dụng lúc tìm kiếm, không nằm trong index. Đổi nó không cần dựng lại.

---

## Tích lũy hay ghi đè?

| Lớp | Hành vi |
|---|---|
| **Dữ liệu** (`corpus_raw.csv`) | **Tích lũy.** crawler nạp corpus cũ, thêm bài mới, dedup theo URL. Không bao giờ xóa bài cũ. |
| **Mô hình** (index TF-IDF) | **Ghi đè.** `fit()` tính lại toàn bộ vocabulary và IDF. Không có cập nhật tăng dần. |

Hệ quả đáng lưu ý: mỗi lần thêm dữ liệu, IDF của **mọi** term đều đổi, nên điểm
số của các bài **đã có sẵn** cũng thay đổi theo. Đây là lý do ngưỡng chấp nhận
cần được dò lại khi corpus lớn lên đáng kể.

## Thời gian huấn luyện (381 bài, đo thực tế)

| Thành phần | Thời gian |
|---|---|
| Intent classifier (150 pattern) | 0.64s |
| Học từ điển teencode (8.372 cặp) | 0.47s |
| **Dựng index truy hồi** | **20.5s** ← chiếm gần hết |
| Tổng khởi động (lần đầu) | 21.3s |
| Tổng khởi động (có cache) | **1.1s** |
| Trả lời một câu hỏi | 24ms |

Xấp xỉ tuyến tính theo số bài (~54ms/bài): 1.000 bài ≈ 1 phút,
10.000 bài ≈ 9 phút.

## Mốc độ mới trượt theo kho (phát hiện 14/09/2026, đã sửa)

Lần crawl thật đầu tiên thêm 151 bài vào kho (381 → 532). Ngay sau đó, câu
`"giá vé tàu cát linh bao nhiêu"` **quay lại trả bài cũ đã sai** — đúng cái lỗi
mà cả mục này sinh ra để chống, và kiểm thử hồi quy `tin_moi_phu_dinh_tin_cu` bắt
được.

Không bài mới nào nói về tàu Cát Linh. Vấn đề nằm ở **mốc tham chiếu**: nó là
ngày đăng mới nhất của cả kho, nên crawl xong mốc nhảy từ 10/09 lên 14/09 và
**cả hai** bài Cát Linh cùng già đi:

| | bài cũ (01/09, đã sai) | bài mới (10/09, đúng) |
|---|---|---|
| cosine | 0.4956 | 0.4098 |
| recency khi mốc = 10/09 | 0.125 → hệ số 1.08 | 1.000 → hệ số 1.60 |
| recency khi mốc = 14/09 | 0.050 → hệ số 1.03 | 0.397 → hệ số 1.24 |
| điểm cuối (mốc 14/09) | **0.5104** ← bot trả bài này | 0.5086 |

Đây là lỗi **thiết kế**, không phải lỗi tham số: thứ hạng của một câu hỏi không
được phép phụ thuộc vào những bài không liên quan tới nó. `evaluate.py` quét cả
lưới (7 nửa chu kỳ × 11 alpha) và xác nhận — với mốc "corpus", **không cấu hình
nào** vừa xử lý đúng ca Cát Linh vừa còn đúng sau khi thêm một bài không liên
quan ở ngày tương lai.

**Cách sửa.** Mốc = ngày mới nhất trong **các bài đang cạnh tranh** cho chính câu
hỏi đó, tức những bài mà nếu được thưởng độ mới tối đa thì vẫn có thể lên hạng 1:

```
điểm(bài) × (1 + alpha)  >=  điểm cao nhất
```

Bài yếu hơn mức đó thì dù mới tới đâu cũng không lật được hạng 1, nên không có lý
do để ngày đăng của nó định nghĩa "thế nào là mới". Hệ quả: **thêm bài không liên
quan không làm đổi thứ hạng** — bất biến theo thiết kế, không phải nhờ may mắn
của tham số. Đây cũng thành ràng buộc cứng thứ hai khi dò (`conflict_case.json`
có sẵn một bài "tương lai không liên quan" để kiểm tra đúng điều này).

Dò lại trên dev với corpus 532 bài: **nửa chu kỳ 3 ngày, alpha 0.6 → 0.3**
(35/77 cấu hình qua được cả hai ràng buộc; chọn cấu hình MRR cao nhất, hòa thì
lấy alpha nhỏ hơn). Alpha 0.6 nay là quá mạnh để gọi là "phá thế hòa": nó cho
phép một bài chỉ liên quan bằng 62,5% bài đúng chiếm hạng 1 — chấp nhận được khi
kho đứng yên, nhưng không khi mỗi ngày có thêm hàng chục bài mới.

## Hạn chế còn lại

1. **Không phát hiện mâu thuẫn.** Bot không hiểu bài B phủ định bài A. Nó chỉ
   ưu tiên bài mới khi hai bài gần ngang nhau về độ liên quan.
2. **Bài cũ liên quan vượt trội vẫn thắng.** Đúng như thiết kế (nếu không, bot
   sẽ chỉ trả tin mới nhất bất kể hỏi gì), nhưng vẫn là một lỗ hổng.
3. **Không có khái niệm "sự kiện".** Hai bài về hai sự việc khác nhau nhưng
   cùng từ khóa vẫn cạnh tranh nhau như thể cùng chủ đề.

Muốn giải quyết triệt để cần **phát hiện cùng-một-sự-việc**: nhóm các bài có độ
tương đồng cao với nhau thành một cụm, rồi chỉ hiển thị bài mới nhất trong cụm
kèm cảnh báo "có bài mới hơn về chủ đề này". Ghi nhận ở phần hướng phát triển.
