# 09 — Cải thiện chatbot: có những "đòn bẩy" nào, và kéo cái nào?

> Tài liệu này viết cho người đọc quen **kỹ thuật phần mềm** nhưng không chuyên về
> học máy. Câu hỏi xuất phát: *"muốn bot tốt hơn thì chỉnh cái gì — tham số à?"*
>
> Trả lời ngắn: tham số là đòn bẩy **rẻ nhất và yếu nhất**. Phần lớn lỗi thật
> không nằm ở giá trị tham số, và chỉnh tham số chỉ **di chuyển dọc theo một đánh
> đổi** chứ không xóa được đánh đổi đó. Tài liệu ghi lại hai lần sửa có thật —
> lỗi độ mới sau khi crawl hằng ngày, và lỗi không nhận ra câu gõ sai chính tả —
> để minh họa cách chọn đòn bẩy.

Mọi số liệu ở đây tái lập được bằng:

```powershell
python src/evaluate.py               # -> data/eval/test_report_v4.txt
python tools/compare_improvements.py # -> data/eval/improvements_report.txt
```

---

## 1. Năm đòn bẩy

| # | Đòn bẩy | Tương đương trong phần mềm | Ở dự án này là gì | Chi phí / rủi ro |
|---|---|---|---|---|
| 1 | **Tham số** | hằng số cấu hình (timeout, số lần retry) | ngưỡng truy hồi 0.13, ngưỡng intent 0.25, α độ mới, nửa chu kỳ, trọng số tiêu đề ×3 | đổi một dòng, nhưng **mọi giá trị đều là một đánh đổi** |
| 2 | **Thuật toán** | sửa logic, không phải sửa config | cách so khớp chữ (từ hay n-gram ký tự), cách tính độ mới | phải viết code, thêm kiểm thử, rồi **dò lại tham số** |
| 3 | **Dữ liệu tham chiếu** | bảng tra, seed data | 164 mẫu câu intent, 342 cặp teencode, 532 bài báo | dễ làm, nhưng thường chỉ sửa được từng ca một |
| 4 | **Dữ liệu đánh giá** | bộ test | dev/test, `conflict_case.json`, `dev_typo`/`test_typo` | **không có thì không biết mình có tiến bộ hay không** |
| 5 | **Đổi mô hình** | thay thư viện / đổi CSDL | TF-IDF → embedding (PhoBERT), hoặc ghép LLM | thay đổi lớn, và vượt ra ngoài phạm vi "from scratch" của đồ án |

**Một điều hay làm người mới bất ngờ: dự án này gần như không có bước "train"
theo nghĩa học sâu.** "Mô hình" chủ yếu là **thống kê đếm từ** trên corpus. Vì
vậy "huấn luyện lại" chỉ là đếm lại (docs/08, mục 1) — chạy vài chục giây, không
GPU, không hạt giống ngẫu nhiên. Thứ duy nhất thật sự "chỉnh được bằng tay" là
nhóm tham số ở đòn bẩy 1, và chúng được dò tự động trên tập DEV.

---

## 2. Vì sao "chỉnh tham số" thường KHÔNG phải là cách sửa

Ca người dùng báo: gõ `thám hiểm Sơn Dòng` (thiếu một chữ `o`, quên gạch của `Đ`)
thì bot từ chối trả lời; gõ đúng `Sơn Đoòng` thì trả lời ngay.

Nguyên nhân đo được: cosine của bài đúng chỉ **0.086**, dưới ngưỡng **0.13**.
Cách sửa "rẻ" nhất là hạ ngưỡng. Đây là bảng dò ngưỡng trên DEV
(`test_report_v4.txt`, PHA 1.4):

| Ngưỡng | Trả lời được câu trong phạm vi | Chặn đúng câu ngoài phạm vi |
|---|---|---|
| 0.08 | 99.0% | **58.3%** |
| 0.10 | 97.1% | 72.2% |
| 0.12 | 94.2% | 94.4% |
| **0.13 (hiện hành)** | 90.4% | **100%** |
| 0.14 | 86.5% | 100% |

Hạ xuống 0.08 để cứu một câu gõ sai thì bot bắt đầu **trả lời bừa gần một nửa số
câu ngoài phạm vi** — hỏi "công thức nấu phở" cũng được đưa cho một bài báo ngẫu
nhiên kèm giọng chắc chắn. Đó chính là lỗi tệ nhất mà cả kiến trúc này sinh ra
để tránh (docs/02, quyết định 1).

> **Quy tắc rút ra:** tham số chỉ cho bạn **trượt dọc theo đường đánh đổi**. Muốn
> *dịch chuyển cả đường* — vừa trả lời được nhiều hơn vừa không đoán bừa nhiều
> hơn — phải đổi **thuật toán** hoặc **dữ liệu**.

---

## 3. Quy trình sửa, nói theo ngôn ngữ kỹ thuật phần mềm

1. **Viết ca kiểm thử đang trượt trước.** Giống test đỏ. Với lỗi gõ sai, không
   có sẵn tập nào đo được nó → sinh `dev_typo` / `test_typo`
   (`tools/build_typo_sets.py`, mỗi truy vấn một lỗi gõ trên một âm tiết nội dung).
2. **Sửa đúng đòn bẩy** (thuật toán hoặc dữ liệu tham chiếu).
3. **Dò lại tham số trên DEV.** Thuật toán mới làm điểm số dịch chuyển, nên
   ngưỡng cũ gần như chắc chắn không còn đúng. `src/evaluate.py` làm việc này
   bằng quét lưới.
4. **Kiểm tra CẢ HAI phía của đánh đổi.** Câu gõ sai có tốt lên **và** câu ngoài
   phạm vi có còn bị chặn không? Một thay đổi làm tốt vế này, hỏng vế kia thì
   không phải cải tiến.
5. **Chạy kiểm thử hồi quy** (`tests/`) — bước CI.
6. **Đo trên TEST đúng một lần, ở cuối.** Nếu chỉnh tiếp sau khi nhìn kết quả
   test thì test đã biến thành dev, và con số báo cáo thành lạc quan giả
   (docs/06 ghi lại đúng sai lầm này ở giai đoạn trước).

---

## 4. Ca 1 — Độ mới: mốc tham chiếu trôi theo kho (đòn bẩy 2)

### Triệu chứng

Sau lần crawl thật đầu tiên (14/09/2026, +151 bài), câu
`"giá vé tàu cát linh bao nhiêu"` **quay lại trả bài cũ đã sai**. Kiểm thử hồi
quy `tin_moi_phu_dinh_tin_cu` bắt được ngay.

Không có bài mới nào nói về tàu Cát Linh. Vấn đề: điểm độ mới tính theo **ngày
mới nhất của cả kho**, nên crawl xong mốc nhảy từ 10/09 lên 14/09, **cả hai** bài
Cát Linh cùng già đi và khoảng cách hệ số thưởng co lại (bảng chi tiết ở docs/05).

Nói theo kiểu kỹ thuật phần mềm: **kết quả của một truy vấn phụ thuộc vào dữ liệu
hoàn toàn không liên quan tới nó** — một dạng phụ thuộc ẩn (hidden coupling).

### Vì sao không sửa bằng tham số

`evaluate.py` quét cả lưới 7 nửa chu kỳ × 11 giá trị α cho mốc "corpus", với hai
ràng buộc cứng: (a) ca Cát Linh phải đúng, (b) vẫn đúng sau khi thêm một bài
**không liên quan, ngày đăng ở tương lai** (mô phỏng các lần crawl sau).

```
mốc corpus    : 0 / 77 cấu hình qua cả hai ràng buộc
mốc candidates: 35 / 77 cấu hình qua cả hai ràng buộc
```

**Không tồn tại** giá trị tham số nào sửa được. Đó là dấu hiệu rõ ràng rằng lỗi
nằm ở tầng thuật toán.

### Cách sửa (đòn bẩy 2)

Mốc tham chiếu = ngày mới nhất trong **các bài đang cạnh tranh** cho chính câu
hỏi đó, tức các bài thỏa `điểm × (1 + α) >= điểm cao nhất`. Bài yếu hơn mức đó
thì dù được thưởng tối đa cũng không lật được hạng 1, nên ngày đăng của nó không
có lý do gì để định nghĩa "mới".

Tính chất thu được — và đây mới là điều quan trọng: **thêm bài không liên quan
không làm đổi thứ hạng**. Đó là bất biến của thiết kế, không phải may mắn của một
bộ tham số. Có hẳn một kiểm thử cho nó
(`crawl_them_bai_khong_lien_quan_khong_doi_thu_hang`).

### Dò lại tham số sau khi đổi thuật toán (bước 3)

α giảm **0.6 → 0.3** (nửa chu kỳ giữ 3 ngày). Với mốc mới, α = 0.6 quá mạnh: nó
cho phép một bài chỉ liên quan bằng 62,5% bài đúng chiếm hạng 1. Khi kho đứng yên
thì không sao; khi mỗi ngày thêm vài chục bài thì đó là nguồn sai thường trực.

---

## 5. Ca 2 — Gõ sai chính tả: thêm một chỉ mục, không phải hạ ngưỡng (đòn bẩy 2 + 4)

### Vì sao TF-IDF mức từ bó tay

TF-IDF so khớp **chính xác theo term**. `dòng` và `đoòng` là hai term khác hẳn
nhau — tệ hơn, `dòng` lại là một từ có thật (xuất hiện trong 100/532 bài), nên không
thể phát hiện bằng kiểu "từ này không có trong từ điển". Sửa lỗi chính tả bằng
khoảng cách sửa (edit distance) cũng không cứu được ca này vì nó là **lỗi thành
từ có thật** (real-word error).

### Thiết kế

Thêm **chỉ mục thứ ba**: n-gram **ký tự** (n = 3) của **tiêu đề đã bỏ dấu**.

```
" son dong "  ->  " so", "son", "on ", " do", "don", "ong", "ng "
" son doong " ->  " so", "son", "on ", " do", "doo", "oon", "ong", "ng "
```

Hai cách viết chung phần lớn n-gram, dù không chung một term nào.

Hai lựa chọn thiết kế quan trọng:

- **Dùng làm ĐƯỜNG DỰ PHÒNG**, chỉ chạy khi đường chính không có bài nào đạt
  ngưỡng. Câu mà đường chính đã trả lời thì kết quả **giữ nguyên** — rủi ro chỉ
  giới hạn ở những câu trước đây bị từ chối. (Trong phần mềm: thêm một nhánh
  fallback, không viết lại đường chính.)
- **Điểm chấp nhận = cosine mức từ + cosine n-gram ký tự.** Các từ **gõ đúng**
  còn lại trong câu là bằng chứng độc lập, cộng vào sẽ phân biệt được "gõ sai một
  chữ" với "chỉ tình cờ na ná cách viết".

### Chọn thiết kế bằng số liệu, không bằng cảm tính

`tools/compare_improvements.py` so 12 thiết kế trên DEV (112 câu sạch + 112 câu
gõ sai + 36 câu ngoài phạm vi), mỗi thiết kế được dò ngưỡng tốt nhất của riêng nó:

| Trường đưa vào chỉ mục | n | Luật chấp nhận | Ngưỡng | Cứu được | Trả sai | Lọt |
|---|---|---|---|---|---|---|
| **chỉ tiêu đề** | **3** | **cộng cosine mức từ** | **0.53** | **36** | **0** | **0** |
| chỉ tiêu đề | 3 | chỉ n-gram ký tự | 0.47 | 34 | 1 | 0 |
| tiêu đề + mô tả | 4 | cộng cosine mức từ | 0.40 | 36 | 4 | 0 |
| cả bài | 4 | chỉ n-gram ký tự | 0.26 | 20 | 5 | 0 |

Thêm thân bài làm **giảm hẳn** chất lượng: vector n-gram ký tự của một bài dài
chứa gần như mọi tổ hợp ba chữ cái, nên bài nào cũng "na ná" câu hỏi nào. Tiêu đề
thì cô đọng đúng tên riêng mà người dùng đang gõ.

### Kết quả trên TEST (đo một lần, ngưỡng 0.53 chốt trên dev)

| Tập | Dự phòng TẮT | Dự phòng BẬT |
|---|---|---|
| test sạch — trả lời đúng | 77.0% (94/122) | **86.1% (105/122)** |
| test sạch — trả lời sai | 3 | 5 |
| test_typo (gõ sai) — trả lời đúng | 60.7% (74/122) | **73.8% (90/122)** |
| test_typo — trả lời sai | 7 | 10 |
| ngoài phạm vi — chặn đúng | 87.5% (21/24) | 87.5% (21/24) |

**Cái giá phải nói rõ:** số câu trả lời SAI tăng (3 → 5 và 7 → 10). Đường dự
phòng biến một phần "từ chối" thành "trả lời", và không phải lần nào cũng đúng
bài. Đổi lại, nó không làm lọt thêm câu ngoài phạm vi nào, và bot **tự nói ra**
rằng đây là khớp gần đúng:

```
_(Không tìm thấy từ khóa khớp chính xác — có thể bạn gõ nhầm.
  Bài có tiêu đề gần nhất với câu hỏi:)_
```

Một điểm bất ngờ: dự phòng còn cứu cả **câu gõ đúng** nhưng diễn đạt khác (94 →
105 câu). Những câu này trước đây trượt ngưỡng vì dùng từ khác với bài báo, nhưng
tiêu đề vẫn gần về mặt chữ.

---

## 6. Bài học về ĐÒN BẨY 4: tập đánh giá phải giống đầu vào thật

Lỗi này suýt lọt, và cách nó lộ ra đáng ghi lại.

Sau khi bật đường dự phòng, kiểm thử hồi quy báo một ca thất bại: câu **ngoài
phạm vi** `"thoi tiet sao hoa hom nay"` được trả lời bằng bài
*"ASIAD 20 khởi tranh hôm nay"*. Trong khi đó, việc dò trên DEV báo **0 câu lọt**.

Vì sao dò trên dev không thấy: tập câu ngoài phạm vi của dev khi đó **toàn câu có
dấu**. Bản không dấu của cùng câu đó đạt 0.513, bản có dấu chỉ 0.481 — bỏ dấu làm
hai câu khác nhau trông giống nhau hơn, mà chỉ mục n-gram ký tự thì luôn chạy
trên bản đã bỏ dấu. Tập dò không đại diện cho đầu vào thật, nên nó "đo" một thế
giới dễ hơn thực tế.

Đã sửa bằng cách **bổ sung dữ liệu đánh giá**, không phải bằng cách vặn ngưỡng
cho qua ca đó:

- Sinh thêm biến thể **không dấu** cho các câu ngoài phạm vi của dev (28 → 36 câu).
- Chỉ sinh từ câu của **DEV**, không bao giờ từ TEST — nếu không, tập test sẽ có
  "anh em sinh đôi" nằm trong tập dò và số liệu test thành lạc quan giả.
- Đưa thẳng các ca ngoài phạm vi dùng trong kiểm thử hồi quy vào dev: ca đã biết
  là khó thì chỗ của nó là tập dò.

Dò lại trên dev mới: ngưỡng 0.51 → **0.53**, và ca kia bị chặn đúng.

> Một lần thử sai đáng ghi: ban đầu tôi sửa bằng cách lọc stopword ở mức âm tiết
> đã bỏ dấu. Hỏng nặng — bỏ dấu làm stopword đụng độ với từ nội dung (`năm` →
> `nam` nuốt luôn `Hải Nam`, `thám` → `tham`). `"tin ve dao hai nam"` chỉ còn
> `"hai"`. Đã bỏ cách này; dấu vết giữ lại ở đây để người sau không thử lại.

---

## 7. Tổng kết trước / sau (cùng kho 532 bài, qua `bot.respond()`)

Nguồn: `data/eval/improvements_report.txt`, phần B.

| Chỉ số | TRƯỚC (cấu hình lúc nộp bài) | SAU |
|---|---|---|
| Recall@1 (thành phần) | 109/122 | **112/122** |
| MRR | 0.927 | **0.940** |
| Ca Cát Linh trên kho hiện tại | ❌ SAI | ✅ đúng |
| Ca Cát Linh sau khi thêm bài tương lai | ❌ SAI | ✅ đúng |
| Câu sạch: đúng / sai | 90 / 5 | **101** / **4** |
| Câu gõ sai: đúng / sai | 74 / 7 | **86** / 8 |
| Câu ngoài phạm vi bị từ chối | 18/24 | 18/24 |

Đầu-cuối trên test với bộ tham số mới: trả lời đúng **82.8%** (101/122), câu gõ
sai **70.5%** (86/122), từ chối đúng 75.0% (18/24).

Tham số thay đổi — tất cả đều là **hệ quả** của việc đổi thuật toán và dữ liệu
đánh giá, không phải là bản thân cách sửa:

| Tham số | Trước | Sau | Vì sao |
|---|---|---|---|
| `FRESHNESS_REFERENCE` | *(chưa có)* | `candidates` | ca 1 |
| `FRESHNESS_ALPHA` | 0.6 | 0.3 | dò lại sau khi đổi mốc |
| `FUZZY_THRESHOLD` | *(chưa có)* | 0.53 | ca 2, dò trên dev đã bổ sung |
| `BM25_B` | 0.9 | 0.75 | dò lại trên kho 532 bài (BM25 vẫn **không** được chọn) |
| `RETRIEVAL_THRESHOLD` | 0.13 | 0.13 | không đổi |
| `INTENT_THRESHOLD` | 0.25 | 0.25 | không đổi |

---

## 8. Những đòn bẩy CHƯA kéo (việc còn lại)

| Vấn đề còn lại | Đòn bẩy đúng | Ghi chú |
|---|---|---|
| Intent chỉ đạt 61.5% — điểm yếu lớn nhất | **3 (dữ liệu tham chiếu)**: thêm mẫu câu cho `dong_y`, `cam_on`, `che_bai`, `huong_dan`, rồi dò lại ngưỡng | Hạ ngưỡng intent **không** phải cách sửa: nó bắt đầu định tuyến nhầm câu hỏi tin tức thành câu soạn sẵn |
| Không hiểu từ đồng nghĩa ("xe hơi" vs "ô tô") | **5 (đổi mô hình)**: embedding | Vượt phạm vi "from scratch" của đồ án |
| 3 câu ngoài phạm vi vẫn lọt qua truy hồi | 1 + 4 | Đều là câu có từ khóa trùng ngẫu nhiên ("miền bắc", "mèo") |
| Nhãn vàng của tập đánh giá cũ dần | **4 (dữ liệu đánh giá)** | Kho lớn dần, có bài **mới hơn** cũng trả lời đúng câu hỏi nhưng không nằm trong nhãn — làm điểm số bị chấm thấp oan. Cần rà lại nhãn định kỳ |
| Không phát hiện mâu thuẫn giữa các bài | 2 | Cần gom cụm "cùng một sự việc" (docs/05) |

---

## 9. Tóm tắt một câu

Khi bot sai, đừng hỏi ngay *"chỉnh tham số nào?"* — hãy hỏi **"lỗi này nằm ở tầng
nào?"**: dữ liệu, cách biến câu chữ thành con số, công thức xếp hạng, hay ngưỡng
quyết định. Tham số chỉ là tầng cuối, và nó luôn được **dò lại** sau khi các tầng
trên thay đổi, chứ hiếm khi là nơi bắt đầu.
