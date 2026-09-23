# 10 — Có nên đưa Word Embedding (Lab 05) vào chatbot không?

> **Câu hỏi:** Lab 05 dạy Word2Vec — biểu diễn từ bằng vector dày, học từ ngữ
> cảnh. Nó được giới thiệu chính là để sửa điểm yếu của TF-IDF mà đồ án này đã
> tự ghi nhận (`"ô_tô" != "xe_hơi"`). Vậy có nên thay, hoặc ghép, Word2Vec vào
> phần truy hồi của chatbot?
>
> **Cách trả lời:** không tranh luận bằng lý thuyết mà **làm thí nghiệm**: cài
> Word2Vec từ đầu, kiểm chứng nó đúng, rồi đo nó trên chính dữ liệu của đồ án —
> cả trên loại câu hỏi mà nó sinh ra để giải. Cùng quy trình đã dùng để thử BM25
> ở docs/06: dò trên DEV, báo cáo TEST một lần, kiểm định thống kê.

Tái lập toàn bộ số liệu:

```powershell
python tests/test_word2vec.py           # kiểm chứng cài đặt (12 kiểm thử)
python tools/build_paraphrase_set.py    # sinh tập câu hỏi diễn đạt lại
python tools/exp_word2vec.py            # thí nghiệm -> data/eval/word2vec_report.txt
```

---

## 1. Lab 05 dạy gì, và khác gì kỹ thuật đồ án đang dùng

| | Lab 04 — đồ án đang dùng | Lab 05 — Word2Vec |
|---|---|---|
| Biểu diễn | vector thưa (sparse), số chiều = kích thước từ vựng (~140.000) | vector dày (dense), `vector_size` chiều (50–100) |
| Hai văn bản giống nhau khi | **dùng chung từ** | dùng những từ **hay xuất hiện trong ngữ cảnh giống nhau** |
| Vector tài liệu | TF-IDF, chuẩn hóa L2 | **trung bình** vector các từ (Lab 05, Phần 4) |
| Đo độ giống | cosine | cosine (giống nhau) |
| Từ chưa gặp | bị bỏ qua, các từ còn lại vẫn tính điểm | **không có vector** |
| Giải thích được | có — biết term nào khớp, đóng góp bao nhiêu | khó — từng chiều không có nghĩa riêng |

Cùng một **nhiệm vụ** (tìm tài liệu giống câu hỏi bằng cosine), cùng tiền xử lý
(tách từ Lab 03), khác nhau ở **cách biến chữ thành số**.

Lab 05 cũng tự cảnh báo một điều quan trọng (Phần E — *Similar ≠ Synonym*):
Word2Vec học **độ giống theo phân bố** (hay đi cùng nhau), không học **đồng
nghĩa**. "vàng" gần "bitcoin" không có nghĩa hai từ cùng nghĩa, chỉ là cùng hay
xuất hiện trong bài tài chính.

---

## 2. Cài đặt: Word2Vec tự viết, không dùng gensim

Lab 05 dùng `gensim.models.Word2Vec`. Đồ án này có nguyên tắc **lõi tự cài đặt**
(TF-IDF, BM25, Naive Bayes đều viết tay, thư viện chỉ dùng để đối chiếu trong
kiểm thử). Để giữ nguyên tắc đó — và để không thêm phụ thuộc mới — Word2Vec cũng
được viết lại bằng NumPy: `src/word2vec.py`, khoảng 200 dòng.

Thuật toán đúng như bản gốc (Mikolov và cộng sự, 2013) và như `gensim` chạy ở
Lab 05: **skip-gram + negative sampling**, cửa sổ động, subsampling từ phổ biến,
phân phối nhiễu unigram^0.75, tốc độ học giảm tuyến tính. Hàm mất mát cho một
cặp (từ tâm c, từ ngữ cảnh o) và k từ nhiễu:

```
L = -log σ(u_o · v_c)  -  Σ_i log σ(-u_{n_i} · v_c)
```

### Làm sao biết cài đặt đúng?

Không có gensim để so, nên dùng hai cách kiểm chứng chuẩn cho mô hình học bằng
gradient (`tests/test_word2vec.py`, **12/12 đạt**):

| Kiểm thử | Kết quả |
|---|---|
| Gradient giải tích vs **đạo hàm số** (sai phân trung tâm), cả 3 ma trận | sai số lớn nhất ~3×10⁻⁸ |
| Kho đồ chơi hai "thế giới" tách biệt (con vật / thiết bị), như corpus mẫu Lab 05 | cosine cùng nhóm **0,965**, khác nhóm **0,135** |
| `most_similar("mèo")` | `gà, vịt, chó` — đúng nhóm |
| Loss giảm qua các epoch | 3,99 → 2,04 |
| Phép cộng dồn gradient nhanh khớp `np.add.at` khi chỉ số trùng | sai số 3,6×10⁻¹⁵ |
| Cùng seed → cùng vector; câu toàn từ lạ → vector 0; `min_count` quá cao → báo lỗi | đạt |

Nghĩa là: **nếu Word2Vec thua trong thí nghiệm dưới đây, đó là giới hạn của kỹ
thuật trên dữ liệu này, không phải lỗi cài đặt.**

---

## 3. Thiết kế thí nghiệm

### Vấn đề: tập đánh giá hiện có không công bằng với embedding

Tập dev/test của đồ án gần như toàn câu hỏi kiểu **từ khóa**: `"giá iphone 18
pro"`, `"scb trương mỹ lan"`. Đó là sân nhà của TF-IDF. Đo Word2Vec chỉ trên đó
thì kết luận "TF-IDF thắng" là điều biết trước, chẳng chứng minh được gì.

Vì vậy làm thêm **tập câu hỏi diễn đạt lại** (`tools/build_paraphrase_set.py`,
47 câu): mỗi câu mô tả lại một bài báo bằng lời khác, **cố tránh các từ nội dung
trong tiêu đề**, dùng từ đồng nghĩa và cách nói vùng miền:

| Tiêu đề bài báo | Câu hỏi diễn đạt lại |
|---|---|
| Vì sao **nem rán** thường **cháy đen** hai đầu? | **chả giò chiên** bị **khét** hai đầu |
| 5 điều cần biết khi **máy bay hạ cánh** khẩn | **phi cơ** phải **đáp** khẩn cấp cần biết gì |
| Tên lửa **tái sử dụng** Trung Quốc bay thành công | Trung Quốc phóng thành công **hỏa tiễn dùng lại được** |
| Gần 20% **học sinh** Việt bị **bắt nạt** | một phần năm **học trò** cấp hai bị **trêu chọc** thường xuyên |

Đây là đúng loại câu Word2Vec sinh ra để giải. Chia 19 câu dev / 28 câu test
theo hạt giống cố định. Câu được viết **trước** khi huấn luyện Word2Vec và không
sửa lại sau khi thấy kết quả.

### Quy trình

- **Độ mới tắt** ở mọi phương pháp, để so chất lượng xếp hạng thuần (giống
  PHA 1.2 khi so TF-IDF với BM25).
- Ba phương pháp: **TF-IDF** (đồ án), **Word2Vec** (vector trung bình, đúng
  Lab 05), và **Lai** — `điểm = cosine TF-IDF + λ · cosine Word2Vec`.
- Mọi lựa chọn chốt trên **DEV**: `vector_size` ∈ {50, 100}, `min_count` ∈
  {1, 3, 5} (window = 5, skip-gram, 30 epoch như `BASE_CONFIG` của Lab 05),
  trung bình thường hay có trọng số IDF, và λ.
- **TEST chạy một lần** ở cuối với cấu hình đã chốt.
- **Kiểm định dấu có cặp**: chỉ coi là "hơn" khi p < 0,05 — cùng tiêu chuẩn đã
  dùng để bác BM25.
- Thêm một câu hỏi thực tế: cosine Word2Vec có dùng được làm **ngưỡng chấp
  nhận** (tách câu trong / ngoài phạm vi) như cosine TF-IDF đang làm không?

---

## 4. Kết quả

Nguồn: `data/eval/word2vec_report.txt`. Kho 532 bài = **163.769 token** sau tách
từ và bỏ stopword, **18.724** từ khác nhau.

### 4.1. Chọn cấu hình trên DEV

12 phương án (2 `vector_size` × 3 `min_count` × 2 cách lấy trung bình). Không
phương án nào tiến lại gần TF-IDF:

| Phương pháp | dev — R@1 | dev — MRR | dev_paraphrase — R@1 | dev_paraphrase — MRR |
|---|---|---|---|---|
| **TF-IDF (đồ án)** | **96,5%** (110/114) | **0,970** | **63,2%** (12/19) | **0,721** |
| Word2Vec tốt nhất theo dev: d=100, min_count=5, TB thường | 72,8% (83/114) | 0,786 | 57,9% (11/19) | 0,698 |
| Word2Vec d=100, min_count=3, TB trọng số IDF | 76,3% (87/114) | 0,810 | 47,4% (9/19) | 0,613 |
| Word2Vec d=50, min_count=1, TB thường (sát `BASE_CONFIG` Lab 05) | 68,4% (78/114) | 0,757 | 42,1% (8/19) | 0,603 |

Ngay trên dev, Word2Vec thua TF-IDF ở **cả hai** tập — kể cả tập diễn đạt lại,
nơi lẽ ra nó phải thắng.

### 4.2. Báo cáo trên TEST (chạy một lần)

Cấu hình đã chốt: d = 100, min_count = 5, trung bình thường, λ = 0,5.

| Tập | Phương pháp | R@1 | R@3 | MRR | Kiểm định dấu so với TF-IDF |
|---|---|---|---|---|---|
| test (122 câu) | **TF-IDF** | **91,0%** | 117 | **0,937** | — |
| | Word2Vec | 68,9% | 95 | 0,742 | tốt hơn 4 câu, **kém hơn 37 câu**, p < 0,001 |
| | Lai | 90,2% | 114 | 0,923 | tốt hơn 4, kém hơn 8, p = 0,39 |
| test_paraphrase (28 câu) | **TF-IDF** | **64,3%** | 21 | **0,729** | — |
| | Word2Vec | 39,3% | 20 | 0,563 | tốt hơn 5, kém hơn 9, p = 0,42 |
| | Lai | 53,6% | 21 | 0,658 | tốt hơn 6, kém hơn 6, p = 1,00 |

Đọc bảng này:

- **Word2Vec đứng một mình kém TF-IDF có ý nghĩa thống kê** trên câu hỏi thật
  (37 câu tệ hơn, 4 câu tốt hơn).
- **Trên câu diễn đạt lại — sân nhà của embedding — nó cũng không thắng**: R@1
  thấp hơn 25 điểm; khác biệt không có ý nghĩa thống kê vì tập nhỏ, nghĩa là
  *không có bằng chứng Word2Vec giúp được*.
- **Bản Lai không hơn**: trên dev, λ = 0,5 nâng MRR diễn đạt lại 0,721 → 0,752,
  nhưng lên test thì **ngược lại** (0,729 → 0,658). Mức tăng trên dev là do khớp
  may với 19 câu — đúng hiện tượng mà quy tắc "chỉ đổi khi p < 0,05" sinh ra để
  chặn. Theo đúng quy tắc đã dùng với BM25: **giữ TF-IDF**.

### 4.3. Theo kiểu câu hỏi (test, R@1)

| Kiểu câu | TF-IDF | Word2Vec | Lai |
|---|---|---|---|
| có dấu (86 câu) | 80 | 69 | 80 |
| **không dấu** (22 câu) | **20** | **7** | 19 |
| teencode (14 câu) | 11 | 8 | 11 |

Câu **không dấu** là chỗ Word2Vec sụp hẳn: từ vựng của nó là token có dấu đã
tách từ, nên câu gõ không dấu chỉ có **36%** token có vector (dev), một câu không
có token nào. TF-IDF có riêng một index âm tiết không dấu cho trường hợp này
(docs/02, quyết định 4). Muốn Word2Vec làm được điều tương tự phải huấn luyện
thêm một mô hình thứ hai trên bản bỏ dấu.

### 4.4. Không dùng được làm ngưỡng chấp nhận

Kiến trúc đồ án xoay quanh một nguyên tắc: **không đủ căn cứ thì nói không
biết** (docs/02, quyết định 1). Điều đó cần một điểm số **tách được** câu trong
phạm vi với câu ngoài phạm vi:

| | Điểm top-1, câu trong phạm vi (trung vị) | Điểm top-1, câu ngoài phạm vi | Ngưỡng tốt nhất → TB(trả lời được, chặn đúng) |
|---|---|---|---|
| TF-IDF | 0,205 | trung vị 0,067, max 0,190 | **91,1%** |
| Word2Vec | 0,826 | trung vị 0,676, **max 0,842** | 79,2% |

Với Word2Vec, câu **ngoài phạm vi** cao nhất (0,842) còn **cao hơn trung vị câu
trong phạm vi** (0,826). Vector trung bình của mọi văn bản đều na ná nhau —
trung bình của hàng trăm vector từ kéo mọi thứ về gần một "vector tiếng Việt
chung" — nên cosine gần như luôn cao. Dùng nó để quyết định trả lời hay không sẽ
làm bot trả lời bừa nhiều hơn hẳn.

Hệ quả thực tế: kể cả khi dùng bản Lai để **xếp hạng**, cổng chấp nhận vẫn phải
là cosine TF-IDF — và trên test_paraphrase, số câu vừa đúng bài vừa qua cổng chỉ
đổi từ **10/28 thành 11/28**.

### 4.5. Vector học được: liên quan theo chủ đề, không phải đồng nghĩa

Cặp đồng nghĩa người dùng hay gõ (cosine; `–` = không có vector):

| Cặp | Cosine | Số lần xuất hiện trong kho |
|---|---|---|
| điện_thoại ~ smartphone | +0,653 | 155 / 55 |
| máy_bay ~ phi_cơ | +0,471 | 176 / **5** |
| học_sinh ~ học_trò | +0,415 | 299 / 20 |
| giá ~ tiền | +0,183 | 398 / 367 |
| bác_sĩ ~ thầy_thuốc | – | 240 / **0** |
| tên_lửa ~ hỏa_tiễn | – | 73 / **0** |
| mỹ ~ hoa_kỳ | – | 297 / **0** |
| ô_tô ~ xe_hơi | – | **0** / **0** |

Hàng xóm gần nhất:

```
máy_bay    : boeing, siêu_thanh, x-59, air_france, bay, hạ_cánh
bác_sĩ     : chuyên_khoa, nội_trú, srimathi, sản_phụ, y, tú
học_sinh   : giáo_viên, năm_học, năng_khiếu, thcs, khai_giảng, trường
dầu        : dầu_thô, diesel, brent, leo_thang, trung_đông, wti
```

Đúng như Lab 05 cảnh báo (*Similar ≠ Synonym*): `máy_bay` gần `boeing`,
`air_france` — cùng chủ đề, không cùng nghĩa. `bác_sĩ` gần `srimathi` — **tên
một người** xuất hiện trong một bài về bác sĩ. Đó là dấu hiệu kinh điển của
embedding học trên quá ít dữ liệu: một lần đồng xuất hiện đã đủ kéo hai từ lại
gần nhau.

---

## 5. Vì sao Word2Vec thua trên dữ liệu này

Bốn nguyên nhân, mỗi nguyên nhân có số đo đi kèm:

1. **Không học được từ đồng nghĩa mà kho không chứa.** Muốn biết "hỏa tiễn" ≈
   "tên lửa", mô hình phải thấy "hỏa tiễn" trong ngữ cảnh giống "tên lửa". Kho
   này có "tên lửa" 73 lần, "hỏa tiễn" **0 lần**; "thầy thuốc" 0 lần; "ô tô" và
   "xe hơi" đều 0. Từ đồng nghĩa người dùng gõ thường chính là từ báo **không
   dùng** — nên Word2Vec huấn luyện trên chính kho báo không có vector cho nó.
   Ví dụ mở đầu của Lab 05 (`ô_tô` ≠ `xe_hơi`) thậm chí không xảy ra được ở đây.

2. **Dữ liệu quá ít cho embedding.** 164 nghìn token, chỉ 4.785 từ xuất hiện ≥ 5
   lần. Word2Vec thường được huấn luyện trên hàng trăm triệu đến hàng tỷ token.
   Với từng này dữ liệu, "gần nhau" chủ yếu nghĩa là "tình cờ cùng một bài".

3. **Lấy trung bình pha loãng đúng thứ quan trọng nhất.** Câu hỏi tin tức xoay
   quanh **tên riêng hiếm** — "SCB", "Trương Mỹ Lan", "Sơn Đoòng". IDF của
   TF-IDF khuếch đại chúng; trung bình vector thì trộn chúng ngang hàng với mọi
   từ chung chung khác. Đó là lý do Word2Vec thua 37/41 câu có khác biệt trên test.

4. **Vector trung bình không tách được trong / ngoài phạm vi** (mục 4.4), mà việc
   biết từ chối lại là nguyên tắc cốt lõi của kiến trúc.

Không nguyên nhân nào là lỗi cài đặt — kiểm thử gradient và kho đồ chơi ở mục 2
cho thấy thuật toán chạy đúng.

---

## 6. Kết luận và quyết định

**Không đưa Word2Vec (huấn luyện trên kho của đồ án) vào chatbot. Giữ TF-IDF.**

Căn cứ, tóm tắt để trình bày:

| Tiêu chí | Kết quả | Nguồn |
|---|---|---|
| Xếp hạng câu hỏi thật | Word2Vec kém TF-IDF **có ý nghĩa** (p < 0,001) | 4.2 |
| Xếp hạng câu diễn đạt lại — mục đích của embedding | **không** tốt hơn TF-IDF | 4.2 |
| Ghép hai tín hiệu | không hơn; mức tăng trên dev không lặp lại trên test | 4.2 |
| Câu không dấu | 7/22 so với 20/22 | 4.3 |
| Làm ngưỡng từ chối | 79,2% so với 91,1% — không an toàn | 4.4 |
| Chất lượng vector | học "cùng chủ đề", không học "cùng nghĩa" | 4.5 |

Đây là một **kết quả âm có kiểm chứng**, cùng loại với BM25 ở docs/06: kỹ thuật
đúng, được cài đặt đúng và đo công bằng — kể cả trên tập câu hỏi thiết kế riêng
để nó có lợi thế — nhưng không cải thiện được hệ thống này.

### Kết luận KHÔNG nói gì

- **Không** nói "embedding vô dụng". Nói là: *embedding tự huấn luyện trên 532
  bài báo* không đủ tốt cho *truy hồi tin tức theo tên riêng*.
- Tập diễn đạt lại nhỏ (28 câu test) nên khoảng tin cậy rộng. Kết luận chắc chắn
  là "không có bằng chứng Word2Vec giúp", còn "Word2Vec chắc chắn kém hơn trên
  câu diễn đạt lại" thì **chưa** khẳng định được.
- Người viết câu hỏi cũng là người làm thí nghiệm; đã giảm thiểu bằng cách viết
  câu trước khi huấn luyện, nhưng không loại bỏ hết được.
- Chỉ thử cách tạo vector tài liệu của Lab 05 (trung bình). Các cách khác
  (Doc2Vec, trung bình có khử thành phần chung như SIF) chưa thử.

### Khi nào embedding đáng thử lại

Số liệu chỉ ra điểm nghẽn là **dữ liệu**, không phải thuật toán. Hướng có triển
vọng thật:

1. **Vector huấn luyện sẵn trên kho tiếng Việt rất lớn** (ví dụ PhoW2V, fastText
   tiếng Việt) — chứa được "hỏa tiễn", "thầy thuốc", "xe hơi". Đổi lại là dùng tài
   nguyên bên ngoài, ra khỏi phạm vi tự cài đặt.
2. **Vector theo n-gram ký tự (fastText)** — xử lý được từ lạ và gõ sai, đúng chỗ
   Word2Vec thuần bó tay.
3. **Phân loại ý định (Buổi 6)** chứ không phải truy hồi. Lab 05 kết thúc bằng
   "cầu nối sang Buổi 6: TF-IDF vs vector trung bình Word2Vec cho **phân loại văn
   bản**". Bài toán đó hợp với embedding hơn: câu ngắn, cần khái quát qua nhiều
   cách nói, không phụ thuộc tên riêng hiếm — và intent classifier đang là điểm
   yếu nhất của đồ án (61,5%). `src/word2vec.py` đã sẵn sàng cho thí nghiệm đó.

