# 07 — RAG với PhoGPT: lần chạy thật đầu tiên, phân tích lỗi, và prompt v2

> Tài liệu này ghi lại **nguyên trạng** lần chạy PhoGPT đầu tiên trên Colab, kể cả
> những chỗ kết quả tệ. Số liệu thô: `data/eval/rag/run1_results.json`,
> `run1_samples.csv`; chấm lại: `run1_results_rescored.csv`
> (`python tools/rescore_rag_run.py`).

## 1. Bối cảnh

Chatbot gốc trích xuất 100% (docs/02). Thí nghiệm n-gram (docs/04) cho thấy tự sinh
văn bản bằng mô hình tự cài đặt thì bịa. Muốn câu trả lời **tự nhiên** mà vẫn
**đúng nguồn** thì dùng RAG: truy hồi TF-IDF tự cài đặt tìm bài, PhoGPT-4B-Chat
(VinAI) diễn đạt lại. Đây là phần **duy nhất** dùng mô hình tiền huấn luyện.

Nguyên tắc từ đầu: **PhoGPT chỉ được gọi khi truy hồi đã có bằng chứng** — câu ngoài
phạm vi bị từ chối trước khi tới mô hình.

Máy cá nhân không có GPU NVIDIA nên mô hình chạy trên Google Colab (T4), người dùng
tự chạy notebook `dist/RAG_PhoGPT_Colab.ipynb` và gửi lại file kết quả.

## 2. Lần chạy 1 — cấu hình

| | |
|---|---|
| Môi trường | Colab, Tesla T4, Python 3.13, torch 2.11, transformers 5.16, underthesea 9.5 |
| Backend | `transformers` **lỗi khi nạp**: mã tùy biến của PhoGPT (kiến trúc MPT) khai báo import `triton_pre_mlir`. Notebook tự chuyển sang **llama.cpp**, GGUF **Q4_K_M** (4 bit) |
| Giải mã | tham lam (temperature 0), repeat_penalty 1.1, tối đa 256 token |
| Prompt | v1: 6 quy tắc đánh số, yêu cầu trích dẫn `[i]`, ngữ cảnh `[i] (đăng dd/mm/yyyy, chuyên mục X)` |
| Câu hỏi | tập **dev** (không đụng test): 25 câu tin tức (seed 2026), 4 câu ca mâu thuẫn, 6 câu bẫy, 5 câu ngoài phạm vi |

**Kiểm tra tái lập:** chạy lại toàn bộ 36 câu (trừ ca mâu thuẫn) trên máy cục bộ —
định tuyến và bài top-1 **trùng 36/36** với Colab. Mọi khác biệt về câu trả lời
đến từ PhoGPT, không phải từ phần tự cài đặt.

## 3. Kết quả tự động — và vì sao một nửa con số là sai

| Chỉ số (21 câu tin tức đi qua RAG) | Notebook báo | Chấm lại (bộ kiểm tra đã sửa) |
|---|---|---|
| Câu có "số bịa" | 12 | **6** |
| Câu có trích dẫn `[i]` hợp lệ | 0 | 0 |
| Độ trễ trung bình / p90 | 1.84 s / 4.9 s | — |
| Ngoài phạm vi: LLM không bị gọi | 5/5 | 5/5 |

Bộ kiểm tra số bịa có **lỗi của chính nó**: nó đếm số thứ tự đầu dòng ("1.", "2.",
...) là số bịa — mô hình hay trả lời bằng danh sách nên 7/12 câu bị đếm oan một
phần. Đã sửa (bỏ ký hiệu danh sách, chấp nhận ngày đăng của nguồn, hiểu
"100 nghìn" = "100.000", "9h40" = "9 giờ 40"). 6 câu còn lại là **bịa thật**:

- 3 câu kết thúc bằng một dòng chép lại định dạng ngữ cảnh kèm **ngày bịa**:
  `[2]  (đăng 06/30/2021, chuyên mục Khoa học)` — sai cả năm lẫn kiểu viết ngày.
- "robot chó dẫn đường ... ra mắt ... vào năm **2016**" — bài không có năm nào như vậy.
- 2 câu gắn năm **2022** vào sự kiện xảy ra năm 2026.

**Trích dẫn:** 0/21 câu trích dẫn đúng kiểu `[i]` ngay sau thông tin. Mô hình chỉ
chép lại đầu mục `[i] (đăng ...)` như một dòng riêng. Kết luận: yêu cầu trích dẫn
bằng prompt không hiệu quả với mô hình 4B; nguồn nên được **gắn bằng code**
(notebook và chatbot vốn đã in danh sách nguồn).

## 4. Đọc từng câu

> Chấm bởi Claude (trợ lý AI) trong lúc phân tích, **đối chiếu từng khẳng định
> với bài gốc** trong corpus (tìm chuỗi trong toàn văn bài). Một người chấm, 21 câu —
> đủ để thấy dạng lỗi, không đủ để báo tỷ lệ chính xác. File CSV có sẵn cột để
> người chấm lại.

| Nhãn | Nghĩa | Số câu | Ví dụ |
|---|---|---|---|
| **A** | Đúng nguồn, trả lời được câu hỏi | **3** | "ten lua spectrum..." → câu trả lời đầy đủ, đúng giờ phóng |
| **B** | Đúng nguồn nhưng trình bày hỏng (danh sách, lặp câu, đuôi chép prompt) | **5** | Blue Origin: nội dung đúng + dòng `[2] (đăng 06/30/2021...)` |
| **C** | Không trả lời (lặp lại truy vấn) | **3** | "Huawei Mate XT2 gập ba" → "Huawei Mate XT2 gập ba." |
| **D** | Có thông tin **sai** | **8** | xem dưới |
| **E** | Từ chối dù bài có câu trả lời | **2** | "bán kết Mỹ Mở rộng nữ có bốn hạt giống" → "không đề cập" |

Các câu **D** — dạng lỗi quan trọng nhất:

| Câu hỏi | PhoGPT viết | Bài gốc |
|---|---|---|
| hạn hán kênh đào panama | "Hạn hán ... đang gây ra sự đình trệ ở eo biển Hormuz" | "**Xung đột** gây ra sự đình trệ ở eo biển Hormuz" — đảo quan hệ nhân quả |
| du khách mắc kẹt trên đảo Cát Bà | "không có du khách nào bị ảnh hưởng" | "hỗ trợ khách **bị ảnh hưởng** vì lệnh cấm biển" |
| vụ đắm tàu Costa Concordia | "trên đường từ Ý đến **Brazil**" | không có Brazil; rồi tự viết thêm "Câu hỏi 1: ... Trả lời: ..." |
| robot chó dẫn đường trung quốc | "năm 2016", "Lidar", trộn robot Lynx của bài khác | không có 2016, Lidar; Lynx là bài khác |
| học sinh việt bị bắt nạt | "Tô Lâm nhận định hồi cuối tháng 6/**2022**" | "Hồi cuối tháng 6" (2026) |
| học sinh huế thiếu sách giáo khoa | "ngày 9/9 năm **2022**" | ngày 9/9, bài đăng 2026 |
| bác sĩ trẻ về trạm y tế | tự đặt ra 8 câu hỏi, trong đó "ngày 10/9 năm 2022" | chương trình bắt đầu 2022, sự kiện 10/9 là 2026 |
| bác sĩ nội trú chọn sản phụ khoa | "Bác sĩ nội trú chọn sản phụ khoa vì chuyên môn cao" | **truy hồi sai bài**, mô hình bịa câu trả lời cho khớp câu hỏi |

Nhận xét: chỉ **3/8** câu D có lỗi nằm ở **con số**. 5/8 là lỗi ngữ nghĩa (đảo nhân
quả, phủ định bịa, trộn hai bài, bịa cho khớp câu hỏi) — không quy tắc so khớp nào
bắt được. Riêng câu truy hồi sai cho thấy RAG **nguy hiểm hơn** trích xuất khi truy
hồi sai: bot trích xuất sẽ trả về một bài không liên quan (người đọc thấy ngay),
còn PhoGPT viết một câu trả lời trôi chảy, sai, trông như đúng.

## 5. Ca mâu thuẫn, câu bẫy, ngoài phạm vi

**Mâu thuẫn** (giá vé tàu Cát Linh: bài 01/09 "tăng lên 15.000", bài 10/09 "hoãn,
giữ 8.000"): **3/3** câu đi qua RAG trả lời theo **tin mới** (8.000 đồng). Câu tốt
nhất: "Giá vé ... giữ nguyên ở mức 8.000 đồng mỗi lượt thay vì tăng lên 15.000 đồng
như thông báo trước đó." Nhưng: 1 câu gán lý do của bài cũ ("nhằm bù đắp chi phí
vận hành") cho quyết định hoãn; 2 câu kèm đuôi ngày bịa. Câu thứ 4 ("tàu cát linh
15.000 đồng") không tới được PhoGPT — xem mục 6.

**Bẫy** (hỏi chi tiết mà bài không có): **0/5** xử lý đúng.

> **Đính chính (sau lần chạy 2):** ban đầu có 6 câu bẫy. Câu "vì sao kem Tràng
> Tiền phải đóng cửa" bị ghi nhầm là bẫy — bài **có** nêu lý do: cơ sở 35 Tràng
> Tiền "hoàn thành sứ mệnh lịch sử" sau 68 năm, việc chuyển đổi nằm trong kế hoạch
> phát triển hệ thống. Lỗi do lúc kiểm tra chỉ tìm các chuỗi "lý do", "vì" thay vì
> đọc cả bài. Câu này được giữ lại như câu hỏi **hợp lệ** (`dev_traps.json`,
> type `hợp_lệ`). Bài học áp dụng cho bộ bẫy test: đọc toàn văn từng bài.

| Câu bẫy | PhoGPT v1 |
|---|---|
| lợi nhuận **năm 2020** của metro Bến Thành Suối Tiên | "Lợi nhuận năm 2020 ... là 27 tỷ đồng." — nhận giả định sai, gán số của mục tiêu 2026-2030 |
| giữ cơm nguội trong tủ lạnh **30 ngày** có sao không | "Không nên giữ ... quá 30 ngày." — bài khuyên không giữ 5-7 ngày |
| đảo Hải Nam miễn visa **từ năm 2015** phải không | "Đúng." |
| AirPods 5 có chống nước chuẩn **IP68** không | chỉ in ra một dòng `[1] (đăng 10/09/2016, ...)` |
| Messi mua CLB Eldense với giá bao nhiêu | đầu mục chép lại + "Câu hỏi chứa giả định sai" |
| *(hợp lệ)* vì sao kem Tràng Tiền phải đóng cửa | chép lại nguyên danh sách quy tắc — trả lời hỏng một câu hỏi có đáp án |

Quy tắc số 6 trong prompt ("nếu câu hỏi chứa giả định sai, hãy chỉ ra") **không có
tác dụng**: mô hình 4B có xu hướng đồng ý với người hỏi.

**Ngoài phạm vi:** 5/5 không gọi LLM — nguyên tắc an toàn hoạt động đúng thiết kế.

## 6. Lỗi tìm ra ngoài lề: bộ chuẩn hóa teencode làm hỏng con số

Khi đo chốt chặn giả định (mục 7) thì câu dev "nhan vien openai tieu **7000** do..."
bị báo nhầm với chi tiết «70». Truy ra: bước gộp ký tự lặp ("đẹppppp" → "đẹp")
dùng regex `(.)\1{2,}` — gộp **cả chữ số**: `15.000 → 15.0`, `2000 → 20`,
`1000 → 10`. Đây là lỗi âm thầm có từ docs/03, ảnh hưởng mọi câu hỏi có con số.

- Sửa: chỉ gộp **chữ cái** `([^\W\d_])\1{2,}`. Có test hồi quy.
- ERR teencode trên ViLexNorm test: 67.54% → **67.84%** (docs/03 đã đính chính).
- Ảnh hưởng tới số đã báo cáo: chuẩn hóa thay đổi ở **1/195** câu dev và **0/178**
  câu test; dò lại tham số trên dev cho **đúng bộ tham số cũ**. Vì vậy các số trên
  tập test ở docs/06 **không đổi** — không cần xem lại tập test.
- Câu mâu thuẫn "tàu cát linh 15.000 đồng" nay giữ đúng con số, nhưng vẫn không
  tới PhoGPT: intent classifier xếp nó vào "ngoài phạm vi" (thêm "giá vé" vào
  câu thì truy hồi đúng). Đây là điểm yếu intent đã biết (docs/06), không sửa ở đây
  vì sửa intent cần tập test mới.

## 7. Prompt v2 và hai chốt chặn

Mỗi thay đổi ứng với một lỗi **quan sát được** ở lần chạy 1:

| Lỗi v1 | Thay đổi v2 |
|---|---|
| Trả lời bằng danh sách, chép lại quy tắc | Hướng dẫn là **một đoạn văn xuôi**, không đánh số |
| Chép đầu mục `[i] (đăng ...)` kèm ngày bịa | Ngữ cảnh viết `Tin i (ngày ...): tiêu đề`; **hậu xử lý** bỏ dòng chép prompt, bỏ ký hiệu danh sách, lấy đoạn đầu, bỏ câu lặp, tối đa 4 câu; chuỗi dừng `### ` |
| 0/21 trích dẫn đúng | Bỏ yêu cầu trích dẫn; nguồn gắn bằng code |
| Lặp lại truy vấn từ khóa / tự đặt câu hỏi | Câu không phải câu hỏi được chuyển thành "Các tin trên cho biết gì về X?" (`is_question`, `frame_question`) |
| Câu hỏi nằm giữa prompt dài | Câu hỏi đặt **cuối**, sát chỗ mô hình bắt đầu viết |
| Đồng ý với giả định sai có con số | **Chốt chặn giả định** (trước LLM): con số / mã hiệu trong câu hỏi mà **không bài nào nhắc tới** → không hiển thị câu sinh, trả lời "các bài báo không nhắc tới «2015»" + trích xuất |
| Bịa con số | **Chốt chặn số bịa** (sau LLM): câu sinh có số không có trong nguồn → hiển thị câu trích xuất gốc thay thế |
| Câu sinh rỗng sau khi làm sạch | → hiển thị câu trích xuất |
| Lượng tử 4 bit | GGUF **Q8_0** (T4 thừa bộ nhớ) |

Hai chốt chặn là **quy tắc tất định** — regex + so khớp số theo giá trị (tinh thần
Lab01) — nên kiểm thử được hoàn toàn không cần GPU (`tests/test_rag.py`, 25 test,
các chuỗi đầu vào chép nguyên từ đầu ra thật của lần chạy 1).

v1 được **giữ nguyên trong code** (`prompt_version="v1"`, không chốt chặn) để lần
chạy 2 so sánh v1/v2 trên cùng mô hình, cùng phiên.

## 8. Đo được ngay, không cần GPU

**Chốt chặn giả định có chặn nhầm câu hợp lệ không?** Chạy trên mọi câu tin tức
tới được RAG (không cần LLM):

| Tập | Câu tới RAG | Chốt chặn kích hoạt | Trong đó truy hồi top-1 **đúng** (= chặn nhầm) |
|---|---|---|---|
| dev | 93 | 0 | 0 |
| test | 97 | 1 | 0 |

Lần kích hoạt duy nhất trên test ("hai bà cụ ngoại **90** tuổi đi du lịch") là câu
mà **truy hồi đã sai bài** — bài lấy về không nhắc tới 90 tuổi. Tức là chốt chặn
còn bắt được một lỗi truy hồi. (Trước khi sửa lỗi mục 6, dev có 1 lần chặn nhầm —
chính là lỗi đó.) Thiết kế chốt chặn chỉ dựa trên dev; tập test được chạy **một
lần** sau khi chốt, để báo cáo.

**Hậu xử lý + chốt chặn của v2 áp lên đúng các câu v1 đã sinh** (`rescore_rag_run.py`)
— tách phần cải thiện đến từ hậu xử lý khỏi phần đến từ prompt mới:

| | v1 như đã chạy | v1 + hậu xử lý v2 + chốt chặn |
|---|---|---|
| Bẫy xử lý an toàn | 0/5 | **5/5** — 4 chốt chặn giả định (IP68, 2020, 30, 2015), Messi rỗng sau làm sạch → trích xuất |
| Câu tin tức hiển thị còn số bịa | 6/21 | **0/18** (3 câu bị thay bằng trích xuất; 3 câu hết số bịa vì dòng đuôi bị cắt) |
| Câu D còn được hiển thị | 8 | **5** (Panama, Cát Bà, Costa, bác sĩ trẻ, sản phụ khoa) |
| Câu C (lặp truy vấn) | 3 | 3 — hậu xử lý không sửa được; cần prompt mới |

Đọc kỹ con số "5/5 bẫy": 4 câu được chặn **bởi thiết kế** (chúng có con số, đúng
loại chốt chặn nhắm tới); câu Messi an toàn **vì may** — đầu ra quá hỏng nên làm
sạch xong không còn gì. Với một đầu ra trôi chảy mà sai, không chốt chặn nào bắt
được — lần chạy 2 xác nhận đúng điều này (mục 10).

## 9. Giới hạn

1. **Hai chốt chặn chỉ bắt lỗi có con số / mã hiệu.** 5/8 lỗi D của v1 là lỗi ngữ
   nghĩa. Bắt chúng cần kiểm tra suy diễn (NLI) — ngoài phạm vi đồ án.
2. **Cỡ mẫu nhỏ**: 21 câu tin tức, 6 bẫy, 1 người chấm. Báo cáo dạng lỗi, không
   báo "độ chính xác của RAG" như một con số có ý nghĩa thống kê.
3. **Truy hồi sai → RAG bịa trôi chảy.** Bot trích xuất lộ lỗi truy hồi ra ngoài;
   RAG che nó đi. RAG chỉ tốt khi truy hồi tốt.
4. Làm sạch câu sinh dùng mẫu regex rút ra từ **chính các đầu ra dev** của lần 1 —
   đo trên cùng các câu đó (mục 8) là lạc quan. Lần chạy 2 mới là phép đo trên
   đầu ra chưa thấy.
5. **Câu hỏi xác nhận bị chặn trước khi tới RAG.** "...đúng không", "...phải
   không" hay bị intent classifier xếp vào intent hội thoại — đúng loại câu mà
   người dùng dùng để kiểm tra một giả định. Đây là điểm yếu của phần tự cài đặt
   (docs/06), không phải của PhoGPT.

## 10. Lần chạy 2 — v1 và v2 trên cùng mô hình (dev)

Cấu hình: Colab T4, llama.cpp, GGUF **Q8_0**, cùng 40 câu dev như lần 1; mỗi câu chạy
qua v1 (prompt cũ, không chốt chặn, 256 token) và v2 (160 token). Khi chốt chặn
giả định chặn, PhoGPT **vẫn được gọi** để ghi lại nó định nói gì. Số liệu thô:
`data/eval/rag/run2_results.json`; chấm lại bằng mã hiện tại:
`run2_results_rescored.csv`.

### Chấm tay 21 câu tin tức (cùng thang A–E như mục 4)

| | v1 · Q4_K_M (lần 1) | v1 · Q8_0 (lần 2) | **v2 · Q8_0 (lần 2)** |
|---|---|---|---|
| **A** đúng, trả lời được | 3 | 4 | **10** |
| **B** đúng, trình bày hỏng | 5 | 3 | 3 |
| **C** không trả lời (lặp truy vấn) | 3 | 4 | 3 |
| **D** có thông tin sai | 8 | 7 | **5** |
| **E** từ chối sai | 2 | 3 | **0** |

Câu PhoGPT sinh, **trước** mọi chốt chặn. Đọc bảng:

- **Lượng tử hóa không phải vấn đề.** v1 trên Q4_K_M và trên Q8_0 cho phân bố lỗi
  gần như nhau. Lỗi nằm ở prompt, không ở độ chính xác số của trọng số.
- **Prompt v2 giúp thật.** So có cặp từng câu: 6 câu v2 đạt A mà v1 không, 0 câu
  ngược lại (kiểm định dấu p ≈ 0.03). Từ chối sai biến mất (Bali, Mỹ Mở rộng,
  Estonia giờ trả lời đúng). **Nhưng** v2 được thiết kế từ đầu ra lần 1 trên chính
  các câu này, và người chấm biết câu nào là v2 — con số này **lạc quan**. Phép đo
  không thiên lệch là lần chạy 3 trên tập test.
- **v2 vẫn sai 5/21 câu.** Không cần con số để sai:

| Câu hỏi | v2 viết | Bài gốc |
|---|---|---|
| hàng nhập dưới 100 nghìn có được miễn thuế | "hàng nhập **dưới** 100 nghìn đồng có thể không còn được miễn thuế" | hạ ngưỡng miễn thuế xuống 100.000 → hàng **trên** 100.000 mất miễn thuế. Mô hình lấy chữ của câu hỏi đảo nghĩa câu trả lời (v1 cũng sai y hệt) |
| bán kết Mỹ Mở rộng nữ có bốn hạt giống | "... Wimbledon 2009 và đây cũng chính là năm Sabalenka lọt vào vòng đấu này" | nửa sau bịa |
| du khách mắc kẹt trên đảo Cát Bà | "sau khi bão số 4 **đổ bộ**" | bài không nói bão đã đổ bộ (lỗi nhẹ) |
| vụ nổ tên lửa blue origin | "Chornobyl năm **196**" | 1986 — chốt chặn số bịa **bắt được** |
| bác sĩ nội trú chọn sản phụ khoa | trả lời như thể bài nói về bác sĩ nội trú | truy hồi sai bài — chốt chặn số bịa chặn được, nhưng **vì may** (xem dưới) |

### Chốt chặn trong lần chạy 2 — cả đúng lẫn sai

| Chốt chặn | Kích hoạt đúng | Kích hoạt sai / may |
|---|---|---|
| Giả định (bẫy) | **4/4** bẫy có con số: PhoGPT v2 định trả lời "Có, ... chuẩn IP68", "Không nên giữ ... quá 30 ngày", "Đúng." — đều sai, đều bị chặn | 0 |
| Số bịa | Blue Origin "196"; bẫy Messi "khoảng 1,5 triệu euro" (bịa hoàn toàn) | "học sinh bị bắt nạt": câu trả lời **đúng** nhưng có nhãn "Tin 1:" chép từ ngữ cảnh → chữ số "1" bị coi là số bịa. "sản phụ khoa": chặn vì "1 bé gái" (bài viết "một bé gái") — câu đáng chặn nhưng chặn vì lý do sai |

Trên bẫy (5 câu thật), PhoGPT v2 **tự nó** chỉ xử lý đúng 1/5 (metro: không nhận
số của năm 2020). Có chốt chặn: **5/5** an toàn. Câu Tràng Tiền (hợp lệ): cả v1 lẫn
v2 trên Q8_0 đều trả lời **đúng** theo bài.

**Ca mâu thuẫn:** v2 đúng 3/3 (theo tin mới, 8.000 đồng, kể cả lý do hoãn lấy đúng
từ bài mới); v1 trên Q8_0 có 1 câu tự mâu thuẫn ("Không ... 8.000" rồi "Đúng vậy").
**Độ trễ** v2: trung bình 1.5 s, p90 2.8 s (v1: 1.6 s / 4.9 s).

### Sửa sau lần chạy 2 (chỉ hậu xử lý — không đổi chữ nào của prompt)

| Lỗi quan sát được | Sửa |
|---|---|
| Nhãn "Tin 1:" làm chốt chặn số bịa bắt nhầm | Bỏ nhãn "Tin N:" khi làm sạch |
| "Các tin trên cho biết về vụ đắm tàu Costa Concordia." — rỗng | **Chốt chặn lặp lại**: câu không thêm âm tiết nội dung nào ngoài câu hỏi → trích xuất. Chỉ áp cho câu dạng từ khóa — câu hỏi có/không được dùng lại chữ của câu hỏi ("Vé tàu Cát Linh không tăng giá." là câu trả lời đúng; bản đầu của chốt chặn đã chặn nhầm câu này khi chấm lại, nên phải thu hẹp) |
| Mở đầu "Các tin trên cho biết..." (người dùng không thấy "các tin trên") | Đổi thành "Theo các bài báo,"; không viết thường danh từ riêng ("tP HCM" là lỗi đã gặp) |
| Ký tự ``` lọt vào câu trả lời | Bỏ |
| (suy luận, không từ dữ liệu) "5 triệu" khớp với một chữ số 5 bất kỳ trong bài | Số có đơn vị chỉ khớp theo **giá trị** (5.000.000) |

Chấm lại đầu ra v2 lần 2 bằng mã đã sửa: 17/21 câu hiển thị câu sinh, 2 chặn vì lặp
lại, 2 chặn vì số bịa; trong 17 câu hiển thị còn **3 câu D** (hàng nhập, Mỹ Mở
rộng, Cát Bà). Chốt chặn giả định sau khi sửa vẫn chặn nhầm **0** câu dev (93) và
**0** câu test đúng bài (97; lần kích hoạt duy nhất là câu truy hồi sai bài).

## 11. Lần chạy 3 — tập test, một lần (đang chờ)

Chốt trước khi chạy (commit riêng):
- Mã `rag.py` như hiện tại; không sửa gì sau khi xem kết quả.
- 40 câu tin tức test (seed 2026), 4 câu ca mâu thuẫn, **24** câu ngoài phạm vi.
- **11 câu bẫy mới** (`data/eval/rag/test_traps.json`), đã đối chiếu toàn văn bài:
  2 bẫy có con số mà chốt chặn nhắm tới, 1 bẫy có con số **nằm ở chỗ khác trong bài**
  (điểm mù đã biết: "cao 500 m" trong khi bài có "500 MW"), 6 giả định sai **không
  có số**, 2 câu hỏi chi tiết bài không có. Bộ này cố ý có nhiều câu chốt chặn
  **không** bắt được, để đo PhoGPT tự nó.
- Phát hiện khi soạn bẫy: **4/11 cách hỏi tự nhiên không tới được RAG** — câu xác
  nhận "...đúng không", "...phải không" bị intent classifier xếp nhầm. 2 câu thử
  lại theo thứ tự biến thể cố định thì tới được; 2 câu (Harvard, Tim Cook) không
  biến thể nào tới được → giữ trong bộ với ghi chú, bot trả lời bằng nhánh trích
  xuất / không tìm thấy.
