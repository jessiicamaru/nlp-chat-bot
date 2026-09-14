# 08 — Runbook: crawl dữ liệu hằng ngày và cập nhật mô hình

> Tài liệu vận hành. Đọc hết mục 1–2 **trước** khi chạy lần đầu.
> Lệnh chạy trên Windows PowerShell, từ thư mục `D:\School\nlp\final-project`.

**Tóm tắt một dòng:**

```powershell
powershell -ExecutionPolicy Bypass -File tools\daily_update.ps1 -RestartApi
```

Lệnh này sao lưu kho bài báo → crawl bài mới từ VnExpress → dựng lại index → chạy kiểm thử
→ khởi động lại web server để nạp dữ liệu mới. Mục 6 hướng dẫn cho nó tự chạy mỗi ngày.

---

## 1. Hiểu trước khi chạy: "huấn luyện lại" nghĩa là gì ở dự án này

Chatbot không có một bước "train" riêng như mô hình học sâu. Cập nhật dữ liệu gồm ba thứ:

| Thành phần | Học từ đâu | Khi dữ liệu mới về thì sao |
|---|---|---|
| **Kho bài báo** `data/raw/corpus_raw.csv` | `src/crawler.py` | **Tích lũy**: chỉ thêm bài có URL mới, không bao giờ xóa bài cũ |
| **Index truy hồi** `models/retriever_index.joblib` | TF-IDF trên toàn kho | **Dựng lại toàn bộ** (IDF của mọi từ đổi). Tự động: index lưu kèm vân tay SHA-256 của kho, kho đổi thì lần nạp sau tự dựng lại |
| **Intent classifier** | `data/intents/intents_vi.json` | Không phụ thuộc bài báo. Học lại mỗi lần bot khởi động (< 1 giây) |
| **Từ điển teencode** `data/resources/teencode_lexicon.json` | ViLexNorm | Không phụ thuộc bài báo. **Không** cần chạy lại hằng ngày |

Ba hệ quả quan trọng:

1. **Web server nạp mô hình một lần lúc khởi động.** Crawl xong mà không khởi động lại
   `src/api.py` thì web vẫn trả lời bằng dữ liệu cũ.
2. **Mốc thời gian của "độ mới" là ngày đăng mới nhất trong kho**, không phải ngày hôm nay
   (docs/05). Crawl thêm bài mới thì mốc tự tiến lên — không cần chỉnh gì.
3. **Dựng index tốn thời gian tỷ lệ với số bài** (~24 giây cho 381 bài, gần như toàn bộ là
   tách từ; ~1 phút cho 1.000 bài, ~9 phút cho 10.000 bài). Crawl mỗi ngày làm kho lớn dần.

---

## 2. ⚠️ Bảo vệ bản nộp bài trước khi cập nhật lần đầu

**Mọi số liệu trong báo cáo** (Recall@1 93,4%, MRR 0,950, kết quả RAG...) được đo trên đúng kho
**381 bài** hiện tại. Kho này được quản lý bằng git. Crawl thêm bài sẽ làm các con số đó **không
tái lập được nữa**, và một số kiểm thử hồi quy có thể không đạt (mục 8).

Làm **một lần**, trước lần cập nhật đầu tiên:

```powershell
cd D:\School\nlp\final-project
git status                               # phải sạch
git tag nop-bai                          # đánh dấu đúng trạng thái đã nộp
git switch -c cap-nhat-du-lieu           # cập nhật hằng ngày trên nhánh riêng
```

Quy tắc:

- **Không commit kho bài báo đã crawl thêm lên nhánh `main`.** Nhánh `main` giữ bản nộp.
- Cần quay lại đúng bản nộp bất cứ lúc nào:

  ```powershell
  git switch main
  .venv\Scripts\python.exe tools\rebuild_index.py   # index tự dựng lại theo kho 381 bài
  ```

---

## 3. Chuẩn bị môi trường (một lần)

```powershell
cd D:\School\nlp\final-project
python -m venv .venv                      # nếu chưa có
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Kiểm tra máy có mạng tới `https://vnexpress.net`. Script dùng thẳng `.venv\Scripts\python.exe`
nên không cần kích hoạt môi trường.

---

## 4. Chạy thủ công từng bước

Dùng khi muốn hiểu từng bước, hoặc khi script tự động báo lỗi.

### 4.1. Sao lưu kho

```powershell
$env:PYTHONIOENCODING = "utf-8"
New-Item -ItemType Directory -Force data\raw\backups | Out-Null
Copy-Item data\raw\corpus_raw.csv "data\raw\backups\corpus_raw_$(Get-Date -Format yyyyMMdd-HHmmss).csv"
```

### 4.2. Crawl bài mới

```powershell
.venv\Scripts\python.exe src\crawler.py --per-category 20
.venv\Scripts\python.exe src\crawler.py --per-category 20 --only "Công nghệ" "Thể thao"
```

| Tham số | Ý nghĩa |
|---|---|
| `--per-category N` | lấy tối đa **N bài mới** mỗi chuyên mục (bài đã có URL trong kho bị bỏ qua). Mặc định 40 |
| `--only ...` | chỉ crawl các chuyên mục được nêu. Tên hợp lệ: `Du lịch`, `Kinh doanh`, `Công nghệ`, `Sức khỏe`, `Giáo dục`, `Thể thao`, `Khoa học`, `Đời sống` |

Crawler đọc tối đa 5 trang danh sách mỗi chuyên mục, nghỉ ngẫu nhiên 0,8–1,6 giây giữa hai request
(lịch sự với máy chủ), loại bài thiếu tiêu đề hoặc phần thân dưới 300 ký tự, và ghi lỗi của **lần
chạy này** vào `data/raw/crawl_errors.csv`. Kho chỉ được ghi ra đĩa **khi crawl xong** — nếu bị
ngắt giữa chừng thì kho cũ vẫn nguyên. Với 20 bài × 8 chuyên mục, một lần chạy mất vài phút.

Cuối lần chạy, crawler in tổng số bài và số bài mỗi chuyên mục.

### 4.3. Dựng lại index và kiểm tra khói

```powershell
.venv\Scripts\python.exe tools\rebuild_index.py
```

Kết quả mong đợi:

```text
Index        : DỰNG LẠI (corpus đã đổi) trong 25.3s
Số bài       : 412
...
Bài mới nhất : 14/09/2026

Kiểm tra khói:
  [OK ] 'tin công nghệ mới nhất' -> route=intent ...
  [OK ] 'có bao nhiêu bài báo' -> route=intent ...
  [OK ] 'thời tiết sao hỏa hôm nay' -> route=fallback ...
```

Nếu dòng đầu ghi "nạp từ cache (corpus không đổi)" sau khi vừa crawl, nghĩa là không có bài mới
nào được thêm — xem `data/raw/crawl_errors.csv`.

### 4.4. Chạy kiểm thử

```powershell
.venv\Scripts\python.exe tests\test_vectorizer.py   # 45 kiểm tra, không phụ thuộc dữ liệu
.venv\Scripts\python.exe tests\test_chatbot.py      # 21 kiểm thử hồi quy, CÓ phụ thuộc dữ liệu (mục 8)
```

### 4.5. Khởi động lại web server

Trong cửa sổ đang chạy `src\api.py`, nhấn **Ctrl+C**, rồi:

```powershell
.venv\Scripts\python.exe src\api.py
```

Mở <http://127.0.0.1:8000> và hỏi thử một tin vừa crawl, ví dụ "tin công nghệ mới nhất" — bài
đầu tiên phải có ngày đăng mới.

---

## 5. Chạy tự động bằng script

`tools/daily_update.ps1` gói toàn bộ mục 4 và tự khôi phục khi có lỗi.

```powershell
powershell -ExecutionPolicy Bypass -File tools\daily_update.ps1                    # crawl + dựng lại + kiểm thử
powershell -ExecutionPolicy Bypass -File tools\daily_update.ps1 -RestartApi        # ... và khởi động lại web
powershell -ExecutionPolicy Bypass -File tools\daily_update.ps1 -PerCategory 30    # lấy nhiều bài hơn
powershell -ExecutionPolicy Bypass -File tools\daily_update.ps1 -SkipCrawl         # chỉ dựng lại + kiểm thử
```

| Tham số | Mặc định | Ý nghĩa |
|---|---|---|
| `-PerCategory` | 20 | số bài mới tối đa mỗi chuyên mục |
| `-SkipCrawl` | tắt | bỏ bước crawl |
| `-SkipTests` | tắt | bỏ kiểm thử (không khuyến nghị) |
| `-RestartApi` | tắt | dừng mọi tiến trình `src/api.py` đang chạy và khởi động một bản mới chạy ẩn |
| `-KeepBackups` | 14 | số bản sao lưu kho giữ lại |

**Các bước script thực hiện:**

1. Sao lưu kho vào `data\raw\backups\corpus_raw_<thời điểm>.csv`.
2. Crawl. Nếu crawler lỗi → **khôi phục kho từ bản sao lưu**, thoát mã 1.
3. Kiểm tra số bài không bị giảm. Nếu giảm → khôi phục, thoát mã 1.
4. Dựng lại index + kiểm tra khói (`tools/rebuild_index.py`). Nếu lỗi → khôi phục, thoát mã 1.
5. Chạy `test_vectorizer.py` và `test_chatbot.py`. Nếu có kiểm thử không đạt → **giữ** dữ liệu
   mới, thoát mã 2 (xem mục 8).
6. Xóa các bản sao lưu cũ hơn 14 bản gần nhất.
7. Khởi động lại web server nếu có `-RestartApi`.

**Mã thoát:**

| Mã | Nghĩa | Việc cần làm |
|---|---|---|
| 0 | cập nhật thành công | không |
| 1 | lỗi, kho đã được khôi phục về trước khi crawl | đọc log, xem mục 10 |
| 2 | dữ liệu đã cập nhật nhưng kiểm thử hồi quy có câu không đạt | đọc mục 8 |

**Log** của mỗi lần chạy: `logs\daily_update_<thời điểm>.log`. Thư mục `logs\` và
`data\raw\backups\` không được đưa vào git.

---

## 6. Lên lịch chạy hằng ngày (Windows Task Scheduler)

Mở PowerShell **với tài khoản của bạn** (không cần quyền quản trị) và chạy:

```powershell
$project  = "D:\School\nlp\final-project"
$action   = New-ScheduledTaskAction -Execute "powershell.exe" `
              -Argument "-NoProfile -ExecutionPolicy Bypass -File `"$project\tools\daily_update.ps1`" -RestartApi" `
              -WorkingDirectory $project
$trigger  = New-ScheduledTaskTrigger -Daily -At 07:00
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -ExecutionTimeLimit (New-TimeSpan -Hours 1)
Register-ScheduledTask -TaskName "NewsChatbot-DailyUpdate" -Action $action -Trigger $trigger `
  -Settings $settings -Description "Crawl VnExpress, dựng lại index, kiểm thử, khởi động lại web chatbot"
```

- `-StartWhenAvailable`: nếu máy tắt lúc 07:00 thì chạy bù ngay khi máy bật lại.
- `-ExecutionTimeLimit 1 giờ`: chặn trường hợp crawl bị treo.
- Mặc định tác vụ chỉ chạy khi bạn **đang đăng nhập** Windows.

**Quản lý tác vụ:**

```powershell
Start-ScheduledTask   -TaskName "NewsChatbot-DailyUpdate"                 # chạy ngay để thử
Get-ScheduledTaskInfo -TaskName "NewsChatbot-DailyUpdate"                 # LastRunTime, LastTaskResult (0/1/2)
Disable-ScheduledTask -TaskName "NewsChatbot-DailyUpdate"                 # tạm dừng
Enable-ScheduledTask  -TaskName "NewsChatbot-DailyUpdate"
Unregister-ScheduledTask -TaskName "NewsChatbot-DailyUpdate" -Confirm:$false   # xóa hẳn
```

Sau lần chạy theo lịch đầu tiên, hãy mở <http://127.0.0.1:8000> để xác nhận web server đã được
khởi động lại và vẫn đang chạy.

---

## 7. Kiểm tra sau mỗi lần cập nhật

- [ ] Dòng cuối log là `Kết thúc với mã 0`.
- [ ] Dòng `Corpus: A -> B bài (+N)` có N > 0. Nếu N = 0 nhiều ngày liền, xem mục 10.
- [ ] `Bài mới nhất` là ngày gần đây.
- [ ] `data\raw\crawl_errors.csv` không toàn lỗi cùng một kiểu (dấu hiệu VnExpress đổi giao diện).
- [ ] Trên web, "tin công nghệ mới nhất" trả về bài có ngày đăng mới.

---

## 8. Khi kiểm thử hồi quy không đạt (mã thoát 2)

`tests/test_vectorizer.py` **không** phụ thuộc dữ liệu: nếu nó không đạt thì đó là lỗi mã thật —
dừng cập nhật và kiểm tra lại mã nguồn.

`tests/test_chatbot.py` có một số kiểm thử **gắn với bài báo cụ thể** của kho 381 bài, ví dụ:

| Kiểm thử | Mong đợi |
|---|---|
| `truy_hoi_cau_khong_dau`, `truy_hoi_cau_teencode` | câu hỏi về đảo Hải Nam trả về đúng bài "Đảo Hải Nam 'chưa dễ hút khách Việt'..." |
| `chuyen_muc_kem_chu_de_thi_tim_trong_muc` | "tin du lịch ninh bình" trả về bài có "Ninh Bình" trong tiêu đề |
| `cau_ngoai_pham_vi_bi_tu_choi` | "thời tiết sao hỏa hôm nay thế nào" bị từ chối — có thể không đạt nếu kho mới có bài trùng nhiều từ khóa với câu này (ví dụ một bài về sao Hỏa) |

Khi kho có thêm bài mới về cùng chủ đề (ví dụ một bài Hải Nam mới hơn), bot có thể **đúng** khi xếp
bài mới lên đầu — kiểm thử vẫn báo không đạt vì nó mong đợi bài cũ. Cách phân biệt:

```powershell
$env:PYTHONIOENCODING = "utf-8"
.venv\Scripts\python.exe src\cli.py --explain "tin ve dao hai nam"
```

- Bài đứng đầu **cùng chủ đề và mới hơn** → hành vi đúng; dữ liệu mới được giữ. Kiểm thử đó chỉ
  đạt lại trên nhánh `main` (kho gốc).
- Bài đứng đầu **không liên quan** → hồi quy thật (thường do bài mới làm loãng từ khóa). Khôi phục
  kho từ bản sao lưu (mục 11) và báo lại.

**Kiểm thử phụ thuộc NGÀY của kho — `tin_moi_phu_dinh_tin_cu`.**

> **Lịch sử.** Lần crawl thật đầu tiên (14/09/2026, +151 bài) làm kiểm thử này **không đạt**, và đó
> là một lỗi thiết kế thật chứ không phải lỗi dữ liệu: độ mới khi đó tính theo ngày đăng mới nhất
> **của cả kho**, nên crawl thêm bài không liên quan cũng làm thứ hạng của câu hỏi khác đi.
> Đã sửa — mốc nay tính theo các bài **đang cạnh tranh** cho chính câu hỏi đó (docs/05, docs/09).
> Kiểm thử này vì vậy **phải luôn đạt trở lại**, kể cả sau nhiều lần crawl.

Có thêm một kiểm thử canh đúng tính chất đó: `crawl_them_bai_khong_lien_quan_khong_doi_thu_hang` —
thêm một bài không liên quan có ngày đăng ở tương lai **không được** làm đổi thứ hạng. Nếu kiểm thử
này không đạt sau khi cập nhật dữ liệu, đừng chỉnh tham số: đó là dấu hiệu phần tính độ mới bị hỏng.

**Kiểm thử đường dự phòng gõ sai** (`go_sai_chinh_ta_van_tim_duoc`) gắn với bài "Chiêm nghiệm từ cuộc
thám hiểm Sơn Đoòng" (crawl ngày 14/09/2026). Nếu kho không còn bài đó, kiểm thử tự bỏ qua và in chú
thích, không báo lỗi giả.

Các kiểm thử không gắn bài cụ thể **và** không phụ thuộc ngày (regex, chuẩn hóa teencode, giữ hệ quy
chiếu dấu, so khớp chuyên mục theo âm tiết, snippet không lặp câu, đầu vào rỗng, đọc ngày VnExpress,
cache tính lại độ mới) **phải luôn đạt**. Nếu một trong số đó không đạt thì đó là lỗi thật.

---

## 9. Dò lại tham số định kỳ (hằng tuần hoặc hằng tháng, KHÔNG hằng ngày)

Kho lớn lên thì IDF của mọi từ thay đổi, nên phân bố điểm cosine cũng dịch dần. Ngưỡng chấp nhận
0,13 được dò trên kho 381 bài có thể không còn tối ưu khi kho lớn gấp đôi.

```powershell
$env:PYTHONIOENCODING = "utf-8"
.venv\Scripts\python.exe src\evaluate.py
```

Script dò lại tham số trên DEV, in bảng so sánh, và **cảnh báo nếu `src/config.py` khác tham số vừa
dò**. Chỉ cập nhật `config.py` khi tham số dò trên DEV thay đổi, rồi chạy lại mục 4.3–4.5.

Nếu vừa crawl thêm nhiều dữ liệu, nên sinh lại tập truy vấn gõ sai trước khi dò, vì chúng được sinh
từ tập dev/test:

```powershell
.venv\Scripts\python.exe tools\build_typo_sets.py
```

Bốn lưu ý bắt buộc:

1. **Nhãn đúng của tập đánh giá là URL bài báo trong kho gốc.** Khi có bài mới về cùng sự việc,
   truy vấn có thể trả về bài mới (đúng về nội dung) nhưng bị chấm là sai. Recall đo trên kho đã
   crawl thêm vì vậy **thấp hơn thực tế** và không so trực tiếp được với số trong báo cáo.
2. **Không chỉnh bất kỳ tham số nào dựa trên kết quả phần TEST.** Đây là quy tắc của docs/06;
   `evaluate.py` in cả phần TEST chỉ để báo cáo.
3. Đừng chạy `evaluate.py` trên nhánh `main` rồi commit: nó ghi lại `data/eval/tuned_params.json`
   và `test_results.json` của bản nộp.
4. **Ngưỡng phải dò trên đúng phân bố câu hỏi người dùng thật sự gõ.** Tập câu ngoài phạm vi của dev
   có thêm biến thể **không dấu**, vì câu không dấu dễ lọt hơn câu có dấu. Nếu bổ sung câu mẫu, chỉ
   bổ sung vào DEV — thêm vào TEST là làm hỏng phép đo (docs/09, mục 6).

---

## 10. Xử lý sự cố

| Triệu chứng | Nguyên nhân thường gặp | Cách xử lý |
|---|---|---|
| `+0` bài nhiều ngày liền; `crawl_errors.csv` toàn "thiếu title" hoặc "text quá ngắn" | VnExpress đổi cấu trúc HTML | Mở một bài bằng trình duyệt → Inspect, sửa `SELECTORS` trong `src/crawler.py` (tiêu đề `h1.title-detail`, mô tả `p.description`, thân bài `article.fck_detail p.Normal`, ngày `span.date`) |
| `crawl_errors.csv` toàn "tải thất bại" | mất mạng, bị chặn tạm thời | Kiểm tra mạng; giảm `-PerCategory`; chạy lại sau |
| `+0` bài nhưng không có lỗi | chưa có bài mới trên 5 trang danh sách, hoặc đã crawl hết | Bình thường nếu chạy nhiều lần trong ngày |
| Script thoát mã 1 "corpus bị giảm" | file kho bị ghi hỏng | Script đã tự khôi phục; xem log để biết bước lỗi |
| Chữ tiếng Việt thành ký tự lạ trong cửa sổ lệnh | console không dùng UTF-8 | `$env:PYTHONIOENCODING = "utf-8"` trước khi chạy Python |
| `running scripts is disabled on this system` | chính sách thực thi PowerShell | Chạy qua `powershell -ExecutionPolicy Bypass -File ...` như trong runbook |
| Web vẫn trả lời tin cũ | web server chưa khởi động lại | Mục 4.5, hoặc chạy script với `-RestartApi` |
| `address already in use` / cổng 8000 bận | còn một web server đang chạy | Dùng bản đang chạy, hoặc dừng nó (lệnh dưới bảng) |
| Bot khởi động báo lỗi khi nạp cache | file cache hỏng | Xóa `models\retriever_index.joblib`, chạy `tools\rebuild_index.py` |
| Dựng index ngày càng chậm | kho lớn dần | Bình thường (tuyến tính theo số bài); xem mục 12 |

Dừng mọi web server đang chạy:

```powershell
Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
  Where-Object { $_.CommandLine -match 'src[\\/]api\.py' } |
  ForEach-Object { Stop-Process -Id $_.ProcessId -Force }
```

---

## 11. Khôi phục

**Về một bản sao lưu gần đây:**

```powershell
Get-ChildItem data\raw\backups | Sort-Object Name -Descending | Select-Object -First 5   # xem các bản
Copy-Item data\raw\backups\corpus_raw_20260914-070000.csv data\raw\corpus_raw.csv -Force
.venv\Scripts\python.exe tools\rebuild_index.py
```

**Về đúng bản nộp bài** (kho 381 bài): xem mục 2.

---

## 12. Quy mô và giới hạn

- **Tốc độ tăng của kho:** tối đa `PerCategory × 8` bài mỗi ngày (mặc định 160), thực tế ít hơn vì
  chỉ lấy bài chưa có. Sau một tháng kho có thể vượt 2.000 bài — dựng index khi đó mất khoảng
  2 phút, và khởi động bot có cache vẫn nhanh.
- **Kho chỉ tăng, không tự dọn.** Bài cũ vẫn được trả lời (độ mới chỉ phá thế hòa, docs/05). Nếu
  muốn giới hạn kho, ví dụ chỉ giữ 90 ngày gần nhất, cần thêm một bước lọc theo `published_at` —
  hiện chưa có.
- **Lịch sự với nguồn tin:** giữ độ trễ giữa các request, không chạy nhiều lần mỗi giờ, và chỉ dùng
  dữ liệu cho mục đích học tập.
- **Lớp RAG (docs/07) không nằm trong quy trình hằng ngày** — nó chạy trên Google Colab. Muốn RAG
  dùng kho mới thì tạo lại gói Colab: `python tools/make_colab_bundle.py`.
- **Phần đã kiểm thử và phần chưa:** script đã được chạy thử đầy đủ ở chế độ `-SkipCrawl
  -RestartApi` (sao lưu, dựng index, 45 + 21 kiểm thử, khởi động lại web). Bước crawl thật, nhánh
  khôi phục khi lỗi, và việc đăng ký Task Scheduler **chưa** được chạy thử trên máy này, vì chạy thật
  sẽ thay đổi kho của bản nộp bài — hãy chạy lần đầu trên nhánh `cap-nhat-du-lieu` (mục 2) và đọc log.

---

## Tham chiếu nhanh

```powershell
cd D:\School\nlp\final-project

# lần đầu
git tag nop-bai; git switch -c cap-nhat-du-lieu

# cập nhật ngay
powershell -ExecutionPolicy Bypass -File tools\daily_update.ps1 -RestartApi

# chỉ dựng lại index + kiểm thử
powershell -ExecutionPolicy Bypass -File tools\daily_update.ps1 -SkipCrawl

# xem log mới nhất
Get-ChildItem logs | Sort-Object Name -Descending | Select-Object -First 1 | Get-Content

# trạng thái tác vụ theo lịch
Get-ScheduledTaskInfo -TaskName "NewsChatbot-DailyUpdate"
```
