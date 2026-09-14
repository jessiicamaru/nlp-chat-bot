"""Sinh notebook báo cáo đồ án cuối kỳ (chạy một lần, kết quả nằm ở notebooks/)."""
import json
from pathlib import Path

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text):
    cells.append({
        "cell_type": "code", "execution_count": None, "metadata": {},
        "outputs": [], "source": text.strip("\n").splitlines(keepends=True),
    })


# ---------------------------------------------------------------- HEADER ----
md(r"""
# ĐỒ ÁN CUỐI KỲ — CHATBOT TIN TỨC TIẾNG VIỆT
## Xây dựng from scratch bằng TF-IDF, Cosine Similarity và Naive Bayes

**Học phần:** Xử lý ngôn ngữ tự nhiên
**Sinh viên:** Hoàng Công Dũng — 23IT036

---

## 🎯 Mục tiêu

Xây dựng một chatbot tiếng Việt **không dùng mô hình ngôn ngữ lớn**, chỉ dùng
các kỹ thuật đã học ở Lab 01–04, và **tự cài đặt** phần lõi bằng NumPy thay vì
gọi thư viện có sẵn.

Chatbot trả lời câu hỏi về tin tức dựa trên kho bài báo VnExpress do nhóm tự thu thập.

## 🔗 Đồ án kế thừa gì từ các lab

| Lab | Nội dung đã học | Dùng ở đâu trong đồ án |
|---|---|---|
| **Lab 01** | `sent_tokenize`, `word_tokenize`, `ner`, Regex | Tách câu để chọn snippet; NER + Regex trích thực thể (`entities.py`) |
| **Lab 02** | Crawl web bằng `requests` + `BeautifulSoup`, validation | Mở rộng corpus lên 381 bài / 8 chuyên mục (`crawler.py`) |
| **Lab 03** | `normalize_basic`, `segment_vi`, stopwords, `preprocess_vi(text, config)` | Pipeline tiền xử lý dùng chung (`preprocess.py`) |
| **Lab 04** | Bag of Words, n-gram, TF-IDF, cosine similarity | **Phần lõi**: `vectorizer.py`, `retriever.py`, và phần TF-IDF + cosine của `intent_classifier.py` |

> **Ngoài phạm vi lab** (tự bổ sung): bộ phân lớp Multinomial Naive Bayes — Lab 04
> chỉ dạy *biểu diễn* văn bản, không có phần phân lớp — cùng với BM25, xếp hạng
> theo độ mới, chuẩn hóa teencode, chỉ mục n-gram ký tự và lớp RAG.

## 🏗️ Kiến trúc

```text
                    câu người dùng
                          |
                          v
        [1] preprocess_vi()            <- Lab 03
                          |
                          v
        [2] IntentClassifier           <- TF-IDF + Naive Bayes (tự cài đặt)
             |                    |
   đủ tin cậy|                    | không đủ tin cậy
             v                    v
   thực thi action        [3] NewsRetriever    <- TF-IDF + cosine similarity
   (chào / thống kê /          |         |
    duyệt mục / tóm tắt)  đạt ngưỡng  dưới ngưỡng
                               |         |
                               v         v
                        trả lời có    fallback
                        dẫn nguồn   ("mình chưa biết")
```

**Nguyên tắc thiết kế quan trọng nhất:** mọi đường đi đều có **ngưỡng tin cậy**.
Bot chỉ trả lời khi có căn cứ định lượng; không đủ căn cứ thì nói thẳng là không
biết, thay vì trả về một bài báo ngẫu nhiên bằng giọng chắc chắn.
""")

# ---------------------------------------------------------------- SETUP -----
md("""
---
# PHẦN A — CHUẨN BỊ MÔI TRƯỜNG
""")

code(r"""
import sys, os, time
from pathlib import Path

# Cho phép import các module trong src/
PROJECT_DIR = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_DIR / "src"))

print("Python     :", sys.version.split()[0])
print("Thư mục dự án:", PROJECT_DIR)
""")

code(r"""
import pandas as pd
import numpy as np
import underthesea

pd.set_option("display.max_colwidth", 90)
pd.set_option("display.width", 160)

print("underthesea:", underthesea.__version__)
print("pandas     :", pd.__version__)
print("numpy      :", np.__version__)
""")

# ---------------------------------------------------------------- PART B ----
md("""
---
# PHẦN B — DỮ LIỆU

Corpus được thu thập bằng `src/crawler.py`, kế thừa kỹ thuật của **Lab 02**
(requests + BeautifulSoup + validation + ghi log lỗi).

So với Lab 02, crawler được mở rộng: crawl nhiều chuyên mục, có delay ngẫu nhiên
giữa các request để không làm phiền server, và có cache theo URL để chạy lại
không tải trùng.
""")

code(r"""
from config import CORPUS_RAW_PATH

df = pd.read_csv(CORPUS_RAW_PATH)
print("Số bài báo:", len(df))
print("Các cột   :", list(df.columns))
df.head(3)[["category", "title", "description"]]
""")

code(r"""
# Phân bố theo chuyên mục
dist = df["category"].value_counts()
print(dist.to_string())
print("\nĐộ dài text (ký tự):")
print(df["text"].str.len().describe().round(0).to_string())
""")

code(r"""
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

dist.sort_values().plot(kind="barh", ax=axes[0], color="#2563eb")
axes[0].set_title("Số bài theo chuyên mục")
axes[0].set_xlabel("số bài")

df["text"].str.len().plot(kind="hist", bins=40, ax=axes[1], color="#0d9488")
axes[1].set_title("Phân bố độ dài bài báo")
axes[1].set_xlabel("số ký tự")

plt.tight_layout()
plt.show()
""")

# ---------------------------------------------------------------- PART C ----
md(r"""
---
# PHẦN C — TIỀN XỬ LÝ (kế thừa Lab 03)

`src/preprocess.py` giữ nguyên tên hàm và cấu trúc `config` của Lab 03:

```python
preprocess_vi(text, config)   # config bật/tắt từng bước
```

## Thứ tự các bước KHÔNG tùy tiện

| Thứ tự | Bước | Vì sao phải đứng ở đây |
|---|---|---|
| 1 | `normalize_basic` | Đưa về NFC trước, nếu không "Hòa" dạng tổ hợp và dạng dựng sẵn thành 2 term khác nhau |
| 2 | `segment_vi` | Phải chạy **trước** lowercase: underthesea dùng chữ hoa để nhận diện tên riêng |
| 3 | `lowercase` | |
| 4 | `remove_numbers` | Phải chạy **trước** bỏ dấu câu, nếu không "3,5%" đã thành "3 5" thì regex số không bắt được |
| 5 | `remove_punctuation` | |
| 6 | `remove_stopwords` | Chạy **cuối cùng**, so khớp trên token đã sạch |
""")

code(r"""
from preprocess import preprocess_vi, normalize_basic, segment_vi, load_stopwords
from config import CONFIG_INTENT, CONFIG_RETRIEVAL

print("Số stopword đã nạp:", len(load_stopwords()))
print()
print("normalize_basic:", repr(normalize_basic("  Sinh viên\n đang   học NLP. ")))
print("segment_vi     :", segment_vi("sinh viên nghiên cứu xử lý ngôn ngữ tự nhiên"))
""")

md(r"""
## So sánh 2 cấu hình (Config A vs Config B)

Đồ án dùng **hai** cấu hình khác nhau cho hai nhiệm vụ khác nhau — đây là một
quyết định thiết kế có chủ đích, không phải quên thống nhất:

| | `CONFIG_INTENT` | `CONFIG_RETRIEVAL` |
|---|---|---|
| Dùng cho | phân loại ý định | truy hồi bài báo |
| `remove_stopwords` | **False** | **True** |
| Lý do | Câu hỏi chỉ 5–10 token. "cho tôi", "là gì", "có không" **chính là** tín hiệu phân biệt intent — bỏ đi thì "bạn là ai" và "tin về ai" giống hệt nhau | Bài báo dài 3000+ ký tự. Stopword chiếm phần lớn token và làm loãng vector, khiến mọi bài trông giống nhau |
""")

code(r"""
EDGE_CASES = [
    "Tôi không thích sản phẩm này.",
    "Công ty ABC đầu tư 10.000 tỷ đồng tại TP.HCM.",
    "Doanh thu tăng 3,5% so với cùng kỳ.",
    "Python, C++ và C# đều được sử dụng trong dự án.",
    "AI/ML đang thay đổi ngành công nghệ.",
]

rows = []
for s in EDGE_CASES:
    rows.append({
        "raw": s,
        "CONFIG_INTENT": preprocess_vi(s, CONFIG_INTENT),
        "CONFIG_RETRIEVAL": preprocess_vi(s, CONFIG_RETRIEVAL),
    })
pd.DataFrame(rows)
""")

md(r"""
### ✍️ Nhận xét về edge case

1. **`remove_punctuation` baseline của Lab 03 quá aggressive.** Regex `[^\w\s]`
   biến `TP.HCM` → `TP HCM`, `3,5%` → `3 5`, `C++` → `C`. Cả ba đều **mất tín
   hiệu**: tên riêng bị tách, số liệu bị phá, tên công nghệ biến mất.
   → Đồ án sửa lại: chỉ bỏ dấu câu ở **rìa** token, giữ dấu nằm **giữa** chữ/số.

2. **`C#` vẫn bị hỏng** (thành `c_#`) vì `word_tokenize` nối `#` vào token trước.
   Đây là giới hạn của bộ tách từ, không sửa được ở tầng regex.

3. **`AI` bị stopword list nuốt mất** trong `CONFIG_RETRIEVAL`. Lý do: trong
   tiếng Việt "ai" là đại từ nghi vấn ("ai đến đó?") nên nằm trong stopword list.
   Đây là **xung đột thật giữa từ viết tắt tiếng Anh và từ thường tiếng Việt** —
   một hạn chế được ghi nhận, chưa xử lý trong phạm vi đồ án.
""")

# ---------------------------------------------------------------- PART D ----
md(r"""
---
# PHẦN D — TF-IDF TỰ CÀI ĐẶT (phần lõi "from scratch")

`src/vectorizer.py` cài đặt lại **toàn bộ** công thức bằng NumPy + SciPy,
không gọi `sklearn.feature_extraction`.

$$\text{tf}(t,d) = \text{số lần } t \text{ xuất hiện trong } d
\qquad\text{hoặc}\qquad 1 + \log(\text{tf}) \;\;(\text{sublinear})$$

$$\text{idf}(t) = \ln\!\left(\frac{1+n}{1+\text{df}(t)}\right) + 1$$

$$\text{tfidf}(t,d) = \text{tf}(t,d)\cdot\text{idf}(t)
\qquad
\vec{v}_d = \frac{\vec{v}_d}{\lVert \vec{v}_d \rVert_2}$$

$$\cos(\vec a, \vec b) = \frac{\vec a \cdot \vec b}{\lVert\vec a\rVert\,\lVert\vec b\rVert}
\;\;\xrightarrow{\text{sau khi chuẩn hóa L2}}\;\; \vec a \cdot \vec b$$

**Vì sao có hai số "+1":**
- `(1 + df)` ở mẫu: tránh chia cho 0 với term chỉ xuất hiện ở query, chưa từng
  thấy lúc fit.
- `+1` ở cuối: đảm bảo `idf > 0`, nên term xuất hiện ở **mọi** document vẫn còn
  trọng số nhỏ thay vì bị triệt tiêu hoàn toàn.
""")

code(r"""
from vectorizer import TfidfVectorizer, CountVectorizer, make_ngrams, cosine_similarity

# n-gram: bigram giữ lại một phần thông tin thứ tự từ mà BoW đánh mất
toks = ["xử_lý", "ngôn_ngữ", "tự_nhiên"]
print("unigram      :", make_ngrams(toks, (1, 1)))
print("unigram+bigram:", make_ngrams(toks, (1, 2)))
""")

code(r"""
demo_docs = [
    "xử_lý ngôn_ngữ tự_nhiên là nhánh của trí_tuệ nhân_tạo",
    "ngôn_ngữ lập_trình python phổ_biến trong trí_tuệ nhân_tạo",
    "python dùng nhiều trong xử_lý dữ_liệu và học_máy",
    "báo_chí việt_nam đưa tin về du_lịch và kinh_tế",
]
docs_tok = [d.split() for d in demo_docs]

vec = TfidfVectorizer(ngram_range=(1, 1))
X = vec.fit_transform(docs_tok)

# Bảng IDF: term càng hiếm -> idf càng cao -> càng "đặc trưng"
idf_table = (pd.DataFrame({"term": vec.feature_names_, "df": vec.counter.document_frequency_,
                           "idf": vec.idf_})
             .sort_values("idf", ascending=False))
print("Term ĐẶC TRƯNG nhất (df thấp, idf cao):")
print(idf_table.head(5).to_string(index=False))
print("\nTerm PHỔ BIẾN nhất (df cao, idf thấp):")
print(idf_table.tail(5).to_string(index=False))
""")

code(r"""
# Cosine similarity giữa query và từng document
query = "python trí_tuệ nhân_tạo"
q = vec.transform([query.split()])
sims = cosine_similarity(q, X)[0]

pd.DataFrame({"document": demo_docs, "cosine": sims.round(4)}).sort_values("cosine", ascending=False)
""")

md(r"""
## ✅ Kiểm chứng: bản tự cài đặt có ĐÚNG không?

Viết được code chưa có nghĩa là công thức đúng. `tests/test_vectorizer.py` đối
chiếu bản tự viết với `scikit-learn` trên 5 cấu hình khác nhau
(unigram/bigram, sublinear on/off, min_df, max_df), so sánh **vocabulary, mảng
idf, ma trận TF-IDF, vector query và cosine similarity**.

> `sklearn` **chỉ** đóng vai trò tham chiếu trong test. Runtime của chatbot
> không import nó ở bất kỳ đâu.
""")

code(r"""
import subprocess
result = subprocess.run(
    [sys.executable, str(PROJECT_DIR / "tests" / "test_vectorizer.py")],
    capture_output=True, text=True, encoding="utf-8",
)
print(result.stdout[-2200:])
""")

# ---------------------------------------------------------------- PART E ----
md(r"""
---
# PHẦN E — PHÂN LOẠI Ý ĐỊNH (Intent Classification)

## Multinomial Naive Bayes tự cài đặt

$$P(c) = \frac{\text{số câu thuộc lớp } c}{\text{tổng số câu}}
\qquad
P(t \mid c) = \frac{\text{count}(t,c) + \alpha}{\sum_{t'}\text{count}(t',c) + \alpha|V|}$$

$$\hat{c} = \arg\max_c \Big[\log P(c) + \sum_t \text{tf}(t,d)\,\log P(t\mid c)\Big]$$

**Vì sao tính trên miền log:** nhân hàng trăm xác suất nhỏ sẽ tràn số về 0.

**Vì sao cần làm mịn Laplace ($\alpha$):** nếu một term chưa từng xuất hiện
trong lớp $c$ thì $P(t\mid c)=0$, và **toàn bộ tích bị triệt tiêu** dù mọi term
khác khớp hoàn hảo.
""")

code(r"""
from intent_classifier import IntentClassifier

clf = IntentClassifier().train_from_file()
print("Số intent      :", len(clf.model.classes_))
print("Kích thước vocab:", len(clf.vectorizer.vocabulary_))
print()
for tag in clf.model.classes_:
    intent = clf.intents[tag]
    print(f"  {tag:<22} action={intent['action']:<16} {len(intent['patterns'])} pattern")
""")

md(r"""
## Vấn đề đo được: Naive Bayes cho câu ngắn có xác suất rất PHẲNG

Khi test bản NB thuần, các câu ngắn nhưng ý định hiển nhiên lại cho độ tin cậy
rất thấp — thấp hơn cả ngưỡng chấp nhận:

| Câu | NB thuần | Cosine tới pattern gần nhất |
|---|---|---|
| "cảm ơn nhé" | **0.189** | 0.657 |
| "bye" | **0.277** | 1.000 |
| "cho mình link" | 0.381 | 1.000 |
| "thời tiết sao hỏa hôm nay" (ngoài phạm vi) | 0.176 | 0.478 |

Nguyên nhân: softmax trên 14 lớp với vector đã chuẩn hóa L2 cho chênh lệch
log-likelihood rất nhỏ khi câu chỉ có 2–3 token.

**Giải pháp — ensemble hai tín hiệu bù trừ nhau:**

$$\text{score}(c) = w \cdot P_{NB}(c) + (1-w)\cdot \max_{p \in c}\cos(\vec q, \vec p)$$

- NB: tổng hợp bằng chứng từ **mọi** term, có cơ sở xác suất — nhưng phẳng với câu ngắn.
- Cosine: trả lời "đã từng thấy câu nào giống thế này chưa?" — rất nhạy với câu
  ngắn, nhưng chỉ nhìn **một** pattern nên dễ bị từ chung chung đánh lừa.

> **Kết quả dò trên DEV lật lại lựa chọn này:** trọng số tốt nhất là $w = 1.0$,
> tức **Naive Bayes thuần** — trên 55 câu dev, ensemble không còn thắng (ô đánh
> giá bên dưới, pha 1.1). Tín hiệu cosine vẫn nằm trong mã để thí nghiệm tái lập
> được, nhưng hiện không đóng góp vào điểm. Hệ quả: câu rất ngắn như
> "thanks nhé" lại rơi xuống dưới ngưỡng trên tập test (Phần I, Phần J case 5).
""")

code(r"""
tests = ["cảm ơn nhé", "bye", "xin chào bạn", "cho mình link",
         "có bao nhiêu bài báo", "tóm tắt bài đó giúp mình",
         "tin du lịch ninh bình", "thời tiết sao hỏa hôm nay", "asdfgh qwerty"]

rows = []
for t in tests:
    tag, conf = clf.predict(t)
    rows.append({"câu": t, "intent": tag or "(không có)", "độ tin cậy": round(conf, 3)})
pd.DataFrame(rows)
""")

md(r"""
## Dò siêu tham số bằng thực nghiệm

Trọng số `w_nb` và ngưỡng chấp nhận **không chọn bằng cảm tính**. `src/evaluate.py`
quét lưới và tối ưu đồng thời hai mục tiêu ngược chiều:

- **accuracy** — câu thuộc intent cố định phải được nhận đúng nhãn;
- **safety** — câu cần truy hồi / ngoài phạm vi **không được** trả lời bằng một
  câu soạn sẵn sai chỗ.

Tham số được dò trên **tập DEV** (`data/eval/dev.json`), rồi báo cáo **một lần**
trên **tập TEST** tách riêng (`data/eval/test.json`) — câu trong TEST chưa từng
được dùng để dò. Cả hai đều viết riêng, không trùng với patterns dùng để train.
Phương pháp chi tiết ở **Phần I**.

Ô dưới chạy toàn bộ `evaluate.py`: pha 1 dò trên dev, pha 2 báo cáo trên test.
Kết quả được lưu vào `data/eval/test_results.json` và dùng lại ở Phần I.
""")

code(r"""
result = subprocess.run(
    [sys.executable, str(PROJECT_DIR / "src" / "evaluate.py")],
    capture_output=True, text=True, encoding="utf-8", cwd=str(PROJECT_DIR / "src"),
)
print(result.stdout)
""")

# ---------------------------------------------------------------- PART F ----
md(r"""
---
# PHẦN F — TRUY HỒI THÔNG TIN (Retrieval)

## Kiến trúc hai tầng

**Tầng 1 — chọn bài báo.** Index gồm `title` + `description` + `text`.
Title được lặp lại **3 lần** và description **2 lần** khi dựng index. Đây là
cách tăng trọng số trường quan trọng mà không phải sửa công thức TF-IDF: lặp
lại làm tăng `tf` của các term trong title.

**Tầng 2 — chọn câu trả lời.** Trong bài đã chọn, tách câu bằng `sent_tokenize`
(Lab 01) rồi so cosine giữa query và từng câu, lấy 2 câu sát nhất. Nhờ vậy bot
trả lời một đoạn ngắn đúng trọng tâm thay vì ném cả bài 3000 ký tự.
""")

code(r"""
from retriever import NewsRetriever

t0 = time.time()
retriever = NewsRetriever().fit(df.dropna(subset=["title", "text"]).reset_index(drop=True))
print(f"Dựng index xong trong {time.time()-t0:.1f}s")

stats = retriever.stats()
for k, v in stats.items():
    if k != "categories":
        print(f"  {k:<18}: {v}")
""")

code(r"""
for q in ["đảo hải nam miễn visa", "giá iphone mới nhất", "học sinh bị bắt nạt"]:
    print("=" * 78)
    print("QUERY:", q)
    for r in retriever.search(q, top_k=3):
        print(f"  {r.score:.3f} [{r.category}] {r.title}")
        print(f"        term khớp: {[t for t, _ in r.matched_terms[:5]]}")
""")

md(r"""
## Giải thích được — bot không phải hộp đen

Ưu điểm lớn của TF-IDF so với embedding/LLM: **mọi kết quả đều truy vết được**
về term cụ thể và đóng góp bao nhiêu điểm.
""")

code(r"""
info = retriever.explain("đảo hải nam miễn visa 30 ngày")
print("Token          :", info["tokens"])
print("Term có trong vocab:", info["terms_in_vocab"])
print("Term OOV       :", info["oov_terms"])
print()
for r in info["results"]:
    print(f"{r['score']:.4f} [{r['category']}] {r['title'][:58]}")
    for term, val in r["matched_terms"]:
        print(f"        {val:.4f}  {term}")
""")

# ---------------------------------------------------------------- PART G ----
md(r"""
---
# PHẦN G — XỬ LÝ CÂU GÕ KHÔNG DẤU

Người Việt chat rất hay bỏ dấu: *"tin ve dao hai nam"*. Nếu chỉ index bản **có
dấu** thì mọi term của câu đó đều là OOV và bot trượt hoàn toàn.

## Tại sao không thể chỉ "bỏ dấu rồi tách từ"

`word_tokenize` được huấn luyện trên tiếng Việt **có dấu**. Đưa câu không dấu
vào, nó tách sai hoàn toàn — đo được trên chính dự án này:
""")

code(r"""
from preprocess import tokenize, fold_query, fold_tokens, has_diacritics

q_no_accent = "tin ve dao hai nam"
print("Tách từ trên câu KHÔNG DẤU (sai):", tokenize(q_no_accent, CONFIG_RETRIEVAL))
print("  -> dính 've dao' thành một từ, cắt rời 'hai nam'")
print()
print("Tách từ trên câu CÓ DẤU (đúng)  :", tokenize("tin về đảo hải nam", CONFIG_RETRIEVAL))
""")

md(r"""
## Giải pháp: index phụ ở mức ÂM TIẾT

Không cố **khôi phục** dấu (bài toán khó, cần mô hình riêng) mà đi hướng ngược
lại — hạ **cả hai phía** xuống cùng mức âm tiết không dấu:

- **Phía document:** vẫn tách từ trên bản có dấu (chính xác), rồi mới bỏ dấu và
  tách từ ghép ra âm tiết: `["đảo","hải_nam"]` → `["dao","hai","nam"]`.
- **Phía query:** bỏ hẳn bước tách từ, chỉ cắt theo khoảng trắng.
- Bigram trong TF-IDF khôi phục lại phần lớn thông tin từ ghép: `"hai nam"`
  xuất hiện như một bigram.

Hai index được giữ **tách biệt**, chỉ dùng index phụ khi câu hỏi không có dấu.
Nhờ vậy index chính không bị pha loãng và các ngưỡng đã dò trên nó vẫn còn hiệu lực.
""")

code(r"""
print("fold_query('tin ve dao hai nam') :", fold_query("tin ve dao hai nam"))
print("fold_tokens(['đảo','hải_nam'])   :", fold_tokens(["đảo", "hải_nam"]))
print()
print("has_diacritics('tin về đảo') :", has_diacritics("tin về đảo"))
print("has_diacritics('tin ve dao') :", has_diacritics("tin ve dao"))
""")

md(r"""
### Kết quả: thay đổi này còn làm retrieval **tốt lên**

| | Trước | Sau |
|---|---|---|
| Recall@1 | 85.7% | **90.5%** |
| Recall@3 | 95.2% | **100%** |
| MRR | 0.905 | **0.952** |

> ⚠️ Bảng này đo trên tập test **cũ** (21 câu, bị dùng để dò tham số). Kết quả
> đo đúng cách trên tập TEST tách riêng: câu không dấu Recall@1 **95.5%** (21/22)
> — xem Phần I.

Lý do ngoài dự kiến: các truy vấn chứa tên riêng nước ngoài như
*"champions league man utd"* vốn **không có dấu nào**, nên trước đây bị tách từ
sai; nay chúng đi qua index âm tiết và khớp đúng.
""")

# ---------------------------------------------------------------- PART G2 ---
md(r"""
---
# PHẦN G2 — XỬ LÝ TEENCODE

Không dấu mới chỉ là mức lệch chuẩn nhẹ. Người dùng thật còn gõ **teencode**:

```text
"bt gì về vụ iphone k b"   ->  "biết gì về vụ iphone không bạn"
```

Đây là lỗi **nặng hơn hẳn** trường hợp không dấu: `k` và `không` không có ký tự
nào chung, nên không mẹo so khớp bề mặt nào cứu được. Bắt buộc phải có bảng ánh xạ.

## Vì sao HỌC bảng ánh xạ thay vì tự liệt kê

Tự ngồi liệt kê vài trăm từ teencode thì vừa thiếu, vừa mang thiên kiến cá nhân,
và **không đo được**. Thay vào đó ta học từ **ViLexNorm** — 10.467 cặp câu
(teencode → chuẩn) do con người gán nhãn, lấy từ bình luận mạng xã hội thật.

Thuật toán: căn token theo vị trí trên các cặp cùng độ dài (79,5% số cặp), đếm
mọi ánh xạ `a → b`, rồi chỉ giữ lại khi thỏa **cả ba** điều kiện an toàn:

| Điều kiện | Ngưỡng | Vì sao cần |
|---|---|---|
| `a` xuất hiện đủ nhiều | `>= 4` | Tránh học từ nhiễu gán nhãn |
| `a` **thường** bị đổi | `>= 0.5` | **Chốt chặn quan trọng nhất** |
| `b` chiếm ưu thế | `>= 0.5` | Tránh chọn bừa khi `a` mơ hồ |

Thiếu điều kiện thứ hai thì các từ thường như "cả", "mà" sẽ bị thay bừa chỉ vì
đôi khi chúng tình cờ đứng ở vị trí có thay đổi — lỗi âm thầm, rất khó phát hiện.
""")

code(r"""
from normalizer import TeencodeNormalizer, learn_lexicon, load_vilexnorm, evaluate

pairs = load_vilexnorm("train")
print(f"Số cặp câu huấn luyện: {len(pairs):,}")

lexicon = learn_lexicon(pairs)
print(f"Số ánh xạ học được  : {len(lexicon):,}")
print()
for k in ["k", "ko", "t", "đc", "bt", "vs", "lun", "j", "cx", "nhìu"]:
    if k in lexicon:
        print(f"    {k:<8} -> {lexicon[k]}")
""")

code(r"""
normalizer = TeencodeNormalizer(lexicon)

demos = [
    "bt gì về vụ iphone k b",
    "cho t hỏi vụ đảo hải nam vs",
    "ko bt là gì lun",
    "đẹppppp quá trờiii",
]
rows = []
for d in demos:
    rows.append({
        "gõ vào": d,
        "sau chuẩn hóa": normalizer.normalize(d),
        "đã sửa": normalizer.explain(d),
    })
pd.DataFrame(rows)
""")

md(r"""
## Đánh giá trên split test (chưa từng thấy khi học)

**Vì sao báo cáo ERR chứ không chỉ accuracy:** khoảng 84% token vốn đã đúng sẵn,
nên accuracy thô bị thổi phồng. **ERR (Error Reduction Rate)** chỉ đo phần lỗi
thực sự được sửa, và là chỉ số chuẩn của bài toán lexical normalization:

$$\text{ERR} = \frac{\text{acc}_{sau} - \text{acc}_{trước}}{1 - \text{acc}_{trước}}$$
""")

code(r"""
rows = []
for split in ["dev", "test"]:
    m = evaluate(normalizer, split)
    rows.append({
        "split": split,
        "số token": f"{m['n_tokens']:,}",
        "acc trước": f"{m['acc_before']:.2%}",
        "acc sau": f"{m['acc_after']:.2%}",
        "ERR": f"{m['ERR']:.2%}",
        "precision": f"{m['precision']:.2%}",
        "recall": f"{m['recall']:.2%}",
    })
pd.DataFrame(rows)
""")

md(r"""
### ✍️ Nhận xét

**Precision (90.7%) cao hơn hẳn Recall (70.2%)** — đây là đánh đổi có chủ đích.
Ba điều kiện an toàn khiến từ điển thận trọng: nó bỏ sót một số từ teencode hiếm,
nhưng gần như không sửa hỏng từ vốn đã đúng. Với chatbot, sửa hỏng một từ đúng
gây hại nhiều hơn bỏ sót một từ lạ.

**Nhập nhằng được ghi nhận, không giấu:** dữ liệu có mâu thuẫn thật —
`t → tôi` (889 lần) và `t → tao` (132 lần). Ta chọn "tôi" vì chiếm ưu thế, và
**chấp nhận sai** ở những câu vốn nói "tao". Đây là giới hạn cố hữu của chuẩn
hóa ở mức từ đơn lẻ, không xét ngữ cảnh.

**Giới hạn của con số:** ERR chỉ tính trên các cặp căn được theo vị trí (79,5%).
Các cặp mà chuẩn hóa làm thay đổi số token bị loại khỏi cả huấn luyện lẫn đánh
giá, nên con số thật trên toàn bộ dữ liệu sẽ **thấp hơn**.
""")

md(r"""
## Hệ quả không lường trước: chuẩn hóa làm LOÃNG vector truy hồi

Sau khi bật chuẩn hóa, một số câu **vẫn** trượt — dù đã được chuẩn hóa đúng
hoàn toàn. Nguyên nhân: chuẩn hóa **bung** từ viết tắt thành từ đầy đủ, tức là
thêm token khung vào câu. Vector query đã chuẩn hóa L2 nên mỗi token thừa đều
chia bớt trọng số của token quan trọng.
""")

code(r"""
# Đo trực tiếp hiện tượng pha loãng
for q in ["giá iphone", "biết gì về vụ iphone không bạn"]:
    res = retriever.search(q, top_k=1, min_score=0.0)
    print(f"{res[0].score:.3f}   {q!r}")
print("\nNgưỡng chấp nhận = 0.12 -> câu thứ hai TRƯỢT dù cùng ý định.")
""")

md(r"""
**Giải pháp — `QUERY_FRAME_WORDS`:** lọc các từ chỉ đóng vai trò *khung câu hỏi*
(`tin`, `biết`, `vụ`, `xem`, `bạn`...) trước khi dựng vector truy hồi.

Các từ này **không** nằm trong stopword list chuẩn, vì trong văn bản thường
chúng vẫn là từ nội dung ("tin" trong *bản tin*). Nên đây là danh sách riêng,
chỉ áp dụng cho **query**, không áp dụng khi index document.

**Chốt an toàn:** nếu lọc hết sạch thì trả lại nguyên bản — câu như
"có tin gì mới không" toàn từ khung, bỏ hết sẽ thành vector rỗng.
""")

code(r"""
# Chatbot hoàn chỉnh với teencode
from chatbot import build_default_bot

bot_tc = build_default_bot()

for u in ["bt gì về vụ iphone k b", "cho t hỏi vụ đảo hải nam vs",
          "ko bt tin sức khỏe gì lun", "cảm ơn nhìu nha"]:
    bot_tc.reset()
    r = bot_tc.respond(u)
    print("=" * 78)
    print(f"Bạn > {u}")
    if r.normalizations:
        print(f"      chuẩn hóa: {r.normalizations}")
        print(f"      -> {r.normalized_input}")
    print(f"      [route={r.route} | intent={r.intent or '-'} | conf={r.confidence:.2f}]")
    print(f"Bot > {r.text[:260]}")
""")

# ---------------------------------------------------------------- PART G4 ---
md(r"""
---
# PHẦN G4 — THÔNG TIN LỖI THỜI VÀ XẾP HẠNG THEO ĐỘ MỚI

Đây là lỗi **nguy hiểm nhất** với một chatbot tin tức. Trả lời sai chủ đề thì
người dùng nhận ra ngay. Trả lời bằng thông tin **đã lỗi thời**, kèm dẫn nguồn
thật, bằng giọng chắc chắn — thì không ai nhận ra.

## Câu hỏi kiểm tra

> Nếu hôm nay tin nói "vấn đề X là A", hôm sau tin nói "vấn đề X là B",
> bot có trả về đúng thông tin mới không?

Dựng lại đúng tình huống bằng hai bài mâu thuẫn nhau:

| Ngày | Tiêu đề |
|---|---|
| 01/09 | Giá vé tàu Cát Linh **tăng lên 15.000 đồng** từ tháng 10 |
| 10/09 | **Hoãn tăng giá** vé tàu Cát Linh, giữ nguyên 8.000 đồng |
""")

code(r"""
import datetime as _dt
from retriever import NewsRetriever as _NR

day1 = {"url": "https://example.test/ngay-1", "source": "VnExpress", "category": "Công nghệ",
        "title": "Giá vé tàu Cát Linh tăng lên 15.000 đồng từ tháng 10",
        "description": "Tổng công ty đường sắt thông báo giá vé mới áp dụng từ đầu tháng 10.",
        "text": ("Tổng công ty đường sắt Hà Nội cho biết giá vé tàu Cát Linh sẽ tăng lên "
                 "15.000 đồng mỗi lượt kể từ ngày 1 tháng 10. Mức giá cũ là 8.000 đồng."),
        "published_at": "2026-09-01", "crawled_at": "2026-09-01"}
day2 = {"url": "https://example.test/ngay-2", "source": "VnExpress", "category": "Công nghệ",
        "title": "Hoãn tăng giá vé tàu Cát Linh, giữ nguyên 8.000 đồng",
        "description": "Quyết định tăng giá vé bị hoãn vô thời hạn sau phản hồi của hành khách.",
        "text": ("Tổng công ty đường sắt Hà Nội vừa thông báo hoãn kế hoạch tăng giá vé tàu "
                 "Cát Linh. Giá vé giữ nguyên ở mức 8.000 đồng mỗi lượt thay vì tăng lên "
                 "15.000 đồng như thông báo trước đó."),
        "published_at": "2026-09-10", "crawled_at": "2026-09-10"}

df_conflict = pd.concat([df.dropna(subset=["title", "text"]), pd.DataFrame([day1, day2])],
                        ignore_index=True)
QUERY = "giá vé tàu cát linh bao nhiêu"

# alpha = 0 tức là TẮT hoàn toàn yếu tố độ mới -> TF-IDF thuần như ban đầu
r_off = _NR(freshness_alpha=0.0).fit(df_conflict)
print("TF-IDF THUẦN (không xét độ mới):")
for res in r_off.search(QUERY, top_k=5, min_score=0.0):
    if "example.test" in res.url:
        nhan = "NGÀY 1 (cũ, nay SAI)" if "ngay-1" in res.url else "NGÀY 10 (mới, ĐÚNG)"
        print(f"  {res.score:.4f}  [{nhan}]  {res.title}")
""")

md(r"""
Bot trả lời bài **cũ, nay đã sai**.

**Vì sao:** tiêu đề bài cũ chứa đúng các từ trong câu hỏi (`giá vé tàu Cát Linh`),
còn bài mới mở đầu bằng `Hoãn tăng giá`. TF-IDF chỉ đo độ trùng lặp từ ngữ — nó
không biết bài nào mới hơn, không biết bài B phủ định bài A, và không có khái
niệm "thông tin bị thay thế".

Nguyên nhân gốc: `published_at` được crawler thu thập và lưu, nhưng **chưa từng
được dùng** ở bất kỳ đâu trong xếp hạng.

### Tệ hơn: kết quả đảo lộn tùy cách diễn đạt
""")

code(r"""
print("TF-IDF thuần — cùng một ý định, 4 cách hỏi:")
for q in ["giá vé tàu cát linh bao nhiêu", "vé tàu cát linh có tăng giá không",
          "tàu cát linh 15.000 đồng", "hoãn tăng giá vé tàu"]:
    res = [x for x in r_off.search(q, top_k=5, min_score=0.0) if "example.test" in x.url]
    if res:
        nhan = "SAI (tin cũ)" if "ngay-1" in res[0].url else "ĐÚNG (tin mới)"
        print(f"  {nhan:<16} <- {q!r}")
""")

md(r"""
Không phải "thường đúng" — mà là **tùy may rủi** theo cách người dùng gõ.

## Giải pháp

$$\text{score}' = \cos(q, d) \times \big(1 + \alpha \cdot \text{recency}(d)\big)
\qquad
\text{recency} = 0.5^{\,\text{tuổi (ngày)} / \text{nửa chu kỳ}}$$

**Vì sao NHÂN chứ không CỘNG:** nếu cộng, một bài hoàn toàn không liên quan
($\cos \approx 0$) nhưng vừa đăng hôm nay vẫn được cộng một lượng lớn và sẽ nổi
lên đầu với **mọi** câu hỏi. Nhân giữ nguyên tính chất: không liên quan thì vẫn
bằng 0 dù mới tinh.

**Vì sao $\alpha$ phải nhỏ:** mục đích không phải luôn ưu tiên tin mới, mà chỉ
**phá thế hòa** khi hai bài liên quan xấp xỉ nhau.

**Mốc tham chiếu** là ngày đăng mới nhất trong corpus, không phải `datetime.now()`
— để kết quả trong báo cáo tái lập được.
""")

code(r"""
# Dò lưới trên TẬP DEV: đo ĐỒNG THỜI chất lượng truy hồi VÀ ca tin mâu thuẫn.
# Dùng lại đúng các hàm của evaluate.py để số liệu khớp với báo cáo.
import json as _json
from evaluate import ranks_for, mrr as _mrr, set_freshness, conflict_ok
from config import DATA_DIR as _DD

_dev = _json.loads((_DD / "eval" / "dev.json").read_text(encoding="utf-8"))
_case = _json.loads((_DD / "eval" / "conflict_case.json").read_text(encoding="utf-8"))
_df_clean = df.dropna(subset=["title", "text"]).reset_index(drop=True)

_rq = _NR().fit_cached(_df_clean)     # corpus gốc -> đo chất lượng
_rc = _NR().fit(df_conflict)          # corpus + 2 bài mâu thuẫn -> kiểm tra ca mâu thuẫn

print(f"{'half-life':>10} {'alpha':>7} {'Recall@1':>9} {'MRR':>7}   ca tin mâu thuẫn")
print("-" * 56)
for hl, alpha in [(3, 0.0), (30, 0.35), (14, 0.60), (7, 0.60), (3, 0.60)]:
    set_freshness(_rq, hl, alpha)
    set_freshness(_rc, hl, alpha)
    ranks = ranks_for(_rq, _dev["retrieval"])
    r1 = sum(1 for x in ranks if x == 1) / len(ranks)
    verdict = "ĐÚNG" if conflict_ok(_rc, _case) else "SAI"
    mark = "  <- chọn" if (hl, alpha) == (3, 0.60) else ("  (tắt độ mới)" if alpha == 0 else "")
    print(f"{hl:>10} {alpha:>7.2f} {r1:>8.1%} {_mrr(ranks):>7.3f}   {verdict}{mark}")
""")

md(r"""
### Vì sao nửa chu kỳ 30 ngày KHÔNG đủ

Hai bài cách nhau 9 ngày:

$$\text{recency}_{\text{cũ}} = 0.5^{9/30} = 0.81 \Rightarrow \text{hệ số } 1.49
\qquad
\text{recency}_{\text{mới}} = 1.00 \Rightarrow \text{hệ số } 1.60$$

Chênh lệch hệ số chỉ **7%**, trong khi khoảng cách cosine là **~22%** → không đủ
để lật thứ hạng. Nửa chu kỳ ngắn hơn làm chênh lệch này lớn lên; dò trên dev
chọn được **3 ngày**.

### Độ mới là một ĐÁNH ĐỔI, không phải cải tiến miễn phí

Bảng trên cho thấy: trên tập dev 112 câu, độ mới **không cải thiện một cách nhất
quán**. Một vài cấu hình tăng nhẹ MRR (khoảng một câu), phần lớn giảm nhẹ — tất
cả đều trong phạm vi nhiễu. Điều quan trọng: **các cấu hình xử lý đúng ca tin mâu
thuẫn đều thấp hơn một chút** so với khi tắt độ mới (cấu hình chọn: 0.964 so với
0.965). Độ mới được giữ vì đổi một chút chất lượng xếp hạng trung bình lấy việc
**không trả lời bằng tin đã lỗi thời**.

> ⚠️ **Đính chính.** Phiên bản trước của báo cáo ghi "độ mới làm Recall@1 tăng
> 93.5% → 96.8%". Con số đó đo trên tập 31 câu **đã dùng để dò tham số**; trên
> tập dev lớn hơn, mức tăng biến mất. Đó là nhiễu của tập nhỏ.

**Một quyết định thiết kế quan trọng khác:** độ mới chỉ dùng để **xếp hạng**.
Việc **chấp nhận** trả lời ("có đủ căn cứ không?") dựa trên cosine thuần. Nếu
áp ngưỡng lên điểm đã nhân độ mới, bài cũ hơn vài tuần sẽ bị âm thầm loại dù
rất liên quan — lỗi này đã thực sự xảy ra và được kiểm thử hồi quy bắt được
(xem Phần I).
""")

code(r"""
# Sau khi bật độ mới (cấu hình đã chốt trong config.py)
bot_fresh = build_default_bot(corpus_path=None) if False else None
r_on = _NR().fit(df_conflict)   # dùng FRESHNESS_ALPHA / HALFLIFE mặc định
print("CÓ XÉT ĐỘ MỚI:")
for res in r_on.search(QUERY, top_k=5, min_score=0.0):
    if "example.test" in res.url:
        nhan = "NGÀY 1 (cũ, SAI)" if "ngay-1" in res.url else "NGÀY 10 (mới, ĐÚNG)"
        print(f"  {res.score:.4f}  [{nhan}]  {res.title}")

print("\nCả 4 cách hỏi:")
for q in ["giá vé tàu cát linh bao nhiêu", "vé tàu cát linh có tăng giá không",
          "tàu cát linh 15.000 đồng", "hoãn tăng giá vé tàu"]:
    res = [x for x in r_on.search(q, top_k=5, min_score=0.0) if "example.test" in x.url]
    if res:
        nhan = "SAI (tin cũ)" if "ngay-1" in res[0].url else "ĐÚNG (tin mới)"
        print(f"  {nhan:<16} <- {q!r}")
""")

md(r"""
Kể cả câu `"tàu cát linh 15.000 đồng"` — **trích đúng con số của tin cũ** — nay
cũng trả về bài nói con số đó đã bị hoãn.

## Vẫn luôn hiện ngày đăng

Xếp hạng theo độ mới **giảm nhẹ** vấn đề chứ không giải quyết triệt để. Bot vẫn
không hiểu bài B phủ định bài A; nó chỉ ưu tiên bài mới khi hai bài gần ngang
nhau. Nếu bài cũ liên quan **vượt trội**, nó vẫn thắng.

Nên mọi câu trả lời nay đều kèm ngày đăng — bot không biết bài nào đã lỗi thời,
nên ít nhất phải cho người đọc đủ dữ kiện để tự đánh giá.
""")

code(r"""
bot_d = build_default_bot()
r = bot_d.respond("tin về đảo hải nam")
print(r.text[:420])
""")

md(r"""
## Tích lũy hay ghi đè?

| Lớp | Hành vi |
|---|---|
| **Dữ liệu** (`corpus_raw.csv`) | **Tích lũy** — crawler nạp corpus cũ, thêm bài mới, dedup theo URL. Không bao giờ xóa. |
| **Mô hình** (index TF-IDF) | **Ghi đè** — `fit()` tính lại toàn bộ vocabulary và IDF. Không có cập nhật tăng dần. |

Hệ quả: mỗi lần thêm dữ liệu, IDF của **mọi** term đều đổi, nên điểm số của các
bài **đã có sẵn** cũng thay đổi theo. Đây là lý do ngưỡng chấp nhận cần được dò
lại khi corpus lớn lên đáng kể.

## Thời gian huấn luyện (đo thực tế, 381 bài)

| Thành phần | Thời gian |
|---|---|
| Intent classifier (164 pattern) | 0.64s |
| Học từ điển teencode (8.372 cặp) | 0.47s |
| **Dựng index truy hồi** | **20.5s** ← chiếm gần hết |
| Tổng khởi động lần đầu | 21.3s |
| Tổng khởi động (có cache đĩa) | **1.1s** |
| Trả lời một câu hỏi | 24ms |

Xấp xỉ tuyến tính (~54ms/bài): 1.000 bài ≈ 1 phút, 10.000 bài ≈ 9 phút.

Index được cache ra đĩa bằng `joblib` + vân tay SHA-256 của (nội dung corpus +
các tham số ảnh hưởng tới index), tự dựng lại khi corpus đổi.
""")

# ---------------------------------------------------------------- PART G3 ---
md(r"""
---
# PHẦN G3 — THÍ NGHIỆM ĐỐI CHỨNG: TRÍCH XUẤT HAY SINH VĂN BẢN?

Chatbot hiện tại **trích xuất**: chỉ trả về câu đã có sẵn trong corpus. Một câu
hỏi hợp lý: *sao không để mô hình tự SINH câu trả lời cho tự nhiên hơn?*

Phần này trả lời bằng **thực nghiệm**, không bằng lời khẳng định.

## Cần nói rõ trước

TF-IDF + cosine similarity là hàm **đo độ giống nhau**. Nó không có bất kỳ cơ
chế nào để tạo ra từ mới — đây là giới hạn **kiến trúc**, không phải giới hạn
cấu hình. Nên để trả lời câu hỏi trên, phải cài đặt một mô hình sinh thật sự:
**n-gram language model** (`src/generator.py`), vẫn hoàn toàn from scratch.

$$P(w_i \mid w_1...w_{i-1}) \approx P(w_i \mid w_{i-n+1}...w_{i-1})
= \frac{\text{count}(\text{context} + w_i)}{\text{count}(\text{context})}$$

Làm mịn bằng nội suy đệ quy, vì ước lượng thô gán xác suất 0 cho mọi n-gram chưa thấy:

$$P_{interp}(w \mid c) = \lambda P_{ML}(w \mid c) + (1-\lambda) P_{interp}(w \mid c_{[1:]})$$

Đo bằng **perplexity** — số lựa chọn trung bình mô hình còn phân vân mỗi bước:

$$PP = \exp\left(-\frac{1}{N}\sum_i \log P(w_i \mid c_i)\right)$$
""")

code(r"""
from generator import NgramLanguageModel, corpus_to_sentences

sentences = corpus_to_sentences(df.dropna(subset=["text"]))
split = int(len(sentences) * 0.9)
train_s, test_s = sentences[:split], sentences[split:]
print(f"Câu huấn luyện: {len(train_s):,}  |  kiểm thử: {len(test_s):,}")
print(f"Tổng token    : {sum(len(s) for s in train_s):,}")
""")

code(r"""
results = {}
for n in [1, 2, 3, 4]:
    lm = NgramLanguageModel(n=n).fit(train_s)
    pp = lm.perplexity(test_s)
    results[n] = pp
    print("=" * 78)
    print(f"n = {n}   perplexity = {pp:,.1f}")
    print("-" * 78)
    for i in range(2):
        print(f"  [{i+1}] {lm.generate(max_tokens=26, seed_text='du lịch')}")
""")

md(r"""
## Kết quả 1 — Perplexity TĂNG theo bậc n (ngược trực giác)

Trực giác "n lớn hơn thì mô hình mạnh hơn" **không đúng** ở quy mô dữ liệu này.

**Giả thuyết:** dữ liệu quá thưa (~217k token), nên hầu hết 4-gram trong tập
test chưa từng xuất hiện. Khi đó thành phần bậc cao bằng 0, và công thức nội suy
nhân thêm hệ số $(1-\lambda)$ ở **mỗi** lần lùi bậc. Với $\lambda=0.7$ và phải
lùi hai bậc, xác suất bị nhân với $0{,}3 \times 0{,}3 = 0{,}09$ — phạt rất nặng.

**Kiểm chứng bằng cách quét $\lambda$** thay vì chỉ suy đoán:
""")

code(r"""
print(f"{'lambda':>7}" + "".join(f"{'n=' + str(n):>12}" for n in (2, 3, 4)))
print("-" * 43)
for lam in (0.3, 0.5, 0.7, 0.9):
    row = f"{lam:>7.1f}"
    for n in (2, 3, 4):
        lm = NgramLanguageModel(n=n, lambda_=lam).fit(train_s)
        row += f"{lm.perplexity(test_s):>12,.0f}"
    print(row)
""")

md(r"""
Hạ $\lambda$ từ 0,9 xuống 0,3 làm perplexity của n=4 giảm **54 lần**
(63.318 → 1.173). Điều này xác nhận đúng cơ chế đã nêu.

Nhưng ở **mọi** $\lambda$, `n=2` vẫn tốt nhất → với lượng dữ liệu này, bigram là
điểm dừng hợp lý. Muốn dùng bậc cao hơn thì phải đổi sang làm mịn tốt hơn
(Kneser-Ney, backoff Katz), **không phải** chỉ tăng n.

## Kết quả 2 — n càng lớn, "sinh" càng biến thành "chép"

Với n=4, phần lớn ngữ cảnh chỉ xuất hiện **đúng một lần** trong corpus nên chỉ
có duy nhất một từ kế tiếp khả dĩ. Quan sát được ở output phía trên: các mẫu n=4
đều mở đầu bằng cùng một chuỗi dài giống hệt nhau.

Tức là mô hình chép nguyên văn — **không thêm giá trị gì** so với truy hồi, mà
lại **mất khả năng dẫn nguồn**.

## Kết quả 3 — Văn bản sinh ra SAI SỰ THẬT ở mọi bậc n

Trích nguyên văn từ output đã chạy:

> *"du lịch phú quốc còn đang xây dựng các **trung tâm điều trị ebola**"*
>
> *"du lịch phú quốc thành lập năm 2014, **cô đã 12 lần vô địch médoc**"*

Mô hình nối từ theo thống kê, **không có khái niệm về sự kiện**. Với một bot
tin tức, đây là lỗi không thể chấp nhận.

## Kết luận

| Tiêu chí | Trích xuất (đang dùng) | Sinh bằng n-gram |
|---|---|---|
| Mạch lạc | Hoàn hảo (câu do người viết) | Trôi dạt sau 5–8 từ |
| Đúng sự thật | Luôn đúng (chép từ nguồn) | **Bịa sự kiện** |
| Dẫn nguồn được | Có | **Không** |
| Ở n cao | — | Suy biến thành chép |

Ở quy mô dữ liệu này, sinh văn bản bằng n-gram **thua truy hồi trên mọi tiêu chí
quan trọng**. Đây là căn cứ thực nghiệm cho lựa chọn kiến trúc trích xuất, chứ
không phải giả định ban đầu.

**Muốn vừa sinh tự nhiên vừa đúng sự thật** thì cần mô hình ngôn ngữ lớn đã tiền
huấn luyện (PhoGPT, Vistral) kết hợp truy hồi kiểu **RAG**. Kiến trúc hiện tại
đã sẵn sàng cho hướng đó — `NewsRetriever` chính là thành phần "R"; phần còn
thiếu là "G", nằm ngoài phạm vi from scratch của đồ án.
""")


# ---------------------------------------------------------------- PART H ----
md(r"""
---
# PHẦN H — QUẢN LÝ HỘI THOẠI

Đây là thứ tách "chatbot" khỏi "công cụ tìm kiếm". Không có trạng thái thì đoạn
hội thoại sau sẽ hỏng:

```text
User: cho tôi biết về đảo Hải Nam
Bot : [bài báo A]
User: tóm tắt bài đó          <- "bài đó" là bài nào?
User: cho mình link           <- link của bài nào?
```

`DialogueState` giữ lại bài báo vừa nhắc tới và chuyên mục đang quan tâm. Đây là
dạng đơn giản nhất của **giải tham chiếu (anaphora resolution)**: không phân
tích cú pháp, chỉ neo vào lượt gần nhất.
""")

code(r"""
from chatbot import build_default_bot

t0 = time.time()
bot = build_default_bot()
print(f"Nạp bot xong trong {time.time()-t0:.1f}s\n")

conversation = [
    "xin chào",
    "có những chuyên mục nào",
    "cho tôi biết về đảo hải nam",
    "tóm tắt bài đó",          # <- tham chiếu tới bài vừa tìm được
    "cho mình link",           # <- vẫn cùng bài đó
    "tin công nghệ về iphone",
    "thời tiết sao hỏa hôm nay",   # <- ngoài phạm vi, phải từ chối
    "cảm ơn nhé",
]

for user in conversation:
    reply = bot.respond(user)
    print("=" * 78)
    print(f"Bạn > {user}")
    print(f"     [route={reply.route} | intent={reply.intent or '-'} | conf={reply.confidence:.2f}]")
    print(f"Bot > {reply.text[:330]}")
""")

md(r"""
### ✍️ Nhận xét

- `"tóm tắt bài đó"` và `"cho mình link"` được giải đúng về bài Hải Nam vừa tìm
  được — tham chiếu hoạt động.
- `"thời tiết sao hỏa hôm nay"` bị từ chối đúng: điểm cosine cao nhất chỉ đạt
  ~0.058, dưới ngưỡng 0.12.
- `route` cho biết câu trả lời đến từ đâu: `intent` (câu soạn sẵn / action),
  `retrieval` (tìm trong corpus), `fallback` (không đủ căn cứ).
""")

# ---------------------------------------------------------------- PART I ----
md(r"""
---
# PHẦN I — ĐÁNH GIÁ TRUNG THỰC

## Lỗi phương pháp đã mắc — và cách sửa

Ở phiên bản đầu, mọi siêu tham số (`w_nb`, các ngưỡng, độ mới) đều được chọn
bằng cách quét lưới trên tập test, rồi số liệu báo cáo lại đo trên **chính tập
đó**. Đó là **rò rỉ tập test**: con số phản ánh mức "khớp" với những câu đã dùng
để dò, không phải hiệu năng trên câu hỏi chưa thấy. Tập lại quá nhỏ (21–31 câu),
một câu sai đã làm Recall@1 dao động 3,2 điểm.

| | DEV | TEST |
|---|---|---|
| Dùng để | dò tham số | **chỉ báo cáo** |
| Truy hồi | 112 | 122 |
| Ngoài phạm vi | 28 | 24 |
| Intent | 55 | 32 |

- Toàn bộ truy vấn **cũ** (đã từng dùng để dò) bắt buộc vào DEV.
- Nhãn đúng theo **URL bài báo**, không theo "tiêu đề chứa từ X".
- Truy vấn đi qua **đúng đường xử lý của chatbot** (chuẩn hóa teencode...).
- Mọi tỷ lệ kèm **khoảng tin cậy Wilson 95%**.
- Thêm **đánh giá đầu-cuối**: gọi `bot.respond()` như người dùng thật.

**Giới hạn:** người viết truy vấn cũng là người xây bot, nên truy vấn có thể
mang cùng "điểm mù" với bot. Cách tốt nhất vẫn là nhờ người khác viết thêm.
""")

code(r"""
import json
from config import DATA_DIR
from evaluate import wilson

res = json.loads((DATA_DIR / "eval" / "test_results.json").read_text(encoding="utf-8"))

def row(name, k, n):
    lo, hi = wilson(k, n)
    return {"chỉ số": name, "kết quả": f"{k / n:.1%}", "k/n": f"{k}/{n}",
            "KTC 95%": f"{lo:.0%}–{hi:.0%}"}

rows = [
    row("Truy hồi — Recall@1", *res["recall_at_1"]),
    row("Truy hồi — Recall@3", *res["recall_at_3"]),
]
for style, v in res["by_style"].items():
    rows.append(row(f"   Recall@1 — {style}", v["r1"], v["n"]))
rows += [
    row("Chặn câu ngoài phạm vi (truy hồi)", *res["oos_blocked"]),
    row("Intent — accuracy", *res["intent_accuracy"]),
    row("ĐẦU-CUỐI: câu tin tức -> đúng bài", *res["e2e_answer"]),
    row("ĐẦU-CUỐI: ngoài phạm vi -> từ chối", *res["e2e_refuse"]),
]
print(f"MRR truy hồi: {res['mrr']:.3f}    |    Intent macro-F1: {res['intent_macro_f1']:.3f}")
pd.DataFrame(rows)
""")

md(r"""
### Đọc bảng này thế nào

| Trước đây báo cáo (rò rỉ) | Nay (TEST tách riêng) |
|---|---|
| Recall@1 96.8% | **93.4%** |
| MRR 0.984 | **0.950** |
| Intent accuracy 88.5% | **61.5%** |

- **Truy hồi vẫn tốt**, giảm vừa phải.
- **Intent classifier là điểm yếu thật sự**, không phải truy hồi. Các câu sai
  đều cùng một kiểu: câu ngắn, cách nói đời thường ("ừm", "thanks nhé") có độ
  tin cậy **dưới ngưỡng** nên bị từ chối.
- **Teencode yếu nhất trong truy hồi**, nhưng chỉ có 14 câu nên khoảng tin cậy
  rất rộng — chưa kết luận chắc được.
- **Khoảng cách thành phần → đầu-cuối** (93% → 77%) chủ yếu do ngưỡng chấp
  nhận: dò trên dev, ngưỡng 0.13 chặn đúng 100% câu ngoài phạm vi, đổi lại từ
  chối một phần câu trả lời được. Đây là đánh đổi có chủ đích.

## Ngưỡng chấp nhận — dò trên DEV, trên thang cosine thuần
""")

code(r"""
from evaluate import top1_scores, set_freshness as _sf
from config import RETRIEVAL_THRESHOLD, FRESHNESS_ALPHA, FRESHNESS_HALFLIFE_DAYS

_r = _NR().fit_cached(df.dropna(subset=["title", "text"]).reset_index(drop=True))
_sf(_r, FRESHNESS_HALFLIFE_DAYS, FRESHNESS_ALPHA)
_ins = top1_scores(_r, _dev["retrieval"], "query")
in_scores = [x.base_score for x, c in zip(_ins, _dev["retrieval"])
             if x is not None and x.url in set(c["gold_urls"])]
oos_scores = [x.base_score if x else 0.0 for x in top1_scores(_r, _dev["out_of_scope"], "text")]

fig, ax = plt.subplots(figsize=(9, 3.4))
ax.hist(in_scores, bins=20, alpha=.75, label="trong phạm vi (trúng bài)", color="#0d9488")
ax.hist(oos_scores, bins=12, alpha=.75, label="ngoài phạm vi", color="#b45309")
ax.axvline(RETRIEVAL_THRESHOLD, color="#dc2626", ls="--", lw=2,
           label=f"ngưỡng = {RETRIEVAL_THRESHOLD}")
ax.set_xlabel("cosine TF-IDF của kết quả top-1 (tập DEV)"); ax.set_ylabel("số truy vấn")
ax.set_title("Ngưỡng chấp nhận nằm trên điểm cao nhất của câu ngoài phạm vi")
ax.legend(); plt.tight_layout(); plt.show()

print(f"Trong phạm vi: min={min(in_scores):.3f}  trung vị={np.median(in_scores):.3f}")
print(f"Ngoài phạm vi: max={max(oos_scores):.3f}  trung vị={np.median(oos_scores):.3f}")
""")

md(r"""
## Kiểm thử hồi quy bắt được hai lỗi thiết kế

`tests/test_chatbot.py` có 21 kiểm thử, mỗi cái ứng với một lỗi thật từng làm
bot trả lời sai mà **không báo lỗi gì**. Ngay lần chạy đầu sau khi áp tham số
mới dò trên dev, chúng bắt được hai lỗi:

**1. Độ mới âm thầm biến thành bộ lọc loại bài cũ.** Ngưỡng áp lên điểm *đã
nhân* độ mới; với nửa chu kỳ 3 ngày, bài 16 ngày tuổi gần như không được
thưởng nên bị loại dù liên quan. Sửa: **xếp hạng** theo điểm có độ mới,
**chấp nhận** theo cosine thuần.

**2. Cùng một câu, kết quả tùy nhánh.** Ngưỡng nới lỏng khi người dùng nêu
chuyên mục chỉ có ở một nhánh. "tin du lịch ninh bình" có độ tin cậy intent
0.246 — dưới ngưỡng 0.25 một chút — nên rơi sang nhánh kia và thất bại. Sửa:
một hằng số dùng chung cho cả hai nhánh.
""")

code(r"""
r = subprocess.run([sys.executable, str(PROJECT_DIR / "tests" / "test_chatbot.py")],
                   capture_output=True, text=True, encoding="utf-8")
print(r.stdout[-1900:])
""")

md(r"""
---
# PHẦN I2 — BM25 TỰ CÀI ĐẶT: MỘT KẾT QUẢ ÂM

BM25 là bước tiếp theo tự nhiên sau TF-IDF, sửa hai điểm yếu của nó:
**tần suất bão hòa** (lần xuất hiện thứ 20 của một từ gần như không thêm bằng
chứng) và **chuẩn hóa độ dài có tham số**.

$$\text{score}(q,d) = \sum_{t \in q} \text{idf}(t)\,
\frac{\text{tf}(t,d)\,(k_1+1)}{\text{tf}(t,d) + k_1\left(1 - b + b\,\frac{|d|}{\text{avgdl}}\right)}
\qquad
\text{idf}(t) = \ln\!\left(\frac{N - \text{df}(t) + 0.5}{\text{df}(t) + 0.5} + 1\right)$$

sklearn không có BM25, nên bản vector hóa được **đối chiếu với một bản cài đặt
ngây thơ viết thẳng từ công thức** (vòng lặp) trên 4 cặp (k1, b) — khớp tuyệt
đối (xem kết quả `test_vectorizer.py` ở Phần D).

Điểm BM25 không bị chặn trên, nên chỉ dùng để **xếp hạng**; việc **chấp nhận**
vẫn dựa trên cosine TF-IDF.
""")

code(r"""
from evaluate import ranks_for as _rf, mrr as _m, sign_test_p
from config import BM25_K1, BM25_B

_r.freshness_alpha = 0.0   # so CHẤT LƯỢNG XẾP HẠNG thuần, tắt độ mới
print(f"{'k1':>5} {'MRR BM25 (dev, b=0.9)':>24}")
for k1 in [0.6, 1.2, 2.0, 3.0, 5.0, 8.0]:
    _r.set_bm25(k1, 0.9, ranking="bm25")
    print(f"{k1:>5} {_m(_rf(_r, _dev['retrieval'])):>24.3f}")

_r.set_bm25(BM25_K1, BM25_B, ranking="tfidf")
rr_tf = [1 / x if x else 0 for x in _rf(_r, _dev["retrieval"])]
_r.set_bm25(BM25_K1, BM25_B, ranking="bm25")
rr_bm = [1 / x if x else 0 for x in _rf(_r, _dev["retrieval"])]
_r.set_bm25(BM25_K1, BM25_B, ranking="tfidf")

w = sum(a > b for a, b in zip(rr_bm, rr_tf)); l = sum(a < b for a, b in zip(rr_bm, rr_tf))
print(f"\nTF-IDF MRR (dev): {np.mean(rr_tf):.3f}   BM25 tốt nhất (k1={BM25_K1}, b={BM25_B}): {np.mean(rr_bm):.3f}")
print(f"Kiểm định dấu có cặp: BM25 thắng {w} câu, thua {l} câu, hòa {len(rr_tf) - w - l} -> p = {sign_test_p(w, l):.3f}")
print("\nTrên TEST:", {k: f"R@1 {v['r1'][0]}/{v['r1'][1]}, MRR {v['mrr']:.3f}"
                     for k, v in res["ranking_comparison"].items()})
""")

md(r"""
### Vì sao BM25 không hơn — giả thuyết được dữ liệu ủng hộ

Index tăng trọng số tiêu đề bằng cách **lặp tiêu đề 3 lần**. Mẹo này chỉ hiệu
quả khi tf còn tăng theo số lần lặp — mà **độ bão hòa tf**, đặc điểm cốt lõi
của BM25, triệt tiêu đúng hiệu ứng đó.

Nếu giả thuyết đúng, BM25 phải tốt dần khi **giảm bão hòa** (tăng $k_1$). Bảng
trên cho thấy đúng như vậy: MRR tăng đều theo $k_1$. Ở $k_1$ lớn, BM25 gần như
quay lại hành vi của TF-IDF. Cách làm đúng là **BM25F** — bão hòa riêng từng
trường rồi mới cộng có trọng số.

### Quyết định: giữ TF-IDF — và một quy tắc quyết định bị sửa

Quy tắc ban đầu "hơn trên dev là đổi" đã chọn BM25 với chênh **0.002 MRR**
(khoảng một câu), rồi BM25 **thua** trên test. Quy tắc được sửa thành: phương
pháp phức tạp hơn phải thắng **có ý nghĩa thống kê** trên dev (kiểm định dấu
có cặp). Kết quả p = 1.0 → giữ TF-IDF.

> **Minh bạch:** quy tắc kiểm định được thêm **sau khi** đã thấy kết quả test
> BM25. Dưới cả hai quy tắc, kết luận không đổi: hai phương pháp không khác nhau
> có ý nghĩa (khoảng tin cậy chồng lấn gần như hoàn toàn).

### Số lần đã xem tập TEST

| Lần | Lý do | Có đổi gì dựa trên test? |
|---|---|---|
| 1 | Báo cáo đầu tiên sau khi tách dev/test | Không |
| 2 | Sau hai sửa lỗi do **kiểm thử hồi quy** phát hiện | Không |
| 3 | Sau khi thêm BM25 | **Có** — quy tắc chọn phương pháp |

Sau lần 3, tham số chatbot không đổi nữa. Tập test chỉ được dùng thêm cho lớp
RAG (Phần K; docs/06 liệt kê đủ), và mỗi lần chạy lại notebook này thì ô đánh giá
ở Phần E tính lại báo cáo test với **đúng** bộ tham số đã chốt — không quyết định
nào dựa trên các lần tính lại đó. Dù vậy, mọi cải tiến tiếp theo (đặc biệt cho
intent) phải được đo trên một **tập test mới**.
""")

# ---------------------------------------------------------------- PART J ----
md(r"""
---
# PHẦN J — PHÂN TÍCH LỖI (Error Analysis)

Yêu cầu tối thiểu 3 case. Dưới đây là **19 lỗi thật** phát hiện trong quá trình
làm, kèm nguyên nhân và cách xử lý.
""")

code(r"""
error_analysis = pd.DataFrame([
    {
        "STT": 1,
        "Tầng": "Regex",
        "Input": "Doanh thu tăng 3,5%",
        "Sai": "regex quantity KHÔNG bắt được '3,5%'",
        "Nguyên nhân": r"Kết thúc pattern bằng \b, nhưng '%' là ký tự non-word và sau nó là khoảng trắng/hết chuỗi (cũng non-word) -> không tồn tại ranh giới từ",
        "Xử lý": r"Đổi \b thành (?!\w). Đã sửa, có test.",
        "Trạng thái": "ĐÃ SỬA",
    },
    {
        "STT": 2,
        "Tầng": "Word segmentation",
        "Input": "tin ve dao hai nam (không dấu)",
        "Sai": "tách thành ['ve_dao', 'hai', 'nam']",
        "Nguyên nhân": "word_tokenize huấn luyện trên tiếng Việt có dấu; text không dấu nằm ngoài phân bố huấn luyện",
        "Xử lý": "Dựng index phụ ở mức âm tiết, bỏ qua tách từ khi câu không có dấu",
        "Trạng thái": "ĐÃ SỬA",
    },
    {
        "STT": 3,
        "Tầng": "NER",
        "Input": "Giám đốc Nguyễn Văn An của Công ty ABC, sđt 0905123456",
        "Sai": "'Công ty ABC' gán LOC (đúng ra ORG); 'sđt 0905123456' cũng bị gán LOC",
        "Nguyên nhân": "Model NER không chắc chắn với tên công ty viết tắt và chuỗi số lạ",
        "Xử lý": "Số điện thoại đã được Regex bắt song song -> không phụ thuộc NER cho định dạng cố định",
        "Trạng thái": "GIẢM NHẸ",
    },
    {
        "STT": 4,
        "Tầng": "Stopwords",
        "Input": "AI/ML đang thay đổi ngành công nghệ",
        "Sai": "'AI' bị xóa khỏi vector truy hồi",
        "Nguyên nhân": "'ai' là đại từ nghi vấn tiếng Việt nên nằm trong stopword list; trùng với viết tắt tiếng Anh 'AI'",
        "Xử lý": "Chưa xử lý. Cần stopword list phân biệt chữ hoa hoặc whitelist thuật ngữ",
        "Trạng thái": "TỒN TẠI",
    },
    {
        "STT": 5,
        "Tầng": "Intent classifier",
        "Input": "cảm ơn nhé",
        "Sai": "NB thuần chỉ cho 0.189 -> rơi xuống dưới ngưỡng",
        "Nguyên nhân": "Câu quá ngắn; softmax trên 14 lớp với vector chuẩn hóa L2 cho phân phối phẳng",
        "Xử lý": "Ensemble NB + cosine tới pattern gần nhất -> 0.470. Nhưng khi dò lại trên DEV, w_nb=1.0 (NB thuần) thắng nên ensemble bị tắt; trên TEST 'thanks nhé' chỉ còn 0.18 và bị từ chối",
        "Trạng thái": "TỒN TẠI (sửa rồi mất khi dò lại)",
    },
    {
        "STT": 6,
        "Tầng": "Retrieval",
        "Input": "tin du lịch ninh bình",
        "Sai": "Bài đúng chỉ đạt 0.097, dưới ngưỡng toàn cục 0.12",
        "Nguyên nhân": "'du_lịch' có idf thấp (xuất hiện ở 72 bài); query chỉ 2 token nội dung; bài dài làm loãng vector",
        "Xử lý": "Khi người dùng đã nêu chuyên mục, tập ứng viên co lại -> dùng ngưỡng thấp hơn (0.6 x ngưỡng gốc)",
        "Trạng thái": "ĐÃ SỬA",
    },
    {
        "STT": 7,
        "Tầng": "Intent classifier",
        "Input": "bạn hỗ trợ được những gì",
        "Sai": "Dự đoán liet_ke_chuyen_muc thay vì huong_dan",
        "Nguyên nhân": "Hai intent này chồng lấn về ngữ nghĩa; cả hai đều hỏi 'bot có gì'",
        "Xử lý": "Chưa xử lý. Cần thêm pattern phân biệt hoặc gộp hai intent",
        "Trạng thái": "TỒN TẠI",
    },
    {
        "STT": 8,
        "Tầng": "Truy hồi / teencode",
        "Input": "bt gì về vụ iphone k b",
        "Sai": "Đã chuẩn hóa ĐÚNG thành 'biết gì về vụ iphone không bạn' nhưng vẫn fallback",
        "Nguyên nhân": "Chuẩn hóa BUNG từ viết tắt thành từ đầy đủ -> thêm token khung vào câu. Vector query chuẩn hóa L2 nên mỗi token thừa chia bớt trọng số token quan trọng: 'giá iphone' đạt 0.167 nhưng 'biết gì về vụ iphone không bạn' chỉ 0.100",
        "Xử lý": "Thêm QUERY_FRAME_WORDS, lọc từ khung câu hỏi trước khi dựng vector truy hồi",
        "Trạng thái": "ĐÃ SỬA",
    },
    {
        "STT": 9,
        "Tầng": "Kiến trúc / trùng lặp định nghĩa",
        "Input": "không biết tin sức khỏe gì luôn",
        "Sai": "Bị coi là có chủ đề cụ thể rồi đem đi tìm kiếm và trượt",
        "Nguyên nhân": "chatbot.py và retriever.py giữ HAI danh sách từ khung riêng, và chúng đã lệch nhau: 'biết' có ở bên retriever nhưng thiếu ở chatbot",
        "Xử lý": "Gộp về một nguồn duy nhất trong config.py. Bài học: một khái niệm không được có hai định nghĩa ở hai nơi",
        "Trạng thái": "ĐÃ SỬA",
    },
    {
        "STT": 10,
        "Tầng": "Word segmentation",
        "Input": "so khớp tên chuyên mục 'Sức khỏe'",
        "Sai": "word_tokenize('Sức khỏe') -> ['sức','khỏe'] nhưng word_tokenize('...tin sức khỏe gì luôn') -> ['sức_khỏe']",
        "Nguyên nhân": "Bộ tách từ cho kết quả KHÁC NHAU cho cùng một cụm tùy ngữ cảnh xung quanh, nên phép so 'sức_khỏe' thuộc {'sức','khỏe'} luôn sai",
        "Xử lý": "Chuyển sang so khớp ở mức ÂM TIẾT (tách theo '_') thay vì so nguyên token",
        "Trạng thái": "ĐÃ SỬA",
    },
    {
        "STT": 11,
        "Tầng": "Mô hình sinh (n-gram LM)",
        "Input": "sinh câu với n = 4",
        "Sai": "Perplexity TĂNG theo bậc n (819 -> 1.951 -> 5.911), ngược trực giác",
        "Nguyên nhân": "Dữ liệu thưa: hầu hết 4-gram ở tập test chưa từng thấy nên thành phần bậc cao = 0, công thức nội suy nhân thêm (1-lambda) ở MỖI lần lùi bậc -> phạt 0,09 lần",
        "Xử lý": "Đã kiểm chứng bằng cách quét lambda: hạ 0,9 -> 0,3 làm perplexity n=4 giảm 54 lần. Kết luận: với dữ liệu này bigram là điểm dừng; muốn n cao hơn phải đổi sang Kneser-Ney",
        "Trạng thái": "ĐÃ GIẢI THÍCH",
    },
    {
        "STT": 13,
        "Tầng": "Xếp hạng / độ mới",
        "Input": "giá vé tàu cát linh bao nhiêu (2 bài mâu thuẫn, cách nhau 9 ngày)",
        "Sai": "Trả về bài CŨ đã lỗi thời (0.4956) thay vì bài mới đúng (0.4098), kèm dẫn nguồn thật",
        "Nguyên nhân": "published_at được crawl và lưu nhưng KHÔNG dùng khi xếp hạng. TF-IDF chỉ đo trùng lặp từ ngữ; tiêu đề bài cũ chứa đúng từ trong câu hỏi",
        "Xử lý": "score' = cosine x (1 + alpha x recency), dò được alpha=0.6 / nửa chu kỳ 7 ngày (dò lại trên dev: 3 ngày). Nhân chứ không cộng để bài không liên quan vẫn ở 0",
        "Trạng thái": "ĐÃ SỬA",
    },
    {
        "STT": 14,
        "Tầng": "Phương pháp đánh giá",
        "Input": "tin ve dao hai nam (không dấu)",
        "Sai": "Đạt 0.168, trượt ngưỡng 0.18 vừa dò được, dù trước đó vẫn trả lời tốt",
        "Nguyên nhân": "Tập test dò ngưỡng CHỈ có câu viết chuẩn có dấu, trong khi bot đã hỗ trợ cả không dấu và teencode -> ngưỡng được dò trên phân bố sai",
        "Xử lý": "Bổ sung 10 truy vấn không dấu/teencode + 4 câu ngoài phạm vi, làm mịn lưới quét quanh vùng ranh giới, dò lại: 0.155",
        "Trạng thái": "ĐÃ SỬA",
    },
    {
        "STT": 15,
        "Tầng": "Chuẩn hóa / định tuyến",
        "Input": "thoi tiet sao hoa hom nay",
        "Sai": "Bot trả lời 'Tạm biệt bạn! Hẹn gặp lại' cho một câu hỏi về thời tiết",
        "Nguyên nhân": "Từ điển teencode luôn trả từ CÓ DẤU. Sửa 'thoi'->'thôi' làm cả câu bỗng 'có dấu' -> định tuyến sang index có dấu, nơi 5 token còn lại đều OOV -> cả câu bị phán đoán dựa trên đúng một từ ('thôi' thuộc pattern 'thôi nhé')",
        "Xử lý": "Nếu câu gốc không có dấu thì bỏ dấu lại sau chuẩn hóa, giữ nguyên hệ quy chiếu dấu người dùng đang gõ",
        "Trạng thái": "ĐÃ SỬA",
    },
    {
        "STT": 16,
        "Tầng": "Phương pháp đánh giá",
        "Input": "toàn bộ quy trình dò tham số",
        "Sai": "Báo cáo Recall@1 96.8%, intent 88.5% — đo trên chính tập đã dùng để dò tham số",
        "Nguyên nhân": "Rò rỉ tập test: không tách tập dò tham số (dev) khỏi tập báo cáo (test); tập lại quá nhỏ (21-31 câu)",
        "Xử lý": "Tách DEV 112 / TEST 122 câu, nhãn theo URL, KTC Wilson 95%. Số thật: Recall@1 93.4%, intent 61.5%",
        "Trạng thái": "ĐÃ SỬA",
    },
    {
        "STT": 17,
        "Tầng": "Xếp hạng / ngưỡng",
        "Input": "tin ve dao hai nam (sau khi dò lại nửa chu kỳ = 3 ngày)",
        "Sai": "Từ trả lời đúng thành 'không tìm thấy'",
        "Nguyên nhân": "Ngưỡng chấp nhận áp lên điểm ĐÃ NHÂN độ mới -> bài 16 ngày tuổi gần như không được thưởng nên bị loại. Độ mới âm thầm thành bộ lọc",
        "Xử lý": "Xếp hạng theo điểm có độ mới, CHẤP NHẬN theo cosine thuần. Kiểm thử hồi quy bắt được",
        "Trạng thái": "ĐÃ SỬA",
    },
    {
        "STT": 18,
        "Tầng": "Định tuyến",
        "Input": "tin du lịch ninh bình",
        "Sai": "Thất bại khi độ tin cậy intent là 0.246 nhưng thành công khi 0.27",
        "Nguyên nhân": "Ngưỡng nới lỏng khi đã nêu chuyên mục chỉ có ở nhánh duyệt mục, không có ở nhánh truy hồi",
        "Xử lý": "Một hằng số CATEGORY_SCOPED_THRESHOLD_FACTOR dùng cho cả hai nhánh",
        "Trạng thái": "ĐÃ SỬA",
    },
    {
        "STT": 19,
        "Tầng": "Quy tắc chọn mô hình",
        "Input": "TF-IDF vs BM25",
        "Sai": "Chọn BM25 vì hơn 0.002 MRR trên dev, rồi BM25 thua trên test",
        "Nguyên nhân": "Quy tắc 'hơn là đổi' bỏ qua nhiễu: chênh 0.002 nhỏ hơn rất nhiều so với độ rộng khoảng tin cậy",
        "Xử lý": "Yêu cầu thắng có ý nghĩa thống kê (kiểm định dấu có cặp): p = 1.0 -> giữ TF-IDF. Quy tắc thêm SAU khi thấy test — ghi rõ",
        "Trạng thái": "ĐÃ SỬA",
    },
    {
        "STT": 12,
        "Tầng": "Chuẩn hóa teencode",
        "Input": "token 't'",
        "Sai": "Luôn chuẩn hóa thành 'tôi', kể cả khi câu vốn nói 'tao'",
        "Nguyên nhân": "Dữ liệu có mâu thuẫn thật: t->tôi (889 lần) vs t->tao (132 lần). Chuẩn hóa ở mức từ đơn lẻ, không xét ngữ cảnh",
        "Xử lý": "Chấp nhận, chọn đích chiếm ưu thế. Muốn đúng phải dùng mô hình seq2seq có ngữ cảnh - ngoài phạm vi đồ án",
        "Trạng thái": "TỒN TẠI",
    },
])
error_analysis
""")

md(r"""
## Một giả thuyết bị dữ liệu BÁC BỎ

Khi gặp lỗi số 6, giả thuyết ban đầu là: *"thay ngưỡng tuyệt đối bằng ngưỡng
biên (margin) — chấp nhận khi top-1 vượt trội hẳn so với top-2"*.

Đã đo tỷ lệ `top1/top2` trên toàn bộ tập test trước khi cài đặt:

| Nhóm | khoảng tỷ lệ |
|---|---|
| Trong phạm vi (trúng bài) | 1.03× – 8.62× |
| Ngoài phạm vi | 1.04× – 1.59× |

Hai khoảng **chồng lấn nặng** → margin **không** phân tách được. Giả thuyết bị
bác bỏ, và ngưỡng tuyệt đối 0.12 được giữ nguyên vì nó tách sạch (0.118 vs 0.108).

> Bài học: đo trước khi sửa. Một ý tưởng nghe hợp lý vẫn có thể sai với dữ liệu thật.
""")

# ---------------------------------------------------------------- PART K ----
md(r"""
---
# PHẦN K — RAG VỚI PhoGPT: THÍ NGHIỆM MỞ RỘNG (NGOÀI PHẠM VI *FROM SCRATCH*)

Phần G3 đã chứng minh: tự sinh văn bản bằng mô hình n-gram tự cài đặt thì **bịa
sự kiện**. Nhưng hạn chế "bot chỉ chép nguyên văn, không diễn đạt lại" vẫn còn.
Cách chuẩn của ngành để vừa **tự nhiên** vừa **bám nguồn** là **RAG**:

```text
câu hỏi ─> [truy hồi TF-IDF tự cài đặt] ─> 1-3 bài ─> [chốt chặn] ─> [PhoGPT-4B-Chat] ─> [chốt chặn]
```

Đây là phần **duy nhất** của đồ án dùng mô hình tiền huấn luyện, nên nó là **lớp
tùy chọn, mặc định tắt**: tắt đi thì chatbot chạy y như cũ. Máy cá nhân không có
GPU NVIDIA nên PhoGPT chạy trên Google Colab (T4), qua `notebooks/RAG_PhoGPT_Colab.ipynb`.

## Hai chốt chặn tất định — hàng rào cho LLM

Cả hai đều là **quy tắc** (regex + so khớp số theo giá trị, tinh thần Lab 01), nên
kiểm thử được không cần GPU (`tests/test_rag.py`, 29 test):

| Chốt chặn | Khi nào | Làm gì |
|---|---|---|
| **Giả định** (trước LLM) | câu hỏi nêu con số/mã hiệu mà **không bài nào** nhắc: "từ năm 2015", "chuẩn IP68" | không gọi LLM; trả lời "các bài báo không nhắc tới ..." + đoạn trích |
| **Số bịa** (sau LLM) | câu sinh chứa số không có trong nguồn | hiển thị đoạn trích gốc thay cho câu sinh |
| **Lặp lại** (sau LLM) | câu sinh không thêm gì ngoài chính câu hỏi | hiển thị đoạn trích gốc |

## Ba lần chạy thật (chi tiết: `docs/07-rag-phogpt.md`)

| Lần | Tập | Mô hình | Mục đích |
|---|---|---|---|
| 1 | dev | GGUF Q4_K_M | prompt v1 (6 quy tắc đánh số) — kết quả **tệ**, dùng để tìm dạng lỗi |
| 2 | dev | GGUF Q8_0 | so v1 với **v2** (prompt ngắn, câu hỏi đặt cuối, hậu xử lý, chốt chặn) |
| 3 | **test** | GGUF Q8_0 | đo **một lần** trên câu hỏi chưa từng thấy, bẫy mới chốt trước khi chạy |
""")

code(r"""
import json
import pandas as pd

RAG_DIR = DATA_DIR / "eval" / "rag"
run3 = json.loads((RAG_DIR / "run3_test_results.json").read_text(encoding="utf-8"))
ann = pd.read_csv(RAG_DIR / "run3_annotation.csv")

NHAN = {"A": "A · đúng, trả lời được", "B": "B · đúng, trình bày hỏng",
        "C": "C · không trả lời (lặp câu hỏi)", "D": "D · CÓ THÔNG TIN SAI",
        "E": "E · từ chối sai"}
bang = pd.DataFrame({
    "prompt v1": ann["nhãn v1"].value_counts(),
    "prompt v2": ann["nhãn v2"].value_counts(),
}).reindex(list(NHAN)).fillna(0).astype(int)
bang.index = [NHAN[k] for k in bang.index]
print(f"Chấm tay {len(ann)} câu tin tức trên tập TEST (câu PhoGPT sinh, TRƯỚC chốt chặn)")
display(bang)

thang = ann[(ann["nhãn v2"] == "A") & (ann["nhãn v1"] != "A")]
thua = ann[(ann["nhãn v1"] == "A") & (ann["nhãn v2"] != "A")]
print(f"So có cặp trên nhãn A — v2 thắng {len(thang)}, thua {len(thua)}")
print(f"kiểm định dấu p = {sign_test_p(len(thang), len(thua)):.4f}")
""")

md(r"""
**Đọc bảng trên cho đúng:**

- Prompt v2 tăng mạnh số câu **dùng được** (9 → 19 trên 33) và xóa sạch **từ chối
  sai**. Đây là tập test, chưa từng dùng để chỉnh prompt, nên kết luận này đứng vững.
- Nhưng **số câu chứa thông tin sai KHÔNG giảm** (3 → 4). RAG chỉ **đổi kiểu sai**:
  v1 sai lộ liễu (bịa ngày tháng, chép lại quy tắc trong prompt), v2 sai trôi chảy
  nên **khó phát hiện hơn** — ví dụ "Messi được đề cử **vì** anh không có tên năm
  2024 và 2025" (đảo nhân quả), hay gán kỷ lục nhiệt độ 56,7 °C cho một ca tử vong
  xảy ra ở 46,7 °C.
""")

code(r"""
traps = pd.read_csv(RAG_DIR / "run3_trap_annotation.csv")
print("11 CÂU BẪY MỚI (chốt trước khi chạy) — kết quả theo LOẠI bẫy")
display(traps[["bẫy", "loại", "v1", "v2"]])
""")

md(r"""
Bảng bẫy là kết quả **quan trọng nhất** của phần này:

- v1 và v2 **hòa nhau**: 6 an toàn / 5 sai — nhưng vì lý do **trái ngược**.
- v2 an toàn **nhờ chốt chặn** ở những bẫy **có con số**; còn với giả định sai
  **không có số**, v2 sai **4/4**: nó khẳng định thẳng điều bài báo phủ định
  ("robot Optimus của Tesla tự bước ra khỏi dây chuyền" — bài nói Tesla **chưa**
  làm được). Ở đúng những câu đó, v1 lại an toàn vì nó **từ chối**.
- Nói cách khác: prompt v2 làm mô hình **quả quyết hơn** → trả lời tốt hơn nhiều
  với câu hỏi thật, nhưng cũng **gật đầu với giả định sai** dễ hơn.
- Một bẫy được thiết kế nhắm **điểm mù đã biết** của chốt chặn ("đập cao **500 m**"
  trong khi bài có "500 MW") — và nó lọt đúng như dự đoán.

Ngoài ra, câu "iPhone 18 Pro Max có mấy màu" cho thấy lỗi **không chỉ ở LLM**: tầng
chọn câu chỉ đưa 4 câu vào ngữ cảnh, câu liệt kê màu không nằm trong đó, nên **cả
hai** prompt đều bịa màu. Truy hồi đúng bài vẫn chưa đủ — phải đưa đúng *câu*.
""")

code(r"""
rows = pd.DataFrame(run3["rows"])
v2 = rows[(rows["prompt"] == "v2") & rows["route"].str.startswith("rag")]

for q in ["vì sao đêm đầu ngủ ở khách sạn hay bị mất ngủ", "giá vé tàu cát linh bao nhiêu",
          "robot Optimus của Tesla tự bước ra khỏi dây chuyền"]:
    r = v2[v2["câu hỏi"] == q].iloc[0]
    print("=" * 100)
    print("HỎI:", q, f"   [chốt chặn: {r['chốt chặn'] or 'không'}]")
    print("\nTRÍCH XUẤT (bot nộp bài):\n ", str(r["trích xuất"])[:260].replace("\n", " "))
    print("\nRAG (PhoGPT v2):\n ", str(r["câu LLM"])[:260].replace("\n", " "))
    if r["ghi chú"]:
        print("\n  ⚠️ ghi chú bẫy:", r["ghi chú"])
""")

md(r"""
## Kết luận phần K

| | Bot trích xuất (bài nộp) | RAG + PhoGPT-4B |
|---|---|---|
| Câu trả lời tự nhiên | không — chép nguyên câu trong bài | **có** |
| Câu dùng được (test, 33 câu) | — (luôn là trích dẫn thật) | 19/33 |
| Khẳng định điều bài báo **không** nói | **không bao giờ** | có, 4/33 câu + 5/11 bẫy |
| Cần GPU | không | có |
| Giải thích được vì sao trả lời vậy | có (`--explain`) | không |

**RAG làm câu trả lời tự nhiên hơn hẳn, nhưng không trung thực hơn.** Vì vậy đồ án
giữ RAG ở đúng vị trí: một lớp **tùy chọn, mặc định tắt**, còn sản phẩm nộp là bản
**trích xuất**. Muốn bật RAG cho người dùng thật thì cần thêm tầng kiểm tra suy
diễn (NLI) cho giả định sai không có con số — ngoài phạm vi đồ án.
""")

# ---------------------------------------------------------------- PART L ----
md(r"""
---
# PHẦN L — HẠN CHẾ VÀ HƯỚNG PHÁT TRIỂN

## Hạn chế đã biết

1. **Không hiểu từ đồng nghĩa.** TF-IDF so khớp trên **mặt chữ**. Hỏi "xe hơi"
   sẽ không tìm ra bài viết dùng từ "ô tô". Đây là hạn chế cốt lõi của mô hình
   túi từ, không sửa được bằng chỉnh tham số.

2. **Không suy luận, không sinh văn bản, không diễn đạt lại.** Bot **trích xuất
   100%** — nó chỉ trả về câu đã có sẵn trong corpus. Không có thành phần nào
   trong hệ thống có khả năng tạo ra một từ chưa có trong dữ liệu. Câu hỏi kiểu
   "so sánh giá iPhone năm nay với năm ngoái" nằm ngoài khả năng.
   Đây là giới hạn **kiến trúc**, đã được kiểm chứng bằng thực nghiệm ở Phần G3:
   mô hình sinh n-gram ở quy mô dữ liệu này thua trích xuất trên mọi tiêu chí.
   Phần K đo thêm lớp **RAG với PhoGPT**: câu trả lời tự nhiên hơn hẳn (19/33 câu
   dùng được so với 9/33 của prompt đầu) nhưng **không trung thực hơn** — số câu
   chứa thông tin sai không giảm, và với giả định sai không có con số thì nó còn
   tệ hơn bản trích xuất. Vì vậy hạn chế này vẫn được coi là **chưa giải quyết**.

3. **Chuẩn hóa teencode không xét ngữ cảnh.** Recall chỉ 70.2%: bỏ sót các từ
   teencode hiếm. Và với từ mơ hồ như "t" (tôi/tao), mô hình luôn chọn một đích
   duy nhất nên sai ở phần còn lại.

4. **Kho tri thức tĩnh.** 381 bài tại thời điểm crawl. Muốn cập nhật phải chạy
   lại crawler và dựng lại index.

5. **Tập intent nhỏ** (14 intent, 164 pattern). Hai intent chồng lấn ngữ nghĩa
   (`huong_dan` / `liet_ke_chuyen_muc`) vẫn nhầm lẫn.

6. **Tham chiếu chỉ neo vào lượt gần nhất.** "bài thứ hai ấy" hoặc "cái lúc nãy
   bạn nói" sẽ không giải đúng.

8. **Intent classifier là điểm yếu lớn nhất** — 61.5% trên test. Câu ngắn,
   cách nói đời thường có độ tin cậy dưới ngưỡng. Cần thêm pattern và/hoặc hiệu
   chỉnh xác suất; phải đo trên một tập test **mới** vì tập hiện tại đã được dùng nhiều lần.

7. **Không phát hiện mâu thuẫn giữa các bài.** Xếp hạng theo độ mới chỉ *giảm
   nhẹ* vấn đề: bot ưu tiên bài mới khi hai bài gần ngang nhau, nhưng nếu bài cũ
   liên quan **vượt trội** thì nó vẫn thắng. Bot không hiểu bài B phủ định bài A.

## Hướng phát triển

| Hướng | Kỹ thuật | Kỳ vọng |
|---|---|---|
| Hiểu từ đồng nghĩa | Word2Vec / PhoBERT embedding, kết hợp lai với TF-IDF | Giải quyết hạn chế 1 |
| ~~Sinh câu trả lời tự nhiên~~ **ĐÃ THỬ — xem Phần K** | RAG: retriever hiện tại làm "R", PhoGPT-4B-Chat làm "G" | Tự nhiên hơn hẳn, nhưng **không** trung thực hơn. Muốn dùng thật cần thêm tầng kiểm tra suy diễn (NLI) cho giả định sai không có con số |
| **Chọn câu đưa vào ngữ cảnh tốt hơn** | Tăng số câu / chọn câu theo thực thể trong câu hỏi, thay vì cố định 4 câu cosine cao nhất | Sửa lỗi thấy ở Phần K: câu liệt kê màu iPhone không lọt vào ngữ cảnh nên **cả hai** prompt đều bịa màu |
| Chuẩn hóa teencode có ngữ cảnh | Mô hình seq2seq (BARTpho) thay cho tra từ điển | Tăng recall, giải được từ mơ hồ như "t" |
| **Phát hiện cùng-một-sự-việc** | Nhóm các bài có độ tương đồng cao thành cụm, chỉ hiện bài mới nhất trong cụm kèm cảnh báo "có bài mới hơn" | **Giải quyết triệt để hạn chế 7** |
| **BM25F** | Bão hòa tf riêng từng trường (tiêu đề, mô tả, thân) rồi cộng có trọng số, thay mẹo lặp tiêu đề | Cho BM25 cơ hội thật sự — hiện mẹo lặp tiêu đề triệt tiêu ưu điểm của nó |
| Cập nhật index tăng dần | Thêm bài mới mà không dựng lại toàn bộ vocabulary/IDF | Giảm chi phí khi corpus lớn |
| Xếp hạng tốt hơn | BM25 thay TF-IDF (chuẩn hóa độ dài tài liệu tốt hơn) | Recall@1 cao hơn với bài dài |
| Khôi phục dấu | Mô hình seq2seq phục hồi dấu thay vì hạ về âm tiết | Chính xác hơn với câu không dấu |
| Mở rộng intent | Thu thập log chat thật để bổ sung pattern | Giảm nhầm lẫn intent chồng lấn |
| Cập nhật tự động | Lên lịch crawl định kỳ + index tăng dần | Kho tri thức luôn mới |
""")

# ---------------------------------------------------------------- PART M ----
md(r"""
---
# PHẦN M — CÁCH CHẠY SẢN PHẨM

```powershell
# 1. Tạo môi trường
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# 2. (tùy chọn) Thu thập thêm dữ liệu
python src/crawler.py --per-category 45

# 3. Chat trong terminal
python src/cli.py
python src/cli.py --debug                      # hiện intent, điểm, thực thể
python src/cli.py --explain "tin về đảo hải nam"

# 4. Chạy web demo -> mở http://127.0.0.1:8000
python src/api.py

# 5. Chạy kiểm thử và đánh giá
python tests/test_vectorizer.py     # TF-IDF vs sklearn, BM25 vs tham chiếu
python tests/test_chatbot.py        # 21 kiểm thử hồi quy
python src/evaluate.py              # dò trên DEV, báo cáo trên TEST
```

---

# PHẦN N — CHECKLIST NỘP BÀI

- [x] Corpus tiếng Việt tự thu thập (381 bài / 8 chuyên mục)
- [x] Pipeline tiền xử lý kế thừa Lab 03, có so sánh 2 cấu hình
- [x] TF-IDF / BoW / n-gram **tự cài đặt**, có kiểm chứng với sklearn
- [x] Cosine similarity tự cài đặt
- [x] Naive Bayes tự cài đặt cho intent classification
- [x] NER + Regex trích thực thể (Lab 01)
- [x] Quản lý trạng thái hội thoại, giải tham chiếu
- [x] Đánh giá định lượng: Accuracy, macro-F1, Recall@k, MRR
- [x] Dò siêu tham số bằng thực nghiệm, không chọn cảm tính
- [x] Chuẩn hóa teencode học từ ViLexNorm, đánh giá bằng ERR
- [x] Thí nghiệm đối chứng: mô hình sinh n-gram vs truy hồi
- [x] Xếp hạng theo độ mới, xử lý tin lỗi thời
- [x] Tách DEV/TEST, báo cáo trên TEST kèm khoảng tin cậy 95%, đánh giá đầu-cuối
- [x] 21 kiểm thử hồi quy cho các lỗi thật đã gặp
- [x] BM25 tự cài đặt, kiểm chứng, so sánh có kiểm định thống kê
- [x] Error analysis (19 case, vượt yêu cầu tối thiểu 3)
- [x] Thí nghiệm mở rộng: RAG với PhoGPT-4B-Chat — 3 lần chạy thật trên Colab,
      bộ câu bẫy chốt **trước** khi chạy, chốt chặn tất định có kiểm thử, và một
      **kết quả âm** được ghi nhận đầy đủ (RAG không trung thực hơn bản trích xuất)
- [x] Giao diện CLI + Web
- [x] Notebook đã Run và lưu output
""")

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.14.0"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).resolve().parent.parent / "notebooks" / "FinalProject_Chatbot_23IT036.ipynb"
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding="utf-8")
print("Wrote", out, f"({len(cells)} cells)")
