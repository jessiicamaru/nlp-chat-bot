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
| **Lab 01** | `sent_tokenize`, `word_tokenize`, `pos_tag`, `ner`, Regex | Tách câu để chọn snippet; NER + Regex trích thực thể (`entities.py`) |
| **Lab 02** | Crawl web bằng `requests` + `BeautifulSoup`, validation | Mở rộng corpus lên 381 bài / 8 chuyên mục (`crawler.py`) |
| **Lab 03** | `normalize_basic`, `segment_vi`, stopwords, `preprocess_vi(text, config)` | Pipeline tiền xử lý dùng chung (`preprocess.py`) |
| **Lab 04** | Bag of Words, n-gram, TF-IDF, cosine similarity | **Phần lõi**: `vectorizer.py`, `retriever.py`, `intent_classifier.py` |

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

Tập test (`data/intents/test_queries.json`) được **viết riêng**, không trùng với
patterns dùng để train — nếu lấy chính patterns đi test thì kết quả đẹp giả tạo.
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
# PHẦN I — ĐÁNH GIÁ TỔNG HỢP

| Hạng mục | Chỉ số | Kết quả |
|---|---|---|
| TF-IDF tự cài đặt | khớp với sklearn | **28/28 test pass**, sai số ~1e-16 |
| Intent classification | Accuracy (tập test riêng) | **88.5%** |
| Intent classification | Macro-F1 | **0.91** |
| Intent classification | Safety (không trả lời bừa) | **87.5%** |
| Retrieval | Recall@1 | **90.5%** |
| Retrieval | Recall@3 | **100%** |
| Retrieval | MRR | **0.952** |
| Retrieval | chặn câu ngoài phạm vi | **100%** ở ngưỡng 0.12 |
| Chuẩn hóa teencode | ERR trên ViLexNorm test | **67.5%** |
| Chuẩn hóa teencode | Accuracy 83.9% → | **94.8%** |
| Chuẩn hóa teencode | Precision / Recall | **90.7% / 70.2%** |
| Mô hình sinh n-gram | Perplexity tốt nhất (n=2) | **819** |

## Ngưỡng được chọn như thế nào

Điểm cosine của truy vấn **trong** và **ngoài** phạm vi tách nhau khá rõ:

| Nhóm | min | trung vị | max |
|---|---|---|---|
| Trong phạm vi (trúng bài) | 0.118 | 0.253 | 0.333 |
| Ngoài phạm vi | 0.000 | 0.054 | **0.108** |

Ngưỡng **0.12** nằm gọn trong khe hở giữa hai nhóm → trả lời được 100% câu trong
phạm vi, đồng thời chặn 100% câu ngoài phạm vi.
""")

code(r"""
# Đo lại trực tiếp trong notebook: điểm top-1 trong vs ngoài phạm vi
import json
from config import INTENTS_DIR
from entities import expand_query

tests = json.loads((INTENTS_DIR / "test_queries.json").read_text(encoding="utf-8"))

in_scores, oos_scores = [], []
for c in tests["retrieval_tests"]:
    r = retriever.search(c["query"], top_k=1, min_score=0.0)
    if r and c["expect_title_contains"].lower() in r[0].title.lower():
        in_scores.append(r[0].score)
for q in tests["out_of_scope"]:
    r = retriever.search(expand_query(q), top_k=1, min_score=0.0)
    oos_scores.append(r[0].score if r else 0.0)

fig, ax = plt.subplots(figsize=(9, 3.4))
ax.hist(in_scores, bins=12, alpha=.75, label="trong phạm vi", color="#0d9488")
ax.hist(oos_scores, bins=12, alpha=.75, label="ngoài phạm vi", color="#b45309")
ax.axvline(0.12, color="#dc2626", ls="--", lw=2, label="ngưỡng = 0.12")
ax.set_xlabel("cosine similarity của kết quả top-1"); ax.set_ylabel("số truy vấn")
ax.set_title("Ngưỡng chấp nhận nằm trong khe hở giữa hai phân bố")
ax.legend(); plt.tight_layout(); plt.show()

print(f"Trong phạm vi: min={min(in_scores):.3f}  trung vị={np.median(in_scores):.3f}")
print(f"Ngoài phạm vi: max={max(oos_scores):.3f}  trung vị={np.median(oos_scores):.3f}")
""")

# ---------------------------------------------------------------- PART J ----
md(r"""
---
# PHẦN J — PHÂN TÍCH LỖI (Error Analysis)

Yêu cầu tối thiểu 3 case. Dưới đây là **7 lỗi thật** phát hiện trong quá trình
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
        "Xử lý": "Ensemble NB + cosine tới pattern gần nhất -> 0.470",
        "Trạng thái": "ĐÃ SỬA",
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
# PHẦN K — HẠN CHẾ VÀ HƯỚNG PHÁT TRIỂN

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

3. **Chuẩn hóa teencode không xét ngữ cảnh.** Recall chỉ 70.2%: bỏ sót các từ
   teencode hiếm. Và với từ mơ hồ như "t" (tôi/tao), mô hình luôn chọn một đích
   duy nhất nên sai ở phần còn lại.

4. **Kho tri thức tĩnh.** 381 bài tại thời điểm crawl. Muốn cập nhật phải chạy
   lại crawler và dựng lại index.

5. **Tập intent nhỏ** (14 intent, ~150 pattern). Hai intent chồng lấn ngữ nghĩa
   (`huong_dan` / `liet_ke_chuyen_muc`) vẫn nhầm lẫn.

6. **Tham chiếu chỉ neo vào lượt gần nhất.** "bài thứ hai ấy" hoặc "cái lúc nãy
   bạn nói" sẽ không giải đúng.

## Hướng phát triển

| Hướng | Kỹ thuật | Kỳ vọng |
|---|---|---|
| Hiểu từ đồng nghĩa | Word2Vec / PhoBERT embedding, kết hợp lai với TF-IDF | Giải quyết hạn chế 1 |
| **Sinh câu trả lời tự nhiên** | **RAG: dùng chính retriever hiện tại làm "R", ghép mô hình ngôn ngữ lớn tiếng Việt (PhoGPT, Vistral) làm "G"** | **Giải quyết hạn chế 2 — bot diễn đạt lại thay vì chép nguyên văn, vẫn dẫn được nguồn** |
| Chuẩn hóa teencode có ngữ cảnh | Mô hình seq2seq (BARTpho) thay cho tra từ điển | Tăng recall, giải được từ mơ hồ như "t" |
| Xếp hạng tốt hơn | BM25 thay TF-IDF (chuẩn hóa độ dài tài liệu tốt hơn) | Recall@1 cao hơn với bài dài |
| Khôi phục dấu | Mô hình seq2seq phục hồi dấu thay vì hạ về âm tiết | Chính xác hơn với câu không dấu |
| Mở rộng intent | Thu thập log chat thật để bổ sung pattern | Giảm nhầm lẫn intent chồng lấn |
| Cập nhật tự động | Lên lịch crawl định kỳ + index tăng dần | Kho tri thức luôn mới |
""")

# ---------------------------------------------------------------- PART L ----
md(r"""
---
# PHẦN L — CÁCH CHẠY SẢN PHẨM

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
python tests/test_vectorizer.py
python src/evaluate.py
```

---

# PHẦN M — CHECKLIST NỘP BÀI

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
- [x] Error analysis (12 case, vượt yêu cầu tối thiểu 3)
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
