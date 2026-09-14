# FINAL PROJECT REPORT — NATURAL LANGUAGE PROCESSING

# A Vietnamese News Question-Answering Chatbot Built from Scratch: Hand-Implemented TF-IDF Retrieval, Non-Standard Text Handling, and an Empirical Evaluation of a RAG Generation Layer

| | |
|---|---|
| **Student** | Hoàng Công Dũng |
| **Student ID** | 23IT036 |
| **Course** | Natural Language Processing |
| **Source code** | <https://github.com/jessiicamaru/nlp-chat-bot> |
| **Project period** | 2026-09-10 – 2026-09-13 (19 commits) |
| **Deliverables** | source code, executed report notebook (86 cells, 0 errors), 7 technical documents, 95 automated tests, Colab notebook for the RAG experiment |

---

## Abstract

This report describes the construction and evaluation of a Vietnamese news question-answering chatbot
over a corpus of 381 VnExpress articles collected by the project itself. The core is implemented
**from scratch** with the techniques of Labs 01–04: Vietnamese preprocessing, bag-of-words, n-grams,
TF-IDF and cosine similarity hand-written in NumPy/SciPy (matching scikit-learn to within
$10^{-16}$), intent classification with a hand-written Multinomial Naive Bayes, and two-tier
retrieval (article → sentence). The system handles three kinds of non-standard input common in chat:
**unaccented** text (via a secondary syllable-level index), **teencode** (via a normalization lexicon
*learned* from the ViLexNorm corpus, 67.84% ERR on its test split), and **outdated news** (via
freshness-aware ranking that is kept separate from the answer-acceptance threshold).

The methodological emphasis of the project is **honest evaluation**. The initial figures (Recall@1
96.8%, intent accuracy 88.5%) were found to suffer from **test-set leakage**; they were re-measured
under a DEV/TEST protocol with URL-based gold labels, 95% Wilson confidence intervals and end-to-end
evaluation. True results on the TEST split: retrieval Recall@1 **93.4%** (CI 88–97%), MRR **0.950**,
intent accuracy **61.5%** (43–78%), and **77.0%** of news questions receive the correct article through
the full system. Three **negative results** are reported in full: n-gram generation loses to retrieval on
every criterion; a hand-written BM25 does not beat TF-IDF (paired sign test, p = 1.0); and a RAG layer
with PhoGPT-4B-Chat — after three real runs on Google Colab — makes answers **far more natural** (usable
answers 9/33 → 19/33, p = 0.006) but **not more truthful** (answers containing false information 3/33 →
4/33), and is worse than the extractive bot on false premises that contain no numbers. The submitted
chatbot is therefore the extractive one; RAG is kept as an optional extension, disabled by default.

**Keywords:** Vietnamese chatbot, TF-IDF, BM25, Naive Bayes, lexical normalization, teencode,
information retrieval, RAG, PhoGPT, held-out evaluation, negative results.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Background and Related Work](#2-background-and-related-work)
3. [Data](#3-data)
4. [System Architecture](#4-system-architecture)
5. [Methods](#5-methods)
6. [Evaluation Methodology](#6-evaluation-methodology)
7. [Development History by Phase](#7-development-history-by-phase)
8. [Experimental Results](#8-experimental-results)
9. [Discussion and Error Analysis](#9-discussion-and-error-analysis)
10. [Limitations and Future Work](#10-limitations-and-future-work)
11. [Conclusion](#11-conclusion)
12. [Statement on the Use of AI Tools](#12-statement-on-the-use-of-ai-tools)
13. [References](#13-references)
14. [Appendices](#14-appendices)

---

## 1. Introduction

### 1.1. Context

Question answering over a document collection is a classic problem in natural language processing and
information retrieval. Vietnamese adds three specific difficulties. First, Vietnamese is an isolating
language whose meaningful unit is the **word**, which may span several syllables (`hải_nam`,
`sức_khỏe`), so **word segmentation** is required before any representation. Second, chat users often
type **without diacritics** (`tin ve dao hai nam`) or in **teencode** (`bt gì về vụ iphone k b`), which
segmenters trained on standard text handle completely wrongly. Third, **news** changes over time: an
article published today may contradict one from last week.

### 1.2. Goals and Constraints

1. Build a Vietnamese chatbot that answers questions about news in a self-collected article corpus.
2. **Implement the core** (text representation, intent classification, retrieval) with exactly the
   techniques taught in Labs 01–04, and **prove** the implementations correct.
3. Handle unaccented input, teencode, and outdated news.
4. Tune every hyper-parameter **empirically** and report every result **honestly**, including negative
   results.
5. (Extension) Test empirically whether a generative model should be used to make answers more natural.

Constraints: no large language model in the core; `scikit-learn` is used only to **cross-check** in tests
and never at runtime; the core runs on a laptop CPU.

### 1.3. Contributions

- **Verified from-scratch implementations** of TF-IDF, cosine similarity, BM25, Multinomial Naive Bayes
  and an n-gram LM, with 45 checks against scikit-learn and a reference implementation.
- **Non-standard text handling:** a syllable-level index for unaccented input; a teencode lexicon
  *learned* from 10,467 annotated sentence pairs under three safety conditions; an "accent frame
  preservation" mechanism.
- **Outdated-news handling:** multiplicative freshness ranking, with **ranking and acceptance
  deliberately decoupled**.
- **An honest evaluation protocol:** detection and repair of test-set leakage, DEV/TEST split,
  URL-based labels, Wilson CIs, sign tests, end-to-end evaluation, and a log of every test-set use.
- **Three fully reported negative results:** n-gram generation, BM25, and RAG with PhoGPT.
- **21 regression tests**, each tied to a real bug; they caught two design bugs on their first run.

### 1.4. Report Structure

Section 2 summarizes the background. Section 3 describes the data. Sections 4–5 present the architecture
and methods. Section 6 describes the evaluation protocol. Section 7 narrates the **development history
phase by phase**, including the figures at each phase and every time the report notebook had to change.
Section 8 presents results. Sections 9–11 discuss, list limitations and conclude.

---

## 2. Background and Related Work

### 2.1. Vietnamese Preprocessing

The preprocessing pipeline inherited from Lab 03 performs Unicode **NFC** normalization (two strings that
render identically may use different code points in NFD form), word segmentation with underthesea's
`word_tokenize`, lower-casing, punctuation removal and (depending on configuration) stopword removal.

### 2.2. Bag-of-Words, n-grams and TF-IDF

A bag-of-words model represents a document $d$ by term frequencies $\mathrm{tf}(t, d)$. Bigrams recover
some word-order information (distinguishing "không tốt" from "tốt"). TF-IDF down-weights common terms:

$$\mathrm{idf}(t) = \ln\frac{1 + N}{1 + \mathrm{df}(t)} + 1 \qquad \mathrm{tfidf}(t,d) = \mathrm{tf}'(t,d)\cdot\mathrm{idf}(t)$$

with $\mathrm{tf}'(t,d) = 1 + \ln\mathrm{tf}(t,d)$ under *sublinear tf*. Vectors are L2-normalized, so cosine
similarity reduces to a dot product: $\cos(\vec a, \vec b) = \vec a\cdot\vec b/(\lVert\vec a\rVert\lVert\vec b\rVert)$.

### 2.3. Okapi BM25

BM25 [Robertson & Zaragoza, 2009] fixes two weaknesses of TF-IDF — term-frequency **saturation** and
**parameterized** length normalization:

$$\mathrm{score}(q,d) = \sum_{t\in q}\mathrm{idf}(t)\cdot\frac{\mathrm{tf}(t,d)\,(k_1+1)}{\mathrm{tf}(t,d) + k_1\left(1 - b + b\,\frac{|d|}{\mathrm{avgdl}}\right)}, \qquad \mathrm{idf}(t) = \ln\!\left(\frac{N-\mathrm{df}(t)+0.5}{\mathrm{df}(t)+0.5}+1\right)$$

The "+1" idf variant (as in Lucene) is always positive, so matching a very common term never lowers a score.

### 2.4. Multinomial Naive Bayes

For class $c$ and a document $x$ represented by term weights, with Laplace smoothing $\alpha$:

$$P(t\mid c) = \frac{\mathrm{count}(t,c) + \alpha}{\sum_{t'}\mathrm{count}(t',c) + \alpha|V|}, \qquad \hat c = \arg\max_c\Big[\log P(c) + \sum_t x_t\log P(t\mid c)\Big]$$

The posterior $P(c\mid x)$ is obtained by a numerically stable softmax over the log-likelihoods.

### 2.5. Lexical Normalization

Lexical normalization maps non-standard tokens to their standard forms (`k` → `không`). ViLexNorm
[Nguyen et al., EACL 2024] contains 10,467 Vietnamese social-media sentence pairs with human-annotated
normalizations. The standard metric is **ERR** (Error Reduction Rate):

$$\mathrm{ERR} = \frac{\mathrm{Acc}_{\text{after}} - \mathrm{Acc}_{\text{before}}}{1 - \mathrm{Acc}_{\text{before}}}$$

ERR is preferred over raw accuracy because most tokens (~84%) are already correct.

### 2.6. n-gram Language Models

An n-gram model makes a Markov assumption of order $n-1$. To avoid zero probabilities we use recursive
interpolation:

$$P_{\text{interp}}(w\mid h) = \lambda\,P_{\text{ML}}(w\mid h) + (1-\lambda)\,P_{\text{interp}}(w\mid h_{2:})$$

evaluated by perplexity $\mathrm{PP} = \exp\!\big(-\tfrac1N\sum_i\log P(w_i\mid h_i)\big)$.

### 2.7. Retrieval-Augmented Generation

RAG [Lewis et al., 2020] couples a retriever with a generator: the retriever fetches relevant documents
and the generator writes an answer grounded in them. This project uses **PhoGPT-4B-Chat** [Nguyen et al.,
2023], VinAI's ~3.7B-parameter Vietnamese generative model, with its official prompt template
`### Câu hỏi: {instruction}\n### Trả lời:`.

### 2.8. Survey of Reference Projects

Before writing code, six open-source projects were surveyed (`docs/01`):

| Project | Technique | Decision |
|---|---|---|
| Dec1mo/Vietnamese-Chatbot-From-Scratch | Keras NN + context handler + underthesea | Inherit the **architecture** intent → context → response; replace the model with hand-written NB + TF-IDF |
| undertheseanlp/chatbot | ChatScript rules + Django 1.11, GPL-3.0 | Not used |
| heraclex12/vietnamese-chat-with-rasa | Rasa framework | Intent organization only (Rasa does all NLU, contrary to the from-scratch requirement) |
| sushant097/Chatbot-using-Python-NLTK | TF-IDF + cosine (English, calls sklearn) | Inherit the retrieval **idea**, rewrite for Vietnamese |
| YUSANITY/TF-IDF-DOCUMENT-RETRIEVAL-CHATBOT | TF-IDF document scoring | Reference |
| stopwords/vietnamese-stopwords | 1,942 Vietnamese stopwords | Used directly |

Key differences from these projects: a hand-written, verified TF-IDF; unaccented and teencode handling;
thresholds tuned empirically on a DEV split; quantitative evaluation with confidence intervals.

---

## 3. Data

### 3.1. Article Corpus

The corpus from Lab 02 had only **28 articles** in a single category (Travel). `src/crawler.py` extended it
to **381 articles across 8 categories** from VnExpress, following the Lab 02 procedure (requests +
BeautifulSoup + validation) and adding: a random 0.8–1.6 s delay between requests, a per-URL cache to avoid
re-downloading, rejection of articles whose body is under 300 characters, a separate error log, and
**merging** into the existing corpus by URL (data accumulates, never overwritten).

| Category | Articles |
|---|---|
| Du lịch (Travel) | 72 |
| Công nghệ (Technology) | 48 |
| Sức khỏe (Health) | 45 |
| Đời sống (Life) | 45 |
| Kinh doanh (Business) | 44 |
| Thể thao (Sports) | 44 |
| Khoa học (Science) | 43 |
| Giáo dục (Education) | 40 |
| **Total** | **381** |

Each article has `url, source, category, title, description, text, published_at, crawled_at`. Mean body
length is **3,358 characters** (median 2,857). Publication dates range from 2025-04-17 to 2026-09-10.

### 3.2. Intent Set

`data/intents/intents_vi.json` contains **14 intents and 164 hand-written patterns**, each intent bound to an
*action*:

| Action | Intents |
|---|---|
| `reply` (canned response) | `chao_hoi`, `tam_biet`, `cam_on`, `hoi_ve_bot`, `dong_y`, `tu_choi`, `che_bai` |
| `help` | `huong_dan` |
| `list_categories` | `liet_ke_chuyen_muc` |
| `stats` | `thong_ke_corpus` |
| `browse_category` | `tin_theo_chuyen_muc` |
| `summarize` | `tom_tat_bai` |
| `source` | `hoi_nguon` |
| `retrieve` | `tim_tin` |

### 3.3. Linguistic Resources

- **Stopwords:** 1,942 entries from `stopwords/vietnamese-stopwords` (loaded in both space-separated and
  underscore-joined forms to match segmenter output).
- **ViLexNorm** [Nguyen et al., 2024]: 10,467 sentence pairs with official train/dev/test splits; licensed
  CC BY-NC-SA 4.0 (research/education use, with attribution).

### 3.4. Chatbot Evaluation Sets (DEV/TEST)

The evaluation sets are generated by `tools/build_eval_sets.py` (seed 2026) with the rule that **every old
query** — any query used for tuning before the leakage was discovered — must go to DEV, while **new**
queries are split randomly 40% DEV / 60% TEST. Each retrieval query is labeled with the **URL(s) of the
correct article(s)** instead of the lenient criterion "the title contains word X".

| | DEV | TEST |
|---|---|---|
| Purpose | tuning, may be viewed repeatedly | **reporting only** |
| Retrieval queries | 112 (accented 78 · unaccented 23 · teencode 11) | 122 (accented 86 · unaccented 22 · teencode 14) |
| Out-of-scope queries | 28 | 24 |
| Intent queries | 55 | 32 (26 belong to fixed intents and are used for accuracy) |

**Conflicting-news case** (`data/eval/conflict_case.json`): two synthetic articles nine days apart about the
Cát Linh metro fare — the 01/09 article says "raised to 15,000 VND", the 10/09 article says "increase
postponed, fare stays at 8,000 VND" — together with four phrasings of the question. It is used as a **hard
constraint** when tuning freshness.

### 3.5. RAG Evaluation Sets

- **DEV traps** (`data/eval/rag/dev_traps.json`): 6 questions, of which 5 are real traps and 1 turned out to be
  a valid question (Section 7.13).
- **TEST traps** (`data/eval/rag/test_traps.json`): **11 new questions**, written and **frozen before the run**,
  every detail checked against the full article text. Types: 2 false premises with a number, 1 wrong number
  that appears elsewhere in the article (a known blind spot), 6 false premises without numbers, and 2 questions
  about details the article does not contain.

---

## 4. System Architecture

### 4.1. Processing a Chat Turn

```text
                               user message
                                     |
       [0] normalizer.prepare_user_text   teencode normalization (lexicon learned from ViLexNorm),
           |                              elongation collapsing; unaccented input -> strip accents again
       [1] entities.extract               category + NER (underthesea) + regex
           |
       [2] IntentClassifier.predict       TF-IDF (1,2) + Naive Bayes; w_nb = 1.0
           |
           +-- conf >= 0.25 and action != retrieve  -->  run the action
           |       (reply / help / list_categories / stats / browse_category / summarize / source)
           |
           +-- otherwise --> [3] NewsRetriever.search
                              - accented query -> main index; unaccented -> syllable index
                              - drop question-frame words (QUERY_FRAME_WORDS)
                              - RANK:    TF-IDF cosine x (1 + 0.6 · recency)
                              - ACCEPT:  plain cosine >= 0.13
                                         (x 0.6 if the user already named a category)
                              - tier 2: pick the 2 closest sentences in the article
                                   |                     |
                             above threshold       below threshold
                                   v                     v
                      answer + date + source          fallback
                                   |
                              DialogueState (last article, category, fallback streak)

   Optional layer, disabled by default: rag.RagChatbot wraps the flow above and runs only when [3]
   returned an above-threshold article: premise guard -> PhoGPT -> cleaning -> empty/echo/number guards.
```

### 4.2. Modules

| File | Responsibility | Lab |
|---|---|---|
| `config.py` | Paths, hyper-parameters (tuned on DEV), question-frame words | — |
| `preprocess.py` | NFC, segmentation (cached), stopwords, accent stripping, syllable folding | Lab 03 |
| `normalizer.py` | Teencode lexicon learning; `prepare_user_text` shared by bot and evaluation | — |
| `vectorizer.py` | BoW, n-grams, TF-IDF, L2, cosine, BM25 — from scratch | Lab 04 |
| `intent_classifier.py` | Hand-written Multinomial NB + cosine-to-pattern signal | TF-IDF + cosine: Lab 04; **NB: outside the lab syllabus** |
| `retriever.py` | Two-tier retrieval, two indexes, freshness, disk cache, explanations | Labs 01 + 04 |
| `dates.py` | VnExpress date parsing, recency scores | — |
| `entities.py` | NER + regex, category detection | Lab 01 |
| `dialogue.py` | Dialogue state, "that article" reference resolution | — |
| `chatbot.py` | Orchestration | — |
| `crawler.py` | VnExpress collection | Lab 02 |
| `generator.py` | n-gram LM — control experiment, not used by the bot | — |
| `evaluate.py` | Tune on DEV, report on TEST, Wilson CI, sign test | — |
| `rag.py` | Optional RAG layer | — |
| `cli.py`, `api.py`, `web/` | Command-line interface, FastAPI + chat page | — |

### 4.3. Six Design Decisions

1. **Every path has a confidence threshold.** The worst failure of a retrieval chatbot is returning a random
   article in a confident voice; below threshold the bot says it does not know.
2. **Ranking and acceptance use different measures.** "Which article comes first?" uses cosine × freshness;
   "is there enough evidence to answer?" uses plain cosine only (Section 7.8).
3. **Two preprocessing configurations:** intents **keep** stopwords (5–10 token questions where "là gì",
   "có không" are the signal); retrieval **removes** them (long articles, stopwords dilute vectors).
4. **Separate indexes** for accented and unaccented text, so the main index is not diluted and the tuned
   thresholds stay valid.
5. **Each business rule lives in exactly one place** (a lesson from two real bugs, Sections 7.4 and 7.8).
6. **The LLM is an optional layer**; no core module depends on it.

---

## 5. Methods

### 5.1. Preprocessing

`preprocess_vi(text, config)` keeps the Lab 03 function signature:
`normalize_basic → segment_vi → lowercase → remove_numbers → remove_punctuation → remove_stopwords`.
`segment_vi` is wrapped in `lru_cache` because the chatbot segments the same sentence several times. Two
configurations are used:

| | `CONFIG_INTENT` | `CONFIG_RETRIEVAL` |
|---|---|---|
| `word_segment`, `lowercase`, `remove_punctuation` | True | True |
| `remove_stopwords` | **False** | **True** |
| `remove_numbers` | False | False |

By default `remove_punctuation` keeps punctuation **inside** letters/digits (`keep_inner=True`) so tokens
like `TP.HCM`, `3,5%`, `C++` survive; the aggressive Lab 03 mode is kept for comparison.

### 5.2. Hand-Written TF-IDF

`vectorizer.py` implements `CountVectorizer` and `TfidfVectorizer` with NumPy + SciPy (sparse CSR):

- Vocabulary sorted alphabetically for stable indices across runs; `min_df`/`max_df` filtering.
- `idf = ln((1+N)/(1+df)) + 1` (smooth), optional sublinear `1 + ln tf`.
- idf multiplied element-wise into `X.data` via `X.indices`, without building a diagonal matrix.
- Row-wise L2 normalization; all-zero rows are left untouched to avoid division by zero.
- `top_terms` and `_matched_terms` support **explanations**: which terms matched and how much each
  contributed.

**Verification:** `tests/test_vectorizer.py` compares against `sklearn.feature_extraction.text` on 5
configurations (unigram/bigram, sublinear, smoothing, min_df/max_df, …) and edge cases — all 28 checks match
with absolute error around $10^{-16}$.

### 5.3. Two-Tier Retrieval

**Tier 1 — article selection.** Each article is indexed as the title repeated 3 times, the description
repeated twice, and the body (`TITLE_WEIGHT = 3`, `DESC_WEIGHT = 2`) — up-weighting important fields without
changing the formula. The index uses unigrams + bigrams, `min_df = 1`, `max_df = 0.85`, sublinear tf; the
main vocabulary has **105,020 terms**.

**Tier 2 — sentence selection.** Within the chosen article, sentences are split with `sent_tokenize`
(Lab 01); sentences shorter than 30 or longer than 400 characters (captions, bylines) are dropped;
**duplicates are removed** (VnExpress often repeats the lead sentence in the body); a **local TF-IDF for that
article alone** is fitted, and the 2 sentences with the highest cosine to the question are returned **in their
original order**.

**Explanations.** `--explain` (CLI) and `/explain` (API) report matched terms and their contributions, OOV
terms, and rejected candidates.

### 5.4. Unaccented Input: a Syllable-Level Index

`word_tokenize` is trained on accented text and mis-segments unaccented input:
`word_tokenize("tin ve dao hai nam") → ['ve_dao', 'hai', 'nam']`. Instead of restoring diacritics (a separate
seq2seq problem), the system projects **both sides** onto the same unaccented syllable space:

- Document side: segment the **accented** text (accurately) first, then strip accents and split compound
  words into syllables (`["đảo","hải_nam"] → ["dao","hai","nam"]`).
- Query side: skip segmentation and split on whitespace.
- Bigrams recover most compound-word information (`"hai nam"` appears as a bigram).

The syllable index (87,294 terms) is kept **separate** from the main index and is used only when the query has
**no diacritics at all** (`has_diacritics`). The intent classifier has a similar secondary model.

### 5.5. Teencode Normalization

**Lexicon learning** (`normalizer.learn_lexicon`) on the ViLexNorm training split:

1. Use only sentence pairs with the **same number of tokens** (79.5% of pairs) for positional alignment.
2. Count every mapping `a → b` at changed positions.
3. Keep `a → b` only if **all three** conditions hold:

| Condition | Threshold | Reason |
|---|---|---|
| `a` occurs often enough | `count ≥ 4` | avoid learning annotation noise |
| `a` is **usually** changed | change rate ≥ 0.5 | the most important guard: without it, ordinary words like "cả", "mà" get replaced |
| `b` dominates | ≥ 0.5 | avoid arbitrary choices when `a` is ambiguous |

The result is **342 mappings** (e.g. `bt/bik/bít → biết`, `b/bn → bạn`, `k → không`). Genuine ambiguity is
recorded: `t → tôi` (889 times) versus `t → tao` (132 times) — `tôi` is chosen, accepting errors when the
sentence meant "tao".

**Application.** `TeencodeNormalizer` runs **before** segmentation: dictionary lookup, then collapsing
characters repeated three or more times (`"khummm" → "khum" → "không"`). Collapsing applies **to letters
only** — the first version also collapsed digits and corrupted numbers (Section 7.12).

**Accent frame preservation** (`prepare_user_text`). The lexicon always returns **accented** words; if the
original input was unaccented and the result were kept as-is, a single corrected token would route the whole
query to the accented index, where the remaining tokens are OOV. Therefore, if the original input has no
diacritics, accents are stripped again after normalization. The function is **shared** by the chatbot and
`evaluate.py`, so evaluation reflects what users actually get.

**Question-frame word filtering.** Normalization *expands* abbreviations into full words and inadvertently
adds frame tokens ("biết", "không", "bạn") that dilute the L2-normalized query: `"giá iphone"` scores 0.167
but `"biết gì về vụ iphone không bạn"` only 0.100. `QUERY_FRAME_WORDS` (52 words, defined **once** in
`config.py`) are removed from the query before retrieval; if everything would be removed, the original is kept.

### 5.6. Intent Classification

Features: unigram + bigram TF-IDF, sublinear tf, stopwords kept (493-term vocabulary). Model: Multinomial NB
with $\alpha = 0.3$ (below 1 because the training set is small and intents are well separated).

The original design was an **ensemble** of two complementary signals:

$$\mathrm{score}(c) = w_{nb}\cdot P_{NB}(c\mid x) + (1-w_{nb})\cdot\max_{p\in c}\cos(\vec x, \vec p)$$

because NB is **flat** on 2–3 token inputs ("cảm ơn nhé" scores only 0.189 despite an obvious intent), while
cosine to the nearest pattern is very sensitive to short inputs ("bye" → 1.000). Re-tuning on DEV selected
$w_{nb} = 1.0$ — i.e. **plain NB**; the cosine signal remains in the code so the experiment is reproducible.
Intent acceptance threshold: 0.25.

### 5.7. Entities and Dialogue State

`entities.py` follows the Lab 01 principle: **fixed formats → regex**, **open entities → NER**.

- Regex: `email`, `phone` (`(0|+84)` + 8–10 digits), `url`, `date` (`12/8/2026`, `2/9`), `quantity` (number +
  unit: %, tỷ, triệu, nghìn, đồng, USD, ngày, giờ, tháng, năm, km, kg, người). The `quantity` pattern ends with
  `(?!\w)` rather than `\b` — there is no word boundary after `%`, so `\b` silently missed `3,5%` (a real bug,
  now tested).
- NER: `underthesea.ner` with BIO tags, merging B-/I- tokens into PER/LOC/ORG entities.
- Category: **accent-insensitive** keyword matching, preferring the longest match.

`DialogueState` stores the last articles mentioned (`last_results`), the current category (`last_category`)
and the fallback streak (to offer help when the user is stuck). This is the simplest form of reference
resolution: "tóm tắt bài đó" (summarize that article) and "cho mình link" (give me the link) anchor to the most
recent turn. The API keeps a separate `DialogueState` per `session_id`.

If the user only names a category ("tin sức khỏe"), the bot browses it newest first; if a topic is added ("tin
sức khỏe về ăn chuối"), it searches within the category with the relaxed threshold $0.13 \times 0.6$ — the
candidate set is smaller, so a lower cosine is still strong evidence. Detecting "a topic beyond the category
name" compares at the **syllable** level (Section 7.4).

### 5.8. Freshness-Aware Ranking

$$\mathrm{recency}(d) = 0.5^{\,\mathrm{age}(d)/h}, \qquad \mathrm{score}'(q,d) = \mathrm{rank}(q,d)\cdot\big(1 + \alpha\cdot\mathrm{recency}(d)\big)$$

with half-life $h$ (days) and a small $\alpha$. Choices and rationale:

- **Multiply, not add:** an irrelevant article (score ≈ 0) stays ≈ 0 however new it is; multiplication is
  scale-free, so it also works with BM25.
- **Small $\alpha$:** the goal is to **break ties**, not to always prefer new articles.
- **The reference date is the newest date in the corpus**, not `datetime.now()`, for reproducibility. Articles
  without a date receive the median recency.
- **The acceptance threshold applies to plain cosine**, not to the freshness-boosted score.
- Every answer shows the **publication date**, and category browsing is sorted newest first.

Current values (tuned on DEV): $h = 3$ days, $\alpha = 0.6$.

### 5.9. Index Caching

Building the index takes about 24 seconds, almost entirely `word_tokenize`. The index is saved with `joblib`
together with a **SHA-256 fingerprint** of the corpus content and every parameter that affects the index
(`ngram_range`, `min_df`, `max_df`, `sublinear_tf`, field weights, preprocessing config). Cached startup takes
**~1.1 seconds**. Recency is always **recomputed** after loading, because the half-life is not part of the
fingerprint (Section 7.7). Count matrices are cached too, so changing BM25's $(k_1, b)$ needs no re-segmentation.

### 5.10. BM25

`bm25_idf`, `bm25_weights` and `bm25_scores` precompute a weight matrix $W$ for the whole corpus such that
$\mathrm{score}(q,d) = W_d\cdot\vec q_{\text{counts}}$; each query is a single sparse matrix product. Since
scikit-learn has no BM25, the vectorized version is checked against **a naive implementation written directly
from the definition** (loops) on 4 $(k_1, b)$ pairs, plus a property test (repeating a term increases the score
towards the ceiling $\mathrm{idf}\cdot(k_1+1)$) — 17 checks. BM25 is used only for **ranking**; acceptance
still uses TF-IDF cosine because BM25 scores are unbounded and not comparable across queries.

### 5.11. Control Experiment: n-gram Generation

`generator.py` implements an n-gram LM with recursive interpolation, trained on sentences split from the
corpus (9,567 training sentences, 1,063 test sentences, 216,976 tokens), measuring perplexity for $n = 1..4$ and
sweeping $\lambda$. Purpose: to answer with data the question "why not let the model generate answers?".

### 5.12. The RAG Layer with PhoGPT-4B-Chat

**Context assembly.** Up to 3 above-threshold articles; for each, the **4** most relevant sentences (reusing
tier 2), truncated to 900 characters, with the publication date.

**Two prompt versions** (full text in Appendix B):

| | v1 | v2 |
|---|---|---|
| Instructions | 6 **numbered** rules | **one prose paragraph** |
| Citations | require `[i]` | dropped; sources attached by code |
| Context format | `[i] (đăng dd/mm/yyyy, chuyên mục X)` | `Tin i (ngày dd/mm/yyyy): title` |
| Keyword queries | passed as-is | rewritten as "Các tin trên cho biết gì về X?" ("What do the articles say about X?") |
| Post-processing | cut at `###` | drop copied-prompt lines, list markers and "Tin N:" labels; keep the first paragraph; drop repeated sentences; at most 4 sentences; rewrite the opening to "Theo các bài báo," ("According to the articles,") |
| Guards | none | yes (below) |
| Max new tokens | 256 | 160 |

**Deterministic guards** (v2 only; all are regex/matching rules, testable without a GPU):

| Guard | When | Rule | Action |
|---|---|---|---|
| Premise | before the LLM | a number or code (letters + digits, e.g. `IP68`) in the question **appears nowhere** in the full text of the retrieved articles; numbers with units ("5 triệu", "100 nghìn") are compared by **value** | do not show generated text; answer "the articles do not mention «…»" + extract |
| Empty | after the LLM | generated text is empty after cleaning | show the extract |
| Echo | after the LLM | keyword-style queries only: the answer adds no content syllable beyond the question | show the extract |
| Invented number | after the LLM | the answer contains a number not in the sources, the question or the publication dates (ignoring list numbering; "9h40" ≡ "9 giờ 40") | show the extract |

**Backends.** `LlamaCppBackend` (GGUF via llama.cpp on GPU; Q4_K_M in run 1, Q8_0 in runs 2–3),
`TransformersBackend` (float16; failed to load on Colab because PhoGPT's custom code declares the
`triton_pre_mlir` package), and `EchoBackend` (a stub for tests). Greedy decoding (temperature 0),
`repeat_penalty = 1.1`.

**Safety principle:** the LLM is called only when retrieval found above-threshold evidence; every other path
(intent, fallback) keeps the extractive answer.

---

## 6. Evaluation Methodology

### 6.1. Metrics

| Component | Metrics |
|---|---|
| Retrieval | Recall@1, Recall@3, MRR (gold article by URL) |
| Intent | Accuracy, macro-F1 (a prediction must be correct **and** above threshold) |
| Out of scope | blocked rate (component) and refused rate (end-to-end) |
| End-to-end | via `bot.respond()`: news question → correct top article; out-of-scope → refusal |
| Teencode | token accuracy before/after, ERR, precision, recall, F1 |
| Generation | perplexity |
| RAG | manual A–E grading; safety on traps; automatic: numbers absent from sources, source-token ratio, valid citations |

### 6.2. 95% Wilson Confidence Intervals

With test sets of a few dozen items, a single error moves a rate by several points. Every rate carries a Wilson
interval:

$$\frac{\hat p + \frac{z^2}{2n}}{1 + \frac{z^2}{n}} \pm \frac{z}{1+\frac{z^2}{n}}\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}, \qquad z = 1.96$$

### 6.3. Paired Sign Test

To decide whether a more complex method should be adopted, results are compared per question, ties are dropped,
and a two-sided exact binomial p-value with $p = 0.5$ is computed:

$$p = \min\!\Big(1,\; 2\sum_{i=0}^{\min(W,L)}\binom{W+L}{i}\,2^{-(W+L)}\Big)$$

**Rule:** a more complex method must win **with statistical significance** on DEV.

### 6.4. DEV/TEST Protocol

- **Phase 1 — tune on DEV** (`evaluate.py`): intent (grid $w_{nb}$ × threshold) → ranking method (TF-IDF or BM25,
  $k_1$, $b$; freshness **off**) → freshness (grid $h$ × $\alpha$, **hard constraint**: the conflicting-news case
  must be correct for all four phrasings, then maximize MRR) → retrieval threshold (grid step 0.005).
- **Phase 2 — report on TEST** with the frozen parameters; nothing is changed after looking.
- Every query goes through `prepare_user_text` exactly like the chatbot.
- Every use of the TEST split is **logged** (Appendix C).

### 6.5. Manual Grading Scale for RAG Answers

| Label | Meaning |
|---|---|
| **A** | faithful to the source and answers the question |
| **B** | faithful but badly formatted (lists, repeated sentences, copied prompt fragments) |
| **C** | non-answer (repeats the question / invents questions) |
| **D** | contains **false** information relative to the source |
| **E** | refuses although the article contains the answer |

Every claim is checked against the **full text** of the source article. Grading is applied to the text PhoGPT
generated, **before** guards. Run-3 labels are stored in `data/eval/rag/run3_annotation.csv` and
`run3_trap_annotation.csv`.

---

## 7. Development History by Phase

This section retells the project chronologically from the commit history. Notably, **many figures changed**
between phases — sometimes up because of improvements, sometimes down because the measurement was corrected.
Both kinds of change are recorded.

### Overview

| Phase | Commit(s) | Content | Report notebook |
|---|---|---|---|
| 1 | `190f608` | NLP core: preprocessing, TF-IDF, NB, retrieval, crawler, evaluation | — |
| 2 | `16ff385` | CLI, web, unaccented input | — |
| 3 | `f45118f`, `bdd5050`, `ad9bd81` | First notebook, docs 01–02, snippet deduplication | **45 cells**, 7 error cases |
| 4 | `2f4d3de` | Teencode normalization + frame-word filtering | — |
| 5 | `1b7f8a2` | n-gram generation experiment, docs 03–04 | **61 cells**, 12 cases |
| 6 | `5b90008`, `72f3f1a` | Freshness + cache, docs 05, fixing a wrong comparison | **72 cells**, 15 cases |
| 7 | `c3da7c2` | **DEV/TEST split** — fixing test-set leakage | — |
| 8 | `c4a6a23` | 21 regression tests + 2 design bugs fixed | — |
| 9 | `6587696` | BM25 (negative result) | — |
| 10 | `5320448` | docs 06, notebook evaluation rewritten | **79 cells**, 19 cases |
| 11 | `98fb144` | RAG layer + Colab notebook | — |
| 12 | `1743132` | RAG run 1 analysis, prompt v2, digit-collapsing bug fix | — |
| 13 | `fe10190` | RAG run 2 analysis, test traps frozen, **code frozen** | — |
| 14 | `be306ce`, `40c1792` | RAG run 3 on TEST, notebook Part K | **86 cells** |
| 15 | (documentation audit commit) | Full documentation audit, this report | 86 cells |

### 7.1. Phase 1 — NLP Core (`190f608`)

Implemented `preprocess`, `vectorizer`, `intent_classifier`, `retriever`, `entities`, `dialogue`, `chatbot`,
`crawler`, `evaluate`, and expanded the corpus from 28 to 381 articles. Figures reported **at that time** (on a
hand-written test set later found to have been used for tuning):

| | Reported then |
|---|---|
| TF-IDF vs sklearn | 28/28, error ~1e-16 |
| Intent accuracy / macro-F1 | 88.5% / 0.91 |
| Retrieval Recall@1 / Recall@3 / MRR | 85.7% / 95.2% / 0.905 |

### 7.2. Phase 2 — Interfaces and Unaccented Input (`16ff385`)

Added `cli.py` (with `/debug` and `/explain`), `api.py` (FastAPI, per-session state), and the chat page.
Discovered that `word_tokenize` mis-segments unaccented input → implemented the syllable index (Section 5.4). On
the same 21 queries: Recall@1 85.7% → **90.5%**, Recall@3 95.2% → **100%**, MRR 0.905 → **0.952** — foreign
proper-noun queries ("champions league man utd"), which carry no diacritics, now matched correctly.

### 7.3. Phase 3 — First Notebook (`f45118f`, `bdd5050`, `ad9bd81`)

Report notebook with **45 cells, 0 errors, 2 charts, 7 error-analysis cases**; docs/01 (survey) and docs/02
(architecture); `build_notebook.py` moved into `tools/` so it is version-controlled. Fixed **duplicated
sentences in snippets**: VnExpress repeats the lead sentence in the body, so the two top-scoring sentences could be
identical → deduplicate at sentence splitting. No metric change.

### 7.4. Phase 4 — Teencode (`2f4d3de`)

Implemented the learned teencode lexicon (Section 5.5). Results at the time on the test split: accuracy 83.88% →
94.77%, **ERR 67.54%**, precision/recall 90.71% / 70.16%. Found that normalization **dilutes** the query vector →
added `QUERY_FRAME_WORDS`. Two bugs found while testing:

1. The frame-word lists in `chatbot.py` and `retriever.py` had **drifted apart** ("biết" existed in only one) →
   merged into a single source in `config.py`.
2. `word_tokenize("Sức khỏe") → ["sức","khỏe"]` but `word_tokenize("... tin sức khỏe gì luôn") → ["sức_khỏe"]` →
   category-name matching moved to the syllable level.

### 7.5. Phase 5 — Generation Experiment (`1b7f8a2`)

Implemented the n-gram LM (Section 5.11). Three findings (detailed in Section 8.7): perplexity **increases** with
$n$; large $n$ turns "generation" into "copying"; generated text is false at every $n$. A comment pre-written in the
module ("perplexity decreases as n grows") was contradicted by the data and corrected. Notebook: **61 cells**, with
new Parts G2 (teencode) and G3 (generation); error analysis 7 → **12 cases**; docs/03 and docs/04.

### 7.6. Phase 6 — Outdated News and Caching (`5b90008`, `72f3f1a`)

Built the conflicting-news case: plain TF-IDF ranked the **old** article first (0.4956 vs 0.4098), and the outcome
**flipped with phrasing** (2 of 4 phrasings returned the outdated article). Root cause: `published_at` was collected
but never used. Added freshness ranking (Section 5.8), always showing dates and sorting browsing by date. The grid
search at the time chose a **7-day** half-life with $\alpha = 0.6$.

Two bugs found while testing:

1. **Unrepresentative tuning set:** only standard-written queries, although the bot now handled unaccented and
   teencode input → added 10 queries + 4 out-of-scope queries and re-tuned the threshold (then 0.12 → 0.155, on
   freshness-boosted scores).
2. **Normalization broke index routing:** `"thoi tiet sao hoa hom nay"` → `"thôi tiet sao hoa hom nay"` (now
   "accented") → intent `tam_biet` (0.280) → the bot replied **"Goodbye!"** to a weather question → added accent
   frame preservation.

Index caching: startup 24.2 s → **1.1 s**. Figures reported then: Recall@1 90.5% → 96.8%, MRR 0.984.

**Fixing a wrong comparison within the phase** (`72f3f1a`): "90.5% → 96.8%" compared two **different** test sets (21
and 31 queries). The correct comparison on the same 31 queries, toggling only freshness, was **93.5% → 96.8%**.
Notebook **72 cells** with new Part G4; error analysis 12 → **15 cases**; docs/05.

### 7.7. Phase 7 — Discovering and Fixing Test-Set Leakage (`c3da7c2`)

**Methodological error:** every hyper-parameter ($w_{nb}$, intent threshold, retrieval threshold, $\alpha$,
half-life) had been chosen by sweeping over `test_queries.json`, and the reported figures were measured on **that
same set**. The set was also tiny (31 retrieval queries, 26 intent queries): one error moved Recall@1 by 3.2 points.

Fixed with the protocol of Section 6.4 and the data of Section 3.4. **TEST report #1:**

| | Before (leaked) | True on TEST |
|---|---|---|
| Retrieval Recall@1 | 96.8% | **93.4%** (114/122) |
| Retrieval MRR | 0.984 | **0.950** |
| Intent accuracy | 88.5% | **61.5%** (16/26) |
| End-to-end: news → correct article | — | 77.9% (95/122) |
| End-to-end: out-of-scope → refusal | — | 83.3% (20/24) |

Two earlier conclusions were **refuted**: (1) "freshness improves retrieval" — on 112 DEV queries freshness does not
improve consistently; the earlier gain was small-sample noise; (2) the NB + cosine ensemble no longer wins — DEV
selected $w_{nb} = 1.0$. A self-inflicted bug was also fixed: the cache stored recency scores but the half-life was
not in the fingerprint, so changing the half-life was silently ignored; recency is now always recomputed.

### 7.8. Phase 8 — Regression Tests Catch Two Design Bugs (`c4a6a23`)

21 tests (listed in Appendix F), each tied to a real bug that had made the bot answer wrongly without any error. On
their first run after applying the DEV-tuned parameters they caught:

1. **Freshness silently became a filter against older articles.** The threshold applied to the **freshness-boosted**
   score; with a 3-day half-life, a 16-day-old article received almost no boost and struggled to pass:
   `"tin ve dao hai nam"` and `"cho t hỏi vụ hải nam vs"` went from correct to "not found". **Fix:** rank by the
   boosted score, **accept by plain cosine**; re-tuned on the cosine scale: **0.13** (DEV: 88.7% answerable, 100%
   out-of-scope blocked).
2. **Same question, different outcome depending on the branch.** The relaxed threshold for named categories
   existed only in the browsing branch; `"tin du lịch ninh bình"` had intent confidence 0.246 — just under 0.25 —
   so it took the retrieval branch (no relaxation) and failed. **Fix:** a single constant,
   `CATEGORY_SCOPED_THRESHOLD_FACTOR`, applied in both branches.

**TEST report #2:** Recall@1 93.4% (unchanged), MRR 0.950; component out-of-scope blocking 95.8% → **87.5%**
because the cosine-scale threshold let 3 "hard" queries through; end-to-end **77.0%** / **75.0%**. Both fixes came
from regression tests, not from test results.

### 7.9. Phase 9 — BM25, a Negative Result (`6587696`)

Implemented and verified BM25 (Section 5.10); `test_vectorizer.py` grew from 28 to **45** checks. On DEV
(freshness off) the best BM25 ($k_1 = 8$, $b = 0.9$) reached MRR 0.967 vs TF-IDF 0.965 — about one query. The
original rule "better on DEV ⇒ switch" selected BM25, which then **lost on TEST** (MRR 0.943 vs 0.950). The rule was
changed to the paired sign test: BM25 better on 3 queries, worse on 3, tied on 106, **p = 1.0** → keep TF-IDF.
**Disclosure:** the significance rule was added **after** seeing test report #3; under either rule the two methods
are not significantly different on test.

### 7.10. Phase 10 — Honest-Evaluation Documentation (`5320448`)

docs/06 records the new methodology; docs/03 and docs/05 gained **correction** boxes (their original content is kept
to preserve history); the README was switched to TEST figures with CIs. The notebook was rewritten to **79 cells**:
Part I "Honest evaluation", Part I2 (BM25), G4 using DEV, error analysis 15 → **19 cases**. Another wrong claim
written during this phase ("turning freshness off gives the highest DEV MRR") was corrected: the notebook output
showed 30 days / $\alpha$ = 0.2 at 0.969 > 0.965 — the correct statement is that freshness does not improve
consistently and all differences are within noise.

### 7.11. Phase 11 — The RAG Layer (`98fb144`)

Implemented `rag.py` (prompt v1, three backends, automatic faithfulness checks), `tests/test_rag.py` (9 tests,
including a backend that **raises if called**, proving the LLM never runs without evidence), a 26-cell Colab notebook
(detects Colab vs. local, dry-runs locally with the stub backend) and a bundling tool (`rag_bundle.zip`, without
the cache because pickles depend on library versions). The laptop has no NVIDIA GPU, so PhoGPT runs on Google Colab
(T4); the student ran the notebook manually and returned the results.

### 7.12. Phase 12 — RAG Run 1 (DEV) and Prompt v2 (`1743132`)

**Setup:** Colab T4; `transformers` failed to load → llama.cpp GGUF **Q4_K_M**; prompt v1; 25 DEV news questions,
4 conflict questions, 6 traps, 5 out-of-scope. **Reproducibility check:** local routing and top-1 articles matched
Colab on **36/36** questions.

**Results** (21 news questions reached RAG): only **3** A answers, **8** D answers, 0/21 correct citations, **0/5**
traps handled (the model answered "Đúng." — "Correct." — to false premises), conflict 3/3 following the newer
article, out-of-scope 5/5 never reached the LLM, latency 1.84 s (p90 4.9 s). The invented-number checker had its own
bug (counting list numbering "1.", "2." as invented numbers): 12 → **6** answers after the fix, all 6 genuine
fabrications (a fake date `06/30/2021`, the year 2016, the year 2022). Only 3/8 D answers had the error in a number;
5/8 were semantic errors (reversed causality, invented negation, merging two articles, inventing an answer to fit
the question when retrieval was wrong).

**Designing v2** from the observed failures (Section 5.12). Two **GPU-free** measurements:

- Premise guard false alarms on valid questions: DEV **0/93**, TEST **1/97** — the single firing was a question whose
  retrieval was already wrong, i.e. **0** false alarms on correctly retrieved questions.
- Applying v2 post-processing + guards to the very answers v1 generated: traps safe 0/5 → 5/5 (4 by design, 1 by
  luck), displayed answers with invented numbers 6/21 → 0/18, displayed D answers 8 → 5, C answers unchanged (3).

**A side discovery:** while measuring the guard, the DEV query "nhan vien openai tieu **7000** do..." was flagged
with «70». The cause was the elongation-collapsing regex `(.)\1{2,}`, which also collapsed **digits**:
`15.000 → 15.0`, `2000 → 20`, `1000 → 10`. Fixed to letters only, `([^\W\d_])\1{2,}`. Teencode test ERR 67.54% →
**67.84%**. The bug touched **1/195** DEV and **0/178** TEST queries; re-tuning on DEV produced **the same
parameters**, so no test figure changed.

### 7.13. Phase 13 — RAG Run 2 (DEV, v1 vs v2) (`fe10190`)

**Setup:** Q8_0; the same 40 DEV questions; every question through both v1 and v2; when the premise guard fires,
PhoGPT is **still called** to record what it would have said.

| Label (21 news questions) | v1 · Q4_K_M (run 1) | v1 · Q8_0 | **v2 · Q8_0** |
|---|---|---|---|
| A | 3 | 4 | **10** |
| B | 5 | 3 | 3 |
| C | 3 | 4 | 3 |
| D | 8 | 7 | **5** |
| E | 2 | 3 | **0** |

- **Quantization was not the cause:** v1 on Q4_K_M and Q8_0 shows almost the same error distribution.
- **v2 genuinely helps:** 6 questions reach A with v2 but not v1, 0 the other way (p ≈ 0.03) — but this is
  **optimistic**, since v2 was designed on these very questions.
- PhoGPT v2 **on its own** handled only 1/5 traps; with guards, 5/5.
- **Correction:** "vì sao kem Tràng Tiền phải đóng cửa" (why did Tràng Tiền ice cream close?) had been recorded as a
  trap, but the article **does** state a reason ("completed its historic mission" after 68 years). The mistake came
  from searching for strings instead of reading the article; the DEV trap set has 5 real traps. The TEST traps were
  therefore checked against the **full text**.

**Fixes after run 2 (post-processing only, prompt unchanged):** remove "Tin N:" labels (they made the number guard
reject a correct answer), strip ``` characters, rewrite the opening to "Theo các bài báo," (without lower-casing
proper nouns — "tP HCM" was a real bug), add the **echo** guard for keyword queries only (a first version applied
to all questions and wrongly blocked the yes/no answer "Vé tàu Cát Linh không tăng giá."), and compare numbers
with units by value. Re-scoring v2's outputs: 17/21 answers displayed, 3 D answers remaining.

**Preparing run 3:** 11 new test traps checked against full texts; **code frozen** at commit `fe10190`. While writing
the traps, **4/11 natural phrasings did not reach RAG** ("...đúng không", "...phải không" — "..., right?" — were
misrouted by the intent classifier); 2 reached it with a fixed-order list of rephrasings, 2 did not reach it with any
variant and were kept unchanged with a note. `test_rag.py`: 9 → 25 → **29** tests.

### 7.14. Phase 14 — RAG Run 3 on TEST and Notebook Completion (`be306ce`, `40c1792`)

Run **once** on TEST with the frozen code (results in Section 8.8.3). Re-running `rescore_rag_run.py` reproduced
**exactly** every guard decision made on Colab, confirming that the local code matched the code that ran. The
report notebook gained **Part K** (A–E table, trap table, three extractive-vs-RAG examples, all read directly from
saved data) and states explicitly that limitation 2 is **not** resolved by RAG; every cell has a fixed id;
**86 cells, 0 errors**.

### 7.15. Phase 15 — Documentation Audit

Before writing this report, all documentation and code comments were checked against the source code, the data and
the commit history. Fixes included: docs/02 still described the 0.8/0.2 ensemble and a 0.12 threshold; `config.py`
comments described a 7-day half-life and repeated the refuted claim "freshness improves retrieval"; the README still
said "Recall@3 100%" (a leaked-set figure) and "~150 patterns" (actually 164); docs/05 quoted a stale score; error
case 5 said "fixed" although DEV tuning had turned the ensemble off; `requirements.txt` did not list `scipy` or
`joblib`. The audit also found a **wrong claim** in docs/07: "tàu cát linh 15.000 đồng" failed to reach RAG not
because of the intent classifier (empty intent, confidence 0) but because of the **retrieval threshold** (cosine
0.096 < 0.13) — now corrected.

---

## 8. Experimental Results

### 8.1. Implementation Verification

| Test suite | Checks | Result |
|---|---|---|
| `test_vectorizer.py` — TF-IDF vs scikit-learn (5 configurations + edge cases) | 28 | pass, error ~1e-16 |
| `test_vectorizer.py` — BM25 vs naive implementation (4 $k_1,b$ pairs) + saturation property | 17 | pass |
| `test_chatbot.py` — regression tests | 21 | pass |
| `test_rag.py` — RAG layer, stub backend | 29 | pass |
| **Total** | **95** | **pass** |

### 8.2. Tuning on DEV

**Intent** (accuracy = fixed-intent queries labeled correctly and above threshold; safety = retrieval/out-of-scope
queries not answered with a canned response; each row shows the best threshold for that $w_{nb}$):

| $w_{nb}$ | threshold | accuracy | safety | mean |
|---|---|---|---|---|
| 0.0 | 0.35 | 91.1% | 61.3% | 76.2% |
| 0.2 | 0.40 | 80.0% | 76.0% | 78.0% |
| 0.4 | 0.35 | 84.4% | 80.7% | 82.6% |
| 0.6 | 0.30 | 86.7% | 82.0% | 84.3% |
| 0.8 | 0.30 | 82.2% | 91.3% | 86.8% |
| **1.0** | **0.25** | **82.2%** | **92.7%** | **87.4%** ← chosen |

**Ranking method** (freshness off): TF-IDF R@1 94.6%, MRR 0.965; best BM25 ($k_1$=8, $b$=0.9) R@1 95.5%, MRR 0.967.
BM25 MRR by $k_1$ ($b$ = 0.9): 0.942 (0.6) · 0.942 (0.9) · 0.944 (1.2) · 0.946 (1.5) · 0.954 (2.0) · 0.959 (3.0) ·
0.960 (5.0) · 0.967 (8.0). Sign test: 3 wins / 3 losses / 106 ties, p = 1.000 → **TF-IDF**.

**Freshness** (hard constraint: conflict case correct):

| $h$ (days) | $\alpha$ | R@1 | MRR | conflict |
|---|---|---|---|---|
| 3 | 0.0 | 94.6% | 0.965 | wrong |
| 3 | 0.4 | 93.8% | 0.960 | correct |
| **3** | **0.6** | **94.6%** | **0.964** | **correct** ← chosen |
| 3 | 0.8 | 94.6% | 0.963 | correct |
| 7 | 0.6 | 93.8% | 0.960 | correct |
| 14 | 0.6 | 93.8% | 0.958 | wrong |
| 14 | 1.0 | 93.8% | 0.958 | correct |
| 30 | 0.2 | 95.5% | 0.969 | wrong |
| 30 | 1.0 | 93.8% | 0.958 | wrong |

(Full 24-configuration grid: `data/eval/test_report_v3.txt`.) The highest-MRR configuration (30 days, $\alpha$ = 0.2)
does **not** handle the conflict case; every configuration that does is slightly lower (0.957–0.964). Freshness is
therefore a **trade-off** of about 0.001 MRR.

**Retrieval threshold.** DEV top-1 cosine distribution: in-scope hits (n = 106) min 0.064, p10 0.128, median 0.218;
out-of-scope (n = 28) median 0.072, p90 0.109, **max 0.125**.

| threshold | answerable | blocked | mean |
|---|---|---|---|
| 0.100 | 96.2% | 82.1% | 89.2% |
| 0.120 | 92.5% | 92.9% | 92.7% |
| **0.130** | **88.7%** | **100.0%** | ← chosen |
| 0.140 | 85.8% | 100.0% | 92.9% |
| 0.160 | 76.4% | 100.0% | 88.2% |

0.13 is the lowest value (on the 0.005-step grid) that blocks 100% of DEV out-of-scope queries.

**Frozen parameters** (`data/eval/tuned_params.json`): $w_{nb}$ = 1.0; intent threshold 0.25; TF-IDF ranking;
$h$ = 3 days; $\alpha$ = 0.6; retrieval threshold 0.13.

### 8.3. Results on TEST

| Category | Metric | Result | 95% Wilson CI |
|---|---|---|---|
| Retrieval | Recall@1 | **93.4%** (114/122) | 88–97% |
| Retrieval | Recall@3 | **95.9%** (117/122) | 91–98% |
| Retrieval | MRR | **0.950** | — |
| — accented | Recall@1 / MRR | 95.3% (82/86) / 0.967 | 89–98% |
| — unaccented | Recall@1 / MRR | 95.5% (21/22) / 0.960 | 78–99% |
| — teencode | Recall@1 / MRR | **78.6%** (11/14) / 0.832 | 52–92% |
| Out of scope (component) | blocked | 87.5% (21/24) | 69–96% |
| Intent | Accuracy | **61.5%** (16/26) | 43–78% |
| Intent | Macro-F1 | 0.712 | — |
| **End-to-end** | news → correct top article | **77.0%** (94/122) | 69–84% |
| **End-to-end** | out-of-scope → refusal | **75.0%** (18/24) | 55–88% |

**Analysis.**

- *Retrieval* is strong and consistent across accented and unaccented input; teencode is the weak spot
  (abbreviations such as `vc`, `ntn` are missing from the lexicon), but with only 14 queries the CI is wide.
- *Intent* is the biggest weakness. All 10 errors are **rejections below threshold** (confidence 0.00–0.25), all
  short colloquial inputs: "ừm", "đúng vậy", "thanks nhé", "ngu thế", "bot làm được những gì".
- *The component → end-to-end gap* is 16 points (93.4% → 77.0%). Of the 28 end-to-end failures, **18** have cosine
  below threshold (fallback), **7** are misrouted by the intent classifier, **3** return a wrong article. This is the
  deliberate trade-off of the 0.13 threshold (100% blocking on DEV, ~11% of answerable questions refused).
- *3 out-of-scope queries pass the retrieval threshold:* "kết quả xổ số miền bắc" (lottery results, 0.142, matching
  a cloud-hunting article about northern Vietnam), "top 10 truyện tranh hay nhất" (top 10 comics, 0.148, matching
  "Internet … top 10"), "bài văn tả con mèo lớp 3" (a 3rd-grade essay about a cat, 0.137). A further 3 are answered
  with a canned response by the intent classifier (e.g. "địa chỉ tiệm sửa laptop gần đây" → `tam_biet`).

### 8.4. BM25 vs TF-IDF on TEST

| | Recall@1 | MRR |
|---|---|---|
| TF-IDF | 93.4% (114/122, 88–97%) | 0.950 |
| BM25 ($k_1$=8, $b$=0.9) | 91.8% (112/122, 86–95%) | 0.943 |

**Explanation (a hypothesis supported by the data):** the index up-weights titles by **repeating** them three times —
a trick that only works while tf keeps increasing with repetitions. BM25's tf saturation cancels exactly that effect;
as saturation is weakened (larger $k_1$), BM25's MRR rises steadily and approaches TF-IDF's behavior. The principled
alternative is BM25F (per-field saturation followed by a weighted sum).

### 8.5. Teencode Normalization

| ViLexNorm split | Tokens (to fix) | Acc before | Acc after | **ERR** | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|
| dev | 10,003 (1,547) | 84.53% | 95.24% | 69.23% | 91.18% | 71.49% | 80.14% |
| **test** | 10,186 (1,642) | 83.88% | 94.82% | **67.84%** | 91.36% | 70.22% | 79.41% |

Precision well above recall is a **deliberate trade-off**: the three safety conditions make the lexicon conservative —
it misses rare forms but almost never corrupts a correct word. **Caveat:** ERR is computed only over positionally
alignable pairs (79.5%); pairs whose token count changes are excluded from both learning and evaluation, so the figure
over the full corpus would be lower.

### 8.6. Outdated News

With `conflict_case.json` and the current configuration:

| Question | Plain TF-IDF (old / new) | With freshness (new / old) | Via `bot.respond()` |
|---|---|---|---|
| giá vé tàu cát linh bao nhiêu | **0.4956 old** / 0.4098 | **0.6556 new** / 0.5328 | returns the new article |
| vé tàu cát linh có tăng giá không | 0.4126 / **0.4577 new** | **0.7323 new** / 0.4435 | returns the new article |
| tàu cát linh 15.000 đồng | **0.1087 old** / 0.0959 | **0.1534 new** / 0.1169 | "not found" (cosine 0.096 < 0.13) |
| hoãn tăng giá vé tàu | 0.1823 / **0.4582 new** | **0.7332 new** / 0.1960 | returns the new article |

Freshness ranking puts the newer article first for **all 4** phrasings (plain TF-IDF: 2/4). End to end, 3/4
phrasings receive the new article; the fourth is blocked by the acceptance threshold — safe, but unanswered.
(Notebook Part G4 rebuilds the two articles inline and therefore reports its own scores: 0.5269/0.4302 without and
0.6883/0.5664 with freshness — same conclusion.)

### 8.7. n-gram Generation

Data: 9,567 training sentences / 1,063 test sentences / 216,976 tokens.

| $n$ | Perplexity ($\lambda$ = 0.7) |
|---|---|
| 1 | 1,285.4 |
| **2** | **818.9** |
| 3 | 1,950.7 |
| 4 | 5,911.2 |

| $\lambda$ | $n$=2 | $n$=3 | $n$=4 |
|---|---|---|---|
| 0.3 | 715 | 844 | 1,173 |
| 0.5 | 716 | 1,113 | 2,109 |
| 0.7 | 819 | 1,951 | 5,911 |
| 0.9 | 1,282 | 7,624 | 63,318 |

1. **Perplexity increases with $n$** — counter-intuitive. Cause: data sparsity makes higher-order components almost
   always zero, and the interpolation multiplies by $(1-\lambda)$ at **each** back-off step (two steps at
   $\lambda$ = 0.7 multiply by 0.09). Verified: lowering $\lambda$ from 0.9 to 0.3 reduces $n$ = 4 perplexity by a
   factor of 54. At every $\lambda$, $n$ = 2 remains best.
2. **Large $n$ turns generation into copying:** at $n$ = 4 samples start with the same long string copied from the
   corpus ("du lịch phú quốc thành lập năm 2014 ở hàng châu, tập trung phát triển ôtô điện...") because most contexts
   occur only once.
3. **False at every $n$:** "du lịch phú quốc còn đang xây dựng các trung tâm điều trị ebola" (Phú Quốc tourism is
   building Ebola treatment centers).

Conclusion: at this data scale, n-gram generation **loses to retrieval on every criterion** (coherence, truthfulness,
attribution) — empirical grounds for the extractive architecture.

### 8.8. The RAG Layer with PhoGPT-4B-Chat

#### 8.8.1. Run 1 (DEV, Q4_K_M, prompt v1)

| Label (21 news questions) | Count | Example |
|---|---|---|
| A | 3 | "ten lua spectrum cua duc phong ve tinh" → complete answer, correct launch time |
| B | 5 | Blue Origin: correct content + a line `[2] (đăng 06/30/2021...)` copied from the context format |
| C | 3 | "Huawei Mate XT2 gập ba" → "Huawei Mate XT2 gập ba." |
| D | 8 | "The Panama drought … caused the stoppage at the Strait of Hormuz" (source: the **conflict** caused it) |
| E | 2 | "bán kết Mỹ Mở rộng nữ có bốn hạt giống" → "not mentioned" |

Traps: 0/5 — "đảo Hải Nam miễn visa từ năm 2015 phải không" (Hainan visa-free since 2015, right?) → "Đúng."; "profit
of the Bến Thành–Suối Tiên metro in 2020" → "27 billion VND" (the 2026–2030 target figure).

#### 8.8.2. Run 2 (DEV, Q8_0, v1 vs v2)

See the table in Section 7.13. The remaining D answers of v2 show that errors **do not need numbers**: "imports
**under** 100,000 VND may no longer be tax-exempt" (source: the threshold is lowered to 100,000 → goods **above**
100,000 lose the exemption; the model took the question's wording and inverted the answer); "... Wimbledon 2009,
which was also the year Sabalenka reached this round" (the second half is invented). The premise guard caught 4/4
numeric traps; the invented-number guard caught "Chornobyl in 196" (1986) and "about 1.5 million euros" (entirely
invented).

#### 8.8.3. Run 3 — TEST, Once, Frozen Code

**Setup:** Q8_0; 40 TEST news questions (33 reached RAG, **all 33 retrieved the correct article**), 4 conflict
questions, 11 new traps (9 reached RAG), 24 out-of-scope; every question through v1 and v2.

**Answer quality (33 news questions, before guards):**

| Label | v1 | **v2** |
|---|---|---|
| A — correct, answers | 9 | **19** |
| B — correct, badly formatted | 10 | **2** |
| C — non-answer | 8 | 8 |
| **D — false information** | **3** | **4** |
| E — wrong refusal | 3 | **0** |

Paired comparison on label A: **11** questions favor v2, **1** favors v1 → sign test **p = 0.0063**. These questions
were never seen while designing the prompt, so "v2 is more useful than v1" holds. **But false information did not
decrease (3 → 4).** v2's D answers:

| Question | v2 wrote | Source |
|---|---|---|
| Messi nominated for the 2026 Ballon d'Or | "nominated **because** he was not on the 2024 and 2025 lists" | reversed causality |
| preparing for old age from 40 | "people whose **average age is 40**" | "people **from** age 40" |
| French tourist died in Death Valley | attributes "the highest temperature ever recorded" to this case | 46.7 °C that day; 56.7 °C is the region's record |
| how many colors does the iPhone 18 Pro Max have | "light blue", "dark red" | "silver", "burgundy red" |

The iPhone case: **both v1 and v2** invented two colors because the 4 sentences put into the context **did not
contain** the sentence listing the colors — the failure starts in the hand-written sentence selector; the LLM only
filled the gap.

**Traps (11 new questions):** v1 and v2 tie at **6 safe / 5 wrong**, for opposite reasons.

| Trap type | Count | v1 | v2 |
|---|---|---|---|
| false premise with a number | 2 | 0 safe ("Đúng.", invented "2031") | **2 safe** (premise guard) |
| wrong number that appears elsewhere | 1 | wrong | **wrong** (the predicted blind spot) |
| false premise **without numbers**, reaching RAG | 4 | 2 safe (refused/evaded by chance) | **0 safe** |
| missing detail | 2 | 2 safe | 2 safe (1 via the number guard — the LLM invented "3.5 billion USD"; 1 via empty output) |
| not reaching RAG (Harvard, Tim Cook) | 2 | safe | safe |

v2 flatly asserts what the article **denies**: "Tesla's Optimus robot walked off the production line" (the article
says Tesla **cannot yet** mass-produce Optimus; the robot that walked off the line is Xpeng's Iron), "twin sisters"
(they are brothers), "Google bought Hugging Face" (Nvidia did). The v2 prompt made the model **more assertive**: much
better on real questions, but also quicker to agree with false premises.

**Other groups:** conflict — v2 **3/3** following the newer article, v1 2/3 (one self-contradictory answer);
out-of-scope — 21/24 never reached the LLM, 3 passed the retrieval threshold (v2: 1 correct refusal, 1 invented "the
top 10 comics are works rated highly for content and form...", 1 blocked by the premise guard because of the "3" in
"lớp 3"); guards — premise **0 false alarms** on 33 news questions, echo fired 4 times and all 4 were justified, the
number guard fired once because of rounding ("2 km" vs "2.1 km"); citations — v1 had 3 answers with valid citations
but 7 with wrong indices, v2 does not ask for citations; latency — v2 1.18 s (p90 3.07 s), v1 1.49 s (p90 4.00 s).

**Summary of the three runs:**

| | Run 1 (DEV) | Run 2 (DEV) | **Run 3 (TEST)** |
|---|---|---|---|
| Model | Q4_K_M | Q8_0 | Q8_0 |
| News questions reaching RAG | 21 | 21 | 33 |
| A: v1 → v2 | 3 → — | 4 → 10 | **9 → 19** |
| D: v1 → v2 | 8 → — | 7 → 5 | **3 → 4** |
| E: v1 → v2 | 2 → — | 3 → 0 | **3 → 0** |
| Traps safe (v2 with guards) | — | 5/5 | **6/11** |
| Conflict follows newer article | 3/3 (v1) | 3/3 (v2) | **3/3 (v2)** |

### 8.9. Performance

| Component | Value (laptop CPU, 381 articles) |
|---|---|
| Index build, no cache | ~24 s |
| Bot startup, cached | ~1.1 s |
| Answer one question (core) | ~21–24 ms |
| Vocabulary: main index / syllable index / intent | 105,020 / 87,294 / 493 |
| RAG v2 on Colab T4 (Q8_0) | 1.18 s mean, 3.07 s p90 |

---

## 9. Discussion and Error Analysis

### 9.1. The 19 Error Cases (Notebook Part J)

| # | Layer | Error | Status |
|---|---|---|---|
| 1 | Regex | `\b` after `%` missed "3,5%" | fixed |
| 2 | Segmentation | unaccented input mis-segmented | fixed (syllable index) |
| 3 | NER | "Công ty ABC" tagged LOC; phone number tagged LOC | mitigated (regex in parallel) |
| 4 | Stopwords | "AI" (English acronym) removed because "ai" is a stopword | open |
| 5 | Intent | short "cảm ơn nhé" below threshold; the ensemble fixed it, but DEV tuning chose plain NB | **open** |
| 6 | Retrieval | "tin du lịch ninh bình" below the global threshold | fixed (relaxed threshold with category) |
| 7 | Intent | `huong_dan` / `liet_ke_chuyen_muc` overlap | open |
| 8 | Retrieval / teencode | correct normalization but diluted vector | fixed (frame-word filtering) |
| 9 | Architecture | two drifting frame-word lists | fixed (single source) |
| 10 | Segmentation | "Sức khỏe" segmented differently by context | fixed (syllable matching) |
| 11 | n-gram generation | perplexity increases with $n$ | explained |
| 12 | Teencode | "t" always → "tôi", even for "tao" | open |
| 13 | Ranking | outdated article returned | fixed (freshness) |
| 14 | Evaluation method | tuning set lacked unaccented/teencode queries | fixed |
| 15 | Normalization / routing | bot said "goodbye" to a weather question | fixed (accent frame preservation) |
| 16 | Evaluation method | test-set leakage | fixed (DEV/TEST) |
| 17 | Ranking / threshold | freshness filtered out older articles | fixed (ranking ≠ acceptance) |
| 18 | Routing | outcome depended on branch at intent 0.246 | fixed (one constant) |
| 19 | Model selection rule | chose BM25 for +0.002 MRR | fixed (sign test) |

### 9.2. Lessons Learned

1. **Measure before fixing.** The hypothesis "use a top-1/top-2 margin instead of an absolute threshold" sounded
   reasonable but was rejected by measurement: the ratios for in-scope (1.03×–8.62×) and out-of-scope
   (1.04×–1.59×) queries overlap heavily.
2. **Evaluation data must match the real input distribution** — extending system capabilities without extending
   the evaluation set biases every subsequently tuned parameter (case 14).
3. **Two different questions need two different measures** — "which article first" vs. "enough evidence to answer"
   (case 17).
4. **Each business rule lives in exactly one place** (cases 9, 18).
5. **A preprocessing step must not silently change the signal used to choose a branch** (case 15).
6. **A more complex method must win with statistical significance** (case 19).
7. **Retrieving the right article is not enough** — the right sentence must be passed on; and when retrieval is
   wrong, RAG **hides** the error behind a fluent answer, whereas the extractive bot **exposes** it to the reader.
8. **Negative results are valuable** — the three negative experiments (n-gram, BM25, RAG) each answered a design
   question with data instead of assumptions.

### 9.3. Threats to Validity

- **The query author is also the system builder**, so queries may share the system's blind spots. Mitigated by
  writing from article descriptions and paraphrasing, but queries written by others would be better.
- **Small samples:** 26 intent queries (CI 43–78%), 14 teencode queries, 33 RAG questions on test, 11 traps. RAG
  figures should be read as **error patterns**, not as statistically strong accuracies.
- **A single grader, and an AI tool,** for RAG answers (Section 12); the grader knew which answer was v1/v2.
- **The TEST split has been used several times** (Appendix C). Chatbot parameters did not change after use #3;
  later uses only reported or recomputed with frozen parameters, but further improvements need a fresh test set.
- **Answer-cleaning patterns** were derived from DEV outputs, so measurements on those DEV answers are optimistic;
  run 3 on TEST measures unseen outputs.
- **One news source (VnExpress), 381 articles:** results should not be generalized to other styles or domains.

---

## 10. Limitations and Future Work

### 10.1. Limitations

1. **No synonym understanding** — TF-IDF matches surface forms ("xe hơi" does not find "ô tô").
2. **The core is 100% extractive**, without reasoning or paraphrasing; the RAG layer paraphrases but is not more
   truthful.
3. **Static knowledge base**, updated by re-crawling and rebuilding the index (every term's IDF changes).
4. **The intent classifier is the weakest component** (61.5%): short colloquial inputs fall below threshold; two
   intents overlap; confirmation questions ("..., right?") are misrouted.
5. **Reference resolution anchors only to the latest turn** ("the second article" is unresolved).
6. **No detection of contradictions between articles**; freshness only breaks ties, so a much more relevant old
   article still wins.
7. **A 16-point component → end-to-end gap** caused by the acceptance threshold.
8. **The sentence selector** can miss the sentence containing the answer.
9. **Context-free teencode normalization** (recall 70.2%; ambiguous "t").
10. **RAG guards only catch errors involving numbers/codes**; false premises without numbers pass completely.

### 10.2. Future Work

| Direction | Technique | Expected benefit |
|---|---|---|
| Intent classifier | More patterns for short and confirmation questions; probability calibration instead of a hard threshold; measure on a fresh test set | Fix the biggest weakness |
| Sentence selection | More sentences / prefer sentences containing the question's entities and numbers | Fix the "iPhone colors" case for both bot and RAG |
| Synonyms | Embeddings (Word2Vec/PhoBERT) combined with TF-IDF | Fix limitation 1 |
| BM25F | Per-field tf saturation | Give BM25 a real chance, replacing title repetition |
| Same-event detection | Cluster similar articles, warn "a newer article exists" | Fix limitation 6 |
| Entailment checking for RAG | NLI between generated text and sources | Catch number-free false premises |
| Contextual normalization | seq2seq (BARTpho) | Higher recall, resolve ambiguous "t" |
| Evaluation | Queries written by others; multiple graders for RAG | Reduce bias |

---

## 11. Conclusion

The project built a Vietnamese news question-answering chatbot whose core is **implemented from scratch and
verified**, handles unaccented input, teencode and outdated news, and runs instantly on a CPU. On a held-out TEST
split, retrieval reaches Recall@1 **93.4%** and MRR **0.950**; through the full system, **77.0%** of news questions
receive the correct article. The intent classifier (61.5%) is the main weakness, and its cause is clearly identified.

The project's main value lies not in a high number but in **how the numbers were produced**: detecting and fixing
test-set leakage, accepting lower but honest figures, using confidence intervals and significance tests for every
method-selection decision, logging every use of the test split, and reporting three negative results in full. The
RAG experiment — three real runs, a trap set frozen in advance, code frozen before the final measurement — yields a
clear, data-backed conclusion: the generative model makes answers **more natural** but **not more truthful** than the
extractive system. Because a news chatbot must put truthfulness before fluency, the submitted product is the extractive
chatbot; RAG remains an optional extension, disabled by default.

---

## 12. Statement on the Use of AI Tools

This project was carried out with the assistance of an AI coding assistant (Claude, by Anthropic), which wrote and
revised code, ran experiments, analyzed results and drafted documentation, under the direction, review and decisions of
the student (choice of topic and approach, requiring a local virtual environment, deciding to run PhoGPT manually on
Colab rather than automating it, running the Colab notebooks, reviewing and pushing the code). In particular, the
**manual grades for RAG answers** (Sections 6.5, 8.8) were produced by the AI assistant by checking each claim against the
full source article; the grade files are stored publicly in the repository so they can be re-checked, and they should be
verified by an independent grader before being treated as definitive.

---

## 13. References

1. Robertson, S., & Zaragoza, H. (2009). *The Probabilistic Relevance Framework: BM25 and Beyond*. Foundations and
   Trends in Information Retrieval, 3(4), 333–389.
2. Manning, C. D., Raghavan, P., & Schütze, H. (2008). *Introduction to Information Retrieval*. Cambridge University
   Press.
3. Jurafsky, D., & Martin, J. H. *Speech and Language Processing* (3rd ed. draft) — chapters on n-gram language
   models, Naive Bayes and perplexity.
4. Nguyen, T.-N., Le, T.-P., & Nguyen, K. V. (2024). *ViLexNorm: A Lexical Normalization Corpus for Vietnamese Social
   Media Text*. EACL 2024. <https://github.com/ngxtnhi/ViLexNorm>
5. Nguyen, D. Q., et al. (2023). *PhoGPT: Generative Pre-training for Vietnamese*. arXiv:2311.02945.
   <https://github.com/VinAIResearch/PhoGPT>
6. Lewis, P., et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks*. NeurIPS 2020.
7. Wilson, E. B. (1927). *Probable Inference, the Law of Succession, and Statistical Inference*. Journal of the
   American Statistical Association, 22(158), 209–212.
8. Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12, 2825–2830.
9. underthesea — Vietnamese NLP Toolkit. <https://github.com/undertheseanlp/underthesea>
10. stopwords/vietnamese-stopwords. <https://github.com/stopwords/vietnamese-stopwords>
11. llama.cpp / llama-cpp-python and the GGUF release `vinai/PhoGPT-4B-Chat-gguf` on Hugging Face.
12. The reference projects in Section 2.8 (Dec1mo, undertheseanlp/chatbot, heraclex12, sushant097, YUSANITY).
13. VnExpress (<https://vnexpress.net>) — source of the articles, collected for educational purposes.
14. Course material: Lab 01 (first NLP pipeline), Lab 02 (data acquisition), Lab 03 (Vietnamese preprocessing), Lab 04
    (BoW, n-grams, TF-IDF, cosine similarity).

---

## 14. Appendices

### Appendix A — Current Hyper-Parameters

| Constant | Value | Source |
|---|---|---|
| `RETRIEVAL_THRESHOLD` | 0.13 (plain cosine) | tuned on DEV |
| `INTENT_THRESHOLD` | 0.25 | tuned on DEV |
| `INTENT_W_NB` | 1.0 (plain NB) | tuned on DEV |
| `CATEGORY_SCOPED_THRESHOLD_FACTOR` | 0.6 | set by hand (too few category queries in DEV) |
| `FRESHNESS_ALPHA` | 0.6 | tuned on DEV |
| `FRESHNESS_HALFLIFE_DAYS` | 3.0 | tuned on DEV |
| `RANKING_METHOD` | `tfidf` | sign test on DEV |
| `BM25_K1`, `BM25_B` | 8.0; 0.9 | best BM25 configuration on DEV |
| `TOP_K` | 3 | — |
| `TFIDF_NGRAM_RANGE`, `MIN_DF`, `MAX_DF` | (1,2); 1; 0.85 | — |
| `TITLE_WEIGHT`, `DESC_WEIGHT` | 3; 2 | — |
| NB $\alpha$ | 0.3 | — |
| Teencode: `min_count`, `min_change_rate`, `min_dominance` | 4; 0.5; 0.5 | — |
| RAG: articles / sentences per article / characters per article | 3; 4; 900 | — |
| RAG: max new tokens v1 / v2 | 256; 160 | — |

### Appendix B — RAG Prompts

Common template (PhoGPT-4B-Chat): `### Câu hỏi: {instruction}\n### Trả lời:`

**v1** (Vietnamese original; English gloss below):

```text
Bạn là trợ lý tin tức. Hãy trả lời câu hỏi CHỈ dựa trên các bài báo được cung cấp dưới đây.

Quy tắc:
1. Trả lời ngắn gọn bằng tiếng Việt tự nhiên, từ 2 đến 4 câu.
2. Chỉ dùng thông tin có trong các bài báo. Không thêm kiến thức bên ngoài, không đoán.
3. Ghi nguồn bằng số trong ngoặc vuông ngay sau thông tin, ví dụ [1] hoặc [2].
4. Nếu các bài báo mâu thuẫn nhau, hãy ưu tiên bài có ngày đăng MỚI HƠN và nói rõ điều đó.
5. Nếu các bài báo không có thông tin để trả lời, hãy trả lời đúng một câu: "Các bài báo hiện có không đề cập đến điều này."
6. Nếu câu hỏi chứa giả định sai so với bài báo, hãy chỉ ra điều đó.

Các bài báo:
{context}

Câu hỏi: {question}
```

*Gloss:* "You are a news assistant. Answer ONLY from the articles below. Rules: (1) answer briefly in natural
Vietnamese, 2–4 sentences; (2) use only information in the articles, no outside knowledge, no guessing; (3) cite
sources with bracketed numbers such as [1]; (4) if articles conflict, prefer the NEWER one and say so; (5) if the
articles lack the answer, reply with exactly one sentence: 'The available articles do not mention this.'; (6) if the
question contains a false premise, point it out."

**v2:**

```text
Đọc các đoạn tin dưới đây rồi trả lời câu hỏi ở cuối bằng 2 đến 3 câu tiếng Việt tự nhiên, chỉ dùng thông tin có trong các đoạn tin. Nếu các đoạn tin không có câu trả lời, chỉ viết: "Các bài báo hiện có không đề cập đến điều này." Nếu hai tin mâu thuẫn nhau, dùng tin có ngày đăng mới hơn.

{context}

Câu hỏi: {question}
```

*Gloss:* "Read the news excerpts below and answer the question at the end in 2–3 natural Vietnamese sentences, using
only information in the excerpts. If the excerpts do not contain the answer, write only: 'The available articles do
not mention this.' If two items conflict, use the one with the newer date."

### Appendix C — Log of TEST-Split Uses

| Use | Activity | Changed anything based on test results? |
|---|---|---|
| 1 | First report after the DEV/TEST split (`test_report_v1.txt`) | No |
| 2 | After the two fixes found by regression tests (`_v2.txt`) | No |
| 3 | After adding BM25 (`_v3.txt`) | **Yes** — method-selection rule changed to the sign test |
| 4 | Premise-guard false-alarm measurement (no LLM) | No |
| 5 | Re-measurement after fixing unit-aware number matching | No |
| 6 | RAG run 3 (code frozen at `fe10190`) | No — reporting only |

In addition, every re-execution of the report notebook re-runs `evaluate.py` in Part E with the frozen parameters;
those recomputations produce an identical `test_results.json` and led to no decisions.

### Appendix D — Reproduction

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

python tests/test_vectorizer.py     # 45 TF-IDF/BM25 checks
python tests/test_chatbot.py        # 21 regression tests
python tests/test_rag.py            # 29 RAG tests (no GPU)
python src/evaluate.py              # tune on DEV, report on TEST
python src/normalizer.py            # learn + evaluate the teencode lexicon
python src/generator.py             # n-gram experiment
python tools/rescore_rag_run.py data/eval/rag/run3_test_results.json   # re-score RAG run 3

python src/cli.py                   # terminal chat
python src/api.py                   # web demo at http://127.0.0.1:8000
```

RAG: run `python tools/build_rag_notebook.py` and `python tools/make_colab_bundle.py`, open
`dist/RAG_PhoGPT_Colab.ipynb` on Google Colab (T4 GPU) → Run all → upload `dist/rag_bundle.zip`.

### Appendix E — Result Files

| File | Content |
|---|---|
| `data/eval/tuned_params.json` | parameters tuned on DEV |
| `data/eval/test_results.json` | TEST figures |
| `data/eval/test_report_v1..v3.txt` | full report for each test-set view |
| `data/eval/rag/run1_results.json`, `run1_samples.csv` | RAG run 1 |
| `data/eval/rag/run2_results.json`, `run2_samples.csv` | RAG run 2 |
| `data/eval/rag/run3_test_results.json`, `run3_test_samples.csv` | RAG run 3 (TEST) |
| `data/eval/rag/run*_rescored.csv` | re-scored with the current code |
| `data/eval/rag/run3_annotation.csv`, `run3_trap_annotation.csv` | run-3 manual grades |
| `data/eval/rag/dev_traps.json`, `test_traps.json` | trap sets |
| `notebooks/FinalProject_Chatbot_23IT036.ipynb` | report notebook (86 cells, executed) |
| `notebooks/RAG_PhoGPT_Colab.ipynb` | Colab notebook for RAG |
| `docs/01..07` | technical documents by topic |

### Appendix F — The 21 Regression Tests

`regex_bat_duoc_phan_tram`, `nhan_dien_cau_khong_dau`, `cau_khong_dau_khong_qua_word_tokenize`,
`loc_tu_khung_khong_lam_rong_query`, `tu_khung_chi_co_mot_nguon_duy_nhat`, `teencode_duoc_chuan_hoa`,
`chuan_hoa_giu_he_quy_chieu_khong_dau`, `hoi_thoi_tiet_khong_bi_chao_tam_biet`,
`so_khop_ten_chuyen_muc_theo_am_tiet`, `cau_chi_neu_chuyen_muc_thi_duyet_muc`,
`chuyen_muc_kem_chu_de_thi_tim_trong_muc`, `truy_hoi_cau_khong_dau`, `truy_hoi_cau_teencode`,
`cau_ngoai_pham_vi_bi_tu_choi`, `snippet_khong_lap_cau`, `tham_chieu_bai_do_qua_nhieu_luot`,
`dau_vao_rong_khong_lam_sap_bot`, `doc_duoc_ngay_vnexpress`, `duyet_muc_sap_theo_ngay_moi_nhat`,
`tin_moi_phu_dinh_tin_cu`, `cache_tinh_lai_do_moi_theo_nua_chu_ky`.

(Test names are in Vietnamese; each corresponds to a real bug described in Sections 7 and 9.1 — for example
`hoi_thoi_tiet_khong_bi_chao_tam_biet` = "a weather question is not answered with goodbye", and
`tin_moi_phu_dinh_tin_cu` = "newer news overrides older news".)

---

### Appendix G — Post-Submission Improvements (branch `cap-nhat-du-lieu`)

The submitted version is frozen at tag `nop-bai` (381-article corpus). The corpus
was then extended by the daily crawl procedure of `docs/08`, and **the first real
crawl (14 Sep 2026, +151 articles, 532 total)** exposed two defects that a static
corpus could not reveal. Both were fixed on branch `cap-nhat-du-lieu`; the full
reasoning and measurements are in `docs/09-cai-thien-mo-hinh.md`.

**G.1. The freshness reference point drifted with the corpus.** Recency was
computed relative to the newest publication date **in the whole corpus**. Adding
151 unrelated articles moved that reference from 10 Sep to 14 Sep, aged both
articles of the "newer article contradicts older article" test case, and the bot
went back to returning the **outdated** article (0.5104 vs 0.5086) — precisely
the failure that Section 5.5 exists to prevent. This is a design defect, not a
parameter defect: a full grid sweep (7 half-lives × 11 values of α) shows that
**no configuration** of the corpus-relative reference both handles the conflict
case and still handles it after an unrelated future-dated article is added. The
fix takes the reference to be the newest date among the articles **actually
competing** for that question (those satisfying `score × (1 + α) ≥ best score`).
The resulting property is an invariant: adding unrelated articles cannot change
the ranking. Re-tuned on DEV, α dropped from 0.6 to 0.3.

**G.2. Misspelled queries were not recognised.** A user typing "thám hiểm Sơn
Dòng" (one missing *o*, and the stroke of *Đ* omitted) was refused, while "Sơn
Đoòng" was answered correctly — cosine was only 0.086, below the 0.13 threshold.
Lowering the threshold is not the fix: at 0.08, correct refusal of out-of-scope
questions falls from 100% to 58%. The fix adds a **third index of character
3-grams over accent-folded titles**, used as a **fallback path** that runs only
when the main path refuses, with acceptance scored as word-level cosine plus
character-level cosine. The design was chosen empirically on DEV (12 variants:
indexed field × n × acceptance rule); "title only, n = 3, additive score"
rescued the most questions while answering none of them incorrectly.

**G.3. A lesson about evaluation data.** After enabling the fallback, a
regression test caught an **out-of-scope** question written without diacritics
being answered, even though DEV tuning reported zero leaks. The cause: DEV's
out-of-scope set contained only **accented** questions, whereas the character
n-gram index always operates on accent-folded text — folding makes distinct
sentences look more alike. The fix was to **extend the evaluation data**
(generating unaccented variants of DEV's out-of-scope questions, 28 → 36) rather
than to bend the threshold around that one case. Variants are generated from DEV
questions only, never from TEST, so the measurement stays sound.

**G.4. Results.** Measured on the same 532-article corpus and the same TEST
questions, through `bot.respond()` (`tools/compare_improvements.py`):

| TEST metric | Before (as submitted) | After |
|---|---|---|
| Recall@1 (component) | 109/122 | **112/122** |
| MRR | 0.927 | **0.940** |
| Ordinary questions → correct article | 90/122 | **101/122** |
| Misspelled questions → correct article | 74/122 | **86/122** |
| Newer article contradicting older one | wrong | **correct** |
| Same case after adding a "future" article | wrong | **correct** |
| Out-of-scope questions refused | 18/24 | 18/24 |

The cost must be stated plainly: the fallback converts some refusals into
answers, so the number of **incorrect** answers rises (7 → 8 on the misspelled
set, 3 → 5 on the clean set). In exchange it leaks no additional out-of-scope
questions, and every answer produced through this path carries an explicit "you
may have mistyped" note. The regression suite grew from 21 to 26 cases, two of
which guard the invariants described above.
