"""
config.py — Cấu hình tập trung cho chatbot.

Mọi đường dẫn và siêu tham số (hyper-parameter) đều khai báo ở đây để notebook,
CLI và web API dùng chung một nguồn sự thật duy nhất.
"""

from pathlib import Path

# ---------------------------------------------------------------- ĐƯỜNG DẪN --
SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent

DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
INTENTS_DIR = DATA_DIR / "intents"
RESOURCES_DIR = DATA_DIR / "resources"
MODELS_DIR = PROJECT_DIR / "models"

CORPUS_RAW_PATH = RAW_DIR / "corpus_raw.csv"
CORPUS_PROCESSED_PATH = PROCESSED_DIR / "corpus_processed.csv"
KB_PATH = PROCESSED_DIR / "knowledge_base.csv"
INTENTS_PATH = INTENTS_DIR / "intents_vi.json"
STOPWORDS_PATH = RESOURCES_DIR / "vietnamese-stopwords.txt"

for _d in (PROCESSED_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------- CONFIG PREPROCESSING --
# Giữ đúng cấu trúc config của Lab 03 để pipeline có thể tái sử dụng nguyên vẹn.
DEFAULT_CONFIG = {
    "word_segment": True,
    "lowercase": True,
    "remove_punctuation": True,
    "remove_stopwords": False,
    "remove_numbers": False,
}

# Config dùng cho INTENT CLASSIFICATION.
# Câu hỏi của người dùng rất ngắn (5-10 token) nên GIỮ stopwords:
# "cho tôi", "là gì", "có không" chính là tín hiệu phân biệt intent.
CONFIG_INTENT = {
    "word_segment": True,
    "lowercase": True,
    "remove_punctuation": True,
    "remove_stopwords": False,
    "remove_numbers": False,
}

# Config dùng cho RETRIEVAL trên corpus tin tức.
# Văn bản dài, cần loại stopwords để TF-IDF tập trung vào từ nội dung.
CONFIG_RETRIEVAL = {
    "word_segment": True,
    "lowercase": True,
    "remove_punctuation": True,
    "remove_stopwords": True,
    "remove_numbers": False,
}


# ------------------------------------------------------------ SIÊU THAM SỐ --
# ---- Các giá trị dưới đây được DÒ TRÊN TẬP DEV (data/eval/dev.json) bằng
# `python src/evaluate.py`, rồi báo cáo MỘT LẦN trên tập TEST chưa từng dùng
# để dò (data/eval/test.json). Kết quả dò mới nhất: data/eval/tuned_params.json.
#
# LƯU Ý LỊCH SỬ: các bảng số trong chú thích bên dưới là từ lần dò CŨ, khi
# tham số còn được dò và báo cáo trên cùng một tập (rò rỉ tập test). Chúng giữ
# lại để đối chiếu; giá trị hiện hành là giá trị gán ở mỗi dòng.

# Ngưỡng để chấp nhận câu trả lời từ retriever — áp lên COSINE THUẦN.
#
# Độ mới chỉ dùng để XẾP HẠNG, không quyết định có trả lời hay không (xem
# retriever.search). Giá trị hiện hành 0.13 dò trên dev: trả lời được 88.7%,
# chặn đúng 100% câu ngoài phạm vi.
#
# (Lịch sử) Trước đây ngưỡng áp lên điểm ĐÃ NHÂN hệ số độ mới, không phải cosine thuần.
# Hệ số (1 + FRESHNESS_ALPHA * recency) thổi mọi điểm lên tối đa 1,6 lần, nên
# ngưỡng phải được dò LẠI sau khi bật độ mới: 0.12 -> 0.18.
#
# Bảng dò (30 truy vấn trong phạm vi + 12 ngoài phạm vi, gồm cả câu không dấu
# và teencode — ngưỡng phải dò trên đúng phân bố truy vấn mà bot thực sự nhận):
#   0.120 -> trả lời được 90.0%, chặn đúng  91.7%
#   0.150 -> trả lời được 90.0%, chặn đúng  91.7%
#   0.155 -> trả lời được 90.0%, chặn đúng 100.0%   <- chọn
#   0.180 -> trả lời được 86.7%, chặn đúng 100.0%   (bắt đầu bỏ sót)
#
# Điểm cao nhất của truy vấn NGOÀI phạm vi là 0.151, nên 0.155 là ngưỡng thấp
# nhất còn chặn được 100% — tối đa hóa số câu trả lời được mà vẫn không đoán bừa.
RETRIEVAL_THRESHOLD = 0.13

# Ngưỡng điểm để chấp nhận nhãn intent từ classifier.
# Dò lưới (w_nb x threshold) trên DEV: w_nb=1.0, threshold=0.25 cho điểm cân
# bằng tốt nhất — accuracy 82.2%, safety 92.7% (data/eval/test_report_v3.txt).
# Trên TEST: accuracy 61.5% (16/26) — điểm yếu lớn nhất của hệ thống (docs/06).
# (Lịch sử: lần dò cũ bị rò rỉ chọn w_nb=0.8 và báo cáo 88.5%.)
INTENT_THRESHOLD = 0.25

# Khi người dùng đã tự nêu chuyên mục ("tin du lịch ninh bình"), tập ứng viên
# co lại còn vài chục bài cùng chủ đề, nên một cosine thấp hơn vẫn là bằng
# chứng đủ mạnh. Ngưỡng trong trường hợp này = RETRIEVAL_THRESHOLD x hệ số này.
# Hằng số đặt tay (chưa dò trên dev vì dev có quá ít câu nêu chuyên mục).
CATEGORY_SCOPED_THRESHOLD_FACTOR = 0.6

# Trọng số của Naive Bayes trong ensemble; phần còn lại (1 - w_nb) là tín hiệu
# cosine tới pattern gần nhất. Xem giải thích trong intent_classifier.py.
# Giá trị 1.0 (dò trên dev) nghĩa là hiện dùng NAIVE BAYES THUẦN — tín hiệu
# cosine vẫn được cài đặt nhưng không đóng góp vào điểm.
INTENT_W_NB = 1.0

# Số document trả về tối đa cho một truy vấn.
TOP_K = 3

# Tham số TF-IDF.
TFIDF_MIN_DF = 1          # bỏ term xuất hiện trong ít hơn min_df document
TFIDF_MAX_DF = 0.85       # bỏ term xuất hiện trong hơn 85% document (quá phổ biến)
TFIDF_NGRAM_RANGE = (1, 2)  # unigram + bigram

# Số lượt hội thoại giữ trong bộ nhớ ngắn hạn của dialogue manager.
HISTORY_MAXLEN = 10

RANDOM_SEED = 42


# ------------------------------------------------- TỪ KHUNG CÂU HỎI --
# Những từ chỉ đóng vai trò "khung" của câu hỏi hội thoại, không mang nội dung
# cần tìm. Chúng KHÔNG nằm trong stopword list chuẩn vì trong văn bản thường
# chúng vẫn là từ nội dung ("tin" trong "bản tin", "biết" trong "hiểu biết").
#
# Vì sao phải loại riêng cho truy hồi: vector query được chuẩn hóa L2, nên mỗi
# token thừa đều chia bớt trọng số của token thực sự quan trọng. Đo được:
#   "giá iphone"                         -> top-1 = 0.167  (đạt ngưỡng)
#   "biết gì về vụ iphone không bạn"     -> top-1 = 0.100  (trượt ngưỡng)
# Cùng một ý định, chỉ khác cách diễn đạt hội thoại. Việc chuẩn hóa teencode
# còn làm nặng thêm vì nó BUNG các từ viết tắt thành từ đầy đủ
# ("k" -> "không", "b" -> "bạn"), tức là thêm token khung vào câu.
QUERY_FRAME_WORDS = {
    "tin", "tức", "tin_tức", "bài", "bài_viết", "viết", "báo", "bài_báo",
    "xem", "đọc", "tìm", "tìm_kiếm", "tra", "tra_cứu", "cho", "biết", "hỏi",
    "vụ", "chuyện", "thông_tin", "nội_dung", "gì", "nào", "sao", "thế_nào",
    "mới", "nhất", "mới_nhất", "hiện", "đi", "nhé", "nha", "ạ", "vậy", "với",
    "tôi", "mình", "bạn", "em", "anh", "chị", "tớ", "cậu", "muốn", "cần",
    "liệt_kê", "danh_sách", "mục", "chuyên_mục", "có", "là", "được",
}


# ------------------------------------------------------- ĐỘ MỚI CỦA TIN --
# Hệ số thưởng cho bài mới khi xếp hạng:  score' = cosine * (1 + alpha * recency)
#
# Cố tình để NHỎ. Mục đích không phải luôn ưu tiên tin mới, mà chỉ PHÁ THẾ HÒA:
# khi hai bài liên quan xấp xỉ nhau thì bài mới thắng. Bài cũ nhưng liên quan
# hơn hẳn vẫn phải thắng — nếu không, bot sẽ chỉ trả tin mới nhất bất kể hỏi gì.
#
# Cả hai giá trị dưới đây được dò bằng `python src/evaluate.py` trên DEV, với
# RÀNG BUỘC CỨNG là ca tin mới phủ định tin cũ phải đúng, rồi chọn MRR cao nhất.
# Bảng quét hiện hành: data/eval/test_report_v3.txt (PHA 1.3).
#
# (Lịch sử — lần dò CŨ trên tập 31 câu bị rò rỉ, giữ để đối chiếu.)
# Bảng quét (half-life x alpha):
#
#   half-life  alpha   Recall@1  MRR     ca tin mâu thuẫn
#          30   0.35      93.5%  0.968   SAI (trả tin cũ)
#          14   0.60      96.8%  0.984   SAI (trả tin cũ)
#           3   0.60      93.5%  0.968   ĐÚNG
#           7   0.60      96.8%  0.984   ĐÚNG   <- chọn
#
# Kết luận cũ "thêm độ mới LÀM TỐT LÊN truy hồi" (93.5% -> 96.8% trên 31 câu)
# đã bị BÁC BỎ khi đo trên dev 112 câu: độ mới không cải thiện nhất quán, và
# mọi cấu hình xử lý đúng ca tin mâu thuẫn đều thấp hơn nhẹ (MRR 0.957–0.964
# so với 0.965 khi tắt). Độ mới là một ĐÁNH ĐỔI có chủ đích (docs/06, mục 4.1).
#
# 0.6 -> 0.3 (nhánh cap-nhat-du-lieu, docs/09): khi đổi mốc tham chiếu sang
# "candidates" (xem FRESHNESS_REFERENCE) và dò lại trên corpus 532 bài, 0.3 là
# alpha NHỎ NHẤT có MRR dev cao nhất mà vẫn qua cả hai ràng buộc cứng. Với 0.6,
# một bài mới chỉ cần cosine bằng 62,5% bài đúng là đã chiếm hạng 1 — quá mạnh
# để gọi là "phá thế hòa" khi corpus có thêm bài mới mỗi ngày.
FRESHNESS_ALPHA = 0.3

# Mốc tính tuổi bài báo khi chấm độ mới.
#   "corpus"     : ngày đăng MỚI NHẤT của cả corpus (cách cũ).
#   "candidates" : ngày đăng mới nhất trong các bài ĐANG CẠNH TRANH cho câu hỏi
#                  này — những bài mà nếu được thưởng độ mới tối đa thì có thể
#                  vượt lên hạng 1:  điểm x (1 + alpha) >= điểm cao nhất.
#
# Lỗi của "corpus" (phát hiện sau lần crawl 14/09/2026): mốc trượt theo bài mới
# nhất của CẢ kho. Crawl thêm 151 bài không liên quan, mốc nhảy từ 10/09 lên
# 14/09, hai bài Cát Linh (01/09 và 10/09) cùng "già" đi và chênh lệch hệ số
# thưởng co lại — bot quay về trả bài cũ đã sai. Tức là THỨ HẠNG của một câu hỏi
# phụ thuộc vào những bài chẳng liên quan gì tới nó. Không cặp (nửa chu kỳ,
# alpha) nào sửa được: evaluate.py quét cả lưới, mode "corpus" không có cấu hình
# nào qua được ràng buộc "thêm bài không liên quan ngày tương lai" (docs/09).
# Với "candidates", thêm bài không liên quan không đổi thứ hạng — bất biến theo
# thiết kế, không phải nhờ may mắn của tham số.
FRESHNESS_REFERENCE = "candidates"

# Nửa chu kỳ suy giảm: sau ngần này ngày, điểm độ mới còn một nửa.
# 3 ngày (dò trên dev; lần dò cũ bị rò rỉ từng chọn 7). Đủ ngắn để hai bài cách
# nhau vài ngày — đúng tình huống tin mới phủ định tin cũ — nhận hệ số thưởng
# khác hẳn nhau. Với 30 ngày, hai bài cách nhau 9 ngày chỉ chênh nhau ~7% hệ số
# thưởng, không đủ để lật thứ hạng. Vì ngưỡng CHẤP NHẬN áp lên cosine thuần,
# nửa chu kỳ ngắn không làm bài cũ bị loại khỏi câu trả lời (docs/06, mục 5.1).
FRESHNESS_HALFLIFE_DAYS = 3.0

# Thư mục cache index đã dựng (tránh phải tách từ lại 381 bài mỗi lần khởi động).
INDEX_CACHE_PATH = MODELS_DIR / "retriever_index.joblib"


# ------------------------------------------------- CHỐNG GÕ SAI CHÍNH TẢ --
# Lỗi người dùng báo: "thám hiểm Sơn Dòng" bị từ chối, "Sơn Đoòng" thì trả lời
# đúng. TF-IDF mức từ so khớp CHÍNH XÁC: "dòng" là một từ có thật (100/532 bài) còn
# "đoòng" là term khác hẳn, nên cosine chỉ 0.086 < 0.13. Hạ ngưỡng không cứu
# được — xuống 0.08 thì chặn đúng câu ngoài phạm vi trên dev rơi từ 100% còn 57%
# (data/eval/test_report_v4.txt, PHA 1.4; docs/09).
#
# Cách sửa: chỉ mục THỨ BA gồm n-gram KÝ TỰ của TIÊU ĐỀ đã bỏ dấu, dùng như
# ĐƯỜNG DỰ PHÒNG — chỉ chạy khi đường chính (mức từ) không có bài nào đạt
# ngưỡng. Câu mà đường chính ĐÃ trả lời thì giữ nguyên kết quả; chỉ câu trước
# đây bị TỪ CHỐI mới có thêm cơ hội (và có thêm rủi ro trả sai).
#   "son dong" và "son doong" chung " so","son","on "," do","ong","ng "
#
# Điểm dự phòng = cosine mức từ + cosine n-gram ký tự  (thang [0, 2]).
# Các lựa chọn được so trên dev (112 câu sạch + 112 câu gõ sai + 28 câu ngoài
# phạm vi) bằng `python tools/compare_improvements.py` — trường đưa vào chỉ mục
# x n x luật chấp nhận, mỗi thiết kế dò ngưỡng tốt nhất của riêng nó:
#   chỉ tiêu đề,   n=3, cộng cosine từ -> cứu 36 câu, 0 trả sai, 0 lọt  <- chọn
#   chỉ tiêu đề,   n=3, chỉ ký tự      -> cứu 34 câu, 1 trả sai, 0 lọt
#   tiêu đề+mô tả, n=4, cộng cosine từ -> cứu 36 câu, 4 trả sai, 0 lọt
#   cả bài,        n=4, chỉ ký tự      -> cứu 20 câu, 5 trả sai, 0 lọt
# Tiêu đề cô đọng tên riêng cần tìm; thêm thân bài làm vector ký tự "đặc" lên và
# mọi câu đều na ná nhau. Cộng thêm cosine mức từ luôn tốt hơn dùng một mình
# n-gram ký tự: các từ GÕ ĐÚNG còn lại trong câu là bằng chứng độc lập.
FUZZY_ENABLED = True
FUZZY_CHAR_N = 3
FUZZY_THRESHOLD = 0.53


# ----------------------------------------------------------- CÁCH XẾP HẠNG --
# "tfidf" : xếp hạng bằng cosine TF-IDF (như ban đầu)
# "bm25"  : xếp hạng bằng Okapi BM25 tự cài đặt (vectorizer.bm25_*)
# Dù chọn cách nào, việc CHẤP NHẬN trả lời vẫn dựa trên cosine TF-IDF, vì điểm
# BM25 không bị chặn trên và không so được giữa các câu hỏi khác nhau.
# Chọn bằng `python src/evaluate.py` trên tập DEV.
#
# Kết quả thí nghiệm (xem docs/06): BM25 KHÔNG hơn TF-IDF. Trên dev, kiểm định
# dấu có cặp cho BM25 tốt hơn ở 3 câu, kém hơn ở 3 câu, hòa 106 câu (p = 1.0);
# trên test BM25 còn kém nhẹ (MRR 0.943 vs 0.950). Giữ TF-IDF vì đơn giản hơn.
# BM25_K1/B dưới đây là cấu hình BM25 tốt nhất trên dev, chỉ dùng khi đổi
# RANKING_METHOD = "bm25".
#
# Dò lại trên corpus 532 bài (data/eval/test_report_v4.txt): kết luận không đổi —
# BM25 tốt hơn 2 câu, kém hơn 4 câu, hòa 106 (p = 0.688); trên test MRR 0.915 so
# với 0.940 của TF-IDF. Cấu hình BM25 tốt nhất trên dev đổi b 0.9 -> 0.75.
RANKING_METHOD = "tfidf"
BM25_K1 = 8.0
BM25_B = 0.75
