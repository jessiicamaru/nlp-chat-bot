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

# Ngưỡng điểm ensemble để chấp nhận nhãn intent từ classifier.
# Dò lưới (w_nb x threshold): w_nb=0.8, threshold=0.25 cho điểm cân bằng
# tốt nhất — accuracy 88.5%, safety 87.5%.
INTENT_THRESHOLD = 0.25

# Khi người dùng đã tự nêu chuyên mục ("tin du lịch ninh bình"), tập ứng viên
# co lại còn vài chục bài cùng chủ đề, nên một cosine thấp hơn vẫn là bằng
# chứng đủ mạnh. Ngưỡng trong trường hợp này = RETRIEVAL_THRESHOLD x hệ số này.
# Hằng số đặt tay (chưa dò trên dev vì dev có quá ít câu nêu chuyên mục).
CATEGORY_SCOPED_THRESHOLD_FACTOR = 0.6

# Trọng số của Naive Bayes trong ensemble; phần còn lại (1 - w_nb) là tín hiệu
# cosine tới pattern gần nhất. Xem giải thích trong intent_classifier.py.
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
# Cả hai giá trị dưới đây được dò bằng `python src/evaluate.py`, tối ưu đồng
# thời hai mục tiêu: (a) không làm giảm Recall@1/MRR, (b) xử lý được ca tin mới
# phủ định tin cũ. Bảng quét (half-life x alpha):
#
#   half-life  alpha   Recall@1  MRR     ca tin mâu thuẫn
#          30   0.35      93.5%  0.968   SAI (trả tin cũ)
#          14   0.60      96.8%  0.984   SAI (trả tin cũ)
#           3   0.60      93.5%  0.968   ĐÚNG
#           7   0.60      96.8%  0.984   ĐÚNG   <- chọn
#
# Đáng chú ý: thêm độ mới còn LÀM TỐT LÊN chất lượng truy hồi, chứ không chỉ là
# đánh đổi. Đo trên CÙNG tập test 31 truy vấn, chỉ bật/tắt yếu tố độ mới:
#   tắt (alpha=0) -> Recall@1 93.5%, MRR 0.968
#   bật (alpha=0.6) -> Recall@1 96.8%, MRR 0.984
FRESHNESS_ALPHA = 0.6

# Nửa chu kỳ suy giảm: sau ngần này ngày, điểm độ mới còn một nửa.
# 7 ngày — đủ ngắn để phân biệt hai bài cách nhau vài ngày (chính là tình huống
# tin mới phủ định tin cũ), nhưng chưa ngắn tới mức vùi lấp bài liên quan hơn
# chỉ vì nó cũ hơn một tuần. Với 30 ngày, hai bài cách nhau 9 ngày chỉ chênh
# nhau 5% điểm thưởng — không đủ để lật thứ hạng.
FRESHNESS_HALFLIFE_DAYS = 3.0

# Thư mục cache index đã dựng (tránh phải tách từ lại 381 bài mỗi lần khởi động).
INDEX_CACHE_PATH = MODELS_DIR / "retriever_index.joblib"


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
RANKING_METHOD = "tfidf"
BM25_K1 = 8.0
BM25_B = 0.9
