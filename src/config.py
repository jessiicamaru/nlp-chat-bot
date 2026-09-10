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
# ---- Các giá trị dưới đây được DÒ BẰNG THỰC NGHIỆM, không chọn cảm tính.
# Chạy `python src/evaluate.py` để tái lập bảng dò ngưỡng.

# Ngưỡng cosine similarity để chấp nhận câu trả lời từ retriever.
# Dò trên 21 truy vấn trong phạm vi + 8 truy vấn ngoài phạm vi:
#   0.08 -> trả lời được 100%, chặn đúng 87.5%
#   0.12 -> trả lời được 100%, chặn đúng 100%   <- chọn
#   0.18 -> trả lời được 77.8%, chặn đúng 100%  (bắt đầu bỏ sót)
RETRIEVAL_THRESHOLD = 0.12

# Ngưỡng điểm ensemble để chấp nhận nhãn intent từ classifier.
# Dò lưới (w_nb x threshold): w_nb=0.8, threshold=0.25 cho điểm cân bằng
# tốt nhất — accuracy 88.5%, safety 87.5%.
INTENT_THRESHOLD = 0.25

# Trọng số của Naive Bayes trong ensemble; phần còn lại (1 - w_nb) là tín hiệu
# cosine tới pattern gần nhất. Xem giải thích trong intent_classifier.py.
INTENT_W_NB = 0.8

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
