"""
build_eval_sets.py — Sinh tập đánh giá DEV và TEST từ một pool truy vấn viết tay.

## Vì sao phải tách dev / test

Trước đây mọi siêu tham số (w_nb, INTENT_THRESHOLD, RETRIEVAL_THRESHOLD,
alpha / nửa chu kỳ độ mới) đều được dò bằng cách quét trên test_queries.json,
rồi các con số trong báo cáo lại được đo trên CHÍNH tập đó. Đó là rò rỉ tập
test: con số phản ánh mức "khớp" với đúng những câu đã dùng để dò, chứ không
phải hiệu năng trên câu hỏi chưa thấy bao giờ.

Quy tắc mới:
  - DEV  : dùng để dò tham số. Được phép nhìn bao nhiêu lần cũng được.
  - TEST : chỉ dùng để BÁO CÁO, một lần, với tham số đã chốt trên dev.
           Không bao giờ được chỉnh tham số dựa trên kết quả test.

## Cách chia

  - Toàn bộ truy vấn CŨ (đã từng dùng để dò) -> bắt buộc vào DEV, vì chúng
    đã "nhiễm" — không còn là câu chưa thấy.
  - Truy vấn MỚI viết thêm -> chia ngẫu nhiên có hạt giống cố định
    (40% dev / 60% test) để tái lập được.

## Nhãn đúng theo URL, không theo chữ trong tiêu đề

Tập cũ chấm đúng khi tiêu đề kết quả CHỨA một từ ("iPhone") — quá dễ dãi, vì
có hàng chục bài chứa chữ đó. Nay mỗi truy vấn ghi rõ (các) bài đúng theo chỉ
số trong corpus, rồi đổi sang URL. Một truy vấn có thể có vài bài đúng khi
corpus thực sự có nhiều bài về cùng một sự việc (ví dụ ba bài về "Match Day").

## Giới hạn cần nói rõ

Người viết truy vấn cũng là người xây bot, nên truy vấn có thể mang cùng "điểm
mù" với bot (vô thức dùng đúng từ khóa mà bot giỏi bắt). Để giảm thiểu, truy
vấn được viết dựa trên phần MÔ TẢ bài báo và diễn đạt lại bằng lời khác, không
chép tiêu đề. Nhưng cách tốt nhất vẫn là nhờ người khác viết thêm truy vấn.

Chạy:  .venv/Scripts/python.exe tools/build_eval_sets.py
"""

import json
import random
import sys
import unicodedata
from pathlib import Path

import pandas as pd

PROJ = Path(__file__).resolve().parent.parent
OUT_DIR = PROJ / "data" / "eval"
CORPUS = PROJ / "data" / "raw" / "corpus_raw.csv"
SEED = 2026
DEV_RATIO_NEW = 0.4
UNACCENT_RATIO = 0.25


def strip_accents(text: str) -> str:
    nfd = unicodedata.normalize("NFD", text)
    out = "".join(c for c in nfd if unicodedata.category(c) != "Mn")
    return out.replace("đ", "d").replace("Đ", "D")


# ===========================================================================
# 1. TRUY VẤN CŨ — đã dùng để dò tham số -> bắt buộc vào DEV
# ===========================================================================
OLD_RETRIEVAL = [
    ("đảo hải nam miễn visa 30 ngày", [0], "có dấu"),
    ("du lịch ninh bình dịp quốc khánh", [2], "có dấu"),
    ("đi tả van ngắm mùa vàng", [1], "có dấu"),
    ("khách mỹ vi phạm lệnh cấm ở venice", [5], "có dấu"),
    ("lịch nghỉ lễ tết của du lịch việt", [3], "có dấu"),
    ("giá iphone 18 pro tại việt nam", [29], "có dấu"),
    ("điện thoại gập đầu tiên của apple", [120, 121, 132], "có dấu"),
    ("apple tăng giá iphone 17", [119], "có dấu"),
    ("giá xăng dầu tăng", [75], "có dấu"),
    ("scb thu hồi nợ trương mỹ lan", [78], "có dấu"),
    ("học sinh việt bị bắt nạt học đường", [210], "có dấu"),
    ("match day đại học y hà nội", [209, 213, 225], "có dấu"),
    ("học sinh huế thiếu sách giáo khoa", [212], "có dấu"),
    ("robot chó dẫn đường trung quốc", [294], "có dấu"),
    ("vụ nổ tên lửa blue origin", [295], "có dấu"),
    ("hạn hán kênh đào panama", [298], "có dấu"),
    ("champions league man utd", [249, 254], "không dấu"),
    ("asiad 20 khởi tranh", [250], "có dấu"),
    ("kẹp tiền để được xạ trị sớm", [164], "có dấu"),
    ("ăn chuối sai cách hại tiêu hóa", [168], "có dấu"),
    ("cơm nguội để được bao lâu", [169], "có dấu"),
    ("tin ve dao hai nam", [0], "không dấu"),
    ("gia iphone 18 pro tai viet nam", [29], "không dấu"),
    ("hoc sinh viet bi bat nat", [210], "không dấu"),
    ("vu no ten lua blue origin", [295], "không dấu"),
    ("com nguoi de duoc bao lau", [169], "không dấu"),
    ("han han kenh dao panama", [298], "không dấu"),
    ("bt gì về giá xăng k", [75], "teencode"),
    ("ko bt gia iphone the nao", [29, 76, 119, 121, 129, 302, 310], "teencode"),
    ("cho t hỏi vụ hải nam vs", [0], "teencode"),
    ("robot cho dan duong trung quoc", [294], "không dấu"),
]

OLD_OUT_OF_SCOPE = [
    "thời tiết sao hỏa hôm nay thế nào",
    "công thức nấu phở bò gia truyền",
    "giải phương trình bậc hai giúp mình",
    "dịch câu này sang tiếng nhật",
    "kể một câu chuyện cười đi",
    "asdfgh qwerty zxcvb",
    "bitcoin sẽ lên bao nhiêu vào năm 2030",
    "viết giúp mình một bài thơ về mùa thu",
    "thoi tiet sao hoa hom nay the nao",
    "cong thuc nau pho bo gia truyen",
    "ke 1 cau chuyen cuoi di b",
    "gia bitcoin nam 2030 la bao nhieu",
]

OLD_INTENT = [
    ("chào bot nhé", "chao_hoi"), ("hế lô bạn ơi", "chao_hoi"),
    ("chào buổi chiều", "chao_hoi"), ("thôi mình đi ngủ đây", "tam_biet"),
    ("bye nhé bot", "tam_biet"), ("cảm ơn bạn rất nhiều", "cam_on"),
    ("thank nha", "cam_on"), ("bạn là con người à", "hoi_ve_bot"),
    ("ai đã lập trình ra bạn vậy", "hoi_ve_bot"),
    ("bạn hoạt động dựa trên nguyên lý gì", "hoi_ve_bot"),
    ("chỉ mình cách hỏi với", "huong_dan"), ("bạn hỗ trợ được những gì", "huong_dan"),
    ("cho mình xem danh sách các mục", "liet_ke_chuyen_muc"),
    ("bạn có những lĩnh vực nào", "liet_ke_chuyen_muc"),
    ("kho của bạn có bao nhiêu tin", "thong_ke_corpus"),
    ("dữ liệu lấy từ nguồn nào vậy", "thong_ke_corpus"),
    ("cho xem tin thể thao đi", "tin_theo_chuyen_muc"),
    ("mục giáo dục có bài gì", "tin_theo_chuyen_muc"),
    ("liệt kê tin kinh doanh", "tin_theo_chuyen_muc"),
    ("tóm tắt lại giúp mình với", "tom_tat_bai"), ("bài vừa rồi nói gì vậy", "tom_tat_bai"),
    ("cho xin đường dẫn bài viết", "hoi_nguon"), ("link gốc ở đâu vậy bạn", "hoi_nguon"),
    ("ừ đúng rồi đó", "dong_y"), ("không phải ý mình", "tu_choi"),
    ("bot trả lời chán quá đi", "che_bai"),
    ("cho tôi biết về giá iphone", "__retrieve__"), ("tin về đảo hải nam", "__retrieve__"),
    ("thông tin về ninh bình", "__retrieve__"), ("bạn biết gì về man utd", "__retrieve__"),
    ("giá xăng dầu thế nào", "__retrieve__"), ("học sinh bị bắt nạt", "__retrieve__"),
    ("robot chó dẫn đường", "__retrieve__"), ("có bài nào về bệnh viện không", "__retrieve__"),
]

# ===========================================================================
# 2. TRUY VẤN MỚI — chưa từng dùng để dò -> chia dev / test
#    Viết dựa trên MÔ TẢ bài báo, diễn đạt lại, không chép tiêu đề.
# ===========================================================================
NEW_RETRIEVAL = [
    # ---- Du lịch
    ("gấu tấn công du khách gần hồ Towada", [7], "có dấu"),
    ("vì sao đêm đầu ngủ ở khách sạn hay bị mất ngủ", [8], "có dấu"),
    ("lồng đèn làm từ nan trúc giấy dó ở đình Sơn Trà", [9], "có dấu"),
    ("những sân bay bị bỏ hoang trên thế giới", [10], "có dấu"),
    ("ng thái lan thik nghỉ dài ngày ở tp nào của vn", [11], "teencode"),
    ("chính sách quá cảnh 240 giờ của Trung Quốc", [12], "có dấu"),
    ("hai bà cụ ngoài 90 tuổi đi du lịch khắp thế giới", [13], "có dấu"),
    ("ưu đãi Sun World Hạ Long dịp lễ 2/9", [14], "có dấu"),
    ("cồn cát cao 100 mét ở Italy và kỳ quan thiên nhiên châu Âu", [15], "có dấu"),
    ("thành phố Venice nhái ở Đại Liên vắng khách", [16], "có dấu"),
    ("thị trấn gần cực Bắc tối suốt 3 tháng có gấu Bắc Cực", [17], "có dấu"),
    ("nhà hàng chay ở Hà Nội mùa Vu Lan", [18], "có dấu"),
    ("chữ SBY trên vé máy bay là j z", [19], "teencode"),
    ("du khách mắc kẹt trên đảo Cát Bà vì bão số 4", [20], "có dấu"),
    ("nc nào ồn nhất tg v", [21], "teencode"),
    ("khách Pháp chết ở Thung lũng Chết vì nắng nóng", [22], "có dấu"),
    ("vụ đắm tàu Costa Concordia", [23], "có dấu"),
    ("bánh flan cỡ lớn gây sốt ở Sài Gòn", [24], "có dấu"),
    ("căn hộ tam giác siêu hẹp ở Tokyo", [25], "có dấu"),
    ("khách Thụy Sĩ đi tù vì chế giễu lễ Nyepi ở Bali", [26], "có dấu"),
    ("hàng bánh cuốn 15 nghìn trong ngõ Thụy Khuê", [33], "có dấu"),
    ("người Trung Quốc đi 233 quốc gia trong 33 năm", [34], "có dấu"),
    ("người dân Estonia mở quán cà phê tại nhà đón khách lạ", [35], "có dấu"),
    ("gia đình Úc sang Đà Nẵng sống để dành tiền mua nhà", [37], "có dấu"),
    ("Shinjuku cấm cho thuê nhà ngắn hạn", [38], "có dấu"),
    ("đảo Cồn Cỏ Quảng Trị nhìn từ trên cao", [39], "có dấu"),
    ("kem Tràng Tiền sắp đóng cửa", [41], "có dấu"),
    ("sân bay Việt Nam cải thiện kết nối hàng không", [42], "có dấu"),
    ("cung đường chữ S ở Y Tý Lào Cai", [43], "có dấu"),
    ("chuột còn sống trong hộp mì ở Hong Kong", [44], "có dấu"),
    ("khách Việt kẹt ở sân bay Jakarta vì núi lửa", [46, 51], "có dấu"),
    ("khách Việt ngại đi Jeju sau vụ phát hiện thi thể", [47], "có dấu"),
    ("rừng dừa do Liên Xô tài trợ ở Quảng Ngãi", [48], "có dấu"),
    ("máy bay Concorde vì sao ngừng bay", [52], "có dấu"),
    ("sao ks ko có phòng 404", [53], "teencode"),
    ("khiếu nại du lịch ở Hàn Quốc tăng kỷ lục", [55], "có dấu"),
    ("du khách Anh chết sau khi làm đẹp ở Thái Lan", [56], "có dấu"),
    ("lừa đảo đặt phòng khách sạn Măng Đen", [63], "có dấu"),
    ("Vietjet bán vé 0 đồng ngày 9/9", [71], "có dấu"),
    ("quán nước sát mép vực trên đèo Lương Sơn", [72], "có dấu"),
    ("leo núi nghe theo AI rồi bị lạc", [61], "có dấu"),
    ("hành khách gây rối bị dán băng keo vào ghế", [68], "có dấu"),
    ("đánh chó ở quảng trường Lâm Viên Đà Lạt bị phạt", [69], "có dấu"),
    ("Vũng Tàu bắn pháo hoa lễ Quốc khánh", [6], "có dấu"),
    ("kinh nghiệm săn mây ở miền Bắc", [67], "có dấu"),
    # ---- Công nghệ
    ("Mỹ tập trận trên quỹ đạo Apollo Maneuvers", [28], "có dấu"),
    ("iPhone 18 Pro Max có mấy màu", [30, 299], "có dấu"),
    ("airpods 5 giá bn", [122], "teencode"),
    ("Apple Watch vỏ gốm quay lại", [123], "có dấu"),
    ("nhân viên OpenAI tiêu 7000 đô tiền token mỗi ngày", [124], "có dấu"),
    ("robot Unitree tự học cách đánh nhau", [125], "có dấu"),
    ("đấu giá băng tần 900 MHz", [127], "có dấu"),
    ("nhà nghiên cứu Anthropic nghỉ việc vì sợ AI", [128], "có dấu"),
    ("iPhone 17 Pro giảm giá trước khi ra 18 Pro", [129], "có dấu"),
    ("robot Iron của Xpeng tự bước khỏi dây chuyền", [130], "có dấu"),
    ("sâu máy tính WeWorm tấn công WeChat", [131], "có dấu"),
    ("máy in 3d có đáng mua ko m", [133], "teencode"),
    ("đồng hồ trẻ em Huawei có hai camera", [134], "có dấu"),
    ("Việt Nam hợp tác với Nga về công nghệ lượng tử", [135], "có dấu"),
    ("Xiaomi ra điện thoại gập trước Apple", [145], "có dấu"),
    ("Huawei Mate XT2 gập ba", [146], "có dấu"),
    ("Microsoft 365 Copilot cho doanh nghiệp", [150], "có dấu"),
    ("tên lửa Spectrum của Đức phóng vệ tinh", [152], "có dấu"),
    ("GPT-6 Astra siêu trí tuệ", [153, 331], "có dấu"),
    ("robot biểu tình ở Warsaw đòi luật về AI", [140], "có dấu"),
    ("tàu hàng đi qua Bắc Cực do băng tan", [142], "có dấu"),
    ("robot chó Lynx chạy trên sa mạc", [143], "có dấu"),
    ("Liên Hợp Quốc dùng bản đồ Equal Earth", [157], "có dấu"),
    ("vườn mưa chống ngập đô thị", [161], "có dấu"),
    ("Huawei đưa mạng 5.5G về Việt Nam", [163], "có dấu"),
    ("robot lau nhà có cánh tay", [156], "có dấu"),
    ("vì sao người ta thích xem robot bị ngã", [155], "có dấu"),
    ("startup edtech Việt Nam đứng thứ mấy Đông Nam Á", [149], "có dấu"),
    # ---- Kinh doanh
    ("khu kinh tế đặc biệt được ưu đãi gì", [77], "có dấu"),
    ("lợi suất trái phiếu Mỹ kỳ hạn 10 năm", [79], "có dấu"),
    ("nhập khẩu linh kiện điện tử tăng vì giá chip", [80, 105], "có dấu"),
    ("phó tổng Vingroup từ chức", [81], "có dấu"),
    ("metro Bến Thành Suối Tiên phải lãi bao nhiêu", [82], "có dấu"),
    ("dạo này ng ta mở tk chứng khoán nhiều ko", [83], "teencode"),
    ("dọn rác sông Tô Lịch", [84], "có dấu"),
    ("xuất khẩu rau quả chế biến", [85], "có dấu"),
    ("Việt Nam có bao nhiêu công ty blockchain", [87], "có dấu"),
    ("Shinhan và HSBC lãi tại Việt Nam", [89], "có dấu"),
    ("khối ngoại bán ròng trước nâng hạng", [90], "có dấu"),
    ("Tiktok đầu tư logistics ở TP HCM", [91], "có dấu"),
    ("giá dầu Brent lên 100 đô", [92, 111], "có dấu"),
    ("ng việt tốn nhiều tiền cho trà sữa ko", [93], "teencode"),
    ("thịt gà nhập khẩu rẻ hơn rau", [95], "có dấu"),
    ("lãi suất tiết kiệm có còn cao ko b", [98], "teencode"),
    ("hàng nhập dưới 100 nghìn có được miễn thuế", [99], "có dấu"),
    ("giải thể công ty không doanh thu bị phạt gì", [100], "có dấu"),
    ("Ariston mua lại Mutosi", [101], "có dấu"),
    ("Trung Quốc mua thêm vàng dự trữ", [103], "có dấu"),
    ("mua giày Nike giả ở Nha Trang", [104], "có dấu"),
    ("casino Phú Quốc báo lỗ", [106], "có dấu"),
    ("giá đô ở ngân hàng giảm r à", [110], "teencode"),
    ("nợ vay của doanh nghiệp bất động sản", [113], "có dấu"),
    ("giảm trừ thuế thu nhập cho học phí và viện phí", [114], "có dấu"),
    ("HDBank thu giữ đất của Hoàng Quân", [115], "có dấu"),
    ("trung tâm tài chính quốc tế Đà Nẵng và TP HCM", [118], "có dấu"),
    ("bếp đun giá 5 đô nhờ tín chỉ carbon ở châu Phi", [112], "có dấu"),
    ("Mexico phải trả nước cho Mỹ", [97], "có dấu"),
    # ---- Sức khỏe
    ("bác sĩ trẻ về trạm y tế TP HCM", [165], "có dấu"),
    ("chuẩn bị cho tuổi già từ năm 40 tuổi", [166], "có dấu"),
    ("tập thể dục ăn ít mà không giảm cân", [167], "có dấu"),
    ("giảm 51 kg phải cắt túi mật", [171], "có dấu"),
    ("áp lực mua nhà mua xe trước khi cưới gây mất ngủ", [172], "có dấu"),
    ("uống mật ong có bị béo bụng ko", [173], "teencode"),
    ("hay dậy giữa đêm có sao ko b", [174], "teencode"),
    ("thói quen khiến ngáy to hơn", [176], "có dấu"),
    ("xúc xích lạp xưởng gây ung thư", [177], "có dấu"),
    ("đồ ăn giúp hấp thụ magie", [180], "có dấu"),
    ("làm sao cho hết đờm ở họng v", [182], "teencode"),
    ("trẻ bị nhược thị có chữa được không", [183], "có dấu"),
    ("dịch Ebola ở Congo WHO thiếu tiền", [193], "có dấu"),
    ("ép tim 15 phút cứu người ngừng tim", [194], "có dấu"),
    ("sốt xuất huyết phải chạy ECMO tốn gần tỷ", [196], "có dấu"),
    ("đồ để lâu trong tủ lạnh có độc ko", [197], "teencode"),
    ("khuyến khích kết hôn trước 30 tuổi sinh hai con", [198], "có dấu"),
    ("YouTuber ăn nhiều đến mức dạ dày phình to", [199, 200, 340], "có dấu"),
    ("bài tập tốt cho người huyết áp cao", [204], "có dấu"),
    ("uống cf có bị vàng răng ko", [206], "teencode"),
    ("sau đột quỵ ít nói có phải trầm cảm", [186], "có dấu"),
    ("bé 6 tuổi viễn thị 7 độ", [192], "có dấu"),
    # ---- Giáo dục
    ("GS Lê Hồng Vân nhà toán học", [211], "có dấu"),
    ("phụ huynh phàn nàn bữa ăn bán trú ở Hà Nội", [214, 221], "có dấu"),
    ("bác sĩ nội trú chọn sản phụ khoa", [215], "có dấu"),
    ("AI giải bài toán Navier-Stokes", [216], "có dấu"),
    ("học sinh Việt đọc hiểu kém theo OECD", [217], "có dấu"),
    ("đại học Mỹ gửi thư mời nhập học không cần hồ sơ", [218], "có dấu"),
    ("trường dạy đờn ca tài tử miễn phí", [220], "có dấu"),
    ("trường tiểu học hơn 10.500 học sinh ở Nha Trang", [222, 226, 232], "có dấu"),
    ("Trung Quốc xử lý giáo sư gian lận học thuật", [223], "có dấu"),
    ("Hà Nội thi đề chung để xét tuyển lớp 10", [224], "có dấu"),
    ("Đại học Chicago cấm dùng laptop và AI", [227], "có dấu"),
    ("giáo viên được dạy thêm môn khác", [229], "có dấu"),
    ("nữ sinh chép bài thủ khoa bị trừ điểm", [230], "có dấu"),
    ("anh em sinh đôi vào thẳng Học viện Kỹ thuật quân sự", [233], "có dấu"),
    ("sửa chương trình sách giáo khoa lớp 1 đến 12", [234], "có dấu"),
    ("trường Việt Đức sửa sang 300 tỷ", [236], "có dấu"),
    ("sv mới ra trường khó kiếm vc vì AI hả", [237], "teencode"),
    ("cắt giảm hiệu trưởng hiệu phó ở TP HCM", [241], "có dấu"),
    ("thủ khoa Bách khoa TP HCM", [242], "có dấu"),
    ("du học mỹ tốn bn 1 năm", [243], "teencode"),
    ("phụ huynh chạy khắp nơi mua sách giáo khoa lớp 9", [239], "có dấu"),
    # ---- Thể thao
    ("Arteta nói về phong độ của Odegaard", [251], "có dấu"),
    ("bán kết Mỹ Mở rộng nữ có bốn hạt giống", [252], "có dấu"),
    ("Ancelotti gọi cầu thủ mới cho tuyển Brazil", [253], "có dấu"),
    ("Raphinha ghi bàn liên tiếp cho Barca", [257, 287], "có dấu"),
    ("Yamal vượt kỷ lục của Mbappe", [258], "có dấu"),
    ("Barca thắng Feyenoord 5-1", [261, 259, 258], "có dấu"),
    ("Chelsea thắng ngược Leeds 6-3", [262], "có dấu"),
    ("Ferran Torres lập hat-trick cho PSG", [263], "có dấu"),
    ("Liverpool thắng Atletico Madrid", [266], "có dấu"),
    ("Ronaldo ghi bàn thứ 979", [267], "có dấu"),
    ("giải marathon ở Pháp phát rượu vang", [268], "có dấu"),
    ("rybakina lên top 1 tg r hả", [269], "teencode"),
    ("Thùy Linh thua ở Vietnam Open", [271], "có dấu"),
    ("Đình Bắc không dự ASIAD", [274], "có dấu"),
    ("Alcaraz thua Shelton", [275], "có dấu"),
    ("lịch đá asean cup ntn", [276], "teencode"),
    ("cựu tuyển thủ Nguyễn Trọng Đại sa sút", [278], "có dấu"),
    ("Messi mua câu lạc bộ Eldense", [285], "có dấu"),
    ("Haaland ngang kỷ lục của Aguero", [283, 288, 256], "có dấu"),
    ("Messi được đề cử Quả bóng vàng 2026", [291], "có dấu"),
    ("HLV Ikeuchi rời U20 Việt Nam", [292], "có dấu"),
    ("Casemiro ghi bàn cho Inter Miami", [260], "có dấu"),
    # ---- Khoa học
    ("dấu hiệu vật chất tối dưới lòng đất South Dakota", [301], "có dấu"),
    ("Nvidia mua Hugging Face", [304, 316], "có dấu"),
    ("laptop Lenovo Yoga Slim 7i", [305], "có dấu"),
    ("đập thủy điện cao nhất thế giới", [306], "có dấu"),
    ("Trung Quốc tìm được mỏ vàng 200 tấn", [308], "có dấu"),
    ("gián cyborg cứu hộ của nhà khoa học Việt", [309], "có dấu"),
    ("OpenAI làm robot hình người", [311], "có dấu"),
    ("tên lửa tái sử dụng Pallas-1", [315, 320], "có dấu"),
    ("khóa 13 triệu SIM chưa xác thực", [318], "có dấu"),
    ("khi nào tắt sóng 2g v", [327], "teencode"),
    ("tốc độ internet Việt Nam vào top 10", [328], "có dấu"),
    ("thiết bị phát hiện camera quay lén", [322], "có dấu"),
    ("lương CEO mới của Apple", [326], "có dấu"),
    ("robot chạy 100 mét trong 8,64 giây", [335], "có dấu"),
    ("CXMT lấy công nghệ DRAM của Samsung", [300], "có dấu"),
    ("vì sao máy bay mất tín hiệu radar trên biển", [330], "có dấu"),
    ("trung tâm dữ liệu AI tốn nước", [303], "có dấu"),
    # ---- Đời sống
    ("vợ phát hiện chồng ngoại tình qua hợp đồng bảo hiểm", [336], "có dấu"),
    ("quy định trên đảo Indian Creek của tỷ phú", [337], "có dấu"),
    ("năm Bính Ngọ Nhật Bản ít sinh con", [338], "có dấu"),
    ("cô gái H'Mông làm bánh nuôi em", [339], "có dấu"),
    ("nữ giao liên trốn dưới lục bình", [341], "có dấu"),
    ("cho t công thức sốt chua ngọt vs", [342], "teencode"),
    ("vợ chồng lâu năm có giống nhau không", [343], "có dấu"),
    ("bà cụ kiện con trai đòi tiền nuôi cháu", [347], "có dấu"),
    ("cô gái bị dị ứng với nước", [349], "có dấu"),
    ("cách thắng nước màu caramel", [351], "có dấu"),
    ("lễ hội đánh nhau ở Peru", [354], "có dấu"),
    ("giấu hồ sơ xin việc trong hộp bánh donut", [360], "có dấu"),
    ("ông bố làm siêu xe bằng gỗ cho con", [363], "có dấu"),
    ("tỷ phú 98 tuổi sống tằn tiện", [357], "có dấu"),
    ("thang máy vách núi cho học sinh Vân Nam", [379], "có dấu"),
    ("chiên khoai tây sao cho giòn z", [380], "teencode"),
    ("nem rán bị cháy hai đầu", [374], "có dấu"),
    ("cô gái Nga lấy chồng thợ điện Vĩnh Long", [378], "có dấu"),
    ("chụp ảnh cưới phong cách xuyên không", [377], "có dấu"),
]

NEW_OUT_OF_SCOPE = [
    ("hướng dẫn nấu bún bò Huế", "có dấu"),
    ("cách cài đặt Windows 11", "có dấu"),
    ("công thức tính diện tích hình tròn", "có dấu"),
    ("ai là tổng thống đầu tiên của Mỹ", "có dấu"),
    ("giá vé xem phim tối nay", "có dấu"),
    ("làm sao để hết mụn trứng cá", "có dấu"),
    ("dự báo thời tiết Đà Lạt cuối tuần", "có dấu"),
    ("lời bài hát Nơi này có anh", "có dấu"),
    ("cách đổi mật khẩu Facebook", "có dấu"),
    ("tuyển lập trình viên Java ở Đà Nẵng", "có dấu"),
    ("bảng xếp hạng K-pop tuần này", "có dấu"),
    ("cách nuôi cá betta", "có dấu"),
    ("viết code python sắp xếp mảng", "có dấu"),
    ("số điện thoại tổng đài điện lực", "có dấu"),
    ("lịch âm hôm nay là ngày bao nhiêu", "có dấu"),
    ("mẹo học tiếng Hàn nhanh", "có dấu"),
    ("review phim Avengers mới nhất", "có dấu"),
    ("cách trồng rau muống trên sân thượng", "có dấu"),
    ("cách làm hộ chiếu online", "có dấu"),
    ("tỷ giá yên Nhật hôm nay", "có dấu"),
    ("kết quả xổ số miền bắc", "có dấu"),
    ("top 10 truyện tranh hay nhất", "có dấu"),
    ("mua bảo hiểm xe máy ở đâu", "có dấu"),
    ("giờ mở cửa bảo tàng Louvre", "có dấu"),
    ("cách chơi cờ tướng cho người mới", "có dấu"),
    ("bài văn tả con mèo lớp 3", "có dấu"),
    ("cách giặt áo len không bị co", "có dấu"),
    ("tỷ số chung kết World Cup 2018", "có dấu"),
    ("cach nau bun bo hue", "không dấu"),
    ("lich chieu phim cuoi tuan", "không dấu"),
    ("gia vang sjc hom nay", "không dấu"),
    ("dang ky wifi viettel the nao", "không dấu"),
    ("cho t xin lời bài hát sơn tùng vs", "teencode"),
    ("chỉ t cách tải game liên quân ik", "teencode"),
    ("hnay ăn j ngon b", "teencode"),
    ("hướng dẫn làm bánh mì sandwich", "có dấu"),
    ("phim hoạt hình hay cho trẻ em", "có dấu"),
    ("địa chỉ tiệm sửa laptop gần đây", "có dấu"),
    ("mẹo giảm cân nhanh trong 1 tuần", "có dấu"),
    ("cách viết CV xin việc bằng tiếng Anh", "có dấu"),
]

NEW_INTENT = [
    ("chào nha", "chao_hoi"), ("helo bot", "chao_hoi"), ("xin chao", "chao_hoi"),
    ("hi bạn", "chao_hoi"),
    ("t đi đây bye", "tam_biet"), ("hẹn gặp lại nhé", "tam_biet"),
    ("tạm biệt bot", "tam_biet"), ("ngủ đây bai", "tam_biet"),
    ("cảm ơn b nhiều", "cam_on"), ("thanks nhé", "cam_on"), ("cam on ban", "cam_on"),
    ("tks bot", "cam_on"),
    ("bạn là chatbot à", "hoi_ve_bot"), ("ai làm ra bạn thế", "hoi_ve_bot"),
    ("ban la gi vay", "hoi_ve_bot"), ("bạn có phải người thật không", "hoi_ve_bot"),
    ("mình có thể hỏi gì", "huong_dan"), ("hướng dẫn mình với", "huong_dan"),
    ("bot làm được những gì", "huong_dan"), ("huong dan su dung", "huong_dan"),
    ("có mấy chuyên mục", "liet_ke_chuyen_muc"), ("liệt kê các chủ đề đi", "liet_ke_chuyen_muc"),
    ("co nhung chu de nao", "liet_ke_chuyen_muc"),
    ("tổng cộng có mấy bài", "thong_ke_corpus"), ("dữ liệu của bạn lấy ở đâu", "thong_ke_corpus"),
    ("co bao nhieu tin", "thong_ke_corpus"),
    ("xem tin kinh doanh", "tin_theo_chuyen_muc"), ("tin khoa học mới", "tin_theo_chuyen_muc"),
    ("cho mình đọc tin đời sống", "tin_theo_chuyen_muc"),
    ("tin the thao moi nhat", "tin_theo_chuyen_muc"),
    ("tóm tắt bài này đi", "tom_tat_bai"), ("nói ngắn gọn bài đó", "tom_tat_bai"),
    ("tom tat giup minh", "tom_tat_bai"),
    ("link đâu b", "hoi_nguon"), ("bài gốc ở đâu", "hoi_nguon"), ("cho xin link bai", "hoi_nguon"),
    ("ok luôn", "dong_y"), ("đúng vậy", "dong_y"), ("ừm", "dong_y"),
    ("không phải vậy", "tu_choi"), ("sai rồi bot", "tu_choi"), ("ko đúng", "tu_choi"),
    ("bot dở quá", "che_bai"), ("trả lời như không", "che_bai"), ("ngu thế", "che_bai"),
    ("tin về robot hình người", "__retrieve__"), ("Nvidia mua lại công ty nào", "__retrieve__"),
    ("có gì mới về Messi không", "__retrieve__"),
    ("vụ lừa đảo khách sạn Măng Đen", "__retrieve__"),
    ("sân bay Jakarta núi lửa", "__retrieve__"), ("học phí du học Mỹ", "__retrieve__"),
    ("tin ve iphone gap", "__retrieve__"), ("bt j về vụ giá dầu k", "__retrieve__"),
]


def main() -> int:
    df = pd.read_csv(CORPUS).dropna(subset=["title", "text"]).reset_index(drop=True)
    urls = df["url"].astype(str).tolist()
    titles = df["title"].astype(str).tolist()

    def to_gold(ids):
        for i in ids:
            if i < 0 or i >= len(urls):
                raise IndexError(f"chỉ số bài {i} nằm ngoài corpus ({len(urls)} bài)")
        return [urls[i] for i in ids]

    rng = random.Random(SEED)

    # Đổi ~25% truy vấn có dấu thành không dấu, mô phỏng người gõ không dấu.
    new_ret = []
    for q, ids, style in NEW_RETRIEVAL:
        if style == "có dấu" and rng.random() < UNACCENT_RATIO:
            q, style = strip_accents(q).lower(), "không dấu"
        new_ret.append({"query": q, "gold_urls": to_gold(ids),
                        "gold_titles": [titles[i] for i in ids], "style": style})

    old_ret = [{"query": q, "gold_urls": to_gold(ids),
                "gold_titles": [titles[i] for i in ids], "style": s, "origin": "cũ"}
               for q, ids, s in OLD_RETRIEVAL]

    def split(items):
        items = list(items)
        rng.shuffle(items)
        k = int(round(len(items) * DEV_RATIO_NEW))
        return items[:k], items[k:]

    ret_dev_new, ret_test = split(new_ret)
    oos_dev_new, oos_test = split([{"text": t, "style": s} for t, s in NEW_OUT_OF_SCOPE])
    int_dev_new, int_test = split([{"text": t, "expected": e} for t, e in NEW_INTENT])

    dev_oos = ([{"text": t, "style": "cũ", "origin": "cũ"} for t in OLD_OUT_OF_SCOPE]
               + oos_dev_new)

    # Thêm biến thể KHÔNG DẤU của chính các câu ngoài phạm vi trong DEV.
    #
    # Vì sao cần: ngưỡng phải được dò trên đúng phân bố truy vấn mà bot thực sự
    # nhận — lập luận đã dùng khi dò RETRIEVAL_THRESHOLD. Tập ngoài phạm vi cũ
    # toàn câu CÓ DẤU, nên không phát hiện được rằng câu không dấu dễ lọt hơn:
    # bỏ dấu làm hai câu khác nhau trông giống nhau hơn, và chỉ mục n-gram ký tự
    # (docs/09) vốn chạy trên bản đã bỏ dấu. Một ca thật lọt qua vì lỗ hổng này:
    # "thoi tiet sao hoa hom nay" (tests/test_chatbot.py).
    #
    # CHỈ sinh từ câu của DEV, không bao giờ từ TEST — nếu không, tập test sẽ có
    # "anh em sinh đôi" nằm trong tập dò, và số liệu test thành lạc quan giả.
    # Hạt giống riêng, đặt SAU khi đã chia dev/test, nên test.json không đổi.
    # Các câu ngoài phạm vi dùng trong kiểm thử hồi quy (tests/test_chatbot.py).
    # Chúng là ca đã BIẾT là khó, nên chỗ của chúng là DEV: ngưỡng phải được dò
    # sao cho chặn được đúng những ca này, thay vì để chúng chỉ nổ ở lúc chạy test.
    have = {x["text"] for x in dev_oos}
    dev_oos += [{"text": t, "style": s, "origin": "kiểm thử hồi quy"} for t, s in [
        ("thoi tiet sao hoa hom nay", "không dấu"),
        ("asdfgh qwerty zxcvb", "vô nghĩa"),
    ] if t not in have]

    oos_rng = random.Random(f"{SEED}-oos-unaccent")
    seen = {x["text"] for x in dev_oos}
    for x in list(dev_oos):
        folded = strip_accents(x["text"]).lower()
        # Bỏ qua câu vốn đã không dấu (danh sách cũ có sẵn vài câu như vậy),
        # nếu không sẽ sinh ra bản sao y hệt và câu đó bị tính hai lần khi dò.
        if folded == x["text"] or folded in seen or oos_rng.random() >= 0.4:
            continue
        seen.add(folded)
        dev_oos.append({"text": folded, "style": "không dấu",
                        "origin": "biến thể không dấu của câu dev"})

    dev = {
        "meta": {
            "split": "dev",
            "purpose": "DÒ THAM SỐ. Được phép xem và dùng nhiều lần.",
            "seed": SEED,
        },
        "retrieval": old_ret + ret_dev_new,
        "out_of_scope": dev_oos,
        "intent": [{"text": t, "expected": e, "origin": "cũ"} for t, e in OLD_INTENT]
                  + int_dev_new,
    }
    test = {
        "meta": {
            "split": "test",
            "purpose": ("CHỈ ĐỂ BÁO CÁO. Không được dò hay chỉnh bất kỳ tham số nào dựa "
                        "trên kết quả của tập này. Toàn bộ câu ở đây CHƯA TỪNG được dùng "
                        "để dò tham số."),
            "seed": SEED,
        },
        "retrieval": ret_test,
        "out_of_scope": oos_test,
        "intent": int_test,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, data in (("dev", dev), ("test", test)):
        (OUT_DIR / f"{name}.json").write_text(
            json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    def styles(items):
        from collections import Counter
        return dict(Counter(x.get("style", "-") for x in items))

    print(f"DEV : retrieval {len(dev['retrieval']):>3}  {styles(dev['retrieval'])}")
    print(f"      out_of_scope {len(dev['out_of_scope']):>3}   intent {len(dev['intent']):>3}")
    print(f"TEST: retrieval {len(test['retrieval']):>3}  {styles(test['retrieval'])}")
    print(f"      out_of_scope {len(test['out_of_scope']):>3}   intent {len(test['intent']):>3}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
