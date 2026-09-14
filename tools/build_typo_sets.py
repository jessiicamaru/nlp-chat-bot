"""
build_typo_sets.py — Sinh tập truy vấn GÕ SAI CHÍNH TẢ từ tập dev/test có sẵn.

## Vì sao cần

Người dùng báo lỗi thật: "thám hiểm Sơn Dòng" (gõ thiếu một chữ o và quên dấu
gạch của Đ) thì bot từ chối, còn "thám hiểm Sơn Đoòng" thì trả lời đúng. Tập
dev/test hiện có chỉ gồm câu gõ đúng chính tả nên không đo được lỗi này — muốn
sửa thì trước hết phải có "ca kiểm thử đang trượt", giống viết test đỏ trước
khi sửa code.

## Cách sinh

Mỗi truy vấn truy hồi của dev (và test) sinh đúng MỘT biến thể gõ sai, trên MỘT
âm tiết nội dung (không phải từ khung như "tin", "cho", không phải stopword),
theo các kiểu lỗi gõ phím tiếng Việt thường gặp:

    sai_dau_thanh   hỏa -> hóa, ninh -> nình      (nhầm/thiếu/thừa dấu thanh)
    mat_dau_phu     đ -> d, ô -> o, ư -> u, ă -> a  (quên dấu mũ/móc/gạch)
    thieu_chu       iphone -> iphne                (gõ thiếu một chữ)
    lap_chu         visa -> vissa                  (gõ thừa một chữ)
    dao_chu         venice -> vneice               (đảo hai chữ cạnh nhau)
    phim_ke_ben     giá -> hiá                     (chạm nhầm phím bên cạnh)

Nhãn đúng (gold_urls) giữ nguyên từ câu gốc — câu gõ sai vẫn hỏi đúng bài đó.
Hạt giống cố định nên chạy lại cho ra đúng cùng tập.

Quy tắc dev/test giữ nguyên: dev_typo dùng để DÒ, test_typo chỉ BÁO CÁO một lần.
Câu lỗi thật của người dùng ("thám hiểm Sơn Dòng") nằm trong dev_typo, vì chính
nó đã được nhìn khi thiết kế cách sửa.

Chạy:  .venv/Scripts/python.exe tools/build_typo_sets.py
"""

import json
import random
import sys
import unicodedata
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJ / "src"))

from config import QUERY_FRAME_WORDS  # noqa: E402
from preprocess import load_stopwords  # noqa: E402

EVAL_DIR = PROJ / "data" / "eval"
SEED = 2026

# Dấu thanh ở dạng tổ hợp (NFD).
TONES = ["\u0300", "\u0301", "\u0303", "\u0309", "\u0323"]   # huyền sắc ngã hỏi nặng
# Dấu phụ của chữ cái: mũ (â ê ô), trăng (ă), móc (ơ ư).
MODIFIERS = {"\u0302", "\u0306", "\u031b"}

KEYBOARD_ROWS = ["qwertyuiop", "asdfghjkl", "zxcvbnm"]
NEIGHBORS: dict[str, str] = {}
for row_i, row in enumerate(KEYBOARD_ROWS):
    for col, ch in enumerate(row):
        near = [row[c] for c in (col - 1, col + 1) if 0 <= c < len(row)]
        for other in (row_i - 1, row_i + 1):
            if 0 <= other < len(KEYBOARD_ROWS):
                near += [KEYBOARD_ROWS[other][c] for c in (col - 1, col, col + 1)
                         if 0 <= c < len(KEYBOARD_ROWS[other])]
        NEIGHBORS[ch] = "".join(near)

# Câu lỗi thật người dùng gặp trên giao diện web (14/09/2026).
USER_REPORTED = [
    {
        "query": "thám hiểm Sơn Dòng",
        "original": "thám hiểm Sơn Đoòng",
        "gold_urls": ["https://vnexpress.net/chiem-nghiem-tu-cuoc-tham-hiem-son-doong-phut-89-cua-ceo-viet-5119565.html"],
        "gold_titles": ["Chiêm nghiệm từ cuộc thám hiểm Sơn Đoòng 'phút 89' của CEO Việt"],
        "style": "có dấu",
        "typo": {"kind": "nguoi_dung_bao_loi", "from": "Đoòng", "to": "Dòng"},
    }
]


# ---------------------------------------------------------------------------
# Các phép gây lỗi trên MỘT âm tiết. Trả None nếu không áp dụng được.
# ---------------------------------------------------------------------------
def _letters(syl: str) -> list[list[str]]:
    """Tách âm tiết thành từng chữ cái kèm các dấu tổ hợp đi theo nó."""
    groups: list[list[str]] = []
    for ch in unicodedata.normalize("NFD", syl):
        if unicodedata.combining(ch) and groups:
            groups[-1].append(ch)
        else:
            groups.append([ch])
    return groups


def _join(groups: list[list[str]]) -> str:
    return unicodedata.normalize("NFC", "".join("".join(g) for g in groups))


def sai_dau_thanh(syl: str, rng: random.Random) -> str | None:
    groups = _letters(syl)
    vowel_idx = [i for i, g in enumerate(groups) if g[0].lower() in "aeiouy"]
    if not vowel_idx:
        return None
    toned = [i for i in vowel_idx if any(m in TONES for m in groups[i][1:])]
    if toned:
        i = toned[0]
        current = next(m for m in groups[i][1:] if m in TONES)
        choice = rng.choice([t for t in TONES if t != current] + [""])
        groups[i] = [c for c in groups[i] if c not in TONES] + ([choice] if choice else [])
    else:
        # Âm tiết không dấu thanh -> gõ thừa một dấu thanh.
        i = rng.choice(vowel_idx)
        groups[i] = groups[i] + [rng.choice(TONES)]
    return _join(groups)


def mat_dau_phu(syl: str, rng: random.Random) -> str | None:
    groups = _letters(syl)
    cand = [i for i, g in enumerate(groups)
            if g[0] in "đĐ" or any(m in MODIFIERS for m in g[1:])]
    if not cand:
        return None
    i = rng.choice(cand)
    if groups[i][0] in "đĐ":
        groups[i] = ["d" if groups[i][0] == "đ" else "D"] + groups[i][1:]
    else:
        groups[i] = [c for c in groups[i] if c not in MODIFIERS]
    return _join(groups)


def thieu_chu(syl: str, rng: random.Random) -> str | None:
    groups = _letters(syl)
    if len(groups) < 3:
        return None
    i = rng.randrange(len(groups))
    return _join(groups[:i] + groups[i + 1:])


def lap_chu(syl: str, rng: random.Random) -> str | None:
    groups = _letters(syl)
    # Chỉ lặp chữ KHÔNG mang dấu: "ưư", "ồồ" không phải lỗi gõ thực tế.
    cand = [i for i, g in enumerate(groups) if len(g) == 1]
    if not cand:
        return None
    i = rng.choice(cand)
    return _join(groups[:i + 1] + [list(groups[i])] + groups[i + 1:])


def dao_chu(syl: str, rng: random.Random) -> str | None:
    groups = _letters(syl)
    pairs = [i for i in range(len(groups) - 1) if groups[i][0].lower() != groups[i + 1][0].lower()]
    if not pairs:
        return None
    i = rng.choice(pairs)
    groups[i], groups[i + 1] = groups[i + 1], groups[i]
    return _join(groups)


def phim_ke_ben(syl: str, rng: random.Random) -> str | None:
    groups = _letters(syl)
    # Chỉ thay chữ KHÔNG mang dấu, nếu không dấu sẽ dính vào phụ âm ("mh̀a").
    cand = [i for i, g in enumerate(groups) if len(g) == 1 and g[0].lower() in NEIGHBORS]
    if not cand:
        return None
    i = rng.choice(cand)
    base = groups[i][0]
    new = rng.choice(NEIGHBORS[base.lower()])
    groups[i] = [new.upper() if base.isupper() else new] + groups[i][1:]
    return _join(groups)


OPERATIONS = [sai_dau_thanh, mat_dau_phu, thieu_chu, lap_chu, dao_chu, phim_ke_ben]


def content_syllables(query: str) -> list[int]:
    """Vị trí các âm tiết mang nội dung (đủ dài, không là từ khung/stopword)."""
    stop = load_stopwords()
    frame = {s for w in QUERY_FRAME_WORDS for s in w.split("_")}
    out = []
    for i, syl in enumerate(query.split()):
        low = syl.lower()
        if len(low) >= 3 and low.isalpha() and low not in frame and low not in stop:
            out.append(i)
    return out


def make_typo(query: str, rng: random.Random) -> tuple[str, dict] | None:
    words = query.split()
    positions = content_syllables(query)
    if not positions:
        return None
    accented = any(unicodedata.normalize("NFD", w) != w for w in words)
    for _ in range(50):
        pos = rng.choice(positions)
        ops = OPERATIONS if accented else [thieu_chu, lap_chu, dao_chu, phim_ke_ben]
        op = rng.choice(ops)
        new = op(words[pos], rng)
        if new and new != words[pos]:
            changed = words[:pos] + [new] + words[pos + 1:]
            return " ".join(changed), {"kind": op.__name__, "from": words[pos], "to": new}
    return None


def build(split: str, rng: random.Random) -> dict:
    data = json.loads((EVAL_DIR / f"{split}.json").read_text(encoding="utf-8"))
    cases = list(USER_REPORTED) if split == "dev" else []
    for c in data["retrieval"]:
        made = make_typo(c["query"], rng)
        if made is None:
            continue
        query, info = made
        cases.append({
            "query": query,
            "original": c["query"],
            "gold_urls": c["gold_urls"],
            "gold_titles": c["gold_titles"],
            "style": c["style"],
            "typo": info,
        })
    return {
        "meta": {
            "split": f"{split}_typo",
            "purpose": ("DÒ THAM SỐ chống gõ sai." if split == "dev"
                        else "CHỈ BÁO CÁO, một lần, với tham số đã chốt trên dev_typo."),
            "seed": SEED,
            "source": f"data/eval/{split}.json (mỗi truy vấn một lỗi gõ trên một âm tiết nội dung)",
        },
        "retrieval": cases,
    }


def main() -> int:
    for split in ("dev", "test"):
        # Hạt giống riêng cho từng tập để thêm/bớt câu ở tập này không đổi tập kia.
        rng = random.Random(f"{SEED}-{split}")
        out = build(split, rng)
        path = EVAL_DIR / f"{split}_typo.json"
        path.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
        kinds: dict[str, int] = {}
        for c in out["retrieval"]:
            kinds[c["typo"]["kind"]] = kinds.get(c["typo"]["kind"], 0) + 1
        print(f"{path.name}: {len(out['retrieval'])} câu  {kinds}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
