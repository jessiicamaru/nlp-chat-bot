"""Đóng gói những gì notebook RAG cần trên Colab thành dist/rag_bundle.zip.

Không đưa cache index (models/*.joblib) vào: file pickle phụ thuộc phiên bản
Python/numpy/scipy, mà Colab chạy phiên bản khác máy này — dựng lại trên Colab
chỉ mất khoảng một phút và an toàn hơn.
"""
import zipfile
from pathlib import Path

PROJ = Path(__file__).resolve().parent.parent
OUT = PROJ / "dist" / "rag_bundle.zip"

INCLUDE = [
    "src/*.py",
    "data/raw/corpus_raw.csv",
    "data/intents/intents_vi.json",
    "data/resources/vietnamese-stopwords.txt",
    "data/resources/teencode_lexicon.json",
    "data/eval/dev.json",
    "data/eval/test.json",
    "data/eval/conflict_case.json",
    "requirements.txt",
]


def main() -> None:
    OUT.parent.mkdir(exist_ok=True)
    files = sorted({p for pattern in INCLUDE for p in PROJ.glob(pattern) if p.is_file()})
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, Path("final-project") / p.relative_to(PROJ))
    size = OUT.stat().st_size / 1024
    print(f"Wrote {OUT}  ({len(files)} file, {size:.0f} KB)")
    for p in files:
        print("  ", p.relative_to(PROJ))


if __name__ == "__main__":
    main()
