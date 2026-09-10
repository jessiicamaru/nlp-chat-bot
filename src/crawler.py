"""
crawler.py — Thu thập bài báo VnExpress để mở rộng corpus.

Kế thừa trực tiếp Lab 02 (requests + BeautifulSoup + validation), nhưng bổ sung:
  - crawl nhiều chuyên mục thay vì một,
  - có delay giữa các request (lịch sự với server),
  - có cache theo URL để chạy lại không tải trùng,
  - ghi log lỗi ra file riêng như Lab 02 yêu cầu.

Chạy:
    .venv/Scripts/python.exe src/crawler.py --per-category 40
"""

from __future__ import annotations

import argparse
import random
import sys
import time
from datetime import datetime, timezone
from urllib.parse import urljoin, urlparse

import pandas as pd
import requests
from bs4 import BeautifulSoup

from config import CORPUS_RAW_PATH, RAW_DIR

# Các chuyên mục của VnExpress dùng làm nguồn tri thức cho chatbot.
CATEGORIES = {
    "Du lịch": "https://vnexpress.net/du-lich",
    "Kinh doanh": "https://vnexpress.net/kinh-doanh",
    "Công nghệ": "https://vnexpress.net/so-hoa",
    "Sức khỏe": "https://vnexpress.net/suc-khoe",
    "Giáo dục": "https://vnexpress.net/giao-duc",
    "Thể thao": "https://vnexpress.net/the-thao",
    "Khoa học": "https://vnexpress.net/khoa-hoc-cong-nghe",
    "Đời sống": "https://vnexpress.net/doi-song",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Accept-Language": "vi-VN,vi;q=0.9",
}

# Selector của một trang bài viết VnExpress (xác định bằng Inspect như Lab 02).
SELECTORS = {
    "title": "h1.title-detail",
    "description": "p.description",
    "text": "article.fck_detail p.Normal",
    "published_at": "span.date",
}

TIMEOUT = 15
DELAY_RANGE = (0.8, 1.6)   # nghỉ ngẫu nhiên giữa 2 request


# ---------------------------------------------------------------------------
# Tải trang
# ---------------------------------------------------------------------------
def fetch(url: str, session: requests.Session) -> str | None:
    try:
        resp = session.get(url, headers=HEADERS, timeout=TIMEOUT)
        if resp.status_code != 200:
            return None
        resp.encoding = resp.apparent_encoding or "utf-8"
        return resp.text
    except requests.RequestException:
        return None


def is_article_url(url: str) -> bool:
    """Bài viết VnExpress có dạng .../slug-<id>.html."""
    if not url or not url.endswith(".html"):
        return False
    path = urlparse(url).path
    slug = path.rsplit("/", 1)[-1]
    stem = slug[: -len(".html")]
    # Phần cuối slug phải là dãy số (id bài viết) -> loại trang chuyên mục, video, ...
    tail = stem.rsplit("-", 1)[-1]
    if not tail.isdigit() or len(tail) < 6:
        return False
    if "/video/" in path or "/podcast/" in path or "/infographic" in path:
        return False
    return True


def collect_article_urls(category_url: str, session: requests.Session, limit: int) -> list[str]:
    """Lấy danh sách URL bài viết từ trang chuyên mục (kể cả trang 2, 3...)."""
    urls: list[str] = []
    seen: set[str] = set()

    for page in range(1, 6):
        page_url = category_url if page == 1 else f"{category_url}-p{page}"
        html = fetch(page_url, session)
        if not html:
            break

        soup = BeautifulSoup(html, "lxml")
        for a in soup.select("a[href]"):
            href = urljoin(category_url, a["href"])
            href = href.split("?")[0].split("#")[0]
            if is_article_url(href) and href not in seen:
                seen.add(href)
                urls.append(href)

        if len(urls) >= limit:
            break
        time.sleep(random.uniform(*DELAY_RANGE))

    return urls[:limit]


# ---------------------------------------------------------------------------
# Parse một bài viết
# ---------------------------------------------------------------------------
def extract_text_or_empty(soup: BeautifulSoup, selector: str) -> str:
    el = soup.select_one(selector)
    return el.get_text(" ", strip=True) if el else ""


def clean_extracted_text(text: str) -> str:
    """Làm sạch cơ bản như Lab 02: gộp whitespace, bỏ ký tự điều khiển."""
    text = text.replace(" ", " ")
    return " ".join(text.split())


def parse_article(html: str, url: str, source: str, category: str) -> dict:
    soup = BeautifulSoup(html, "lxml")

    paragraphs = [p.get_text(" ", strip=True) for p in soup.select(SELECTORS["text"])]
    body = clean_extracted_text(" ".join(paragraphs))

    return {
        "url": url,
        "source": source,
        "category": category,
        "title": clean_extracted_text(extract_text_or_empty(soup, SELECTORS["title"])),
        "description": clean_extracted_text(extract_text_or_empty(soup, SELECTORS["description"])),
        "text": body,
        "published_at": clean_extracted_text(extract_text_or_empty(soup, SELECTORS["published_at"])),
        "crawled_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def validate_article(record: dict, min_text_length: int = 300) -> tuple[bool, str]:
    """Kiểm tra chất lượng như Lab 02. Trả (hợp lệ, lý do loại)."""
    if not record["title"]:
        return False, "thiếu title"
    if not record["text"]:
        return False, "thiếu text"
    if len(record["text"]) < min_text_length:
        return False, f"text quá ngắn ({len(record['text'])} ký tự)"
    return True, ""


# ---------------------------------------------------------------------------
# Vòng crawl chính
# ---------------------------------------------------------------------------
def crawl(per_category: int = 40, categories: dict | None = None) -> pd.DataFrame:
    categories = categories or CATEGORIES
    session = requests.Session()

    records: list[dict] = []
    errors: list[dict] = []

    # Nạp corpus cũ để không tải lại bài đã có.
    existing_urls: set[str] = set()
    if CORPUS_RAW_PATH.exists():
        old = pd.read_csv(CORPUS_RAW_PATH)
        existing_urls = set(old["url"].astype(str))
        records.extend(old.to_dict("records"))
        print(f"Đã có sẵn {len(existing_urls)} bài trong corpus.")

    for category, cat_url in categories.items():
        print(f"\n[{category}] {cat_url}")
        urls = collect_article_urls(cat_url, session, limit=per_category * 2)
        urls = [u for u in urls if u not in existing_urls][:per_category]
        print(f"  tìm thấy {len(urls)} URL mới")

        ok = 0
        for i, url in enumerate(urls, 1):
            html = fetch(url, session)
            if not html:
                errors.append({"url": url, "category": category, "error": "tải thất bại"})
                continue

            record = parse_article(html, url, "VnExpress", category)
            valid, reason = validate_article(record)
            if valid:
                records.append(record)
                existing_urls.add(url)
                ok += 1
            else:
                errors.append({"url": url, "category": category, "error": reason})

            if i % 10 == 0:
                print(f"    {i}/{len(urls)} ... hợp lệ {ok}")
            time.sleep(random.uniform(*DELAY_RANGE))

        print(f"  -> thu được {ok} bài hợp lệ")

    df = pd.DataFrame(records).drop_duplicates(subset=["url"]).reset_index(drop=True)

    if errors:
        err_path = RAW_DIR / "crawl_errors.csv"
        pd.DataFrame(errors).to_csv(err_path, index=False, encoding="utf-8-sig")
        print(f"\nGhi {len(errors)} lỗi vào {err_path}")

    return df


def main() -> int:
    ap = argparse.ArgumentParser(description="Crawl VnExpress mở rộng corpus")
    ap.add_argument("--per-category", type=int, default=40,
                    help="số bài tối đa lấy cho mỗi chuyên mục")
    ap.add_argument("--only", nargs="*", default=None,
                    help="chỉ crawl các chuyên mục này")
    args = ap.parse_args()

    cats = CATEGORIES
    if args.only:
        cats = {k: v for k, v in CATEGORIES.items() if k in args.only}
        if not cats:
            print(f"Không có chuyên mục nào khớp. Chọn trong: {list(CATEGORIES)}")
            return 1

    df = crawl(per_category=args.per_category, categories=cats)
    CORPUS_RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CORPUS_RAW_PATH, index=False, encoding="utf-8-sig")

    print(f"\nTổng cộng {len(df)} bài -> {CORPUS_RAW_PATH}")
    print(df["category"].value_counts().to_string())
    return 0


if __name__ == "__main__":
    sys.exit(main())
