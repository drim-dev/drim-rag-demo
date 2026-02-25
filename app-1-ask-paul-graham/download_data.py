"""Download Paul Graham essays for RAG pipeline."""

import os
import re
import time

import requests
from bs4 import BeautifulSoup

DATA_DIR = os.path.join(os.path.dirname(__file__), "data", "essays")
ARTICLES_URL = "http://paulgraham.com/articles.html"


def discover_essay_urls() -> list[str]:
    """Scrape the articles index page to get all essay URLs."""
    resp = requests.get(ARTICLES_URL, timeout=15)
    resp.raise_for_status()

    soup = BeautifulSoup(resp.text, "html.parser")
    urls = []

    for link in soup.find_all("a", href=True):
        href = link["href"]
        if not href.endswith(".html"):
            continue
        if href == "articles.html":
            continue

        if href.startswith("http"):
            url = href
        else:
            url = f"http://paulgraham.com/{href}"

        if "paulgraham.com" not in url:
            continue

        if url not in urls:
            urls.append(url)

    return urls


def extract_essay_text(html: str) -> tuple[str, str]:
    """Extract title and body text from a Paul Graham essay page.

    Returns (title, body_text). Paul Graham's site uses a simple HTML
    structure — the main content is inside <font> tags within a <table>.
    """
    soup = BeautifulSoup(html, "html.parser")

    title_tag = soup.find("title")
    title = title_tag.get_text(strip=True) if title_tag else "Untitled"

    body = soup.find("body")
    if not body:
        return title, ""

    for tag in body.find_all(["script", "style", "img"]):
        tag.decompose()

    text = body.get_text(separator="\n")

    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()

    return title, text


def slugify(title: str) -> str:
    """Convert essay title to a safe filename."""
    slug = title.lower()
    slug = re.sub(r"[^a-z0-9]+", "-", slug)
    slug = slug.strip("-")
    return slug[:80] if slug else "untitled"


def download_essays() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    print("Discovering essays from articles page...\n")
    essay_urls = discover_essay_urls()
    print(f"Found {len(essay_urls)} essay links\n")

    downloaded = 0
    skipped = 0
    failed = 0
    total_bytes = 0

    for i, url in enumerate(essay_urls, 1):
        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()

            title, text = extract_essay_text(resp.text)

            if len(text) < 200:
                print(f"  [{i}/{len(essay_urls)}] SKIP (too short): {url}")
                skipped += 1
                continue

            filename = slugify(title) + ".txt"
            filepath = os.path.join(DATA_DIR, filename)

            header = f"Title: {title}\nSource: {url}\n\n"
            content = header + text

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

            size_kb = len(content.encode("utf-8")) / 1024
            total_bytes += len(content.encode("utf-8"))
            downloaded += 1
            print(f"  [{i}/{len(essay_urls)}] OK: {title} ({size_kb:.1f} KB) -> {filename}")

            time.sleep(1)

        except requests.RequestException as e:
            print(f"  [{i}/{len(essay_urls)}] FAIL: {url} — {e}")
            failed += 1

    print(f"\nDone: {downloaded} downloaded, {skipped} skipped, {failed} failed")
    print(f"Total size: {total_bytes / 1024:.1f} KB")
    print(f"Location: {DATA_DIR}")


if __name__ == "__main__":
    print("Downloading Paul Graham essays...\n")
    download_essays()
