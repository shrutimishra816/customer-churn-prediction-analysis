"""
web_scraping/scrape_market_signal.py

Web scraping module added to the Churn Prediction project to demonstrate
the data-collection side of the pipeline (BeautifulSoup / Selenium),
alongside the existing SQL exploration and modeling layers.

What it scrapes:
This project's own Python dependency stack (pandas, scikit-learn, shap,
requests, beautifulsoup4, selenium, scrapy, matplotlib, seaborn) — pulling
each package's live PyPI project page for its current version, summary,
and author. It's a small, self-contained "dependency intelligence" report
(useful for spotting outdated pins before a release) and, more broadly,
a template for scraping any list of target pages politely and reliably.

robots.txt: pypi.org's robots.txt disallows /search*, /simple/, /packages/,
etc., but NOT individual /project/<name>/ pages, so this script's target
is allowed to be crawled.

Libraries: requests + BeautifulSoup for static HTML (used below).
A Selenium fallback class is included for pages that render their
content via JavaScript and can't be parsed from the raw HTML response.

Usage:
    python web_scraping/scrape_market_signal.py

Output:
    web_scraping/market_signal.csv
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, asdict

import requests
from bs4 import BeautifulSoup

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; ChurnResearchBot/1.0; "
        "+https://github.com/shrutimishra816/customer-churn-prediction-analysis)"
    )
}

# This project's own dependencies (see requirements.txt) — a self-referential,
# always-relevant scrape target instead of a hardcoded arbitrary page.
PACKAGES = [
    "pandas", "numpy", "scikit-learn", "shap", "requests",
    "beautifulsoup4", "selenium", "scrapy", "matplotlib", "seaborn",
]

PROJECT_URL_TMPL = "https://pypi.org/project/{package}/"


@dataclass
class PackageRecord:
    package: str
    latest_version: str
    summary: str
    author: str


def fetch_html(url: str, retries: int = 3, backoff: float = 1.5) -> str:
    """GET a page with basic retry/backoff, politely identified via User-Agent."""
    last_exc = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            return resp.text
        except requests.RequestException as exc:
            last_exc = exc
            time.sleep(backoff * attempt)
    raise RuntimeError(f"Failed to fetch {url} after {retries} attempts") from last_exc


def parse_package_page(html: str, package: str) -> PackageRecord:
    soup = BeautifulSoup(html, "html.parser")

    header = soup.select_one("h1.package-header__name")
    name_and_version = header.get_text(strip=True) if header else f"{package} unknown"
    latest_version = name_and_version.split()[-1] if name_and_version else "unknown"

    summary_tag = soup.select_one("p.package-description__summary")
    summary = summary_tag.get_text(strip=True) if summary_tag else ""

    author_tag = soup.select_one('a[href^="/user/"]')
    author = author_tag.get_text(strip=True) if author_tag else "unknown"

    return PackageRecord(
        package=package, latest_version=latest_version, summary=summary, author=author
    )


def save_to_csv(records: list[PackageRecord], path: str) -> None:
    if not records:
        print("No records scraped — nothing to save.")
        return
    fieldnames = list(asdict(records[0]).keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(asdict(r))
    print(f"Saved {len(records)} records to {path}")


class SeleniumFallbackScraper:
    """
    Fallback for pages that require JS rendering (dynamic tables, infinite
    scroll, "load more" buttons, etc.) and can't be parsed from a plain
    requests.get() response. Not used by default in this project — kept
    here to show the Selenium path alongside BeautifulSoup/Scrapy.
    """

    def __init__(self, headless: bool = True):
        self.headless = headless

    def scrape(self, url: str, wait_selector: str, timeout: int = 10) -> str:
        from selenium import webdriver
        from selenium.webdriver.common.by import By
        from selenium.webdriver.support.ui import WebDriverWait
        from selenium.webdriver.support import expected_conditions as EC
        from selenium.webdriver.chrome.options import Options

        options = Options()
        if self.headless:
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")

        driver = webdriver.Chrome(options=options)
        try:
            driver.get(url)
            WebDriverWait(driver, timeout).until(
                EC.presence_of_element_located((By.CSS_SELECTOR, wait_selector))
            )
            return driver.page_source
        finally:
            driver.quit()


def main() -> None:
    records = []
    for package in PACKAGES:
        url = PROJECT_URL_TMPL.format(package=package)
        html = fetch_html(url)
        records.append(parse_package_page(html, package))
        time.sleep(0.5)  # be polite between requests

    save_to_csv(records, "web_scraping/market_signal.csv")


if __name__ == "__main__":
    main()
