"""
web_scraping/scrape_market_signal.py

Web scraping module added to the Churn Prediction project to demonstrate
the data-collection side of the pipeline (BeautifulSoup / Selenium),
alongside the existing SQL exploration and modeling layers.

Motivation for this project specifically:
Churn models here are trained on internal customer data only. In a real
deployment you'd also want an external "market pressure" signal -- e.g.
how much competitor/industry activity is happening in a given period --
to add as a feature or as context in the retention report. This script
scrapes a live, public source and produces a tidy CSV that can be joined
against the customer dataset by date/segment.

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

# Live, public, scrape-friendly source used to demonstrate the pipeline.
# In production this would point at a competitor pricing/plans page or a
# market-activity feed relevant to the business; swap SOURCE_URL and the
# parsing logic in `parse_listing` for that target.
SOURCE_URL = "https://github.com/trending/python?since=daily"


@dataclass
class ListingRecord:
    name: str
    description: str
    stars_today: str


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


def parse_listing(html: str) -> list[ListingRecord]:
    """Parse repo cards into structured records (name, description, stars gained today)."""
    soup = BeautifulSoup(html, "html.parser")
    records: list[ListingRecord] = []

    for article in soup.select("article.Box-row"):
        title_tag = article.select_one("h2 a")
        if not title_tag:
            continue
        name = title_tag.get_text(strip=True).replace("\n", "").replace(" ", "")

        desc_tag = article.select_one("p")
        description = desc_tag.get_text(strip=True) if desc_tag else ""

        stars_tag = article.select_one("span.d-inline-block.float-sm-right")
        stars_today = stars_tag.get_text(strip=True) if stars_tag else "0"

        records.append(ListingRecord(name=name, description=description, stars_today=stars_today))

    return records


def save_to_csv(records: list[ListingRecord], path: str) -> None:
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
    html = fetch_html(SOURCE_URL)
    records = parse_listing(html)
    save_to_csv(records, "web_scraping/market_signal.csv")


if __name__ == "__main__":
    main()
