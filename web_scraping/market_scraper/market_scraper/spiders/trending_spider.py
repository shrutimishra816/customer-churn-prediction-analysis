"""
pypi_deps_spider.py

Scrapy spider for the same "dependency intelligence" scraping task as
scrape_market_signal.py (which uses requests + BeautifulSoup). This
version demonstrates the Scrapy framework instead — start_requests,
CSS selectors, and Scrapy's built-in CSV export — for projects where a
full crawling framework (many pages, retries, concurrency, pipelines)
is the better fit than a one-off script.

Target: each package's own PyPI project page (pypi.org/project/<name>/).
robots.txt on pypi.org disallows /search*, /simple/, /packages/, etc.,
but not individual /project/<name>/ pages, so ROBOTSTXT_OBEY=True (the
Scrapy default here) still allows this crawl to run.

Run from web_scraping/market_scraper/:
    scrapy crawl pypi_deps -o ../market_signal_scrapy.csv
"""
import scrapy

PACKAGES = [
    "pandas", "numpy", "scikit-learn", "shap", "requests",
    "beautifulsoup4", "selenium", "scrapy", "matplotlib", "seaborn",
]


class PyPIDepsSpider(scrapy.Spider):
    name = "pypi_deps"
    allowed_domains = ["pypi.org"]

    custom_settings = {
        "USER_AGENT": (
            "Mozilla/5.0 (compatible; ChurnResearchBot/1.0; "
            "+https://github.com/shrutimishra816/customer-churn-prediction-analysis)"
        ),
        "ROBOTSTXT_OBEY": True,
        "DOWNLOAD_DELAY": 0.5,
    }

    async def start(self):
        # Scrapy >=2.13 async entry point.
        for package in PACKAGES:
            url = f"https://pypi.org/project/{package}/"
            yield scrapy.Request(url, callback=self.parse, cb_kwargs={"package": package})

    def start_requests(self):
        # Kept for compatibility with Scrapy <2.13, which doesn't call start().
        for package in PACKAGES:
            url = f"https://pypi.org/project/{package}/"
            yield scrapy.Request(url, callback=self.parse, cb_kwargs={"package": package})

    def parse(self, response, package):
        header = response.css("h1.package-header__name::text").get(default="")
        latest_version = header.strip().split()[-1] if header.strip() else "unknown"

        summary = response.css("p.package-description__summary::text").get(default="").strip()
        author = response.css(
            'a[href^="/user/"] span.sidebar-section__user-gravatar-text::text'
        ).get(default="unknown").strip()

        yield {
            "package": package,
            "latest_version": latest_version,
            "summary": summary,
            "author": author,
        }
