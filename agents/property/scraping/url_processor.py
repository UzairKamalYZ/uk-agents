import logging
import os
import re

from .web_scraper import WebScraper

logger = logging.getLogger(__name__)

class UrlProcessor:
    """Scrapes URLs and extracts property listings."""

    def __init__(self, web_scraper: WebScraper):
        self.web_scraper = web_scraper

    def process_urls_from_file(self, file_path: str) -> list[dict]:
        listings = []

        if not os.path.exists(file_path):
            logger.warning("url-processor: %s not found — no URLs to scrape (set URLS_FILE)", file_path)
            return listings

        with open(file_path, "r") as f:
            # One URL per line; blank lines and "#" comments are skipped
            urls = [u.strip() for u in f.readlines() if u.strip() and not u.strip().startswith("#")]

        for url in urls:
            print(f"----------Scraping----->>>>> {url}...")
            content = self.web_scraper.scrape(url)

            if not content:
                print(f"Failed to scrape {url}")
                continue

            extracted = self.extract_listings(content, url)

            listings.extend(extracted)

            print(f"Extracted {len(extracted)} listings from {url}")

        return listings

    def extract_listings(self, content: str, source_url: str) -> list[dict]:
        listings = []
        # Split content into candidate blocks
        blocks = re.split(r"\n\s*\n", content)

        for block in blocks:
            # Heuristic: must contain a rent indicator
            if not re.search(r"€|\beur\b", block.lower()):
                continue

            rent = self._extract_rent(block)
            if rent is None:
                continue

            listing = {
                "type": self._extract_property_type(block),
                "city": self._extract_city(block),
                "rent": rent,
                "description": self._extract_description(block),
                "url": source_url,
            }

            listings.append(listing)

        return listings

    def _extract_rent(self, text: str) -> int | None:
        match = re.search(r"€\s?(\d{3,5})", text)
        if match:
            return int(match.group(1))
        return None


    def _extract_property_type(self, text: str) -> str:
        text = text.lower()
        if "house" in text or "maison" in text:
            return "House"
        if "apartment" in text or "appartement" in text or "flat" in text:
            return "Apartment"
        return "Apartment"


    def _extract_city(self, text: str) -> str:
        cities = [
            # Belgium
            "brussels", "antwerp", "ghent", "bruges", "leuven", "liège", "namur",
            # Poland
            "warsaw", "kraków", "krakow", "łódź", "lodz", "wrocław", "wroclaw",
            "poznań", "poznan", "gdańsk", "gdansk", "szczecin", "bydgoszcz",
            "lublin", "białystok", "bialystok", "katowice", "gdynia", "częstochowa",
            "radom", "sosnowiec", "toruń", "torun", "kielce", "rzeszów", "rzeszow",
            "gliwice", "zabrze", "olsztyn", "bytom", "bielsko-biała", "bielsko-biala",
        ]

        text_lower = text.lower()
        for city in cities:
            if city in text_lower:
                return city.capitalize()

        return "Unknown"


    def _extract_description(self, text: str) -> str:
        # Keep it short and clean
        cleaned = re.sub(r"\s+", " ", text).strip()
        return cleaned[:160]
