import logging
from pathlib import Path
from unittest.mock import MagicMock

from agents.property.scraping.url_processor import UrlProcessor

ROOT = Path(__file__).resolve().parents[3]


def test_url_file_skips_comments_and_blank_lines(tmp_path):
    urls = tmp_path / "urls.txt"
    urls.write_text("# one URL per line\n\nhttps://a.example/flat\n  # indented comment\nhttps://b.example\n")
    scraper = MagicMock()
    scraper.scrape.return_value = ""
    UrlProcessor(scraper).process_urls_from_file(str(urls))
    assert [c.args[0] for c in scraper.scrape.call_args_list] == ["https://a.example/flat", "https://b.example"]


def test_missing_url_file_is_reported(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        assert UrlProcessor(MagicMock()).process_urls_from_file(str(tmp_path / "urls.txt")) == []
    assert "urls.txt" in caplog.text


def test_shipped_url_file_holds_no_urls_yet():
    lines = [ln.strip() for ln in (ROOT / "agents/property/urls.txt").read_text().splitlines()]
    assert all(not ln or ln.startswith("#") for ln in lines)
