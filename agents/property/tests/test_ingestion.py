import logging
from pathlib import Path
from unittest.mock import MagicMock

from agents.property.gatherers.pl_housing_data_collector import ingest
from agents.property.scraping.url_processor import UrlProcessor
from core.src.config.config import Config

ROOT = Path(__file__).resolve().parents[3]


def _collector():
    collector = MagicMock()
    collector.stream_csv_files.side_effect = lambda d: iter([{"id": i, "city": "Warsaw"} for i in range(5)])
    collector.generateEmbededDocument.side_effect = lambda row: f"flat {row['id']}"
    return collector


def test_housing_dataset_dir_resolves_inside_the_project():
    assert Path(Config.HOUSING_DATASET_DIR) == ROOT / "agents/property/datasets/pl-housing"


def test_ingest_persists_rows_in_batches(tmp_path):
    (tmp_path / "a.csv").write_text("id\n1\n")
    collector = _collector()
    assert ingest(collector, tmp_path, batch_size=2) == 5
    batches = [list(call.args[0]) for call in collector.persist.call_args_list]
    assert batches == [[0, 1], [2, 3], [4]]
    collector.stream_csv_files.assert_called_once_with(tmp_path)


def test_ingest_without_csv_files_says_where_it_looked(tmp_path, caplog):
    collector = _collector()
    with caplog.at_level(logging.ERROR):
        assert ingest(collector, tmp_path / "missing") == 0
    assert str(tmp_path / "missing") in caplog.text
    collector.persist.assert_not_called()


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
