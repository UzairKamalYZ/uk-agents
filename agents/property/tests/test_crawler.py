import json
import logging
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from agents.property.crawler.crawler import PropertyCrawler
from agents.property.crawler.extractor import Listing, extract_listings, next_page_url, parse_number
from agents.property.crawler.fetcher import PlaywrightFetcher, RobotsChecker
from agents.property.crawler.indexer import ListingIndexer
from agents.property.crawler.main import CrawlerService
from agents.property.crawler.seen_store import SeenStore
from agents.property.crawler.sites import Site, FieldSelector, load_sites
from core.src.model.context_builder import build_context_from_listings

ROOT = Path(__file__).resolve().parents[3]


def _site(**overrides) -> Site:
    values = dict(
        name="portal",
        start_urls=["https://portal.example/rent?page=1"],
        listing_selector="article.offer",
        fields={
            "title": FieldSelector("h2"),
            "price": FieldSelector(".price"),
            "rooms": FieldSelector(".rooms"),
            "area": FieldSelector(".area"),
            "city": FieldSelector(".city"),
        },
        next_page_selector="a.next",
        max_pages=5,
        city="Warsaw",
        property_type="Apartment",
    )
    values.update(overrides)
    return Site(**values)


def _card(n: int, price: str = "3 500 zł", city: str = "") -> str:
    return (f'<article class="offer"><a href="/offer/{n}"><h2>Flat {n}</h2></a>'
            f'<span class="price">{price}</span><span class="rooms">2 rooms</span>'
            f'<span class="area">48,5 m²</span><span class="city">{city}</span></article>')


def _page(cards: list[str], next_href: str | None = None) -> str:
    nxt = f'<a class="next" href="{next_href}">Next</a>' if next_href else ""
    return f"<html><body>{''.join(cards)}{nxt}</body></html>"


# --- extraction -------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("3 500 zł", 3500), ("€1,250.50", 1250.5), ("1.200 PLN", 1200), ("48,5 m²", 48.5),
    ("2 rooms", 2), ("PLN 4 200 / month", 4200), ("ask for price", None), ("", None), (None, None),
])
def test_parse_number_reads_european_and_english_formats(text, expected):
    assert parse_number(text) == expected


def test_extract_listings_reads_fields_and_makes_links_absolute():
    [listing] = extract_listings(_page([_card(7, city="Kraków")]), _site(), "https://portal.example/rent?page=1")
    assert listing.url == "https://portal.example/offer/7"
    assert (listing.title, listing.price, listing.rooms, listing.area) == ("Flat 7", 3500, 2, 48.5)
    assert listing.city == "Kraków"
    assert listing.property_type == "Apartment"
    assert "Flat 7. Apartment in Kraków. Price 3500 PLN. 2 rooms. 48.5 square meters." in listing.text


def test_extract_listings_falls_back_to_the_site_city_and_counts_a_repeated_ad_once():
    listings = extract_listings(_page([_card(1), _card(1)]), _site(), "https://portal.example/rent")
    assert len(listings) == 1
    assert listings[0].city == "Warsaw"


def test_card_without_a_link_gets_a_stable_url_from_its_text():
    html = _page(['<article class="offer"><h2>No link</h2><span class="price">900 €</span></article>'])
    first = extract_listings(html, _site(), "https://portal.example/rent")
    again = extract_listings(html, _site(), "https://portal.example/rent")
    assert first[0].url.startswith("https://portal.example/rent#listing-")
    assert first[0].id == again[0].id


def test_unknown_fields_are_kept_as_extra_metadata():
    site = _site(fields={"title": FieldSelector("h2"), "floor": FieldSelector(".floor"),
                         "photo": FieldSelector("img", attr="src")})
    html = _page(['<article class="offer"><a href="/o/1"><h2>A</h2></a>'
                  '<span class="floor">3/5</span><img src="/p.jpg"></article>'])
    [listing] = extract_listings(html, site, "https://portal.example/")
    assert listing.extra == {"floor": "3/5", "photo": "/p.jpg"}
    assert listing.metadata("now")["floor"] == "3/5"


def test_metadata_uses_the_keys_the_property_agent_reads_and_drops_empty_values():
    [listing] = extract_listings(_page([_card(3)]), _site(), "https://portal.example/")
    meta = listing.metadata("2026-11-25T10:00:00+00:00")
    assert meta["squareMeters"] == 48.5 and meta["price"] == 3500 and meta["source"] == "crawler"
    assert meta["text"] == listing.text and meta["url"] == listing.url
    assert all(v not in (None, "") for v in meta.values())


def test_next_page_url_is_absolute_and_none_on_the_last_page():
    site = _site()
    assert next_page_url(_page([], "?page=2"), site, "https://portal.example/rent?page=1") == \
        "https://portal.example/rent?page=2"
    assert next_page_url(_page([]), site, "https://portal.example/rent") is None
    assert next_page_url(_page([], "javascript:void(0)"), site, "https://portal.example/") is None


# --- site list --------------------------------------------------------------

def test_load_sites_skips_disabled_and_malformed_sites(tmp_path, caplog):
    path = tmp_path / "sites.json"
    path.write_text(json.dumps({"sites": [
        {"name": "ok", "url": "https://a.example", "listing_selector": ".x",
         "fields": {"price": ".p", "photo": {"selector": "img", "attr": "src"}}},
        {"name": "off", "start_urls": ["https://b.example"], "listing_selector": ".x", "enabled": False},
        {"name": "broken"},
    ]}))
    with caplog.at_level(logging.ERROR):
        sites = load_sites(path)
    assert [s.name for s in sites] == ["ok"]
    assert sites[0].start_urls == ["https://a.example"]
    assert sites[0].wait_for == ".x"
    assert sites[0].fields["photo"] == FieldSelector("img", "src")
    assert "site #3" in caplog.text


def test_missing_sites_file_is_reported(tmp_path, caplog):
    with caplog.at_level(logging.WARNING):
        assert load_sites(tmp_path / "sites.json") == []
    assert "sites.json" in caplog.text


def test_shipped_sites_file_parses_and_crawls_nothing_until_a_site_is_enabled():
    raw = json.loads((ROOT / "agents/property/crawler/sites.json").read_text())
    assert raw["sites"] and all(not s.get("enabled", True) for s in raw["sites"])
    assert load_sites(ROOT / "agents/property/crawler/sites.json") == []


# --- seen store -------------------------------------------------------------

def _listing(n: int, price: float = 1000) -> Listing:
    return Listing(site="portal", url=f"https://portal.example/offer/{n}", title=f"Flat {n}", price=price)


def test_seen_store_returns_only_new_or_changed_listings(tmp_path):
    with SeenStore(tmp_path / "sub" / "seen.db") as store:
        store.touch([_listing(1), _listing(2)])
        fresh = store.unseen([_listing(1), _listing(2, price=900), _listing(3)])
        assert [l.url[-1] for l in fresh] == ["2", "3"]
        assert store.count("portal") == 2


def test_seen_store_survives_a_restart(tmp_path):
    with SeenStore(tmp_path / "seen.db") as store:
        store.touch([_listing(1)])
    with SeenStore(tmp_path / "seen.db") as store:
        assert store.unseen([_listing(1)]) == []


# --- crawler ----------------------------------------------------------------

class FakeFetcher:
    def __init__(self, pages: dict[str, str], fail: set[str] = frozenset()):
        self.pages, self.fail = pages, fail
        self.fetched: list[str] = []
        self.opened = self.closed = 0
        self.sites = 0

    def open(self):
        self.opened += 1
        return self

    def new_site(self):
        self.sites += 1

    def fetch(self, url, wait_for=None, scrolls=0):
        self.fetched.append(url)
        if url in self.fail:
            raise RuntimeError("navigation failed")
        return self.pages[url]

    def close(self):
        self.closed += 1


class FakeIndexer:
    def __init__(self, fail: bool = False):
        self.indexed: list[Listing] = []
        self.fail = fail

    def index(self, listings):
        if self.fail:
            raise RuntimeError("pinecone down")
        self.indexed.extend(listings)
        return len(listings)


BASE = "https://portal.example/rent?page="


def _three_pages() -> dict[str, str]:
    return {
        BASE + "1": _page([_card(1), _card(2)], "?page=2"),
        BASE + "2": _page([_card(3)], "?page=3"),
        BASE + "3": _page([_card(4)]),
    }


def _crawler(tmp_path, sites, fetcher, indexer, **kw) -> PropertyCrawler:
    store = SeenStore(tmp_path / "seen.db")
    return PropertyCrawler(sites, fetcher, store, indexer, page_delay_seconds=0, sleep=lambda s: None, **kw)


def test_crawler_follows_pages_and_indexes_every_listing(tmp_path):
    fetcher, indexer = FakeFetcher(_three_pages()), FakeIndexer()
    report = _crawler(tmp_path, [_site()], fetcher, indexer).run_once()
    assert fetcher.fetched == [BASE + "1", BASE + "2", BASE + "3"]
    assert [l.title for l in indexer.indexed] == ["Flat 1", "Flat 2", "Flat 3", "Flat 4"]
    assert report.indexed == 4 and report.sites[0].pages == 3
    assert (fetcher.opened, fetcher.closed) == (1, 1)


def test_second_pass_indexes_only_new_listings_and_stops_paging_at_known_ones(tmp_path):
    pages = _three_pages()
    _crawler(tmp_path, [_site()], FakeFetcher(pages), FakeIndexer()).run_once()

    pages[BASE + "1"] = _page([_card(9), _card(1)], "?page=2")   # one new ad on top
    fetcher, indexer = FakeFetcher(pages), FakeIndexer()
    _crawler(tmp_path, [_site()], fetcher, indexer).run_once()
    assert [l.title for l in indexer.indexed] == ["Flat 9"]
    assert fetcher.fetched == [BASE + "1", BASE + "2"]   # page 2 held nothing new → stop


def test_a_price_change_is_reindexed(tmp_path):
    pages = {BASE + "1": _page([_card(1)])}
    _crawler(tmp_path, [_site()], FakeFetcher(pages), FakeIndexer()).run_once()
    pages[BASE + "1"] = _page([_card(1, price="3 200 zł")])
    indexer = FakeIndexer()
    _crawler(tmp_path, [_site()], FakeFetcher(pages), indexer).run_once()
    assert [l.price for l in indexer.indexed] == [3200]


def test_max_pages_and_page_loops_end_the_crawl(tmp_path):
    loop = {BASE + "1": _page([_card(1)], "?page=2"), BASE + "2": _page([_card(2)], "?page=1")}
    fetcher = FakeFetcher(loop)
    _crawler(tmp_path, [_site(max_pages=10)], fetcher, FakeIndexer()).run_once()
    assert fetcher.fetched == [BASE + "1", BASE + "2"]

    fetcher = FakeFetcher(_three_pages())
    _crawler(tmp_path / "b", [_site(max_pages=2)], fetcher, FakeIndexer()).run_once()
    assert fetcher.fetched == [BASE + "1", BASE + "2"]


def test_one_failing_site_does_not_stop_the_next(tmp_path):
    other = _site(name="other", start_urls=["https://other.example/"], next_page_selector=None)
    pages = {"https://other.example/": _page([_card(5)])}
    fetcher, indexer = FakeFetcher(pages, fail={BASE + "1"}), FakeIndexer()
    report = _crawler(tmp_path, [_site(), other], fetcher, indexer).run_once()
    assert "navigation failed" in report.sites[0].error
    assert report.sites[1].indexed == 1 and fetcher.sites == 2


def test_listings_are_not_marked_seen_when_indexing_fails(tmp_path):
    pages = {BASE + "1": _page([_card(1)])}
    report = _crawler(tmp_path, [_site()], FakeFetcher(pages), FakeIndexer(fail=True)).run_once()
    assert "pinecone down" in report.sites[0].error
    indexer = FakeIndexer()
    _crawler(tmp_path, [_site()], FakeFetcher(pages), indexer).run_once()
    assert len(indexer.indexed) == 1   # retried on the next pass


def test_robots_disallowed_pages_are_skipped(tmp_path, caplog):
    robots = MagicMock()
    robots.allowed.return_value = False
    fetcher = FakeFetcher(_three_pages())
    with caplog.at_level(logging.WARNING):
        _crawler(tmp_path, [_site()], fetcher, FakeIndexer(), robots=robots).run_once()
    assert fetcher.fetched == []
    assert "robots.txt disallows" in caplog.text


def test_stop_request_ends_the_pass_before_the_next_site(tmp_path):
    fetcher = FakeFetcher(_three_pages())
    _crawler(tmp_path, [_site()], fetcher, FakeIndexer(), should_stop=lambda: True).run_once()
    assert fetcher.fetched == [] and fetcher.closed == 1


def test_no_sites_means_no_browser(tmp_path):
    fetcher = FakeFetcher({})
    _crawler(tmp_path, [], fetcher, FakeIndexer()).run_once()
    assert fetcher.opened == 0


# --- robots.txt -------------------------------------------------------------

def test_robots_checker_reads_robots_txt_once_per_host():
    response = MagicMock(status_code=200, text="User-agent: *\nDisallow: /private\n")
    with patch("agents.property.crawler.fetcher.requests.get", return_value=response) as get:
        robots = RobotsChecker()
        assert robots.allowed("https://a.example/rent")
        assert not robots.allowed("https://a.example/private/1")
    get.assert_called_once()


def test_unreachable_robots_txt_allows_crawling():
    import requests
    with patch("agents.property.crawler.fetcher.requests.get", side_effect=requests.ConnectionError("down")):
        assert RobotsChecker().allowed("https://a.example/rent")


# --- indexer ----------------------------------------------------------------

def test_indexer_upserts_in_batches_with_listing_metadata():
    embedder = MagicMock()
    embedder.embed_documents_to_vectors.side_effect = lambda docs: list(docs)
    listings = [_listing(n) for n in range(5)]
    assert ListingIndexer(embedder, batch_size=2).index(listings) == 5
    batches = [c.args[0] for c in embedder.embed_documents_to_vectors.call_args_list]
    assert [len(b) for b in batches] == [2, 2, 1]
    doc = batches[0][listings[0].id]
    assert doc["text"] == listings[0].text and doc["url"] == listings[0].url
    assert embedder.save_vectors_in_store.call_count == 3


# --- service ----------------------------------------------------------------

def test_service_runs_a_pass_with_the_configured_files(tmp_path):
    sites = tmp_path / "sites.json"
    sites.write_text(json.dumps({"sites": [{
        "name": "portal", "start_urls": [BASE + "1"], "listing_selector": "article.offer",
        "fields": {"title": "h2"}, "next_page_selector": "a.next", "max_pages": 3}]}))
    indexer = FakeIndexer()
    with patch.multiple("agents.property.crawler.main.Config", CRAWLER_SITES_FILE=str(sites),
                        CRAWLER_DB_FILE=str(tmp_path / "seen.db"), CRAWLER_RESPECT_ROBOTS=False,
                        CRAWLER_PAGE_DELAY_SECONDS=0):
        service = CrawlerService(interval_minutes=60, indexer=indexer,
                                 fetcher_factory=lambda: FakeFetcher(_three_pages()))
        report = service.run_once()
    assert report.indexed == 4
    assert SeenStore(tmp_path / "seen.db").count() == 4


def test_service_stop_ends_start():
    service = CrawlerService(interval_minutes=60, indexer=FakeIndexer(), fetcher_factory=MagicMock())
    service.run_once = MagicMock(side_effect=lambda: service.stop())
    service.start()   # returns instead of sleeping an hour
    service.run_once.assert_called_once()


# --- what the property agent sees -------------------------------------------

def test_context_shows_the_listing_link_and_its_currency():
    context = build_context_from_listings([{"city": "Berlin", "price": 1200, "currency": "EUR",
                                            "url": "https://portal.example/offer/1", "text": "Flat"}])
    assert "Price: 1200 EUR" in context
    assert "Link: https://portal.example/offer/1" in context
    assert "Price: 900 PLN" in build_context_from_listings([{"price": 900}])


# --- real browser -----------------------------------------------------------

def test_playwright_fetcher_returns_javascript_rendered_html(tmp_path):
    page = tmp_path / "list.html"
    page.write_text(
        "<html><body><div id='root'></div><script>"
        "setTimeout(() => { document.getElementById('root').innerHTML ="
        " '<article class=\"offer\"><a href=\"/offer/1\"><h2>Rendered</h2></a></article>'; }, 200);"
        "</script></body></html>"
    )
    fetcher = PlaywrightFetcher(timeout_seconds=10, executable_path=os.getenv("CRAWLER_BROWSER_PATH"))
    try:
        fetcher.open()
    except Exception as exc:   # no Chromium here (e.g. CI without `playwright install chromium`)
        pytest.skip(f"Chromium not available: {exc}")
    try:
        fetcher.new_site()
        html = fetcher.fetch(page.as_uri(), wait_for="article.offer")
    finally:
        fetcher.close()
    [listing] = extract_listings(html, _site(), page.as_uri())
    assert listing.title == "Rendered"
