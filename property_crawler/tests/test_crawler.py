import json
import logging
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from property_crawler.classify import (deal_from_label, detect_availability, detect_deal, detect_type,
                                       split_locality)
from property_crawler.blocking import block_reason, parse_retry_after
from property_crawler.crawler import PropertyCrawler
from property_crawler.extractor import Listing, extract_listings, next_page_url, parse_number
from property_crawler.fetcher import FetchedPage, PlaywrightFetcher, RobotsChecker
from property_crawler.indexer import ListingIndexer
from property_crawler.main import CrawlerService
from property_crawler.schedule import Schedule, parse_times
from property_crawler.purge import purge
from property_crawler.settings import PACKAGE_DIR, ROOT, Settings
from property_crawler.vector_index import IncompatibleIndexError, VectorIndex, check_compatible
from property_crawler.seen_store import SeenStore
from property_crawler.sites import Site, FieldSelector, load_sites

SITES_FILE = PACKAGE_DIR / "sites.json"


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
        city="Ghent",
        property_type="Apartment",
        currency="EUR",
    )
    values.update(overrides)
    return Site(**values)


def _card(n: int, price: str = "3 500 €", city: str = "") -> str:
    return (f'<article class="offer"><a href="/offer/{n}"><h2>Flat {n}</h2></a>'
            f'<span class="price">{price}</span><span class="rooms">2 rooms</span>'
            f'<span class="area">48,5 m²</span><span class="city">{city}</span></article>')


def _page(cards: list[str], next_href: str | None = None) -> str:
    nxt = f'<a class="next" href="{next_href}">Next</a>' if next_href else ""
    return f"<html><body>{''.join(cards)}{nxt}</body></html>"


# --- extraction -------------------------------------------------------------

@pytest.mark.parametrize("text, expected", [
    ("3 500 €", 3500), ("€1,250.50", 1250.5), ("1.200 EUR", 1200), ("48,5 m²", 48.5),
    ("2 rooms", 2), ("€1,250 1250€", 1250), ("1 250 000 €", 1250000), ("EUR 4 200 / month", 4200), ("ask for price", None), ("", None), (None, None),
])
def test_parse_number_reads_european_and_english_formats(text, expected):
    assert parse_number(text) == expected


def test_extract_listings_reads_fields_and_makes_links_absolute():
    [listing] = extract_listings(_page([_card(7, city="Liège")]), _site(), "https://portal.example/rent?page=1")
    assert listing.url == "https://portal.example/offer/7"
    assert (listing.title, listing.price, listing.rooms, listing.area) == ("Flat 7", 3500, 2, 48.5)
    assert listing.city == "Liège"
    assert listing.property_type == "Apartment"
    assert "Flat 7. Apartment for rent in Liège. Price 3500 EUR. 2 rooms. 48.5 square meters." in listing.text


def test_extract_listings_falls_back_to_the_site_city_and_counts_a_repeated_ad_once():
    listings = extract_listings(_page([_card(1), _card(1)]), _site(), "https://portal.example/rent")
    assert len(listings) == 1
    assert listings[0].city == "Ghent"


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
    assert listing.extra == {"floor": "3/5"}
    assert listing.photo == "https://portal.example/p.jpg"
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


def test_shipped_sites_file_crawls_immoweb_only():
    sites = load_sites(SITES_FILE)
    assert [s.name for s in sites] == ["immoweb"]
    assert sites[0].start_urls[0].startswith("https://www.immoweb.be/en/search/")
    assert sites[0].currency == "EUR"
    assert [detect_deal(u) for u in sites[0].start_urls] == ["rent", "sale"]


def test_immoweb_config_reads_a_result_card():
    # Markup as Immoweb's result cards were last known to look; the live site could not be
    # loaded to confirm it (Cloudflare refuses automated browsers from datacenter IPs).
    [immoweb] = load_sites(SITES_FILE)
    card = (
        '<article class="card card--result">'
        '<h2 class="card__title"><a class="card__title-link" href="https://www.immoweb.be/en/classified/'
        'apartment/for-rent/ixelles/1050/12345678">Apartment for rent</a></h2>'
        '<p class="card--result__price"><span aria-hidden="true">€1,250</span><span class="sr-only">1250€</span></p>'
        '<p class="card__information--property">2 bdr. 85 m²</p>'
        '<p class="card__information--locality">1050 Ixelles</p></article>'
        '<a class="pagination__link pagination__link--next" href="?countries=BE&orderBy=newest&page=2">Next</a>'
    )
    url = immoweb.start_urls[0]
    [listing] = extract_listings(_page([card]), immoweb, url)
    assert listing.url.endswith("/12345678")
    assert (listing.price, listing.rooms, listing.area, listing.city) == (1250, 2, 85, "Ixelles")
    assert (listing.postcode, listing.deal, listing.availability) == ("1050", "rent", "available")
    assert next_page_url(_page([card]), immoweb, url).endswith("page=2")


def test_a_field_pattern_keeps_its_first_group():
    site = _site(fields={"rooms": FieldSelector(".info", pattern=r"(\d+)\s*bdr"),
                         "area": FieldSelector(".info", pattern=r"(\d+)\s*m²"),
                         "garden": FieldSelector(".info", pattern=r"garden")})
    html = _page(['<article class="offer"><a href="/o/1">x</a><p class="info">3 bdr. 120 m²</p></article>'])
    [listing] = extract_listings(html, site, "https://portal.example/")
    assert (listing.rooms, listing.area) == (3, 120)
    assert "garden" not in listing.extra


# --- classification ---------------------------------------------------------

@pytest.mark.parametrize("url, text, expected", [
    ("https://www.immoweb.be/en/search/house/for-sale?page=2", "", "sale"),
    ("https://www.immoweb.be/en/classified/apartment/for-rent/ixelles/1050/1", "", "rent"),
    ("https://www.zimmo.be/nl/te-koop/", "", "sale"),
    ("https://portal.example/a-louer/appartement", "", "rent"),
    ("https://portal.example/search", "Flat €950 / month", "rent"),
    ("https://portal.example/search", "Huis te koop, vraagprijs € 325.000", "sale"),
    ("https://portal.example/search", "Nice flat, €950", None),
    ("https://rent-and-sale.example/search", "", None),     # the host's own words don't count
    ("https://portal.example/buy/search?location=gent", "", "sale"),   # "location=" is a search parameter
    ("https://portal.example/current-offers", "", None),     # "rent" inside a word doesn't count
    ("https://portal.example/wholesale", "", None),
    ("https://portal.example/search?transaction=rent", "", "rent"),
    ("https://portal.example/search", "Flat €950/month", "rent"),
    ("https://portal.example/search", "Flat 950€/mois", "rent"),
])
def test_detect_deal_reads_the_url_then_the_words(url, text, expected):
    assert detect_deal(url, text) == expected


@pytest.mark.parametrize("label, expected", [
    ("For rent", "rent"), ("Te koop", "sale"), ("SALE", "sale"), ("À louer", "rent"), ("", None),
])
def test_deal_from_label(label, expected):
    assert deal_from_label(label) == expected


@pytest.mark.parametrize("status, text, expected", [
    ("", "Bright flat, 2 bedrooms", "available"),
    ("Under option", "", "under offer"),
    ("Sold STC", "", "under offer"),
    ("Let", "", "rented"),
    ("", "Verkocht! Prachtige woning", "sold"),
    ("", "Let us show you this lovely home", "available"),   # "let" in prose is not a status
    ("", "Recently renovated, can be rented furnished", "available"),
    ("", "Sous compromis", "under offer"),
])
def test_detect_availability(status, text, expected):
    assert detect_availability(status, text) == expected


@pytest.mark.parametrize("text, expected", [
    ("Apartment for rent", "Apartment"), ("Studio apartment", "Studio"), ("Villa with pool", "Villa"),
    ("Charming town house", "House"), ("Penthouse with terrace", "Penthouse"),
    ("Bouwgrond 800 m²", "Land"), ("Appartement te huur", "Apartment"), ("Lovely place", None),
])
def test_detect_type(text, expected):
    assert detect_type(text) == expected


@pytest.mark.parametrize("locality, expected", [
    ("1050 Ixelles", ("Ixelles", "1050")), ("9000 GENT", ("Gent", "9000")), ("Ghent (9000)", ("Ghent", "9000")),
    ("1012 AB Amsterdam", ("Amsterdam", "1012 AB")), ("75011 Paris", ("Paris", "75011")),
    ("Leuven", ("Leuven", None)), ("", (None, None)), (None, (None, None)),
])
def test_split_locality(locality, expected):
    assert split_locality(locality) == expected


def test_extracted_listing_is_classified_from_its_card():
    site = _site(property_type=None, city=None,
                 fields={**_site().fields, "availability": FieldSelector(".badge")})
    card = ('<article class="offer"><a href="/o/9"><h2>Villa with garden</h2></a><span class="price">€ 495.000</span>'
            '<span class="badge">Under offer</span><span class="city">3000 Leuven</span></article>')
    [listing] = extract_listings(_page([card]), site, "https://portal.example/for-sale?page=1")
    assert (listing.property_type, listing.deal, listing.availability) == ("Villa", "sale", "under offer")
    assert (listing.city, listing.postcode, listing.price) == ("Leuven", "3000", 495000)
    assert "Villa for sale in Leuven. Postcode 3000. Status: under offer. Price 495000 EUR." in listing.text
    meta = listing.metadata("2026-11-25T10:00:00+00:00")
    assert (meta["deal"], meta["availability"], meta["postcode"]) == ("sale", "under offer", "3000")


def test_site_deal_wins_over_detection_and_is_validated():
    [listing] = extract_listings(_page([_card(1)]), _site(deal="sale"), "https://portal.example/rent?page=1")
    assert listing.deal == "sale"


def test_a_site_with_an_unknown_deal_is_skipped(tmp_path, caplog):
    path = tmp_path / "sites.json"
    path.write_text(json.dumps({"sites": [
        {"name": "a", "url": "https://a.example/", "listing_selector": "li", "deal": "swap"},
        {"name": "b", "url": "https://b.example/", "listing_selector": "li", "deal": "Buy"}]}))
    assert [(s.name, s.deal) for s in load_sites(path)] == [("b", "sale")]
    assert "swap" in caplog.text


# --- seen store -------------------------------------------------------------

def _listing(n: int, price: float = 1000) -> Listing:
    return Listing(site="portal", url=f"https://portal.example/offer/{n}", title=f"Flat {n}", price=price)


def test_seen_store_returns_only_new_or_changed_listings(tmp_path):
    with SeenStore(tmp_path / "sub" / "seen.db") as store:
        store.touch([_listing(1), _listing(2)])
        fresh = store.unseen([_listing(1), _listing(2, price=900), _listing(3)])
        assert [l.url[-1] for l in fresh] == ["2", "3"]
        assert store.count("portal") == 2


def test_seen_store_keeps_listing_details_and_first_seen(tmp_path):
    db = tmp_path / "seen.db"
    listing = Listing(site="portal", url="https://portal.example/offer/1", title="Flat 1", city="Ghent",
                      postcode="9000", price=950, currency="EUR", rooms=2, area=60, property_type="Apartment",
                      deal="rent", photo="https://portal.example/1.jpg", extra={"floor": "3"})
    with SeenStore(db) as store:
        store.touch([listing], when="2026-11-20T10:00:00+00:00")
        listing.price, listing.availability = 900, "under offer"
        store.touch([listing], when="2026-11-24T10:00:00+00:00")
        store.seen_again([listing], when="2026-11-25T10:00:00+00:00")
    conn = sqlite3.connect(db)
    conn.row_factory = sqlite3.Row
    row = dict(conn.execute("SELECT * FROM listings").fetchone())
    assert (row["first_seen"], row["updated_at"], row["last_seen"]) == (
        "2026-11-20T10:00:00+00:00", "2026-11-24T10:00:00+00:00", "2026-11-25T10:00:00+00:00")
    assert (row["price"], row["availability"], row["deal"], row["city"], row["postcode"]) == (
        900, "under offer", "rent", "Ghent", "9000")
    assert json.loads(row["extra"]) == {"floor": "3"}


def test_seen_store_adds_detail_columns_to_an_old_database(tmp_path):
    db = tmp_path / "seen.db"
    conn = sqlite3.connect(db)
    conn.execute("CREATE TABLE listings (id TEXT PRIMARY KEY, site TEXT NOT NULL, url TEXT NOT NULL, "
                 "content_hash TEXT NOT NULL, first_seen TEXT NOT NULL, last_seen TEXT NOT NULL)")
    conn.execute("INSERT INTO listings VALUES ('crawl-old', 'portal', 'u', 'h', '2026-01-01', '2026-01-01')")
    conn.commit()
    conn.close()
    with SeenStore(db) as store:
        store.touch([_listing(1)])
        assert store.count() == 2
    columns = {r[1] for r in sqlite3.connect(db).execute("PRAGMA table_info(listings)")}
    assert {"deal", "availability", "city", "price", "photo", "updated_at"} <= columns


def test_seen_store_survives_a_restart(tmp_path):
    with SeenStore(tmp_path / "seen.db") as store:
        store.touch([_listing(1)])
    with SeenStore(tmp_path / "seen.db") as store:
        assert store.unseen([_listing(1)]) == []


# --- crawler ----------------------------------------------------------------

class FakeFetcher:
    """pages: url → HTML, or a FetchedPage to answer with a status."""

    def __init__(self, pages: dict, fail: set[str] = frozenset()):
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
        page = self.pages[url]
        return page if isinstance(page, FetchedPage) else FetchedPage(page, 200)

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
    kw.setdefault("sleep", lambda s: None)
    return PropertyCrawler(sites, fetcher, store, indexer, page_delay_seconds=0, **kw)


# --- wait, stop and see -----------------------------------------------------

CLOUDFLARE = FetchedPage("<html><head><title>Just a moment...</title></head><body></body></html>", 403)


def test_crawler_waits_a_random_while_between_page_loads(tmp_path):
    waits = []
    crawler = PropertyCrawler([_site()], FakeFetcher(_three_pages()), SeenStore(tmp_path / "seen.db"),
                              FakeIndexer(), page_delay_seconds=8, sleep=waits.append,
                              jitter=lambda d: d + 1)
    crawler.run_once()
    assert waits == [9, 9]   # before pages 2 and 3, not before the first


def test_default_wait_is_between_one_and_two_delays():
    from property_crawler.crawler import _jitter
    assert all(8 <= _jitter(8) <= 16 for _ in range(200))


def test_a_block_stops_the_site_and_backs_off(tmp_path, caplog):
    pages = _three_pages()
    pages[BASE + "2"] = CLOUDFLARE
    indexer = FakeIndexer()
    with caplog.at_level(logging.WARNING):
        report = _crawler(tmp_path, [_site()], FakeFetcher(pages), indexer).run_once()
    assert report.sites[0].blocked == "HTTP 403"
    assert report.sites[0].pages == 2                      # page 3 never requested
    assert [l.title for l in indexer.indexed] == ["Flat 1", "Flat 2"]   # page 1 still indexed
    assert "refused the crawler" in caplog.text
    with SeenStore(tmp_path / "seen.db") as store:
        until = store.blocked_until("portal")
    assert until is not None and until > datetime.now(timezone.utc) + timedelta(minutes=55)


def test_a_backing_off_site_is_skipped_without_a_browser(tmp_path):
    pages = {BASE + "1": CLOUDFLARE}
    _crawler(tmp_path, [_site()], FakeFetcher(pages), FakeIndexer()).run_once()
    fetcher = FakeFetcher(_three_pages())
    report = _crawler(tmp_path, [_site()], fetcher, FakeIndexer()).run_once()
    assert fetcher.fetched == [] and fetcher.opened == 0
    assert "backing off" in report.sites[0].skipped


def test_a_blocked_site_does_not_stop_the_next_one(tmp_path):
    other = _site(name="other", start_urls=["https://other.example/"], next_page_selector=None)
    pages = {BASE + "1": CLOUDFLARE, "https://other.example/": _page([_card(5)])}
    report = _crawler(tmp_path, [_site(), other], FakeFetcher(pages), FakeIndexer()).run_once()
    assert report.sites[0].blocked and report.sites[1].indexed == 1


def test_back_off_doubles_per_block_up_to_the_cap_and_honours_retry_after(tmp_path):
    now = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)
    base, cap = timedelta(hours=1), timedelta(hours=24)
    with SeenStore(tmp_path / "seen.db") as store:
        waits = [store.record_block("s", "HTTP 429", base, cap, now=now) - now for _ in range(7)]
        assert [w.total_seconds() / 3600 for w in waits] == [1, 2, 4, 8, 16, 24, 24]
        assert store.record_block("t", "HTTP 429", base, cap, timedelta(hours=3), now=now) - now == timedelta(hours=3)
        store.clear_block("s")
        assert store.blocked_until("s") is None
        assert store.record_block("s", "HTTP 429", base, cap, now=now) - now == base   # strikes reset


def test_a_normal_pass_clears_an_expired_back_off(tmp_path):
    with SeenStore(tmp_path / "seen.db") as store:
        past = datetime.now(timezone.utc) - timedelta(days=2)
        store.record_block("portal", "HTTP 429", timedelta(hours=1), timedelta(hours=1), now=past)
    _crawler(tmp_path, [_site()], FakeFetcher(_three_pages()), FakeIndexer()).run_once()
    with SeenStore(tmp_path / "seen.db") as store:
        assert store.record_block("portal", "x", timedelta(hours=1), timedelta(hours=24)) \
            - datetime.now(timezone.utc) < timedelta(hours=1, minutes=1)   # strike 1 again


def test_retry_after_seconds_is_used(tmp_path):
    pages = {BASE + "1": FetchedPage("slow down", 429, retry_after="7200")}
    _crawler(tmp_path, [_site()], FakeFetcher(pages), FakeIndexer()).run_once()
    with SeenStore(tmp_path / "seen.db") as store:
        assert store.blocked_until("portal") > datetime.now(timezone.utc) + timedelta(minutes=115)


@pytest.mark.parametrize("page, listings, expected", [
    (FetchedPage("x", 429), 0, "HTTP 429"),
    (FetchedPage("x", 503), 3, "HTTP 503"),
    (FetchedPage("<title>Attention Required! | Cloudflare</title>", 200), 0, "challenge page"),
    (FetchedPage('<div id="cf-turnstile"></div>', 200), 0, "challenge page (cf-turnstile)"),
    (FetchedPage('<script src="https://ct.captcha-delivery.com/c.js"></script>', 200), 0, "captcha-delivery.com"),
    (FetchedPage('<title>Rentals</title><div class="g-recaptcha"></div>', 200), 4, None),  # real content
    (FetchedPage("<title>Rentals</title>", 200), 0, None),   # just no matches: a selector problem
    (FetchedPage("x", 404), 0, None),
])
def test_block_reason(page, listings, expected):
    reason = block_reason(page, listings)
    assert (reason is None) if expected is None else (expected in reason)


def test_parse_retry_after_reads_seconds_and_dates():
    now = datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc)
    assert parse_retry_after("120") == timedelta(seconds=120)
    assert parse_retry_after("Wed, 25 Nov 2026 13:00:00 GMT", now=now) == timedelta(hours=1)
    assert parse_retry_after(None) is None and parse_retry_after("soon") is None


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


def test_unchanged_listings_still_on_the_site_move_last_seen_only(tmp_path):
    pages = {BASE + "1": _page([_card(1)])}
    _crawler(tmp_path, [_site()], FakeFetcher(pages), FakeIndexer()).run_once()
    conn = sqlite3.connect(tmp_path / "seen.db")
    conn.execute("UPDATE listings SET last_seen = '2026-01-01T00:00:00+00:00'")
    conn.commit()
    indexer = FakeIndexer()
    _crawler(tmp_path, [_site()], FakeFetcher(pages), indexer).run_once()
    assert indexer.indexed == []
    first_seen, last_seen = conn.execute("SELECT first_seen, last_seen FROM listings").fetchone()
    assert last_seen > "2026-01-01T00:00:00+00:00" and first_seen <= last_seen


def test_a_price_change_is_reindexed(tmp_path):
    pages = {BASE + "1": _page([_card(1)])}
    _crawler(tmp_path, [_site()], FakeFetcher(pages), FakeIndexer()).run_once()
    pages[BASE + "1"] = _page([_card(1, price="3 200 €")])
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
    with patch("property_crawler.fetcher.requests.get", return_value=response) as get:
        robots = RobotsChecker()
        assert robots.allowed("https://a.example/rent")
        assert not robots.allowed("https://a.example/private/1")
    get.assert_called_once()


def test_unreachable_robots_txt_allows_crawling():
    import requests
    with patch("property_crawler.fetcher.requests.get", side_effect=requests.ConnectionError("down")):
        assert RobotsChecker().allowed("https://a.example/rent")


# --- indexer ----------------------------------------------------------------

def test_indexer_upserts_in_batches_with_listing_metadata():
    vector_index = MagicMock()
    listings = [_listing(n) for n in range(5)]
    assert ListingIndexer(vector_index, batch_size=2).index(listings) == 5
    batches = [c.args[0] for c in vector_index.upsert.call_args_list]
    assert [len(b) for b in batches] == [2, 2, 1]
    doc = batches[0][listings[0].id]
    assert doc["text"] == listings[0].text and doc["url"] == listings[0].url


def _vector_index() -> VectorIndex:
    index = VectorIndex("key", "listings", "some-model")
    index._model, index._index = MagicMock(), MagicMock()
    return index


def test_vector_index_embeds_the_text_and_upserts_metadata():
    import numpy as np
    index = _vector_index()
    index._model.encode.return_value = np.array([[0.5, 0.25], [1.0, 0.0]])
    index.upsert({"crawl-a": {"text": "flat a", "city": "Ghent"}, "crawl-b": {"text": "flat b"}})
    index._model.encode.assert_called_once_with(["flat a", "flat b"])
    vectors = index._index.upsert.call_args.kwargs["vectors"]
    assert vectors[0] == {"id": "crawl-a", "values": [0.5, 0.25], "metadata": {"text": "flat a", "city": "Ghent"}}
    assert isinstance(vectors[1]["values"], list)


def test_vector_index_deletes_in_chunks_of_1000():
    index = _vector_index()
    index.delete([str(i) for i in range(2500)])
    assert [len(c.kwargs["ids"]) for c in index._index.delete.call_args_list] == [1000, 1000, 500]


def _description(dimension=384, fields=None):
    schema = None if fields is None else MagicMock(fields=fields)
    return MagicMock(dimension=dimension, schema=schema)


def test_compatible_index_passes_the_check():
    check_compatible(_description(), "listings", 384)


def test_index_with_a_document_schema_is_rejected():
    with pytest.raises(IncompatibleIndexError, match="document schema"):
        check_compatible(_description(1024, {"title": 1, "chunk_text": 2}), "listings", 384)


def test_index_with_another_dimension_is_rejected():
    with pytest.raises(IncompatibleIndexError, match="dimension 1024"):
        check_compatible(_description(1024), "listings", 384)


def test_vector_index_checks_an_existing_index_before_using_it():
    index = VectorIndex("key", "listings", "some-model")
    index._model = MagicMock()
    index._model.get_sentence_embedding_dimension.return_value = 384
    client = MagicMock()
    client.list_indexes.return_value.names.return_value = ["listings"]
    client.describe_index.return_value = _description(1024)
    with patch("pinecone.Pinecone", return_value=client):
        with pytest.raises(IncompatibleIndexError):
            index.index
    client.create_index.assert_not_called()


def test_vector_index_needs_pinecone_settings():
    with pytest.raises(ValueError, match="PINECONE"):
        VectorIndex("", "listings", "model")


# --- purge ------------------------------------------------------------------

def test_purge_deletes_only_vectors_the_crawler_did_not_write():
    store = MagicMock()
    store.list_ids.return_value = iter([["crawl-abc", "123", "456"], ["crawl-def", "789"]])
    assert purge(store, delete=True) == 3
    store.delete.assert_called_once_with(["123", "456", "789"])


def test_purge_dry_run_deletes_nothing():
    store = MagicMock()
    store.list_ids.return_value = iter([["1", "crawl-x"]])
    assert purge(store, delete=False) == 1
    store.delete.assert_not_called()


# --- settings ---------------------------------------------------------------

def test_settings_defaults(monkeypatch, tmp_path):
    for name in list(os.environ):
        if name.startswith("CRAWLER_"):
            monkeypatch.delenv(name)
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))
    s = Settings.from_env()
    assert s.sites_file == SITES_FILE
    assert s.db_file == tmp_path / "property-crawler" / "seen.db"
    assert (s.interval_minutes, s.page_delay_seconds, s.back_off_minutes, s.max_back_off_hours) == (60, 8, 60, 24)
    assert s.headless and s.respect_robots
    assert (s.run_at, s.timezone) == ("04:30,13:00", "Europe/Brussels")


def test_an_empty_run_at_means_every_interval(monkeypatch):
    monkeypatch.setenv("CRAWLER_RUN_AT", "")
    assert Settings.from_env().run_at == ""


def test_settings_resolve_relative_paths_from_the_repository_root(monkeypatch):
    monkeypatch.setenv("CRAWLER_SITES_FILE", "my/sites.json")
    monkeypatch.setenv("CRAWLER_HEADLESS", "false")
    monkeypatch.setenv("CRAWLER_INTERVAL_MINUTES", "30")
    s = Settings.from_env()
    assert s.sites_file == ROOT / "my" / "sites.json"
    assert not s.headless and s.interval_minutes == 30


def test_the_package_imports_nothing_from_uk_agents():
    # It is meant to move to its own repository: only the standard library, third-party
    # packages and its own modules.
    import ast
    foreign = {"core", "agents", "orchestrator", "clients", "logging_config"}
    for path in PACKAGE_DIR.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
            for name in names:
                assert name.split(".")[0] not in foreign, f"{path.name} imports {name}"


# --- service ----------------------------------------------------------------

def _settings(tmp_path, sites) -> Settings:
    return Settings(sites_file=sites, db_file=tmp_path / "seen.db", respect_robots=False,
                    page_delay_seconds=0)


def test_service_runs_a_pass_with_the_configured_files(tmp_path):
    sites = tmp_path / "sites.json"
    sites.write_text(json.dumps({"sites": [{
        "name": "portal", "start_urls": [BASE + "1"], "listing_selector": "article.offer",
        "fields": {"title": "h2"}, "next_page_selector": "a.next", "max_pages": 3}]}))
    service = CrawlerService(_settings(tmp_path, sites), indexer=FakeIndexer(),
                             fetcher_factory=lambda: FakeFetcher(_three_pages()))
    assert service.run_once().indexed == 4
    assert SeenStore(tmp_path / "seen.db").count() == 4


def test_service_stop_ends_start(tmp_path):
    service = CrawlerService(_settings(tmp_path, tmp_path / "none.json"), indexer=FakeIndexer(),
                             fetcher_factory=MagicMock())
    service.run_once = MagicMock(side_effect=lambda: service.stop())
    service.start()   # returns instead of sleeping an hour
    service.run_once.assert_called_once()


# --- schedule ---------------------------------------------------------------

BRUSSELS = Schedule(tuple(parse_times("13:00, 04:30")), "Europe/Brussels")


def _utc(text: str) -> datetime:
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


def test_parse_times_sorts_and_rejects_bad_times():
    assert [t.strftime("%H:%M") for t in parse_times("13:00, 4:30,13:00")] == ["04:30", "13:00"]
    assert parse_times("") == []
    with pytest.raises(ValueError):
        parse_times("25:00")


def test_unknown_time_zone_is_a_configuration_error():
    with pytest.raises(ValueError, match="CRAWLER_TIMEZONE"):
        Schedule((), "Mars/Olympus")


@pytest.mark.parametrize("now, last_pass, expected", [
    # Brussels is UTC+1 in winter: 04:30 local = 03:30 UTC, 13:00 local = 12:00 UTC
    ("2026-11-26T05:00:00", "2026-11-26T03:30:00", "2026-11-26T12:00:00"),   # morning pass done → 13:00
    ("2026-11-26T13:00:00", "2026-11-26T12:00:00", "2026-11-27T03:30:00"),   # both done → tomorrow 04:30
    ("2026-11-26T02:00:00", "2026-11-25T12:00:00", "2026-11-26T03:30:00"),   # night → 04:30
    ("2026-11-26T10:00:00", "2026-11-25T12:00:00", "2026-11-26T10:00:00"),   # 04:30 missed (restart) → now
    ("2026-06-26T05:00:00", "2026-06-26T02:30:00", "2026-06-26T11:00:00"),   # summer time: UTC+2
])
def test_fixed_times_run_after_4_30_and_at_13_00_brussels_time(now, last_pass, expected):
    assert BRUSSELS.next_run(_utc(now), _utc(last_pass)) == _utc(expected)


def test_a_first_start_runs_a_pass_now():
    assert BRUSSELS.next_run(_utc("2026-11-26T10:00:00"), None) == _utc("2026-11-26T10:00:00")


def test_interval_schedule():
    every_hour = Schedule((), "UTC", 60)
    assert every_hour.next_run(_utc("2026-11-26T10:00:00"), _utc("2026-11-26T09:30:00")) == _utc("2026-11-26T10:30:00")
    assert every_hour.describe() == "a pass every 60 minutes"
    assert BRUSSELS.describe() == "passes at 04:30, 13:00 (Europe/Brussels)"


def test_service_waits_for_the_next_scheduled_time(tmp_path):
    clock = {"now": _utc("2026-11-26T05:00:00")}
    with SeenStore(tmp_path / "seen.db") as store:
        store.record_pass(_utc("2026-11-26T03:30:00"))      # this morning's pass is done
    slept = []

    def sleep(seconds):
        slept.append(seconds)
        clock["now"] += timedelta(minutes=30)

    service = CrawlerService(_settings(tmp_path, tmp_path / "none.json"), indexer=FakeIndexer(),
                             fetcher_factory=MagicMock(), clock=lambda: clock["now"], sleep=sleep)
    passes = []
    service.run_once = MagicMock(side_effect=lambda: (passes.append(clock["now"]), service.stop()))
    service.start()
    assert passes == [_utc("2026-11-26T12:00:00")]           # 13:00 Brussels
    assert len(slept) == 14


def test_run_once_records_the_pass(tmp_path):
    service = CrawlerService(_settings(tmp_path, tmp_path / "none.json"), indexer=FakeIndexer(),
                             fetcher_factory=MagicMock(), clock=lambda: _utc("2026-11-26T12:00:00"))
    service.run_once()
    with SeenStore(tmp_path / "seen.db") as store:
        assert store.last_pass() == _utc("2026-11-26T12:00:00")


# --- duplicates -------------------------------------------------------------

def test_tracking_parameters_do_not_make_a_new_listing():
    cards = ['<article class="offer"><a href="/offer/1?utm_source=x&fbclid=y&id=7"><h2>A</h2></a></article>',
             '<article class="offer"><a href="/offer/1?id=7"><h2>A</h2></a></article>']
    [listing] = extract_listings(_page(cards), _site(fields={"title": FieldSelector("h2")}), "https://portal.example/")
    assert listing.url == "https://portal.example/offer/1?id=7"


def test_fingerprint_needs_price_place_and_size():
    flat = Listing(site="a", url="u", price=950, postcode="9000", area=60.4, rooms=2, deal="rent",
                   property_type="Apartment")
    assert flat.fingerprint == "rent|9000|950|60|2|apartment"
    assert Listing(site="b", url="v", price=950.0, city="Gent", area=60, deal="rent").fingerprint == "rent|gent|950|60|?|?"
    assert Listing(site="a", url="u", price=950, postcode="9000").fingerprint is None      # no size
    assert Listing(site="a", url="u", area=60, postcode="9000").fingerprint is None        # no price


def _flat(site: str, n: int, price: float = 950, area: float = 60) -> Listing:
    return Listing(site=site, url=f"https://{site}.example/{n}", title=f"Flat {n}", price=price, postcode="9000",
                   area=area, rooms=2, deal="rent", property_type="Apartment")


def test_the_same_property_on_another_site_is_stored_once(tmp_path):
    indexer = FakeIndexer()
    crawler = _crawler(tmp_path, [], FakeFetcher({}), indexer)
    store = crawler._store
    crawler._index([_flat("agency", 1)])
    fresh, indexed = crawler._index([_flat("portal", 7), _flat("portal", 8, price=1200)])
    assert (len(fresh), indexed) == (2, 1)                    # flat 7 is the agency's flat 1
    assert [l.url for l in indexer.indexed] == ["https://agency.example/1", "https://portal.example/8"]
    conn = sqlite3.connect(tmp_path / "seen.db")
    rows = dict(conn.execute("SELECT url, duplicate_of FROM listings").fetchall())
    assert rows["https://portal.example/7"] == _flat("agency", 1).id
    assert rows["https://agency.example/1"] is None and rows["https://portal.example/8"] is None
    # Seen again unchanged: still a duplicate, still not indexed
    assert crawler._index([_flat("portal", 7)]) == ([], 0)
    assert store.count() == 3


def test_identical_flats_on_one_site_are_kept_apart(tmp_path):
    # A new-build project lists several identical flats: same site, different ads → all kept
    indexer = FakeIndexer()
    crawler = _crawler(tmp_path, [], FakeFetcher({}), indexer)
    crawler._index([_flat("agency", 1), _flat("agency", 2)])
    assert len(indexer.indexed) == 2


def test_a_seen_db_without_dedup_columns_gets_them(tmp_path):
    db = tmp_path / "seen.db"
    with SeenStore(db) as store:
        store.touch([_flat("agency", 1)])
    conn = sqlite3.connect(db)
    conn.execute("ALTER TABLE listings DROP COLUMN duplicate_of")
    conn.commit()
    conn.close()
    with SeenStore(db) as store:
        assert store.original_of(_flat("portal", 9)) == _flat("agency", 1).id


def test_a_site_without_a_currency_shows_the_bare_price():
    [listing] = extract_listings(_page([_card(1)]), _site(currency=None), "https://portal.example/")
    assert "Price 3500." in listing.text and "currency" not in listing.metadata("now")


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
        fetched = fetcher.fetch(page.as_uri(), wait_for="article.offer")
    finally:
        fetcher.close()
    [listing] = extract_listings(fetched.html, _site(), page.as_uri())
    assert listing.title == "Rendered"


def test_a_switched_off_site_needs_no_selectors(tmp_path, caplog):
    path = tmp_path / "sites.json"
    path.write_text(json.dumps({"sites": [{"name": "later", "enabled": False, "start_urls": ["https://x.example/"]}]}))
    assert load_sites(path) == []
    assert "skipped" not in caplog.text


def test_shipped_sites_file_lists_every_requested_site_switched_off_until_its_selectors_are_set():
    raw = json.loads(SITES_FILE.read_text(encoding="utf-8"))["sites"]
    waiting = [s for s in raw if not s.get("enabled", True) and s.get("todo")]
    assert len(waiting) == 53
    assert len({s["name"] for s in raw}) == len(raw)
