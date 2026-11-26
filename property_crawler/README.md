# Property listing crawler

`property_crawler` keeps a Pinecone index of property listings fresh. Twice a day (04:30 and 13:00, Brussels time) it visits the websites in `sites.json` one at a time. It renders each listing page in headless Chromium (Playwright) so JavaScript-built pages work, reads the listing cards with CSS selectors, and upserts every **new or changed** listing into the index. uk-agents' property agent searches that index.

It is a self-contained package that imports nothing from uk-agents (a test enforces this), so it can move to its own repository unchanged. See [Moving to its own repository](#moving-to-its-own-repository). It shares only an interface with uk-agents: the Pinecone index, the embedding model and the metadata keys.

It runs on the Ubuntu server as one of the services of `deployment/deploy.sh` (see [Running it on the server](#running-it-on-the-server)). By hand:

```bash
uv run playwright install chromium                 # the browser (deploy.sh does this on every build)
python -m property_crawler.main             # passes at CRAWLER_RUN_AT (04:30 and 13:00 by default)
python -m property_crawler.main --once      # one pass now, then exit
```

## One pass

```
for each enabled site in sites.json (in order):
    still backing off from a block?    → skip it this pass
    fresh browser context (no cookies carried between sites)
    for each start URL:
        page = start URL
        while page and pages < max_pages:
            robots.txt allows it?          (CRAWLER_RESPECT_ROBOTS)
            wait a random 8–16 s           (not before the pass's first page)
            render in Chromium, wait for `wait_for`, scroll `scrolls` times
            refused (403/429/503, bot check)?  → stop this site, back off
            cards → listings               (extractor.py, CSS selectors)
            new or changed? (seen.db)      → embed + upsert to Pinecone, then mark seen
            nothing new on this page?      → stop paging (stop_when_seen)
            page = the `next_page_selector` link
```

### Wait, stop and see

The crawler is paced so that it doesn't get blocked, and backs away when a site pushes back:

- **Wait.** Before every page load except the first of a pass, it pauses for a random time between `CRAWLER_PAGE_DELAY_SECONDS` and twice that: 8–16 s by default. With `max_pages: 3`, a site gets at most 3 page loads an hour.
- **Stop.** A refusal ends that site's crawl for this pass at once. A refusal is HTTP 401/403/429/503, or a page with no listings that is a bot check (Cloudflare "Just a moment…", "Attention Required", Turnstile, DataDome, reCAPTCHA/hCaptcha). Listings from pages before the refusal are kept.
- **See.** The site is skipped for `CRAWLER_BACK_OFF_MINUTES` (60), then 2 h, 4 h, 8 h and so on, up to `CRAWLER_MAX_BACK_OFF_HOURS` (24), each time it refuses again. When the site sends `Retry-After` and asks for longer, the crawler waits that long. The first normal answer resets this. The back-off is kept in `CRAWLER_DB_FILE`, so restarts and deploys don't reset it.

The log says what happened: `immoweb refused the crawler (HTTP 403) — stopped; not trying again before 2026-11-26 10:41 UTC`, then `immoweb: skipped (backing off …)` on each pass until then.

The crawler never tries to get *past* a block. It doesn't solve CAPTCHAs, fake a fingerprint or rotate IPs. A site that keeps refusing stays at one attempt a day.

- **Only new data is embedded.** `CRAWLER_DB_FILE` (SQLite) keeps a content hash for each listing it has indexed. A listing is embedded again only when its text changes, for example after a price drop.
- **No duplicates in Pinecone.** The vector ID comes from the listing's URL (`crawl-<sha1>`), so a re-upsert overwrites the old vector.
- **A broken site doesn't stop the pass.** A failure is logged with its traceback and the crawler moves on to the next site. A failed Pinecone upsert leaves those listings unmarked, so they are retried on the next pass.
- **`sites.json` is re-read every pass.** Adding or fixing a site needs no restart.
- **Each pass starts Chromium and closes it at the end**, so no browser sits in memory between passes. The embedding model is loaded once and kept.

## sites.json

```json
{
  "sites": [
    {
      "name": "my-portal",
      "start_urls": ["https://portal.example/rent/apartments?sort=newest"],
      "listing_selector": "article.listing",
      "fields": {
        "title": "h2",
        "price": ".price",
        "city": ".location",
        "rooms": ".rooms",
        "area": ".area",
        "floor": ".floor",
        "photo": {"selector": "img", "attr": "src"}
      },
      "next_page_selector": "a[rel=next]",
      "max_pages": 3,
      "currency": "EUR",
      "property_type": "Apartment"
    }
  ]
}
```

| Key | Default | Meaning |
|---|---|---|
| `name` | required | Shown in logs, stored as the `site` metadata |
| `start_urls` (or `url`) | required | Listing pages to start from. Sort them newest first, so `stop_when_seen` can stop early |
| `listing_selector` | required | CSS selector matching one listing card |
| `fields` | `{}` | Field → CSS selector *inside the card* (its text), or `{"selector", "attr", "pattern"}`: an attribute instead of the text, and/or a regex whose first group is kept (`"(\\d+)\\s*bdr"` reads `2` from `2 bdr. 85 m²`). `price`, `rooms` and `area` are parsed into numbers (`3 500 €` → 3500, `48,5 m²` → 48.5). `title`, `city`, `type`, `description`, `url`, `deal` (a "For rent" / "Te koop" label), `availability` (a status badge such as "Under offer") and `photo` (use `"attr": "src"`) are known fields; any other field (e.g. `floor`) is added to the text and metadata as is |
| `link_selector` | `a[href]` | Link to the listing inside the card. It is the listing's identity. A card without a link gets an ID from its text |
| `next_page_selector` | none | Link to the next results page |
| `max_pages` | `1` | Pages followed per start URL |
| `wait_for` | `listing_selector` | Selector to wait for before reading the page |
| `scrolls` | `0` | Scroll to the bottom this many times first (infinite-scroll sites) |
| `stop_when_seen` | `true` | Stop paging when a page has no new or changed listing |
| `city`, `property_type` | none | Used when the card doesn't name one |
| `currency` | none | Shown with the price, e.g. `EUR` |
| `deal` | detected | `rent` or `sale` when every listing on the site is one (`buy`/`sell` are accepted too). Left out, it is detected per listing (see below) |
| `enabled` | `true` | `false` keeps the entry without crawling it; a switched-off entry may lack its selectors |
| `todo` | none | A note on what is missing before the site can be switched on (ignored by the crawler) |

The shipped `sites.json` crawls Immoweb (see below) and holds one disabled example, plus 53 more Belgian portals, agencies, classifieds and student-housing sites, switched off with a `todo`: each needs its newest-first search page in `start_urls`, its card selectors, and `"enabled": true`. To find selectors, open a listing page in Chrome, right-click a listing → Inspect, and pick the class shared by every card. A site whose selectors match nothing logs `no listings matched … check the selectors`.

## What the crawler works out for each listing

Besides the fields read with selectors, every listing is classified (`classify.py`, English, Dutch and French keywords). A field or `sites.json` key the site provides always wins; detection only fills what's missing.

| Attribute | Values | How |
|---|---|---|
| `deal` | `rent`, `sale` or none | The `deal` field, then the site's `deal`, then the page URL (`/for-rent/`, `/te-koop/`, `/a-vendre/` …), then the card's words (`€950 / month`, `vraagprijs`, `for sale` …) |
| `availability` | `available`, `under offer`, `sold`, `rented` | The `availability` field (a badge), else status words anywhere on the card (`under option`, `sold STC`, `verkocht`, `sous compromis` …). A card with no such mark is `available` |
| `property_type` | `Apartment`, `House`, `Studio`, `Villa`, `Penthouse`, `Duplex`, `Loft`, `Room`, `Land`, `Garage`, `Office`, `Commercial`, … | The `type` field, then the site's `property_type`, then the title, then the card text |
| `city` / `postcode` | `Ixelles` / `1050` | The locality is split: `1050 Ixelles`, `Ghent (9000)`, `1012 AB Amsterdam` |

So one site can be crawled for rent *and* sale: give it one start URL of each (Immoweb does). A change of status or price changes the listing's text, so it is re-embedded and its stored details are updated.

## The listings table

`CRAWLER_DB_FILE` (`seen.db`) keeps every listing's details, so other apps can list and filter listings without Pinecone. **my-stats' Properties page reads it read-only**, so these columns are a contract: add columns freely, but don't rename or drop them.

| Column | Meaning |
|---|---|
| `id`, `site`, `url` | Vector id (`crawl-…`), site name, listing link |
| `title`, `description`, `photo` | As read from the card (`photo` is an absolute image URL) |
| `city`, `postcode` | Split from the locality |
| `price`, `price_text`, `currency` | Parsed number, the text as shown, e.g. `EUR` |
| `rooms`, `area` | Bedrooms / rooms, m² |
| `property_type`, `deal`, `availability` | See above |
| `extra` | JSON object of the site's other fields |
| `first_seen` | When the crawler first found it (ISO 8601, UTC) — "newest" means this |
| `updated_at` | When its details last changed (e.g. a price drop) |
| `last_seen` | The last pass whose pages showed it. The crawler reads only the newest pages, so an old `last_seen` means "dropped out of the newest results", not necessarily "gone" |
| `content_hash` | Changes when anything indexed changes |
| `fingerprint` | The property, whichever site lists it (see [Duplicates](#duplicates)) |
| `duplicate_of` | The `id` of the same property found first on another site, or empty. Apps listing properties should leave these rows out |

A `seen.db` from before these columns existed gets them added on the next start; its old rows keep empty details until the listing is seen again.

## When it runs

At `CRAWLER_RUN_AT`: by default 04:30, after the night's new listings are up, and 13:00. The start of each pass is kept in `CRAWLER_DB_FILE`, so the crawler knows what it missed: when it starts (first run, a reboot, a deploy) and the last scheduled pass didn't happen, it runs one straight away, then goes back to the schedule. A deploy in between two passes doesn't trigger an extra one. `--once` runs a pass now, whatever the time.

## Duplicates

The same property is often listed by its agency, by one or more portals and by aggregators. Three checks keep it out of the index twice:

- **Same ad, same URL** — the vector id comes from the listing's URL, so seeing it again updates it.
- **Same ad, different URL** — tracking parameters (`utm_*`, `fbclid`, `gclid`, `ref`, `searchId` …) are removed from listing links before they become the id.
- **Same property, another site** — each listing gets a fingerprint: rent/sale, postcode (or city), price, m², bedrooms and type. When another site already has a listing with the same fingerprint, the new one is stored in `seen.db` with `duplicate_of` pointing at the first, and is **not** embedded. A listing needs a price, a place and m² or bedrooms to get a fingerprint; with less, it is always kept. Identical listings on the *same* site (flats in a new-build project) are kept apart.

## What lands in Pinecone

Each listing is embedded as text, for example `Sunny 2-bed. Apartment in Ghent. Price 1150 EUR. 2 rooms. 52 square meters. …`. Queries must be embedded with the same model (`SENTENCE_TRANSFORMER_MODEL`), and the index is searched in its default namespace. The text comes with metadata, which uk-agents' `core/src/model/context_builder.py` shows the property agent: `text`, `city`, `price`, `currency`, `rooms`, `squareMeters`, `type` and `url` (shown to the agent as `Link:`), plus `source: "crawler"`, `site`, `title`, `deal` (shown as `For:`), `availability` (shown as `Status:`), `postcode`, `photo`, `crawled_at` (shown as `Last seen:`) and any extra fields.

## Immoweb

`sites.json` crawls two Immoweb searches, newest first: houses and apartments `for-rent` and `for-sale` anywhere in Belgium. That's up to 3 result pages (about 90 listings) per search per pass; the URL tells the crawler which listings are for rent and which for sale. To narrow it, change the search on immoweb.be and paste the resulting URL into `start_urls`. Keep `orderBy=newest`, so the crawler stops at the first page with nothing new.

**Unverified.** When this was set up, Immoweb's Cloudflare refused every automated request from a datacenter IP, robots.txt included (HTTP 403, "Just a moment…"). So the selectors (`article.card--result`, `.card--result__price`, `.card__information--property`, `.card__information--locality`, `a.pagination__link--next`) follow Immoweb's last known markup and are not checked against the live site. On the server, look at the first pass in `logs/crawler.log`:

| Log says | Meaning |
|---|---|
| `immoweb: 3 page(s), 90 listing(s), 90 new/changed` | It works |
| `no listings matched 'article.card--result'` | The page loaded but the markup changed: update the selectors |
| `immoweb refused the crawler (HTTP 403)` | Cloudflare blocks the server. The crawler backs off and tries again later, at most once a day |

Check Immoweb's terms of use before relying on this. Listing portals often forbid automated collection, and the robots.txt could not be read from here. If Immoweb keeps refusing, saved-search e-mail alerts are an alternative: Immoweb sends new listings by e-mail, which the mail agent can read.

## Clearing out old data

The property agent searches the whole index, so vectors written before the crawler (the old Polish CSV data) keep showing up until they are deleted. Every vector the crawler writes has an id starting with `crawl-`; this deletes all the others:

```bash
python -m property_crawler.purge          # dry run: how many would go
python -m property_crawler.purge --yes    # delete them
```

It deletes from `PINECONE_INDEX_NAME`, in the default namespace, and can't be undone.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `CRAWLER_SITES_FILE` | `property_crawler/sites.json` | Site list |
| `CRAWLER_RUN_AT` | `04:30,13:00` | When passes start: times of day (`HH:MM`, comma-separated) in `CRAWLER_TIMEZONE`. Empty → every `CRAWLER_INTERVAL_MINUTES` instead |
| `CRAWLER_TIMEZONE` | `Europe/Brussels` | Time zone of `CRAWLER_RUN_AT` (summer time is handled) |
| `CRAWLER_INTERVAL_MINUTES` | `60` | Only with an empty `CRAWLER_RUN_AT`: minutes between the start of one pass and the next |
| `CRAWLER_DB_FILE` | per-user data dir `/property-crawler/seen.db` (Linux: `~/.local/share/property-crawler/seen.db`) | Which listings are already indexed (delete it to re-index everything) |
| `CRAWLER_HEADLESS` | `true` | `false` shows the browser window (debugging selectors) |
| `CRAWLER_TIMEOUT_SECONDS` | `30` | Page load / `wait_for` timeout |
| `CRAWLER_PAGE_DELAY_SECONDS` | `8` | Random wait between page loads: this many seconds to twice as many |
| `CRAWLER_BACK_OFF_MINUTES` | `60` | How long a site that refused the crawler is skipped; doubled each time it refuses again |
| `CRAWLER_MAX_BACK_OFF_HOURS` | `24` | The longest back-off |
| `CRAWLER_RESPECT_ROBOTS` | `true` | Skip URLs the site's robots.txt disallows |
| `CRAWLER_BROWSER_PATH` | _(empty)_ | A Chromium binary to use instead of Playwright's own |

| `PINECONE_API_KEY` / `PINECONE_INDEX_NAME` | — | The index listings are written to (created if missing: serverless, cosine, AWS us-east-1) |
| `SENTENCE_TRANSFORMER_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model, which must be the one the searching app uses |

Settings are read from the environment and from `.env` in the repository root (`property_crawler/settings.py`). Inside uk-agents, that is uk-agents' own `.env`.

## Good to know

- **Check each site's terms of use.** Many listing portals forbid scraping. Some block headless browsers or show a CAPTCHA. The crawler does not try to get around either: it backs off (see [Wait, stop and see](#wait-stop-and-see)).
- **Listings are not removed** when they disappear from a site. `crawled_at` tells you how fresh a listing is.

## Running it on the server

`deployment/deploy.sh` starts the crawler with the REST API, Telegram bot and Streamlit UI (service name `crawler`). It restarts the crawler if it dies and redeploys it on every merge to `main`. Every build also runs `playwright install chromium`, so the browser always matches the Playwright version in `uv.lock`. Set-up on Ubuntu (22.04 or 24.04), once, on top of [deployment/README.md](../deployment/README.md):

```bash
cd ~/uk-agents
uv sync
sudo .venv/bin/playwright install-deps chromium   # the system libraries Chromium needs (apt)
uv run python -m property_crawler.main --once   # try a pass; check logs/crawler.log
```

- **No screen is needed.** `CRAWLER_HEADLESS=true` is the default. Leave it on: `false` needs a display, and the server has none.
- **Run it as your normal user, not root.** The browser download goes to `~/.cache/ms-playwright` of the user running `deploy.sh`.
- **Logs**: `logs/crawler.log` (every pass, every site; rotating) and `logs/deploy-crawler.log` (the process's raw output). A failed browser download is in `logs/deploy-playwright.log`.
- **Memory**: about 0.5–1 GB while a pass runs (embedding model plus Chromium). Chromium is closed between passes.
- **To leave it out**, set `DEPLOY_SERVICES="rest telegram streamlit"`.

## Moving to its own repository

Copy `property_crawler/` into the new repository's root and add:

- **`pyproject.toml`** with Python ≥ 3.13 and the dependencies `playwright`, `beautifulsoup4`, `requests`, `python-dotenv`, `numpy`, `pinecone` and `sentence-transformers`, plus `pytest` for development, with `testpaths = ["property_crawler/tests"]`.
- **`.env`** with the `CRAWLER_*`, `PINECONE_*` and `SENTENCE_TRANSFORMER_MODEL` settings above.
- **A way to run it on the server.** For example, a systemd unit running `uv run python -m property_crawler.main`. In uk-agents, `deployment/deploy.sh` does this; afterwards, set `DEPLOY_SERVICES="rest telegram streamlit"` there.

Then delete `property_crawler/` from uk-agents and drop `property_crawler/tests` from its `testpaths`. uk-agents keeps reading the same index, so nothing else changes. The tests already run from a copy outside uk-agents (`python -m pytest property_crawler/tests` from the copy's root).
