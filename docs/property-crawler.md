# Property listing crawler

`agents/property/crawler/` keeps the property agent's Pinecone index fresh. Every hour it visits the websites in `sites.json` one at a time. It renders each listing page in headless Chromium (Playwright) so JavaScript-built pages work, reads the listing cards with CSS selectors, and upserts every **new or changed** listing into the same index the property agent searches.

```bash
uv run playwright install chromium                 # once per machine (add --with-deps on a bare Linux server)
python -m agents.property.crawler.main             # a pass now, then every CRAWLER_INTERVAL_MINUTES
python -m agents.property.crawler.main --once      # one pass, then exit (for a system crontab)
```

## One pass

```
for each enabled site in sites.json (in order):
    fresh browser context (no cookies carried between sites)
    for each start URL:
        page = start URL
        while page and pages < max_pages:
            robots.txt allows it?          (CRAWLER_RESPECT_ROBOTS)
            render in Chromium, wait for `wait_for`, scroll `scrolls` times
            cards → listings               (extractor.py, CSS selectors)
            new or changed? (seen.db)      → embed + upsert to Pinecone, then mark seen
            nothing new on this page?      → stop paging (stop_when_seen)
            page = the `next_page_selector` link
        wait CRAWLER_PAGE_DELAY_SECONDS between page loads
```

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
      "start_urls": ["https://portal.example/rent/warsaw?sort=newest"],
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
      "currency": "PLN",
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
| `fields` | `{}` | Field → CSS selector *inside the card* (its text), or `{"selector", "attr"}` for an attribute. `price`, `rooms` and `area` are parsed into numbers (`3 500 zł` → 3500, `48,5 m²` → 48.5). `title`, `city`, `type`, `description` and `url` are known fields; any other field (e.g. `floor`) is added to the text and metadata as is |
| `link_selector` | `a[href]` | Link to the listing inside the card. It is the listing's identity. A card without a link gets an ID from its text |
| `next_page_selector` | none | Link to the next results page |
| `max_pages` | `1` | Pages followed per start URL |
| `wait_for` | `listing_selector` | Selector to wait for before reading the page |
| `scrolls` | `0` | Scroll to the bottom this many times first (infinite-scroll sites) |
| `stop_when_seen` | `true` | Stop paging when a page has no new or changed listing |
| `city`, `property_type` | none | Used when the card doesn't name one |
| `currency` | `PLN` | Shown with the price |
| `enabled` | `true` | `false` keeps the entry without crawling it |

The shipped `sites.json` holds one disabled example. To find selectors, open a listing page in Chrome, right-click a listing → Inspect, and pick the class shared by every card. A site whose selectors match nothing logs `no listings matched … check the selectors`.

## What lands in Pinecone

Each listing is embedded as text, for example `Sunny 2-bed. Apartment in Warsaw. Price 3400 PLN. 2 rooms. 52 square meters. …`. The text comes with metadata that `core/src/model/context_builder.py` shows the agent: `text`, `city`, `price`, `currency`, `rooms`, `squareMeters`, `type` and `url` (shown to the agent as `Link:`), plus `source: "crawler"`, `site`, `title`, `crawled_at` and any extra fields. Crawled listings share the default namespace with the CSV data, so the property agent searches both at once.

## Settings

| Variable | Default | Meaning |
|---|---|---|
| `CRAWLER_SITES_FILE` | `agents/property/crawler/sites.json` | Site list |
| `CRAWLER_INTERVAL_MINUTES` | `60` | Minutes between the end of one pass and the start of the next |
| `CRAWLER_DB_FILE` | per-user data dir `/crawler/seen.db` | Which listings are already indexed (delete it to re-index everything) |
| `CRAWLER_HEADLESS` | `true` | `false` shows the browser window (debugging selectors) |
| `CRAWLER_TIMEOUT_SECONDS` | `30` | Page load / `wait_for` timeout |
| `CRAWLER_PAGE_DELAY_SECONDS` | `3` | Pause between page loads |
| `CRAWLER_RESPECT_ROBOTS` | `true` | Skip URLs the site's robots.txt disallows |
| `CRAWLER_BROWSER_PATH` | _(empty)_ | A Chromium binary to use instead of Playwright's own |

Pinecone and the embedding model come from the usual `PINECONE_*` and `SENTENCE_TRANSFORMER_MODEL` settings.

## Good to know

- **Check each site's terms of use.** Many listing portals forbid scraping. Some block headless browsers or show a CAPTCHA, and the crawler does not try to get around either. Such a site logs `no listings matched`.
- **Listings are not removed** when they disappear from a site. `crawled_at` tells you how fresh a listing is.
- **The crawler is not started by `deployment/deploy.sh`** (it runs clients only). Run it alongside, like the mail and WhatsApp monitors: `nohup uv run python -m agents.property.crawler.main >> logs/crawler.log 2>&1 &`, or add `--once` to a crontab line: `0 * * * * cd ~/uk-agents && uv run python -m agents.property.crawler.main --once`.
- The older `agents/property/gatherers/data_collector.py` (requests + regex over `urls.txt`) still works for static pages without per-site selectors.
