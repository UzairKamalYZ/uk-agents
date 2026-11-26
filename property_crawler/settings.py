"""
The crawler's settings, read from the environment and from `.env` in the directory that
holds this package (the repository root). Self-contained: nothing here depends on the
code of the application that searches the index.
"""
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

PACKAGE_DIR = Path(__file__).resolve().parent
ROOT = PACKAGE_DIR.parent

load_dotenv(ROOT / ".env")


def _user_data_dir() -> Path:
    """Per-user data directory, outside the checkout (survives re-clones)."""
    if os.name == "nt":
        base = Path(os.getenv("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.getenv("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "property-crawler"


def _path(name: str, default: Path) -> Path:
    """A relative value resolves from the repository root; "~" is expanded."""
    return ROOT / Path(os.getenv(name) or default).expanduser()


def _bool(name: str, default: str) -> bool:
    return (os.getenv(name) or default).strip().lower() == "true"


@dataclass(frozen=True)
class Settings:
    sites_file: Path
    db_file: Path
    interval_minutes: int = 60          # used when run_at is empty
    run_at: str = "04:30,13:00"         # fixed pass times, comma-separated HH:MM in `timezone`
    timezone: str = "Europe/Brussels"
    headless: bool = True
    timeout_seconds: float = 30
    page_delay_seconds: float = 8
    back_off_minutes: int = 60
    max_back_off_hours: int = 24
    respect_robots: bool = True
    browser_path: str = ""
    # Where listings go — the same index, embedding model and namespace the searching app uses
    pinecone_api_key: str = ""
    pinecone_index_name: str = ""
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            sites_file=_path("CRAWLER_SITES_FILE", PACKAGE_DIR / "sites.json"),
            db_file=_path("CRAWLER_DB_FILE", _user_data_dir() / "seen.db"),
            interval_minutes=int(os.getenv("CRAWLER_INTERVAL_MINUTES") or 60),
            # Unset → twice a day; set but empty → every CRAWLER_INTERVAL_MINUTES instead
            run_at=os.environ.get("CRAWLER_RUN_AT", "04:30,13:00"),
            timezone=os.getenv("CRAWLER_TIMEZONE") or "Europe/Brussels",
            headless=_bool("CRAWLER_HEADLESS", "true"),
            timeout_seconds=float(os.getenv("CRAWLER_TIMEOUT_SECONDS") or 30),
            page_delay_seconds=float(os.getenv("CRAWLER_PAGE_DELAY_SECONDS") or 8),
            back_off_minutes=int(os.getenv("CRAWLER_BACK_OFF_MINUTES") or 60),
            max_back_off_hours=int(os.getenv("CRAWLER_MAX_BACK_OFF_HOURS") or 24),
            respect_robots=_bool("CRAWLER_RESPECT_ROBOTS", "true"),
            browser_path=os.getenv("CRAWLER_BROWSER_PATH", ""),
            pinecone_api_key=os.getenv("PINECONE_API_KEY", ""),
            pinecone_index_name=os.getenv("PINECONE_INDEX_NAME", ""),
            embedding_model=os.getenv("SENTENCE_TRANSFORMER_MODEL") or "sentence-transformers/all-MiniLM-L6-v2",
        )
