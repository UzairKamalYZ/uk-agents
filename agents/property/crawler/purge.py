"""
Remove every vector the crawler did not write (e.g. the old Polish CSV data) from the
property index, so the property agent only finds crawled listings.

Run from the project root:
    python -m agents.property.crawler.purge          # dry run: only counts
    python -m agents.property.crawler.purge --yes    # delete them
"""
import argparse
import logging

from .extractor import ID_PREFIX

logger = logging.getLogger(__name__)


def foreign_ids(store) -> list[str]:
    """Ids in the store that the crawler did not write."""
    return [i for page in store.list_ids() for i in page if not i.startswith(ID_PREFIX)]


def purge(store, delete: bool) -> int:
    ids = foreign_ids(store)
    if delete and ids:
        store.delete(ids)
    return len(ids)


def main(argv: list[str] | None = None) -> None:
    from logging_config import setup_logging
    from core.src.config.config import Config
    from core.src.persistence.factory import create_vector_store

    setup_logging()
    parser = argparse.ArgumentParser(description="Delete every non-crawler vector from the property index.")
    parser.add_argument("--yes", action="store_true", help="really delete (default: dry run)")
    args = parser.parse_args(argv)
    store = create_vector_store(Config.STORE_TYPE, dim=384, config={})
    count = purge(store, delete=args.yes)
    if args.yes:
        logger.info("[crawler purge] deleted %d vector(s) the crawler did not write", count)
    else:
        logger.info("[crawler purge] dry run: %d vector(s) would be deleted — run with --yes to delete", count)


if __name__ == "__main__":
    main()
