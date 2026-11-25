"""
Remove every vector the crawler did not write (e.g. data loaded into the index some other
way) from the index, so searches only find crawled listings.

Run from the repository root:
    python -m property_crawler.purge          # dry run: only counts
    python -m property_crawler.purge --yes    # delete them
"""
import argparse
import logging

from .extractor import ID_PREFIX

logger = logging.getLogger(__name__)


def foreign_ids(vector_index) -> list[str]:
    """Ids in the index that the crawler did not write."""
    return [i for page in vector_index.list_ids() for i in page if not i.startswith(ID_PREFIX)]


def purge(vector_index, delete: bool) -> int:
    ids = foreign_ids(vector_index)
    if delete and ids:
        vector_index.delete(ids)
    return len(ids)


def main(argv: list[str] | None = None) -> None:
    from .runtime import setup_logging
    from .settings import Settings
    from .vector_index import VectorIndex

    setup_logging()
    parser = argparse.ArgumentParser(description="Delete every non-crawler vector from the index.")
    parser.add_argument("--yes", action="store_true", help="really delete (default: dry run)")
    args = parser.parse_args(argv)
    s = Settings.from_env()
    count = purge(VectorIndex(s.pinecone_api_key, s.pinecone_index_name, s.embedding_model), delete=args.yes)
    if args.yes:
        logger.info("[crawler purge] deleted %d vector(s) the crawler did not write", count)
    else:
        logger.info("[crawler purge] dry run: %d vector(s) would be deleted — run with --yes to delete", count)


if __name__ == "__main__":
    main()
