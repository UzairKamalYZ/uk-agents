"""
Index the Polish housing CSVs into the vector store.

Run from the project root:
    python -m agents.property.gatherers.pl_housing_data_collector

Reads every *.csv in HOUSING_DATASET_DIR (default agents/property/datasets/pl-housing).
"""
import logging
from pathlib import Path

from core.src.config.config import Config
from agents.property.housing.housing_data_collector import HousingDataCollector

logger = logging.getLogger(__name__)

BATCH_SIZE = 100


def ingest(collector, dataset_dir: Path, batch_size: int = BATCH_SIZE) -> int:
    """Embed every CSV row in dataset_dir and persist them in batches; returns the row count."""
    if not any(Path(dataset_dir).glob("*.csv")):
        logger.error("pl-housing: no CSV files in %s — set HOUSING_DATASET_DIR or add the files", dataset_dir)
        return 0
    documents: dict = {}
    count = 0
    for row in collector.stream_csv_files(dataset_dir):
        doc = dict(row)  # copy all CSV fields
        doc["text"] = collector.generateEmbededDocument(row)
        documents[row["id"]] = doc
        count += 1
        if len(documents) == batch_size:
            collector.persist(documents)
            documents = {}
    if documents:  # the last, partial batch
        collector.persist(documents)
    logger.info("pl-housing: indexed %d rows from %s", count, dataset_dir)
    return count


if __name__ == "__main__":
    from logging_config import setup_logging

    setup_logging()
    with HousingDataCollector() as collector:
        ingest(collector, Path(Config.HOUSING_DATASET_DIR))
