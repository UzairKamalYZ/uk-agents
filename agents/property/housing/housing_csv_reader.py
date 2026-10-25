from pathlib import Path
import pandas as pd

from core.src.config.config import Config

DATASET_DIR = Path(Config.HOUSING_DATASET_DIR)
CHUNK_SIZE = 50_000


def stream_csv_files(dataset_dir: Path, chunk_size: int = CHUNK_SIZE):
    """
    Generator that yields rows (dict) from all CSV files in a folder,
    reading in chunks to avoid memory issues.
    """
    for csv_file in dataset_dir.glob("*.csv"):
        print(f"Reading {csv_file.name} ...")
        for chunk in pd.read_csv(
            csv_file,
            chunksize=chunk_size,
            encoding="utf-8",
            on_bad_lines="skip",
        ):
            for row in chunk.to_dict(orient="records"):
                yield row


def main():
    count = 0

    for row in stream_csv_files(DATASET_DIR):
        count += 1

        # Example: print first 3 rows only
        if count <= 3:
            print(row)

    print(f"\nTotal rows processed: {count}")


if __name__ == "__main__":
    main()
