#!/usr/bin/env python3

import argparse
from pathlib import Path

import duckdb


def normalize(text: str) -> str:
    return " ".join(text.replace("\r", " ").replace("\n", " ").split())


def export_split(con, pattern: str, limit: int, out_path: Path) -> None:
    sql = f"""
    SELECT text
    FROM read_parquet('{pattern}')
    LIMIT {limit}
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for (text,) in con.execute(sql).fetchall():
            if not text:
                continue
            cleaned = normalize(text)
            if cleaned:
                handle.write(cleaned + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract compact PG-19 samples from parquet via DuckDB."
    )
    parser.add_argument(
        "--root",
        default="/mnt/exfat-data/Downloads/pg19/data",
        help="directory containing the pg19 parquet shards (default: %(default)s)",
    )
    parser.add_argument("--train-limit", type=int, default=2000)
    parser.add_argument("--valid-limit", type=int, default=200)
    parser.add_argument("--test-limit", type=int, default=200)
    parser.add_argument(
        "--output-dir",
        default="data_sets",
        help="destination for the *.raw files (default: %(default)s)",
    )
    args = parser.parse_args()
    root = Path(args.root)
    con = duckdb.connect()
    export_split(
        con,
        str(root / "train-*.parquet"),
        args.train_limit,
        Path(args.output_dir) / "pg19.train.raw",
    )
    export_split(
        con,
        str(root / "validation-*.parquet"),
        args.valid_limit,
        Path(args.output_dir) / "pg19.valid.raw",
    )
    export_split(
        con,
        str(root / "test-*.parquet"),
        args.test_limit,
        Path(args.output_dir) / "pg19.test.raw",
    )


if __name__ == "__main__":
    main()
