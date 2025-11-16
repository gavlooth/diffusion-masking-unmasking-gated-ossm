#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import List, TextIO


def interleave(paths: List[Path], out_path: Path) -> None:
    streams: List[TextIO] = []
    for path in paths:
        if path.exists():
            streams.append(path.open("r", encoding="utf-8"))
    if not streams:
        raise RuntimeError("No source files found to interleave")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        active = list(enumerate(streams))
        with out_path.open("w", encoding="utf-8") as handle:
            while active:
                next_active = []
                for idx, stream in active:
                    line = stream.readline()
                    if not line:
                        continue
                    handle.write(line.rstrip("\n") + "\n")
                    next_active.append((idx, stream))
                active = next_active
    finally:
        for stream in streams:
            stream.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Interleave pg19 and OpenWebText2 raw datasets.")
    parser.add_argument(
        "--output-dir",
        default="data_sets",
        help="directory for the unified *.raw files (default: %(default)s)",
    )
    parser.add_argument(
        "--prefix",
        default="unified",
        help="base name for interleaved files (default: %(default)s)",
    )
    args = parser.parse_args()
    out_dir = Path(args.output_dir)
    mapping = {
        "train": [
            out_dir / "pg19.train.raw",
            out_dir / "openwebtext2.train.raw",
        ],
        "valid": [
            out_dir / "pg19.valid.raw",
            out_dir / "openwebtext2.valid.raw",
        ],
        "test": [
            out_dir / "pg19.test.raw",
            out_dir / "openwebtext2.test.raw",
        ],
    }
    for split, sources in mapping.items():
        target = out_dir / f"{args.prefix}.{split}.raw"
        interleave(sources, target)
        print(f"Wrote {target}")


if __name__ == "__main__":
    main()
