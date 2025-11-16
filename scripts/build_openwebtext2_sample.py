#!/usr/bin/env python3

import argparse
import io
import json
from pathlib import Path
import sys
from typing import Iterator

import requests
import zstandard
from datasets import load_dataset


def normalize_text(text: str) -> str:
    sanitized = text.replace("\r", " ").replace("\n", " ")
    parts = sanitized.split()
    return " ".join(parts)


def write_split(stream_iter: Iterator[str], out_path: Path, count: int) -> int:
    written = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        while written < count:
            try:
                candidate = next(stream_iter)
            except StopIteration as exc:
                raise RuntimeError(
                    f"Source stream exhausted before {out_path.name} reached {count} lines"
                ) from exc
            text = normalize_text(candidate)
            if not text:
                continue
            handle.write(text + "\n")
            written += 1
    return written


class LimitedTarStream:
    def __init__(self, base_stream, remaining: int):
        self.base_stream = base_stream
        self.remaining = remaining

    def read(self, size: int = -1) -> bytes:
        if self.remaining <= 0:
            return b""
        if size < 0 or size > self.remaining:
            size = self.remaining
        chunk = self.base_stream.read(size)
        self.remaining -= len(chunk)
        return chunk


def _parse_tar_size(field: bytes) -> int:
    if not field:
        return 0
    if field[0] & 0x80:
        return int.from_bytes(field, byteorder="big", signed=True)
    size_text = field.decode("ascii", errors="ignore").strip("\x00 ")
    return int(size_text or "0", 8)


def _read_exact(stream, size: int) -> bytes:
    if size <= 0:
        return b""
    chunks = []
    remaining = size
    while remaining > 0:
        chunk = stream.read(remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def remote_text_stream(url: str) -> Iterator[str]:
    with requests.get(url, stream=True) as resp:
        resp.raise_for_status()
        while True:
            header = resp.raw.read(512)
            if not header or header == b"\0" * 512:
                break
            name = header[0:100].decode("utf-8", errors="ignore").rstrip("\x00")
            size = _parse_tar_size(header[124:136])
            typeflag = header[156:157]
            if typeflag in (b"0", b"\0") and name.endswith(".jsonl.zst"):
                limiter = LimitedTarStream(resp.raw, size)
                dctx = zstandard.ZstdDecompressor()
                reader = dctx.stream_reader(limiter)
                text_stream = io.TextIOWrapper(reader, encoding="utf-8")
                try:
                    for line in text_stream:
                        payload = json.loads(line)
                        text = payload.get("text", "")
                        if text:
                            yield text
                finally:
                    text_stream.close()
                padding = (512 - (size % 512)) % 512
                if padding:
                    _read_exact(resp.raw, padding)
                continue
            _ = _read_exact(resp.raw, size)
            padding = (512 - (size % 512)) % 512
            if padding:
                _read_exact(resp.raw, padding)


def dataset_text_stream(dataset_id: str, split: str) -> Iterator[str]:
    dataset = load_dataset(dataset_id, split=split, streaming=True)
    for sample in dataset:
        text = sample.get("text", "")
        if text:
            yield text


def build_source_iterator(args) -> Iterator[str]:
    if not args.use_dataset:
        return remote_text_stream(args.remote_url)
    try:
        return dataset_text_stream(args.dataset, args.split)
    except Exception as exc:  # pragma: no cover - fallback path
        print(
            f"[warn] dataset '{args.dataset}' unavailable ({exc}); streaming from {args.remote_url}",
            file=sys.stderr,
        )
        return remote_text_stream(args.remote_url)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Stream a lightweight OpenWebText2 sample into raw text files."
    )
    parser.add_argument(
        "--dataset",
        default="allenai/openwebtext2",
        help="🤗 dataset repository to stream from (default: %(default)s)",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="dataset split to stream (default: %(default)s)",
    )
    parser.add_argument(
        "--remote-url",
        default="https://huggingface.co/datasets/segyges/OpenWebText2/resolve/main/openwebtext2.jsonl.zst.tar",
        help="remote url to stream tarred zstd payload (default: %(default)s)",
    )
    parser.add_argument(
        "--output-dir",
        default="data_sets",
        help="directory to place the *.raw files (default: %(default)s)",
    )
    parser.add_argument(
        "--base-name",
        default="openwebtext2",
        help="base filename for the emitted raw files (default: %(default)s)",
    )
    parser.add_argument("--train-count", type=int, default=10000)
    parser.add_argument("--val-count", type=int, default=1000)
    parser.add_argument("--test-count", type=int, default=1000)
    parser.add_argument(
        "--use-dataset",
        action="store_true",
        help="try huggingface datasets streaming first (falls back to remote url on failure)",
    )
    args = parser.parse_args()

    source_iter = build_source_iterator(args)
    out_dir = Path(args.output_dir)
    splits = [
        ("train", args.train_count),
        ("valid", args.val_count),
        ("test", args.test_count),
    ]
    for split_name, count in splits:
        out_path = out_dir / f"{args.base_name}.{split_name}.raw"
        print(f"Writing {count} samples to {out_path}")
        actual = write_split(source_iter, out_path, count)
        print(f"Finished {split_name}: wrote {actual} entries")


if __name__ == "__main__":
    main()
