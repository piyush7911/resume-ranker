"""
loader.py — Transparent JSONL / gzipped-JSONL streaming.

The participant bundle ships candidates.jsonl.gz (~52MB) which unzips to
candidates.jsonl (~465MB). We support both so the same code path works on the
raw bundle and on an unpacked file, with no extra disk required for the .gz.
"""

import gzip
import orjson


def open_bytes(path):
    """Open a path in binary mode, decompressing transparently if it's .gz."""
    if str(path).endswith(".gz"):
        return gzip.open(path, "rb")
    return open(path, "rb")


def iter_candidates(path):
    """Yield candidate dicts from a (optionally gzipped) JSONL file."""
    with open_bytes(path) as f:
        for line in f:
            if not line.strip():
                continue
            yield orjson.loads(line)
