#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""
Seed a global MinHash deduplication database from JSONL inputs.

This script mirrors the doc_id and MinHash logic used by JSONLQualityScorer
so the seeded store remains compatible with runtime checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections.abc import Iterable, Iterator
from pathlib import Path

# Allow running from a source checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sievio.core.dedup_store import GlobalDedupStore
from sievio.core.qc_utils import minhash_signature_for_text, open_jsonl_maybe_gz


def _iter_signatures(paths: Iterable[str], *, k: int, n_perm: int) -> Iterator[tuple[str, tuple[int, ...], str]]:
    for path in paths:
        with open_jsonl_maybe_gz(path) as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                text = rec.get("text", "")
                meta = rec.get("meta") or {}
                doc_id = meta.get("doc_id") or rec.get("id")
                if doc_id:
                    doc_id = str(doc_id)
                else:
                    doc_id = hashlib.sha1(text.encode("utf-8")).hexdigest()
                content_hash = meta.get("sha256") or hashlib.sha256(text.encode("utf-8")).hexdigest()
                sig = minhash_signature_for_text(text, k=k, n_perm=n_perm)
                yield doc_id, sig, content_hash


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed a SQLite MinHash deduplication store from JSONL files.")
    parser.add_argument("db_path", help="Path to the dedup SQLite database.")
    parser.add_argument("inputs", nargs="+", help="Input JSONL/JSONL.GZ paths to index.")
    parser.add_argument("--k", type=int, default=5, help="Shingle size (bytes) for MinHash signatures.")
    parser.add_argument("--perm", type=int, default=128, help="Number of MinHash permutations.")
    parser.add_argument("--bands", type=int, default=32, help="Number of LSH bands.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.82,
        help="Jaccard threshold used for duplicate decisions.",
    )
    args = parser.parse_args(argv)

    print(
        f"Using MinHash k={args.k}, n_perm={args.perm}. "
        "Ensure these match qc.scorer_options.minhash_* in your config.",
        file=sys.stderr,
    )

    with GlobalDedupStore(
        args.db_path,
        n_perm=args.perm,
        bands=args.bands,
        jaccard_threshold=args.threshold,
        persistent_connection=True,
    ) as store:
        store.bulk_add(_iter_signatures(args.inputs, k=args.k, n_perm=args.perm))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
