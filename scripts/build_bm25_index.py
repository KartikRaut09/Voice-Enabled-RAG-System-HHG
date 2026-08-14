"""Build and persist BM25 lexical index from development dataset.

Constructs an Okapi BM25 inverted index for the configured chunking strategy
and saves index, metadata, and manifest to disk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
import yaml

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.bm25 import BM25Index
from backend.app.chunking import get_chunker
from scripts.build_chunks import get_or_stream_dev_records


def load_config() -> dict:
    config_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and save BM25 lexical index.")
    parser.add_argument("--strategy", type=str, default="structure_aware", help="Chunking strategy")
    parser.add_argument("--k1", type=float, default=1.5, help="BM25 k1 parameter")
    parser.add_argument("--b", type=float, default=0.75, help="BM25 b parameter")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/indexes/bm25_structure_aware",
        help="Directory to persist BM25 index",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent.parent
    config = load_config()
    dev_path = base_dir / "data" / "processed" / "dev" / "dev.jsonl"

    print("Loading corpus records...")
    dev_records = get_or_stream_dev_records(config, dev_path)
    print(f"Loaded {len(dev_records)} corpus records.")

    chunker = get_chunker(args.strategy, config)
    all_chunks = []
    for rec in dev_records:
        rec_meta = {
            "query_id": rec.get("query_id"),
            "query_type": rec.get("query_type"),
            "source_lang": rec.get("source_lang"),
            "target_lang": rec.get("target_lang"),
        }
        for p in rec.get("passages", []):
            all_chunks.extend(chunker.chunk_passage(p, rec_meta))

    print(f"Generated {len(all_chunks)} chunks with strategy '{args.strategy}'.")

    chunk_texts = [c.text for c in all_chunks]
    chunk_meta = [c.to_dict() for c in all_chunks]

    t0 = time.perf_counter()
    index = BM25Index(k1=args.k1, b=args.b)
    index.build(chunk_texts, chunk_meta)
    build_time = time.perf_counter() - t0
    print(f"BM25 index built in {build_time:.3f}s ({len(all_chunks) / build_time:.1f} chunks/s).")

    out_path = base_dir / args.output_dir
    index.save(out_path)
    print(f"BM25 index saved to {out_path} ({index.size} documents).")

    # Verify reloading
    loaded = BM25Index.load(out_path)
    print(f"VERIFIED: Reloaded BM25 index from {out_path} successfully ({loaded.size} documents).")


if __name__ == "__main__":
    main()
