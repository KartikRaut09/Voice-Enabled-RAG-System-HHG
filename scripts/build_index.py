"""Script to build and persist FAISS dense vector index for retrieval.

Usage:
  python scripts/build_index.py --model sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --strategy passage
  python scripts/build_index.py --model intfloat/multilingual-e5-small --strategy passage
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

from backend.app.chunking import get_chunker
from backend.app.embeddings import SentenceTransformerEmbedder
from backend.app.vector_store import FAISSVectorStore


def load_config() -> dict:
    config_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_and_save_index(
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    strategy: str = "passage",
    output_dir: Path | None = None,
) -> Path:
    base_dir = Path(__file__).resolve().parent.parent
    config = load_config()
    dev_path = base_dir / "data" / "processed" / "dev" / "dev.jsonl"

    from scripts.build_chunks import get_or_stream_dev_records
    dev_records = get_or_stream_dev_records(config, dev_path)
    print(f"Loaded {len(dev_records)} corpus records.")

    # 1. Chunk records
    chunker = get_chunker(strategy, config)
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

    print(f"Generated {len(all_chunks)} chunks with strategy '{strategy}'.")

    # 2. Instantiate embedder
    embedder = SentenceTransformerEmbedder(model_name=model_name, device="cpu", normalize=True)

    # 3. Encode chunks in batches
    chunk_texts = [c.text for c in all_chunks]
    chunk_meta = [c.to_dict() for c in all_chunks]

    print(f"Encoding {len(chunk_texts)} chunks with {model_name}...")
    t0 = time.perf_counter()
    embeddings = embedder.encode_documents(chunk_texts, batch_size=64)
    t_encode = time.perf_counter() - t0
    print(f"Encoding completed in {t_encode:.2f}s ({len(chunk_texts)/t_encode:.1f} chunks/s).")

    # 4. Build FAISS vector store
    store = FAISSVectorStore(dimension=embedder.dimension, metric="cosine")
    store.add(embeddings, chunk_meta)

    # 5. Persist to disk
    model_slug = model_name.split("/")[-1].replace("-", "_")
    target_dir = output_dir or (base_dir / "data" / "indexes" / f"{model_slug}_{strategy}")
    store.save(target_dir)
    print(f"Vector index saved to {target_dir} ({store.size} vectors).")

    # 6. Verify index reload
    reloaded = FAISSVectorStore.load(target_dir)
    assert reloaded.size == store.size, "Reloaded index size mismatch!"
    print(f"VERIFIED: Reloaded index from {target_dir} successfully ({reloaded.size} vectors).")

    return target_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and persist dense vector index.")
    parser.add_argument(
        "--model",
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Embedding model identifier",
    )
    parser.add_argument(
        "--strategy",
        default="passage",
        choices=["passage", "fixed", "overlap", "structure_aware"],
        help="Chunking strategy",
    )
    args = parser.parse_args()
    build_and_save_index(model_name=args.model, strategy=args.strategy)


if __name__ == "__main__":
    main()
