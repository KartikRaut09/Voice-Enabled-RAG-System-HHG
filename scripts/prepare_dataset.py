"""MSMARCO-XI dataset preparation script.

Deterministically streams, normalizes, and partitions subsets of ai4bharat/MSMARCO-XI
into development (data/processed/dev/dev.jsonl) and evaluation (data/processed/evaluation/evaluation.jsonl)
datasets, along with a reproducible manifest (data/processed/manifest.json).
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import sys
import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset

from backend.app.dataset import normalize_record


def load_config() -> dict:
    config_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def prepare_dataset() -> dict:
    config = load_config()
    ds_config = config.get("dataset", {})
    seed = int(ds_config.get("sample_seed", 42))
    languages = ds_config.get("languages", [])
    dev_per_lang = int(ds_config.get("dev_samples_per_language", 200))
    eval_per_lang = int(ds_config.get("eval_samples_per_language", 50))
    output_dir = Path(__file__).resolve().parent.parent / ds_config.get("output_dir", "data/processed")

    random.seed(seed)

    dev_dir = output_dir / "dev"
    eval_dir = output_dir / "evaluation"
    dev_dir.mkdir(parents=True, exist_ok=True)
    eval_dir.mkdir(parents=True, exist_ok=True)

    dev_file = dev_dir / "dev.jsonl"
    eval_file = eval_dir / "evaluation.jsonl"
    manifest_file = output_dir / "manifest.json"

    print(f"Preparing MSMARCO-XI subsets (seed={seed}, dev_per_lang={dev_per_lang}, eval_per_lang={eval_per_lang})...")

    dev_count = 0
    eval_count = 0
    dev_query_ids: set[int] = set()

    with open(dev_file, "w", encoding="utf-8") as f_dev, open(eval_file, "w", encoding="utf-8") as f_eval:
        for lang in languages:
            code = lang["code"]
            name = lang["name"]

            # 1. Dev Partition (using train file if available, otherwise val file)
            train_file = lang.get("train_file") or lang.get("val_file")
            train_url = f"https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/{train_file}"
            print(f"Streaming dev records for {name} ({code}) from {train_file}...")

            stream_train = load_dataset("parquet", data_files={code: train_url}, streaming=True)
            lang_dev_count = 0
            for raw in stream_train[code]:
                norm = normalize_record(raw)
                norm["lang_code"] = code
                f_dev.write(json.dumps(norm, ensure_ascii=False) + "\n")
                f_dev.flush()
                dev_query_ids.add(norm["query_id"])
                lang_dev_count += 1
                dev_count += 1
                if lang_dev_count >= dev_per_lang:
                    break

            # 2. Evaluation Partition (using validation file, ensuring disjoint queries)
            val_file = lang.get("val_file")
            val_url = f"https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/{val_file}"
            print(f"Streaming evaluation records for {name} ({code}) from {val_file}...")

            stream_val = load_dataset("parquet", data_files={code: val_url}, streaming=True)
            lang_eval_count = 0
            skip_count = dev_per_lang if train_file == val_file else 0
            skipped = 0

            for raw in stream_val[code]:
                if skipped < skip_count:
                    skipped += 1
                    continue

                norm = normalize_record(raw)
                if norm["query_id"] in dev_query_ids and train_file != val_file:
                    continue

                norm["lang_code"] = code
                f_eval.write(json.dumps(norm, ensure_ascii=False) + "\n")
                f_eval.flush()
                lang_eval_count += 1
                eval_count += 1
                if lang_eval_count >= eval_per_lang:
                    break

    # Compute Schema & Manifest Hash
    schema_definition = {
        "query_id": "int64",
        "query": "string",
        "english_query": "string",
        "answer": "string",
        "english_answer": "string",
        "query_type": "string",
        "source_lang": "string",
        "target_lang": "string",
        "passages": "list[dict(passage_id, text, english_text, language, is_selected)]",
        "meta": "dict",
    }
    schema_hash = hashlib.sha256(json.dumps(schema_definition, sort_keys=True).encode()).hexdigest()

    manifest = {
        "dataset": ds_config.get("name", "ai4bharat/MSMARCO-XI"),
        "hf_url": ds_config.get("hf_url"),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "dev_sample_size": dev_count,
        "evaluation_sample_size": eval_count,
        "languages": [l["code"] for l in languages],
        "language_names": [l["name"] for l in languages],
        "schema_hash": schema_hash,
        "schema_definition": schema_definition,
        "disjoint_dev_eval": True,
    }

    with open(manifest_file, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"Manifest written to {manifest_file}")
    print(f"VERIFIED: dev.jsonl ({dev_file.stat().st_size} bytes, {dev_count} rows), eval.jsonl ({eval_file.stat().st_size} bytes, {eval_count} rows), manifest.json ({manifest_file.stat().st_size} bytes)")
    return manifest


if __name__ == "__main__":
    prepare_dataset()
