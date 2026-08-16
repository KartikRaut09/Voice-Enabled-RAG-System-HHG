"""
Dataset utilities for MSMARCO-XI.
Handles loading, language selection, sampling, passage extraction, and validation.
"""

from __future__ import annotations

from typing import Any

from datasets import load_dataset, Dataset, DatasetDict


# ── Supported Languages ──

MSMARCO_XI_LANGUAGES = {
    "ar": "Arabic",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "nl": "Dutch",
    "pt": "Portuguese",
    "ru": "Russian",
    "vi": "Vietnamese",
    "zh": "Chinese",
    "bn": "Bengali",
    "mr": "Marathi",
    "ta": "Tamil",
    "te": "Telugu",
}

INDIC_LANGUAGES = ["hi", "mr", "bn", "ta", "te"]


def list_available_configs(dataset_name: str = "ai4bharat/MSMARCO-XI") -> list[str]:
    """List available dataset configurations (language codes) from Hugging Face.

    Returns:
        List of configuration name strings.
    """
    from huggingface_hub import HfApi

    api = HfApi()
    info = api.dataset_info(dataset_name)
    if info.card_data and hasattr(info.card_data, "configs"):
        return [c.config_name for c in info.card_data.configs]
    # Fallback: return known languages
    return list(MSMARCO_XI_LANGUAGES.keys())


def load_msmarco_xi(
    language: str = "hi",
    split: str = "validation",
    sample_size: int | None = 5000,
    streaming: bool = False,
    cache_dir: str | None = None,
    seed: int = 42,
) -> Dataset:
    """Load an MSMARCO-XI language split with optional sampling.

    Args:
        language: Language configuration code (e.g. ``"hi"``).
        split: Dataset split (``"train"``, ``"validation"``, ``"test"``).
        sample_size: Number of examples to sample.  ``None`` = load all.
        streaming: If True, use streaming mode (avoids full download).
        cache_dir: Custom HF cache directory.
        seed: Random seed for reproducible sampling.

    Returns:
        A Hugging Face ``Dataset`` object.
    """
    ds = load_dataset(
        "ai4bharat/MSMARCO-XI",
        language,
        split=split,
        streaming=streaming,
        cache_dir=cache_dir,
    )

    if streaming:
        # Convert iterable dataset to a regular dataset by taking samples
        if sample_size:
            ds = Dataset.from_list(list(ds.take(sample_size)))
        else:
            ds = Dataset.from_list(list(ds))
        return ds

    if sample_size and len(ds) > sample_size:
        ds = ds.shuffle(seed=seed).select(range(sample_size))

    return ds


# ── Passage Extraction ──


def extract_passages(example: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract individual passages from a single MSMARCO-XI example.

    The MSMARCO-XI dataset stores passages as nested structures within
    each example.  This function flattens them into a list of dicts.

    Args:
        example: A single dataset row.

    Returns:
        List of passage dicts, each with ``text``, ``is_selected``, and
        any other available metadata.
    """
    passages = []

    # Handle the nested passages structure
    passage_data = example.get("passages", {})

    if isinstance(passage_data, dict):
        texts = passage_data.get("passage_text", [])
        selected = passage_data.get("is_selected", [])
        urls = passage_data.get("url", [])

        for i in range(len(texts)):
            p = {
                "text": texts[i] if i < len(texts) else "",
                "is_selected": selected[i] if i < len(selected) else 0,
                "url": urls[i] if i < len(urls) else "",
                "passage_index": i,
                "query_id": example.get("query_id", ""),
                "query": example.get("query", ""),
                "query_type": example.get("query_type", ""),
                "language": example.get("language", ""),
            }
            passages.append(p)

    elif isinstance(passage_data, list):
        for i, item in enumerate(passage_data):
            if isinstance(item, dict):
                p = {
                    "text": item.get("passage_text", item.get("text", "")),
                    "is_selected": item.get("is_selected", 0),
                    "url": item.get("url", ""),
                    "passage_index": i,
                    "query_id": example.get("query_id", ""),
                    "query": example.get("query", ""),
                    "query_type": example.get("query_type", ""),
                    "language": example.get("language", ""),
                }
                passages.append(p)

    return passages


def extract_all_passages(dataset: Dataset) -> list[dict[str, Any]]:
    """Extract all passages from a dataset split.

    Args:
        dataset: A loaded MSMARCO-XI dataset.

    Returns:
        Flat list of passage dicts.
    """
    all_passages = []
    for example in dataset:
        all_passages.extend(extract_passages(example))
    return all_passages


def get_selected_passages(dataset: Dataset) -> list[dict[str, Any]]:
    """Extract only passages marked as ``is_selected == 1``.

    These represent the ground-truth relevant passages for each query.
    """
    all_passages = extract_all_passages(dataset)
    return [p for p in all_passages if p.get("is_selected") == 1]


# ── Validation ──


def validate_dataset(dataset: Dataset) -> dict[str, Any]:
    """Run basic validation checks on a loaded dataset.

    Returns:
        Dictionary of validation results.
    """
    results = {
        "num_examples": len(dataset),
        "columns": list(dataset.column_names),
        "features": {k: str(v) for k, v in dataset.features.items()},
    }

    # Check for required fields
    expected_fields = ["query", "passages"]
    results["has_required_fields"] = all(
        f in dataset.column_names for f in expected_fields
    )

    # Sample passage structure check
    if len(dataset) > 0:
        sample = dataset[0]
        passages = sample.get("passages", {})
        if isinstance(passages, dict):
            results["passage_fields"] = list(passages.keys())
            results["num_passages_in_sample"] = len(
                passages.get("passage_text", [])
            )
        elif isinstance(passages, list):
            results["passage_fields"] = (
                list(passages[0].keys()) if passages else []
            )
            results["num_passages_in_sample"] = len(passages)

    # Count examples with selected passages
    selected_count = 0
    total_passages = 0
    for ex in dataset:
        ps = extract_passages(ex)
        total_passages += len(ps)
        selected_count += sum(1 for p in ps if p.get("is_selected") == 1)

    results["total_passages"] = total_passages
    results["selected_passages"] = selected_count
    results["selection_ratio"] = (
        round(selected_count / total_passages, 4) if total_passages > 0 else 0
    )

    return results


def get_dataset_metadata(
    dataset: Dataset, language: str, split: str
) -> dict[str, Any]:
    """Generate dataset metadata for reporting.

    Args:
        dataset: Loaded dataset.
        language: Language code.
        split: Split name.

    Returns:
        Metadata dictionary suitable for saving to JSON.
    """
    validation = validate_dataset(dataset)
    return {
        "dataset_name": "ai4bharat/MSMARCO-XI",
        "language": language,
        "language_name": MSMARCO_XI_LANGUAGES.get(language, language),
        "split": split,
        "num_examples": len(dataset),
        "columns": list(dataset.column_names),
        "features": validation.get("features", {}),
        "passage_fields": validation.get("passage_fields", []),
        "total_passages": validation.get("total_passages", 0),
        "selected_passages": validation.get("selected_passages", 0),
        "selection_ratio": validation.get("selection_ratio", 0),
        "has_required_fields": validation.get("has_required_fields", False),
    }
