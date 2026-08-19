"""
Dataset utilities for MSMARCO-XI.
Handles direct parquet loading from ai4bharat/MSMARCO-XI, language mapping,
sampling, robust passage extraction, and schema validation.
"""

from __future__ import annotations

from typing import Any
from datasets import load_dataset, Dataset


# ── Language Code Mapping ──

LANG_TO_PARQUET = {
    "hi": {"name": "Hindi", "val": "validation/hinval.parquet", "train": "train/hintrain.parquet"},
    "hin": {"name": "Hindi", "val": "validation/hinval.parquet", "train": "train/hintrain.parquet"},
    "mr": {"name": "Marathi", "val": "validation/marval.parquet", "train": "train/martrain.parquet"},
    "mar": {"name": "Marathi", "val": "validation/marval.parquet", "train": "train/martrain.parquet"},
    "bn": {"name": "Bengali", "val": "validation/benval.parquet", "train": "train/bentrain.parquet"},
    "ben": {"name": "Bengali", "val": "validation/benval.parquet", "train": "train/bentrain.parquet"},
    "ta": {"name": "Tamil", "val": "validation/tamval.parquet", "train": "train/tamtrain.parquet"},
    "tam": {"name": "Tamil", "val": "validation/tamval.parquet", "train": "train/tamtrain.parquet"},
    "te": {"name": "Telugu", "val": "validation/telval.parquet", "train": "validation/telval.parquet"},
    "tel": {"name": "Telugu", "val": "validation/telval.parquet", "train": "validation/telval.parquet"},
}

MSMARCO_XI_LANGUAGES = {
    "hi": "Hindi",
    "mr": "Marathi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "ar": "Arabic",
    "de": "German",
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "nl": "Dutch",
    "pt": "Portuguese",
    "ru": "Russian",
    "vi": "Vietnamese",
    "zh": "Chinese",
}

INDIC_LANGUAGES = ["hi", "mr", "bn", "ta", "te"]


def list_available_configs(dataset_name: str = "ai4bharat/MSMARCO-XI") -> list[str]:
    """List available dataset language configurations."""
    return list(LANG_TO_PARQUET.keys())


def load_msmarco_xi(
    language: str = "hi",
    split: str = "validation",
    sample_size: int | None = 5000,
    streaming: bool = False,
    cache_dir: str | None = None,
    seed: int = 42,
) -> Dataset:
    """Load an MSMARCO-XI language split directly from Hugging Face Parquet.

    Args:
        language: Language code (e.g. ``"hi"``, ``"mr"``, ``"bn"``, ``"ta"``, ``"te"``).
        split: Dataset split (``"validation"`` or ``"train"``).
        sample_size: Number of examples to load. ``None`` = load all.
        streaming: If True, uses streaming mode.
        cache_dir: Custom HF cache directory.
        seed: Random seed for reproducible sampling.

    Returns:
        A Hugging Face ``Dataset`` object.
    """
    lang_key = language.lower().strip()
    if lang_key not in LANG_TO_PARQUET:
        raise ValueError(
            f"Unsupported language code '{language}'. Supported Indic codes: {list(LANG_TO_PARQUET.keys())}"
        )

    file_subpath = (
        LANG_TO_PARQUET[lang_key]["val"]
        if split.startswith("val")
        else LANG_TO_PARQUET[lang_key]["train"]
    )
    hf_parquet_url = (
        f"https://huggingface.co/datasets/ai4bharat/MSMARCO-XI/resolve/main/{file_subpath}"
    )

    if streaming:
        stream = load_dataset(
            "parquet",
            data_files={split: hf_parquet_url},
            streaming=True,
            cache_dir=cache_dir,
        )
        data_iter = stream[split]
        if sample_size:
            sampled_list = []
            for item in data_iter:
                sampled_list.append(item)
                if len(sampled_list) >= sample_size:
                    break
            return Dataset.from_list(sampled_list)
        else:
            return Dataset.from_list(list(data_iter))

    # Standard non-streaming download and load
    ds_dict = load_dataset(
        "parquet",
        data_files={split: hf_parquet_url},
        cache_dir=cache_dir,
    )
    ds = ds_dict[split]

    if sample_size and len(ds) > sample_size:
        ds = ds.shuffle(seed=seed).select(range(sample_size))

    return ds


# ── Robust Passage Extraction ──


def extract_passages(example: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract individual passages from a single MSMARCO-XI example.

    Supports both MSMARCO-XI schemas:
    - Native MSMARCO-XI: `passages.Translated_passages`, `passages.English_passages`, `passages.is_selected`
    - Generic/Normalized: `passages.passage_text`, `passages.text`

    Args:
        example: A single dataset row.

    Returns:
        List of passage dicts, each with ``text``, ``is_selected``, and metadata.
    """
    passages = []
    passage_data = example.get("passages") or {}

    query_id = str(example.get("query_id", "") or "")
    query_text = str(example.get("query", "") or "")
    query_type = str(example.get("query_type", "") or "")
    target_lang = str(example.get("target_lang", example.get("language", "")) or "")

    if isinstance(passage_data, dict):
        # Primary MSMARCO-XI schema: Translated_passages
        trans_texts = (
            passage_data.get("Translated_passages")
            or passage_data.get("passage_text")
            or passage_data.get("text")
            or []
        )
        eng_texts = passage_data.get("English_passages") or []
        selected_flags = passage_data.get("is_selected") or []
        urls = passage_data.get("url") or []

        num_passages = max(len(trans_texts), len(eng_texts), len(selected_flags))

        for i in range(num_passages):
            t_text = str(trans_texts[i] if i < len(trans_texts) else "").strip()
            e_text = str(eng_texts[i] if i < len(eng_texts) else "").strip()
            # If translated text is missing, fall back to English text
            main_text = t_text if t_text else e_text
            is_sel = int(selected_flags[i]) if i < len(selected_flags) else 0
            url_str = str(urls[i]) if i < len(urls) else ""

            if main_text:
                passages.append(
                    {
                        "passage_id": f"{query_id}_p{i}",
                        "text": main_text,
                        "translated_text": t_text,
                        "english_text": e_text,
                        "is_selected": is_sel,
                        "url": url_str,
                        "passage_index": i,
                        "query_id": query_id,
                        "query": query_text,
                        "query_type": query_type,
                        "language": target_lang,
                    }
                )

    elif isinstance(passage_data, list):
        for i, item in enumerate(passage_data):
            if isinstance(item, dict):
                t_text = str(
                    item.get("Translated_passages")
                    or item.get("passage_text")
                    or item.get("text", "")
                ).strip()
                is_sel = int(item.get("is_selected", 0))
                if t_text:
                    passages.append(
                        {
                            "passage_id": f"{query_id}_p{i}",
                            "text": t_text,
                            "is_selected": is_sel,
                            "url": str(item.get("url", "")),
                            "passage_index": i,
                            "query_id": query_id,
                            "query": query_text,
                            "query_type": query_type,
                            "language": target_lang,
                        }
                    )

    return passages


def extract_all_passages(dataset: Dataset | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract all passages from a dataset split."""
    all_passages = []
    for example in dataset:
        all_passages.extend(extract_passages(example))
    return all_passages


def get_selected_passages(dataset: Dataset | list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract only passages marked as ``is_selected == 1``."""
    all_passages = extract_all_passages(dataset)
    return [p for p in all_passages if p.get("is_selected") == 1]


# ── Validation ──


def validate_dataset(dataset: Dataset | list[dict[str, Any]]) -> dict[str, Any]:
    """Run validation checks on a loaded dataset."""
    num_examples = len(dataset)
    columns = list(dataset[0].keys()) if num_examples > 0 else []

    selected_count = 0
    total_passages = 0
    sample_passage_fields = []

    for idx, ex in enumerate(dataset):
        ps = extract_passages(ex)
        total_passages += len(ps)
        selected_count += sum(1 for p in ps if p.get("is_selected") == 1)
        if idx == 0 and ex.get("passages"):
            raw_p = ex.get("passages")
            sample_passage_fields = list(raw_p.keys()) if isinstance(raw_p, dict) else []

    return {
        "num_examples": num_examples,
        "columns": columns,
        "passage_fields": sample_passage_fields,
        "total_passages": total_passages,
        "selected_passages": selected_count,
        "selection_ratio": (selected_count / total_passages) if total_passages > 0 else 0.0,
        "has_required_fields": "query" in columns and "passages" in columns,
    }


def get_dataset_metadata(
    dataset: Dataset, language: str, split: str
) -> dict[str, Any]:
    """Generate dataset metadata for reporting."""
    validation = validate_dataset(dataset)
    return {
        "dataset_name": "ai4bharat/MSMARCO-XI",
        "language": language,
        "language_name": MSMARCO_XI_LANGUAGES.get(language, language),
        "split": split,
        "num_examples": len(dataset),
        "columns": validation.get("columns", []),
        "passage_fields": validation.get("passage_fields", []),
        "total_passages": validation.get("total_passages", 0),
        "selected_passages": validation.get("selected_passages", 0),
        "selection_ratio": validation.get("selection_ratio", 0.0),
        "has_required_fields": validation.get("has_required_fields", False),
    }
