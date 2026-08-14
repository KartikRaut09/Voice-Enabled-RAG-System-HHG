# MSMARCO-XI Dataset Analysis Report

## 1. Dataset Identity

- **Dataset Name**: `ai4bharat/MSMARCO-XI`
- **Source**: [ai4bharat/MSMARCO-XI on Hugging Face](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI)
- **Verified Configuration**: Parquet format across 14 Indic languages
- **Verified Splits**: `train` (13 language files, ~10.08M rows) and `validation` (14 language files, ~1.37M rows)
- **Analysis Sample**: 500 total records streamed with seed 42 across 5 representative Indic languages (Hindi, Marathi, Bengali, Tamil, Telugu)
- **Streaming Speed**: 500 records analyzed in 9.35 seconds without full dataset materialization in RAM

## 2. Verified Schema

Programmatically verified schema structure per record:

```text
├── query_id: int64                 (e.g., 1185869)
├── query: string                   (translated query in Indic script: Devanagari, Bengali, Tamil, Telugu)
├── Eng_Query: string               (original English query)
├── Answer: string                  (translated answer)
├── Eng_Answer: string              (original English answer)
├── query_type: string              (e.g., "DESCRIPTION", "NUMERIC", "ENTITY", "PERSON", "LOCATION")
├── source_lang: string             (e.g., "eng_Latn")
├── target_lang: string             (e.g., "hin_Deva", "mar_Deva", "ben_Beng", "tam_Taml", "tel_Telu")
├── meta: dict                      (translation model metadata: model_name, temperature, etc.)
└── passages: dict
    ├── is_selected: list[int64]    (1 = relevant ground truth passage, 0 = distractor passage)
    ├── English_passages: list[str] (original English passage candidate texts)
    └── Translated_passages: list[str] (translated passage candidate texts in Indic script)
```

## 3. Language Coverage

| Language Code | Language Name | Language Family | Script Tag | Verified Files |
|---|---|---|---|---|
| `hin` | Hindi | Indo-Aryan | `hin_Deva` | `train/hintrain.parquet`, `validation/hinval.parquet` |
| `mar` | Marathi | Indo-Aryan | `mar_Deva` | `train/martrain.parquet`, `validation/marval.parquet` |
| `ben` | Bengali | Indo-Aryan | `ben_Beng` | `train/bentrain.parquet`, `validation/benval.parquet` |
| `tam` | Tamil | Dravidian | `tam_Taml` | `train/tamtrain.parquet`, `validation/tamval.parquet` |
| `tel` | Telugu | Dravidian | `tel_Telu` | `validation/telval.parquet` (validation partition) |

*Full 14 languages supported in dataset*: Assamese (`asm`), Bengali (`ben`), Gujarati (`guj`), Hindi (`hin`), Kannada (`kan`), Malayalam (`mal`), Marathi (`mar`), Nepali (`nep`), Odia (`ori`), Punjabi (`pan`), Sanskrit (`san`), Tamil (`tam`), Telugu (`tel`), Urdu (`urd`).

## 4. Query Statistics (N = 500)

- **Character Length**: Min = 15, Max = 182, Mean = 54.3, Median = 51.0
- **Word Length**: Min = 3, Max = 28, Mean = 8.7, Median = 8.0
- **Empty Queries**: 0 (0.0%)
- **Duplicate Queries in Sample**: 0 (0.0%)

## 5. Answer Statistics

- **Character Length**: Min = 17, Max = 1642, Mean = 305.8, Median = 242.0
- **Word Length**: Min = 3, Max = 221, Mean = 45.9, Median = 37.0
- **Empty Answers**: 0 (0.0%)

## 6. Passage & Relevance Statistics

- **Total Passages Analyzed**: 4,999 (average 10.0 passages per query)
- **Passage Character Length**: Min = 9, Max = 2,724, Mean = 401.7, Median = 353.0
- **Passage Word Length**: Min = 1, Max = 387, Mean = 60.1, Median = 53.0
- **Selected (Relevant) Passages**: 526
- **Non-Selected (Distractor) Passages**: 4,473
- **Relevance Ratio**: **10.52%** of candidate passages are marked `is_selected=1` (~1.05 relevant passages per query)

## 7. Query Type Distribution

- **DESCRIPTION**: 361 queries (72.2%)
- **NUMERIC**: 88 queries (17.6%)
- **ENTITY**: 27 queries (5.4%)
- **PERSON**: 15 queries (3.0%)
- **LOCATION**: 9 queries (1.8%)

## 8. Development & Evaluation Sampling Strategy

- **Streaming Ingestion**: Uses Hugging Face parquet streaming directly from the remote repository.
- **Stratified Partitioning**: Balanced sample sizes per target language across language families.
- **Strict Separation**: 
  - `dev.jsonl` extracted from `train` files.
  - `evaluation.jsonl` extracted from `validation` files with disjoint `query_id`s.
- **Ground Truth Preservation**: The `is_selected` label is preserved for every passage to enable Recall@K and MRR evaluation in later phases.

## 9. Storage Considerations

- **Full Dataset Size**: ~55.6 GB compressed / 146.6 GB uncompressed (11.45M total rows).
- **RAM Protection**: Never loaded into memory all at once.
- **Local Artifacts**: Deterministic, compact JSONL development (`dev.jsonl`) and evaluation (`evaluation.jsonl`) subsets stored locally in `data/processed/`.

## 10. Phase 1 Conclusions for Subsequent Phases

1. **Passage Granularity**: Each query has ~10 pre-segmented passages (~50-80 words each). In Phase 2, passage-level chunking will serve as an essential baseline strategy.
2. **Bilingual Retrieval Opportunity**: Both English and translated Indic text exist for every passage, enabling cross-lingual and native multilingual dense retrieval benchmarks in Phase 3 & 4.
3. **Relevance Benchmarking**: Ground truth `is_selected` flags enable exact calculation of Recall@1, Recall@5, Recall@10, and MRR.