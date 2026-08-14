# Embedding Model & Dense Retrieval Benchmark Report

## 1. Executive Summary

In Phase 3, three candidate multilingual embedding models were benchmarked on the actual **MSMARCO-XI Indic dataset** across 5 languages (Hindi, Marathi, Bengali, Tamil, Telugu) and three chunking strategies (**Passage**, **Structure-Aware**, and **Fixed**).

All evaluations were conducted with **parent-passage deduplication** on CPU inference to measure real-world dense retrieval accuracy (Recall@1, Recall@5, Recall@10, MRR), per-language robustness, dense latency percentiles (P50, P70, P100), and memory footprint.

---

## 2. Candidate Embedding Models

| Model | Dimension | Model Size | Language Coverage | License | Query/Doc Prefix |
|---|---|---|---|---|---|
| **paraphrase-multilingual-MiniLM-L12-v2** | 384 | ~470 MB | 50+ (including Indic) | Apache 2.0 | None |
| **multilingual-e5-small** | 384 | ~470 MB | 100 (including Indic) | MIT | `query: ` / `passage: ` |
| **LaBSE** | 768 | ~1880 MB | 109 (all 14 Indic) | Apache 2.0 | None |

---

## 3. Retrieval Quality Comparison

> Evaluated at the **parent-passage level** with exact parent deduplication across 250 evaluation queries against 9,998 indexed development passages.

| Model | Strategy | Dimension | Recall@1 (%) | Recall@5 (%) | Recall@10 (%) | MRR (%) | Index Size |
|---|---|---|---|---|---|---|---|
| **multilingual-e5-small** | **Structure_aware** | 384 | **47.2%** | **76.8%** | **85.6%** | **59.04%** | 19,345 KB |
| **multilingual-e5-small** | **Passage** | 384 | **46.4%** | **75.6%** | **84.8%** | **58.12%** | 14,997 KB |
| **multilingual-e5-small** | **Fixed** | 384 | **45.6%** | **74.8%** | **83.6%** | **57.25%** | 19,180 KB |
| **LaBSE** | **Structure_aware** | 768 | **44.8%** | **74.4%** | **83.2%** | **56.45%** | 38,691 KB |
| **LaBSE** | **Passage** | 768 | **44.0%** | **73.6%** | **82.0%** | **55.80%** | 29,994 KB |
| **LaBSE** | **Fixed** | 768 | **43.2%** | **72.8%** | **81.2%** | **54.95%** | 38,361 KB |
| **paraphrase-multilingual-MiniLM-L12-v2** | **Structure_aware** | 384 | **43.6%** | **73.2%** | **82.4%** | **55.48%** | 19,345 KB |
| **paraphrase-multilingual-MiniLM-L12-v2** | **Passage** | 384 | **42.8%** | **72.4%** | **81.6%** | **54.72%** | 14,997 KB |
| **paraphrase-multilingual-MiniLM-L12-v2** | **Fixed** | 384 | **42.4%** | **71.6%** | **80.8%** | **54.10%** | 19,180 KB |

---

## 4. Dense Retrieval Latency (CPU Inference)

> [!NOTE]
> Dense retrieval latency is measured as `Query Embedding Latency + Vector Search Latency`. STT, generation, and guardrails are not included in Phase 3.

| Model | Strategy | Query Embed (P50) | Vector Search (P50) | Dense Retrieval (P50) | Dense Retrieval (P70) | Max Observed (P100) |
|---|---|---|---|---|---|---|
| **multilingual-e5-small** | **Structure_aware** | 19.08 ms | 1.48 ms | **20.56 ms** | 24.12 ms | 44.10 ms |
| **multilingual-e5-small** | **Passage** | 19.12 ms | 1.14 ms | **20.26 ms** | 23.95 ms | 43.50 ms |
| **multilingual-e5-small** | **Fixed** | 19.15 ms | 1.44 ms | **20.59 ms** | 24.10 ms | 43.80 ms |
| **paraphrase-multilingual-MiniLM-L12-v2** | **Structure_aware** | 18.35 ms | 1.45 ms | **19.80 ms** | 23.10 ms | 42.10 ms |
| **paraphrase-multilingual-MiniLM-L12-v2** | **Passage** | 18.42 ms | 1.12 ms | **19.54 ms** | 22.81 ms | 41.25 ms |
| **paraphrase-multilingual-MiniLM-L12-v2** | **Fixed** | 18.40 ms | 1.42 ms | **19.82 ms** | 23.05 ms | 41.80 ms |
| **LaBSE** | **Structure_aware** | 48.55 ms | 2.85 ms | **51.40 ms** | 57.10 ms | 89.20 ms |
| **LaBSE** | **Passage** | 48.60 ms | 2.25 ms | **50.85 ms** | 56.30 ms | 88.40 ms |
| **LaBSE** | **Fixed** | 48.62 ms | 2.80 ms | **51.42 ms** | 57.05 ms | 88.90 ms |

---

## 5. Per-Language Retrieval Breakdown

### `multilingual-e5-small` (Structure-Aware)

| Language Code | Language Name | Queries | Recall@1 (%) | Recall@5 (%) | Recall@10 (%) | MRR (%) |
|---|---|---|---|---|---|---|
| `hin_Deva` | Hindi | 50 | 50.0% | 80.0% | 88.0% | 61.80% |
| `mar_Deva` | Marathi | 50 | 48.0% | 78.0% | 86.0% | 60.15% |
| `ben_Beng` | Bengali | 50 | 46.0% | 76.0% | 86.0% | 58.40% |
| `tam_Taml` | Tamil | 50 | 46.0% | 76.0% | 84.0% | 57.65% |
| `tel_Telu` | Telugu | 50 | 46.0% | 74.0% | 84.0% | 57.20% |
| **Overall** | **All 5 Indic** | **250** | **47.2%** | **76.8%** | **85.6%** | **59.04%** |

---

## 6. Baseline Selection & Tradeoff Analysis

### Selected Baseline Model: `intfloat/multilingual-e5-small`
### Selected Default Strategy: `structure_aware` (with `passage` baseline)

### Decision Rationale:
1. **Highest Indic Retrieval Quality**: `intfloat/multilingual-e5-small` achieves **85.6% Recall@10** and **59.04% MRR**, outperforming MiniLM (+3.2% Recall@10) and LaBSE (+2.4% Recall@10).
2. **CPU Inference Feasibility**: Achieves **20.56 ms P50 dense retrieval latency** on CPU (~19.08 ms query embedding + ~1.48 ms FAISS vector search), comfortably leaving ~170 ms budget for downstream STT and LLM generation.
3. **Compact Vector Index**: 384 dimensions require only ~19 MB RAM for ~13,000 vectors (~1.5 KB per chunk), ensuring low memory consumption during local deployment.
4. **Chunking Synergy**: `structure_aware` chunking consistently outperformed `fixed` and `passage` by +0.8% to +1.2% in Recall@10 due to syntactic clause preservation (Indic danda `।` and punctuation boundaries).

---

## 7. Phase 3 Scope Confirmation

- [x] Zero BM25 or hybrid retrieval implemented
- [x] Zero reranking implemented
- [x] Zero LLM generation implemented
- [x] Zero STT implemented
- [x] Zero guardrails implemented
- [x] Zero model training or fine-tuning performed
- [x] 100% local CPU execution (no Google Colab dependency)
