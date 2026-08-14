# BM25 Lexical Retrieval Benchmark & Complementarity Report

## 1. Executive Summary

In Phase 4, an independent **Okapi BM25 lexical retrieval baseline** with multilingual Unicode tokenization was implemented, benchmarked on the 250 evaluation queries across 5 Indic languages (Hindi, Marathi, Bengali, Tamil, Telugu), and comparatively evaluated against the Phase 3 dense retrieval baseline (`intfloat/multilingual-e5-small` + `structure_aware`).

All retrieval metrics are computed at the **parent-passage level** with exact parent deduplication to ensure 1:1 mathematical parity with Phase 3.

---

## 2. BM25 Configuration Benchmark

| Strategy | k1 | b | Chunks | Vocabulary | Build Time | Recall@1 (%) | Recall@5 (%) | Recall@10 (%) | MRR (%) | Latency (P50) |
|---|---|---|---|---|---|---|---|---|---|---|
| **Structure_aware** | **1.5** | **0.75** | 12,897 | 37,284 | 0.165s | **34.0%** | **58.4%** | **68.8%** | **44.15%** | **0.62 ms** |
| **Passage** | 1.5 | 0.75 | 9,998 | 33,142 | 0.130s | **33.2%** | **57.6%** | **67.6%** | **43.28%** | **0.58 ms** |
| **Structure_aware** | 1.2 | 0.75 | 12,897 | 37,284 | 0.165s | **33.6%** | **58.0%** | **68.4%** | **43.88%** | **0.61 ms** |
| **Structure_aware** | 1.5 | 0.50 | 12,897 | 37,284 | 0.161s | **33.6%** | **58.4%** | **68.8%** | **44.02%** | **0.62 ms** |

---

## 3. Dense vs BM25 Baseline Comparison

| System | Chunking Strategy | Recall@1 (%) | Recall@5 (%) | Recall@10 (%) | MRR (%) | Latency (P50) | Memory / Size |
|---|---|---|---|---|---|---|---|
| **E5-small (Dense Baseline - Phase 3 Reference)** | **Structure_aware** | **47.2%** | **76.8%** | **85.6%** | **59.04%** | 20.56 ms | ~470 MB (Model) + 19 MB (FAISS) |
| **BM25 (Lexical Baseline - Phase 4)** | **Structure_aware** | **34.0%** | **58.4%** | **68.8%** | **44.15%** | **0.62 ms** | ~1,840 KB (Inverted Index) |
| **BM25 (Lexical Baseline - Phase 4)** | **Passage** | **33.2%** | **57.6%** | **67.6%** | **43.28%** | **0.58 ms** | ~1,420 KB (Inverted Index) |

---

## 4. Per-Language Retrieval Breakdown for BM25 (`structure_aware`, k1=1.5, b=0.75)

| Language Code | Language Name | Queries | Recall@1 (%) | Recall@5 (%) | Recall@10 (%) | MRR (%) |
|---|---|---|---|---|---|---|
| `hin_Deva` | Hindi | 50 | 36.0% | 62.0% | 72.0% | 47.12% |
| `mar_Deva` | Marathi | 50 | 34.0% | 58.0% | 70.0% | 44.80% |
| `ben_Beng` | Bengali | 50 | 34.0% | 58.0% | 68.0% | 43.95% |
| `tam_Taml` | Tamil | 50 | 32.0% | 56.0% | 66.0% | 42.10% |
| `tel_Telu` | Telugu | 50 | 34.0% | 58.0% | 68.0% | 42.78% |
| **Overall** | **All 5 Indic** | **250** | **34.0%** | **58.4%** | **68.8%** | **44.15%** |

---

## 5. Dense-Lexical Complementarity Analysis

To determine whether lexical retrieval provides distinct, non-redundant signal for Phase 5 hybrid retrieval, individual query outcomes were mapped between **Dense (E5-small)** and **BM25** (evaluated at Top-10):

| Outcome Category | Queries Count | Percentage (%) | Interpretation |
|---|---|---|---|
| **Both Succeed** | 156 | 62.4% | High-confidence overlap across both semantic and lexical channels. |
| **Dense Only** | 58 | 23.2% | Semantic abstraction matches conceptual queries without exact token match. |
| **BM25 Only** | 16 | 6.4% | Exact keyword, entity name, and numeric matches that dense embedding missed. |
| **Neither** | 20 | 8.0% | Difficult or under-specified queries. |

### Theoretical Maximum Hybrid Upper Bound:
- Combining Dense + BM25 provides a theoretical Recall@10 ceiling of **92.0%** (an absolute gain of **+6.4%** over Dense alone).
- **Conclusion**: BM25 uniquely recovers **16 queries (6.4%)** where dense embedding failed, proving that hybrid fusion in Phase 5 is mathematically justified.

---

## 6. Lexical Retrieval Latency (CPU)

| Metric | Latency |
|---|---|
| **Index Build Time (12,897 chunks)** | 0.165 seconds (~78,163 chunks/sec) |
| **Query Tokenization (P50)** | <0.05 ms |
| **BM25 Search (P50)** | 0.62 ms |
| **BM25 Search (P70)** | 0.81 ms |
| **BM25 Search (P100 / Max)** | 2.45 ms |

---

## 7. Phase 4 Scope Confirmation

- [x] Zero Dense + BM25 fusion implemented
- [x] Zero Reciprocal Rank Fusion (RRF) implemented
- [x] Zero weighted combination implemented
- [x] Zero reranking / cross-encoders implemented
- [x] Zero LLM generation implemented
- [x] Zero STT implemented
- [x] Zero guardrails implemented
- [x] 100% CPU execution (no Google Colab)
