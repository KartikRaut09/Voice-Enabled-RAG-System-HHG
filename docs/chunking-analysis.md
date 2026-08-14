# Multi-Strategy Chunking Analysis Report

## 1. Executive Summary

In accordance with HH Goa Task 2 requirements and Phase 1 findings, four distinct chunking strategies were implemented, executed on the 1,000-record MSMARCO-XI development dataset (9,998 total input passages across 5 Indic languages), and rigorously benchmarked for boundary integrity, corpus expansion, and relevance preservation.

### Data-Driven Threshold Selection
Phase 1 revealed that the average passage in MSMARCO-XI is **60.1 words** (median **53.0 words**, min **1 word**, max **387 words**). Consequently, a fixed threshold of `max_words = 80` was established:
- **~75–80% of passages** are naturally concise and stay intact without arbitrary fragmentation.
- Only genuinely long, multi-sentence passages (>80 words) undergo splitting.

---

## 2. Comparative Benchmark Results

| Strategy | Passages | Chunks | Chunks/Passage | Mean Words | Median Words | P95 Words | Split % | Duplication | Throughput (pass/s) |
|---|---|---|---|---|---|---|---|---|---|
| **Passage** | 9,998 | 9,998 | 1.0x | 60.1 | 53.0 | 129.0 | 0.0% | 1.0x | 53,325 |
| **Fixed** | 9,998 | 12,787 | 1.28x | 47.0 | 48.0 | 80.0 | 21.57% | 1.0x | 49,455 |
| **Overlap** | 9,998 | 13,716 | 1.37x | 48.1 | 50.0 | 80.0 | 21.57% | 1.096x | 46,193 |
| **Structure_aware** | 9,998 | 12,897 | 1.29x | 46.6 | 47.0 | 80.0 | 21.57% | 1.0x | 37,722 |


---

## 3. Strategy-by-Strategy Analysis

### 1. Passage-Level Baseline (`passage`)
- **Corpus Expansion**: 1.0x (9,998 chunks produced from 9,998 passages).
- **Split Ratio**: 0.0% split. Every passage is preserved as a cohesive contextual unit.
- **Pros**: Zero corpus duplication, retains complete author context, lowest storage and indexing overhead.
- **Cons**: For the top 5% longest passages (>150 words), may dilute dense embedding representation.

### 2. Fixed Token-Bounded Chunking (`fixed`)
- **Corpus Expansion**: ~1.28x (12,787 chunks).
- **Split Ratio**: 21.6% of passages split into consecutive non-overlapping 80-word windows.
- **Pros**: Strict upper-bound on sequence length; fits cleanly into fixed-token embedding models (e.g. 128/256 token windows).
- **Cons**: Unaware of sentence or punctuation boundaries, which can occasionally truncate compound Indic clauses.

### 3. Overlapping Chunking (`overlap`)
- **Corpus Expansion**: ~1.37x (13,716 chunks).
- **Split Ratio**: 21.6% of passages split with a 20-word sliding overlap (`max_words=80`, `overlap_words=20`).
- **Duplication Rate**: ~1.09x total word duplication.
- **Pros**: Eliminates boundary truncation effects across chunks for long passages.
- **Cons**: Increases vector index size by ~37% and creates near-duplicate candidate retrievals that require downstream deduplication/reranking.

### 4. Structure-Aware Chunking (`structure_aware`)
- **Corpus Expansion**: ~1.29x (12,897 chunks).
- **Split Ratio**: 21.6% of passages split honoring paragraph breaks and Indic sentence terminators (`।`, `.`, `?`, `!`, `;`).
- **Pros**: Preserves complete syntactic and semantic clauses without cutting mid-sentence. High boundary coherence for LLM context injection.
- **Cons**: Slightly more variable chunk lengths depending on individual sentence sizes.

---

## 4. Relevance Label (`is_selected`) Preservation

Across all four strategies, ground-truth relevance labels were preserved with 100% fidelity:
- **Baseline Relevant Chunks**: 1,051 selected chunks (~10.51% relevance ratio).
- When a relevant passage is split into multiple chunks, all derived sub-chunks inherit `is_selected = True`. This guarantees that downstream Recall@K and MRR evaluation in Phase 4 & 5 remain mathematically valid.

---

## 5. Data Integrity & Verification Checklist

- [x] **No Query Text Contamination**: Chunk text contains strictly the passage text (no query text prepended).
- [x] **Zero Empty Chunks**: No empty or whitespace-only chunks generated across all strategies.
- [x] **Zero Duplicate Chunk IDs**: Every chunk has a deterministic unique ID (`{query_id}_p{idx}_c{chunk_index}`).
- [x] **Strict Traceability**: All chunks retain `parent_passage_id`, `query_id`, `language`, `source_lang`, `target_lang`, and `query_type`.
- [x] **Deterministic Progression**: Identical input + seed produces identical chunk outputs.

---

## 6. Recommendations for Phase 3 & Phase 4

Rather than arbitrarily locking into a single chunker before retrieval benchmarks exist:
1. **Passage-Level (`passage`)** and **Structure-Aware (`structure_aware`)** should proceed as primary candidates to Phase 3/4 embedding and retrieval benchmarking.
2. **Fixed (`fixed`)** and **Overlap (`overlap`)** will serve as comparative benchmarks to measure whether the 28%–37% corpus expansion yields a statistically meaningful improvement in Recall@5 / MRR on Indic languages.
