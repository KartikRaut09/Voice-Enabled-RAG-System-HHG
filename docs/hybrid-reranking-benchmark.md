# Hybrid Retrieval Fusion & Neural Reranking Benchmark Report

## 1. Executive Summary

In Phase 5, **Hybrid Retrieval (Dense + BM25)** and **Multilingual Neural Reranking** were implemented and benchmarked on the 250 evaluation queries across 5 Indic languages (Hindi, Marathi, Bengali, Tamil, Telugu) against 12,897 indexed `structure_aware` development chunks.

The benchmark comparatively evaluates:
1. **Dense Retrieval alone** (`intfloat/multilingual-e5-small` + `structure_aware`)
2. **BM25 Lexical Retrieval alone** (`BM25Index` + `indic_unicode`)
3. **Reciprocal Rank Fusion (RRF)** ($k=20, 60, 100$)
4. **Normalized Weighted Score Fusion** ($\alpha = 0.5, 0.7, 0.8, 0.9$)
5. **Neural Cross-Encoder Reranking** (`BAAI/bge-reranker-base` over candidate pool)

All metrics are evaluated at the **parent-passage level** with exact parent deduplication.

---

## 2. Comprehensive Retrieval Quality & Latency Matrix

| System / Configuration | Strategy / Parameters | Recall@1 (%) | Recall@5 (%) | Recall@10 (%) | MRR (%) | Latency P50 (ms) | Latency P70 (ms) | Max Latency (ms) |
|---|---|---|---|---|---|---|---|---|
| **Dense Baseline (E5-small)** | `k=10` | 47.2% | 76.8% | 85.6% | 59.04% | 20.56 ms | 24.12 ms | 44.10 ms |
| **BM25 Baseline** | `k=10` | 34.0% | 58.4% | 68.8% | 44.15% | 0.62 ms | 0.81 ms | 2.45 ms |
| **Hybrid RRF (k=20)** | `k=20, cand_k=50` | 49.6% | 79.6% | 89.2% | 62.40% | 21.22 ms | 24.90 ms | 45.80 ms |
| **Hybrid RRF (k=60)** | `k=60, cand_k=50` | **50.8%** | **80.4%** | **90.0%** | **63.85%** | **21.25 ms** | **24.95 ms** | **45.90 ms** |
| **Hybrid RRF (k=100)** | `k=100, cand_k=50` | 50.4% | 80.0% | 89.6% | 63.15% | 21.28 ms | 25.00 ms | 46.10 ms |
| **Hybrid Weighted (α=0.5)** | `dense=0.5, bm25=0.5` | 48.0% | 78.4% | 88.0% | 60.85% | 21.20 ms | 24.85 ms | 45.70 ms |
| **Hybrid Weighted (α=0.7)** | `dense=0.7, bm25=0.3` | 49.6% | 79.6% | 89.2% | 62.60% | 21.22 ms | 24.88 ms | 45.75 ms |
| **Hybrid Weighted (α=0.8)** | `dense=0.8, bm25=0.2` | 50.0% | 80.0% | 89.6% | 63.10% | 21.24 ms | 24.90 ms | 45.80 ms |
| **Hybrid Weighted (α=0.9)** | `dense=0.9, bm25=0.1` | 48.8% | 78.8% | 87.6% | 61.20% | 21.22 ms | 24.86 ms | 45.72 ms |
| **Hybrid RRF + Reranker (Top-10)** | `rrf_k=60, rerank_k=10` | 56.8% | 83.6% | 90.0% | 68.90% | 64.50 ms | 76.20 ms | 148.20 ms |
| **Hybrid RRF + Reranker (Top-20)** | `rrf_k=60, rerank_k=20` | **58.4%** | **84.8%** | **90.8%** | **70.24%** | **108.40 ms** | **126.80 ms** | **235.40 ms** |

---

## 3. Component Latency Breakdown (CPU Inference)

| Configuration | Dense P50 | BM25 P50 | Fusion P50 | Rerank P50 | Aggregation P50 | Total P50 |
|---|---|---|---|---|---|---|
| **Dense Baseline (E5-small)** | 19.08 ms | 0.00 ms | 0.00 ms | 0.00 ms | 1.48 ms | **20.56 ms** |
| **BM25 Baseline** | 0.00 ms | 0.58 ms | 0.00 ms | 0.00 ms | 0.04 ms | **0.62 ms** |
| **Hybrid RRF (k=60)** | 19.08 ms | 0.58 ms | 0.12 ms | 0.00 ms | 1.47 ms | **21.25 ms** |
| **Hybrid Weighted (α=0.8)** | 19.08 ms | 0.58 ms | 0.11 ms | 0.00 ms | 1.47 ms | **21.24 ms** |
| **Hybrid RRF + Reranker (Top-10)** | 19.08 ms | 0.58 ms | 0.12 ms | 43.25 ms | 1.47 ms | **64.50 ms** |
| **Hybrid RRF + Reranker (Top-20)** | 19.08 ms | 0.58 ms | 0.12 ms | 87.15 ms | 1.47 ms | **108.40 ms** |

---

## 4. Per-Language Retrieval Breakdown

### Best Hybrid (RRF k=60) vs Hybrid + Reranker (`BAAI/bge-reranker-base`)

| Language Code | Language Name | Queries | Dense Baseline R@10 | BM25 Baseline R@10 | Hybrid (RRF k=60) R@10 | Hybrid + Rerank R@10 | Hybrid MRR | Hybrid+Rerank MRR |
|---|---|---|---|---|---|---|---|---|
| `hin_Deva` | Hindi | 50 | 88.0% | 72.0% | **92.0%** | **94.0%** | 66.40% | **73.15%** |
| `mar_Deva` | Marathi | 50 | 86.0% | 70.0% | **90.0%** | **92.0%** | 64.80% | **71.20%** |
| `ben_Beng` | Bengali | 50 | 86.0% | 68.0% | **90.0%** | **90.0%** | 63.90% | **69.80%** |
| `tam_Taml` | Tamil | 50 | 84.0% | 66.0% | **88.0%** | **88.0%** | 61.50% | **68.25%** |
| `tel_Telu` | Telugu | 50 | 84.0% | 68.0% | **90.0%** | **90.0%** | 62.65% | **68.80%** |
| **Overall** | **All 5 Indic** | **250** | **85.6%** | **68.8%** | **90.0%** | **90.8%** | **63.85%** | **70.24%** |

---

## 5. Reranker Delta: Quality Improvement vs Latency Tradeoff

| Metric | Hybrid without Reranker (RRF k=60) | Hybrid with Reranker (bge-reranker-base) | Incremental Delta (Δ) |
|---|---|---|---|
| **Recall@1** | 50.8% | **58.4%** | **+7.6%** |
| **Recall@5** | 80.4% | **84.8%** | **+4.4%** |
| **Recall@10** | **90.0%** | **90.8%** | **+0.8%** |
| **MRR** | 63.85% | **70.24%** | **+6.39%** |
| **P50 Latency** | **21.25 ms** | 108.40 ms | +87.15 ms |
| **P70 Latency** | **24.95 ms** | 126.80 ms | +101.85 ms |

> [!IMPORTANT]
> **Key Architectural Takeaways**:
> 1. **Hybrid RRF (k=60)** is the undisputed efficiency champion: It boosts Recall@10 from **85.6% -> 90.0% (+4.4%)** and MRR from **59.04% -> 63.85% (+4.81%)** for only **+0.69 ms of latency overhead** (P50 = **21.25 ms**).
> 2. **Neural Cross-Encoder Reranking** provides massive precision in top ranks (**+7.6% Recall@1**, **+6.39% MRR**), placing the gold passage at rank 1 in 58.4% of all Indic queries.
> 3. However, on CPU inference, cross-encoding 20 candidates adds **~87 ms**.
> 4. **Production Recommendation**: Use **Hybrid RRF (k=60)** as the default retrieval pipeline to maintain ultra-fast ~21 ms latency (preserving ~175 ms headroom for STT and LLM generation), with neural reranking available as a configurable toggle when sub-100 ms RAG budgets are available.

---

## 6. Failure & Complementarity Analysis

| Sample Query ID | Language | Query Snippet | Dense Result | BM25 Result | Hybrid Result | Root Cause Category |
|---|---|---|---|---|---|---|
| `1102432` | Hindi | "कॉर्पोरेशन क्या है?" | PASS (Rank 1) | PASS (Rank 2) | **PASS (Rank 1)** | Definition / Semantic match |
| `1102431` | Hindi | "रेचल कार्सन ने क्यों एक दायित्व बर्दाश्त करने के लिए लिखा" | FAIL (Rank 14) | PASS (Rank 1) | **PASS (Rank 1)** | Exact Named Entity ('रेचल कार्सन') |
| `90836` | Hindi | "पोटेशियम में कम खाद्य पदार्थों का चार्ट।" | PASS (Rank 3) | PASS (Rank 1) | **PASS (Rank 1)** | Keyword Overlap |
| `55665` | Marathi | "मालवाहू जहाजाचा पुढचा खालचा भाग" | PASS (Rank 2) | FAIL (Rank 18) | **PASS (Rank 2)** | Semantic Paraphrase |
| `104231` | Bengali | "পিসিতে গুগল ক্রোম কীভাবে আপডেট করবেন" | FAIL (Rank 12) | PASS (Rank 2) | **PASS (Rank 2)** | Technical Term / Transliteration |
| `204561` | Tamil | "இரத்த அழுத்தத்தை குறைக்கும் உணவுகள்" | PASS (Rank 1) | PASS (Rank 4) | **PASS (Rank 1)** | Health domain query |

---

## 7. Selected Production Retrieval Baseline

```yaml
retrieval:
  pipeline: "hybrid_rrf"
  embedding:
    model: "intfloat/multilingual-e5-small"
    dimension: 384
    chunk_strategy: "structure_aware"
  bm25:
    k1: 1.5
    b: 0.75
    tokenizer: "indic_unicode"
  hybrid:
    fusion: "rrf"
    rrf_k: 60
    candidate_k: 50
    parent_aggregation: "max"
  reranking:
    enabled: false # Default false for ultra-low 21ms latency; toggleable to true for 70.24% MRR
    model: "BAAI/bge-reranker-base"
    candidate_k: 20
```

---

## 8. Phase 5 Scope Confirmation

- [x] Zero LLM generation implemented
- [x] Zero STT implemented
- [x] Zero guardrails implemented
- [x] Zero model training/fine-tuning
- [x] 100% CPU execution
