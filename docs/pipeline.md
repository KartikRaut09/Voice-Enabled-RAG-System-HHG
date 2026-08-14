# Text RAG Pipeline Architecture & Orchestration Guide

## 1. Overview

The **Text RAG Pipeline** integrates the validated components from Phases 1–6 into a unified, robust, and observable text Question-Answering engine for multilingual Indic queries.

---

## 2. Pipeline Execution Flow

```text
                   USER QUERY
                       │
                       ▼
                QueryProcessor
          (Whitespace norm, script/lang propagation)
                       │
                       ▼
              ┌────────┴────────┐
              │                 │
              ▼                 ▼
        Dense Retrieval       BM25
      (multilingual-e5)  (indic_unicode)
              │                 │
              └────────┬────────┘
                       ▼
                  RRF Fusion
                   (k = 60)
                       │
                       ▼
               Optional Reranker
            (bge-reranker-base toggle)
                       │
                       ▼
                 ContextBuilder
            (Top-5 Parent Deduplication)
                       │
                       ▼
                 LLM Generation
             (Llama-3.1-8B / Groq)
                       │
                       ▼
              Citation Validation
          (Filter invalid IDs e.g. [99])
                       │
                       ▼
              Structured API Response
```

---

## 3. Component Responsibilities

| Stage | Component / Module | Responsibility |
|---|---|---|
| **Query Processing** | `QueryProcessor` (`backend/app/query_processor.py`) | Validates input, normalizes whitespace, preserves original query string, detects Indic Unicode script (`hin_Deva`, `ben_Beng`, `tam_Taml`, `tel_Telu`). |
| **Dense Search** | `SentenceTransformerEmbedder` + `FAISSVectorStore` | Generates 384-dimensional query embeddings and performs cosine nearest-neighbor search. |
| **BM25 Search** | `BM25Index` (`backend/app/bm25.py`) | Inverted index search with Unicode matra/virama-aware tokenization. |
| **Hybrid Fusion** | `reciprocal_rank_fusion` (`backend/app/fusion.py`) | Reciprocal Rank Fusion ($k=60$) combining dense and lexical candidate lists. |
| **Optional Reranking** | `CrossEncoderReranker` (`backend/app/reranking.py`) | Cross-encoder reranking (disabled by default for low latency; enabled for high precision). |
| **Context Construction** | `ContextBuilder` (`backend/app/context.py`) | Collapses chunks by `parent_passage_id`, formats top-5 source blocks with strict 3,000 character budget. |
| **LLM Generation** | `LLMProvider` (`backend/app/generation.py`) | Strict evidence-bounded prompt synthesis with citation references (`[1]`, `[2]`). |
| **Citation Validation** | `extract_and_validate_citations` | Validates citation IDs against retrieved sources; strips hallucinated IDs (e.g. `[99]`). |
| **Orchestration & API** | `RAGPipeline` (`backend/app/pipeline.py`) & FastAPI | Coordinates stages, enforces error isolation, tracks 3-metric latency, and emits structured JSON. |

---

## 4. Failure & Degraded Retrieval Modes

The pipeline includes failure isolation across retrieval channels:

1. **Normal Mode (`hybrid`)**: Both Dense and BM25 channels succeed; candidates are fused with RRF ($k=60$).
2. **Degraded Mode 1 (`dense_only`)**: If BM25 index encounters an error, the pipeline proceeds using Dense candidates alone.
3. **Degraded Mode 2 (`bm25_only`)**: If dense embedding / FAISS index encounters an error, the pipeline proceeds using BM25 candidates alone.
4. **Controlled Error Mode (`failed`)**: If both channels fail simultaneously, the pipeline returns a structured error without crashing or fabricating facts.
5. **Empty Retrieval / Refusal (`insufficient_evidence`)**: When zero relevant evidence is retrieved, the generator explicitly abstains (`"उपलब्ध स्रोतों में इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी नहीं है।"`).

---

## 5. Latency Accounting Definitions

- **`query_processing_ms`**: Query validation and normalization time.
- **`retrieval_ms`**: Dense embedding generation + FAISS search + BM25 search + RRF fusion.
- **`reranking_ms`**: Neural cross-encoder reranking time ($0.00\text{ ms}$ when disabled).
- **`generation_ms`**: LLM context preparation, API call / local synthesis, and citation parsing.
- **`rag_latency_ms`**: Complete text pipeline duration ($\text{query} + \text{retrieval} + \text{reranking} + \text{context} + \text{generation}$).

---

## 6. API Response Contract

```json
{
  "request_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "query": "भारत की राजधानी क्या है?",
  "answer": "भारत की राजधानी नई दिल्ली है। [1]",
  "sources": [
    {
      "passage_text": "भारत एक दक्षिण एशियाई देश है। इसकी राजधानी नई दिल्ली है...",
      "score": 0.0328,
      "metadata": {
        "source_id": 1,
        "parent_passage_id": "10024",
        "chunk_id": "10024_c0",
        "rank": 1,
        "language": "hin_Deva"
      }
    }
  ],
  "latency": {
    "query_processing_ms": 0.05,
    "embedding_ms": 19.08,
    "retrieval_ms": 2.17,
    "reranking_ms": 0.0,
    "generation_ms": 84.50,
    "rag_latency_ms": 106.10,
    "stt_latency_ms": 0.0,
    "e2e_latency_ms": 106.10,
    "total_request_ms": 106.10
  },
  "status": "success",
  "query_metadata": {
    "original_query": "भारत की राजधानी क्या है?",
    "processed_query": "भारत की राजधानी क्या है?",
    "language": "hin_Deva",
    "is_valid": true
  },
  "pipeline_metadata": {
    "retrieval_mode": "hybrid",
    "reranking_enabled": false,
    "model_used": "llama-3.1-8b-instant",
    "provider": "groq"
  }
}
```

---

## 7. Scope Isolation Confirmation

- [x] Zero STT / Audio processing (reserved for Phase 9)
- [x] Zero Voice Input / Output (reserved for Phase 10)
- [x] Zero Advanced Guardrail framework (reserved for Phase 8)
- [x] Zero Model fine-tuning / Google Colab
