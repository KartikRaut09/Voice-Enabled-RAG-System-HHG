# HH Goa Task 2 — Colab Research Pipeline

A streamlined, reproducible Google Colab research and development notebook for building, benchmarking, and exporting the Voice-Enabled RAG system described in the HH Goa 2026 Task 2 specification.

## Purpose

This directory contains the complete end-to-end research, benchmarking, vector indexing, and artifact export infrastructure in a single unified notebook:

📁 `colab/notebooks/voice_rag_pipeline_research.ipynb`

It covers all required phases:
1. **Environment Setup & Hardware Diagnostics** (GPU, CUDA, PyTorch, SEED=42)
2. **Dataset Loading & Extraction** (`ai4bharat/MSMARCO-XI` for Indic languages)
3. **Exploratory Data Analysis** (Query/Passage distributions, relevance statistics)
4. **Chunking Strategy Benchmark** (Original, Fixed, Sentence-Aware, Metadata-Aware, Semantic)
5. **Multilingual Embedding Model Benchmark** (`intfloat/multilingual-e5-small`, `BAAI/bge-m3`, etc.)
6. **Hybrid Retrieval Benchmark** (FAISS IndexFlatIP Dense + BM25 Lexical + Reciprocal Rank Fusion, Recall@1/5/10, MRR)
7. **Latency Benchmark & SLA Verification** (< 200 ms target, P50, P70, P90, P100 breakdown)
8. **Production Vector Index Build** (`faiss_index.bin` + `metadata.pkl`)
9. **Artifact Export & Backend Integration** (`manifest.json` generation and FastAPI loading)

---

## How to Use in Google Colab

### 1. Clone the repository in Colab

```python
!git clone <YOUR-REPOSITORY-URL>
%cd <REPOSITORY-NAME>
```

### 2. Install dependencies

```python
!pip install -r colab/requirements-colab.txt -q
```

### 3. Open and Run the Notebook

Open `colab/notebooks/voice_rag_pipeline_research.ipynb` and run cells sequentially from top to bottom. Every code cell includes a dedicated markdown cell detailing:
- **What this cell does**
- **Why we need it**
- **Expected output**

---

## Required Secrets

API keys are managed securely via Colab Secrets (`userdata.get`) or environment variables. **Never hard-code keys.**

| Secret | Required For | How to Set |
|---|---|---|
| `HF_TOKEN` | Hugging Face dataset access (if gated) | Colab Secrets or `os.environ` |
| `SARVAM_API_KEY` | Sarvam STT API calls (`saaras:v3`) | Colab Secrets or `os.environ` |
| `GROQ_API_KEY` | Groq LLM generation (`llama-3.1-8b-instant`) | Colab Secrets or `os.environ` |

---

## Directory Structure

```
colab/
├── README.md                                  ← Documentation & usage guide
├── requirements-colab.txt                     ← Pip dependencies for Colab
├── notebooks/
│   └── voice_rag_pipeline_research.ipynb      ← Consolidated End-to-End Notebook
├── src/                                       ← Reusable Python modules
│   ├── __init__.py
│   ├── dataset_utils.py                       ← Dataset loading & passage flattening
│   ├── chunking.py                            ← 5 Chunking strategies & comparison
│   ├── embeddings.py                          ← Multilingual bi-encoder wrapper & benchmarks
│   ├── retrieval.py                           ← FAISS vector store, BM25, and RRF fusion
│   ├── evaluation.py                          ← Recall@K, MRR & evaluation harness
│   ├── latency.py                             ← P50/P70/P100 timing instrumentation
│   └── utils.py                               ← Path discovery, configs, seeding & secrets
├── configs/
│   └── experiment_config.yaml                 ← Central experiment configuration
├── data/                                      ← Local data cache
├── artifacts/                                 ← Exported vector index & metadata
└── reports/                                   ← JSON experiment reports & benchmarks
```

---

## Backend Integration

The exported artifacts in `colab/artifacts/final_index/`:
- `faiss_index.bin`
- `metadata.pkl`
- `manifest.json`

integrate directly with the production FastAPI backend (`backend/app/vector_store.py` and `backend/app/pipeline.py`) to serve live voice queries through Sarvam STT and the HH Goa web frontend.
