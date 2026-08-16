# HH Goa Task 2 — Colab Research Pipeline

A reproducible Google Colab research pipeline for building, benchmarking, and
exporting the Voice-Enabled RAG system described in the HH Goa 2026 Task 2
specification.

## Purpose

This directory contains the complete experimental infrastructure required to:

1. **Analyze** the MSMARCO-XI Indic dataset
2. **Experiment** with chunking strategies, embedding models, and retrieval configurations
3. **Benchmark** retrieval quality (Recall@K, MRR) and latency (P50/P70/P100)
4. **Build** the production vector index
5. **Export** artifacts for integration with the existing HH Goa backend

## Execution Order

```
01_environment_and_dataset.ipynb
        ↓
02_dataset_analysis.ipynb
        ↓
03_chunking_experiments.ipynb
        ↓
04_embedding_benchmark.ipynb
        ↓
05_retrieval_benchmark.ipynb
        ↓
06_latency_benchmark.ipynb
        ↓
07_semantic_chunking.ipynb
        ↓
   (optional) 08_optional_retriever_finetuning.ipynb
        ↓
09_final_index_build.ipynb
        ↓
10_export_and_integration.ipynb
```

Notebooks 01–07 are sequential research experiments. Notebook 08 is optional
and should only be run if baseline retrieval quality is insufficient. Notebooks
09–10 use the experimentally validated configuration to build and export the
final production artifacts.

## How to Use in Google Colab

### 1. Clone the repository

```python
!git clone <YOUR-REPOSITORY-URL>
%cd <REPOSITORY-NAME>
```

Replace `<YOUR-REPOSITORY-URL>` with your actual GitHub repository URL.

### 2. Install dependencies

```python
!pip install -r colab/requirements-colab.txt -q
```

### 3. Open a notebook

Navigate to `colab/notebooks/` and open the first notebook.

### 4. Run cells sequentially

Each notebook is designed to be run top-to-bottom. Markdown cells explain what
each section does, why it is needed, and what output to expect.

## Required Secrets

The following API keys may be required. **Never hard-code them.**

| Secret | Required For | How to Set |
|--------|-------------|------------|
| `HF_TOKEN` | Hugging Face dataset access (if gated) | Colab Secrets or `os.environ` |
| `SARVAM_API_KEY` | Sarvam STT API calls | Colab Secrets or `os.environ` |
| `GROQ_API_KEY` | Groq LLM generation | Colab Secrets or `os.environ` |

In Google Colab, add secrets via: **🔑 Secrets** (left sidebar) → Add secret.

In code, access them via:
```python
from google.colab import userdata
api_key = userdata.get('SARVAM_API_KEY')
```

## Directory Structure

```
colab/
├── README.md                          ← This file
├── requirements-colab.txt             ← Colab pip dependencies
├── notebooks/                         ← Jupyter notebooks (run in order)
│   ├── 01_environment_and_dataset.ipynb
│   ├── 02_dataset_analysis.ipynb
│   ├── 03_chunking_experiments.ipynb
│   ├── 04_embedding_benchmark.ipynb
│   ├── 05_retrieval_benchmark.ipynb
│   ├── 06_latency_benchmark.ipynb
│   ├── 07_semantic_chunking.ipynb
│   ├── 08_optional_retriever_finetuning.ipynb
│   ├── 09_final_index_build.ipynb
│   └── 10_export_and_integration.ipynb
├── src/                               ← Shared Python modules
│   ├── __init__.py
│   ├── dataset_utils.py
│   ├── chunking.py
│   ├── embeddings.py
│   ├── retrieval.py
│   ├── evaluation.py
│   ├── latency.py
│   └── utils.py
├── configs/
│   └── experiment_config.yaml         ← Central experiment configuration
├── data/                              ← Downloaded/cached dataset files
├── artifacts/                         ← Generated indexes, models, exports
└── reports/                           ← Benchmark results, analysis outputs
```

## Expected Outputs

| Notebook | Primary Output |
|----------|---------------|
| 01 | `reports/dataset_metadata.json` |
| 02 | `reports/dataset_analysis.json`, distribution plots |
| 03 | `reports/chunking_comparison.json` |
| 04 | `reports/embedding_benchmark.json` |
| 05 | `reports/retrieval_benchmark.json` |
| 06 | `reports/latency_benchmark.json` |
| 07 | `reports/semantic_chunking_results.json` |
| 08 | `reports/finetuning_comparison.json` (optional) |
| 09 | `artifacts/final_index/` (FAISS index + metadata) |
| 10 | `artifacts/export/` (manifest + integration config) |

## Final Artifacts

After running notebooks 09 and 10, the `artifacts/export/` directory will contain:

- `faiss_index.bin` — The production FAISS vector index
- `metadata.pkl` — Passage metadata mapping (text, language, query_id, etc.)
- `manifest.json` — Build manifest with model name, dimensions, strategy, timestamps
- `config.yaml` — Recommended backend configuration

These artifacts integrate with the existing backend via the `VectorStore` and
`RAGPipeline` interfaces defined in `backend/app/`.

## Performance Target

The HH Goa Task 2 specification requires:

> "The full process — chunking + vector DB retrieval + everything through to
> final output — should complete in under 200 ms."

**Important distinction:**
- **Offline latency** (corpus preprocessing, chunking, embedding, index creation)
  is a one-time cost and is NOT part of the 200 ms target.
- **Online latency** (query embedding → vector search → context construction →
  LLM generation → guardrails) is what must be under 200 ms.

The latency benchmark (Notebook 06) measures online latency separately and
reports P50, P70, and P100 percentiles.

## Reproducibility

All experiments use `SEED = 42` by default. Each notebook records:
- Python version, package versions
- GPU name, CUDA version (when available)
- Model versions, configuration parameters
- Dataset configuration and sample sizes
