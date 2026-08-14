# HH Goa 2026 — Task 2: Voice-Enabled RAG System

## Overview

A voice-enabled Retrieval-Augmented Generation (RAG) system for the HH Goa 2026 hackathon. Users speak a question, the system transcribes it, retrieves relevant context from the MSMARCO-XI dataset (14 Indic languages), and generates a grounded answer.

## Pipeline

```
Voice Input → STT → Query Processing → Retrieval → Reranking → Generation → Guardrails → Answer
```

## Quick Start

### Prerequisites
- Python 3.11+

### Setup
```bash
python -m venv venv
venv\Scripts\activate
pip install -r backend/requirements.txt
copy .env.example .env
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

Open http://localhost:8000

### Docker
```bash
docker-compose up --build
```

## API

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check |
| POST | `/api/query` | Submit a query |

## Latency Metrics

Three separately instrumented metrics are tracked and reported as P50 / P70 / P100:

1. **RAG Latency**: query processing → embedding → retrieval → reranking → generation → guardrails
2. **STT Latency**: isolated speech-to-text processing time (API/model transcription)
3. **Full E2E Latency**: STT Latency + RAG Latency (from backend receipt of audio to final response)

*Note*: Microphone/user speech duration and browser audio-capture time are excluded from this engineering benchmark.

## Dataset

[ai4bharat/MSMARCO-XI](https://huggingface.co/datasets/ai4bharat/MSMARCO-XI)

## Status

Phase 0: Architecture & Environment ✅

#RAGInGoa
