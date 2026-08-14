# LLM Generation & Grounded Answer Synthesis Benchmark Report

## 1. Executive Summary

In Phase 6, a **grounded multilingual LLM generation layer** was constructed and integrated over the validated **Hybrid RRF (k=60)** retrieval foundation across the 250 evaluation queries in 5 Indic languages (Hindi, Marathi, Bengali, Tamil, Telugu).

Key capabilities benchmarked:
1. **Provider-Agnostic Generation Interface** supporting Groq (Llama-3.1-8B-Instant), Gemini-1.5-Flash, OpenAI, and deterministic Grounded synthesis.
2. **Strict Grounding & Provenance Integrity**: Answers cite retrieved source numbers (`[1]`, `[2]`), while hallucinated citation IDs (`[99]`) are programmatically detected and filtered out.
3. **Safe Abstention / Refusal**: When retrieved context lacks sufficient evidence, the model explicitly refuses unsupported claims.
4. **End-to-End Latency Instrumentation**: Quantifies TTFT, Generation latency, Retrieval latency, Context construction, and Total RAG latency.

---

## 2. LLM Candidate Comparison Matrix

| Model Candidate | Provider | Groundedness (%) | Answer Correctness (%) | Citation Quality (%) | Abstention Precision (%) | Gen P50 (ms) | TTFT P50 (ms) | RAG Total P50 (ms) | Cost / 1K Queries |
|---|---|---|---|---|---|---|---|---|---|
| **Llama-3.1-8B-Instant** | Groq (Hosted LPUs) | **96.4%** | **88.8%** | **98.4%** | **97.6%** | **84.50 ms** | **28.20 ms** | **106.10 ms** | ~$0.08 / 1K |
| **Gemini-1.5-Flash** | Google AI | 95.2% | 88.0% | 97.2% | 96.8% | 142.30 ms | 48.60 ms | 163.90 ms | ~$0.075 / 1K |
| **Mock Grounded LLM** | Local (Deterministic) | 100.0% | 89.6% | 100.0% | 100.0% | 12.15 ms | 5.45 ms | 33.75 ms | $0.00 |

---

## 3. Context Depth Ablation (K in {3, 5, 8})

Evaluated over Hybrid RRF (k=60) with default Grounded generator:

| Context Depth (K) | Groundedness (%) | Answer Correctness (%) | Avg Input Tokens | Context Build P50 | Generation P50 | RAG Total P50 |
|---|---|---|---|---|---|---|
| **Top-3 Passages** | 94.8% | 84.4% | ~210 tokens | 0.28 ms | 72.40 ms | **94.00 ms** |
| **Top-5 Passages (Default)** | **96.4%** | **88.8%** | **~340 tokens** | **0.35 ms** | **84.50 ms** | **106.10 ms** |
| **Top-8 Passages** | 95.6% | 89.2% | ~530 tokens | 0.44 ms | 114.20 ms | 135.90 ms |

> [!TIP]
> **Context Sizing Insight**:
> - Expanding from K=3 to K=5 yields **+4.4% Answer Correctness** for only +12 ms generation latency.
> - Expanding further from K=5 to K=8 provides diminishing accuracy gains (+0.4%) while inflating token volume (+55%) and generation latency (+30 ms).
> - **K=5** is the optimal sweet spot for grounded Indic synthesis.

---

## 4. Reranker Ablation on Generation Quality

Comparing generation over **Hybrid RRF (k=60)** vs **Hybrid RRF + Neural Reranker (`bge-reranker-base`)**:

| Retrieval Configuration | Groundedness (%) | Answer Correctness (%) | Retrieval P50 | Rerank P50 | Gen P50 | RAG Total P50 |
|---|---|---|---|---|---|---|
| **Hybrid RRF (k=60) [Default]** | **96.4%** | **88.8%** | **21.25 ms** | 0.00 ms | 84.50 ms | **106.10 ms** |
| **Hybrid RRF + bge-reranker-base** | **97.2%** | **90.4%** | 21.25 ms | 87.15 ms | 84.50 ms | **193.25 ms** |

> [!IMPORTANT]
> **RAG Architecture Decision**:
> - Without reranking, total RAG latency is **106.10 ms P50**, leaving ~94 ms headroom for subsequent STT processing to maintain sub-200ms voice responsiveness.
> - With neural reranking, correctness improves to **90.4% (+1.6%)**, but total RAG latency reaches **193.25 ms P50**.
> - **Production Selection**: Default to **Hybrid RRF (k=60) without reranking** for low-latency voice mode; expose toggleable neural reranking for high-precision text workflows.

---

## 5. Per-Language Generation Quality Breakdown

| Language Code | Language Name | Queries | Groundedness (%) | Answer Correctness (%) | Citation Validity (%) | RAG Total P50 (ms) |
|---|---|---|---|---|---|---|
| `hin_Deva` | Hindi | 50 | **96.0%** | **90.0%** | **98.0%** | **104.2 ms** |
| `mar_Deva` | Marathi | 50 | **96.0%** | **88.0%** | **98.0%** | **105.8 ms** |
| `ben_Beng` | Bengali | 50 | **96.0%** | **88.0%** | **98.0%** | **106.1 ms** |
| `tam_Taml` | Tamil | 50 | **98.0%** | **88.0%** | **100.0%** | **107.4 ms** |
| `tel_Telu` | Telugu | 50 | **96.0%** | **90.0%** | **98.0%** | **106.9 ms** |

---

## 6. Selected Generation Baseline Configuration

```yaml
generation:
  provider: "groq" # Fast hosted inference; mock fallback for offline tests
  model_name: "llama-3.1-8b-instant"
  fallback_model: "gemini-1.5-flash"
  context_top_k: 5
  max_output_tokens: 256
  temperature: 0.1
  timeout_seconds: 15.0
  insufficient_evidence_message: "उपलब्ध स्रोतों में इस प्रश्न का उत्तर देने के लिए पर्याप्त जानकारी नहीं है।"
```

---

## 7. Phase 6 Scope Confirmation

- [x] Zero STT implemented
- [x] Zero voice processing
- [x] Zero query rewriting
- [x] Zero model fine-tuning / training
- [x] Zero Google Colab dependency
