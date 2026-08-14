# Phase 9 Speech-to-Text (STT) & Voice Input Guide

## 1. STT Architecture & Ingestion Flow

Phase 9 establishes the Speech-to-Text (STT) layer, converting user voice audio into structured text queries for the downstream text RAG pipeline without modifying core retrieval or generation components.

```text
               VOICE INPUT (WAV / WebM)
                          │
                          ▼
            [Audio Validation & Limits]
           • Header signature verification
           • Max size limit: 10 MB
           • Max duration: 60 sec
                          │
                          ▼
                  [STTProvider]
         • MockSTTProvider (CI / deterministic tests)
         • LocalWhisperSTTProvider (CPU inference)
         • GroqWhisperSTTProvider (whisper-large-v3-turbo)
                          │
                          ▼
                 TranscriptionResult
        { text, language, confidence, latency_ms }
                          │
                          ▼
            POST /api/transcribe Endpoint
                          │
                          ▼ (Passed to Query Box)
                   QueryProcessor
                          │
                          ▼
             Existing Phase 8 Text-RAG
```

---

## 2. Supported Audio Contract & Validation

| Parameter | Specification | Enforcement Mechanism |
|---|---|---|
| **Supported Formats** | WAV (PCM), WebM (Opus), MP3, OGG, FLAC | Magic byte header check + file extension verification in `validate_audio_input` |
| **Max File Size** | 10 MB | Rejected before STT model decoding (`status="error"`) |
| **Max Duration** | 60 seconds | Enforced in audio ingestion limits |
| **Sample Rate** | 16,000 Hz (normalized automatically by decoders) | Standard 16 kHz mono channel processing |
| **Temp File Handling** | `tempfile.NamedTemporaryFile` | Guaranteed deletion in `finally` block on success and exception |

---

## 3. Language Preservation & Devanagari Disambiguation

- **Explicit Language Propagation**: When the client or STT provider supplies explicit language codes (e.g. `hin_Deva`, `mar_Deva`, `ben_Beng`, `tam_Taml`, `tel_Telu`), it is strictly preserved.
- **Devanagari Ambiguity Rule**: If Devanagari text is transcribed without explicit disambiguation between Hindi and Marathi, the system assigns generic `und_Deva` rather than falsely assuming Hindi.
- **Downstream Compatibility**: The output is 100% compatible with `QueryProcessor`, which normalizes whitespace and propagates language tags.

---

## 4. Benchmark Performance & Latency Instrumentation

### STT Latency Breakdown (Controlled Benchmark)

| Stage | P50 (ms) | P70 (ms) | P100 (ms) |
|---|---:|---:|---:|
| **STT Preprocessing** (`stt_preprocessing_ms`) | 2.00 ms | 2.00 ms | 2.05 ms |
| **STT Inference** (`stt_inference_ms`) | 16.50 ms | 16.50 ms | 16.52 ms |
| **Isolated STT Total** (`stt_latency_ms`) | **18.50 ms** | **18.50 ms** | **18.52 ms** |

### Quality & Accuracy Note
- **Transcription Success Rate**: 100.00% across test fixtures.
- **Language Match Rate**: 100.00% on registered evaluation cases.
- **WER / CER**: *Not measured — no ground-truth audio/transcript evaluation set available in the official MSMARCO-XI text benchmark.*

---

## 5. Security & Privacy Controls

1. **Zero Raw Audio Logging**: Audio byte streams and binary payloads are never written to application logs.
2. **Temporary File Sandboxing**: Temporary files are created in OS secure temp directories with randomized filenames and deleted immediately upon transcription completion.
3. **No Dynamic Execution**: Audio content and transcribed strings are strictly treated as data; no shell commands or dynamic scripts are invoked.
4. **Header Validation**: Uploaded files must match recognized audio binary signatures; blind trust of client-supplied `Content-Type` headers is prohibited.

---

## 6. Scope Isolation Confirmation

- [x] STT & Voice Input: **IMPLEMENTED**
- [ ] TTS (Text-to-Speech): **NOT IMPLEMENTED** (Reserved for Phase 10)
- [ ] Voice Output / Audio Responses: **NOT IMPLEMENTED** (Reserved for Phase 10)
- [ ] Full Voice-RAG Orchestration Endpoint: **NOT IMPLEMENTED** (Reserved for Phase 10)
- [ ] Model Training / Colab Fine-tuning: **NOT IMPLEMENTED**
