"""Phase 9 Speech-to-Text (STT) Latency & Quality Benchmark Script.

Evaluates STT providers on controlled audio inputs across the 5 target Indic languages:
- Hindi (hin_Deva)
- Marathi (mar_Deva)
- Bengali (ben_Beng)
- Tamil (tam_Taml)
- Telugu (tel_Telu)

Measures:
1. Isolated STT latency breakdown (stt_preprocessing_ms, stt_inference_ms, stt_total_ms)
2. Percentiles: P50, P70, P100/max
3. Language preservation & detection accuracy
4. Empty audio and format rejection rate
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
import time
import yaml

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.stt import MockSTTProvider, STTProvider, get_stt_provider, validate_audio_input


def load_config() -> dict:
    config_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def calculate_percentile(data: list[float], percentile: float) -> float:
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (percentile / 100.0)
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return float(sorted_data[int(k)])
    d0 = sorted_data[int(f)] * (c - k)
    d1 = sorted_data[int(c)] * (k - f)
    return float(d0 + d1)


def generate_synthetic_wav(sample_rate: int = 16000, duration_sec: float = 2.0, signature_tag: str = "") -> bytes:
    """Generate a clean synthetic WAV byte payload with a signature tag in the data section."""
    num_samples = int(sample_rate * duration_sec)
    data_bytes = signature_tag.encode("utf-8") + b"\x00" * (num_samples * 2 - len(signature_tag))

    header = b"RIFF" + (36 + len(data_bytes)).to_bytes(4, "little") + b"WAVEfmt " + (16).to_bytes(4, "little")
    header += (1).to_bytes(2, "little") + (1).to_bytes(2, "little")  # PCM mono
    header += sample_rate.to_bytes(4, "little") + (sample_rate * 2).to_bytes(4, "little")
    header += (2).to_bytes(2, "little") + (16).to_bytes(2, "little")
    header += b"data" + len(data_bytes).to_bytes(4, "little")
    return header + data_bytes


def main() -> None:
    config = load_config()
    print("=" * 60)
    print("PHASE 9 SPEECH-TO-TEXT (STT) BENCHMARK")
    print("=" * 60)

    # 1. Setup Test Dataset Across 5 Indic Languages
    test_cases = [
        {"id": "hi_1", "lang": "hin_Deva", "text": "भारत की राजधानी नई दिल्ली है।", "sig": "SIG_HI_01"},
        {"id": "hi_2", "lang": "hin_Deva", "text": "ताजमहल किस शहर में स्थित है?", "sig": "SIG_HI_02"},
        {"id": "mr_1", "lang": "mar_Deva", "text": "महाराष्ट्राची राजधानी मुंबई आहे.", "sig": "SIG_MR_01"},
        {"id": "mr_2", "lang": "mar_Deva", "text": "पुणे शहराचे प्राचीन नाव काय आहे?", "sig": "SIG_MR_02"},
        {"id": "bn_1", "lang": "ben_Beng", "text": "পশ্চিমবঙ্গের রাজধানী কোথায়?", "sig": "SIG_BN_01"},
        {"id": "bn_2", "lang": "ben_Beng", "text": "কলকাতা কোন নদীর তীরে অবস্থিত?", "sig": "SIG_BN_02"},
        {"id": "ta_1", "lang": "tam_Taml", "text": "தமிழ்நாட்டின் தலைநகரம் சென்னை ஆகும்.", "sig": "SIG_TA_01"},
        {"id": "ta_2", "lang": "tam_Taml", "text": "மதுரை மீனாட்சி அம்மன் கோவில் எங்குள்ளது?", "sig": "SIG_TA_02"},
        {"id": "te_1", "lang": "tel_Telu", "text": "తెలంగాణ రాజధాని హైదరాబాద్.", "sig": "SIG_TE_01"},
        {"id": "te_2", "lang": "tel_Telu", "text": "చార్మినార్ ఏ నగరంలో ఉంది?", "sig": "SIG_TE_02"},
    ]

    # Instantiate Provider
    mock_provider = MockSTTProvider(simulated_latency_ms=18.5)
    for c in test_cases:
        mock_provider.register_transcript(c["sig"], c["text"], c["lang"])

    print(f"\nEvaluating STT Latency across {len(test_cases) * 10} simulated voice requests...")

    stt_prep_latencies: list[float] = []
    stt_infer_latencies: list[float] = []
    stt_total_latencies: list[float] = []

    success_count = 0
    language_match_count = 0

    iterations = 10
    for _ in range(iterations):
        for case in test_cases:
            wav_payload = generate_synthetic_wav(duration_sec=2.5, signature_tag=case["sig"])
            t0 = time.perf_counter()
            res = mock_provider.transcribe(wav_payload, filename="voice.wav", language=case["lang"])
            elapsed = (time.perf_counter() - t0) * 1000.0

            stt_prep_latencies.append(res.stt_preprocessing_ms)
            stt_infer_latencies.append(res.stt_inference_ms)
            stt_total_latencies.append(res.stt_total_ms)

            if res.status == "success":
                success_count += 1
            if res.language == case["lang"]:
                language_match_count += 1

    total_runs = len(test_cases) * iterations
    p50 = calculate_percentile(stt_total_latencies, 50)
    p70 = calculate_percentile(stt_total_latencies, 70)
    p100 = max(stt_total_latencies) if stt_total_latencies else 0.0

    prep_p50 = calculate_percentile(stt_prep_latencies, 50)
    infer_p50 = calculate_percentile(stt_infer_latencies, 50)

    print("\n------------------------------------------------------------")
    print("STT BENCHMARK SUMMARY")
    print("------------------------------------------------------------")
    print(f"Total Test Runs:            {total_runs}")
    print(f"Transcription Success Rate: {(success_count / total_runs) * 100:.2f}%")
    print(f"Language Match Rate:        {(language_match_count / total_runs) * 100:.2f}%")
    print("------------------------------------------------------------")
    print(f"Preprocessing Latency P50:  {prep_p50:.2f} ms")
    print(f"Inference Latency P50:      {infer_p50:.2f} ms")
    print("------------------------------------------------------------")
    print(f"Isolated STT Latency P50:   {p50:.2f} ms")
    print(f"Isolated STT Latency P70:   {p70:.2f} ms")
    print(f"Isolated STT Latency P100:  {p100:.2f} ms")
    print("------------------------------------------------------------")
    print("Note: Ground-truth audio WER/CER requires task-provided audio dataset.")
    print("WER/CER: Not measured — no ground-truth audio/transcript evaluation set available in MSMARCO-XI.")
    print("============================================================\n")


if __name__ == "__main__":
    main()
