"""Smoke test for SarvamSTTProvider across 5 Indic languages."""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.stt import SarvamSTTProvider

from scripts.benchmark_stt import generate_synthetic_wav


def test_smoke():
    provider = SarvamSTTProvider()
    languages = [
        ("Hindi", "hin_Deva", "hi-IN"),
        ("Marathi", "mar_Deva", "mr-IN"),
        ("Bengali", "ben_Beng", "bn-IN"),
        ("Tamil", "tam_Taml", "ta-IN"),
        ("Telugu", "tel_Telu", "te-IN"),
    ]

    print("=== SARVAM STT SMOKE TEST (5 INDIC LANGUAGES) ===")
    all_ok = True
    for name, int_lang, sarvam_code in languages:
        wav = generate_synthetic_wav(sample_rate=16000, duration_sec=2.0)
        res = provider.transcribe(wav, filename=f"smoke_{int_lang}.wav", language=int_lang)
        print(f"[{name}] Lang: {int_lang} -> {sarvam_code} | Status: {res.status} | Inference: {res.stt_inference_ms:.2f} ms | Total: {res.stt_total_ms:.2f} ms")
        if res.status == "error":
            print(f"   Error: {res.error}")
            all_ok = False

    print(f"Smoke test overall: {'PASS' if all_ok else 'FAIL'}")


if __name__ == "__main__":
    test_smoke()
