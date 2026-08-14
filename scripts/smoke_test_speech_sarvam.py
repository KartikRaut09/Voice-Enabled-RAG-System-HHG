"""Smoke test for SarvamSTTProvider with REAL spoken speech audio across 5 Indic languages."""

import io
import sys
from pathlib import Path
from dotenv import load_dotenv
from gtts import gTTS

load_dotenv()
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.app.stt import SarvamSTTProvider


def test_speech_smoke():
    provider = SarvamSTTProvider()
    languages = [
        ("Hindi", "hin_Deva", "hi", "भारत की राजधानी क्या है?"),
        ("Marathi", "mar_Deva", "mr", "महाराष्ट्राची राजधानी कोणती?"),
        ("Bengali", "ben_Beng", "bn", "কলকাতা কোন নদীর তীরে অবস্থিত?"),
        ("Tamil", "tam_Taml", "ta", "சென்னையின் முக்கிய இடம் எது?"),
        ("Telugu", "tel_Telu", "te", "హైదరాబాద్ యొక్క ముఖ్యమైన ప్రదేశం ఏది?"),
    ]

    print("=== SARVAM REAL SPEECH SMOKE TEST (5 INDIC LANGUAGES) ===")
    all_ok = True
    for name, int_lang, tts_code, spoken_text in languages:
        # 1. Synthesize legitimate spoken speech audio
        tts = gTTS(spoken_text, lang=tts_code)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        audio_bytes = fp.getvalue()

        # 2. Transcribe using live Sarvam STT (saaras:v3)
        res = provider.transcribe(audio_bytes, filename=f"speech_{int_lang}.mp3", language=int_lang)

        print(f"[{name}]")
        print(f"   Ground Truth:  {spoken_text}")
        print(f"   Transcribed:   {res.text}")
        print(f"   Language:      {res.language}")
        print(f"   Status:        {res.status}")
        print(f"   Inference ms:  {res.stt_inference_ms:.2f} ms")
        print(f"   Total ms:      {res.stt_total_ms:.2f} ms")

        if res.status != "success" or not res.text.strip():
            print(f"   [!] Failed or empty transcript")
            all_ok = False
        else:
            print(f"   [+] OK")
        print()

    print(f"Smoke test overall: {'PASS' if all_ok else 'FAIL'}")
    return all_ok


if __name__ == "__main__":
    test_speech_smoke()
