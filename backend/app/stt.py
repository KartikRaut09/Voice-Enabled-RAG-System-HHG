"""Speech-to-Text (STT) Provider and Audio Ingestion Module.

Provides deterministic audio validation, language preservation, and transcription:
1. Validates audio input size and headers without blind MIME trust.
2. Abstract STTProvider interface and concrete implementations:
   - MockSTTProvider (deterministic for unit tests & CI)
   - LocalWhisperSTTProvider (lightweight CPU inference via whisper)
   - GroqWhisperSTTProvider (fast cloud inference via whisper-large-v3-turbo)
3. Preserves explicit language metadata and maps ambiguous Devanagari to und_Deva.
4. Ensures secure temporary file handling and zero audio leakage in logs.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import re
import tempfile
import time
from typing import Any

from backend.app.config import get_logger
from backend.app.query_processor import detect_indic_script

logger = get_logger(__name__)


@dataclass
class TranscriptionResult:
    """Structured result from speech transcription."""

    text: str
    language: str | None = None
    confidence: float | None = None
    provider: str = "mock"
    model: str = "mock"
    stt_preprocessing_ms: float = 0.0
    stt_inference_ms: float = 0.0
    stt_total_ms: float = 0.0
    status: str = "success"  # success, empty_transcription, error
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_audio_input(
    audio_bytes: bytes,
    max_size_mb: int = 10,
    filename: str | None = None,
) -> tuple[bool, str | None]:
    """Validate audio payload size and basic structure."""
    if not audio_bytes or len(audio_bytes) == 0:
        return False, "Audio file is empty"

    max_bytes = max_size_mb * 1024 * 1024
    if len(audio_bytes) > max_bytes:
        return False, f"Audio file exceeds maximum size limit of {max_size_mb} MB"

    # Header signature verification
    is_wav = audio_bytes.startswith(b"RIFF") and b"WAVE" in audio_bytes[:16]
    is_ogg_opus = audio_bytes.startswith(b"OggS")
    is_webm = audio_bytes.startswith(b"\x1a\x45\xdf\xa3")
    is_mp3 = audio_bytes.startswith(b"\xff\xfb") or audio_bytes.startswith(b"\xff\xf3") or audio_bytes.startswith(b"ID3")

    if not (is_wav or is_ogg_opus or is_webm or is_mp3):
        # If extension is provided and recognizable, allow it, else flag unsupported
        ext = Path(filename or "").suffix.lower()
        if ext not in (".wav", ".mp3", ".ogg", ".webm", ".m4a", ".flac"):
            return False, "Unsupported or corrupt audio file format"

    return True, None


def normalize_stt_language(detected_lang: str | None, text: str, explicit_lang: str | None = None) -> str:
    """Normalize language from STT provider while respecting Devanagari ambiguity rules."""
    if explicit_lang:
        return explicit_lang

    if detected_lang:
        lang_lower = detected_lang.lower()
        if lang_lower in ("hi", "hindi", "hin_deva"):
            # Check if script is Devanagari; if explicit language was not given, use und_Deva
            return "und_Deva"
        elif lang_lower in ("mr", "marathi", "mar_deva"):
            return "mar_Deva"
        elif lang_lower in ("bn", "bengali", "ben_beng"):
            return "ben_Beng"
        elif lang_lower in ("ta", "tamil", "tam_taml"):
            return "tam_Taml"
        elif lang_lower in ("te", "telugu", "tel_telu"):
            return "tel_Telu"
        elif lang_lower in ("en", "english"):
            return "en"

    # Fallback to deterministic script detection on transcribed text
    script_lang = detect_indic_script(text)
    if script_lang:
        return script_lang

    return "en"


class STTProvider(ABC):
    """Abstract interface for Speech-to-Text providers."""

    @abstractmethod
    def transcribe(
        self,
        audio_bytes: bytes,
        filename: str | None = None,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Transcribe audio bytes to structured text."""
        pass


class MockSTTProvider(STTProvider):
    """Deterministic mock STT provider for unit testing, CI, and benchmarks."""

    def __init__(
        self,
        default_transcript: str = "भारत की राजधानी क्या है?",
        default_language: str = "und_Deva",
        model_name: str = "mock-stt-v1",
        simulated_latency_ms: float = 15.0,
    ) -> None:
        self.default_transcript = default_transcript
        self.default_language = default_language
        self.model_name = model_name
        self.simulated_latency_ms = simulated_latency_ms
        self.custom_transcripts: dict[str, tuple[str, str]] = {}
        self.should_fail = False
        self.should_return_empty = False

    def register_transcript(self, audio_signature: str, text: str, language: str) -> None:
        """Register deterministic transcription mapping."""
        self.custom_transcripts[audio_signature] = (text, language)

    def transcribe(
        self,
        audio_bytes: bytes,
        filename: str | None = None,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Return deterministic mock transcription with simulated timings."""
        t_start = time.perf_counter()

        if self.should_fail:
            return TranscriptionResult(
                text="",
                provider="mock",
                model=self.model_name,
                status="error",
                error="Simulated STT provider failure",
            )

        if self.should_return_empty:
            return TranscriptionResult(
                text="",
                provider="mock",
                model=self.model_name,
                status="empty_transcription",
                error="Speech not detected in audio",
            )

        # Check audio validity
        valid, err = validate_audio_input(audio_bytes, filename=filename)
        if not valid:
            return TranscriptionResult(
                text="",
                provider="mock",
                model=self.model_name,
                status="error",
                error=err,
            )

        t_prep = 2.0  # Simulated preprocessing ms
        t_infer = max(1.0, self.simulated_latency_ms - t_prep)

        # Check for signature matches in audio header
        sig = audio_bytes[:16].decode("latin-1", errors="ignore")
        text, detected_lang = self.default_transcript, self.default_language
        for k, (c_text, c_lang) in self.custom_transcripts.items():
            if k in sig:
                text, detected_lang = c_text, c_lang
                break

        final_lang = normalize_stt_language(detected_lang, text, explicit_lang=language)

        t_total = (time.perf_counter() - t_start) * 1000.0 + self.simulated_latency_ms

        return TranscriptionResult(
            text=text,
            language=final_lang,
            confidence=0.98,
            provider="mock",
            model=self.model_name,
            stt_preprocessing_ms=t_prep,
            stt_inference_ms=t_infer,
            stt_total_ms=t_total,
            status="success",
        )


class LocalWhisperSTTProvider(STTProvider):
    """Local inference STT provider using OpenAI Whisper on CPU."""

    def __init__(
        self,
        model_name: str = "tiny",
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load_model(self) -> Any:
        if self._model is None:
            import whisper

            logger.info("loading_whisper_model", model_name=self.model_name, device=self.device)
            # Map standard HF names to whisper names
            w_name = self.model_name.replace("openai/whisper-", "").replace("whisper-", "")
            if w_name not in ("tiny", "base", "small", "medium", "large"):
                w_name = "tiny"
            self._model = whisper.load_model(w_name, device=self.device)
        return self._model

    def transcribe(
        self,
        audio_bytes: bytes,
        filename: str | None = None,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Write audio to temporary file, perform Whisper transcription, and clean up."""
        valid, err = validate_audio_input(audio_bytes, filename=filename)
        if not valid:
            return TranscriptionResult(
                text="",
                provider="whisper_local",
                model=self.model_name,
                status="error",
                error=err,
            )

        t_start = time.perf_counter()
        suffix = Path(filename or "audio.wav").suffix or ".wav"

        # Secure temporary file creation with guaranteed cleanup
        temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        temp_path = Path(temp_file.name)
        try:
            temp_file.write(audio_bytes)
            temp_file.flush()
            temp_file.close()

            t_prep = (time.perf_counter() - t_start) * 1000.0

            t_infer_start = time.perf_counter()
            model = self._load_model()
            whisper_lang = None
            if language:
                # Map language codes to whisper language keys
                w_map = {
                    "hin_Deva": "hi",
                    "mar_Deva": "mr",
                    "ben_Beng": "bn",
                    "tam_Taml": "ta",
                    "tel_Telu": "te",
                    "en": "en",
                }
                whisper_lang = w_map.get(language, language.split("_")[0])

            res = model.transcribe(str(temp_path), language=whisper_lang)
            t_infer = (time.perf_counter() - t_infer_start) * 1000.0

            raw_text = res.get("text", "").strip()
            detected_lang = res.get("language")

            if not raw_text:
                return TranscriptionResult(
                    text="",
                    language=None,
                    provider="whisper_local",
                    model=self.model_name,
                    stt_preprocessing_ms=t_prep,
                    stt_inference_ms=t_infer,
                    stt_total_ms=t_prep + t_infer,
                    status="empty_transcription",
                    error="No speech detected in audio",
                )

            final_lang = normalize_stt_language(detected_lang, raw_text, explicit_lang=language)

            return TranscriptionResult(
                text=raw_text,
                language=final_lang,
                confidence=0.90,
                provider="whisper_local",
                model=self.model_name,
                stt_preprocessing_ms=t_prep,
                stt_inference_ms=t_infer,
                stt_total_ms=t_prep + t_infer,
                status="success",
            )
        except Exception as e:
            logger.error("whisper_transcription_failed", error=str(e))
            return TranscriptionResult(
                text="",
                provider="whisper_local",
                model=self.model_name,
                status="error",
                error=f"Local Whisper transcription failed: {str(e)}",
            )
        finally:
            if temp_path.exists():
                try:
                    os.unlink(temp_path)
                except Exception as del_err:
                    logger.warn("temp_audio_cleanup_failed", path=str(temp_path), error=str(del_err))


class GroqWhisperSTTProvider(STTProvider):
    """Fast cloud STT provider via Groq Whisper API (whisper-large-v3-turbo)."""

    def __init__(
        self,
        model_name: str = "whisper-large-v3-turbo",
        api_key: str | None = None,
        timeout_seconds: float = 15.0,
    ) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.getenv("GROQ_API_KEY")
        self.timeout_seconds = timeout_seconds

    def transcribe(
        self,
        audio_bytes: bytes,
        filename: str | None = None,
        language: str | None = None,
    ) -> TranscriptionResult:
        """Call Groq Whisper audio transcriptions API with temporary file cleanup."""
        valid, err = validate_audio_input(audio_bytes, filename=filename)
        if not valid:
            return TranscriptionResult(
                text="",
                provider="groq_whisper",
                model=self.model_name,
                status="error",
                error=err,
            )

        if not self.api_key:
            return TranscriptionResult(
                text="",
                provider="groq_whisper",
                model=self.model_name,
                status="error",
                error="GROQ_API_KEY environment variable is not configured",
            )

        t_start = time.perf_counter()
        suffix = Path(filename or "audio.wav").suffix or ".wav"
        temp_file = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        temp_path = Path(temp_file.name)

        try:
            temp_file.write(audio_bytes)
            temp_file.flush()
            temp_file.close()

            t_prep = (time.perf_counter() - t_start) * 1000.0

            from openai import OpenAI

            client = OpenAI(
                base_url="https://api.groq.com/openai/v1",
                api_key=self.api_key,
                timeout=self.timeout_seconds,
            )

            w_lang = None
            if language:
                w_lang = language.split("_")[0]

            t_infer_start = time.perf_counter()
            with open(temp_path, "rb") as audio_file:
                transcription = client.audio.transcriptions.create(
                    file=audio_file,
                    model=self.model_name,
                    language=w_lang,
                    response_format="json",
                )
            t_infer = (time.perf_counter() - t_infer_start) * 1000.0

            raw_text = getattr(transcription, "text", "").strip()
            if not raw_text:
                return TranscriptionResult(
                    text="",
                    provider="groq_whisper",
                    model=self.model_name,
                    stt_preprocessing_ms=t_prep,
                    stt_inference_ms=t_infer,
                    stt_total_ms=t_prep + t_infer,
                    status="empty_transcription",
                    error="Empty transcription returned by Groq Whisper",
                )

            final_lang = normalize_stt_language(None, raw_text, explicit_lang=language)

            return TranscriptionResult(
                text=raw_text,
                language=final_lang,
                confidence=0.95,
                provider="groq_whisper",
                model=self.model_name,
                stt_preprocessing_ms=t_prep,
                stt_inference_ms=t_infer,
                stt_total_ms=t_prep + t_infer,
                status="success",
            )
        except Exception as e:
            logger.error("groq_whisper_failed", error=str(e))
            return TranscriptionResult(
                text="",
                provider="groq_whisper",
                model=self.model_name,
                status="error",
                error=f"Groq Whisper transcription failed: {str(e)}",
            )
        finally:
            if temp_path.exists():
                try:
                    os.unlink(temp_path)
                except Exception as del_err:
                    logger.warn("temp_audio_cleanup_failed", path=str(temp_path), error=str(del_err))


def get_stt_provider(config: dict | None = None) -> STTProvider:
    """Factory to instantiate the configured STT provider."""
    cfg = config or {}
    stt_cfg = cfg.get("stt", {})
    provider_name = stt_cfg.get("provider", "mock").lower()

    if provider_name in ("whisper_local", "local_whisper"):
        return LocalWhisperSTTProvider(
            model_name=stt_cfg.get("model_name", "tiny"),
            device=stt_cfg.get("device", "cpu"),
        )
    elif provider_name in ("groq", "groq_whisper"):
        return GroqWhisperSTTProvider(
            model_name=stt_cfg.get("model_name", "whisper-large-v3-turbo"),
            timeout_seconds=float(stt_cfg.get("timeout_seconds", 15.0)),
        )
    else:
        return MockSTTProvider(
            model_name=stt_cfg.get("model_name", "mock-stt-v1"),
        )
