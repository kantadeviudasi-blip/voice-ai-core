import io
import os
import httpx
import asyncio
from typing import Callable, Optional
from src.adapters.stt.base import BaseSTT

class GroqWhisperSTT(BaseSTT):
    """
    Groq Cloud Whisper STT Adapter (whisper-large-v3 / whisper-large-v3-turbo).
    Ultra-low latency STT powered by Groq LPUs using the EXACT same GROQ_API_KEY as the LLM.
    Supports Hindi, Hinglish, and multilingual audio.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "whisper-large-v3-turbo",
        language: Optional[str] = "hi",
        prompt: Optional[str] = "नमस्ते, हाँ जी, solar panel, subsidy, 3kW, cost, bijli bill, EMI, WhatsApp, rooftop solar, survey, Sneha, telecaller.",
        sample_rate: int = 16000
    ):
        super().__init__(language=language, sample_rate=sample_rate)
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        if not self.api_key:
            env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), ".env")
            if os.path.exists(env_path):
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("GROQ_API_KEY="):
                            self.api_key = line.split("=", 1)[1].strip()
        self.model = model
        self.prompt = prompt
        self.url = "https://api.groq.com/openai/v1/audio/transcriptions"

    def _normalize_pcm(self, pcm_data: bytes) -> bytes:
        """Boosts speech only when legitimate voice amplitude is present, ignoring background hiss/rustle."""
        if not pcm_data:
            return pcm_data
        import struct
        count = len(pcm_data) // 2
        if count == 0:
            return pcm_data
        try:
            samples = struct.unpack(f"<{count}h", pcm_data)
            max_val = max(abs(s) for s in samples)
            # Only boost if genuine voice signal exists (max_val >= 500) and isn't already loud
            if 500 <= max_val < 18000:
                gain = min(3.0, 24000.0 / max_val)
                boosted = [max(-32768, min(32767, int(s * gain))) for s in samples]
                return struct.pack(f"<{count}h", *boosted)
        except Exception:
            pass
        return pcm_data

    async def transcribe(self, audio_data: bytes) -> str:
        if not self.api_key:
            return "[Groq API Key not set]"

        if not audio_data or len(audio_data) < 320:
            return ""

        # Normalize audio levels
        normalized_pcm = self._normalize_pcm(audio_data)

        # Wrap raw PCM into standard WAV format header for Groq Whisper
        wav_buffer = io.BytesIO()
        self._write_wav_header(wav_buffer, len(normalized_pcm), self.sample_rate)
        wav_buffer.write(normalized_pcm)
        wav_bytes = wav_buffer.getvalue()

        headers = {
            "Authorization": f"Bearer {self.api_key}"
        }
        files = {
            "file": ("audio.wav", wav_bytes, "audio/wav")
        }
        data = {
            "model": self.model,
            "response_format": "json",
            "temperature": "0.0"
        }
        if self.language:
            data["language"] = self.language
        if self.prompt:
            data["prompt"] = self.prompt

        # ── Whisper Hallucination Blacklist ────────────────────────────────────
        # Groq Whisper (large-v3-turbo) has known ghost-transcriptions on:
        #   • Near-silent audio / background fan noise
        #   • Very short utterances < 300ms
        #   • Audio frames containing only breath or hiss
        # All entries are normalised to lowercase; comparison strips punctuation.
        # -----------------------------------------------------------------------
        WHISPER_HALLUCINATIONS = {
            # English video / social media hallucinations
            "thank you", "thank you.", "thank you!", "thanks", "thanks.",
            "thanks for watching", "thanks for watching!",
            "you", "the", ".", "..", "...", "a", "i",
            "bye", "goodbye", "bye bye", "see you", "see you soon",
            "subtitles by", "subtitles", "closed captioning",
            "[music]", "(music)", "[applause]", "[laughter]",
            "subscribe", "like and subscribe", "amara.org",
            "watching", "youtube", "please subscribe",
            "like comment subscribe", "don't forget to subscribe",
            "do subscribe", "share karo",

            # Common English ambient / silence triggers
            "oh", "ah", "um", "mm", "hmm", "uh", "er",
            "okay", "ok", "okay.", "ok.", "yes", "no",

            # Hindi-specific hallucinations on near-silent Whisper runs
            "सुप्रभात", "थैंक यू", "धन्यवाद", "नमस्ते",
            "हाँ", "हां", "जी", "ठीक है", "ठीक",
            "हेलो", "हैलो", "बाय", "अच्छा",

            # Whisper filler artifacts on Hindi audio
            "आप", "मैं", "और", "के", "है", "हैं",
            "की", "में", "को", "से", "पर",
        }

        async with httpx.AsyncClient(timeout=8.0) as client:
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    response = await client.post(self.url, headers=headers, files=files, data=data)
                    if response.status_code == 200:
                        result = response.json()
                        text = result.get("text", "").strip()
                        # Discard known Whisper silence/hallucination artifacts
                        cleaned = text.lower().strip(" .,!?:;\u0964")
                        if cleaned in WHISPER_HALLUCINATIONS or len(cleaned) < 2:
                            return ""
                        return text
                    elif response.status_code == 429 or response.status_code >= 500:
                        if attempt < max_retries - 1:
                            await asyncio.sleep(0.5 * (attempt + 1))
                            continue
                        return f"[Groq STT Error: {response.status_code}]"
                    else:
                        return f"[Groq STT Error: {response.status_code}]"
                except httpx.RequestError as e:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(0.5 * (attempt + 1))
                        continue
                    return f"[Groq STT Exception: {str(e)}]"
                except Exception as e:
                    return f"[Groq STT Exception: {str(e)}]"
            return ""


    def _write_wav_header(self, buffer: io.BytesIO, data_length: int, sample_rate: int):
        """Helper to prepend standard 44-byte WAV header to linear PCM."""
        import struct
        channels = 1
        bits_per_sample = 16
        byte_rate = sample_rate * channels * (bits_per_sample // 8)
        block_align = channels * (bits_per_sample // 8)

        # RIFF header
        buffer.write(b"RIFF")
        buffer.write(struct.pack("<I", 36 + data_length))
        buffer.write(b"WAVE")
        # fmt chunk
        buffer.write(b"fmt ")
        buffer.write(struct.pack("<I", 16)) # Subchunk1Size
        buffer.write(struct.pack("<H", 1))  # PCM format
        buffer.write(struct.pack("<H", channels))
        buffer.write(struct.pack("<I", sample_rate))
        buffer.write(struct.pack("<I", byte_rate))
        buffer.write(struct.pack("<H", block_align))
        buffer.write(struct.pack("<H", bits_per_sample))
        # data chunk
        buffer.write(b"data")
        buffer.write(struct.pack("<I", data_length))

    async def connect_stream(self, on_transcript_callback: Callable[[str, bool], None]):
        pass

    async def push_audio(self, chunk: bytes):
        pass

    async def close(self):
        pass
