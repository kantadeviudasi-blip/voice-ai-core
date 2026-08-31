import io
import os
import httpx
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
        language: str = "hi",
        prompt: Optional[str] = "Namaste, main Hindi aur Hinglish mein baat kar raha hoon.",
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

    async def transcribe(self, audio_data: bytes) -> str:
        if not self.api_key:
            return "[Groq API Key not set]"

        if not audio_data or len(audio_data) < 100:
            return ""

        # Wrap raw PCM into standard WAV format header for Groq Whisper
        wav_buffer = io.BytesIO()
        self._write_wav_header(wav_buffer, len(audio_data), self.sample_rate)
        wav_buffer.write(audio_data)
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
            "language": self.language,
            "temperature": "0.0"
        }
        if self.prompt:
            data["prompt"] = self.prompt

        # Known Whisper hallucination phrases on near-silent audio — discard them
        WHISPER_HALLUCINATIONS = {
            "thank you", "thanks for watching", "thanks for watching!",
            "you", ".", "the", "bye", "goodbye",
            "subtitles by", "[music]", "(music)"
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.post(self.url, headers=headers, files=files, data=data)
                if response.status_code == 200:
                    result = response.json()
                    text = result.get("text", "").strip()
                    # Discard known Whisper silence/hallucination artifacts
                    if text.lower() in WHISPER_HALLUCINATIONS or len(text) < 2:
                        return ""
                    return text
                else:
                    return f"[Groq STT Error: {response.status_code}]"
            except Exception as e:
                return f"[Groq STT Exception: {str(e)}]"


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
