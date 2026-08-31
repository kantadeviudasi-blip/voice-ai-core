import os
import httpx
from typing import AsyncGenerator, Optional
from src.adapters.tts.base import BaseTTS


class CartesiaTTS(BaseTTS):
    """
    Cartesia Sonic TTS Adapter.
    Ultra-low-latency streaming TTS with multilingual support (including Indian English/Hindi).
    Uses the Cartesia /tts/bytes endpoint for chunked PCM streaming.
    Docs: https://docs.cartesia.ai/api-reference/tts/bytes
    """

    API_URL = "https://api.cartesia.ai/tts/bytes"
    API_VERSION = "2024-06-10"

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_id: str = "sonic-multilingual",
        voice_id: str = "694f120f-baa9-4938-8996-9b603e30dceb",
        output_format: str = "pcm_16000",   # 16-bit PCM @ 16kHz – pipeline native
        language: str = "hi",
    ):
        super().__init__(voice=voice_id)
        self.api_key = api_key or os.getenv("CARTESIA_API_KEY", "")
        self.model_id = model_id
        self.voice_id = voice_id
        self.output_format = output_format
        self.language = language

    def _build_request_body(self, text: str) -> dict:
        return {
            "model_id": self.model_id,
            "transcript": text,
            "voice": {
                "mode": "id",
                "id": self.voice_id,
            },
            "output_format": {
                "container": "raw",
                "encoding": "pcm_s16le",
                "sample_rate": 16000,
            },
            "language": self.language,
        }

    def _build_headers(self) -> dict:
        return {
            "X-API-Key": self.api_key,
            "Cartesia-Version": self.API_VERSION,
            "Content-Type": "application/json",
        }

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """Stream raw PCM audio chunks from Cartesia in real-time."""
        if not text.strip() or not self.api_key:
            return

        async with httpx.AsyncClient(timeout=15.0) as client:
            try:
                async with client.stream(
                    "POST",
                    self.API_URL,
                    headers=self._build_headers(),
                    json=self._build_request_body(text),
                ) as response:
                    if response.status_code == 200:
                        async for chunk in response.aiter_bytes(chunk_size=4096):
                            if chunk:
                                yield chunk
                    else:
                        # Log but don't crash — fallback to silence
                        error = await response.aread()
                        print(f"[CartesiaTTS] Error {response.status_code}: {error[:200]}")
            except Exception as e:
                print(f"[CartesiaTTS] Stream error: {e}")

    async def synthesize(self, text: str) -> bytes:
        """Collect full audio response as a single bytes object."""
        chunks = []
        async for chunk in self.synthesize_stream(text):
            chunks.append(chunk)
        return b"".join(chunks)
