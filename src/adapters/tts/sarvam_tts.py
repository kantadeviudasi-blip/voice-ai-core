import os
import base64
import logging
from typing import AsyncGenerator, Optional
import httpx
from src.adapters.tts.base import BaseTTS

logger = logging.getLogger(__name__)

class SarvamTTS(BaseTTS):
    """
    Sarvam AI Text-to-Speech Adapter (Bulbul:v3).
    Supports high-quality Indian languages, natural female (girl) voices (e.g. 'ritu', 'priya'),
    and dynamic sample rate configuration (8000 Hz, 16000 Hz, 24000 Hz).
    """

    SUPPORTED_SAMPLE_RATES = (8000, 16000, 24000)
    FEMALE_SPEAKERS = [
        "ritu", "priya", "neha", "pooja", "simran", 
        "kavya", "ishita", "shreya", "roopa", "tanya", 
        "shruti", "suhani", "kavitha", "rupali", "niharika"
    ]

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "bulbul:v3",
        speaker: str = "ritu",
        sample_rate: int = 8000,
        target_language_code: str = "hi-IN",
        pace: float = 1.0,
        pitch: float = 0.0,
        loudness: float = 1.0,
        enable_preprocessing: bool = True
    ):
        super().__init__(voice=speaker)
        self.api_key = api_key or os.getenv("SARVAM_API_KEY", "")
        if not self.api_key:
            try:
                from dotenv import load_dotenv
                load_dotenv()
                self.api_key = os.getenv("SARVAM_API_KEY", "")
            except Exception:
                pass

        self.model = model
        self.speaker = speaker
        self.sample_rate = sample_rate if sample_rate in self.SUPPORTED_SAMPLE_RATES else 8000
        self.target_language_code = target_language_code
        self.pace = pace
        self.pitch = pitch
        self.loudness = loudness
        self.enable_preprocessing = enable_preprocessing
        self.api_url = "https://api.sarvam.ai/text-to-speech"
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(timeout=15.0)
        return self._client

    async def synthesize(self, text: str) -> bytes:
        """
        Synthesizes complete text to WAV audio bytes via Sarvam AI API.
        """
        if not text or not text.strip():
            return b""

        if not self.api_key:
            logger.error("[SarvamTTS] Missing SARVAM_API_KEY")
            return b""

        headers = {
            "api-subscription-key": self.api_key,
            "Content-Type": "application/json"
        }

        payload = {
            "inputs": [text.strip()],
            "target_language_code": self.target_language_code,
            "speaker": self.speaker,
            "pitch": self.pitch,
            "pace": self.pace,
            "loudness": self.loudness,
            "speech_sample_rate": self.sample_rate,
            "enable_preprocessing": self.enable_preprocessing,
            "model": self.model
        }

        try:
            client = await self._get_client()
            response = await client.post(self.api_url, headers=headers, json=payload)
            if response.status_code == 200:
                data = response.json()
                audios = data.get("audios", [])
                if audios:
                    return base64.b64decode(audios[0])
                logger.warning("[SarvamTTS] Response received but no audio found in payload.")
                return b""
            else:
                logger.error(f"[SarvamTTS] API Error {response.status_code}: {response.text}")
                return b""
        except Exception as e:
            logger.error(f"[SarvamTTS] Request failed: {e}")
            return b""

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Yields synthesized audio bytes for the given text segment.
        """
        audio_bytes = await self.synthesize(text)
        if audio_bytes:
            yield audio_bytes

    async def close(self):
        """Clean up HTTP client connection pool."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()
            self._client = None
