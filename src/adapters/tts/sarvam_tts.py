import base64
import os
import httpx
from typing import AsyncGenerator, Optional
from src.adapters.tts.base import BaseTTS

class SarvamTTS(BaseTTS):
    """
    Sarvam AI (Bulbul) TTS Adapter.
    Specialized for authentic Indian accents, local nuances, and regional languages.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        speaker: str = "meera",
        language_code: str = "hi-IN",
        model: str = "bulbul:v1"
    ):
        super().__init__(voice=speaker)
        self.api_key = api_key or os.getenv("SARVAM_API_KEY", "")
        self.speaker = speaker
        self.language_code = language_code
        self.model = model
        self.url = "https://api.sarvam.ai/text-to-speech"

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        audio_bytes = await self.synthesize(text)
        if audio_bytes:
            # Yield in 2KB chunks to simulate streaming transport
            chunk_size = 2048
            for i in range(0, len(audio_bytes), chunk_size):
                yield audio_bytes[i:i + chunk_size]

    async def synthesize(self, text: str) -> bytes:
        if not text.strip() or not self.api_key:
            return b""

        headers = {
            "api-subscription-key": self.api_key,
            "Content-Type": "application/json"
        }
        body = {
            "inputs": [text],
            "target_language_code": self.language_code,
            "speaker": self.speaker,
            "model": self.model
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                response = await client.post(self.url, headers=headers, json=body)
                if response.status_code == 200:
                    data = response.json()
                    audios = data.get("audios", [])
                    if audios:
                        return base64.b64decode(audios[0])
            except Exception:
                pass
        return b""
