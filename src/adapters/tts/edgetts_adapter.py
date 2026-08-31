import io
from typing import AsyncGenerator
import edge_tts
from src.adapters.tts.base import BaseTTS

class EdgeTTSAdapter(BaseTTS):
    """
    Microsoft EdgeTTS Adapter.
    Zero-Cost, High-Quality Neural Voices for Hindi, Indian English, and Indian Vernacular.
    """
    def __init__(self, voice: str = "hi-IN-SwaraNeural", rate: str = "+10%", pitch: str = "+0Hz"):
        super().__init__(voice=voice, rate=rate, pitch=pitch)

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        if not text.strip():
            return
        
        communicate = edge_tts.Communicate(text, self.voice, rate=self.rate, pitch=self.pitch)
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                yield chunk["data"]

    async def synthesize(self, text: str) -> bytes:
        if not text.strip():
            return b""
        buffer = bytearray()
        async for chunk in self.synthesize_stream(text):
            buffer.extend(chunk)
        return bytes(buffer)
