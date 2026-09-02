import asyncio
from typing import AsyncGenerator
from src.adapters.tts.base import BaseTTS

class MockTTS(BaseTTS):
    """
    Mock TTS Adapter for unit testing and local development without cloud TTS costs.
    """
    def __init__(self, voice: str = "mock-voice", rate: str = "+0%", pitch: str = "+0Hz"):
        super().__init__(voice=voice, rate=rate, pitch=pitch)

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        # Yield mock 16-bit linear PCM audio chunk
        for _ in range(3):
            await asyncio.sleep(0.01)
            yield b"\x10\x20" * 160  # 320 bytes = 10ms of 16kHz 16-bit PCM

    async def synthesize(self, text: str) -> bytes:
        return b"\x10\x20" * 480
