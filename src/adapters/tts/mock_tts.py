import asyncio
from typing import AsyncGenerator
from src.adapters.tts.base import BaseTTS

class MockTTS(BaseTTS):
    """
    Mock TTS Adapter generating synthetic PCM silence/sine tones for rapid local testing.
    """
    def __init__(self, voice: str = "mock-female-hindi", rate: str = "+0%", pitch: str = "+0Hz"):
        super().__init__(voice=voice, rate=rate, pitch=pitch)

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        # Generate 10 small chunks simulating audio arrival
        for _ in range(5):
            await asyncio.sleep(0.03)
            yield b"\x00\x00" * 320 # 20ms of silence/PCM

    async def synthesize(self, text: str) -> bytes:
        return b"\x00\x00" * 1600
