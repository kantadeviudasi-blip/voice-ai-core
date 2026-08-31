import asyncio
from typing import Callable
from src.adapters.stt.base import BaseSTT

class MockSTT(BaseSTT):
    """
    Mock STT Adapter for local development, sandbox testing, and latency benchmarking.
    """
    def __init__(self, language: str = "hi-IN", sample_rate: int = 16000, simulated_text: str = "Haan main property dekhna chahta hoon."):
        super().__init__(language=language, sample_rate=sample_rate)
        self.simulated_text = simulated_text

    async def transcribe(self, audio_data: bytes) -> str:
        # Simulate network & inference latency ~80ms
        await asyncio.sleep(0.08)
        return self.simulated_text

    async def connect_stream(self, on_transcript_callback: Callable[[str, bool], None]):
        pass

    async def push_audio(self, chunk: bytes):
        pass

    async def close(self):
        pass
