from typing import Callable, Optional
from src.adapters.stt.base import BaseSTT

class MockSTT(BaseSTT):
    """
    Mock STT Adapter for unit tests and local development without API keys.
    """
    def __init__(
        self,
        simulated_text: str = "Kya aap CRM demo provide karte hain?",
        language: str = "hi",
        sample_rate: int = 16000
    ):
        super().__init__(language=language, sample_rate=sample_rate)
        self.simulated_text = simulated_text
        self.on_transcript_callback: Optional[Callable[[str, bool], None]] = None
        self.is_connected = False

    async def transcribe(self, audio_data: bytes) -> str:
        return self.simulated_text

    async def connect_stream(self, on_transcript_callback: Callable[[str, bool], None]):
        self.on_transcript_callback = on_transcript_callback
        self.is_connected = True

    async def push_audio(self, chunk: bytes):
        # When audio chunk is received in stream mode, trigger the callback
        if self.on_transcript_callback and self.simulated_text:
            self.on_transcript_callback(self.simulated_text, True)

    async def close(self):
        self.is_connected = False
