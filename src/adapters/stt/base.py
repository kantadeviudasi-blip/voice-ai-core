from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional, Callable

class BaseSTT(ABC):
    """
    Abstract Base Interface for Speech-To-Text Adapters.
    """
    def __init__(self, language: str = "en-IN", sample_rate: int = 16000):
        self.language = language
        self.sample_rate = sample_rate

    @abstractmethod
    async def transcribe(self, audio_data: bytes) -> str:
        """
        Transcribe complete audio chunk/utterance to text.
        """
        pass

    @abstractmethod
    async def connect_stream(self, on_transcript_callback: Callable[[str, bool], None]):
        """
        Connect to real-time streaming STT (if supported by provider).
        """
        pass

    @abstractmethod
    async def push_audio(self, chunk: bytes):
        """
        Push raw audio chunk to stream.
        """
        pass

    @abstractmethod
    async def close(self):
        """Clean up connection."""
        pass
