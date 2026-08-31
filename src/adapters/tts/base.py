from abc import ABC, abstractmethod
from typing import AsyncGenerator

class BaseTTS(ABC):
    """
    Abstract Base Interface for Text-To-Speech Adapters.
    """
    def __init__(self, voice: str, rate: str = "+0%", pitch: str = "+0Hz"):
        self.voice = voice
        self.rate = rate
        self.pitch = pitch

    @abstractmethod
    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        """
        Stream audio bytes as they become available.
        """
        pass

    @abstractmethod
    async def synthesize(self, text: str) -> bytes:
        """
        Synthesize entire text to complete audio bytes.
        """
        pass
