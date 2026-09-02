import asyncio
from typing import AsyncGenerator, List, Dict, Any, Optional
from src.adapters.llm.base import BaseLLM

class MockLLM(BaseLLM):
    """
    Mock LLM Brain Adapter for testing and local zero-cost emulation.
    """
    def __init__(
        self,
        model: str = "mock-model",
        temperature: float = 0.3,
        max_tokens: int = 150,
        simulated_response: str = "Haan ji, hum CRM automation aur live demo provide karte hain."
    ):
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        self.simulated_response = simulated_response

    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[str, None]:
        # Yield word by word with minimal delay to simulate token streaming
        words = self.simulated_response.split(" ")
        for i, word in enumerate(words):
            token = word + (" " if i < len(words) - 1 else "")
            await asyncio.sleep(0.01)
            yield token

    async def generate(self, messages: List[Dict[str, str]], system_prompt: str) -> str:
        return self.simulated_response
