import asyncio
from typing import AsyncGenerator, List, Dict, Any, Optional
from src.adapters.llm.base import BaseLLM

class MockLLM(BaseLLM):
    """
    Mock LLM Adapter for ultra-fast local testing and benchmarking without external API keys.
    """
    def __init__(self, model: str = "mock-telecaller-v1", temperature: float = 0.3, max_tokens: int = 150):
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)

    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[str, None]:
        # Realistic Indian telecaller conversational response chunks
        chunks = [
            "Namaste sir! ",
            "Humare paas 2 aur 3 BHK ",
            "luxury flats available hain, ",
            "starting price sirf 45 Lakhs hai. ",
            "Kya aap Saturday ko site visit ke liye comfortable hain?"
        ]
        for chunk in chunks:
            await asyncio.sleep(0.04) # ~40ms token delay
            yield chunk

    async def generate(self, messages: List[Dict[str, str]], system_prompt: str) -> str:
        return "Namaste sir! Humare paas 2 aur 3 BHK luxury flats available hain. Kya aap Saturday ko site visit ke liye comfortable hain?"
