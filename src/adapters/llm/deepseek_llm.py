import json
import os
import httpx
from typing import AsyncGenerator, List, Dict, Any, Optional
from src.adapters.llm.base import BaseLLM

class DeepSeekLLM(BaseLLM):
    """
    DeepSeek V3 / R1 LLM Adapter.
    Unbeatable reasoning and objection-handling at minimal cost.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: str = "https://api.deepseek.com",
        model: str = "deepseek-chat",
        temperature: float = 0.3,
        max_tokens: int = 150
    ):
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY", "")
        self.base_url = f"{base_url.rstrip('/')}/chat/completions"

    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[str, None]:
        payload_messages = [{"role": "system", "content": system_prompt}] + messages
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": payload_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                async with client.stream("POST", self.base_url, headers=headers, json=body) as response:
                    if response.status_code != 200:
                        yield f"Error from DeepSeek ({response.status_code})"
                        return

                    async for line in response.aiter_lines():
                        if line.startswith("data: ") and line != "data: [DONE]":
                            data_str = line[6:].strip()
                            if not data_str:
                                continue
                            try:
                                chunk = json.loads(data_str)
                                delta = chunk.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                continue
            except Exception as e:
                yield f"[DeepSeek Error: {str(e)}]"

    async def generate(self, messages: List[Dict[str, str]], system_prompt: str) -> str:
        tokens = []
        async for t in self.generate_stream(messages, system_prompt):
            tokens.append(t)
        return "".join(tokens)
