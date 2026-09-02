import json
import os
import httpx
from typing import AsyncGenerator, List, Dict, Any, Optional
from src.adapters.llm.base import BaseLLM

class GeminiLLM(BaseLLM):
    """
    Google Gemini 2.0 Flash Streaming Adapter.
    Ultra-low cost ($0.10/M tokens), multilingual Indian vernacular mastery.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gemini-3.8-flash",
        temperature: float = 0.3,
        max_tokens: int = 150
    ):
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        self.api_key = api_key or os.getenv("GEMINI_API_KEY", "")

    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[str, None]:
        if not self.api_key:
            yield "Namaste! Main aapki sahayata ke liye tayar hoon."
            return

        url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:streamGenerateContent?alt=sse&key={self.api_key}"
        
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})

        body = {
            "system_instruction": {"parts": [{"text": system_prompt}]},
            "contents": contents if contents else [{"role": "user", "parts": [{"text": "Hello"}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
            }
        }

        async with httpx.AsyncClient(timeout=10.0) as client:
            try:
                async with client.stream("POST", url, json=body) as response:
                    if response.status_code != 200:
                        yield f"Error from Gemini ({response.status_code})"
                        return

                    async for line in response.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:].strip()
                            if not data_str:
                                continue
                            try:
                                chunk = json.loads(data_str)
                                candidates = chunk.get("candidates", [{}])
                                if candidates:
                                    parts = candidates[0].get("content", {}).get("parts", [{}])
                                    for p in parts:
                                        text = p.get("text", "")
                                        if text:
                                            yield text
                            except json.JSONDecodeError:
                                continue
            except Exception as e:
                yield f"[Gemini Error: {str(e)}]"

    async def generate(self, messages: List[Dict[str, str]], system_prompt: str) -> str:
        tokens = []
        async for t in self.generate_stream(messages, system_prompt):
            tokens.append(t)
        return "".join(tokens)
