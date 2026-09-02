import json
import os
import asyncio
import logging
import httpx
from typing import AsyncGenerator, List, Dict, Any, Optional
from src.adapters.llm.base import BaseLLM

logger = logging.getLogger(__name__)

try:
    from google import genai
    from google.genai import types
    HAS_GOOGLE_GENAI = True
except ImportError:
    HAS_GOOGLE_GENAI = False

class GeminiLLM(BaseLLM):
    """
    Google Gemini 3.8 Flash Adapter with multi-model failover & streaming.
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
        self.client = None
        if HAS_GOOGLE_GENAI and self.api_key:
            try:
                self.client = genai.Client(api_key=self.api_key)
            except Exception:
                self.client = None

    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[str, None]:
        if not self.api_key:
            yield "Namaste! Main aapki sahayata ke liye tayar hoon."
            return

        models_to_try = [self.model]
        for fallback in ["gemini-3.7-flash", "gemini-3.6-flash", "gemini-2.5-flash-lite"]:
            if fallback not in models_to_try:
                models_to_try.append(fallback)

        prompt = messages[-1].get("content", "Hello") if messages else "Hello"

        # 1. Primary path: Official Google GenAI SDK
        if self.client:
            for active_model in models_to_try:
                try:
                    chat = self.client.aio.chats.create(
                        model=active_model,
                        config=types.GenerateContentConfig(
                            system_instruction=system_prompt,
                            temperature=self.temperature,
                            max_output_tokens=self.max_tokens,
                        )
                    )
                    # Attempt true streaming first
                    emitted = False
                    try:
                        stream = await chat.send_message_stream(prompt)
                        async for chunk in stream:
                            if chunk.text:
                                emitted = True
                                yield chunk.text
                        if emitted:
                            return
                    except Exception as stream_err:
                        logger.warning(f"[GeminiLLM] Streaming attempt with {active_model} failed ({stream_err}), falling back to direct generation.")
                    
                    # Fall back to single-turn chat if streaming had a temporary spike
                    res = await chat.send_message(prompt)
                    if res.text:
                        words = res.text.split(" ")
                        for i, w in enumerate(words):
                            token = w + (" " if i < len(words) - 1 else "")
                            yield token
                        return
                except Exception as model_err:
                    logger.warning(f"[GeminiLLM] Model {active_model} failed: {model_err}")
                    continue

        # 2. Secondary path: Direct REST SSE via httpx
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
            for active_model in models_to_try:
                url = f"https://generativelanguage.googleapis.com/v1beta/models/{active_model}:streamGenerateContent?alt=sse&key={self.api_key}"
                try:
                    async with client.stream("POST", url, json=body) as response:
                        if response.status_code != 200:
                            continue

                        tokens_emitted = False
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
                                                tokens_emitted = True
                                                yield text
                                except json.JSONDecodeError:
                                    continue
                        if tokens_emitted:
                            return
                except Exception:
                    continue

        # 3. Graceful human holding response if all free-tier quotas or endpoints are temporarily busy
        yield "Ji, main sun rahi hoon. Kripya apna sawaal ek baar phir bataiye."

    async def generate(self, messages: List[Dict[str, str]], system_prompt: str) -> str:
        tokens = []
        async for t in self.generate_stream(messages, system_prompt):
            tokens.append(t)
        return "".join(tokens)
