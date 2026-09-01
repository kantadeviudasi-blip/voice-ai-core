import json
import os
import httpx
from typing import AsyncGenerator, List, Dict, Any, Optional
from src.adapters.llm.base import BaseLLM
from src.core.config import config
from src.adapters.llm.gemini_llm import GeminiLLM

# Ordered fallback chain — newest/fastest first
# NOTE: qwen3.x are THINKING/REASONING models — they output internal <think> blocks
# that leak into TTS. Use NON-THINKING models for voice telecalling.
GROQ_MODEL_FALLBACK_CHAIN = [
    "llama-3.3-70b-versatile",     # ✅ Best quality, natural Hindi/Hinglish, non-thinking
    "llama-3.1-8b-instant",        # ✅ Ultra-fast fallback, non-thinking
    "llama-3.2-1b-preview",        # Fallback for free tiers
    "llama-3.2-3b-preview",
    "gemma2-9b-it",
    "mixtral-8x7b-32768",
    "llama3-70b-8192",             # Secondary fallback
    "llama3-8b-8192",              # Emergency fallback
]

class GroqLLM(BaseLLM):
    """
    Groq Cloud LLM Adapter with automatic model fallback.
    Tries models in order until one succeeds — handles deprecated/unavailable models gracefully.
    Logs exact Groq error messages to server console for easy debugging.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "llama-3.3-70b-versatile",   # ✅ Non-thinking fast model — no thought leakage
        temperature: float = 0.3,
        max_tokens: int = 150
    ):
        super().__init__(model=model, temperature=temperature, max_tokens=max_tokens)
        self.api_key = api_key or os.getenv("GROQ_API_KEY", "")
        if not self.api_key:
            env_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
                ".env"
            )
            if os.path.exists(env_path):
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("GROQ_API_KEY="):
                            self.api_key = line.split("=", 1)[1].strip()
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self._working_model: Optional[str] = None  # Cached model that worked last time

    def _get_model_chain(self) -> List[str]:
        """Build fallback chain starting from the configured model."""
        chain = [self.model]
        for m in GROQ_MODEL_FALLBACK_CHAIN:
            if m not in chain:
                chain.append(m)
        return chain

    async def _stream_model(
        self,
        model: str,
        payload_messages: List[Dict],
        client: httpx.AsyncClient
    ) -> AsyncGenerator[str, None]:
        """Stream tokens from a specific model. Raises on non-200."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        body: Dict[str, Any] = {
            "model": model,
            "messages": payload_messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": True
        }
        async with client.stream("POST", self.base_url, headers=headers, json=body) as response:
            if response.status_code != 200:
                raw = await response.aread()
                err_text = raw.decode("utf-8", errors="ignore")
                try:
                    err_msg = json.loads(err_text).get("error", {}).get("message", err_text[:150])
                except Exception:
                    err_msg = err_text[:150]
                print(f"[GroqLLM] '{model}' HTTP {response.status_code}: {err_msg}")
                raise ValueError(f"HTTP {response.status_code}")

            async for line in response.aiter_lines():
                if line.startswith("data: ") and line != "data: [DONE]":
                    data_str = line[6:].strip()
                    if not data_str:
                        continue
                    try:
                        chunk = json.loads(data_str)
                        delta = chunk.get("choices", [{}])[0].get("delta", {})
                        # Discard internal reasoning_content or thinking tags
                        if "reasoning_content" in delta and not delta.get("content"):
                            continue
                        content = delta.get("content", "")
                        if content:
                            yield content
                    except json.JSONDecodeError:
                        continue


    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        system_prompt: str,
        tools: Optional[List[Dict[str, Any]]] = None
    ) -> AsyncGenerator[str, None]:
        payload_messages = [{"role": "system", "content": system_prompt}] + messages

        # Build deduplicated model chain — cached working model goes first
        chain = ([self._working_model] + self._get_model_chain()) if self._working_model else self._get_model_chain()
        seen: set = set()
        model_chain = [m for m in chain if not (m in seen or seen.add(m))]

        async with httpx.AsyncClient(timeout=15.0) as client:
            for model in model_chain:
                try:
                    got_token = False
                    async for token in self._stream_model(model, payload_messages, client):
                        if not got_token:
                            got_token = True
                            self._working_model = model
                            if model != self.model:
                                print(f"[GroqLLM] Fallback OK: using '{model}'")
                        yield token
                    return  # Done
                except Exception:
                    continue  # Try next model

            # All models failed
            print(f"[GroqLLM] All Groq models failed. Attempting cross-provider fallback to Gemini...")
            gemini_key = config.llm.gemini.api_key
            if gemini_key:
                try:
                    fallback_llm = GeminiLLM(
                        api_key=gemini_key, 
                        model=config.llm.gemini.model,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    async for token in fallback_llm.generate_stream(messages, system_prompt):
                        yield token
                    return
                except Exception as e:
                    print(f"[GroqLLM] Gemini fallback also failed: {e}")
                    pass
            
            yield "[Service unavailable — check API keys or try again]"

    async def generate(self, messages: List[Dict[str, str]], system_prompt: str) -> str:
        collected = []
        async for token in self.generate_stream(messages, system_prompt):
            collected.append(token)
        return "".join(collected)
