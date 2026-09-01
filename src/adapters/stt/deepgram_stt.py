import os
import json
import asyncio
import logging
from typing import Callable, Optional
import websockets
from src.adapters.stt.base import BaseSTT

logger = logging.getLogger(__name__)

class DeepgramSTT(BaseSTT):
    """
    Deepgram Real-Time WebSocket STT Adapter.
    Provides ultra-low latency transcription via WebSockets.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "nova-2",
        language: str = "hi",
        sample_rate: int = 16000
    ):
        super().__init__(language=language, sample_rate=sample_rate)
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY", "")
        if not self.api_key:
            env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), ".env")
            if os.path.exists(env_path):
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        if line.startswith("DEEPGRAM_API_KEY="):
                            self.api_key = line.split("=", 1)[1].strip()
        self.model = model
        
        self.ws_connection: Optional[websockets.WebSocketClientProtocol] = None
        self.on_transcript_callback: Optional[Callable[[str, bool], None]] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._is_closing = False
        self.is_streaming_supported = True

    async def transcribe(self, audio_data: bytes) -> str:
        """Fallback for batch processing if needed."""
        # For simplicity, returning empty string as this adapter is meant for streaming
        return ""

    async def connect_stream(self, on_transcript_callback: Callable[[str, bool], None]):
        """Connect to Deepgram WebSocket for real-time streaming."""
        if not self.api_key:
            logger.warning("[DeepgramSTT] No API key provided, cannot connect.")
            return

        self.on_transcript_callback = on_transcript_callback
        
        url = (
            f"wss://api.deepgram.com/v1/listen?"
            f"model={self.model}&language={self.language}&"
            f"encoding=linear16&sample_rate={self.sample_rate}&"
            f"endpointing=300&smart_format=true"
        )
        
        headers = {
            "Authorization": f"Token {self.api_key}"
        }

        try:
            self._is_closing = False
            self.ws_connection = await websockets.connect(url, extra_headers=headers)
            self._receive_task = asyncio.create_task(self._receive_loop())
            logger.info("[DeepgramSTT] Connected to streaming WebSocket.")
        except Exception as e:
            logger.error(f"[DeepgramSTT] Connection failed: {e}")

    async def _receive_loop(self):
        try:
            while not self._is_closing and self.ws_connection:
                message = await self.ws_connection.recv()
                if isinstance(message, str):
                    data = json.loads(message)
                    if data.get("type") == "Results":
                        is_final = data.get("is_final", False)
                        alternatives = data.get("channel", {}).get("alternatives", [])
                        if alternatives:
                            transcript = alternatives[0].get("transcript", "").strip()
                            if transcript and self.on_transcript_callback:
                                self.on_transcript_callback(transcript, is_final)
        except websockets.exceptions.ConnectionClosed:
            logger.info("[DeepgramSTT] WebSocket connection closed.")
        except Exception as e:
            if not self._is_closing:
                logger.error(f"[DeepgramSTT] Receive loop error: {e}")
        finally:
            await self.close()

    async def push_audio(self, chunk: bytes):
        """Send raw PCM chunk to Deepgram."""
        if self.ws_connection and not self._is_closing:
            try:
                await self.ws_connection.send(chunk)
            except Exception as e:
                logger.error(f"[DeepgramSTT] Error pushing audio: {e}")

    async def close(self):
        """Clean up connection."""
        self._is_closing = True
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
        if self.ws_connection:
            try:
                # Send empty binary payload to close Deepgram stream cleanly (CloseStream)
                await self.ws_connection.send(b"")
                await self.ws_connection.close()
            except Exception:
                pass
            self.ws_connection = None
