import os
import json
import asyncio
import logging
from typing import Callable, Optional
import websockets
from src.adapters.tts.base import BaseTTS

logger = logging.getLogger(__name__)

class DeepgramTTS(BaseTTS):
    """
    Deepgram Aura Text-to-Speech Adapter via WebSockets.
    """
    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "aura-2-asteria-en",
        sample_rate: int = 24000
    ):
        super().__init__(voice=model)
        self.sample_rate = sample_rate
        self.api_key = api_key or os.getenv("DEEPGRAM_API_KEY", "")
        if not self.api_key:
            try:
                from dotenv import load_dotenv
                load_dotenv()
                self.api_key = os.getenv("DEEPGRAM_API_KEY", "")
            except Exception:
                pass
        self.model = model
        
        self.ws_connection: Optional[websockets.WebSocketClientProtocol] = None
        self.on_audio_callback: Optional[Callable[[bytes], None]] = None
        self._receive_task: Optional[asyncio.Task] = None
        self._is_closing = False
        
    async def synthesize(self, text: str) -> bytes:
        """Batch synthesis is not implemented for this adapter. Use synthesize_stream."""
        return b""

    async def connect_stream(self, on_audio_callback: Callable[[bytes], None]):
        """Connect to Deepgram TTS WebSocket."""
        if not self.api_key:
            logger.warning("[DeepgramTTS] No API key provided.")
            return

        self.on_audio_callback = on_audio_callback
        url = f"wss://api.deepgram.com/v1/speak?model={self.model}&encoding=linear16&sample_rate={self.sample_rate}"
        headers = {"Authorization": f"Token {self.api_key}"}

        try:
            self._is_closing = False
            self.ws_connection = await websockets.connect(url, additional_headers=headers)
            self._receive_task = asyncio.create_task(self._receive_loop())
            logger.info("[DeepgramTTS] Connected to streaming WebSocket.")
        except Exception as e:
            logger.error(f"[DeepgramTTS] Connection failed: {e}")

    async def _receive_loop(self):
        try:
            while not self._is_closing and self.ws_connection:
                message = await self.ws_connection.recv()
                if isinstance(message, bytes):
                    if self.on_audio_callback:
                        self.on_audio_callback(message)
                else:
                    # Ignore control messages (e.g. Warning, Metadata, Flush)
                    pass
        except websockets.exceptions.ConnectionClosed:
            logger.info("[DeepgramTTS] WebSocket closed.")
        except Exception as e:
            if not self._is_closing:
                logger.error(f"[DeepgramTTS] Receive loop error: {e}")
        finally:
            await self.close()

    async def synthesize_stream(self, text: str):
        """Send text chunk to Deepgram."""
        if not text.strip() or not self.ws_connection or self._is_closing:
            return

        try:
            payload = json.dumps({"type": "Speak", "text": text})
            await self.ws_connection.send(payload)
            # Flush after every chunk to force audio generation immediately
            await self.ws_connection.send(json.dumps({"type": "Flush"}))
        except Exception as e:
            logger.error(f"[DeepgramTTS] Error sending text: {e}")

    async def flush(self):
        """Force flush of current audio buffer."""
        if self.ws_connection and not self._is_closing:
            try:
                await self.ws_connection.send(json.dumps({"type": "Flush"}))
            except Exception:
                pass

    async def close(self):
        """Clean up connection."""
        self._is_closing = True
        if self._receive_task and not self._receive_task.done():
            self._receive_task.cancel()
        if self.ws_connection:
            try:
                await self.ws_connection.send(json.dumps({"type": "Close"}))
                await self.ws_connection.close()
            except Exception:
                pass
            self.ws_connection = None
