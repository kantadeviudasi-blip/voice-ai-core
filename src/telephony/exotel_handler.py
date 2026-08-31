import json
import base64
from typing import Optional, Callable
from src.telephony.base_telephony import BaseTelephonyBridge

class ExotelStreamHandler:
    """
    Exotel Voice Stream WebSocket Protocol Handler.
    Handles inbound and outbound audio streaming over Exotel's Indian telephony gateway.
    """
    def __init__(self, on_audio_chunk: Callable[[bytes], None], on_stop: Optional[Callable[[], None]] = None):
        self.on_audio_chunk = on_audio_chunk
        self.on_stop = on_stop
        self.stream_sid: Optional[str] = None
        self.call_sid: Optional[str] = None

    def handle_message(self, message_str: str) -> Optional[dict]:
        """
        Parses incoming JSON message from Exotel WebSocket.
        """
        try:
            data = json.loads(message_str)
            event = data.get("event")

            if event == "start":
                self.stream_sid = data.get("stream_sid")
                self.call_sid = data.get("start", {}).get("call_sid")
                return {"type": "start", "call_sid": self.call_sid, "stream_sid": self.stream_sid}

            elif event == "media":
                payload_b64 = data.get("media", {}).get("payload", "")
                if payload_b64:
                    mulaw_chunk = base64.b64decode(payload_b64)
                    # Exotel uses standard μ-law 8kHz audio
                    pcm16k = BaseTelephonyBridge.mulaw_to_pcm16(mulaw_chunk)
                    self.on_audio_chunk(pcm16k)
                return {"type": "media"}

            elif event == "stop":
                if self.on_stop:
                    self.on_stop()
                return {"type": "stop"}

        except Exception as e:
            return {"type": "error", "error": str(e)}
        return None

    def create_media_response(self, pcm16_chunk: bytes) -> str:
        """
        Encodes outbound PCM audio to μ-law and formats Exotel media frame.
        """
        mulaw = BaseTelephonyBridge.pcm16_to_mulaw(pcm16_chunk)
        payload_b64 = base64.b64encode(mulaw).decode("utf-8")
        msg = {
            "event": "media",
            "stream_sid": self.stream_sid,
            "media": {
                "payload": payload_b64
            }
        }
        return json.dumps(msg)

    def create_clear_response(self) -> str:
        """
        Flushes playback buffer when user interrupts (Barge-in).
        """
        msg = {
            "event": "clear",
            "stream_sid": self.stream_sid
        }
        return json.dumps(msg)
