import json
import base64
from typing import Optional, Callable
from src.telephony.base_telephony import BaseTelephonyBridge

class PlivoStreamHandler:
    """
    Plivo Bidirectional Audio Stream WebSocket Protocol Handler.
    """
    def __init__(self, on_audio_chunk: Callable[[bytes], None], on_stop: Optional[Callable[[], None]] = None):
        self.on_audio_chunk = on_audio_chunk
        self.on_stop = on_stop
        self.stream_id: Optional[str] = None

    def handle_message(self, message_str: str) -> Optional[dict]:
        try:
            data = json.loads(message_str)
            event = data.get("event")

            if event == "start":
                self.stream_id = data.get("stream_id")
                return {"type": "start", "stream_id": self.stream_id}

            elif event == "media":
                payload_b64 = data.get("media", {}).get("payload", "")
                if payload_b64:
                    mulaw_chunk = base64.b64decode(payload_b64)
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
        mulaw = BaseTelephonyBridge.pcm16_to_mulaw(pcm16_chunk)
        payload_b64 = base64.b64encode(mulaw).decode("utf-8")
        msg = {
            "event": "playAudio",
            "media": {
                "contentType": "audio/x-mulaw",
                "sampleRate": 8000,
                "payload": payload_b64
            }
        }
        return json.dumps(msg)

    def create_clear_response(self) -> str:
        msg = {"event": "clearAudio"}
        return json.dumps(msg)
