import time
import uuid
from enum import Enum
from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field

class SessionState(str, Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    ENDED = "ended"

class TurnMetric(BaseModel):
    turn_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    user_transcript: str = ""
    agent_response: str = ""
    vad_latency_ms: float = 0.0
    stt_latency_ms: float = 0.0
    ttft_ms: float = 0.0              # Time to first LLM token
    tts_first_chunk_ms: float = 0.0   # Time until first audio chunk is ready
    total_latency_ms: float = 0.0     # End of user speech to start of audio playback
    interrupted: bool = False

class CallSession:
    """
    Manages the lifecycle, state, history, and metrics of a single call session.
    """
    def __init__(self, session_id: Optional[str] = None, tenant_id: str = "default", caller_number: str = "anonymous"):
        self.session_id = session_id or str(uuid.uuid4())
        self.tenant_id = tenant_id
        self.caller_number = caller_number
        self.created_at = time.time()
        self.state = SessionState.IDLE
        self.messages: List[Dict[str, str]] = []
        self.metrics: List[TurnMetric] = []
        self.current_turn: Optional[TurnMetric] = None
        self.metadata: Dict[str, Any] = {}
        self.active_stream_id: Optional[str] = None
        self.is_interrupted = False

    def start_turn(self) -> TurnMetric:
        self.current_turn = TurnMetric()
        self.is_interrupted = False
        return self.current_turn

    def complete_turn(self, user_text: str, agent_text: str):
        if self.current_turn:
            self.current_turn.user_transcript = user_text
            self.current_turn.agent_response = agent_text
            self.current_turn.interrupted = self.is_interrupted
            self.metrics.append(self.current_turn)
            self.current_turn = None

        if user_text:
            self.messages.append({"role": "user", "content": user_text})
        if agent_text:
            self.messages.append({"role": "assistant", "content": agent_text})

    def interrupt(self):
        """Mark session as interrupted (Barge-in triggered)."""
        self.is_interrupted = True
        self.state = SessionState.INTERRUPTED
        if self.current_turn:
            self.current_turn.interrupted = True

    def get_summary_metrics(self) -> Dict[str, Any]:
        if not self.metrics:
            return {"turns": 0, "avg_total_latency_ms": 0.0, "avg_ttft_ms": 0.0}
        
        valid_latencies = [m.total_latency_ms for m in self.metrics if m.total_latency_ms > 0]
        valid_ttft = [m.ttft_ms for m in self.metrics if m.ttft_ms > 0]
        
        return {
            "session_id": self.session_id,
            "tenant_id": self.tenant_id,
            "duration_seconds": round(time.time() - self.created_at, 2),
            "total_turns": len(self.metrics),
            "interruption_count": sum(1 for m in self.metrics if m.interrupted),
            "avg_total_latency_ms": round(sum(valid_latencies) / len(valid_latencies), 1) if valid_latencies else 0.0,
            "avg_ttft_ms": round(sum(valid_ttft) / len(valid_ttft), 1) if valid_ttft else 0.0,
        }
