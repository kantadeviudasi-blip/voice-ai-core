import time
from typing import Dict, Any

class LatencyTracker:
    """
    Precision latency timer tracking each stage of the conversational turn:
    - VAD speech end to STT completion (stt_latency)
    - STT completion to First LLM token (ttft)
    - First LLM token to First TTS audio packet ready (tts_first_chunk)
    - Total turnaround time (End of user speech -> Sound heard by user)
    """
    def __init__(self):
        self.t_user_speech_end: float = 0.0
        self.t_stt_complete: float = 0.0
        self.t_llm_first_token: float = 0.0
        self.t_tts_first_audio: float = 0.0
        self.t_turn_complete: float = 0.0

    def mark_user_speech_end(self):
        self.t_user_speech_end = time.perf_counter()
        self.t_stt_complete = 0.0
        self.t_llm_first_token = 0.0
        self.t_tts_first_audio = 0.0
        self.t_turn_complete = 0.0

    def mark_stt_complete(self):
        if self.t_stt_complete == 0.0:
            self.t_stt_complete = time.perf_counter()

    def mark_llm_first_token(self):
        if self.t_llm_first_token == 0.0:
            self.t_llm_first_token = time.perf_counter()

    def mark_tts_first_audio(self):
        if self.t_tts_first_audio == 0.0:
            self.t_tts_first_audio = time.perf_counter()

    def mark_turn_complete(self):
        self.t_turn_complete = time.perf_counter()

    def get_metrics(self) -> Dict[str, float]:
        stt_latency = (self.t_stt_complete - self.t_user_speech_end) * 1000 if self.t_stt_complete and self.t_user_speech_end else 0.0
        ttft = (self.t_llm_first_token - self.t_stt_complete) * 1000 if self.t_llm_first_token and self.t_stt_complete else 0.0
        tts_latency = (self.t_tts_first_audio - self.t_llm_first_token) * 1000 if self.t_tts_first_audio and self.t_llm_first_token else 0.0
        total_latency = (self.t_tts_first_audio - self.t_user_speech_end) * 1000 if self.t_tts_first_audio and self.t_user_speech_end else 0.0

        return {
            "stt_latency_ms": round(stt_latency, 1),
            "ttft_ms": round(ttft, 1),
            "tts_latency_ms": round(tts_latency, 1),
            "total_latency_ms": round(total_latency, 1),
        }

class CostEstimator:
    """
    Calculates exact running cost per call minute based on active providers.
    """
    RATES = {
        "telephony": 0.35,          # Direct SIP trunking (~₹0.35 / min)
        "stt_groq": 0.08,           # Groq Whisper LPU (~₹0.08 / min)
        "stt_whisper": 0.05,        # Self-Hosted Faster-Whisper Server cost share (~₹0.05 / min)
        "llm_groq": 0.10,           # ~₹0.10 / min
        "llm_gemini_flash": 0.08,   # ~₹0.08 / min
        "llm_deepseek": 0.09,       # ~₹0.09 / min
        "tts_edgetts": 0.00,        # Zero cost (100% Free)
        "tts_sarvam": 0.18,         # ₹0.18 / min
        "tts_cartesia": 0.25,       # ₹0.25 / min
        "server": 0.04              # Ingress / WebSocket node
    }

    @classmethod
    def estimate_minute_cost(cls, stt_provider: str, llm_provider: str, tts_provider: str) -> float:
        stt_cost = cls.RATES.get(f"stt_{stt_provider}", 0.08)
        llm_cost = cls.RATES.get(f"llm_{llm_provider}", 0.10)
        tts_cost = cls.RATES.get(f"tts_{tts_provider}", 0.00)
        telephony_cost = cls.RATES["telephony"]
        server_cost = cls.RATES["server"]

        return round(stt_cost + llm_cost + tts_cost + telephony_cost + server_cost, 2)
