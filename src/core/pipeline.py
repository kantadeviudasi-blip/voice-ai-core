import asyncio
import re
from typing import AsyncGenerator, Callable, Optional, Dict, Any
from src.core.session import CallSession, SessionState
from src.core.vad import VoiceActivityDetector
from src.core.metrics import LatencyTracker, CostEstimator
from src.adapters.stt.base import BaseSTT
from src.adapters.llm.base import BaseLLM
from src.adapters.tts.base import BaseTTS
from src.knowledge.prompt_builder import AgentProfile

def clean_text_for_tts(text: str) -> str:
    """Cleans markdown, emojis, formatting, and normalizes spoken units for TTS."""
    if not text:
        return ""
    # Strip markdown symbols (*, _, #, ~, `, >)
    cleaned = re.sub(r'[\*\_\#\~\`\>]', '', text)
    # Convert ₹ and Rs to spoken 'rupaye'
    cleaned = re.sub(r'₹\s*([0-9,]+)', r'\1 rupaye', cleaned)
    cleaned = re.sub(r'Rs\.?\s*([0-9,]+)', r'\1 rupaye', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'([0-9]+)\s*kW\b', r'\1 kilo-watt', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'%', ' percent ', cleaned)
    # Remove bracketed stage directions like (smiling), [laughs]
    cleaned = re.sub(r'\[.*?\]|\(.*?\)', '', cleaned)
    # Normalize whitespace
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

class VoicePipeline:
    """
    Core Real-Time Voice AI Pipeline Orchestrator.
    Handles Bi-directional Audio Streams, Barge-in Cancellation, and Hot-Swappable Adapters.
    """
    def __init__(
        self,
        stt: BaseSTT,
        llm: BaseLLM,
        tts: BaseTTS,
        agent_profile: AgentProfile,
        sample_rate: int = 16000,
        silence_threshold_ms: int = 450,
        energy_threshold: float = 0.015,
        barge_in_enabled: bool = True
    ):
        self.stt = stt
        self.llm = llm
        self.tts = tts
        self.agent_profile = agent_profile
        self.sample_rate = sample_rate
        self.barge_in_enabled = barge_in_enabled

        self.session = CallSession(tenant_id=agent_profile.tenant_id)
        self.vad = VoiceActivityDetector(
            sample_rate=sample_rate,
            energy_threshold=energy_threshold,
            silence_threshold_ms=silence_threshold_ms,
            on_speech_start=self._on_user_speech_start,
            on_speech_end=self._on_user_speech_end,
            on_barge_in=self._on_barge_in
        )

        self.latency_tracker = LatencyTracker()
        self.active_generation_task: Optional[asyncio.Task] = None
        self.outbound_audio_callback: Optional[Callable[[bytes], None]] = None
        self.event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None

    def set_callbacks(
        self,
        outbound_audio_callback: Callable[[bytes], None],
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
    ):
        self.outbound_audio_callback = outbound_audio_callback
        self.event_callback = event_callback

    def _emit_event(self, event_type: str, data: Dict[str, Any]):
        if self.event_callback:
            self.event_callback(event_type, data)

    def _on_user_speech_start(self):
        self.session.state = SessionState.LISTENING
        self._emit_event("user_speech_start", {"session_id": self.session.session_id})

    def _on_barge_in(self):
        if not self.barge_in_enabled:
            return
        
        # User interrupted the AI
        self.session.interrupt()
        self.vad.set_agent_speaking(False)

        # Cancel active LLM and TTS tasks immediately
        if self.active_generation_task and not self.active_generation_task.done():
            self.active_generation_task.cancel()

        self._emit_event("barge_in_interrupted", {
            "session_id": self.session.session_id,
            "message": "AI voice playback cancelled due to user interruption"
        })

    def _on_user_speech_end(self, audio_data: bytes):
        self.latency_tracker.mark_user_speech_end()
        self.session.state = SessionState.THINKING
        self.session.start_turn()

        # Launch response generation task asynchronously
        self.active_generation_task = asyncio.create_task(self._process_turn(audio_data))

    async def _process_turn(self, audio_data: bytes):
        try:
            # 1. Speech-To-Text (STT)
            self._emit_event("status", {"state": "transcribing"})
            user_transcript = await self.stt.transcribe(audio_data)
            self.latency_tracker.mark_stt_complete()

            if not user_transcript or user_transcript.startswith("["):
                self.session.state = SessionState.IDLE
                return

            await self._run_llm_and_tts(user_transcript)

        except asyncio.CancelledError:
            # Clean interruption
            self.vad.set_agent_speaking(False)
            self.session.state = SessionState.IDLE
        except Exception as e:
            self._emit_event("error", {"error": str(e)})
            self.vad.set_agent_speaking(False)
            self.session.state = SessionState.IDLE

    async def process_text_turn(self, user_transcript: str):
        """Direct text turn for instant testing without microphone."""
        if not user_transcript or not user_transcript.strip():
            return None
        self.session.state = SessionState.THINKING
        self.session.start_turn()
        self.latency_tracker.mark_user_speech_end()
        self.latency_tracker.mark_stt_complete()
        self.active_generation_task = asyncio.create_task(self._run_llm_and_tts(user_transcript.strip()))
        return self.active_generation_task

    async def _run_llm_and_tts(self, user_transcript: str):
        try:
            self._emit_event("transcript", {"role": "user", "text": user_transcript})

            # 2. LLM Streaming
            self._emit_event("status", {"state": "thinking"})
            agent_response_full = []
            sentence_buffer = ""
            first_token_received = False

            system_prompt = self.agent_profile.system_prompt
            messages = self.session.messages + [{"role": "user", "content": user_transcript}]

            # 3. Stream Tokens & Synthesize in Sentences for low latency
            async for token in self.llm.generate_stream(messages, system_prompt):
                if not first_token_received:
                    first_token_received = True
                    self.latency_tracker.mark_llm_first_token()
                    self._emit_event("status", {"state": "speaking"})

                agent_response_full.append(token)
                sentence_buffer += token

                # If sentence boundary reached (. ! ? or \n), synthesize speech immediately
                if any(punct in token for punct in [".", "!", "?", "\n", "।"]):
                    if sentence_buffer.strip():
                        await self._stream_tts(sentence_buffer.strip())
                        sentence_buffer = ""

            # Synthesize remaining text
            if sentence_buffer.strip():
                await self._stream_tts(sentence_buffer.strip())

            full_response_text = "".join(agent_response_full).strip()
            self._emit_event("transcript", {"role": "assistant", "text": full_response_text})

            # Complete turn and capture latency metrics
            self.latency_tracker.mark_turn_complete()
            metrics = self.latency_tracker.get_metrics()
            
            if self.session.current_turn:
                self.session.current_turn.vad_latency_ms = 450.0 # configured threshold
                self.session.current_turn.stt_latency_ms = metrics["stt_latency_ms"]
                self.session.current_turn.ttft_ms = metrics["ttft_ms"]
                self.session.current_turn.tts_first_chunk_ms = metrics["tts_latency_ms"]
                self.session.current_turn.total_latency_ms = metrics["total_latency_ms"]

            self.session.complete_turn(user_transcript, full_response_text)
            self._emit_event("turn_metrics", metrics)

            self.vad.set_agent_speaking(False)
            self.session.state = SessionState.IDLE

        except asyncio.CancelledError:
            self.vad.set_agent_speaking(False)
            self.session.state = SessionState.IDLE
        except Exception as e:
            self._emit_event("error", {"error": str(e)})
            self.vad.set_agent_speaking(False)
            self.session.state = SessionState.IDLE

    async def _stream_tts(self, text: str):
        """
        Synthesizes a sentence and sends the complete audio buffer over the WebSocket.
        """
        cleaned = clean_text_for_tts(text)
        if not cleaned:
            return

        self.vad.set_agent_speaking(True)
        self.session.state = SessionState.SPEAKING

        # Collect all audio chunks into a single buffer for this sentence
        audio_buffer = bytearray()
        async for audio_chunk in self.tts.synthesize_stream(cleaned):
            audio_buffer.extend(audio_chunk)

        if audio_buffer:
            self.latency_tracker.mark_tts_first_audio()
            if self.outbound_audio_callback:
                self.outbound_audio_callback(bytes(audio_buffer))

    def process_incoming_audio(self, pcm_chunk: bytes):
        """Ingests live audio chunk from telephony/browser WebSocket stream."""
        self.vad.process_chunk(pcm_chunk)

    async def trigger_greeting(self):
        """Plays initial outbound/inbound agent greeting script."""
        greeting_text = self.agent_profile.greeting
        self._emit_event("transcript", {"role": "assistant", "text": greeting_text})
        self._emit_event("status", {"state": "speaking"})
        await self._stream_tts(greeting_text)
        self.vad.set_agent_speaking(False)
        self.session.state = SessionState.IDLE
