import asyncio
import re
import logging
from typing import AsyncGenerator, Callable, Optional, Dict, Any
from src.core.session import CallSession, SessionState
from src.core.vad import VoiceActivityDetector
from src.core.metrics import LatencyTracker, CostEstimator
from src.adapters.stt.base import BaseSTT
from src.adapters.llm.base import BaseLLM
from src.adapters.tts.base import BaseTTS
from src.knowledge.prompt_builder import AgentProfile

logger = logging.getLogger(__name__)

def clean_text_for_tts(text: str) -> str:
    """Cleans markdown, formatting, and phonetically normalizes words into Hindi Devanagari for natural Indian TTS."""
    if not text:
        return ""
    # Strip common XML-like tags and their contents (e.g. <think>...</think>)
    cleaned = re.sub(r'<(think|thought|thinking|xml|code|step).*?</\1>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Strip any remaining standalone XML tags
    cleaned = re.sub(r'<[^>]+>', '', cleaned)
    # Strip thinking preambles and internal monologues
    cleaned = re.sub(r'^(thinking process|thought|analysis|this is thinking process|internal monologue|user is saying).*?(:|\n)', '', cleaned, flags=re.IGNORECASE)
    # Strip markdown symbols (*, _, #, ~, `, >, quotes)
    cleaned = re.sub(r'[\*\_\#\~\`\>\"\'\“\”]', '', cleaned)

    
    # Common English telecalling names and terms to Devanagari phonetics
    phonetic_map = [
        (r'\bSunrise\b', 'सनराइज'),
        (r'\bHeights\b', 'हाइट्स'),
        (r'\bApex\b', 'अपेक्स'),
        (r'\bGreenTech\b', 'ग्रीनटेक'),
        (r'\bDental\b', 'डेंटल'),
        (r'\bCare\b', 'केयर'),
        (r'\bSolar\b', 'सोलर'),
        (r'\bSolutions\b', 'सॉल्यूशंस'),
        (r'\bSneha\b', 'स्नेहा'),
        (r'\bPooja\b', 'पूजा'),
        (r'\bRiya\b', 'रिया'),
        (r'\bWhatsApp\b', 'व्हाट्सएप'),
        (r'\bEMI\b', 'ईएमआई'),
        (r'\bBHK\b', 'बीएचके'),
        (r'\bflats?\b', 'फ्लैट्स'),
        (r'\bclubhouse\b', 'क्लबहाउस'),
        (r'\bswimming pool\b', 'स्विमिंग पूल'),
        (r'\bamenities\b', 'सुविधाएं'),
        (r'\bdiscount\b', 'डिस्काउंट'),
        (r'\boffer\b', 'ऑफर'),
        (r'\bfree\b', 'फ्री'),
        (r'\bsite visit\b', 'साइट विजिट'),
        (r'\bsurvey\b', 'सर्वे'),
        (r'\bdetails\b', 'डिटेल्स'),
        (r'\bbrochure\b', 'ब्रोशर'),
        (r'\bcall\b', 'कॉल'),
        (r'\bbusy\b', 'बिजी'),
        (r'\bproblem\b', 'प्रॉब्लम'),
        (r'\btime\b', 'टाइम'),
        (r'\bbill\b', 'बिल'),
        (r'\bsubsidy\b', 'सब्सिडी'),
        (r'\bmonthly\b', 'मंथली'),
        (r'\bweekend\b', 'वीकेंड'),
        (r'\bSaturday\b', 'शनिवार'),
        (r'\bSunday\b', 'रविवार'),
        (r'\bMain\s+([A-Z\u0900-\u097F])', r'मैं \1'),
    ]
    for pattern, repl in phonetic_map:
        cleaned = re.sub(pattern, repl, cleaned, flags=re.IGNORECASE)

    # Convert ₹ and Rs to spoken 'रुपये'
    cleaned = re.sub(r'₹\s*([0-9,]+)', r'\1 रुपये', cleaned)
    cleaned = re.sub(r'Rs\.?\s*([0-9,]+)', r'\1 रुपये', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'([0-9]+)\s*kW\b', r'\1 किलोवाट', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'%', ' प्रतिशत ', cleaned)
    # Remove bracketed stage directions like (smiling), [laughs]
    cleaned = re.sub(r'\[.*?\]|\(.*?\)', '', cleaned)
    # Normalize multiple punctuation marks
    cleaned = re.sub(r'[\.\!\?]{2,}', '.', cleaned)
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
        energy_threshold: float = 0.022,
        barge_in_enabled: bool = True,
        speech_onset_ms: float = 120.0,
        min_utterance_speech_ms: float = 300.0,
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
            speech_onset_ms=speech_onset_ms,
            min_utterance_speech_ms=min_utterance_speech_ms,
            on_speech_start=self._on_user_speech_start,
            on_speech_end=self._on_user_speech_end,
            on_barge_in=self._on_barge_in
        )

        self.latency_tracker = LatencyTracker()
        self.active_generation_task: Optional[asyncio.Task] = None
        self.audio_output_queue: asyncio.Queue = asyncio.Queue(maxsize=50)
        self.audio_worker_task: Optional[asyncio.Task] = None
        self.outbound_audio_callback: Optional[Callable[[bytes], None]] = None
        self.event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
        self._state_lock = asyncio.Lock()
        
        # Check if STT supports streaming
        self.using_streaming_stt = getattr(self.stt, 'is_streaming_supported', False)
        # Check if TTS uses event-based streaming
        self.using_streaming_tts = getattr(self.tts, 'connect_stream', None) is not None
        self._tts_first_chunk_received = False
    async def start(self):
        """Initializes streaming STT and TTS connections if supported."""
        if self.using_streaming_stt:
            await self.stt.connect_stream(self._on_realtime_transcript)
        if self.using_streaming_tts:
            await self.tts.connect_stream(self._on_tts_audio_chunk)

    def _on_realtime_transcript(self, transcript: str, is_final: bool):
        """Callback for real-time STT streaming (e.g. Deepgram)."""
        if is_final and transcript.strip():
            words = transcript.strip().split()
            if len(words) >= 2:
                asyncio.create_task(self._locked_speech_end_streaming(transcript.strip()))
            else:
                logger.info(f"Skipping short background noise transcript: {transcript}")

    def _on_tts_audio_chunk(self, chunk: bytes):
        """Callback for real-time TTS streaming (e.g. Deepgram)."""
        if self.session.state == SessionState.LISTENING:
            return

        if not self._tts_first_chunk_received:
            self.latency_tracker.mark_tts_first_audio()
            self._tts_first_chunk_received = True

        self.vad.set_agent_speaking(True)
        self.session.state = SessionState.SPEAKING
        self._ensure_audio_worker()

        try:
            self.audio_output_queue.put_nowait(chunk)
        except asyncio.QueueFull:
            pass

    def set_callbacks(
        self,
        outbound_audio_callback: Callable[[bytes], None],
        event_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None
    ):
        self.outbound_audio_callback = outbound_audio_callback
        self.event_callback = event_callback
        self._ensure_audio_worker()

    def _ensure_audio_worker(self):
        """Ensures the background audio dispatch worker is active."""
        if self.audio_worker_task is None or self.audio_worker_task.done():
            try:
                loop = asyncio.get_running_loop()
                self.audio_worker_task = loop.create_task(self._audio_dispatch_worker())
            except RuntimeError:
                pass

    async def _audio_dispatch_worker(self):
        """Dispatches queued audio chunks to the outbound audio callback."""
        while True:
            try:
                chunk = await self.audio_output_queue.get()
                if chunk and self.outbound_audio_callback:
                    try:
                        self.outbound_audio_callback(chunk)
                    except Exception as e:
                        logger.error(f"Error in outbound_audio_callback: {e}")
                self.audio_output_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Audio dispatch worker error: {e}")

    def flush_audio_queue(self):
        """Flushes all queued audio frames immediately on interruption."""
        while not self.audio_output_queue.empty():
            try:
                self.audio_output_queue.get_nowait()
                self.audio_output_queue.task_done()
            except (asyncio.QueueEmpty, ValueError):
                break

    def _emit_event(self, event_type: str, data: Dict[str, Any]):
        if self.event_callback:
            try:
                self.event_callback(event_type, data)
            except Exception as e:
                logger.error(f"Error in event_callback: {e}")

    def _on_user_speech_start(self):
        # Fire-and-forget state transition without blocking audio thread
        asyncio.create_task(self._locked_speech_start())

    async def _locked_speech_start(self):
        async with self._state_lock:
            self.session.state = SessionState.LISTENING
            self._emit_event("user_speech_start", {"session_id": self.session.session_id})

    def _on_barge_in(self):
        """
        Triggered immediately by VAD or STT when user speech activity is detected during AI speech.
        """
        if not self.barge_in_enabled:
            return
        
        logger.info("⚡ [BARGE-IN DETECTED] Aborting active generation & clearing audio buffer.")

        # Trigger async safe interruption
        asyncio.create_task(self._locked_barge_in())

    async def _locked_barge_in(self):
        async with self._state_lock:
            # 1. Update session & VAD state
            self.session.interrupt()
            self.vad.set_agent_speaking(False)

            # 2. Cancel active LLM and TTS tasks immediately
            if self.active_generation_task and not self.active_generation_task.done():
                self.active_generation_task.cancel()

            # 3. Flush internal Python audio queue
            self.flush_audio_queue()

            # 4. Emit barge-in event so Telephony Bridges send WebSocket clear signal
            self._emit_event("barge_in_interrupted", {
                "session_id": self.session.session_id,
                "message": "AI voice playback cancelled due to user interruption"
            })

    def _on_user_speech_end(self, audio_data: bytes):
        asyncio.create_task(self._locked_speech_end(audio_data))

    async def _locked_speech_end_streaming(self, transcript: str):
        """Used by true Streaming STT (Deepgram) to instantly trigger LLM."""
        async with self._state_lock:
            # If an LLM task is already running, avoid double-firing.
            if self.session.state != SessionState.LISTENING:
                return
            
            self.latency_tracker.mark_user_speech_end()
            self.session.state = SessionState.THINKING
            self.session.start_turn()
            self._ensure_audio_worker()
            self.active_generation_task = asyncio.create_task(self._run_llm_and_tts(transcript))

    async def _locked_speech_end(self, audio_data: bytes):
        """Used by Batch VAD-STT (Groq). Ignored if using true Streaming STT."""
        if self.using_streaming_stt:
            return  # Rely on _on_realtime_transcript instead

        async with self._state_lock:
            self.latency_tracker.mark_user_speech_end()
            self.session.state = SessionState.THINKING
            self.session.start_turn()

            # Launch response generation task asynchronously
            self._ensure_audio_worker()
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
            self.flush_audio_queue()
        except Exception as e:
            self._emit_event("error", {"error": str(e)})
            self.vad.set_agent_speaking(False)
            self.session.state = SessionState.IDLE
            self.flush_audio_queue()

    async def process_text_turn(self, user_transcript: str):
        """Direct text turn for instant testing without microphone."""
        if not user_transcript or not user_transcript.strip():
            return None
        async with self._state_lock:
            self._ensure_audio_worker()
            self.session.state = SessionState.THINKING
            self.session.start_turn()
            self.latency_tracker.mark_user_speech_end()
            self.latency_tracker.mark_stt_complete()
            self.active_generation_task = asyncio.create_task(self._run_llm_and_tts(user_transcript.strip()))
            return self.active_generation_task

    async def _run_llm_and_tts(self, user_transcript: str):
        try:
            self._tts_first_chunk_received = False
            self._emit_event("transcript", {"role": "user", "text": user_transcript})

            # 2. LLM Streaming
            self._emit_event("status", {"state": "thinking"})
            agent_response_full = []
            sentence_buffer = ""
            first_token_received = False

            system_prompt = self.agent_profile.system_prompt
            messages = self.session.messages + [{"role": "user", "content": user_transcript}]

            # 3. Stream Tokens & Synthesize with Early-Token Progressive Chunking
            in_think_tag = False
            skip_preamble = True
            accumulated_preamble = ""
            first_chunk_emitted = False

            # Clause boundary pattern (Devanagari danda, period, exclamation, question, comma, semicolon, newline)
            clause_delimiters = [".", "!", "?", "\n", "।", ",", ";"]

            async for token in self.llm.generate_stream(messages, system_prompt):
                # Filter <think>...</think> tags and other XML-like tags
                if re.search(r'<(think|thought|thinking|xml|code|step).*?>', token, re.IGNORECASE):
                    in_think_tag = True
                    continue
                if in_think_tag:
                    if re.search(r'</(think|thought|thinking|xml|code|step)>', token, re.IGNORECASE):
                        in_think_tag = False
                    continue


                # Strip accidental thinking preambles like "Thinking Process:" or "This is thinking"
                if skip_preamble:
                    accumulated_preamble += token
                    preamble_lower = accumulated_preamble.lower()
                    # Check if it starts with known thinking prefixes or XML tags
                    if re.search(r'^(<[^>]+>|thinking process|thought|analysis|this is thinking process|internal monologue|user is saying|let me|okay, |okay so|alright|i need to|i should|i will|i am going to|```).*?(\n\n|\n|:|>|```)', accumulated_preamble, flags=re.IGNORECASE):
                        # Strip the preamble prefix and resume from clean content
                        accumulated_preamble = re.sub(r'^(<[^>]+>|thinking process|thought|analysis|this is thinking process|internal monologue|user is saying|let me|okay,|okay so|alright|i need to|i should|i will|i am going to|```).*?(\n\n|\n|:|>|```)', '', accumulated_preamble, flags=re.IGNORECASE).strip()
                        if not accumulated_preamble:
                            continue
                        token = accumulated_preamble
                        skip_preamble = False
                    elif len(accumulated_preamble) > 120 and not any(p in preamble_lower for p in ["think", "thought", "process", "user is", "i need", "i should", "okay", "alright", "<", "```"]):
                        # Long enough and no thinking keywords — safe to emit
                        skip_preamble = False
                        token = accumulated_preamble
                    else:
                        # Still accumulating — check for dangerous thinking keywords
                        if any(p in preamble_lower for p in ["thinking process", "this is thinking", "thought:", "user is saying", "/think", "<xml", "<think", "```"]):
                            continue
                        if len(accumulated_preamble) > 120:
                            # Force emit after 120 chars regardless
                            skip_preamble = False
                            token = accumulated_preamble
                        else:
                            continue

                if not first_token_received:
                    first_token_received = True
                    self.latency_tracker.mark_llm_first_token()
                    self._emit_event("status", {"state": "speaking"})

                agent_response_full.append(token)
                sentence_buffer += token

                # Progressive Early-Token Chunking:
                # Chunk 1: Fire immediately upon 3-4 words or early punctuation for near-zero latency
                if not first_chunk_emitted:
                    words = sentence_buffer.strip().split()
                    has_early_punct = any(p in sentence_buffer for p in clause_delimiters)
                    if len(words) >= 3 or (len(words) >= 2 and has_early_punct):
                        if sentence_buffer.strip():
                            first_chunk_emitted = True
                            await self._stream_tts(sentence_buffer.strip())
                            sentence_buffer = ""
                else:
                    # Subsequent Chunks: Fire on clause boundaries
                    if any(punct in token for punct in clause_delimiters):
                        if len(sentence_buffer.strip().split()) >= 2 or any(p in sentence_buffer for p in [".", "!", "?", "\n", "।"]):
                            if sentence_buffer.strip():
                                await self._stream_tts(sentence_buffer.strip())
                                sentence_buffer = ""

            # Flush accumulated_preamble if it was never emitted
            if skip_preamble and accumulated_preamble:
                if not first_token_received:
                    first_token_received = True
                    self.latency_tracker.mark_llm_first_token()
                    self._emit_event("status", {"state": "speaking"})
                agent_response_full.append(accumulated_preamble)
                sentence_buffer += accumulated_preamble

            # Synthesize remaining text
            if sentence_buffer.strip():
                await self._stream_tts(sentence_buffer.strip())

            if self.using_streaming_tts:
                try:
                    await asyncio.sleep(0.4)
                    await asyncio.wait_for(self.audio_output_queue.join(), timeout=3.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass

            full_response_text = "".join(agent_response_full).strip()
            self._emit_event("transcript", {"role": "assistant", "text": full_response_text})

            # Complete turn and capture latency metrics
            self.latency_tracker.mark_turn_complete()
            metrics = self.latency_tracker.get_metrics()
            
            if self.session.current_turn:
                self.session.current_turn.vad_latency_ms = 280.0
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
            self.flush_audio_queue()
        except Exception as e:
            self._emit_event("error", {"error": str(e)})
            self.vad.set_agent_speaking(False)
            self.session.state = SessionState.IDLE
            self.flush_audio_queue()

    async def _stream_tts(self, text: str):
        """
        Synthesizes a chunk and streams audio buffers through the queue.
        """
        cleaned = clean_text_for_tts(text)
        if not cleaned:
            return

        self.vad.set_agent_speaking(True)
        self.session.state = SessionState.SPEAKING
        self._ensure_audio_worker()

        if self.using_streaming_tts:
            await self.tts.synthesize_stream(cleaned)
        else:
            # Collect all audio chunks for this sentence to preserve valid MP3/audio container structure
            sentence_buffer = bytearray()
            first_chunk = True
            async for audio_chunk in self.tts.synthesize_stream(cleaned):
                if first_chunk:
                    self.latency_tracker.mark_tts_first_audio()
                    first_chunk = False
                sentence_buffer.extend(audio_chunk)

            if sentence_buffer:
                await self.audio_output_queue.put(bytes(sentence_buffer))


    def process_incoming_audio(self, pcm_chunk: bytes):
        """Ingests live audio chunk from telephony/browser WebSocket stream."""
        self.vad.process_chunk(pcm_chunk)
        
        # True Streaming STT path (Deepgram)
        if self.using_streaming_stt and not self.vad.is_agent_speaking:
            asyncio.create_task(self.stt.push_audio(pcm_chunk))

    async def trigger_greeting(self):
        """Plays initial outbound/inbound agent greeting script."""
        greeting_text = self.agent_profile.greeting
        self._emit_event("transcript", {"role": "assistant", "text": greeting_text})
        self._emit_event("status", {"state": "speaking"})
        await self._stream_tts(greeting_text)
        self.vad.set_agent_speaking(False)
        self.session.state = SessionState.IDLE

    async def close(self):
        """Cleans up background tasks and flushes queues."""
        if self.active_generation_task and not self.active_generation_task.done():
            self.active_generation_task.cancel()
        if self.audio_worker_task and not self.audio_worker_task.done():
            self.audio_worker_task.cancel()
        self.flush_audio_queue()
        if self.using_streaming_stt:
            await self.stt.close()

