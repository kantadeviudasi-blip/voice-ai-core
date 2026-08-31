import math
import struct
import time
from typing import Callable, Optional

class VADState:
    SILENCE = "silence"
    SPEAKING = "speaking"

class VoiceActivityDetector:
    """
    Real-time Frame-by-Frame Voice Activity Detector (VAD) with Barge-in Interruption Support.
    Optimized for high-concurrency with minimal CPU overhead.
    """
    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 20,
        energy_threshold: float = 0.015,
        silence_threshold_ms: int = 500,
        on_speech_start: Optional[Callable[[], None]] = None,
        on_speech_end: Optional[Callable[[bytes], None]] = None,
        on_barge_in: Optional[Callable[[], None]] = None,
    ):
        self.sample_rate = sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.energy_threshold = energy_threshold
        self.silence_threshold_ms = silence_threshold_ms
        
        self.on_speech_start = on_speech_start
        self.on_speech_end = on_speech_end
        self.on_barge_in = on_barge_in
        
        self.state = VADState.SILENCE
        self.audio_buffer = bytearray()
        self.silence_duration_ms = 0
        self.speech_duration_ms = 0
        self.is_agent_speaking = False

    def set_agent_speaking(self, speaking: bool):
        """Notifies VAD whether the AI agent is currently playing TTS audio."""
        self.is_agent_speaking = speaking

    def _calculate_rms(self, pcm_data: bytes) -> float:
        """Calculates Root Mean Square (RMS) energy of 16-bit PCM audio."""
        if not pcm_data:
            return 0.0
        count = len(pcm_data) // 2
        if count == 0:
            return 0.0
        format_str = f"<{count}h"
        try:
            shorts = struct.unpack(format_str, pcm_data)
        except Exception:
            return 0.0
        
        sum_squares = sum(s * s for s in shorts)
        mean_square = sum_squares / count
        rms = math.sqrt(mean_square) / 32768.0  # Normalize to 0.0 - 1.0
        return rms

    def process_chunk(self, chunk: bytes):
        """
        Process a single audio chunk (typically 20ms of PCM 16-bit).
        """
        rms = self._calculate_rms(chunk)
        is_speech = rms > self.energy_threshold

        if is_speech:
            self.silence_duration_ms = 0
            self.speech_duration_ms += self.chunk_duration_ms
            self.audio_buffer.extend(chunk)

            # If user starts speaking while AI is talking -> Trigger Barge-In
            if self.is_agent_speaking and self.speech_duration_ms >= 60: # 60ms of sustained speech
                if self.on_barge_in:
                    self.on_barge_in()

            if self.state == VADState.SILENCE and self.speech_duration_ms >= 60:
                self.state = VADState.SPEAKING
                if self.on_speech_start:
                    self.on_speech_start()
        else:
            if self.state == VADState.SPEAKING:
                self.audio_buffer.extend(chunk)
                self.silence_duration_ms += self.chunk_duration_ms

                # User stopped speaking for longer than silence_threshold_ms -> Utterance Finished
                if self.silence_duration_ms >= self.silence_threshold_ms:
                    self.state = VADState.SILENCE
                    utterance_audio = bytes(self.audio_buffer)
                    self.audio_buffer.clear()
                    self.speech_duration_ms = 0
                    self.silence_duration_ms = 0
                    if self.on_speech_end:
                        self.on_speech_end(utterance_audio)
            else:
                self.speech_duration_ms = 0
                self.silence_duration_ms = 0

    def reset(self):
        self.state = VADState.SILENCE
        self.audio_buffer.clear()
        self.speech_duration_ms = 0
        self.silence_duration_ms = 0
