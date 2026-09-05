import asyncio
import os
import pytest
from dotenv import load_dotenv
load_dotenv()
from src.knowledge.prompt_builder import PromptBuilder
from src.knowledge.extractor import DocumentExtractor
from src.core.vad import VoiceActivityDetector, VADState
from src.core.metrics import LatencyTracker, CostEstimator
from src.core.pipeline import VoicePipeline
from src.adapters.stt.mock_stt import MockSTT
from src.adapters.llm.mock_llm import MockLLM
from src.adapters.tts.deepgram_tts import DeepgramTTS

def test_document_to_agent_synthesis():
    sample_text = """GreenTech Solar Solutions Jaipur.
We provide 3kW, 5kW, and 10kW rooftop solar panel installations.
Get up to 40% government subsidy under PM Surya Ghar Muft Bijli Yojana.
ROI in 3 years. Free site feasibility survey available."""

    profile = PromptBuilder.synthesize_profile(
        tenant_id="greentech-solar",
        company_name="GreenTech Solar",
        extracted_text=sample_text,
        agent_name="Aakash",
        language_mode="Hinglish",
        primary_goal="Check electricity bill amount and schedule free solar rooftop survey"
    )

    assert profile.company_name == "GreenTech Solar"
    assert profile.agent_name == "Aakash"
    assert "GreenTech Solar" in profile.system_prompt
    assert "PM Surya Ghar" in profile.system_prompt
    assert "price_too_high" in profile.objection_matrix
    assert "नमस्ते" in profile.greeting or "Namaste" in profile.greeting

def test_unit_economics_cost_calculator():
    cost_min = CostEstimator.estimate_minute_cost(
        stt_provider="groq",
        llm_provider="groq",
        tts_provider="cartesia"
    )
    # Total cost for pure Groq + Cartesia + SIP stack should be estimated here
    assert cost_min <= 0.85
    assert cost_min >= 0.45

def test_vad_speech_and_silence_detection():
    speech_started = False
    speech_ended = False

    def on_start():
        nonlocal speech_started
        speech_started = True

    def on_end(audio):
        nonlocal speech_ended
        speech_ended = True

    vad = VoiceActivityDetector(
        sample_rate=16000,
        chunk_duration_ms=20,
        energy_threshold=0.01,
        silence_threshold_ms=60,
        speech_onset_ms=40.0,
        min_utterance_speech_ms=80.0,
        on_speech_start=on_start,
        on_speech_end=on_end
    )

    # 1. Feed loud PCM frame (Simulate speech - 5 frames * 20ms = 100ms)
    loud_frame = b"\x50\x20" * 320 # high amplitude PCM
    for _ in range(5):
        vad.process_chunk(loud_frame)

    assert speech_started is True
    assert vad.state == VADState.SPEAKING

    # 2. Feed silence frames to trigger utterance completion
    silent_frame = b"\x00\x00" * 320
    for _ in range(5):
        vad.process_chunk(silent_frame)

    assert speech_ended is True
    assert vad.state == VADState.SILENCE

def test_vad_ignores_background_noise_and_transient_clicks():
    """Verifies that background room noise, mic hiss, fan rumble, and short clicks ('खड़का/हुश-फुश') are ignored."""
    speech_started = False
    speech_ended = False

    def on_start():
        nonlocal speech_started
        speech_started = True

    def on_end(audio):
        nonlocal speech_ended
        speech_ended = True

    vad = VoiceActivityDetector(
        sample_rate=16000,
        chunk_duration_ms=20,
        energy_threshold=0.022,
        silence_threshold_ms=200,
        speech_onset_ms=120.0,
        min_utterance_speech_ms=300.0,
        on_speech_start=on_start,
        on_speech_end=on_end
    )

    # 1. Feed background ambient noise / fan hiss (low amplitude ~ RMS 0.005)
    noise_frame = b"\x10\x01" * 320
    for _ in range(20): # 400ms of room noise
        vad.process_chunk(noise_frame)

    assert speech_started is False
    assert vad.state == VADState.SILENCE

    # 2. Feed a transient click/pop (single 20ms frame)
    click_frame = b"\x50\x30" * 320
    vad.process_chunk(click_frame)
    vad.process_chunk(noise_frame)
    vad.process_chunk(noise_frame)

    assert speech_started is False
    assert vad.state == VADState.SILENCE

    # 3. Feed sustained real human speech (350ms of strong voice)
    speech_frame = b"\x80\x25" * 320
    for _ in range(18): # 360ms of speech
        vad.process_chunk(speech_frame)

    assert speech_started is True
    assert vad.state == VADState.SPEAKING

    # 4. Silence after speaking
    for _ in range(12): # 240ms of silence
        vad.process_chunk(b"\x00\x00" * 320)

    assert speech_ended is True
    assert vad.state == VADState.SILENCE

@pytest.mark.asyncio
async def test_end_to_end_voice_pipeline_turn():
    profile = PromptBuilder.synthesize_profile(
        tenant_id="test-co",
        company_name="Test Enterprise",
        extracted_text="We offer CRM and Automation services."
    )

    stt = MockSTT(simulated_text="Kya aap CRM demo provide karte hain?")
    llm = MockLLM()
    tts = DeepgramTTS()

    pipeline = VoicePipeline(stt=stt, llm=llm, tts=tts, agent_profile=profile)
    await pipeline.start()
    
    received_audio_chunks = []
    received_events = []

    pipeline.set_callbacks(
        outbound_audio_callback=lambda chunk: received_audio_chunks.append(chunk),
        event_callback=lambda evt, data: received_events.append((evt, data))
    )

    # Trigger simulated user speech turn
    await pipeline._process_turn(b"\x00" * 3200)

    import asyncio
    await asyncio.sleep(1.0) # Yield for real Deepgram streaming

    # Verify turn completed
    assert len(pipeline.session.messages) == 2
    assert pipeline.session.messages[0]["role"] == "user"
    assert pipeline.session.messages[0]["content"] == "Kya aap CRM demo provide karte hain?"
    assert pipeline.session.messages[1]["role"] == "assistant"
    assert len(received_audio_chunks) > 0
    assert any(evt == "turn_metrics" for evt, _ in received_events)
    await pipeline.close()

def test_text_normalization_for_tts():
    from src.core.pipeline import clean_text_for_tts
    raw_text = "**Namaste!** Aapka Sunrise Heights cost ₹78,000 aayega for 3kW system with 40% subsidy. [smiling]"
    cleaned = clean_text_for_tts(raw_text)
    assert "*" not in cleaned
    assert "₹" not in cleaned
    assert "78,000 रुपये" in cleaned
    assert "3 किलोवाट" in cleaned
    assert "प्रतिशत" in cleaned
    assert "सनराइज" in cleaned
    assert "हाइट्स" in cleaned
    assert "[smiling]" not in cleaned


@pytest.mark.asyncio
async def test_direct_text_turn_processing():
    profile = PromptBuilder.synthesize_profile(
        tenant_id="solar-co",
        company_name="Apex Solar",
        extracted_text="Solar systems available."
    )
    stt = MockSTT()
    llm = MockLLM()
    tts = DeepgramTTS()
    pipeline = VoicePipeline(stt=stt, llm=llm, tts=tts, agent_profile=profile)
    await pipeline.start()
    
    received_audio = []
    pipeline.set_callbacks(outbound_audio_callback=lambda c: received_audio.append(c))

    task = await pipeline.process_text_turn("Subsidy kitni milegi?")
    if task:
        await task

    await asyncio.sleep(1.0) # Yield for Deepgram real streaming audio
    assert len(pipeline.session.messages) == 2
    assert pipeline.session.messages[0]["content"] == "Subsidy kitni milegi?"
    assert len(received_audio) > 0
    await pipeline.close()

def test_multidomain_prompt_synthesis():
    solar_profile = PromptBuilder.synthesize_profile(
        tenant_id="apex-solar-solutions",
        company_name="Apex Solar Solutions",
        extracted_text="We offer 3kW and 5kW rooftop solar panels with Rs. 78,000 subsidy under PM Surya Ghar Yojana.",
        agent_name="Sneha",
        primary_goal="Book free rooftop solar site inspection"
    )
    assert "Apex Solar Solutions" in solar_profile.system_prompt
    assert "Sneha" in solar_profile.system_prompt
    assert "PM Surya Ghar" in solar_profile.system_prompt
    assert "78,000" in solar_profile.system_prompt


@pytest.mark.asyncio
async def test_barge_in_task_cancellation_and_queue_flush():
    profile = PromptBuilder.synthesize_profile(
        tenant_id="test-co",
        company_name="Test Enterprise",
        extracted_text="We offer CRM and Automation services."
    )

    stt = MockSTT(simulated_text="Hello")
    llm = MockLLM()
    tts = DeepgramTTS()

    pipeline = VoicePipeline(stt=stt, llm=llm, tts=tts, agent_profile=profile)
    await pipeline.start()
    dispatched_audio = []
    events = []

    pipeline.set_callbacks(
        outbound_audio_callback=lambda chunk: dispatched_audio.append(chunk),
        event_callback=lambda evt, data: events.append((evt, data))
    )

    # Start generation
    task = asyncio.create_task(pipeline._run_llm_and_tts("Tell me more"))
    pipeline.active_generation_task = task

    # Give it a moment to begin streaming
    await asyncio.sleep(0.06)

    # Simulate Barge-In Interruption
    pipeline._on_barge_in()

    # Wait briefly for cancellation to propagate
    await asyncio.sleep(0.02)

    assert task.done() or task.cancelled()
    assert pipeline.vad.is_agent_speaking is False
    assert pipeline.audio_output_queue.empty()
    assert any(evt == "barge_in_interrupted" for evt, _ in events)
    await pipeline.close()

def test_telephony_clear_buffer_signals():
    import json
    from src.telephony.exotel_handler import ExotelStreamHandler
    from src.telephony.plivo_handler import PlivoStreamHandler

    # Exotel clear signal
    exotel = ExotelStreamHandler(on_audio_chunk=lambda c: None)
    exotel.stream_sid = "EX98765"
    exotel_clear = json.loads(exotel.create_clear_response())
    assert exotel_clear["event"] == "clear"
    assert exotel_clear["stream_sid"] == "EX98765"

    # Plivo clear signal
    plivo = PlivoStreamHandler(on_audio_chunk=lambda c: None)
    plivo_clear = json.loads(plivo.create_clear_response())
    assert plivo_clear["event"] == "clearAudio"

def test_telephony_multi_sample_rate_bridge():
    from src.telephony.base_telephony import BaseTelephonyBridge

    # 1 second of audio at 8kHz, 16kHz, 24kHz
    pcm8k = b"\x10\x00" * 8000
    pcm16k = b"\x10\x00" * 16000
    pcm24k = b"\x10\x00" * 24000

    mu8k = BaseTelephonyBridge.pcm8k_to_mulaw(pcm8k)
    mu16k = BaseTelephonyBridge.pcm16_to_mulaw(pcm16k)
    mu24k = BaseTelephonyBridge.pcm_to_mulaw(pcm24k, sample_rate=24000)

    # All should be converted to standard 8000 bytes per second telephony mu-law
    assert len(mu8k) == 8000
    assert len(mu16k) == 8000
    assert len(mu24k) == 8000

    # Test WAV container header stripping
    wav_with_header = b"RIFF" + b"\x00" * 40 + pcm8k
    mu_stripped = BaseTelephonyBridge.pcm8k_to_mulaw(wav_with_header)
    assert len(mu_stripped) == 8000

@pytest.mark.asyncio
async def test_sarvam_tts_synthesis():
    from src.adapters.tts.sarvam_tts import SarvamTTS
    sarvam_key = os.getenv("SARVAM_API_KEY")
    if not sarvam_key:
        pytest.skip("SARVAM_API_KEY not set")

    tts = SarvamTTS(api_key=sarvam_key, speaker="ritu", sample_rate=8000)
    audio = await tts.synthesize("नमस्ते, मैं आपकी क्या सहायता कर सकती हूँ?")
    await tts.close()
    assert len(audio) > 1000
    assert audio.startswith(b"RIFF")

