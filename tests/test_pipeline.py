import asyncio
import pytest
from src.knowledge.prompt_builder import PromptBuilder
from src.knowledge.extractor import DocumentExtractor
from src.core.vad import VoiceActivityDetector, VADState
from src.core.metrics import LatencyTracker, CostEstimator
from src.core.pipeline import VoicePipeline
from src.adapters.stt.mock_stt import MockSTT
from src.adapters.llm.mock_llm import MockLLM
from src.adapters.tts.mock_tts import MockTTS

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
    assert "Namaste" in profile.greeting

def test_unit_economics_cost_calculator():
    cost_min = CostEstimator.estimate_minute_cost(
        stt_provider="deepgram",
        llm_provider="groq",
        tts_provider="edgetts"
    )
    # Total cost should strictly be under ₹1.10/min
    assert cost_min <= 1.15
    assert cost_min >= 0.85

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
        on_speech_start=on_start,
        on_speech_end=on_end
    )

    # 1. Feed loud PCM frame (Simulate speech)
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

@pytest.mark.asyncio
async def test_end_to_end_voice_pipeline_turn():
    profile = PromptBuilder.synthesize_profile(
        tenant_id="test-co",
        company_name="Test Enterprise",
        extracted_text="We offer CRM and Automation services."
    )

    stt = MockSTT(simulated_text="Kya aap CRM demo provide karte hain?")
    llm = MockLLM()
    tts = MockTTS()

    pipeline = VoicePipeline(stt=stt, llm=llm, tts=tts, agent_profile=profile)
    
    received_audio_chunks = []
    received_events = []

    pipeline.set_callbacks(
        outbound_audio_callback=lambda chunk: received_audio_chunks.append(chunk),
        event_callback=lambda evt, data: received_events.append((evt, data))
    )

    # Trigger simulated user speech turn
    await pipeline._process_turn(b"\x00" * 3200)

    # Verify turn completed
    assert len(pipeline.session.messages) == 2
    assert pipeline.session.messages[0]["role"] == "user"
    assert pipeline.session.messages[0]["content"] == "Kya aap CRM demo provide karte hain?"
    assert pipeline.session.messages[1]["role"] == "assistant"
    assert len(received_audio_chunks) > 0
    assert any(evt == "turn_metrics" for evt, _ in received_events)

def test_text_normalization_for_tts():
    from src.core.pipeline import clean_text_for_tts
    raw_text = "**Namaste!** Aapka cost ₹78,000 aayega for 3kW system with 40% subsidy. [smiling]"
    cleaned = clean_text_for_tts(raw_text)
    assert "*" not in cleaned
    assert "₹" not in cleaned
    assert "78,000 rupaye" in cleaned
    assert "3 kilo-watt" in cleaned
    assert "40 percent" in cleaned
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
    tts = MockTTS()
    pipeline = VoicePipeline(stt=stt, llm=llm, tts=tts, agent_profile=profile)
    
    received_audio = []
    pipeline.set_callbacks(outbound_audio_callback=lambda c: received_audio.append(c))

    task = await pipeline.process_text_turn("Subsidy kitni milegi?")
    if task:
        await task

    assert len(pipeline.session.messages) == 2
    assert pipeline.session.messages[0]["content"] == "Subsidy kitni milegi?"
    assert len(received_audio) > 0

def test_multidomain_prompt_synthesis():
    real_estate_profile = PromptBuilder.synthesize_profile(
        tenant_id="sunrise-heights",
        company_name="Sunrise Heights",
        extracted_text="We offer 2 BHK and 3 BHK luxury apartments starting from 45 Lakhs in Jaipur.",
        agent_name="Pooja",
        primary_goal="Book Saturday site visit"
    )
    assert "Sunrise Heights" in real_estate_profile.system_prompt
    assert "Pooja" in real_estate_profile.system_prompt
    assert "2 BHK" in real_estate_profile.system_prompt
    assert "FEW-SHOT EXAMPLES" in real_estate_profile.system_prompt
