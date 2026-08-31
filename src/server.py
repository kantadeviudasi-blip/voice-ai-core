import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import yaml
import asyncio
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn


# Auto-load .env file if present
ENV_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
if os.path.exists(ENV_PATH):
    with open(ENV_PATH, "r", encoding="utf-8") as env_f:
        for line in env_f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ[k.strip()] = v.strip()


from src.core.pipeline import VoicePipeline
from src.core.metrics import CostEstimator
from src.adapters.stt.deepgram_stt import DeepgramSTT
from src.adapters.stt.groq_whisper_stt import GroqWhisperSTT
from src.adapters.stt.mock_stt import MockSTT
from src.adapters.llm.groq_llm import GroqLLM
from src.adapters.llm.gemini_llm import GeminiLLM
from src.adapters.llm.deepseek_llm import DeepSeekLLM
from src.adapters.llm.mock_llm import MockLLM
from src.adapters.tts.edgetts_adapter import EdgeTTSAdapter
from src.adapters.tts.sarvam_tts import SarvamTTS
from src.adapters.tts.cartesia_tts import CartesiaTTS
from src.adapters.tts.mock_tts import MockTTS
from src.knowledge.extractor import DocumentExtractor
from src.knowledge.prompt_builder import PromptBuilder, AgentProfile
from src.knowledge.tenant_store import TenantStore
from src.telephony.exotel_handler import ExotelStreamHandler
from src.telephony.plivo_handler import PlivoStreamHandler
from src.telephony.twilio_handler import TwilioStreamHandler

# Load Configuration
CONFIG_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")
with open(CONFIG_PATH, "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

app = FastAPI(
    title="Voice AI Calling Agent Platform",
    description="High-concurrency, ultra-low latency, cost-optimized Voice Calling AI platform for Indian businesses.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

tenant_store = TenantStore()

def build_adapters(stt_choice: Optional[str] = None, llm_choice: Optional[str] = None, tts_choice: Optional[str] = None):
    """
    Factory function to instantiate hot-swappable adapters.
    STT  : Groq Whisper-large-v3-turbo (best & fastest Hindi/Hinglish)
    LLM  : Groq openai/gpt-oss-20b (fast everyday chat) — fallback chain auto-tries newer models
    TTS  : EdgeTTS (free, zero-key) | Sarvam | Cartesia
    """
    groq_key = os.getenv("GROQ_API_KEY", "")
    deepgram_key = os.getenv("DEEPGRAM_API_KEY", "")
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    deepseek_key = os.getenv("DEEPSEEK_API_KEY", "")
    sarvam_key = os.getenv("SARVAM_API_KEY", "")

    default_stt_cfg = config.get("pipeline", {}).get("default_stt", "groq")
    target_stt = stt_choice or default_stt_cfg

    # STT Adapter Resolution: Groq Whisper -> Deepgram -> Mock
    groq_stt_model = config.get("stt", {}).get("groq", {}).get("model", "whisper-large-v3-turbo")
    if target_stt == "groq" and groq_key:
        stt = GroqWhisperSTT(api_key=groq_key, model=groq_stt_model)
    elif target_stt == "deepgram" and deepgram_key:
        stt = DeepgramSTT(api_key=deepgram_key)
    elif groq_key:
        stt = GroqWhisperSTT(api_key=groq_key, model=groq_stt_model)
    elif deepgram_key:
        stt = DeepgramSTT(api_key=deepgram_key)
    else:
        stt = MockSTT()

    # LLM Adapter Resolution: Groq -> Gemini -> DeepSeek -> Mock
    default_llm_cfg = config.get("pipeline", {}).get("default_llm", "groq")
    target_llm = llm_choice or default_llm_cfg
    groq_llm_model = config.get("llm", {}).get("groq", {}).get("model", "qwen/qwen3.8-27b")

    if target_llm == "groq" and groq_key:
        llm = GroqLLM(api_key=groq_key, model=groq_llm_model)
    elif target_llm == "gemini" and gemini_key:
        llm = GeminiLLM(api_key=gemini_key)
    elif target_llm == "deepseek" and deepseek_key:
        llm = DeepSeekLLM(api_key=deepseek_key)
    elif groq_key:
        llm = GroqLLM(api_key=groq_key, model=groq_llm_model)
    else:
        llm = MockLLM()

    # TTS Adapter Resolution: Cartesia -> Sarvam -> EdgeTTS (Free zero-key)
    cartesia_key = os.getenv("CARTESIA_API_KEY", "")
    cartesia_cfg = config.get("tts", {}).get("cartesia", {})

    if tts_choice == "cartesia" and cartesia_key:
        tts = CartesiaTTS(
            api_key=cartesia_key,
            model_id=cartesia_cfg.get("model_id", "sonic-multilingual"),
            voice_id=cartesia_cfg.get("voice_id", "694f120f-baa9-4938-8996-9b603e30dceb"),
        )
    elif tts_choice == "sarvam" and sarvam_key:
        tts = SarvamTTS(api_key=sarvam_key)
    else:
        tts = EdgeTTSAdapter(voice=config["tts"]["edgetts"]["voice"])

    return stt, llm, tts

@app.get("/")
def root():
    return {
        "status": "online",
        "service": "Voice AI Calling Agent Core Engine",
        "supported_telephony": ["Exotel", "Plivo", "Twilio", "Browser-WebSockets"],
        "active_tenants": tenant_store.list_tenants()
    }

@app.get("/api/config")
def get_config():
    stt_def = config["pipeline"]["default_stt"]
    llm_def = config["pipeline"]["default_llm"]
    tts_def = config["pipeline"]["default_tts"]
    est_cost = CostEstimator.estimate_minute_cost(stt_def, llm_def, tts_def)
    
    return {
        "pipeline_defaults": {
            "stt": stt_def,
            "llm": llm_def,
            "tts": tts_def
        },
        "unit_economics": {
            "estimated_cost_per_minute_inr": f"₹{est_cost}",
            "recommended_retail_price_inr": "₹3.50",
            "projected_gross_margin": f"{round(((3.50 - est_cost) / 3.50) * 100, 1)}%"
        },
        "tenants": tenant_store.list_tenants()
    }

@app.post("/api/tenant/onboard")
async def onboard_company(
    company_name: str = Form(...),
    agent_name: str = Form("Sneha"),
    language_mode: str = Form("Hinglish"),
    primary_goal: Optional[str] = Form(None),
    document_text: Optional[str] = Form(None),
    pdf_file: Optional[UploadFile] = File(None)
):
    """
    Dynamic Instant Onboarding:
    Upload company PDF or text profile -> Instantly compiles into a custom Voice AI Telecaller.
    """
    raw_content = ""
    if pdf_file:
        content_bytes = await pdf_file.read()
        raw_content = DocumentExtractor.extract_from_pdf(content_bytes)
    elif document_text:
        raw_content = DocumentExtractor.extract_from_text(document_text)
    else:
        raise HTTPException(status_code=400, detail="Provide either document_text or pdf_file")

    tenant_id = company_name.lower().replace(" ", "-") + "-" + os.urandom(3).hex()
    
    profile = PromptBuilder.synthesize_profile(
        tenant_id=tenant_id,
        company_name=company_name,
        extracted_text=raw_content,
        agent_name=agent_name,
        language_mode=language_mode,
        primary_goal=primary_goal
    )
    
    tenant_store.save_profile(profile)
    
    return {
        "success": True,
        "tenant_id": tenant_id,
        "agent_name": profile.agent_name,
        "company_name": profile.company_name,
        "greeting": profile.greeting,
        "primary_goal": profile.primary_goal,
        "objection_rules_count": len(profile.objection_matrix),
        "test_url": f"/test?tenant={tenant_id}"
    }

@app.get("/api/tenant/{tenant_id}")
def get_tenant_profile(tenant_id: str):
    profile = tenant_store.get_profile(tenant_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Tenant not found")
    return profile

# ==============================================================================
# TELEPHONY & REAL-TIME AUDIO WEBSOCKET ENDPOINTS
# ==============================================================================

@app.websocket("/ws/audio/{tenant_id}")
async def universal_audio_stream(websocket: WebSocket, tenant_id: str):
    """
    Universal Bidirectional Audio WebSocket for Browser Testing & General SIP Gateways.
    """
    await websocket.accept()
    profile = tenant_store.get_profile(tenant_id)
    if not profile:
        profile = tenant_store.get_profile("real-estate-demo")

    stt, llm, tts = build_adapters()
    pipeline = VoicePipeline(stt=stt, llm=llm, tts=tts, agent_profile=profile)

    # Outbound callback to stream audio chunks back over WebSocket
    def on_outbound_audio(chunk: bytes):
        async def _safe_send_bytes():
            try:
                await websocket.send_bytes(chunk)
            except Exception:
                pass
        asyncio.create_task(_safe_send_bytes())

    def on_event(event_type: str, data: dict):
        async def _safe_send_text():
            try:
                await websocket.send_text(json.dumps({"event": event_type, "data": data}))
            except Exception:
                pass
        asyncio.create_task(_safe_send_text())

    pipeline.set_callbacks(outbound_audio_callback=on_outbound_audio, event_callback=on_event)

    # Trigger Initial Agent Greeting
    asyncio.create_task(pipeline.trigger_greeting())

    try:
        while True:
            data = await websocket.receive()
            if "bytes" in data and data["bytes"]:
                pipeline.process_incoming_audio(data["bytes"])
            elif "text" in data and data["text"]:
                msg = json.loads(data["text"])
                if msg.get("action") == "interrupt":
                    pipeline._on_barge_in()
                elif msg.get("action") == "text_input":
                    # Direct text turn for testing without microphone
                    text = msg.get("text", "")
                    if text:
                        asyncio.create_task(pipeline.process_text_turn(text))
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"WS error: {e}")

@app.websocket("/ws/exotel/{tenant_id}")
async def exotel_voice_stream(websocket: WebSocket, tenant_id: str):
    """
    Exotel Voice Stream WebSocket Bridge.
    """
    await websocket.accept()
    profile = tenant_store.get_profile(tenant_id) or tenant_store.get_profile("real-estate-demo")
    stt, llm, tts = build_adapters()
    pipeline = VoicePipeline(stt=stt, llm=llm, tts=tts, agent_profile=profile)

    handler = ExotelStreamHandler(on_audio_chunk=pipeline.process_incoming_audio)

    def on_outbound_audio(chunk: bytes):
        msg = handler.create_media_response(chunk)
        asyncio.create_task(websocket.send_text(msg))

    def on_event(event_type: str, data: dict):
        if event_type == "barge_in_interrupted":
            clear_msg = handler.create_clear_response()
            asyncio.create_task(websocket.send_text(clear_msg))

    pipeline.set_callbacks(outbound_audio_callback=on_outbound_audio, event_callback=on_event)

    try:
        while True:
            text_data = await websocket.receive_text()
            res = handler.handle_message(text_data)
            if res and res.get("type") == "start":
                asyncio.create_task(pipeline.trigger_greeting())
            elif res and res.get("type") == "stop":
                break
    except WebSocketDisconnect:
        pass

# ==============================================================================
# INTERACTIVE BROWSER VOICE TEST CONSOLE
# ==============================================================================

STATIC_HTML_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "index.html")

@app.get("/test")
def test_console():
    from fastapi.responses import FileResponse
    if os.path.exists(STATIC_HTML_PATH):
        return FileResponse(STATIC_HTML_PATH)
    return HTMLResponse("<h1>Voice AI Test Console: static/index.html not found</h1>")

if __name__ == "__main__":
    uvicorn.run("src.server:app", host=config["server"]["host"], port=config["server"]["port"], reload=True)
