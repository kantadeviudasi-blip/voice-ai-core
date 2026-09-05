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
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
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
from src.adapters.llm.gemini_llm import GeminiLLM
from src.adapters.tts.deepgram_tts import DeepgramTTS
from src.adapters.tts.sarvam_tts import SarvamTTS
from src.knowledge.extractor import DocumentExtractor
from src.knowledge.prompt_builder import PromptBuilder, AgentProfile
from src.knowledge.tenant_store import TenantStore
from src.telephony.exotel_handler import ExotelStreamHandler
from src.telephony.plivo_handler import PlivoStreamHandler

from src.core.config import config as app_settings

config = app_settings.model_dump()

app = FastAPI(
    title="Voice AI Calling Agent Platform",
    description="High-concurrency, ultra-low latency, cost-optimized Voice Calling AI platform for Indian businesses.",
    version="1.0.0"
)

# NOTE: StaticFiles is intentionally NOT mounted here at root "/".
# Mounting StaticFiles at root BEFORE WebSocket routes causes all WebSocket
# upgrade requests to be intercepted and rejected with:
#   AssertionError: assert scope["type"] == "http"
# Instead, index.html is served via an explicit GET route below, and
# StaticFiles is mounted at "/static" at the END of the file.
static_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

tenant_store = TenantStore()

def build_adapters(channel: str = "web", sample_rate: Optional[int] = None):
    """
    Factory function to instantiate STT, LLM, and TTS adapters based on active config.yaml
    and dynamic client channel (telephony=8000Hz, mobile=16000Hz, web=24000Hz).
    """
    gemini_key = os.getenv("GEMINI_API_KEY", "")
    deepgram_key = os.getenv("DEEPGRAM_API_KEY", "")
    sarvam_key = os.getenv("SARVAM_API_KEY", "")

    stt_cfg = config.get("stt", {}).get("deepgram", {})
    llm_cfg = config.get("llm", {}).get("gemini", {})
    default_tts = config.get("pipeline", {}).get("default_tts", "sarvam")

    # Determine channel-specific sample rate if not explicitly supplied
    if sample_rate is None:
        if channel == "telephony":
            sample_rate = 8000
        elif channel == "mobile":
            sample_rate = 16000
        else:
            sample_rate = 24000

    stt = DeepgramSTT(
        api_key=deepgram_key,
        model=stt_cfg.get("model", "nova-3"),
        language=stt_cfg.get("language", "hi")
    )
    llm = GeminiLLM(
        api_key=gemini_key,
        model=llm_cfg.get("model", "gemini-3.8-flash"),
        temperature=llm_cfg.get("temperature", 0.25),
        max_tokens=llm_cfg.get("max_tokens", 150)
    )

    if default_tts == "sarvam":
        sarvam_cfg = config.get("tts", {}).get("sarvam", {})
        tts = SarvamTTS(
            api_key=sarvam_key,
            model=sarvam_cfg.get("model", "bulbul:v3"),
            speaker=sarvam_cfg.get("speaker", "ritu"),
            sample_rate=sample_rate,
            target_language_code=sarvam_cfg.get("target_language_code", "hi-IN"),
            pace=sarvam_cfg.get("pace", 1.0)
        )
    else:
        tts_cfg = config.get("tts", {}).get("deepgram", {})
        tts = DeepgramTTS(
            api_key=deepgram_key,
            model=tts_cfg.get("model", "aura-2-asteria-en"),
            sample_rate=sample_rate
        )

    return stt, llm, tts

def create_pipeline(profile, channel: str = "web", sample_rate: Optional[int] = None) -> VoicePipeline:
    """Factory helper to build a fully configured VoicePipeline with adaptive VAD & noise rejection."""
    stt, llm, tts = build_adapters(channel=channel, sample_rate=sample_rate)
    vad_cfg = config.get("vad", {})
    return VoicePipeline(
        stt=stt,
        llm=llm,
        tts=tts,
        agent_profile=profile,
        sample_rate=vad_cfg.get("sample_rate", 16000),
        silence_threshold_ms=vad_cfg.get("silence_threshold_ms", 450),
        energy_threshold=vad_cfg.get("energy_threshold", 0.022),
        barge_in_enabled=vad_cfg.get("barge_in_enabled", True),
        speech_onset_ms=vad_cfg.get("speech_onset_ms", 120.0),
        min_utterance_speech_ms=vad_cfg.get("min_utterance_speech_ms", 300.0)
    )

@app.get("/api/health")
def health_check():
    """Health check endpoint — returns platform status and active tenants."""
    return {
        "status": "online",
        "service": "Voice AI Calling Agent Core Engine",
        "supported_telephony": ["FreeSWITCH/Asterisk-SIP", "Exotel", "Plivo", "Browser-WebSockets"],
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
async def universal_audio_stream(
    websocket: WebSocket,
    tenant_id: str,
    channel: Optional[str] = None,
    sample_rate: Optional[int] = None
):
    """
    Universal Bidirectional Audio WebSocket for Browser Testing & General SIP Gateways.
    Dynamically supports 8kHz (telephony), 16kHz (mobile HD), and 24kHz (web studio).
    Can be controlled via query params (?channel=mobile or ?sample_rate=16000),
    User-Agent auto-detection, or environment variable AUDIO_SAMPLE_RATE.
    """
    await websocket.accept()
    profile = tenant_store.get_profile(tenant_id)
    if not profile:
        profile = tenant_store.get_profile("apex-solar-solutions")

    # 1. Determine active channel & sample_rate dynamically
    active_channel = channel or os.getenv("AUDIO_CHANNEL", "").strip() or None
    active_sr = sample_rate
    if active_sr is None and os.getenv("AUDIO_SAMPLE_RATE"):
        try:
            active_sr = int(os.getenv("AUDIO_SAMPLE_RATE", "0"))
        except ValueError:
            pass

    # 2. Auto-detect mobile device from User-Agent if not specified
    user_agent = websocket.headers.get("user-agent", "").lower()
    if not active_channel and any(m in user_agent for m in ["mobi", "android", "iphone", "ipad"]):
        active_channel = "mobile"

    # 3. Fall back to standard defaults
    if not active_channel:
        active_channel = "web"

    if active_sr not in (8000, 16000, 24000):
        if active_channel == "telephony":
            active_sr = 8000
        elif active_channel == "mobile":
            active_sr = 16000
        else:
            active_sr = 24000

    pipeline = create_pipeline(profile, channel=active_channel, sample_rate=active_sr)

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

    # Initialize streaming STT if supported
    await pipeline.start()

    # Emit session negotiated configuration
    on_event("session_config", {"channel": active_channel, "sample_rate": active_sr})

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
    finally:
        await pipeline.close()

@app.websocket("/ws/exotel/{tenant_id}")
async def exotel_voice_stream(websocket: WebSocket, tenant_id: str):
    """
    Exotel Voice Stream WebSocket Bridge.
    """
    await websocket.accept()
    profile = tenant_store.get_profile(tenant_id) or tenant_store.get_profile("apex-solar-solutions")
    pipeline = create_pipeline(profile, channel="telephony", sample_rate=8000)

    handler = ExotelStreamHandler(on_audio_chunk=pipeline.process_incoming_audio)

    def on_outbound_audio(chunk: bytes):
        msg = handler.create_media_response(chunk)
        asyncio.create_task(websocket.send_text(msg))

    def on_event(event_type: str, data: dict):
        if event_type == "barge_in_interrupted":
            clear_msg = handler.create_clear_response()
            asyncio.create_task(websocket.send_text(clear_msg))

    pipeline.set_callbacks(outbound_audio_callback=on_outbound_audio, event_callback=on_event)
    await pipeline.start()

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
    finally:
        await pipeline.close()

@app.websocket("/ws/sip/{tenant_id}")
async def direct_sip_voice_stream(websocket: WebSocket, tenant_id: str):
    """
    Direct FreeSWITCH / Asterisk / Kamailio SIP Trunking WebSocket Bridge (mod_audio_fork).
    Direct low-latency μ-law / linear PCM media stream without third-party aggregator markups.
    """
    await websocket.accept()
    profile = tenant_store.get_profile(tenant_id) or tenant_store.get_profile("apex-solar-solutions")
    pipeline = create_pipeline(profile, channel="telephony", sample_rate=8000)

    # Outbound callback to send raw audio frames back to the SIP Gateway
    def on_outbound_audio(chunk: bytes):
        async def _safe_send_bytes():
            try:
                await websocket.send_bytes(chunk)
            except Exception:
                pass
        asyncio.create_task(_safe_send_bytes())

    def on_event(event_type: str, data: dict):
        if event_type == "barge_in_interrupted":
            async def _safe_send_clear():
                try:
                    await websocket.send_text(json.dumps({"event": "clear_buffer"}))
                except Exception:
                    pass
            asyncio.create_task(_safe_send_clear())

    pipeline.set_callbacks(outbound_audio_callback=on_outbound_audio, event_callback=on_event)
    await pipeline.start()

    # Greet upon SIP session connect
    asyncio.create_task(pipeline.trigger_greeting())

    try:
        while True:
            data = await websocket.receive()
            if "bytes" in data and data["bytes"]:
                pipeline.process_incoming_audio(data["bytes"])
            elif "text" in data and data["text"]:
                msg = json.loads(data["text"])
                if msg.get("event") == "hangup" or msg.get("type") == "stop":
                    break
    except WebSocketDisconnect:
        pass
    finally:
        await pipeline.close()

@app.websocket("/ws/plivo/{tenant_id}")
async def plivo_voice_stream(websocket: WebSocket, tenant_id: str):
    """
    Plivo Audio Streams WebSocket Bridge.
    """
    await websocket.accept()
    profile = tenant_store.get_profile(tenant_id) or tenant_store.get_profile("apex-solar-solutions")
    pipeline = create_pipeline(profile, channel="telephony", sample_rate=8000)

    handler = PlivoStreamHandler(on_audio_chunk=pipeline.process_incoming_audio)

    def on_outbound_audio(chunk: bytes):
        msg = handler.create_media_response(chunk)
        asyncio.create_task(websocket.send_text(msg))

    def on_event(event_type: str, data: dict):
        if event_type == "barge_in_interrupted":
            clear_msg = handler.create_clear_response()
            asyncio.create_task(websocket.send_text(clear_msg))

    pipeline.set_callbacks(outbound_audio_callback=on_outbound_audio, event_callback=on_event)
    await pipeline.start()

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
    finally:
        await pipeline.close()

# ==============================================================================
# STATIC FILE SERVING  (MUST BE LAST — after all WebSocket routes are defined)
# ==============================================================================
# CRITICAL: app.mount() with StaticFiles must come AFTER all @app.websocket()
# routes. If mounted at root "/" before websocket routes, StaticFiles intercepts
# every WebSocket upgrade and crashes: assert scope["type"] == "http"
# ==============================================================================

STATIC_HTML_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static", "index.html")

@app.get("/")
@app.get("/test")
def test_console(request: Request):
    """
    Serves the browser Voice AI Test Console UI (index.html) for web browsers,
    or JSON health status for API/test clients requesting root.
    """
    accept_header = request.headers.get("accept", "")
    if request.url.path == "/" and "text/html" not in accept_header:
        return health_check()

    if os.path.exists(STATIC_HTML_PATH):
        return FileResponse(STATIC_HTML_PATH)
    return HTMLResponse("<h1>Voice AI Test Console: static/index.html not found</h1>")

# Mount static assets (JS, CSS, images) at /static prefix — SAFE because it
# does NOT conflict with root WebSocket paths like /ws/browser/...
app.mount("/static", StaticFiles(directory=static_path), name="static_assets")

if __name__ == "__main__":
    uvicorn.run("src.server:app", host=config["server"]["host"], port=config["server"]["port"], reload=True)

