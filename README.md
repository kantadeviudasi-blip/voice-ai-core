# 🎙️ Voice AI Calling Agent Platform (Indian Business Focus)

High-concurrency, ultra-low latency, and cost-optimized Voice Calling AI Agent platform engineered for Indian businesses (Real Estate, Clinics, Solar, EdTech, Debt Collection, etc.).

---

## ⚡ Key Highlights
- **Sub-Second Latency**: ~500ms – 700ms end-to-end response time.
- **Ultra-Low Operating Cost**: Strictly **₹0.94 – ₹1.12 per call minute** (Telephony + STT + LLM + TTS + Server).
- **80%+ Gross Margin**: Billed to clients at ₹3.50 – ₹4.00/min or ₹8,000 – ₹10,000/month retainers.
- **Hinglish & Vernacular Support**: Code-mixing, Indian accents, and local dialect understanding.
- **Dynamic Instant Onboarding**: Upload any company PDF/profile text $\rightarrow$ Instantly generates a customized telecaller persona, objection handling matrix, and greetings.
- **Hot-Swappable Model Adapters**: Switch STT, LLM, and TTS models on the fly via `config.yaml` or API.
- **Telephony Bridges Included**: Out-of-the-box WebSocket support for **Exotel**, **Plivo**, **Twilio**, and Browser WebRTC.

---

## 🏗️ Architecture & Component Selection

| Layer | Recommended Provider | Cost (INR / Min) | Key Advantage |
| :--- | :--- | :--- | :--- |
| **STT** | **Deepgram Nova-2** | ₹0.36 | High accuracy with Hinglish & Indian accents (~150ms) |
| **LLM** | **Groq (Llama 3.3 70B)** / **Gemini 2.0 Flash** | ₹0.10 | ~300 tokens/sec & sub-120ms TTFT |
| **TTS** | **EdgeTTS** / **Sarvam AI (Bulbul)** | ₹0.00 – ₹0.18 | Zero-cost or authentic Indian regional voices |
| **Telephony** | **Exotel** / **Plivo** / **SIP** | ₹0.42 | Indian compliance, DND filtering, CLI routing |
| **VAD** | **Silero VAD / Frame-by-Frame Energy** | Free | Instant Barge-In (Interruption handling) |

---

## 🚀 Quickstart Guide

### 1. Install Dependencies
```bash
cd voice-ai-core
pip install -r requirements.txt
```

### 2. Configure Environment Variables (Optional for Cloud APIs)
```bash
# STT
export DEEPGRAM_API_KEY="your_deepgram_key"

# LLM
export GROQ_API_KEY="your_groq_key"
export GEMINI_API_KEY="your_gemini_key"
export DEEPSEEK_API_KEY="your_deepseek_key"

# TTS
export SARVAM_API_KEY="your_sarvam_key"
```
*(Note: If no API keys are supplied, the platform automatically runs in Mock / EdgeTTS mode so you can test immediately.)*

### 3. Start the Voice Engine & Web Console
```bash
python src/server.py
```
Visit `http://localhost:8000/test` in your browser to interact with the Voice AI agent using your computer microphone!

---

## 📄 Instant Company Onboarding API

To onboard any new business dynamically:

```bash
curl -X POST http://localhost:8000/api/tenant/onboard \
  -F "company_name=Apex Solar Solutions" \
  -F "agent_name=Sneha" \
  -F "language_mode=Hinglish" \
  -F "document_text=We install 3kW and 5kW rooftop solar panels in Jaipur with 40% government subsidy. Free site survey available."
```

Returns:
```json
{
  "success": true,
  "tenant_id": "apex-solar-solutions-3a8f1b",
  "agent_name": "Sneha",
  "greeting": "Namaste! Main Apex Solar Solutions se Sneha baat kar rahi hoon...",
  "primary_goal": "Qualify customer interest and schedule a follow-up or appointment",
  "test_url": "/test?tenant=apex-solar-solutions-3a8f1b"
}
```

---

## 📞 Telephony WebSocket Endpoints

- **Exotel Stream**: `ws://<your-server>:8000/ws/exotel/{tenant_id}`
- **Plivo Stream**: `ws://<your-server>:8000/ws/plivo/{tenant_id}`
- **Twilio Stream**: `ws://<your-server>:8000/ws/twilio/{tenant_id}`
- **Universal / Browser Stream**: `ws://<your-server>:8000/ws/audio/{tenant_id}`
