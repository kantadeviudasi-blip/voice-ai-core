"""Quick test to verify Gemini + Deepgram TTS APIs are working."""
import asyncio
import json
import os
from dotenv import load_dotenv
load_dotenv()

async def test_gemini():
    print("=== Testing Gemini API ===")
    try:
        from google import genai
        api_key = os.environ.get("GEMINI_API_KEY")
        print(f"API Key present: {bool(api_key)}, length: {len(api_key) if api_key else 0}")
        client = genai.Client(api_key=api_key)
        
        # Try multiple models to find one that works
        for model_name in ["gemini-3.8-flash", "gemini-2.5-flash", "gemini-2.0-flash", "gemini-1.5-flash"]:
            try:
                print(f"  Trying model: {model_name}...")
                chat = client.aio.chats.create(model=model_name)
                response = await chat.send_message("Say hello in one sentence.")
                print(f"  Response: {response.text}")
                print(f"  OK - Working model: {model_name}")
                return model_name
            except Exception as e:
                err_str = str(e)
                if "404" in err_str or "not available" in err_str.lower() or "NOT_FOUND" in err_str:
                    print(f"  Model {model_name} not available")
                else:
                    print(f"  Model {model_name} error: {err_str[:100]}")
        
        print("  FAILED: No working model found")
        return None
    except Exception as e:
        print(f"  Gemini setup FAILED: {e}")
        return None

async def test_deepgram_tts():
    print("\n=== Testing Deepgram TTS ===")
    try:
        import websockets
        api_key = os.environ.get("DEEPGRAM_API_KEY")
        print(f"API Key present: {bool(api_key)}, length: {len(api_key) if api_key else 0}")
        tts_url = "wss://api.deepgram.com/v1/speak?model=aura-asteria-en&encoding=linear16&sample_rate=24000"
        headers = {"Authorization": f"Token {api_key}"}
        
        async with websockets.connect(tts_url, additional_headers=headers) as ws:
            print("  TTS WebSocket connected")
            
            await ws.send(json.dumps({"type": "Speak", "text": "Hello, this is a test."}))
            await ws.send(json.dumps({"type": "Flush"}))
            print("  Sent text to TTS")
            
            total_bytes = 0
            try:
                async for msg in ws:
                    if isinstance(msg, bytes):
                        total_bytes += len(msg)
                        if total_bytes > 1000:
                            break
                    else:
                        data = json.loads(msg)
                        print(f"  TTS control message: {data}")
            except Exception as e:
                print(f"  TTS receive ended: {e}")
            
            print(f"  Received {total_bytes} bytes of audio")
            if total_bytes > 0:
                print("  OK - Deepgram TTS working")
            else:
                print("  FAILED - Deepgram TTS returned no audio")
    except Exception as e:
        print(f"  Deepgram TTS FAILED: {e}")

async def main():
    working_model = await test_gemini()
    await test_deepgram_tts()
    if working_model:
        print(f"\n=== RESULT: Use model '{working_model}' in pipeline.py ===")

asyncio.run(main())
