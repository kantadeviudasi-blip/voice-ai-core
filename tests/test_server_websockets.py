import json
import pytest
from fastapi.testclient import TestClient
from src.server import app

client = TestClient(app)

def test_rest_endpoints():
    # Test Root
    res = client.get("/")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "online"
    assert "Exotel" in data["supported_telephony"]

    # Test Config
    res_cfg = client.get("/api/config")
    assert res_cfg.status_code == 200
    assert "unit_economics" in res_cfg.json()

    # Test Onboarding
    res_onboard = client.post("/api/tenant/onboard", data={
        "company_name": "Test RealEstate Co",
        "agent_name": "Riya",
        "language_mode": "Hinglish",
        "document_text": "We sell 2BHK and 3BHK flats in Mumbai starting at 80 Lakhs."
    })
    assert res_onboard.status_code == 200
    onboard_data = res_onboard.json()
    assert onboard_data["success"] is True
    assert onboard_data["agent_name"] == "Riya"
    assert "test_url" in onboard_data

def test_browser_websocket_flow_and_barge_in():
    with client.websocket_connect("/ws/audio/test-tenant") as ws:
        # 1. Send text_input action
        ws.send_text(json.dumps({"action": "text_input", "text": "Flat ki price kya hai?"}))
        
        # Receive events and audio bytes
        events_received = []
        for _ in range(5):
            try:
                msg = ws.receive()
                if "text" in msg and msg["text"]:
                    data = json.loads(msg["text"])
                    events_received.append(data.get("event"))
                elif "bytes" in msg and msg["bytes"]:
                    events_received.append("audio_bytes")
            except Exception:
                break

        # 2. Trigger Barge-in interrupt
        ws.send_text(json.dumps({"action": "interrupt"}))
        
        # Verify barge_in_interrupted event received
        found_interrupt = False
        for _ in range(5):
            try:
                msg = ws.receive()
                if "text" in msg and msg["text"]:
                    data = json.loads(msg["text"])
                    if data.get("event") == "barge_in_interrupted":
                        found_interrupt = True
                        break
            except Exception:
                break
        
        assert found_interrupt is True

def test_dynamic_sample_rate_and_channel_negotiation():
    # 1. Test mobile channel negotiation (16kHz)
    with client.websocket_connect("/ws/audio/test-tenant?channel=mobile&sample_rate=16000") as ws:
        msg = ws.receive()
        assert "text" in msg
        data = json.loads(msg["text"])
        assert data.get("event") == "session_config"
        assert data.get("data", {}).get("channel") == "mobile"
        assert data.get("data", {}).get("sample_rate") == 16000

    # 2. Test telephony channel negotiation (8kHz)
    with client.websocket_connect("/ws/audio/test-tenant?channel=telephony&sample_rate=8000") as ws:
        msg = ws.receive()
        assert "text" in msg
        data = json.loads(msg["text"])
        assert data.get("event") == "session_config"
        assert data.get("data", {}).get("channel") == "telephony"
        assert data.get("data", {}).get("sample_rate") == 8000

def test_direct_sip_websocket_flow():
    with client.websocket_connect("/ws/sip/test-tenant") as ws:
        # Send audio media chunk (binary linear PCM)
        ws.send_bytes(b"\x00\x00" * 320)
        
        # Send text hangup event
        stop_msg = {
            "event": "hangup"
        }
        ws.send_text(json.dumps(stop_msg))

def test_exotel_websocket_flow():
    with client.websocket_connect("/ws/exotel/test-tenant") as ws:
        start_msg = {
            "event": "start",
            "stream_sid": "EX_TEST_5678",
            "start": {"call_sid": "CA_EX_5678"}
        }
        ws.send_text(json.dumps(start_msg))
        
        media_msg = {
            "event": "media",
            "stream_sid": "EX_TEST_5678",
            "media": {"payload": "//////8="}
        }
        ws.send_text(json.dumps(media_msg))
        
        stop_msg = {
            "event": "stop",
            "stream_sid": "EX_TEST_5678"
        }
        ws.send_text(json.dumps(stop_msg))

def test_plivo_websocket_flow():
    with client.websocket_connect("/ws/plivo/test-tenant") as ws:
        start_msg = {
            "event": "start",
            "stream_id": "PL_TEST_9999"
        }
        ws.send_text(json.dumps(start_msg))
        
        media_msg = {
            "event": "media",
            "media": {"payload": "//////8="}
        }
        ws.send_text(json.dumps(media_msg))
        
        stop_msg = {
            "event": "stop"
        }
        ws.send_text(json.dumps(stop_msg))
