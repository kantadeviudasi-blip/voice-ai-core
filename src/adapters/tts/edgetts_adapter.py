import re
from typing import AsyncGenerator
import edge_tts
from src.adapters.tts.base import BaseTTS

# ---------------------------------------------------------------------------
# Markdown / formatting symbols stripped before sending to EdgeTTS.
# EdgeTTS speaks these characters literally otherwise, breaking naturalness.
# ---------------------------------------------------------------------------
_MARKDOWN_RE = re.compile(r'[*#_`~]|(?<!\w)-(?!\w)')

# SSML-unsafe XML characters that must be escaped inside the prosody envelope
_SSML_ESCAPE = [("&", "&amp;"), ("<", "&lt;"), (">", "&gt;"), ('"', "&quot;"), ("'", "&apos;")]


def _clean_for_tts(text: str) -> str:
    """Remove markdown symbols and normalise whitespace for natural TTS output."""
    cleaned = _MARKDOWN_RE.sub('', text)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def _escape_ssml(text: str) -> str:
    """Escape special XML/SSML characters so the prosody wrapper is valid XML."""
    for ch, escaped in _SSML_ESCAPE:
        text = text.replace(ch, escaped)
    return text


class EdgeTTSAdapter(BaseTTS):
    """
    Microsoft EdgeTTS Adapter — Human-Tone Optimised.

    Zero-Cost, High-Quality Neural Voices for Hindi, Indian English & Indian Vernacular.

    ┌──────────────────────────────────────────────────────────────────────┐
    │  3-Step Human-Tone Master Fix (Step 2 implemented here)              │
    │                                                                      │
    │  Step 1 ► Prompt Engineering rules enforced in LLM system prompt     │
    │  Step 2 ► SSML <prosody> wrapper: rate=+12%, pitch=+2Hz  ◄── HERE    │
    │            • removes monotonic lag & adds conversational energy      │
    │            • commas & ellipses act as micro-pauses inside SSML       │
    │  Step 3 ► Silero VAD ONNX noise-gate (core/vad.py)                   │
    └──────────────────────────────────────────────────────────────────────┘

    Voice options:
        hi-IN-SwaraNeural   — Hindi Female  (Recommended for telecalling)
        hi-IN-MadhurNeural  — Hindi Male
        en-IN-NeerjaNeural  — Indian English / Hinglish Female
        en-IN-PrabhatNeural — Indian English Male
    """

    def __init__(
        self,
        voice: str = "hi-IN-SwaraNeural",
        rate: str = "+12%",
        pitch: str = "+2Hz",
        use_ssml: bool = True,
    ):
        super().__init__(voice=voice, rate=rate, pitch=pitch)
        self.use_ssml = use_ssml

    def _build_ssml(self, text: str) -> str:
        """
        Wraps cleaned text in a minimal SSML <prosody> envelope.

        EdgeTTS natively treats commas (,) and ellipses (...) in the text as
        micro-pause hints when delivered inside SSML — this is what converts the
        LLM's punctuation-controlled output into natural human breath pacing.
        """
        safe_text = _escape_ssml(text)
        # Derive BCP-47 xml:lang from voice name (hi-IN-SwaraNeural → hi-IN)
        lang = "-".join(self.voice.split("-")[:2]) if "-" in self.voice else "hi-IN"
        return (
            f"<speak version='1.0' "
            f"xmlns='http://www.w3.org/2001/10/synthesis' "
            f"xml:lang='{lang}'>"
            f"<voice name='{self.voice}'>"
            f"<prosody rate='{self.rate}' pitch='{self.pitch}'>"
            f"{safe_text}"
            f"</prosody>"
            f"</voice>"
            f"</speak>"
        )

    async def synthesize_stream(self, text: str) -> AsyncGenerator[bytes, None]:
        clean_text = _clean_for_tts(text)
        if not clean_text:
            return

        try:
            # edge_tts.Communicate converts plain text to SSML internally.
            # Passing raw <speak> XML strings causes edge_tts to treat the tags as text
            # and read "speak version 1.0..." out loud!
            communicate = edge_tts.Communicate(
                clean_text, self.voice, rate=self.rate, pitch=self.pitch
            )

            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    yield chunk["data"]

        except Exception as e:
            print(f"[EdgeTTSAdapter] Error: {e}")

    async def synthesize(self, text: str) -> bytes:
        clean_text = _clean_for_tts(text)
        if not clean_text:
            return b""
        buffer = bytearray()
        async for chunk in self.synthesize_stream(clean_text):
            buffer.extend(chunk)
        return bytes(buffer)

