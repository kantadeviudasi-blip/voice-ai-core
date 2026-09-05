import struct
from abc import ABC, abstractmethod

# Standard G.711 mu-law decoding table
_ULAW_DECODE_TABLE = []
for i in range(256):
    u = ~i
    sign = (u & 0x80)
    exponent = (u & 0x70) >> 4
    mantissa = (u & 0x0F)
    sample = (mantissa << 3) + 132
    sample <<= exponent
    sample -= 132
    if sign:
        sample = -sample
    _ULAW_DECODE_TABLE.append(sample)

# Standard G.711 mu-law encoding lookup
def _linear2ulaw(sample: int) -> int:
    BIAS = 0x84
    CLIP = 32635
    sign = 0
    if sample < 0:
        sample = -sample
        sign = 0x80
    if sample > CLIP:
        sample = CLIP
    sample += BIAS
    
    # Exponent search
    exponent = 7
    for exp, mask in enumerate([0x4000, 0x2000, 0x1000, 0x0800, 0x0400, 0x0200, 0x0100, 0x0080]):
        if sample & mask:
            exponent = 7 - exp
            break
    
    mantissa = (sample >> (exponent + 3)) & 0x0F
    ulawbyte = ~(sign | (exponent << 4) | mantissa) & 0xFF
    return ulawbyte

class BaseTelephonyBridge(ABC):
    """
    Telephony Bridge to decode inbound 8kHz mu-law streams and encode outbound AI audio.
    Compatible with Python 3.10 through 3.14+ (No legacy audioop dependency).
    """
    def __init__(self, sample_rate: int = 8000):
        self.sample_rate = sample_rate

    @staticmethod
    def mulaw_to_pcm16(mulaw_bytes: bytes) -> bytes:
        """Converts standard 8kHz mu-law telephony audio to 16kHz PCM linear audio."""
        pcm16k_samples = bytearray()
        for b in mulaw_bytes:
            sample = _ULAW_DECODE_TABLE[b]
            packed = struct.pack("<h", sample)
            # Upsample 8kHz to 16kHz (2x linear interpolation / sample duplication)
            pcm16k_samples.extend(packed)
            pcm16k_samples.extend(packed)
        return bytes(pcm16k_samples)

    @staticmethod
    def pcm_to_mulaw(pcm_bytes: bytes, sample_rate: int = 8000) -> bytes:
        """
        Converts PCM audio (8kHz, 16kHz, or 24kHz) to standard 8kHz mu-law for telephony playback.
        Automatically strips RIFF/WAVE container header if present.
        """
        if pcm_bytes.startswith(b"RIFF") and len(pcm_bytes) >= 44:
            pcm_bytes = pcm_bytes[44:]

        step = 2
        if sample_rate == 16000:
            step = 4
        elif sample_rate == 24000:
            step = 6

        mulaw_bytes = bytearray()
        for i in range(0, len(pcm_bytes) - 1, step):
            if i + 2 <= len(pcm_bytes):
                sample = struct.unpack("<h", pcm_bytes[i:i+2])[0]
                mulaw_bytes.append(_linear2ulaw(sample))
        return bytes(mulaw_bytes)

    @staticmethod
    def pcm8k_to_mulaw(pcm8k_bytes: bytes) -> bytes:
        """Converts 8kHz PCM audio directly to 8kHz mu-law without downsampling."""
        return BaseTelephonyBridge.pcm_to_mulaw(pcm8k_bytes, sample_rate=8000)

    @staticmethod
    def pcm16_to_mulaw(pcm16k_bytes: bytes) -> bytes:
        """Converts 16kHz PCM audio to 8kHz mu-law for telephony playback."""
        return BaseTelephonyBridge.pcm_to_mulaw(pcm16k_bytes, sample_rate=16000)
