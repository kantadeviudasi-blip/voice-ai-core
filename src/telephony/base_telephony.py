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
    def pcm16_to_mulaw(pcm16k_bytes: bytes) -> bytes:
        """Converts 16kHz PCM audio to 8kHz mu-law for telephony playback."""
        mulaw_bytes = bytearray()
        # Downsample 16kHz to 8kHz by taking every 2nd sample (4 bytes per 2 samples)
        for i in range(0, len(pcm16k_bytes) - 3, 4):
            sample = struct.unpack("<h", pcm16k_bytes[i:i+2])[0]
            mulaw_byte = _linear2ulaw(sample)
            mulaw_bytes.append(mulaw_byte)
        return bytes(mulaw_bytes)
