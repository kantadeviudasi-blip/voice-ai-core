import os
import yaml
from typing import Dict, Any, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import BaseModel, Field, field_validator

class ServerConfig(BaseModel):
    host: str = "0.0.0.0"
    port: int = Field(8000, ge=1024, le=65535)
    log_level: str = "info"

class PipelineConfig(BaseModel):
    default_stt: str
    default_llm: str
    default_tts: str

class DeepgramSTTConfig(BaseModel):
    api_key: Optional[str] = None
    model: str = "nova-3"
    language: str = "hi"

class STTConfig(BaseModel):
    deepgram: DeepgramSTTConfig = Field(default_factory=DeepgramSTTConfig)

class GeminiLLMConfig(BaseModel):
    api_key: Optional[str] = None
    model: str = "gemini-3.8-flash"
    temperature: float = 0.25
    max_tokens: int = 150

class LLMConfig(BaseModel):
    gemini: GeminiLLMConfig = Field(default_factory=GeminiLLMConfig)

class DeepgramTTSConfig(BaseModel):
    api_key: Optional[str] = None
    model: str = "aura-2-asteria-en"
    sample_rate: int = 24000

class SarvamTTSConfig(BaseModel):
    api_key: Optional[str] = None
    model: str = "bulbul:v3"
    speaker: str = "ritu"
    target_language_code: str = "hi-IN"
    sample_rate: int = 8000
    pace: float = 1.0

class TTSConfig(BaseModel):
    deepgram: DeepgramTTSConfig = Field(default_factory=DeepgramTTSConfig)
    sarvam: SarvamTTSConfig = Field(default_factory=SarvamTTSConfig)

class VADConfig(BaseModel):
    chunk_duration_ms: int = Field(20, ge=10)
    sample_rate: int = Field(16000)
    silence_threshold_ms: int = Field(450, ge=100)
    confidence_threshold: float = Field(0.55, ge=0.0, le=1.0)
    speech_onset_ms: int = Field(100, ge=50)
    min_utterance_speech_ms: int = Field(250, ge=100)
    barge_in_enabled: bool = True

    @field_validator('sample_rate')
    def validate_sample_rate(cls, v):
        if v not in (8000, 16000, 48000):
            raise ValueError("Sample rate must be 8000, 16000, or 48000")
        return v

class TelephonyConfig(BaseModel):
    provider: str = "sip"
    sample_rate: int = 8000

class AppSettings(BaseSettings):
    server: ServerConfig = Field(default_factory=ServerConfig)
    pipeline: PipelineConfig
    stt: STTConfig = Field(default_factory=STTConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    telephony: TelephonyConfig = Field(default_factory=TelephonyConfig)

    # Env variables mapping
    GEMINI_API_KEY: Optional[str] = None
    DEEPGRAM_API_KEY: Optional[str] = None
    SARVAM_API_KEY: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), ".env"),
        env_file_encoding='utf-8',
        extra='ignore'
    )

def load_config() -> AppSettings:
    """Loads config.yaml and applies Pydantic validation."""
    config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        raw_config = yaml.safe_load(f)
    
    settings = AppSettings(**raw_config)

    if settings.DEEPGRAM_API_KEY:
        settings.stt.deepgram.api_key = settings.DEEPGRAM_API_KEY
        settings.tts.deepgram.api_key = settings.DEEPGRAM_API_KEY
    if settings.GEMINI_API_KEY:
        settings.llm.gemini.api_key = settings.GEMINI_API_KEY
    if settings.SARVAM_API_KEY:
        settings.tts.sarvam.api_key = settings.SARVAM_API_KEY

    # Fail fast if default keys are missing
    if settings.pipeline.default_llm == "gemini" and not settings.llm.gemini.api_key:
        raise ValueError("GEMINI_API_KEY is required")
    if settings.pipeline.default_stt == "deepgram" and not settings.stt.deepgram.api_key:
        raise ValueError("DEEPGRAM_API_KEY is required for STT")
    if settings.pipeline.default_tts == "deepgram" and not settings.tts.deepgram.api_key:
        raise ValueError("DEEPGRAM_API_KEY is required for TTS")
    if settings.pipeline.default_tts == "sarvam" and not settings.tts.sarvam.api_key:
        raise ValueError("SARVAM_API_KEY is required for Sarvam TTS")

    return settings

config = load_config()
