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

class GroqSTTConfig(BaseModel):
    api_key: Optional[str] = None
    model: str = "whisper-large-v3-turbo"
    language: str = "hi"
    prompt: Optional[str] = None

class STTConfig(BaseModel):
    groq: GroqSTTConfig = Field(default_factory=GroqSTTConfig)

class GroqLLMConfig(BaseModel):
    api_key: Optional[str] = None
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.25
    max_tokens: int = 150

class GeminiLLMConfig(BaseModel):
    api_key: Optional[str] = None
    model: str = "gemini-2.0-flash"
    temperature: float = 0.25
    max_tokens: int = 150

class DeepSeekLLMConfig(BaseModel):
    api_key: Optional[str] = None
    base_url: str = "https://api.deepseek.com"
    model: str = "deepseek-chat"
    temperature: float = 0.25

class LLMConfig(BaseModel):
    groq: GroqLLMConfig = Field(default_factory=GroqLLMConfig)
    gemini: GeminiLLMConfig = Field(default_factory=GeminiLLMConfig)
    deepseek: DeepSeekLLMConfig = Field(default_factory=DeepSeekLLMConfig)

class EdgeTTSConfig(BaseModel):
    voice: str = "hi-IN-SwaraNeural"
    rate: str = "+12%"
    pitch: str = "+2Hz"

class SarvamTTSConfig(BaseModel):
    api_key: Optional[str] = None
    speaker: str = "meera"
    language_code: str = "hi-IN"
    model: str = "bulbul:v1"

class CartesiaTTSConfig(BaseModel):
    api_key: Optional[str] = None
    model_id: str = "sonic-multilingual"
    voice_id: str = "694f120f-baa9-4938-8996-9b603e30dceb"

class TTSConfig(BaseModel):
    edgetts: EdgeTTSConfig = Field(default_factory=EdgeTTSConfig)
    sarvam: SarvamTTSConfig = Field(default_factory=SarvamTTSConfig)
    cartesia: CartesiaTTSConfig = Field(default_factory=CartesiaTTSConfig)

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
    # App Settings
    server: ServerConfig = Field(default_factory=ServerConfig)
    pipeline: PipelineConfig
    stt: STTConfig = Field(default_factory=STTConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    telephony: TelephonyConfig = Field(default_factory=TelephonyConfig)

    # Env variables mapping
    GROQ_API_KEY: Optional[str] = None
    GEMINI_API_KEY: Optional[str] = None
    DEEPSEEK_API_KEY: Optional[str] = None
    SARVAM_API_KEY: Optional[str] = None
    CARTESIA_API_KEY: Optional[str] = None

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
    
    # Initialize Pydantic BaseSettings with raw_config values.
    # Any missing or invalid values will raise ValidationError.
    settings = AppSettings(**raw_config)

    # Map API keys from env to the config objects dynamically if they exist
    if settings.GROQ_API_KEY:
        settings.stt.groq.api_key = settings.GROQ_API_KEY
        settings.llm.groq.api_key = settings.GROQ_API_KEY
    if settings.GEMINI_API_KEY:
        settings.llm.gemini.api_key = settings.GEMINI_API_KEY
    if settings.DEEPSEEK_API_KEY:
        settings.llm.deepseek.api_key = settings.DEEPSEEK_API_KEY
    if settings.SARVAM_API_KEY:
        settings.tts.sarvam.api_key = settings.SARVAM_API_KEY
    if settings.CARTESIA_API_KEY:
        settings.tts.cartesia.api_key = settings.CARTESIA_API_KEY

    # Fail fast if default keys are missing
    if settings.pipeline.default_llm == "groq" and not settings.llm.groq.api_key:
        raise ValueError("GROQ_API_KEY is required when default_llm is 'groq'")
    if settings.pipeline.default_llm == "gemini" and not settings.llm.gemini.api_key:
        raise ValueError("GEMINI_API_KEY is required when default_llm is 'gemini'")
    
    if settings.pipeline.default_stt == "groq" and not settings.stt.groq.api_key:
        raise ValueError("GROQ_API_KEY is required when default_stt is 'groq'")

    return settings

# Singleton instance
config = load_config()
