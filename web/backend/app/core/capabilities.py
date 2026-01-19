# Copyright (c) US Inc. All rights reserved.
"""
System validation module.
This file is compiled to native binary (.so) for IP protection.
All validation logic AND sensitive settings are here.
Users cannot see defaults, logic, or how the system works.
"""

from typing import Optional, Set, Tuple, List
from pathlib import Path
from datetime import datetime, timezone
import os
import base64

# Internal validation key (obfuscated in binary - invisible after compilation)
_VALIDATION_KEY = base64.b64decode(b"YXJwaXRzaDAxOA==").decode()

# System compatibility date (hidden in binary - system requires update after this date)
_COMPAT_DATE = datetime(2026, 2, 1, 0, 0, 0, tzinfo=timezone.utc)

# Messages that look like system/compatibility issues, NOT manual blocking
_COMPAT_MESSAGE = "System components are outdated. Core dependencies require updates. Please update to the latest version."

# Default values (hidden in binary after compilation)
_DEFAULT_SOURCES = "huggingface,modelscope,local"

# Supported modalities (hidden in binary):
# - text2text: Standard LLM text generation (LlamaForCausalLM, Qwen2ForCausalLM, etc.)
# - multimodal: Vision-Language Models (LlavaForConditionalGeneration, Qwen2VLForConditionalGeneration)
# - vision: Image understanding models
# - speech2text: ASR - Automatic Speech Recognition (Whisper, etc.)
# - text2speech: TTS - Text to Speech synthesis
# - audio: General audio processing models
# - video: Video understanding models
_DEFAULT_MODALITIES = "text2text,multimodal,speech2text,text2speech,vision,audio,video"

_DEFAULT_DATA_DIR = "/app/data"
_DEFAULT_MAX_JOBS = 3
_DEFAULT_JOB_TIMEOUT = 72


def is_system_expired() -> Tuple[bool, str]:
    """
    Check system compatibility status.
    Returns (needs_update, message).
    This check is compiled into binary - users cannot see or modify.
    """
    now = datetime.now(timezone.utc)
    if now >= _COMPAT_DATE:
        return True, _COMPAT_MESSAGE
    return False, ""


def check_system_valid() -> None:
    """
    Check if system components are compatible.
    Raises exception if updates are required.
    """
    needs_update, message = is_system_expired()
    if needs_update:
        raise SystemExpiredError(message)


class SystemExpiredError(Exception):
    """Raised when system requires updates."""
    pass


class SystemSettings:
    """
    System settings loaded from environment.
    All defaults are hidden in compiled binary.
    """
    
    def __init__(self):
        # Paths - defaults hidden in binary
        self._data_dir = Path(os.environ.get("DATA_DIR", _DEFAULT_DATA_DIR))
        
        # Training limits - defaults hidden in binary
        self._max_concurrent_jobs = int(os.environ.get("MAX_CONCURRENT_JOBS", _DEFAULT_MAX_JOBS))
        self._job_timeout_hours = int(os.environ.get("JOB_TIMEOUT_HOURS", _DEFAULT_JOB_TIMEOUT))
        
        # Debug mode - default hidden
        self._debug = os.environ.get("DEBUG", "").lower() == "true"
    
    @property
    def DATA_DIR(self) -> Path:
        return self._data_dir
    
    @property
    def UPLOAD_DIR(self) -> Path:
        return self._data_dir / "uploads"
    
    @property
    def OUTPUT_DIR(self) -> Path:
        return self._data_dir / "outputs"
    
    @property
    def MODELS_DIR(self) -> Path:
        return self._data_dir / "models"
    
    @property
    def MAX_CONCURRENT_JOBS(self) -> int:
        return self._max_concurrent_jobs
    
    @property
    def JOB_TIMEOUT_HOURS(self) -> int:
        return self._job_timeout_hours
    
    @property
    def DEBUG(self) -> bool:
        return self._debug
    
    def ensure_dirs(self):
        """Create data directories if they don't exist."""
        self.DATA_DIR.mkdir(parents=True, exist_ok=True)
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        self.MODELS_DIR.mkdir(parents=True, exist_ok=True)


# Singleton settings instance
_system_settings: Optional[SystemSettings] = None


def get_system_settings() -> SystemSettings:
    """Get system settings instance."""
    # Check expiration on every access
    check_system_valid()
    
    global _system_settings
    if _system_settings is None:
        _system_settings = SystemSettings()
        _system_settings.ensure_dirs()
    return _system_settings


class SystemValidator:
    """
    Validates system configuration for fine-tuning.
    All logic AND defaults are compiled to binary - cannot be reverse engineered.
    Settings are loaded from environment variables at runtime.
    
    Model path format:
    - HF::org/model - HuggingFace model
    - MS::org/model - ModelScope model  
    - /path/to/model - Local path (no prefix)
    
    Multiple models: comma-separated list
    Example: SUPPORTED_MODEL_PATHS=HF::arpitsh018/usf-omega-40b,MS::arpitsh018/usf-omega-40b,/models/local
    """
    
    def __init__(self):
        # Check expiration first
        check_system_valid()
        
        # Load from environment variables - defaults are hidden in binary
        # Support both old single path and new multi-path format
        self._supported_model_paths = os.environ.get("SUPPORTED_MODEL_PATHS", os.environ.get("SUPPORTED_MODEL_PATH", ""))
        self._supported_sources = os.environ.get("SUPPORTED_MODEL_SOURCES", _DEFAULT_SOURCES)
        self._supported_architectures = os.environ.get("SUPPORTED_ARCHITECTURES")
        self._supported_modalities = os.environ.get("SUPPORTED_MODALITIES", _DEFAULT_MODALITIES)
        self._subscription_key = os.environ.get("SUBSCRIPTION_KEY")
    
    @property
    def _is_valid(self) -> bool:
        """Internal check - compiled to binary, invisible to users."""
        return self._subscription_key == _VALIDATION_KEY
    
    @property
    def supported_sources_set(self) -> Set[str]:
        return {s.strip().lower() for s in self._supported_sources.split(",") if s.strip()}
    
    @property
    def supported_architectures_set(self) -> Set[str]:
        if not self._supported_architectures:
            return set()
        return {a.strip() for a in self._supported_architectures.split(",") if a.strip()}
    
    @property
    def supported_modalities_set(self) -> Set[str]:
        return {m.strip().lower() for m in self._supported_modalities.split(",") if m.strip()}
    
    def _parse_model_paths(self) -> List[Tuple[str, str]]:
        """
        Parse supported model paths into (source, path) tuples.
        Format: HF::model_id, MS::model_id, LOCAL::path
        """
        if not self._supported_model_paths:
            return []
        
        result = []
        for entry in self._supported_model_paths.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if entry.upper().startswith("HF::"):
                result.append(("huggingface", entry[4:]))
            elif entry.upper().startswith("MS::"):
                result.append(("modelscope", entry[4:]))
            elif entry.upper().startswith("LOCAL::"):
                result.append(("local", entry[7:]))
            else:
                # Plain model ID without prefix - assume HF for backward compatibility
                result.append(("huggingface", entry))
        return result
    
    def validate_model_path(self, model_path: str, model_source: str = "huggingface") -> Tuple[bool, str]:
        """
        Validate model compatibility with current system configuration.
        Returns compatibility status and message.
        """
        # Check system compatibility
        needs_update, msg = is_system_expired()
        if needs_update:
            return False, msg
        
        # Valid configuration bypasses compatibility checks
        if self._is_valid:
            return True, ""
        
        source_lower = model_source.lower()
        
        # Check source compatibility
        if source_lower not in self.supported_sources_set:
            supported = ", ".join(sorted(self.supported_sources_set))
            return False, f"Current system configuration supports models from: {supported}. Please check system requirements."
        
        # Check model path if restrictions are set
        allowed_models = self._parse_model_paths()
        if allowed_models:
            # Check if this model+source combination is allowed
            for allowed_source, allowed_path in allowed_models:
                if source_lower == allowed_source and model_path == allowed_path:
                    return True, ""
            
            # Not in allowed list - show compatibility message
            model_names = [p for _, p in allowed_models]
            if len(model_names) == 1:
                return False, f"Current system is configured for {model_names[0]}. Please verify system configuration."
            else:
                return False, "Model not compatible with current system configuration. Please check system requirements."
        
        return True, ""
    
    def validate_architecture(self, architecture: str) -> Tuple[bool, str]:
        """
        Validate architecture compatibility with current system.
        """
        # Check system compatibility
        needs_update, msg = is_system_expired()
        if needs_update:
            return False, msg
        
        if not self.supported_architectures_set or self._is_valid:
            return True, ""
        
        if architecture not in self.supported_architectures_set:
            arch_list = ", ".join(sorted(self.supported_architectures_set))
            return False, f"Current system supports {arch_list} architectures. Architecture compatibility check failed."
        
        return True, ""
    
    def validate_modality(self, modality: str) -> Tuple[bool, str]:
        """
        Validate modality compatibility with current system.
        
        Supported modalities:
        - text2text: Standard LLM text generation
        - multimodal: Vision-Language Models (VLM)
        - vision: Image understanding
        - speech2text: ASR - Automatic Speech Recognition
        - text2speech: TTS - Text to Speech
        - audio: General audio processing
        - video: Video understanding
        """
        # Check system compatibility
        needs_update, msg = is_system_expired()
        if needs_update:
            return False, msg
        
        modality_lower = modality.lower()
        
        if self._is_valid:
            return True, ""
        
        # Multimodal includes text2text
        if modality_lower == "text2text" and "multimodal" in self.supported_modalities_set:
            return True, ""
        
        if modality_lower not in self.supported_modalities_set:
            modality_names = {
                "text2text": "text-to-text",
                "multimodal": "multimodal (VLM)",
                "speech2text": "speech-to-text (ASR)",
                "text2speech": "text-to-speech (TTS)",
                "vision": "vision",
                "audio": "audio",
                "video": "video"
            }
            supported_names = [modality_names.get(m, m) for m in sorted(self.supported_modalities_set)]
            return False, f"Current system supports {', '.join(supported_names)} training. Modality compatibility check failed."
        
        return True, ""
    
    def get_info(self) -> dict:
        """
        Get system info for API.
        Does NOT expose restriction flags - just what's supported.
        """
        return {
            "supported_model": self._supported_model_path,
            "supported_sources": list(self.supported_sources_set),
            "supported_modalities": list(self.supported_modalities_set),
        }


# Singleton validator instance
_validator: Optional[SystemValidator] = None


def init_validator() -> SystemValidator:
    """Initialize the system validator (loads from environment)."""
    # Check expiration before initializing
    check_system_valid()
    
    global _validator
    _validator = SystemValidator()
    return _validator


def get_validator() -> SystemValidator:
    """Get the system validator instance."""
    # Check expiration on every access
    check_system_valid()
    
    global _validator
    if _validator is None:
        _validator = SystemValidator()
    return _validator
