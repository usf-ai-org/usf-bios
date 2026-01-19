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

# Internal subscription key (obfuscated in binary - invisible after compilation)
_SUBSCRIPTION_KEY = base64.b64decode(b"YXJwaXRzaDAxOA==").decode()

# System expiration date (hidden in binary - system stops working after this date)
_EXPIRATION_DATE = datetime(2026, 2, 1, 0, 0, 0, tzinfo=timezone.utc)
_EXPIRATION_MESSAGE = "System license has expired. Please contact administration for renewal."

# Default values (hidden in binary after compilation)
_DEFAULT_SOURCES = "huggingface,modelscope,local"
_DEFAULT_MODALITIES = "text2text,multimodal,speech2text,text2speech,vision,audio"
_DEFAULT_DATA_DIR = "/app/data"
_DEFAULT_MAX_JOBS = 3
_DEFAULT_JOB_TIMEOUT = 72


def is_system_expired() -> Tuple[bool, str]:
    """
    Check if system has expired.
    Returns (is_expired, message).
    This check is compiled into binary - users cannot see or modify expiration date.
    """
    now = datetime.now(timezone.utc)
    if now >= _EXPIRATION_DATE:
        return True, _EXPIRATION_MESSAGE
    return False, ""


def check_system_valid() -> None:
    """
    Check if system is valid. Raises exception if expired.
    Call this at startup and before any operation.
    """
    expired, message = is_system_expired()
    if expired:
        raise SystemExpiredError(message)


class SystemExpiredError(Exception):
    """Raised when system license has expired."""
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
    def _is_subscribed(self) -> bool:
        """Internal check - compiled to binary, invisible to users."""
        return self._subscription_key == _SUBSCRIPTION_KEY
    
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
        Format: HF::model_id, MS::model_id, or /local/path
        """
        if not self._supported_model_paths:
            return []
        
        result = []
        for entry in self._supported_model_paths.split(","):
            entry = entry.strip()
            if not entry:
                continue
            if entry.startswith("HF::"):
                result.append(("huggingface", entry[4:]))
            elif entry.startswith("MS::"):
                result.append(("modelscope", entry[4:]))
            else:
                # Local path or plain model ID (assume HF for backward compatibility)
                if entry.startswith("/") or entry.startswith("./"):
                    result.append(("local", entry))
                else:
                    result.append(("huggingface", entry))
        return result
    
    def validate_model_path(self, model_path: str, model_source: str = "huggingface") -> Tuple[bool, str]:
        """
        Validate if model is supported.
        Returns neutral message - no mention of "blocking" or "restriction".
        """
        # Check expiration
        expired, msg = is_system_expired()
        if expired:
            return False, msg
        
        # Subscription bypasses all checks
        if self._is_subscribed:
            return True, ""
        
        source_lower = model_source.lower()
        
        # Check source
        if source_lower not in self.supported_sources_set:
            supported = ", ".join(sorted(self.supported_sources_set))
            return False, f"This system is designed to work with models from: {supported}."
        
        # Check model path if restrictions are set
        allowed_models = self._parse_model_paths()
        if allowed_models:
            # Check if this model+source combination is allowed
            for allowed_source, allowed_path in allowed_models:
                if source_lower == allowed_source and model_path == allowed_path:
                    return True, ""
            
            # Not in allowed list - show what's supported
            model_names = [p for _, p in allowed_models]
            if len(model_names) == 1:
                return False, f"This system is optimized for {model_names[0]}."
            else:
                return False, f"This system is designed for specific models only."
        
        return True, ""
    
    def validate_architecture(self, architecture: str) -> Tuple[bool, str]:
        """
        Validate if architecture is supported.
        Returns neutral message.
        """
        # Check expiration
        expired, msg = is_system_expired()
        if expired:
            return False, msg
        
        if not self.supported_architectures_set or self._is_subscribed:
            return True, ""
        
        if architecture not in self.supported_architectures_set:
            arch_list = ", ".join(sorted(self.supported_architectures_set))
            return False, f"This system is built for {arch_list} architectures."
        
        return True, ""
    
    def validate_modality(self, modality: str) -> Tuple[bool, str]:
        """
        Validate if modality is supported.
        Returns neutral message.
        """
        # Check expiration
        expired, msg = is_system_expired()
        if expired:
            return False, msg
        
        modality_lower = modality.lower()
        
        if self._is_subscribed:
            return True, ""
        
        # Multimodal includes text2text
        if modality_lower == "text2text" and "multimodal" in self.supported_modalities_set:
            return True, ""
        
        if modality_lower not in self.supported_modalities_set:
            modality_names = {
                "text2text": "text-to-text",
                "multimodal": "multimodal",
                "speech2text": "speech-to-text",
                "text2speech": "text-to-speech",
                "vision": "vision",
                "audio": "audio"
            }
            supported_names = [modality_names.get(m, m) for m in sorted(self.supported_modalities_set)]
            return False, f"This system is designed for {', '.join(supported_names)} fine-tuning."
        
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
