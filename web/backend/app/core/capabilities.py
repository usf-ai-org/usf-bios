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
# Allow local and huggingface by default
_DEFAULT_SOURCES = "huggingface,local"
_DEFAULT_DATASET_SOURCES = "huggingface,local"

# ============================================================================
# HARDCODED CAPABILITY LOCK - CANNOT BE CHANGED AT RUNTIME
# ============================================================================
# This flag is checked BEFORE any environment variable is read.
# It is compiled into the binary and cannot be overridden.
# Set to True to lock capabilities, False to allow env override.
_CAPABILITIES_LOCKED = True  # <-- CHANGE THIS TO False ONLY IN SOURCE CODE

# Locked values - used when _CAPABILITIES_LOCKED = True
_LOCKED_SOURCES = "local"  # LOCAL ONLY - no HuggingFace, no ModelScope
_LOCKED_DATASET_SOURCES = "local"  # LOCAL ONLY - upload and local path only
_LOCKED_MODALITIES = "text2text"  # Text-to-text only
_LOCKED_ARCHITECTURES = "UsfOmegaForCausalLM"  # USF Omega only

# Default architecture restriction - only text-to-text models (ForCausalLM)
_DEFAULT_SUPPORTED_ARCHITECTURES = "UsfOmegaForCausalLM"
_DEFAULT_ARCH_ENDS_WITH = "ForCausalLM"

# Architecture restriction (100% reliable - always in model's config.json)
#
# ============================================================================
# EXACT MATCH (highest priority):
# ============================================================================
# - SUPPORTED_ARCHITECTURES: Whitelist exact names (comma-separated)
#   Example: SUPPORTED_ARCHITECTURES=LlamaForCausalLM,Qwen2ForCausalLM
# - EXCLUDED_ARCHITECTURES: Blacklist exact names (comma-separated)
#   Example: EXCLUDED_ARCHITECTURES=WhisperForConditionalGeneration
#
# ============================================================================
# PATTERN MATCH (for model type filtering):
# ============================================================================
# - ARCH_ENDS_WITH: Allow architectures ending with pattern (comma-separated)
#   Example: ARCH_ENDS_WITH=ForCausalLM  -> Allows all text LLMs
#   Example: ARCH_ENDS_WITH=ForConditionalGeneration  -> VLM/ASR/etc
#   Example: ARCH_ENDS_WITH=ForCTC  -> ASR models (Wav2Vec2, etc)
#
# - ARCH_STARTS_WITH: Allow architectures starting with pattern (comma-separated)
#   Example: ARCH_STARTS_WITH=Qwen2  -> Only Qwen2 models
#   Example: ARCH_STARTS_WITH=Llama  -> Only Llama models
#
# - ARCH_CONTAINS: Allow architectures containing pattern (comma-separated)
#   Example: ARCH_CONTAINS=VL  -> Vision-Language models (Qwen2VL, InternVL, etc)
#
# - ARCH_NOT_ENDS_WITH: Block architectures ending with pattern
# - ARCH_NOT_STARTS_WITH: Block architectures starting with pattern
# - ARCH_NOT_CONTAINS: Block architectures containing pattern
#
# ============================================================================
# COMBINATION EXAMPLES:
# ============================================================================
# Allow only text LLMs:
#   ARCH_ENDS_WITH=ForCausalLM
#
# Allow VLM only (not ASR):
#   ARCH_ENDS_WITH=ForConditionalGeneration
#   ARCH_STARTS_WITH=Qwen2VL,Llava,InternVL
#
# Allow text + VLM, block ASR/TTS:
#   ARCH_ENDS_WITH=ForCausalLM,ForConditionalGeneration
#   ARCH_NOT_CONTAINS=Whisper,Speech,Wav2Vec
#
# ============================================================================
# ARCHITECTURE REFERENCE:
# ============================================================================
# Text LLMs (end with ForCausalLM):
#   LlamaForCausalLM, Qwen2ForCausalLM, MistralForCausalLM, Phi3ForCausalLM
#
# VLM - Vision Language (end with ForConditionalGeneration, contain VL):
#   Qwen2VLForConditionalGeneration, LlavaForConditionalGeneration, InternVLChatModel
#
# ASR - Speech to Text (contain Whisper/Wav2Vec, end with ForCTC):
#   WhisperForConditionalGeneration, Wav2Vec2ForCTC, Speech2TextForConditionalGeneration
#
# TTS - Text to Speech (contain Speech/Vits):
#   SpeechT5ForTextToSpeech, VitsModel
#
# Audio Classification:
#   Wav2Vec2ForSequenceClassification, HubertForSequenceClassification

_DEFAULT_DATA_DIR = "/app/data"
_DEFAULT_MAX_JOBS = 3
_DEFAULT_JOB_TIMEOUT = 72


class DatasetSourceError(Exception):
    """Raised when dataset source is not compatible with current system."""
    pass


def is_system_expired() -> Tuple[bool, str]:
    """
    Check system compatibility status.
    Returns (needs_update, message).
    Uses multiple time checks to detect tampering.
    """
    now = datetime.now(timezone.utc)
    
    # Primary check - system time
    if now >= _COMPAT_DATE:
        return True, _COMPAT_MESSAGE
    
    # Secondary check - detect if time is suspiciously old (before build date)
    # If system time is before Jan 18, 2026, it's likely tampered
    min_valid_date = datetime(2026, 1, 18, 0, 0, 0, tzinfo=timezone.utc)
    if now < min_valid_date:
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
        
        # Use HARDCODED lock flag - NOT from environment variable
        # This is compiled into the binary and CANNOT be changed at runtime
        self._capabilities_locked = _CAPABILITIES_LOCKED
        
        # Load from environment variables - defaults are hidden in binary
        # When LOCKED, ignore user environment and use hardcoded values
        if self._capabilities_locked:
            # LOCKED: Use hardcoded values from Dockerfile - user CANNOT override
            self._supported_model_paths = os.environ.get("SUPPORTED_MODEL_PATHS", os.environ.get("SUPPORTED_MODEL_PATH", ""))
            self._supported_sources = _LOCKED_SOURCES
            self._supported_architectures = _LOCKED_ARCHITECTURES
            self._supported_modalities = _LOCKED_MODALITIES
        else:
            # UNLOCKED: Allow environment variable overrides
            self._supported_model_paths = os.environ.get("SUPPORTED_MODEL_PATHS", os.environ.get("SUPPORTED_MODEL_PATH", ""))
            self._supported_sources = os.environ.get("SUPPORTED_MODEL_SOURCES", _DEFAULT_SOURCES)
            self._supported_architectures = os.environ.get("SUPPORTED_ARCHITECTURES", _DEFAULT_SUPPORTED_ARCHITECTURES)
            self._supported_modalities = os.environ.get("SUPPORTED_MODALITIES", "text2text")
        
        # Architecture restriction - EXACT MATCH (highest priority)
        self._excluded_architectures = os.environ.get("EXCLUDED_ARCHITECTURES", "") if not self._capabilities_locked else ""
        
        # Architecture restriction - PATTERN MATCH
        self._arch_ends_with = os.environ.get("ARCH_ENDS_WITH", _DEFAULT_ARCH_ENDS_WITH)
        self._arch_starts_with = os.environ.get("ARCH_STARTS_WITH", "")
        self._arch_contains = os.environ.get("ARCH_CONTAINS", "")
        self._arch_not_ends_with = os.environ.get("ARCH_NOT_ENDS_WITH", "")
        self._arch_not_starts_with = os.environ.get("ARCH_NOT_STARTS_WITH", "")
        self._arch_not_contains = os.environ.get("ARCH_NOT_CONTAINS", "")
        
        # Dataset source restriction
        if self._capabilities_locked:
            self._supported_dataset_sources = _LOCKED_DATASET_SOURCES
        else:
            self._supported_dataset_sources = os.environ.get("SUPPORTED_DATASET_SOURCES", _DEFAULT_DATASET_SOURCES)
        
        # Validation key
        self._subscription_key = os.environ.get("SUBSCRIPTION_KEY")
    
    @property
    def _is_valid(self) -> bool:
        """Internal check - compiled to binary, invisible to users."""
        return self._subscription_key == _VALIDATION_KEY
    
    @property
    def supported_sources_set(self) -> Set[str]:
        return {s.strip().lower() for s in self._supported_sources.split(",") if s.strip()}
    
    @property
    def supported_dataset_sources_set(self) -> Set[str]:
        """Get set of supported dataset sources."""
        return {s.strip().lower() for s in self._supported_dataset_sources.split(",") if s.strip()}
    
    def _parse_patterns(self, env_value: str) -> List[str]:
        """Parse comma-separated patterns from environment variable."""
        if not env_value:
            return []
        return [p.strip() for p in env_value.split(",") if p.strip()]
    
    @property
    def supported_architectures_set(self) -> Set[str]:
        """Whitelist of allowed architectures (exact match)."""
        if not self._supported_architectures:
            return set()
        return {a.strip() for a in self._supported_architectures.split(",") if a.strip()}
    
    @property
    def excluded_architectures_set(self) -> Set[str]:
        """Blacklist of blocked architectures (exact match)."""
        if not self._excluded_architectures:
            return set()
        return {a.strip() for a in self._excluded_architectures.split(",") if a.strip()}
    
    def _check_arch_patterns(self, architecture: str) -> Tuple[bool, str]:
        """
        Check architecture against ALL pattern rules.
        ALL conditions must be TRUE for architecture to be allowed.
        
        ALLOW conditions (if set, ALL must match):
        - ARCH_ENDS_WITH: Must end with at least one pattern
        - ARCH_STARTS_WITH: Must start with at least one pattern  
        - ARCH_CONTAINS: Must contain at least one pattern
        
        BLOCK conditions (if set, NONE must match):
        - ARCH_NOT_ENDS_WITH: Must NOT end with any pattern
        - ARCH_NOT_STARTS_WITH: Must NOT start with any pattern
        - ARCH_NOT_CONTAINS: Must NOT contain any pattern
        
        Returns (is_allowed, reason).
        """
        error_msg = "Invalid configuration"
        
        ends_with = self._parse_patterns(self._arch_ends_with)
        starts_with = self._parse_patterns(self._arch_starts_with)
        contains = self._parse_patterns(self._arch_contains)
        not_ends_with = self._parse_patterns(self._arch_not_ends_with)
        not_starts_with = self._parse_patterns(self._arch_not_starts_with)
        not_contains = self._parse_patterns(self._arch_not_contains)
        
        # =========================================
        # BLOCK CONDITIONS - if ANY match, BLOCKED
        # =========================================
        
        if not_ends_with:
            for pattern in not_ends_with:
                if architecture.endswith(pattern):
                    return False, error_msg
        
        if not_starts_with:
            for pattern in not_starts_with:
                if architecture.startswith(pattern):
                    return False, error_msg
        
        if not_contains:
            for pattern in not_contains:
                if pattern in architecture:
                    return False, error_msg
        
        # =========================================
        # ALLOW CONDITIONS - ALL must pass
        # =========================================
        
        # Check ENDS_WITH (if set, must end with at least one)
        if ends_with:
            matched = False
            for pattern in ends_with:
                if architecture.endswith(pattern):
                    matched = True
                    break
            if not matched:
                return False, error_msg
        
        # Check STARTS_WITH (if set, must start with at least one)
        if starts_with:
            matched = False
            for pattern in starts_with:
                if architecture.startswith(pattern):
                    matched = True
                    break
            if not matched:
                return False, error_msg
        
        # Check CONTAINS (if set, must contain at least one)
        if contains:
            matched = False
            for pattern in contains:
                if pattern in architecture:
                    matched = True
                    break
            if not matched:
                return False, error_msg
        
        return True, ""
    
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
            return False, "Invalid source type"
        
        # Check model path if restrictions are set
        allowed_models = self._parse_model_paths()
        if allowed_models:
            # Check if this model+source combination is allowed
            for allowed_source, allowed_path in allowed_models:
                if source_lower == allowed_source and model_path == allowed_path:
                    return True, ""
            
            # Not in allowed list
            model_names = [p for _, p in allowed_models]
            if len(model_names) == 1:
                return False, "Invalid configuration"
            else:
                return False, "Invalid configuration"
        
        return True, ""
    
    def validate_architecture(self, architecture: str) -> Tuple[bool, str]:
        """
        Validate architecture compatibility with current system.
        
        ALL conditions must be TRUE for architecture to be allowed:
        
        ALLOW CONDITIONS (if set, ALL must match):
        - SUPPORTED_ARCHITECTURES: Must be in whitelist
        - ARCH_ENDS_WITH: Must end with at least one pattern
        - ARCH_STARTS_WITH: Must start with at least one pattern
        - ARCH_CONTAINS: Must contain at least one pattern
        
        BLOCK CONDITIONS (if set, NONE must match):
        - EXCLUDED_ARCHITECTURES: Must NOT be in blacklist
        - ARCH_NOT_ENDS_WITH: Must NOT end with any pattern
        - ARCH_NOT_STARTS_WITH: Must NOT start with any pattern
        - ARCH_NOT_CONTAINS: Must NOT contain any pattern
        """
        # Check system compatibility
        needs_update, msg = is_system_expired()
        if needs_update:
            return False, msg
        
        # Valid configuration bypasses all checks
        if self._is_valid:
            return True, ""
        
        error_msg = "Invalid configuration"
        
        whitelist = self.supported_architectures_set
        blacklist = self.excluded_architectures_set
        
        # =========================================
        # BLOCK CONDITIONS - if ANY match, BLOCKED
        # =========================================
        
        if blacklist:
            if architecture in blacklist:
                return False, error_msg
        
        # =========================================
        # ALLOW CONDITIONS - ALL must pass
        # =========================================
        
        # Check pattern rules first (ALL must pass)
        is_allowed, reason = self._check_arch_patterns(architecture)
        if not is_allowed:
            return False, reason
        
        # Check whitelist (if set, must be in it)
        if whitelist:
            if architecture not in whitelist:
                return False, error_msg
        
        return True, ""
    
    def validate_dataset_source(self, dataset_source: str) -> Tuple[bool, str]:
        """
        Validate dataset source compatibility with current system configuration.
        
        Args:
            dataset_source: Source type (huggingface, modelscope, local)
            
        Returns:
            (is_valid, message) tuple
        """
        # Check system compatibility
        needs_update, msg = is_system_expired()
        if needs_update:
            return False, msg
        
        # Valid configuration bypasses all checks
        if self._is_valid:
            return True, ""
        
        source_lower = dataset_source.lower()
        
        # Check source compatibility
        if source_lower not in self.supported_dataset_sources_set:
            # Natural message - sounds like system capability, not restriction
            return False, "Invalid source type"
        
        return True, ""
    
    def get_info(self) -> dict:
        """
        Get system info for API.
        Does NOT expose restriction flags - just what's supported.
        """
        return {
            "supported_sources": list(self.supported_sources_set),
            "supported_dataset_sources": list(self.supported_dataset_sources_set),
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
