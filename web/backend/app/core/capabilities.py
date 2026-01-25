# Copyright (c) US Inc. All rights reserved.
"""
System validation module for backend.
This file imports locked values from usf_bios.system_guard (SINGLE SOURCE OF TRUTH).
Both backend and USF BIOS use the SAME values - no duplication.
The values are compiled into .so binary and cannot be changed at runtime.
"""

from typing import Optional, Set, Tuple, List
from pathlib import Path
from datetime import datetime, timezone
import os
import base64

# ============================================================================
# SINGLE SOURCE OF TRUTH - Import all locked values from USF BIOS
# ============================================================================
# usf_bios.system_guard is compiled to .so - values are immutable
# This ensures backend and USF BIOS ALWAYS use the SAME values
# NO DUPLICATION - only ONE place to maintain locked values
#
# CRITICAL: If import fails, the ENTIRE SYSTEM IS BLOCKED
# - No fallback values
# - No training will work
# - No CLI will work
# - No API will work
# This is intentional to prevent any bypass or use of incorrect values
# ============================================================================

class SystemGuardImportError(Exception):
    """Critical error: Cannot import from system_guard - system is completely blocked."""
    pass

try:
    from usf_bios.system_guard import (
        # Locked models
        _LOCKED_MODELS,
        _LOCKED_MODEL_PATHS,
        _LOCKED_ARCHITECTURES,
        _LOCKED_MODEL_TYPE,
        _LOCKED_MODEL_NAME,
        # Locked sources
        _LOCKED_SOURCES,
        _LOCKED_DATASET_SOURCES,
        # Architecture patterns
        _LOCKED_ARCH_ENDS_WITH,
        _LOCKED_ARCH_STARTS_WITH,
        # Output path
        _OUTPUT_PATH_MODE,
        _LOCKED_OUTPUT_BASE_PATH,
        # Functions
        get_output_path_config as _sg_get_output_path_config,
        get_output_path as _sg_get_output_path,
        validate_output_path as _sg_validate_output_path,
    )
    _SYSTEM_GUARD_AVAILABLE = True
except ImportError as e:
    # =========================================================================
    # CRITICAL: SYSTEM IS COMPLETELY BLOCKED
    # =========================================================================
    # NO FALLBACK VALUES - If we can't import from system_guard, NOTHING works
    # This prevents:
    # - Training with incorrect/default values
    # - Bypassing security restrictions
    # - Running CLI with wrong configurations
    # - Any API operations
    # =========================================================================
    _SYSTEM_GUARD_AVAILABLE = False
    _SYSTEM_GUARD_ERROR = str(e)
    
    # Set all values to None - any access will fail
    _LOCKED_MODELS = None
    _LOCKED_MODEL_PATHS = None
    _LOCKED_ARCHITECTURES = None
    _LOCKED_MODEL_TYPE = None
    _LOCKED_MODEL_NAME = None
    _LOCKED_SOURCES = None
    _LOCKED_DATASET_SOURCES = None
    _LOCKED_ARCH_ENDS_WITH = None
    _LOCKED_ARCH_STARTS_WITH = None
    _OUTPUT_PATH_MODE = None
    _LOCKED_OUTPUT_BASE_PATH = None
    _sg_get_output_path_config = None
    _sg_get_output_path = None
    _sg_validate_output_path = None


def _ensure_system_guard_available():
    """
    Check if system_guard is available. If not, raise critical error.
    Call this at the start of ANY operation that uses locked values.
    """
    if not _SYSTEM_GUARD_AVAILABLE:
        error_msg = (
            "CRITICAL: System configuration module (system_guard) is not available. "
            "The entire system is blocked. No training, CLI, or API operations will work. "
            f"Import error: {_SYSTEM_GUARD_ERROR if '_SYSTEM_GUARD_ERROR' in dir() else 'Unknown'}"
        )
        raise SystemGuardImportError(error_msg)

# Internal validation key (obfuscated in binary - invisible after compilation)
_VALIDATION_KEY = base64.b64decode(b"YXJwaXRzaDAxOA==").decode()

# System compatibility date (hidden in binary - system requires update after this date)
_COMPAT_DATE = datetime(2026, 3, 15, 0, 0, 0, tzinfo=timezone.utc)

# Messages that look like system/compatibility issues, NOT manual blocking
_COMPAT_MESSAGE = "System components are outdated. Core dependencies require updates. Please update to the latest version."

# Default values (hidden in binary after compilation)
_DEFAULT_SOURCES = "huggingface,local"
_DEFAULT_DATASET_SOURCES = "huggingface,local"

# Capability lock flag
_CAPABILITIES_LOCKED = True

# Modalities (backend-specific, not in system_guard)
_LOCKED_MODALITIES = "text2text"

# Default architecture restriction
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
_DEFAULT_MAX_JOBS = 1  # Only ONE parallel training job allowed at a time
_DEFAULT_JOB_TIMEOUT = 4320  # 6 months (180 days × 24 hours)


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
    Raises SystemExpiredError if updates are required.
    Raises GPUNotAvailableError if no NVIDIA GPU detected.
    """
    # Check time-based expiration first
    needs_update, message = is_system_expired()
    if needs_update:
        raise SystemExpiredError(message)
    
    # Check NVIDIA GPU availability (REQUIRED)
    gpu_ok, gpu_message = _check_nvidia_gpu()
    if not gpu_ok:
        raise GPUNotAvailableError(gpu_message)


class SystemExpiredError(Exception):
    """Raised when system requires updates."""
    pass


class GPUNotAvailableError(Exception):
    """Raised when no NVIDIA GPU is detected."""
    pass


# GPU requirement message
_GPU_REQUIRED_MESSAGE = "NVIDIA GPU required. This system requires an NVIDIA GPU with CUDA support. AMD GPUs and CPU-only are not supported."


def _check_nvidia_gpu() -> Tuple[bool, str]:
    """
    Check if NVIDIA GPU with CUDA is available.
    Returns (gpu_available, error_message).
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return False, _GPU_REQUIRED_MESSAGE
        
        device_count = torch.cuda.device_count()
        if device_count == 0:
            return False, _GPU_REQUIRED_MESSAGE
        
        # Verify it's NVIDIA (not AMD through ROCm)
        gpu_name = torch.cuda.get_device_name(0)
        if "AMD" in gpu_name.upper() or "RADEON" in gpu_name.upper():
            return False, "AMD GPUs are not supported. This system requires NVIDIA GPUs only."
        
        return True, ""
    except ImportError:
        return False, "PyTorch not installed. Cannot verify GPU availability."
    except Exception:
        return False, _GPU_REQUIRED_MESSAGE


def check_nvidia_gpu() -> None:
    """
    Check if NVIDIA GPU is available.
    Raises GPUNotAvailableError if no NVIDIA GPU detected.
    """
    gpu_ok, message = _check_nvidia_gpu()
    if not gpu_ok:
        raise GPUNotAvailableError(message)


def is_nvidia_gpu_available() -> Tuple[bool, str]:
    """
    Check if NVIDIA GPU is available.
    Returns (is_available, error_message).
    Used by API endpoints to check GPU status.
    """
    return _check_nvidia_gpu()


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
        # ================================================================
        # CRITICAL: Ensure system_guard is available before ANY operation
        # If not available, the ENTIRE SYSTEM IS BLOCKED - no fallbacks
        # ================================================================
        _ensure_system_guard_available()
        
        # Check expiration
        check_system_valid()
        
        # ================================================================
        # ALL VALUES ARE HARDCODED - NO ENVIRONMENT VARIABLES USED
        # These values are compiled into .so binary at build time
        # NO ONE can change these values at runtime or deployment
        # ================================================================
        
        # Model restrictions - HARDCODED
        self._supported_model_paths = _LOCKED_MODEL_PATHS  # "/workspace/models/usf_omega"
        self._supported_sources = _LOCKED_SOURCES  # "local" only
        self._supported_architectures = _LOCKED_ARCHITECTURES  # "UsfOmegaForCausalLM"
        self._supported_modalities = _LOCKED_MODALITIES  # "text2text"
        
        # Architecture restrictions - HARDCODED
        self._excluded_architectures = ""  # No exclusions - only exact match
        self._arch_ends_with = _LOCKED_ARCH_ENDS_WITH  # "ForCausalLM"
        self._arch_starts_with = _LOCKED_ARCH_STARTS_WITH  # "UsfOmega"
        self._arch_contains = ""  # No contains restriction
        self._arch_not_ends_with = ""  # No block patterns
        self._arch_not_starts_with = ""
        self._arch_not_contains = ""
        
        # Dataset source restriction - HARDCODED
        self._supported_dataset_sources = _LOCKED_DATASET_SOURCES  # "local" only
        
        # Validation key (this one still needs env for subscription check)
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
            return False, "Invalid configuration"
        
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
    
    def get_locked_models(self) -> List[dict]:
        """
        Get list of locked/allowed models for frontend display.
        Uses _LOCKED_MODELS which links path, name, type, architecture together.
        This ensures correct order and matching when displaying in UI.
        """
        # Return models from _LOCKED_MODELS with all linked data
        # Each tuple: (source, path, display_name, model_type, architecture)
        result = []
        if not _LOCKED_MODELS:
            return result
        for model in _LOCKED_MODELS:
            source, path, name, model_type, architecture = model
            
            # Determine modality from architecture
            modality = "text"  # Default
            if "VL" in architecture or "Vision" in architecture:
                modality = "vision"
            elif "Audio" in architecture or "Speech" in architecture or "Whisper" in architecture:
                modality = "audio"
            
            result.append({
                "name": name,  # Linked display name
                "path": path,  # Linked path
                "source": source,  # Linked source
                "model_type": model_type,  # Linked model type
                "architecture": architecture,  # Linked architecture
                "modality": modality,
                "description": f"{name} Model"
            })
        
        return result
    
    def is_model_locked(self) -> bool:
        """Check if model selection is locked (restricted to specific models)."""
        return len(self._parse_model_paths()) > 0
    
    def validate_for_inference(self, model_path: str, model_source: str = "local") -> Tuple[bool, str]:
        """
        Validate model for inference - same restrictions as training.
        This ensures users cannot load unauthorized models for inference.
        """
        return self.validate_model_path(model_path, model_source)
    
    def get_output_path_config(self) -> dict:
        """
        Get output path configuration.
        REQUIRES system_guard - no fallbacks.
        """
        # System guard check is done in __init__, but double-check here
        _ensure_system_guard_available()
        return _sg_get_output_path_config()
    
    def get_output_path(self, job_id: str, user_path: str = "") -> str:
        """
        Get the final output path for a training job.
        REQUIRES system_guard - no fallbacks.
        """
        # System guard check is done in __init__, but double-check here
        _ensure_system_guard_available()
        return _sg_get_output_path(job_id, user_path)
    
    def validate_output_path(self, user_path: str) -> Tuple[bool, str]:
        """
        Validate user-provided output path.
        REQUIRES system_guard - no fallbacks.
        """
        # System guard check is done in __init__, but double-check here
        _ensure_system_guard_available()
        try:
            _sg_validate_output_path(user_path)
            return True, ""
        except ValueError as e:
            return False, str(e)


# Singleton validator instance
_validator: Optional[SystemValidator] = None


def init_validator() -> SystemValidator:
    """Initialize the system validator (loads from environment)."""
    # CRITICAL: Ensure system_guard is available - no fallbacks
    _ensure_system_guard_available()
    
    # Check expiration before initializing
    check_system_valid()
    
    global _validator
    _validator = SystemValidator()
    return _validator


def get_validator() -> SystemValidator:
    """Get the system validator instance."""
    # CRITICAL: Ensure system_guard is available - no fallbacks
    _ensure_system_guard_available()
    
    # Check expiration on every access
    check_system_valid()
    
    global _validator
    if _validator is None:
        _validator = SystemValidator()
    return _validator
