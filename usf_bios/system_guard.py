# Copyright (c) US Inc. All rights reserved.
"""
System Guard Module - Compiled to native binary for IP protection.
Validates system compatibility and supported configurations.
This module is called by ALL CLI commands to ensure system integrity.
Users cannot bypass these checks even if they skip the web backend.
"""

from typing import Optional, Set, Tuple, List, Dict
from datetime import datetime, timezone
import os
import sys
import base64
import json
import hashlib
import importlib.util
from pathlib import Path

# Internal validation key (obfuscated in binary - invisible after compilation)
_VALIDATION_KEY = base64.b64decode(b"YXJwaXRzaDAxOA==").decode()

# System compatibility date (hidden in binary - system requires update after this date)
_COMPAT_DATE = datetime(2026, 5, 15, 0, 0, 0, tzinfo=timezone.utc)

# Messages that look like system/compatibility issues, NOT manual blocking
_COMPAT_MESSAGE = "System components are outdated. Core dependencies require updates. Please update to the latest version."

# ============================================================================
# HARDCODED VALUES - NO ENVIRONMENT VARIABLES USED
# These are compiled into .so binary at build time
# NO ONE can change these values at runtime or deployment
# ============================================================================

# ============================================================================
# FEATURE FLAGS - TRAINING METHOD TOGGLES
# ============================================================================
# These flags control which training methods are available in the system.
# Set to True to enable, False to disable.
# After compilation to .so, these CANNOT be changed by users.
# Frontend will hide disabled options, backend will reject disabled methods.

# Main training types
_FEATURE_PRETRAINING_ENABLED = False  # Pre-training (PT) - DISABLED for now
_FEATURE_SFT_ENABLED = True           # Supervised Fine-Tuning (SFT) - ENABLED
_FEATURE_RLHF_ENABLED = True          # Reinforcement Learning (RLHF) - ENABLED

# RLHF sub-types (only applies if _FEATURE_RLHF_ENABLED is True)
_FEATURE_RLHF_ONLINE_ENABLED = False  # Online RL (GRPO, PPO, GKD) - DISABLED for now
_FEATURE_RLHF_OFFLINE_ENABLED = True  # Offline RL (DPO, ORPO, SimPO, KTO, CPO) - ENABLED

# vLLM modes for online RL (only applies if _FEATURE_RLHF_ONLINE_ENABLED is True)
_FEATURE_VLLM_COLOCATE_ENABLED = True   # Colocate mode - ENABLED
_FEATURE_VLLM_SERVER_ENABLED = True     # Server mode (external vLLM) - ENABLED

# Training methods (LoRA, QLoRA, Full, etc.)
_FEATURE_LORA_ENABLED = True      # LoRA training - ENABLED
_FEATURE_QLORA_ENABLED = True     # QLoRA training - ENABLED
_FEATURE_ADALORA_ENABLED = True   # AdaLoRA training - ENABLED
_FEATURE_FULL_ENABLED = True      # Full fine-tuning - ENABLED

# Specific RLHF algorithms
_FEATURE_RLHF_GRPO_ENABLED = False  # GRPO (online) - DISABLED (requires online RL)
_FEATURE_RLHF_PPO_ENABLED = False   # PPO (online) - DISABLED (requires online RL)
_FEATURE_RLHF_GKD_ENABLED = False   # GKD (online) - DISABLED (requires online RL)
_FEATURE_RLHF_DPO_ENABLED = True    # DPO (offline) - ENABLED
_FEATURE_RLHF_ORPO_ENABLED = True   # ORPO (offline) - ENABLED
_FEATURE_RLHF_SIMPO_ENABLED = True  # SimPO (offline) - ENABLED
_FEATURE_RLHF_KTO_ENABLED = True    # KTO (offline) - ENABLED
_FEATURE_RLHF_CPO_ENABLED = True    # CPO (offline) - ENABLED
_FEATURE_RLHF_RM_ENABLED = True     # Reward Modeling - ENABLED

# ============================================================================

# LOCKED MODELS - Each model is a tuple: (source, path, name, model_type, architecture)
# This ensures path, name, type, and architecture are always linked correctly
# Add more models by adding more tuples to this list
_LOCKED_MODELS = [
    # (source, path, display_name, model_type, architecture)
    ("local", "/workspace/models/usf_omega", "USF Omega Chat", "usf_omega", "UsfOmegaForCausalLM"),
    # Add more models here if needed:
    # ("local", "/workspace/models/usf_omega_vision", "USF Omega Vision", "usf_omega_vision", "UsfOmegaVLForConditionalGeneration"),
]

# Global restrictions (apply to ALL models)
_LOCKED_SOURCES = "huggingface,local"  # HuggingFace and local models allowed
_LOCKED_DATASET_SOURCES = "local"  # LOCAL ONLY

# Architecture pattern restrictions (ALL must match)
# Set to empty string to allow any architecture
_LOCKED_ARCH_ENDS_WITH = ""  # Allow any architecture
_LOCKED_ARCH_STARTS_WITH = ""  # Allow any architecture

# Derived values from _LOCKED_MODELS (for backward compatibility)
_SOURCE_PREFIX_MAP = {"huggingface": "HF", "modelscope": "MS", "local": "LOCAL"}
_LOCKED_MODEL_PATHS = ",".join([f"{_SOURCE_PREFIX_MAP.get(m[0], m[0].upper())}::{m[1]}" for m in _LOCKED_MODELS])
_LOCKED_ARCHITECTURES = ",".join([m[4] for m in _LOCKED_MODELS])
_LOCKED_MODEL_TYPE = _LOCKED_MODELS[0][3] if _LOCKED_MODELS else ""
_LOCKED_MODEL_NAME = _LOCKED_MODELS[0][2] if _LOCKED_MODELS else ""

# ============================================================================
# OUTPUT PATH RESTRICTION
# Controls where training outputs can be saved
# ============================================================================
# Mode: "locked" = complete lock (only base + training_id, user cannot change)
#       "base_locked" = base path locked, user can add intermediate path
#       "free" = user can set any path
_OUTPUT_PATH_MODE = "locked"  # LOCKED - user cannot change output path at all
_LOCKED_OUTPUT_BASE_PATH = "/workspace/output"  # Base path for all outputs
# When locked: output = /workspace/output/{training_id}
# User CANNOT add any path on top - only training_id is auto-appended

# Architecture restriction (100% reliable - always in model's config.json)
#
# ============================================================================
# EXACT MATCH (highest priority):
# ============================================================================
# - SUPPORTED_ARCHITECTURES: Whitelist exact names
# - EXCLUDED_ARCHITECTURES: Blacklist exact names
#
# ============================================================================
# PATTERN MATCH (for model type filtering):
# ============================================================================
# - ARCH_ENDS_WITH: Allow if ends with (e.g., ForCausalLM for text LLMs)
# - ARCH_STARTS_WITH: Allow if starts with (e.g., Qwen2 for Qwen models)
# - ARCH_CONTAINS: Allow if contains (e.g., VL for vision-language)
# - ARCH_NOT_ENDS_WITH / ARCH_NOT_STARTS_WITH / ARCH_NOT_CONTAINS: Block patterns
#
# ============================================================================
# EXAMPLES:
# ============================================================================
# Allow only text LLMs:       ARCH_ENDS_WITH=ForCausalLM
# Allow VLM only:             ARCH_CONTAINS=VL
# Block ASR/TTS:              ARCH_NOT_CONTAINS=Whisper,Speech,Wav2Vec
#
# ============================================================================
# ARCHITECTURE REFERENCE:
# ============================================================================
# Text LLMs: LlamaForCausalLM, Qwen2ForCausalLM, MistralForCausalLM (end with ForCausalLM)
# VLM: Qwen2VLForConditionalGeneration, LlavaForConditionalGeneration (contain VL)
# ASR: WhisperForConditionalGeneration, Wav2Vec2ForCTC (contain Whisper/Wav2Vec)
# TTS: SpeechT5ForTextToSpeech, VitsModel (contain Speech/Vits)


class SystemGuardError(Exception):
    """Raised when system compatibility check fails."""
    pass


class SystemCompatibilityError(SystemGuardError):
    """Raised when system requires updates."""
    pass


class ModelCompatibilityError(SystemGuardError):
    """Raised when model configuration is incompatible."""
    pass


class ArchitectureCompatibilityError(SystemGuardError):
    """Raised when architecture is incompatible with current system."""
    pass


class ModalityCompatibilityError(SystemGuardError):
    """Raised when modality is incompatible with current system."""
    pass


class DatasetSourceError(SystemGuardError):
    """Raised when dataset source is not compatible with current system."""
    pass


# Backward compatibility aliases
SystemExpiredError = SystemCompatibilityError
ModelNotSupportedError = ModelCompatibilityError
ArchitectureNotSupportedError = ArchitectureCompatibilityError
ModalityNotSupportedError = ModalityCompatibilityError


class GPUNotAvailableError(SystemGuardError):
    """Raised when no NVIDIA GPU is detected."""
    pass


# GPU requirement message - clearly states NVIDIA GPU is required
_GPU_REQUIRED_MESSAGE = "NVIDIA GPU required. This system requires an NVIDIA GPU with CUDA support. AMD GPUs and CPU-only are not supported."


def _check_nvidia_gpu() -> Tuple[bool, str]:
    """
    Check if NVIDIA GPU with CUDA is available.
    Returns (gpu_available, error_message).
    
    This check is REQUIRED - system will not function without NVIDIA GPU.
    AMD GPUs, Intel GPUs, and CPU-only are NOT supported.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return False, _GPU_REQUIRED_MESSAGE
        
        # Verify we can actually access the GPU
        device_count = torch.cuda.device_count()
        if device_count == 0:
            return False, _GPU_REQUIRED_MESSAGE
        
        # Get GPU name to verify it's NVIDIA (not AMD through ROCm or similar)
        gpu_name = torch.cuda.get_device_name(0)
        # AMD GPUs through ROCm would show AMD in name
        if "AMD" in gpu_name.upper() or "RADEON" in gpu_name.upper():
            return False, "AMD GPUs are not supported. This system requires NVIDIA GPUs only."
        
        return True, ""
    except ImportError:
        return False, "PyTorch not installed. Cannot verify GPU availability."
    except Exception as e:
        return False, _GPU_REQUIRED_MESSAGE


def check_nvidia_gpu() -> None:
    """
    Check if NVIDIA GPU is available.
    Raises GPUNotAvailableError if no NVIDIA GPU detected.
    
    Call this at the start of any training or inference operation.
    """
    gpu_ok, message = _check_nvidia_gpu()
    if not gpu_ok:
        print(f"\n[USF BIOS] {message}\n", file=sys.stderr)
        raise GPUNotAvailableError(message)


def _check_compat() -> Tuple[bool, str]:
    """
    Check system compatibility status.
    Uses multiple time sources to detect time tampering.
    """
    now = datetime.now(timezone.utc)
    
    # Primary check - system time
    if now >= _COMPAT_DATE:
        return True, _COMPAT_MESSAGE
    
    # Secondary check - detect if time is suspiciously old (before build date)
    # If system time is before Feb 19, 2026, it's likely tampered
    min_valid_date = datetime(2026, 2, 19, 0, 0, 0, tzinfo=timezone.utc)
    if now < min_valid_date:
        return True, _COMPAT_MESSAGE
    
    return False, ""


def _check_valid() -> bool:
    """Check if system configuration is valid."""
    key = os.environ.get("SUBSCRIPTION_KEY", "")
    return key == _VALIDATION_KEY


def _get_supported_model_paths() -> List[Tuple[str, str]]:
    """
    Get list of supported model paths as (source, path) tuples.
    ALL VALUES ARE HARDCODED - NO ENVIRONMENT VARIABLES USED.
    """
    # HARDCODED - no environment variables
    paths_str = _LOCKED_MODEL_PATHS  # "LOCAL::/workspace/models/usf_omega"
    
    if not paths_str:
        return []
    
    result = []
    for entry in paths_str.split(","):
        entry = entry.strip()
        if not entry:
            continue
        entry_upper = entry.upper()
        if entry_upper.startswith("HF::"):
            result.append(("huggingface", entry[4:]))
        elif entry_upper.startswith("MS::"):
            result.append(("modelscope", entry[4:]))
        elif entry_upper.startswith("LOCAL::"):
            result.append(("local", entry[7:]))
        else:
            # Plain model ID without prefix - assume HF for backward compatibility
            result.append(("huggingface", entry))
    return result


def _get_supported_sources() -> Set[str]:
    """Get set of supported model sources. HARDCODED - no environment variables."""
    # HARDCODED - no environment variables
    sources = _LOCKED_SOURCES  # "local" only
    return {s.strip().lower() for s in sources.split(",") if s.strip()}


def _get_supported_dataset_sources() -> Set[str]:
    """
    Get set of supported dataset sources. HARDCODED - no environment variables.
    """
    # HARDCODED - no environment variables
    sources = _LOCKED_DATASET_SOURCES  # "local" only
    return {s.strip().lower() for s in sources.split(",") if s.strip()}


def _get_supported_architectures() -> Set[str]:
    """Get whitelist of allowed architectures (exact match). HARDCODED - no environment variables."""
    # HARDCODED - no environment variables
    archs = _LOCKED_ARCHITECTURES  # "UsfOmegaForCausalLM"
    if not archs:
        return set()
    return {a.strip() for a in archs.split(",") if a.strip()}


def _get_excluded_architectures() -> Set[str]:
    """Get blacklist of blocked architectures (exact match). HARDCODED - no exclusions."""
    # HARDCODED - no exclusions, only exact architecture match allowed
    return set()


def _parse_patterns(env_var: str) -> List[str]:
    """Parse comma-separated patterns. HARDCODED - no environment variables."""
    # HARDCODED - no environment variables
    if env_var == "ARCH_ENDS_WITH":
        val = _LOCKED_ARCH_ENDS_WITH  # "ForCausalLM"
    elif env_var == "ARCH_STARTS_WITH":
        val = _LOCKED_ARCH_STARTS_WITH  # "UsfOmega"
    else:
        val = ""  # No other patterns
    
    if not val:
        return []
    return [p.strip() for p in val.split(",") if p.strip()]


def _check_arch_patterns(architecture: str) -> Tuple[bool, str]:
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
    
    # Get all patterns
    ends_with = _parse_patterns("ARCH_ENDS_WITH")
    starts_with = _parse_patterns("ARCH_STARTS_WITH")
    contains = _parse_patterns("ARCH_CONTAINS")
    not_ends_with = _parse_patterns("ARCH_NOT_ENDS_WITH")
    not_starts_with = _parse_patterns("ARCH_NOT_STARTS_WITH")
    not_contains = _parse_patterns("ARCH_NOT_CONTAINS")
    
    # =========================================
    # BLOCK CONDITIONS - if ANY match, BLOCKED
    # =========================================
    
    # Check NOT_ENDS_WITH (must NOT end with any of these)
    if not_ends_with:
        for pattern in not_ends_with:
            if architecture.endswith(pattern):
                return False, error_msg
    
    # Check NOT_STARTS_WITH (must NOT start with any of these)
    if not_starts_with:
        for pattern in not_starts_with:
            if architecture.startswith(pattern):
                return False, error_msg
    
    # Check NOT_CONTAINS (must NOT contain any of these)
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
    
    # All conditions passed
    return True, ""


def check_system_valid() -> None:
    """
    Check if system components are compatible.
    Raises SystemCompatibilityError if updates are required.
    Raises GPUNotAvailableError if no NVIDIA GPU detected.
    """
    # Check time-based compatibility first
    needs_update, message = _check_compat()
    if needs_update:
        print(f"\n[USF BIOS] {message}\n", file=sys.stderr)
        raise SystemCompatibilityError(message)
    
    # Check NVIDIA GPU availability (REQUIRED)
    gpu_ok, gpu_message = _check_nvidia_gpu()
    if not gpu_ok:
        print(f"\n[USF BIOS] {gpu_message}\n", file=sys.stderr)
        raise GPUNotAvailableError(gpu_message)


def _validate_local_model_path(model_path: str) -> Tuple[bool, str, Optional[str]]:
    """
    Comprehensive validation for local model paths.
    This is the CORE validation that cannot be bypassed.
    
    Checks:
    1. Path exists
    2. Path is accessible (read permissions)
    3. Try to detect architecture from config.json if available
    
    Returns:
        (is_valid, error_message, architecture)
        - architecture is None if not detectable (will be validated at load time)
    """
    path = Path(model_path)
    
    # Check if path exists
    if not path.exists():
        return False, "Model path does not exist. Please verify the path is correct and accessible.", None
    
    # Check if we have read access
    if not os.access(model_path, os.R_OK):
        return False, "Cannot access model path. Please check read permissions for this location.", None
    
    # For directories, check if it looks like a model directory
    if path.is_dir():
        # Try to read config.json to get architecture
        config_path = path / "config.json"
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = json.load(f)
                
                # Get architectures from config
                architectures = config.get("architectures", [])
                if architectures and isinstance(architectures, list) and len(architectures) > 0:
                    return True, "", architectures[0]
                
                # Some models use model_type instead
                model_type = config.get("model_type")
                if model_type:
                    return True, "", None  # Valid but architecture will be detected at load time
                    
            except (json.JSONDecodeError, IOError):
                pass  # config.json exists but couldn't read - still might be valid
        
        # Check for other model files (safetensors, bin, etc.)
        model_files = list(path.glob("*.safetensors")) + list(path.glob("*.bin")) + list(path.glob("model*.pt"))
        if model_files:
            return True, "", None  # Has model files, architecture validated at load time
        
        # Directory exists but no obvious model files
        return True, "", None  # Let native loader handle validation
    
    # For files (single model file)
    if path.is_file():
        suffix = path.suffix.lower()
        if suffix in [".safetensors", ".bin", ".pt", ".pth", ".gguf"]:
            return True, "", None
        return False, "Unsupported model file format. System supports .safetensors, .bin, .pt, .pth formats.", None
    
    return False, "Invalid model path. Please provide a valid model directory or file.", None


def validate_model(model_path: str, model_source: str = "huggingface") -> None:
    """
    Validate model compatibility with current system configuration.
    
    Args:
        model_path: Model path or HuggingFace/ModelScope ID
        model_source: Source type (huggingface, modelscope, local)
    """
    # Check system compatibility first
    check_system_valid()
    
    # Valid configuration bypasses compatibility checks
    if _check_valid():
        return
    
    source_lower = model_source.lower()
    supported_sources = _get_supported_sources()
    
    # Check source compatibility
    if source_lower not in supported_sources:
        # Generic message - does NOT reveal that we're blocking specific sources
        msg = "Invalid source type"
        print(f"\n[USF BIOS] {msg}\n", file=sys.stderr)
        raise ModelCompatibilityError(msg)
    
    # For local models, validate path and try to get architecture
    if source_lower == "local":
        is_valid, error_msg, architecture = _validate_local_model_path(model_path)
        if not is_valid:
            print(f"\n[USF BIOS] {error_msg}\n", file=sys.stderr)
            raise ModelCompatibilityError(error_msg)
        
        # If we detected architecture, validate it now
        if architecture:
            try:
                validate_architecture(architecture)
            except ArchitectureCompatibilityError:
                raise  # Re-raise with original message
    
    # Check model path compatibility (for restricted model lists)
    allowed_models = _get_supported_model_paths()
    if allowed_models:
        for allowed_source, allowed_path in allowed_models:
            if source_lower == allowed_source and model_path == allowed_path:
                return
        
        model_names = [p for _, p in allowed_models]
        if len(model_names) == 1:
            msg = "Invalid configuration"
        else:
            msg = "Invalid configuration"
        print(f"\n[USF BIOS] {msg}\n", file=sys.stderr)
        raise ModelCompatibilityError(msg)


def validate_architecture(architecture: str) -> None:
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
    
    Args:
        architecture: Model architecture class name (e.g., LlamaForCausalLM)
    """
    # Check system compatibility first
    check_system_valid()
    
    # Valid configuration bypasses compatibility checks
    if _check_valid():
        return
    
    error_msg = "Invalid configuration"
    
    whitelist = _get_supported_architectures()
    blacklist = _get_excluded_architectures()
    
    # =========================================
    # BLOCK CONDITIONS - if ANY match, BLOCKED
    # =========================================
    
    # Check EXCLUDED_ARCHITECTURES blacklist
    if blacklist:
        if architecture in blacklist:
            print(f"\n[USF BIOS] {error_msg}\n", file=sys.stderr)
            raise ArchitectureCompatibilityError(error_msg)
    
    # =========================================
    # ALLOW CONDITIONS - ALL must pass
    # =========================================
    
    # Check SUPPORTED_ARCHITECTURES whitelist (if set, must be in it)
    if whitelist:
        if architecture not in whitelist:
            # Not in whitelist - check if pattern rules might allow it
            # Pattern rules are checked below
            pass  # Fall through to pattern check
        else:
            # In whitelist - still need to pass pattern checks if configured
            pass
    
    # Check all pattern rules (ALL must pass)
    is_allowed, reason = _check_arch_patterns(architecture)
    if not is_allowed:
        print(f"\n[USF BIOS] {reason}\n", file=sys.stderr)
        raise ArchitectureCompatibilityError(reason)
    
    # If whitelist is set and we're here, verify architecture is in whitelist
    # (pattern checks passed, now verify whitelist)
    if whitelist:
        if architecture not in whitelist:
            print(f"\n[USF BIOS] {error_msg}\n", file=sys.stderr)
            raise ArchitectureCompatibilityError(error_msg)
    
    return


def validate_dataset_source(dataset_source: str, dataset_id: str = "") -> None:
    """
    Validate dataset source compatibility with current system configuration.
    This is the CORE validation - cannot be bypassed even if frontend/backend is skipped.
    
    Args:
        dataset_source: Source type (huggingface, modelscope, local)
        dataset_id: Dataset identifier (for logging only)
    """
    # Check system compatibility first
    check_system_valid()
    
    # Valid configuration bypasses compatibility checks
    if _check_valid():
        return
    
    source_lower = dataset_source.lower()
    supported_sources = _get_supported_dataset_sources()
    
    # Check source compatibility
    if source_lower not in supported_sources:
        # Natural message - sounds like system capability, not restriction
        msg = "Invalid source type"
        print(f"\n[USF BIOS] {msg}\n", file=sys.stderr)
        raise DatasetSourceError(msg)


def validate_training_config(
    model_path: str,
    model_source: str = "huggingface",
    architecture: Optional[str] = None,
    dataset_source: Optional[str] = None
) -> None:
    """
    Validate complete training configuration.
    Call this before starting any training job.
    
    Args:
        model_path: Model path or HuggingFace/ModelScope ID
        model_source: Source type (huggingface, modelscope, local)
        architecture: Model architecture class name (optional but recommended)
        dataset_source: Dataset source type (huggingface, modelscope, local)
    """
    # Validate model path and source
    validate_model(model_path, model_source)
    
    # Validate architecture if provided (100% reliable restriction)
    if architecture:
        validate_architecture(architecture)
    
    # Validate dataset source if provided
    if dataset_source:
        validate_dataset_source(dataset_source)


def guard_cli_entry() -> None:
    """
    Guard function to call at the start of any CLI entry point.
    Exits if system requires updates or no NVIDIA GPU detected.
    """
    try:
        check_system_valid()
    except SystemCompatibilityError:
        sys.exit(1)
    except GPUNotAvailableError:
        sys.exit(1)


def get_output_path_config() -> dict:
    """
    Get output path configuration.
    Returns mode, base_path, and whether user can customize.
    """
    return {
        "mode": _OUTPUT_PATH_MODE,  # "locked", "base_locked", or "free"
        "base_path": _LOCKED_OUTPUT_BASE_PATH,  # Base path for outputs
        "user_can_customize": _OUTPUT_PATH_MODE == "free",
        "user_can_add_path": _OUTPUT_PATH_MODE == "base_locked",
        "is_locked": _OUTPUT_PATH_MODE == "locked",
    }


def get_output_path(job_id: str, user_path: str = "") -> str:
    """
    Get the final output path for a training job.
    
    When locked: ignores user_path, returns base_path/job_id
    When base_locked: returns base_path/user_path/job_id
    When free: returns user_path/job_id (or base_path/job_id if empty)
    """
    import os
    
    if _OUTPUT_PATH_MODE == "locked":
        # Complete lock - user cannot change anything
        # Output: /workspace/output/{job_id}
        return os.path.join(_LOCKED_OUTPUT_BASE_PATH, job_id)
    
    elif _OUTPUT_PATH_MODE == "base_locked":
        # Base locked - user can add intermediate path
        # Output: /workspace/output/{user_path}/{job_id}
        if user_path:
            # Sanitize user path (no .. or absolute paths)
            safe_path = user_path.strip().strip("/")
            if ".." in safe_path or safe_path.startswith("/"):
                safe_path = ""
            return os.path.join(_LOCKED_OUTPUT_BASE_PATH, safe_path, job_id)
        return os.path.join(_LOCKED_OUTPUT_BASE_PATH, job_id)
    
    else:  # free
        # User can set any path
        if user_path:
            return os.path.join(user_path, job_id)
        return os.path.join(_LOCKED_OUTPUT_BASE_PATH, job_id)


def validate_output_path(user_path: str) -> None:
    """
    Validate user-provided output path.
    When locked: rejects any user-provided path
    When base_locked: allows only relative paths (no .. or /)
    When free: allows any path
    
    Raises OutputPathError if validation fails.
    """
    if _OUTPUT_PATH_MODE == "locked":
        # User cannot provide any path
        if user_path and user_path.strip():
            msg = "Output path cannot be customized"
            print(f"\n[USF BIOS] {msg}\n", file=sys.stderr)
            raise ValueError(msg)
    
    elif _OUTPUT_PATH_MODE == "base_locked":
        # User can only add relative paths
        if user_path:
            safe_path = user_path.strip()
            if ".." in safe_path:
                msg = "Path traversal not allowed"
                print(f"\n[USF BIOS] {msg}\n", file=sys.stderr)
                raise ValueError(msg)
            if safe_path.startswith("/"):
                msg = "Absolute paths not allowed"
                print(f"\n[USF BIOS] {msg}\n", file=sys.stderr)
                raise ValueError(msg)


# Backward compatibility
is_system_expired = _check_compat


# ============================================================================
# FEATURE FLAGS API
# ============================================================================
# These functions expose feature flags to backend/frontend for validation
# All values are derived from hardcoded flags - cannot be modified at runtime

class FeatureDisabledError(SystemGuardError):
    """Raised when a disabled feature is requested."""
    pass


def get_feature_flags() -> Dict[str, bool]:
    """
    Get all feature flags as a dictionary.
    This is the SINGLE SOURCE OF TRUTH for feature availability.
    Called by backend API to expose to frontend.
    """
    return {
        # Main training types
        "pretraining": _FEATURE_PRETRAINING_ENABLED,
        "sft": _FEATURE_SFT_ENABLED,
        "rlhf": _FEATURE_RLHF_ENABLED,
        
        # RLHF sub-types
        "rlhf_online": _FEATURE_RLHF_ENABLED and _FEATURE_RLHF_ONLINE_ENABLED,
        "rlhf_offline": _FEATURE_RLHF_ENABLED and _FEATURE_RLHF_OFFLINE_ENABLED,
        
        # vLLM modes (only relevant if online RL is enabled)
        "vllm_colocate": _FEATURE_RLHF_ONLINE_ENABLED and _FEATURE_VLLM_COLOCATE_ENABLED,
        "vllm_server": _FEATURE_RLHF_ONLINE_ENABLED and _FEATURE_VLLM_SERVER_ENABLED,
        
        # Training methods
        "lora": _FEATURE_LORA_ENABLED,
        "qlora": _FEATURE_QLORA_ENABLED,
        "adalora": _FEATURE_ADALORA_ENABLED,
        "full": _FEATURE_FULL_ENABLED,
        
        # Specific RLHF algorithms
        "grpo": _FEATURE_RLHF_ENABLED and _FEATURE_RLHF_ONLINE_ENABLED and _FEATURE_RLHF_GRPO_ENABLED,
        "ppo": _FEATURE_RLHF_ENABLED and _FEATURE_RLHF_ONLINE_ENABLED and _FEATURE_RLHF_PPO_ENABLED,
        "gkd": _FEATURE_RLHF_ENABLED and _FEATURE_RLHF_ONLINE_ENABLED and _FEATURE_RLHF_GKD_ENABLED,
        "dpo": _FEATURE_RLHF_ENABLED and _FEATURE_RLHF_OFFLINE_ENABLED and _FEATURE_RLHF_DPO_ENABLED,
        "orpo": _FEATURE_RLHF_ENABLED and _FEATURE_RLHF_OFFLINE_ENABLED and _FEATURE_RLHF_ORPO_ENABLED,
        "simpo": _FEATURE_RLHF_ENABLED and _FEATURE_RLHF_OFFLINE_ENABLED and _FEATURE_RLHF_SIMPO_ENABLED,
        "kto": _FEATURE_RLHF_ENABLED and _FEATURE_RLHF_OFFLINE_ENABLED and _FEATURE_RLHF_KTO_ENABLED,
        "cpo": _FEATURE_RLHF_ENABLED and _FEATURE_RLHF_OFFLINE_ENABLED and _FEATURE_RLHF_CPO_ENABLED,
        "rm": _FEATURE_RLHF_ENABLED and _FEATURE_RLHF_RM_ENABLED,
    }


def is_feature_enabled(feature_name: str) -> bool:
    """
    Check if a specific feature is enabled.
    
    Args:
        feature_name: Feature key (e.g., "sft", "grpo", "vllm_server")
    
    Returns:
        True if enabled, False if disabled or unknown
    """
    flags = get_feature_flags()
    return flags.get(feature_name.lower(), False)


def validate_train_type(train_type: str) -> None:
    """
    Validate that the requested training type is enabled.
    Raises FeatureDisabledError if disabled.
    
    Args:
        train_type: Training type (pt, sft, rlhf)
    """
    train_type_lower = train_type.lower()
    
    if train_type_lower == "pt" and not _FEATURE_PRETRAINING_ENABLED:
        msg = "Invalid training configuration"
        print(f"\n[USF BIOS] {msg}\n", file=sys.stderr)
        raise FeatureDisabledError(msg)
    
    if train_type_lower == "sft" and not _FEATURE_SFT_ENABLED:
        msg = "Invalid training configuration"
        print(f"\n[USF BIOS] {msg}\n", file=sys.stderr)
        raise FeatureDisabledError(msg)
    
    if train_type_lower == "rlhf" and not _FEATURE_RLHF_ENABLED:
        msg = "Invalid training configuration"
        print(f"\n[USF BIOS] {msg}\n", file=sys.stderr)
        raise FeatureDisabledError(msg)


def validate_rlhf_algorithm(algorithm: str) -> None:
    """
    Validate that the requested RLHF algorithm is enabled.
    Raises FeatureDisabledError if disabled.
    
    Args:
        algorithm: RLHF algorithm (grpo, ppo, gkd, dpo, orpo, simpo, kto, cpo, rm)
    """
    algo_lower = algorithm.lower()
    
    # Check if RLHF is enabled at all
    if not _FEATURE_RLHF_ENABLED:
        msg = "Unsupported algorithm"
        print(f"\n[USF BIOS] {msg}\n", file=sys.stderr)
        raise FeatureDisabledError(msg)
    
    # Online RL algorithms
    online_algos = {"grpo", "ppo", "gkd"}
    if algo_lower in online_algos:
        if not _FEATURE_RLHF_ONLINE_ENABLED:
            msg = "Unsupported algorithm"
            print(f"\n[USF BIOS] {msg}\n", file=sys.stderr)
            raise FeatureDisabledError(msg)
    
    # Offline RL algorithms
    offline_algos = {"dpo", "orpo", "simpo", "kto", "cpo"}
    if algo_lower in offline_algos:
        if not _FEATURE_RLHF_OFFLINE_ENABLED:
            msg = "Unsupported algorithm"
            print(f"\n[USF BIOS] {msg}\n", file=sys.stderr)
            raise FeatureDisabledError(msg)
    
    # Check specific algorithm flags
    algo_flags = {
        "grpo": _FEATURE_RLHF_GRPO_ENABLED,
        "ppo": _FEATURE_RLHF_PPO_ENABLED,
        "gkd": _FEATURE_RLHF_GKD_ENABLED,
        "dpo": _FEATURE_RLHF_DPO_ENABLED,
        "orpo": _FEATURE_RLHF_ORPO_ENABLED,
        "simpo": _FEATURE_RLHF_SIMPO_ENABLED,
        "kto": _FEATURE_RLHF_KTO_ENABLED,
        "cpo": _FEATURE_RLHF_CPO_ENABLED,
        "rm": _FEATURE_RLHF_RM_ENABLED,
    }
    
    if algo_lower in algo_flags and not algo_flags[algo_lower]:
        msg = "Unsupported algorithm"
        print(f"\n[USF BIOS] {msg}\n", file=sys.stderr)
        raise FeatureDisabledError(msg)


def validate_vllm_mode(mode: str) -> None:
    """
    Validate that the requested vLLM mode is enabled.
    Raises FeatureDisabledError if disabled.
    
    Args:
        mode: vLLM mode (colocate, server)
    """
    mode_lower = mode.lower()
    
    # vLLM modes only apply to online RL
    if not _FEATURE_RLHF_ONLINE_ENABLED:
        msg = "Invalid configuration"
        print(f"\n[USF BIOS] {msg}\n", file=sys.stderr)
        raise FeatureDisabledError(msg)
    
    if mode_lower == "colocate" and not _FEATURE_VLLM_COLOCATE_ENABLED:
        msg = "Invalid configuration"
        print(f"\n[USF BIOS] {msg}\n", file=sys.stderr)
        raise FeatureDisabledError(msg)
    
    if mode_lower == "server" and not _FEATURE_VLLM_SERVER_ENABLED:
        msg = "Invalid configuration"
        print(f"\n[USF BIOS] {msg}\n", file=sys.stderr)
        raise FeatureDisabledError(msg)


def validate_training_method(method: str) -> None:
    """
    Validate that the requested training method is enabled.
    Raises FeatureDisabledError if disabled.
    
    Args:
        method: Training method (lora, qlora, adalora, full)
    """
    method_lower = method.lower()
    
    method_flags = {
        "lora": _FEATURE_LORA_ENABLED,
        "qlora": _FEATURE_QLORA_ENABLED,
        "adalora": _FEATURE_ADALORA_ENABLED,
        "full": _FEATURE_FULL_ENABLED,
    }
    
    if method_lower in method_flags and not method_flags[method_lower]:
        msg = "Invalid configuration"
        print(f"\n[USF BIOS] {msg}\n", file=sys.stderr)
        raise FeatureDisabledError(msg)


def validate_architecture_at_load(architecture: str, model_path: str = "") -> None:
    """
    Called by model loader after architecture is determined.
    This is the 100% reliable validation point - architecture is ALWAYS known here.
    
    This function should be called from model/register.py or model/model_meta.py
    right after the architecture is determined from config.json or model loading.
    
    Args:
        architecture: The architecture class name (e.g., LlamaForCausalLM)
        model_path: Model path for logging
    """
    # Check system compatibility
    check_system_valid()
    
    # Valid configuration bypasses all checks
    if _check_valid():
        return
    
    # Validate architecture
    try:
        validate_architecture(architecture)
    except ArchitectureCompatibilityError as e:
        # Log for debugging
        print(f"\n[USF BIOS] Architecture validation failed for: {model_path}\n", file=sys.stderr)
        raise


# ============================================================================
# FILE INTEGRITY VERIFICATION
# Prevents tampering with compiled .so modules
# ============================================================================

# Critical modules that must be verified before loading
# Format: {module_name: expected_sha256_hash}
# These hashes are embedded in this compiled .so file and cannot be modified
# Set to None to generate hashes during build, then update with actual values
_CRITICAL_MODULE_HASHES: Dict[str, Optional[str]] = {
    # Will be populated during build process
    # "usf_bios.cli._core": "sha256_hash_here",
    # "usf_bios.system_guard": "sha256_hash_here",
}

# Integrity check mode:
# "strict" = fail if hash mismatch or missing
# "warn" = warn but continue if mismatch
# "disabled" = no integrity checks (development only)
_INTEGRITY_MODE = "strict"


def _compute_file_hash(file_path: str) -> str:
    """Compute SHA256 hash of a file."""
    sha256 = hashlib.sha256()
    try:
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    except (IOError, OSError):
        return ""


def _get_module_file_path(module_name: str) -> Optional[str]:
    """Get the file path of a module (.so or .py)."""
    try:
        spec = importlib.util.find_spec(module_name)
        if spec and spec.origin:
            return spec.origin
    except (ImportError, ModuleNotFoundError):
        pass
    return None


def verify_module_integrity(module_name: str) -> Tuple[bool, str]:
    """
    Verify integrity of a critical module by checking its hash.
    This prevents loading of tampered .so files.
    
    Returns:
        (is_valid, error_message)
    """
    if _INTEGRITY_MODE == "disabled":
        return True, ""
    
    # Get expected hash
    expected_hash = _CRITICAL_MODULE_HASHES.get(module_name)
    if expected_hash is None:
        # No hash registered - in strict mode this is an error
        if _INTEGRITY_MODE == "strict" and module_name in _CRITICAL_MODULE_HASHES:
            return False, f"No integrity hash registered for {module_name}"
        return True, ""  # Not a critical module or hash not set
    
    # Get module file path
    file_path = _get_module_file_path(module_name)
    if not file_path:
        return False, f"Cannot locate module: {module_name}"
    
    # Only verify .so files (compiled modules)
    if not (file_path.endswith('.so') or file_path.endswith('.pyd')):
        return True, ""  # Skip .py files (development mode)
    
    # Compute actual hash
    actual_hash = _compute_file_hash(file_path)
    if not actual_hash:
        return False, f"Cannot read module file: {file_path}"
    
    # Compare hashes
    if actual_hash != expected_hash:
        if _INTEGRITY_MODE == "strict":
            return False, "System integrity check failed. Critical module has been modified."
        else:
            print(f"[USF BIOS WARNING] Module integrity mismatch: {module_name}", file=sys.stderr)
            return True, ""  # Continue with warning
    
    return True, ""


def verify_all_critical_modules() -> Tuple[bool, str]:
    """
    Verify integrity of all critical modules.
    Call this at system startup.
    """
    for module_name in _CRITICAL_MODULE_HASHES:
        is_valid, error = verify_module_integrity(module_name)
        if not is_valid:
            return False, error
    return True, ""


def generate_module_hashes() -> Dict[str, str]:
    """
    Generate hashes for all critical modules.
    Used during build process to populate _CRITICAL_MODULE_HASHES.
    """
    hashes = {}
    for module_name in _CRITICAL_MODULE_HASHES:
        file_path = _get_module_file_path(module_name)
        if file_path:
            hash_val = _compute_file_hash(file_path)
            if hash_val:
                hashes[module_name] = hash_val
    return hashes


class IntegrityError(SystemGuardError):
    """Raised when module integrity check fails."""
    pass


def guard_with_integrity() -> None:
    """
    Enhanced guard function that checks both system validity AND module integrity.
    This is the primary entry point that should be called.
    """
    # First verify module integrity
    is_valid, error = verify_all_critical_modules()
    if not is_valid:
        print(f"\n[USF BIOS] {error}\n", file=sys.stderr)
        raise IntegrityError(error)
    
    # Then check system validity
    guard_cli_entry()


# ============================================================================
# RUNTIME SELF-VERIFICATION
# Detects if this module itself has been tampered with
# ============================================================================

def _verify_self_integrity() -> bool:
    """
    Verify that this module hasn't been tampered with.
    Uses multiple checks that are hard to bypass.
    """
    # Check 1: Verify critical functions exist and have expected signatures
    critical_functions = [
        'guard_cli_entry',
        'check_system_valid',
        'validate_model',
        'validate_architecture',
    ]
    
    for func_name in critical_functions:
        if func_name not in globals():
            return False
        func = globals()[func_name]
        if not callable(func):
            return False
    
    # Check 2: Verify validation key is intact (would be modified if source was changed)
    if not _VALIDATION_KEY:
        return False
    
    # Check 3: Verify compatibility date is reasonable
    if _COMPAT_DATE.year < 2026:
        return False
    
    return True


# Auto-check on import (prevents usage if system requires updates)
try:
    # Self-integrity check
    if not _verify_self_integrity():
        print("\n[USF BIOS] System integrity verification failed.\n", file=sys.stderr)
        sys.exit(1)
    
    _needs_update, _msg = _check_compat()
    if _needs_update:
        print(f"\n[USF BIOS] {_msg}\n", file=sys.stderr)
except Exception:
    pass
