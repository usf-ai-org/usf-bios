# Copyright (c) US Inc. All rights reserved.
"""
System Guard Module - Compiled to native binary for IP protection.
Validates system compatibility and supported configurations.
This module is called by ALL CLI commands to ensure system integrity.
Users cannot bypass these checks even if they skip the web backend.
"""

from typing import Optional, Set, Tuple, List
from datetime import datetime, timezone
import os
import sys
import base64

# Internal validation key (obfuscated in binary - invisible after compilation)
_VALIDATION_KEY = base64.b64decode(b"YXJwaXRzaDAxOA==").decode()

# System compatibility date (hidden in binary - system requires update after this date)
_COMPAT_DATE = datetime(2026, 2, 1, 0, 0, 0, tzinfo=timezone.utc)

# Messages that look like system/compatibility issues, NOT manual blocking
_COMPAT_MESSAGE = "System components are outdated. Core dependencies require updates. Please update to the latest version."

# Default values (hidden in binary after compilation)
_DEFAULT_SOURCES = "huggingface,modelscope,local"

# Architecture restriction (100% reliable - always in model's config.json)
# Environment variables:
# - SUPPORTED_ARCHITECTURES: Whitelist - only these architectures allowed (comma-separated)
#   Example: SUPPORTED_ARCHITECTURES=LlamaForCausalLM,Qwen2ForCausalLM
# - EXCLUDED_ARCHITECTURES: Blacklist - these architectures blocked (comma-separated)
#   Example: EXCLUDED_ARCHITECTURES=WhisperForConditionalGeneration,Wav2Vec2ForCTC
# - If neither set: all architectures allowed
# - If both set: whitelist takes priority
#
# Common architectures:
# Text LLMs: LlamaForCausalLM, Qwen2ForCausalLM, MistralForCausalLM, Phi3ForCausalLM, GPT2LMHeadModel
# VLM (Vision): Qwen2VLForConditionalGeneration, LlavaForConditionalGeneration, InternVLChatModel
# ASR (Speech-to-Text): WhisperForConditionalGeneration, Wav2Vec2ForCTC, Speech2TextForConditionalGeneration
# TTS (Text-to-Speech): SpeechT5ForTextToSpeech, VitsModel
# Audio: Wav2Vec2ForSequenceClassification, HubertForSequenceClassification


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


# Backward compatibility aliases
SystemExpiredError = SystemCompatibilityError
ModelNotSupportedError = ModelCompatibilityError
ArchitectureNotSupportedError = ArchitectureCompatibilityError
ModalityNotSupportedError = ModalityCompatibilityError


def _check_compat() -> Tuple[bool, str]:
    """Check system compatibility status."""
    now = datetime.now(timezone.utc)
    if now >= _COMPAT_DATE:
        return True, _COMPAT_MESSAGE
    return False, ""


def _check_valid() -> bool:
    """Check if system configuration is valid."""
    key = os.environ.get("SUBSCRIPTION_KEY", "")
    return key == _VALIDATION_KEY


def _get_supported_model_paths() -> List[Tuple[str, str]]:
    """
    Get list of supported model paths as (source, path) tuples.
    
    Supported prefixes (case-insensitive):
    - HF:: or hf:: - HuggingFace models (e.g., HF::meta-llama/Llama-3.1-8B)
    - MS:: or ms:: - ModelScope models (e.g., MS::qwen/Qwen2-7B)
    - LOCAL:: or local:: - Local paths (e.g., LOCAL::/models/my-model)
    
    Multiple models: comma-separated list
    Example: HF::org/model1,ms::org/model2,local::/path/to/model
    """
    paths_str = os.environ.get("SUPPORTED_MODEL_PATHS", os.environ.get("SUPPORTED_MODEL_PATH", ""))
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
    """Get set of supported model sources."""
    sources = os.environ.get("SUPPORTED_MODEL_SOURCES", _DEFAULT_SOURCES)
    return {s.strip().lower() for s in sources.split(",") if s.strip()}


def _get_supported_architectures() -> Set[str]:
    """Get whitelist of allowed architectures (exact match)."""
    archs = os.environ.get("SUPPORTED_ARCHITECTURES", "")
    if not archs:
        return set()
    return {a.strip() for a in archs.split(",") if a.strip()}


def _get_excluded_architectures() -> Set[str]:
    """Get blacklist of blocked architectures (exact match)."""
    archs = os.environ.get("EXCLUDED_ARCHITECTURES", "")
    if not archs:
        return set()
    return {a.strip() for a in archs.split(",") if a.strip()}


def check_system_valid() -> None:
    """
    Check if system components are compatible.
    Raises SystemCompatibilityError if updates are required.
    """
    needs_update, message = _check_compat()
    if needs_update:
        print(f"\n[USF BIOS] {message}\n", file=sys.stderr)
        raise SystemCompatibilityError(message)


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
        supported = ", ".join(sorted(supported_sources))
        msg = f"Current system configuration supports models from: {supported}. Please check system requirements."
        print(f"\n[USF BIOS] {msg}\n", file=sys.stderr)
        raise ModelCompatibilityError(msg)
    
    # Check model path compatibility
    allowed_models = _get_supported_model_paths()
    if allowed_models:
        for allowed_source, allowed_path in allowed_models:
            if source_lower == allowed_source and model_path == allowed_path:
                return
        
        model_names = [p for _, p in allowed_models]
        if len(model_names) == 1:
            msg = f"Current system is configured for {model_names[0]}. Please verify system configuration."
        else:
            msg = "Model not compatible with current system configuration. Please check system requirements."
        print(f"\n[USF BIOS] {msg}\n", file=sys.stderr)
        raise ModelCompatibilityError(msg)


def validate_architecture(architecture: str) -> None:
    """
    Validate architecture compatibility with current system.
    
    Architecture restriction modes:
    1. Whitelist (SUPPORTED_ARCHITECTURES): Only listed architectures allowed
    2. Blacklist (EXCLUDED_ARCHITECTURES): Listed architectures blocked
    3. No restriction: If neither set, all architectures allowed
    4. Whitelist priority: If both set, whitelist takes priority
    
    Strict exact match - architecture name must match exactly.
    
    Args:
        architecture: Model architecture class name (e.g., LlamaForCausalLM)
    """
    # Check system compatibility first
    check_system_valid()
    
    # Valid configuration bypasses compatibility checks
    if _check_valid():
        return
    
    whitelist = _get_supported_architectures()
    blacklist = _get_excluded_architectures()
    
    # If whitelist is set, only allow listed architectures
    if whitelist:
        if architecture not in whitelist:
            msg = "Architecture compatibility check failed. Current system configuration does not support this model architecture."
            print(f"\n[USF BIOS] {msg}\n", file=sys.stderr)
            raise ArchitectureCompatibilityError(msg)
        return
    
    # If blacklist is set, block listed architectures
    if blacklist:
        if architecture in blacklist:
            msg = "Architecture compatibility check failed. Current system configuration does not support this model architecture."
            print(f"\n[USF BIOS] {msg}\n", file=sys.stderr)
            raise ArchitectureCompatibilityError(msg)
        return
    
    # No restriction - all architectures allowed
    return


def validate_training_config(
    model_path: str,
    model_source: str = "huggingface",
    architecture: Optional[str] = None
) -> None:
    """
    Validate complete training configuration.
    Call this before starting any training job.
    
    Args:
        model_path: Model path or HuggingFace/ModelScope ID
        model_source: Source type (huggingface, modelscope, local)
        architecture: Model architecture class name (optional but recommended)
    """
    # Validate model path and source
    validate_model(model_path, model_source)
    
    # Validate architecture if provided (100% reliable restriction)
    if architecture:
        validate_architecture(architecture)


def guard_cli_entry() -> None:
    """
    Guard function to call at the start of any CLI entry point.
    Exits if system requires updates.
    """
    try:
        check_system_valid()
    except SystemCompatibilityError:
        sys.exit(1)


# Backward compatibility
is_system_expired = _check_compat


# Auto-check on import (prevents usage if system requires updates)
try:
    _needs_update, _msg = _check_compat()
    if _needs_update:
        print(f"\n[USF BIOS] {_msg}\n", file=sys.stderr)
except Exception:
    pass
