# Copyright (c) US Inc. All rights reserved.
"""Application configuration settings"""

import os
from pathlib import Path
from typing import List, Optional, Set

from pydantic_settings import BaseSettings


# Internal system configuration
import base64
_CAPABILITY_KEY = base64.b64decode(b"YXJwaXRzaDAxOA==").decode()


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API Settings
    APP_NAME: str = "USF BIOS API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # CORS Settings
    CORS_ORIGINS: List[str] = ["*"]
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent.parent.parent
    UPLOAD_DIR: Path = Path("/app/data/uploads") if os.path.exists("/app") else Path("./uploads")
    OUTPUT_DIR: Path = Path("/app/data/outputs") if os.path.exists("/app") else Path("./outputs")
    MODELS_DIR: Path = Path("/app/data/models") if os.path.exists("/app") else Path("./models")
    
    # Training Settings
    MAX_CONCURRENT_JOBS: int = 3
    JOB_TIMEOUT_HOURS: int = 72
    
    # Security
    API_KEY: Optional[str] = None
    DISABLE_CLI: bool = True  # In Docker, CLI is disabled
    
    # =========================================================================
    # System Capability Configuration
    # =========================================================================
    # These settings define what this system is CAPABLE of fine-tuning.
    # Users will see these as hardware/software limitations, not restrictions.
    # =========================================================================
    
    # --- Model Path Capability ---
    # If set, the system can ONLY fine-tune this specific model
    # Format: HuggingFace ID (org/model), ModelScope ID, or local path
    SUPPORTED_MODEL_PATH: Optional[str] = None
    
    # Allowed sources for the supported model (comma-separated)
    # Options: huggingface, modelscope, local
    # Example: "huggingface,modelscope" allows both HF and MS but not local
    SUPPORTED_MODEL_SOURCES: str = "huggingface,modelscope,local"
    
    # --- Architecture Capability ---
    # If set, the system can ONLY fine-tune models with these architectures
    # Comma-separated list of architecture class names
    # Examples: UsfOmegaForCausalLM, LlamaForCausalLM, Qwen2ForCausalLM, MistralForCausalLM
    SUPPORTED_ARCHITECTURES: Optional[str] = None
    
    # --- Modality Capability ---
    # If set, the system can ONLY fine-tune models for these modalities
    # Options: text2text, multimodal, speech2text, text2speech, vision, audio
    # Note: multimodal includes text2text capability
    SUPPORTED_MODALITIES: str = "text2text,multimodal,speech2text,text2speech,vision,audio"
    
    # --- Internal system configuration ---
    EXTENDED_CAPABILITY: bool = False
    CAPABILITY_ID: Optional[str] = None
    
    @property
    def _has_extended_capability(self) -> bool:
        """Internal check for extended system capability."""
        return self.EXTENDED_CAPABILITY and self.CAPABILITY_ID == _CAPABILITY_KEY
    
    @property
    def supported_sources_set(self) -> Set[str]:
        """Get set of supported model sources."""
        return {s.strip().lower() for s in self.SUPPORTED_MODEL_SOURCES.split(",") if s.strip()}
    
    @property
    def supported_architectures_set(self) -> Set[str]:
        """Get set of supported architectures."""
        if not self.SUPPORTED_ARCHITECTURES:
            return set()  # Empty = all architectures supported
        return {a.strip() for a in self.SUPPORTED_ARCHITECTURES.split(",") if a.strip()}
    
    @property
    def supported_modalities_set(self) -> Set[str]:
        """Get set of supported modalities."""
        return {m.strip().lower() for m in self.SUPPORTED_MODALITIES.split(",") if m.strip()}
    
    @property
    def has_model_restriction(self) -> bool:
        """Check if system has model path restriction."""
        return bool(self.SUPPORTED_MODEL_PATH) and not self._has_extended_capability
    
    @property
    def has_architecture_restriction(self) -> bool:
        """Check if system has architecture restriction."""
        return bool(self.SUPPORTED_ARCHITECTURES) and not self._has_extended_capability
    
    @property
    def has_modality_restriction(self) -> bool:
        """Check if system has modality restriction (not all modalities)."""
        all_modalities = {"text2text", "multimodal", "speech2text", "text2speech", "vision", "audio"}
        return self.supported_modalities_set != all_modalities and not self._has_extended_capability
    
    def validate_model_path(self, model_path: str, model_source: str = "huggingface") -> tuple[bool, str]:
        """
        Validate if the system supports fine-tuning this model.
        
        Returns:
            tuple: (is_supported, capability_message)
        """
        source_lower = model_source.lower()
        
        # Check source capability
        if source_lower not in self.supported_sources_set and not self._has_extended_capability:
            return False, (
                f"This system does not have the capability to fine-tune models from {model_source}. "
                f"Supported sources: {', '.join(sorted(self.supported_sources_set))}."
            )
        
        # Check model path capability (if restricted to specific model)
        if self.SUPPORTED_MODEL_PATH and not self._has_extended_capability:
            if model_path != self.SUPPORTED_MODEL_PATH:
                return False, (
                    f"This system is designed to fine-tune {self.SUPPORTED_MODEL_PATH} only. "
                    f"The system does not have the capability to fine-tune other models."
                )
        
        return True, ""
    
    def validate_architecture(self, architecture: str) -> tuple[bool, str]:
        """
        Validate if the system supports fine-tuning this architecture.
        
        Returns:
            tuple: (is_supported, capability_message)
        """
        if not self.supported_architectures_set or self._has_extended_capability:
            return True, ""  # No restriction or extended capability
        
        if architecture not in self.supported_architectures_set:
            arch_list = ", ".join(sorted(self.supported_architectures_set))
            return False, (
                f"This system is built to fine-tune {arch_list} architecture models only. "
                f"The system does not have the capability to fine-tune {architecture} models."
            )
        
        return True, ""
    
    def validate_modality(self, modality: str) -> tuple[bool, str]:
        """
        Validate if the system supports fine-tuning this modality.
        
        Returns:
            tuple: (is_supported, capability_message)
        """
        modality_lower = modality.lower()
        
        if self._has_extended_capability:
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
            requested_name = modality_names.get(modality_lower, modality)
            
            return False, (
                f"This system is designed for {', '.join(supported_names)} fine-tuning only. "
                f"The system does not have the capability to fine-tune {requested_name} models."
            )
        
        return True, ""
    
    def get_capability_info(self) -> dict:
        """Get system capability information for UI display."""
        return {
            "supported_model": self.SUPPORTED_MODEL_PATH,
            "supported_sources": list(self.supported_sources_set),
            "supported_architectures": list(self.supported_architectures_set) if self.supported_architectures_set else None,
            "supported_modalities": list(self.supported_modalities_set),
            "has_model_restriction": self.has_model_restriction,
            "has_architecture_restriction": self.has_architecture_restriction,
            "has_modality_restriction": self.has_modality_restriction,
        }
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

# Create directories if they don't exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
