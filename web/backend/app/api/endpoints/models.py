# Copyright (c) US Inc. All rights reserved.
"""Model-related endpoints"""

import os
from pathlib import Path
from typing import Optional, Literal, List

from fastapi import APIRouter, Query, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ...models.schemas import ModelSource, ModelValidation
from ...core.database import get_db
from ...core.capabilities import get_validator
from ...services.model_registry_service import ModelRegistryService

router = APIRouter()


# Request/Response models for Model Registry
class ModelRegistration(BaseModel):
    """Request model for registering a model"""
    name: str
    source: str = "local"  # Model source
    source_id: str  # Local path to model directory
    description: Optional[str] = None
    model_type: Optional[str] = None  # llm, vlm, etc.
    model_size: Optional[str] = None  # 7B, 14B, 40B, etc.


class RegisteredModelInfo(BaseModel):
    """Registered model info"""
    id: str
    name: str
    source: str
    source_id: str
    description: Optional[str]
    model_type: Optional[str]
    model_size: Optional[str]
    times_used: int
    last_used_at: Optional[str]
    created_at: Optional[str]
    trainings_count: int


class SupportedModels(BaseModel):
    usf_models: list
    popular_models: list


@router.get("/supported", response_model=SupportedModels, include_in_schema=False)
async def get_supported_models():
    """Get list of supported models (hidden - returns empty)"""
    # Do not expose model suggestions - only local models supported
    return SupportedModels(
        usf_models=[],
        popular_models=[]
    )


@router.post("/validate", response_model=ModelValidation)
async def validate_model(
    model_path: str = Query(..., description="Path to model directory"),
    source: ModelSource = Query(ModelSource.LOCAL, description="Model source")
):
    """Validate if a model exists and is accessible"""
    try:
        # First validate source is allowed
        validator = get_validator()
        source_str = source.value if hasattr(source, 'value') else str(source)
        is_valid, msg = validator.validate_model_path(model_path, source_str)
        if not is_valid:
            return ModelValidation(valid=False, error=msg)
        
        if source == ModelSource.LOCAL:
            path = Path(model_path)
            
            # Check if path exists
            if not path.exists():
                return ModelValidation(valid=False, error="Model path does not exist. Please verify the path is correct.")
            
            # Check read permissions
            if not os.access(model_path, os.R_OK):
                return ModelValidation(valid=False, error="Cannot access model path. Please check read permissions.")
            
            # For directories, check for model files
            if path.is_dir():
                config_path = path / "config.json"
                if config_path.exists():
                    # Try to read config and get architecture
                    import json
                    try:
                        with open(config_path) as f:
                            config = json.load(f)
                        
                        model_type = config.get("model_type", "unknown")
                        architectures = config.get("architectures", [])
                        
                        # Validate architecture if available
                        if architectures and len(architectures) > 0:
                            arch_valid, arch_msg = validator.validate_architecture(architectures[0])
                            if not arch_valid:
                                return ModelValidation(valid=False, error=arch_msg)
                        
                        return ModelValidation(valid=True, model_type=model_type)
                    except (json.JSONDecodeError, IOError):
                        return ModelValidation(valid=False, error="Could not read model configuration.")
                else:
                    # Check for model files without config.json
                    model_files = list(path.glob("*.safetensors")) + list(path.glob("*.bin"))
                    if model_files:
                        return ModelValidation(valid=True, model_type="unknown")
                    return ModelValidation(valid=False, error="No model files found in directory.")
            else:
                # Single file - check extension
                if path.suffix.lower() in [".safetensors", ".bin", ".pt", ".pth", ".gguf"]:
                    return ModelValidation(valid=True, model_type="file")
                return ModelValidation(valid=False, error="Unsupported model file format.")
        
        else:
            # For remote models - validate source is allowed
            source_key = "huggingface" if source == ModelSource.HUGGINGFACE else "modelscope"
            is_source_valid, source_msg = validator.validate_model_path(model_path, source_key)
            if not is_source_valid:
                return ModelValidation(valid=False, error=source_msg)
            
            if "/" not in model_path:
                return ModelValidation(valid=False, error="Invalid model ID format. Use 'organization/model-name'")
            
            return ModelValidation(valid=True, model_type="remote")
    
    except Exception as e:
        return ModelValidation(valid=False, error="Failed to validate model")


# ============================================================================
# Model Registry Endpoints
# ============================================================================

# Valid source values (internal - not exposed in schema)
_VALID_MODEL_SOURCES = {"local", "huggingface", "modelscope"}


@router.post("/registry/register")
async def register_model(registration: ModelRegistration, db: Session = Depends(get_db)):
    """Register a model in the global registry"""
    try:
        # Validate source value is valid
        if registration.source not in _VALID_MODEL_SOURCES:
            raise HTTPException(status_code=400, detail="Invalid source type")
        
        # Validate source is allowed by system configuration
        validator = get_validator()
        is_valid, msg = validator.validate_model_path(registration.source_id, registration.source)
        if not is_valid:
            raise HTTPException(status_code=403, detail=msg)
        
        # For local models, validate path exists and is accessible
        if registration.source == "local":
            path = Path(registration.source_id)
            if not path.exists():
                raise HTTPException(status_code=400, detail="Model path does not exist. Please verify the path is correct.")
            if not os.access(registration.source_id, os.R_OK):
                raise HTTPException(status_code=400, detail="Cannot access model path. Please check read permissions.")
            
            # Validate architecture if config.json exists
            if path.is_dir():
                config_path = path / "config.json"
                if config_path.exists():
                    try:
                        import json
                        with open(config_path) as f:
                            config = json.load(f)
                        architectures = config.get("architectures", [])
                        if architectures and len(architectures) > 0:
                            arch_valid, arch_msg = validator.validate_architecture(architectures[0])
                            if not arch_valid:
                                raise HTTPException(status_code=403, detail=arch_msg)
                    except (json.JSONDecodeError, IOError):
                        pass  # Continue if config can't be read
        
        service = ModelRegistryService(db)
        result = service.register_model(
            name=registration.name,
            source=registration.source,
            source_id=registration.source_id,
            description=registration.description,
            model_type=registration.model_type,
            model_size=registration.model_size
        )
        return {"success": True, "model": result}
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to register model")


@router.get("/registry/list")
async def list_registered_models(
    source: Optional[str] = Query(None, description="Filter by source"),
    db: Session = Depends(get_db)
):
    """List all registered models"""
    try:
        service = ModelRegistryService(db)
        models = service.list_models(source=source)
        return {"models": models, "total": len(models)}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to list models")


@router.get("/registry/{model_id}")
async def get_registered_model(model_id: str, db: Session = Depends(get_db)):
    """Get a specific registered model by ID"""
    try:
        service = ModelRegistryService(db)
        model = service.get_model(model_id)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
        return model
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to get model")


@router.get("/registry/{model_id}/delete-info")
async def get_model_delete_info(model_id: str, db: Session = Depends(get_db)):
    """Get information needed for delete confirmation (returns model name)"""
    try:
        service = ModelRegistryService(db)
        model = service.get_model(model_id)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
        
        # Check if model is used by any trainings
        trainings_count = len(model.training_jobs) if hasattr(model, 'training_jobs') else 0
        
        return {
            "model_id": model_id,
            "model_name": model.name,
            "source": model.source,
            "trainings_count": trainings_count,
            "can_delete": trainings_count == 0,
            "confirm_text": model.name  # User must type this to confirm deletion
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to get model info")


@router.delete("/registry/{model_id}")
async def unregister_model(
    model_id: str,
    confirm: str = Query(..., description="Type the model NAME to confirm"),
    force: bool = Query(False, description="Force delete even if used by trainings"),
    db: Session = Depends(get_db)
):
    """Unregister a model from the registry. User must type the model name to confirm."""
    try:
        service = ModelRegistryService(db)
        model = service.get_model(model_id)
        if not model:
            raise HTTPException(status_code=404, detail=f"Model not found: {model_id}")
        
        # Validate confirmation matches the model name
        if confirm != model.name:
            raise HTTPException(
                status_code=400, 
                detail=f"Confirmation failed. You must type '{model.name}' to delete this model."
            )
        
        result = service.delete_model(model_id, force=force)
        result["deleted_name"] = model.name
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to unregister model")


@router.get("/registry/popular")
async def get_popular_models(limit: int = Query(10, ge=1, le=50), db: Session = Depends(get_db)):
    """Get most frequently used models"""
    try:
        service = ModelRegistryService(db)
        models = service.get_popular_models(limit=limit)
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to get popular models")


# ============================================================================
# Model Info Endpoint - Get context length and other model-specific info
# ============================================================================

# Known context lengths for popular models (avoids network calls for HF/MS)
# Format: "model_name_pattern" -> context_length
# This is checked BEFORE making any network requests
KNOWN_MODEL_CONTEXT_LENGTHS: dict = {
    # Qwen family
    "qwen": 32768,
    "qwen2": 131072,
    "qwen2.5": 131072,
    "qwen1.5": 32768,
    # Llama family
    "llama-2": 4096,
    "llama-3": 8192,
    "llama-3.1": 131072,
    "llama-3.2": 131072,
    "llama-3.3": 131072,
    # Mistral family
    "mistral": 32768,
    "mixtral": 32768,
    "mistral-nemo": 131072,
    # DeepSeek family
    "deepseek": 16384,
    "deepseek-v2": 131072,
    "deepseek-v3": 131072,
    "deepseek-coder": 16384,
    # Yi family
    "yi": 4096,
    "yi-1.5": 16384,
    # Phi family
    "phi-2": 2048,
    "phi-3": 131072,
    "phi-4": 16384,
    # Gemma family
    "gemma": 8192,
    "gemma-2": 8192,
    # InternLM
    "internlm": 8192,
    "internlm2": 32768,
    # ChatGLM
    "chatglm": 8192,
    "chatglm2": 32768,
    "chatglm3": 32768,
    "glm-4": 131072,
    # Baichuan
    "baichuan": 4096,
    "baichuan2": 4096,
    # Others
    "vicuna": 4096,
    "falcon": 2048,
    "bloom": 2048,
    "opt": 2048,
    "gpt2": 1024,
    "gpt-j": 2048,
    "codellama": 16384,
    "starcoder": 8192,
    "command-r": 131072,
}

# Cache for model info (avoids repeated disk reads or network calls)
_model_info_cache: dict = {}


def _get_context_from_known_models(model_path: str) -> Optional[int]:
    """
    Get context length from known models lookup table.
    Checks if model_path contains any known model name patterns.
    Returns None if not found in lookup table.
    """
    model_lower = model_path.lower()
    
    # Check for exact or partial matches
    for pattern, context_length in KNOWN_MODEL_CONTEXT_LENGTHS.items():
        if pattern in model_lower:
            return context_length
    
    return None


class ModelInfo(BaseModel):
    """Model information including context length"""
    model_path: str
    model_type: Optional[str] = None
    architecture: Optional[str] = None
    context_length: int = 4096  # Default fallback
    vocab_size: Optional[int] = None
    hidden_size: Optional[int] = None
    num_layers: Optional[int] = None
    num_attention_heads: Optional[int] = None


def _get_context_length_from_config(config: dict) -> int:
    """
    Extract context length from model config.json.
    Different models use different field names for context length.
    
    Returns the context length or 4096 as default.
    """
    # Common field names for context/sequence length in different model architectures
    context_fields = [
        "max_position_embeddings",      # Most common (Llama, Mistral, Qwen, etc.)
        "max_sequence_length",          # Some models
        "n_positions",                  # GPT-2 style
        "seq_length",                   # Some models
        "max_seq_len",                  # Some models
        "context_length",               # Direct field
        "sliding_window",               # Mistral sliding window (use as hint)
        "original_max_position_embeddings",  # For extended context models
    ]
    
    context_length = None
    
    for field in context_fields:
        if field in config and config[field] is not None:
            value = config[field]
            if isinstance(value, int) and value > 0:
                # Use the largest value found (some models have both original and extended)
                if context_length is None or value > context_length:
                    context_length = value
    
    # Check for rope_scaling which indicates extended context
    if "rope_scaling" in config and config["rope_scaling"]:
        rope_config = config["rope_scaling"]
        if isinstance(rope_config, dict):
            factor = rope_config.get("factor", 1)
            if factor > 1 and context_length:
                # Some models use rope scaling to extend context
                # But max_position_embeddings already reflects this, so don't multiply
                pass
    
    return context_length or 4096  # Default fallback


@router.get("/info")
async def get_model_info(
    model_path: str = Query(..., description="Path to model or HuggingFace model ID"),
    source: ModelSource = Query(ModelSource.LOCAL, description="Model source")
):
    """
    Get model information including context length.
    
    This endpoint uses a tiered approach for efficiency:
    1. Check cache first (instant)
    2. For local models: read config.json from disk (fast, no GPU)
    3. For remote models: check known models lookup table (instant, no network)
    4. Only fetch from HuggingFace if model not in lookup table
    
    NOTE: This does NOT load the model into GPU memory.
    It only reads the small config.json file (~50KB).
    
    Frontend uses this to set dynamic ranges for:
    - max_length (sequence length for training)
    - max_completion_length (for RLHF algorithms)
    - max_new_tokens (for inference)
    """
    global _model_info_cache
    
    try:
        import json
        
        # Check cache first
        cache_key = f"{source.value}:{model_path}"
        if cache_key in _model_info_cache:
            return _model_info_cache[cache_key]
        
        result = ModelInfo(model_path=model_path)
        
        if source == ModelSource.LOCAL:
            # LOCAL: Read config.json from disk (fast, no GPU needed)
            path = Path(model_path)
            
            if not path.exists():
                # Try known models lookup as fallback
                known_context = _get_context_from_known_models(model_path)
                if known_context:
                    result.context_length = known_context
                return result
            
            config_path = path / "config.json" if path.is_dir() else None
            
            if config_path and config_path.exists():
                try:
                    with open(config_path) as f:
                        config = json.load(f)
                    
                    # Extract context length
                    result.context_length = _get_context_length_from_config(config)
                    
                    # Extract other info
                    result.model_type = config.get("model_type")
                    architectures = config.get("architectures", [])
                    if architectures:
                        result.architecture = architectures[0]
                    result.vocab_size = config.get("vocab_size")
                    result.hidden_size = config.get("hidden_size")
                    result.num_layers = config.get("num_hidden_layers")
                    result.num_attention_heads = config.get("num_attention_heads")
                    
                except (json.JSONDecodeError, IOError):
                    # Fallback to known models lookup
                    known_context = _get_context_from_known_models(model_path)
                    if known_context:
                        result.context_length = known_context
        
        else:
            # HUGGINGFACE / MODELSCOPE: Use lookup table first (no network call)
            known_context = _get_context_from_known_models(model_path)
            if known_context:
                result.context_length = known_context
            else:
                # Only fetch from HuggingFace if model not in lookup table
                # This is a last resort and may be slow
                try:
                    from huggingface_hub import hf_hub_download
                    import tempfile
                    
                    with tempfile.TemporaryDirectory() as tmp_dir:
                        config_file = hf_hub_download(
                            repo_id=model_path,
                            filename="config.json",
                            cache_dir=tmp_dir,
                            local_dir=tmp_dir
                        )
                        with open(config_file) as f:
                            config = json.load(f)
                        
                        result.context_length = _get_context_length_from_config(config)
                        result.model_type = config.get("model_type")
                        architectures = config.get("architectures", [])
                        if architectures:
                            result.architecture = architectures[0]
                        result.vocab_size = config.get("vocab_size")
                        
                except Exception:
                    # Final fallback: use reasonable default
                    result.context_length = 8192  # Safe default for most modern models
        
        # Cache the result for future requests
        _model_info_cache[cache_key] = result
        
        return result
        
    except Exception as e:
        # Return defaults on any error
        return ModelInfo(model_path=model_path)
