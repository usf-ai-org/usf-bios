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
from ...services.model_registry_service import ModelRegistryService

router = APIRouter()


# Request/Response models for Model Registry
class ModelRegistration(BaseModel):
    """Request model for registering a model"""
    name: str
    source: Literal["huggingface", "modelscope", "local"]
    source_id: str  # HF/MS model ID or local path
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


@router.get("/supported", response_model=SupportedModels)
async def get_supported_models():
    """Get list of supported models"""
    return SupportedModels(
        usf_models=[
            {"id": "arpitsh018/usf-omega-40b-base", "name": "USF Omega 40B Base", "type": "text", "size": "40B"},
            {"id": "arpitsh018/usf-omega-40b-instruct", "name": "USF Omega 40B Instruct", "type": "text", "size": "40B"},
            {"id": "arpitsh018/usf-mini-7b-base", "name": "USF Mini 7B Base", "type": "text", "size": "7B"},
            {"id": "arpitsh018/usf-mini-7b-instruct", "name": "USF Mini 7B Instruct", "type": "text", "size": "7B"},
            {"id": "arpitsh018/usf-omega-vl-40b", "name": "USF Omega VL 40B", "type": "vision", "size": "40B"},
        ],
        popular_models=[
            {"id": "Qwen/Qwen2.5-7B-Instruct", "name": "Qwen 2.5 7B Instruct", "type": "text", "size": "7B"},
            {"id": "Qwen/Qwen2.5-14B-Instruct", "name": "Qwen 2.5 14B Instruct", "type": "text", "size": "14B"},
            {"id": "meta-llama/Llama-3.1-8B-Instruct", "name": "Llama 3.1 8B Instruct", "type": "text", "size": "8B"},
            {"id": "mistralai/Mistral-7B-Instruct-v0.3", "name": "Mistral 7B Instruct", "type": "text", "size": "7B"},
            {"id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "name": "DeepSeek R1 7B", "type": "text", "size": "7B"},
            {"id": "Qwen/Qwen2.5-VL-7B-Instruct", "name": "Qwen 2.5 VL 7B", "type": "vision", "size": "7B"},
        ]
    )


@router.post("/validate", response_model=ModelValidation)
async def validate_model(
    model_path: str = Query(..., description="Model path or HF ID"),
    source: ModelSource = Query(ModelSource.HUGGINGFACE, description="Model source")
):
    """Validate if a model exists and is accessible"""
    try:
        if source == ModelSource.LOCAL:
            path = Path(model_path)
            if not path.exists():
                return ModelValidation(valid=False, error="Local path does not exist")
            
            config_path = path / "config.json"
            if not config_path.exists():
                return ModelValidation(valid=False, error="No config.json found in model directory")
            
            # Try to read config
            import json
            with open(config_path) as f:
                config = json.load(f)
            
            model_type = config.get("model_type", "unknown")
            return ModelValidation(valid=True, model_type=model_type)
        
        else:
            # For HF/MS models, we assume they're valid if format is correct
            if "/" not in model_path:
                return ModelValidation(valid=False, error="Invalid model ID format. Use 'organization/model-name'")
            
            return ModelValidation(valid=True, model_type="remote")
    
    except Exception as e:
        return ModelValidation(valid=False, error="Failed to validate model")


# ============================================================================
# Model Registry Endpoints
# ============================================================================

@router.post("/registry/register")
async def register_model(registration: ModelRegistration, db: Session = Depends(get_db)):
    """Register a model in the global registry"""
    try:
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
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to register model")


@router.get("/registry/list")
async def list_registered_models(
    source: Optional[str] = Query(None, description="Filter by source (huggingface, modelscope, local)"),
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
