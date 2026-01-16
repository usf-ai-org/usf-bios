# Copyright (c) US Inc. All rights reserved.
"""Model-related endpoints"""

import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel

from ...models.schemas import ModelSource, ModelValidation

router = APIRouter()


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
        return ModelValidation(valid=False, error=str(e))
