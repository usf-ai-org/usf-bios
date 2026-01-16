# Copyright (c) US Inc. All rights reserved.
"""
Inference endpoints for model testing
- Test models before fine-tuning
- Test intermediate checkpoints during fine-tuning  
- Test final fine-tuned models
"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ...services.inference_service import (
    inference_service, 
    InferenceRequest, 
    InferenceResponse,
    ModelInfo
)

router = APIRouter()


class ChatMessage(BaseModel):
    """Chat message format"""
    role: str = Field(..., description="Role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    """Chat inference request"""
    model_path: str = Field(..., description="Model path (HF ID, local path, or checkpoint)")
    messages: List[ChatMessage] = Field(..., description="Chat messages")
    adapter_path: Optional[str] = Field(default=None, description="LoRA adapter path (for fine-tuned models)")
    max_new_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.9, ge=0, le=1)
    top_k: int = Field(default=50, ge=0)
    repetition_penalty: float = Field(default=1.0, ge=1, le=2)
    do_sample: bool = Field(default=True)


class ChatResponse(BaseModel):
    """Chat inference response"""
    success: bool
    response: Optional[str] = None
    tokens_generated: int = 0
    inference_time_ms: int = 0
    model_loaded: str = ""
    error: Optional[str] = None


class MemoryStatus(BaseModel):
    """GPU memory status"""
    success: bool
    memory_freed_gb: float = 0.0
    memory_used_gb: float = 0.0
    error: Optional[str] = None


class LoadModelRequest(BaseModel):
    """Request to pre-load a model"""
    model_path: str
    adapter_path: Optional[str] = None


@router.post("/chat", response_model=ChatResponse)
async def chat_inference(request: ChatRequest):
    """
    Run inference on a model with chat messages.
    
    Use cases:
    - Test a base model before fine-tuning
    - Test intermediate checkpoints during fine-tuning
    - Test the final fine-tuned model with LoRA adapter
    """
    try:
        # Convert to internal format
        messages = [{"role": m.role, "content": m.content} for m in request.messages]
        
        inference_request = InferenceRequest(
            model_path=request.model_path,
            messages=messages,
            adapter_path=request.adapter_path,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            do_sample=request.do_sample
        )
        
        result = await inference_service.generate(inference_request)
        
        return ChatResponse(
            success=result.success,
            response=result.response,
            tokens_generated=result.tokens_generated,
            inference_time_ms=result.inference_time_ms,
            model_loaded=result.model_loaded,
            error=result.error
        )
    
    except Exception as e:
        return ChatResponse(
            success=False,
            error=f"Inference failed: {str(e)}"
        )


@router.post("/load", response_model=dict)
async def load_model(request: LoadModelRequest):
    """
    Pre-load a model into GPU memory.
    
    This is useful to warm up the model before running inference,
    reducing the latency of the first inference request.
    """
    try:
        result = await inference_service.load_model(
            request.model_path, 
            request.adapter_path
        )
        return result
    
    except Exception as e:
        return {
            "success": False,
            "error": f"Failed to load model: {str(e)}"
        }


@router.post("/clear-memory", response_model=MemoryStatus)
async def clear_memory():
    """
    Clear the loaded model from GPU memory.
    
    Call this before:
    - Loading a different model
    - Starting a new fine-tuning job
    - When you need to free up GPU memory
    """
    try:
        result = await inference_service.clear_memory()
        return MemoryStatus(**result)
    
    except Exception as e:
        return MemoryStatus(
            success=False,
            error=f"Failed to clear memory: {str(e)}"
        )


@router.get("/status", response_model=dict)
async def get_inference_status():
    """
    Get the current inference service status.
    
    Returns information about the currently loaded model (if any)
    and GPU memory usage.
    """
    try:
        model_info = await inference_service.get_model_info()
        
        if model_info is None:
            return {
                "model_loaded": False,
                "model_path": None,
                "adapter_path": None,
                "memory_used_gb": 0.0
            }
        
        return {
            "model_loaded": True,
            "model_path": model_info.model_path,
            "adapter_path": model_info.adapter_path,
            "loaded_at": model_info.loaded_at.isoformat(),
            "memory_used_gb": model_info.memory_used_gb
        }
    
    except Exception as e:
        return {
            "model_loaded": False,
            "error": str(e)
        }


@router.get("/checkpoints/{job_id}")
async def list_checkpoints(job_id: str):
    """
    List available checkpoints for a training job.
    
    These can be used to test intermediate model states during fine-tuning.
    """
    try:
        from pathlib import Path
        from ...core.config import settings
        
        output_dir = settings.OUTPUT_DIR / job_id
        
        if not output_dir.exists():
            return {
                "success": False,
                "checkpoints": [],
                "error": "Job output directory not found"
            }
        
        checkpoints = []
        
        # Find checkpoint directories
        for item in output_dir.iterdir():
            if item.is_dir() and item.name.startswith("checkpoint-"):
                try:
                    step = int(item.name.split("-")[1])
                    checkpoints.append({
                        "name": item.name,
                        "path": str(item),
                        "step": step
                    })
                except (IndexError, ValueError):
                    continue
        
        # Sort by step
        checkpoints.sort(key=lambda x: x["step"])
        
        # Check for final model
        final_adapter = output_dir / "adapter_model.safetensors"
        if final_adapter.exists():
            checkpoints.append({
                "name": "final",
                "path": str(output_dir),
                "step": -1,
                "is_final": True
            })
        
        return {
            "success": True,
            "job_id": job_id,
            "checkpoints": checkpoints,
            "output_dir": str(output_dir)
        }
    
    except Exception as e:
        return {
            "success": False,
            "checkpoints": [],
            "error": str(e)
        }
