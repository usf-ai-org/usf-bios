# Copyright (c) US Inc. All rights reserved.
"""
Inference endpoints for model testing
Supports multiple backends: Transformers, SGLang, vLLM

Features:
- Multiple inference backends with streaming
- Multimodal support (text, image, audio, video)
- OpenAI-compatible API format
- Memory management for training/inference switching
"""

from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ...services.inference_service import (
    inference_service, 
    InferenceRequest, 
    InferenceResponse,
    InferenceBackend,
    ModelInfo,
    ModelCapabilities
)
from ...services.sanitized_log_service import sanitized_log_service

router = APIRouter()


class ContentItem(BaseModel):
    """Multimodal content item (OpenAI format)"""
    type: str = Field(..., description="Content type: text, image_url, audio, video")
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None
    audio: Optional[Dict[str, str]] = None
    video: Optional[Dict[str, str]] = None


class ChatMessage(BaseModel):
    """Chat message with multimodal support (OpenAI format)"""
    role: str = Field(..., description="Role: system, user, or assistant")
    content: Union[str, List[ContentItem]] = Field(..., description="Message content (text or multimodal)")


class ChatRequest(BaseModel):
    """Chat inference request with backend selection"""
    model_path: str = Field(..., description="Model path (local path or HuggingFace ID)")
    messages: List[ChatMessage] = Field(..., description="Chat messages (OpenAI format)")
    adapter_path: Optional[str] = Field(default=None, description="LoRA adapter path")
    # Backend selection
    backend: str = Field(default="transformers", description="Backend: transformers, sglang, vllm")
    # Generation parameters
    max_new_tokens: int = Field(default=512, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.9, ge=0, le=1)
    top_k: int = Field(default=50, ge=0)
    repetition_penalty: float = Field(default=1.0, ge=1, le=2)
    do_sample: bool = Field(default=True)
    # Streaming
    stream: bool = Field(default=False, description="Enable streaming response")


class ChatResponse(BaseModel):
    """
    Chat inference response with multimodal output support.
    
    Supports different output types based on model:
    - LLM/VLM: text response
    - T2I: images list
    - TTS: audio
    - T2V: video
    - MLLM: any combination
    """
    success: bool
    response: Optional[str] = None
    # Multimodal outputs
    images: Optional[List[Dict[str, str]]] = None  # [{"data": "base64...", "format": "png"}]
    audio: Optional[Dict[str, str]] = None  # {"data": "base64...", "format": "wav"}
    video: Optional[Dict[str, str]] = None  # {"data": "base64...", "format": "mp4"}
    # Metadata
    tokens_generated: int = 0
    inference_time_ms: int = 0
    model_loaded: str = ""
    backend_used: str = "transformers"
    output_type: str = "text"  # text, image, audio, video, multimodal
    error: Optional[str] = None


class MemoryStatus(BaseModel):
    """GPU memory status"""
    success: bool
    memory_freed_gb: float = 0.0
    memory_used_gb: float = 0.0
    total_memory_gb: float = 0.0
    cleanup_type: str = "standard"
    error: Optional[str] = None


class LoadModelRequest(BaseModel):
    """Request to pre-load a model"""
    model_path: str
    adapter_path: Optional[str] = None
    backend: str = Field(default="transformers", description="Backend: transformers, sglang, vllm")


@router.post("/chat")
async def chat_inference(request: ChatRequest):
    """
    Run inference on a model with chat messages.
    
    Supports:
    - Multiple backends: transformers (default), sglang, vllm
    - Streaming responses (for sglang/vllm)
    - Multimodal inputs (text, image, audio, video)
    - OpenAI-compatible message format
    
    BLOCKED during training to prevent GPU memory conflicts.
    """
    # CRITICAL: Check if training is active - block inference during training
    from ...services.training_status_service import training_status_service
    can_load, reason = await training_status_service.can_load_inference()
    if not can_load:
        return ChatResponse(
            success=False,
            error=f"Inference blocked: {reason}"
        )
    
    try:
        # Convert backend string to enum
        backend_map = {
            "transformers": InferenceBackend.TRANSFORMERS,
            "sglang": InferenceBackend.SGLANG,
            "vllm": InferenceBackend.VLLM
        }
        backend = backend_map.get(request.backend.lower(), InferenceBackend.TRANSFORMERS)
        
        # Convert messages to internal format (handles multimodal content)
        messages = []
        for m in request.messages:
            if isinstance(m.content, str):
                messages.append({"role": m.role, "content": m.content})
            else:
                # Handle multimodal content (list of ContentItems)
                content_list = []
                for item in m.content:
                    content_item = {"type": item.type}
                    if item.text:
                        content_item["text"] = item.text
                    if item.image_url:
                        content_item["image_url"] = item.image_url
                    if item.audio:
                        content_item["audio"] = item.audio
                    if item.video:
                        content_item["video"] = item.video
                    content_list.append(content_item)
                messages.append({"role": m.role, "content": content_list})
        
        inference_request = InferenceRequest(
            model_path=request.model_path,
            messages=messages,
            adapter_path=request.adapter_path,
            backend=backend,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
            do_sample=request.do_sample,
            stream=request.stream
        )
        
        # Handle streaming response
        if request.stream:
            async def generate_stream():
                async for chunk in inference_service.generate_stream(inference_request):
                    # SSE format
                    yield f"data: {chunk}\n\n"
                yield "data: [DONE]\n\n"
            
            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                }
            )
        
        # Non-streaming response
        result = await inference_service.generate(inference_request)
        
        return ChatResponse(
            success=result.success,
            response=result.response,
            # Include multimodal outputs
            images=result.images,
            audio=result.audio,
            video=result.video,
            output_type=result.output_type,
            # Metadata
            tokens_generated=result.tokens_generated,
            inference_time_ms=result.inference_time_ms,
            model_loaded=result.model_loaded,
            backend_used=result.backend_used,
            error=result.error
        )
    
    except Exception as e:
        sanitized = sanitized_log_service.sanitize_error(str(e))
        return ChatResponse(
            success=False,
            error=sanitized['user_message']
        )


@router.post("/load", response_model=dict)
async def load_model(request: LoadModelRequest):
    """
    Pre-load a model into GPU memory with selected backend.
    
    Backends:
    - transformers: Default, uses USF BIOS TransformersEngine
    - sglang: High-performance serving with SGLang
    - vllm: Optimized serving with vLLM
    
    BLOCKED during training to prevent GPU memory conflicts.
    """
    # CRITICAL: Check if training is active - block inference loading during training
    from ...services.training_status_service import training_status_service
    can_load, reason = await training_status_service.can_load_inference()
    if not can_load:
        return {
            "success": False,
            "error": reason,
            "blocked_by_training": True
        }
    
    try:
        backend_map = {
            "transformers": InferenceBackend.TRANSFORMERS,
            "sglang": InferenceBackend.SGLANG,
            "vllm": InferenceBackend.VLLM
        }
        backend = backend_map.get(request.backend.lower(), InferenceBackend.TRANSFORMERS)
        
        result = await inference_service.load_model(
            request.model_path, 
            request.adapter_path,
            backend
        )
        return result
    
    except Exception as e:
        sanitized = sanitized_log_service.sanitize_error(str(e))
        return {
            "success": False,
            "error": sanitized['user_message']
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
        sanitized = sanitized_log_service.sanitize_error(str(e))
        return MemoryStatus(
            success=False,
            error=sanitized['user_message']
        )


@router.get("/status", response_model=dict)
async def get_inference_status():
    """
    Get the current inference service status.
    
    Returns:
    - Currently loaded model info
    - Backend being used
    - Model capabilities (multimodal support)
    - GPU memory usage
    - Available backends
    """
    try:
        model_info = await inference_service.get_model_info()
        available_backends = inference_service.get_available_backends()
        
        if model_info is None:
            return {
                "model_loaded": False,
                "model_path": None,
                "adapter_path": None,
                "backend": None,
                "capabilities": None,
                "memory_used_gb": 0.0,
                "available_backends": available_backends
            }
        
        return {
            "model_loaded": True,
            "model_path": model_info.model_path,
            "adapter_path": model_info.adapter_path,
            "backend": model_info.backend,
            "loaded_at": model_info.loaded_at.isoformat(),
            "memory_used_gb": model_info.memory_used_gb,
            "capabilities": model_info.capabilities.model_dump() if model_info.capabilities else None,
            "loaded_adapters": model_info.loaded_adapters,
            "available_backends": available_backends
        }
    
    except Exception as e:
        return {
            "model_loaded": False,
            "error": "Failed to get model status",
            "available_backends": {"transformers": False, "sglang": False, "vllm": False}
        }


@router.get("/backends")
async def get_available_backends():
    """
    Get available inference backends and their status.
    
    Returns which backends are installed and available:
    - transformers: USF BIOS TransformersEngine
    - sglang: SGLang high-performance serving
    - vllm: vLLM optimized serving
    """
    backends = inference_service.get_available_backends()
    return {
        "backends": backends,
        "default": "transformers",
        "recommended_for_streaming": "vllm" if backends.get("vllm") else "sglang" if backends.get("sglang") else "transformers"
    }


class LoadAdapterRequest(BaseModel):
    """Request to load a LoRA adapter"""
    adapter_path: str


@router.post("/load-adapter", response_model=dict)
async def load_adapter(request: LoadAdapterRequest):
    """
    Load a LoRA adapter onto the currently loaded model.
    
    The base model must already be loaded. This endpoint allows
    dynamically adding adapters after training (e.g., for LoRA fine-tuning).
    """
    try:
        result = await inference_service.load_adapter(request.adapter_path)
        return result
    except Exception as e:
        sanitized = sanitized_log_service.sanitize_error(str(e))
        return {
            "success": False,
            "error": sanitized['user_message']
        }


@router.post("/switch-adapter", response_model=dict)
async def switch_adapter(request: LoadAdapterRequest):
    """
    Switch to a different LoRA adapter.
    
    This reloads the model with the specified adapter.
    Useful when you have multiple fine-tuned adapters and want to switch between them.
    """
    try:
        result = await inference_service.switch_adapter(request.adapter_path)
        return result
    except Exception as e:
        sanitized = sanitized_log_service.sanitize_error(str(e))
        return {
            "success": False,
            "error": sanitized['user_message']
        }


@router.post("/deep-clear-memory", response_model=MemoryStatus)
async def deep_clear_memory():
    """
    Aggressive memory cleanup - use before training.
    
    This performs a thorough cleanup including:
    - Clearing all model instances
    - Multiple garbage collection passes
    - CUDA cache clearing on all GPUs
    - Memory stats reset
    
    Use this before starting a new training job to ensure
    maximum GPU memory is available.
    """
    try:
        result = await inference_service.deep_clear_memory()
        return MemoryStatus(**result)
    
    except Exception as e:
        sanitized = sanitized_log_service.sanitize_error(str(e))
        return MemoryStatus(
            success=False,
            error=sanitized['user_message']
        )


@router.get("/checkpoints/{job_id}")
async def list_checkpoints(job_id: str):
    """
    List available checkpoints for a training job.
    
    USF BIOS writes output in a nested structure:
      output_dir/{job_id}/v0-YYYYMMDD-HHMMSS/checkpoint-N/
    
    This endpoint handles both flat and nested checkpoint layouts.
    It also detects whether checkpoints contain LoRA adapters or full models.
    """
    try:
        from pathlib import Path
        from ...core.capabilities import get_system_settings
        
        output_dir = get_system_settings().OUTPUT_DIR / job_id
        
        if not output_dir.exists():
            return {
                "success": False,
                "checkpoints": [],
                "error": "Job output directory not found"
            }
        
        checkpoints = []
        seen_steps = set()
        
        # Recursively find all checkpoint-* directories at any depth
        for ckpt_dir in sorted(output_dir.glob("**/checkpoint-*")):
            if not ckpt_dir.is_dir():
                continue
            try:
                step = int(ckpt_dir.name.split("-")[1])
            except (IndexError, ValueError):
                continue
            
            if step in seen_steps:
                continue
            seen_steps.add(step)
            
            # Detect checkpoint type: LoRA adapter or full model
            has_adapter = (ckpt_dir / "adapter_model.safetensors").exists() or (ckpt_dir / "adapter_model.bin").exists()
            has_adapter_config = (ckpt_dir / "adapter_config.json").exists()
            has_full_model = (ckpt_dir / "model.safetensors").exists() or (ckpt_dir / "pytorch_model.bin").exists()
            has_model_index = (ckpt_dir / "model.safetensors.index.json").exists()
            
            ckpt_type = "unknown"
            if has_adapter or has_adapter_config:
                ckpt_type = "lora"
            elif has_full_model or has_model_index:
                ckpt_type = "full"
            
            checkpoints.append({
                "name": ckpt_dir.name,
                "path": str(ckpt_dir),
                "step": step,
                "type": ckpt_type,
                "is_final": False,
            })
        
        # Sort by step number
        checkpoints.sort(key=lambda x: x["step"])
        
        # Mark the highest checkpoint as "final" if no separate final model exists
        # Also search for a standalone final adapter/model outside checkpoint dirs
        final_adapters = list(output_dir.glob("**/adapter_model.safetensors"))
        final_models = list(output_dir.glob("**/model.safetensors"))
        
        # Filter to only adapters NOT inside checkpoint directories
        non_ckpt_adapters = [a for a in final_adapters if "checkpoint-" not in str(a)]
        non_ckpt_models = [m for m in final_models if "checkpoint-" not in str(m)]
        
        if non_ckpt_adapters:
            checkpoints.append({
                "name": "final",
                "path": str(non_ckpt_adapters[0].parent),
                "step": -1,
                "type": "lora",
                "is_final": True,
            })
        elif non_ckpt_models:
            checkpoints.append({
                "name": "final",
                "path": str(non_ckpt_models[0].parent),
                "step": -1,
                "type": "full",
                "is_final": True,
            })
        elif checkpoints:
            # Mark the last checkpoint as the final one
            checkpoints[-1]["is_final"] = True
        
        # Determine the best adapter/model path for quick loading
        best_path = None
        if checkpoints:
            final_ckpts = [c for c in checkpoints if c.get("is_final")]
            best_path = final_ckpts[0]["path"] if final_ckpts else checkpoints[-1]["path"]
        
        return {
            "success": True,
            "job_id": job_id,
            "checkpoints": checkpoints,
            "output_dir": str(output_dir),
            "best_adapter_path": best_path,
        }
    
    except Exception as e:
        return {
            "success": False,
            "checkpoints": [],
            "error": "Failed to list checkpoints"
        }
