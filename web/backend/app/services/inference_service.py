# Copyright (c) US Inc. All rights reserved.
"""
Inference service for model testing using USF BIOS package
- Test models before fine-tuning
- Test intermediate checkpoints during fine-tuning
- Test final fine-tuned models

IMPORTANT: This service uses the USF BIOS custom transformers fork,
NOT the standard HuggingFace transformers library.
"""

import asyncio
import gc
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from .sanitized_log_service import sanitized_log_service

# Add the project root to path for usf_bios imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Check if USF BIOS and torch are available
USF_BIOS_AVAILABLE = False
TORCH_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None

try:
    from usf_bios.model import get_model_processor
    from usf_bios.infer_engine import TransformersEngine
    from usf_bios.infer_engine.protocol import InferRequest, RequestConfig
    USF_BIOS_AVAILABLE = True
except ImportError:
    get_model_processor = None
    TransformersEngine = None
    InferRequest = None
    RequestConfig = None


class InferenceRequest(BaseModel):
    """Inference request model"""
    model_path: str = Field(..., description="Path to model (HF ID or local path)")
    messages: List[Dict[str, str]] = Field(..., description="Chat messages")
    max_new_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.9, ge=0, le=1)
    top_k: int = Field(default=50, ge=0)
    repetition_penalty: float = Field(default=1.0, ge=1, le=2)
    do_sample: bool = Field(default=True)
    # LoRA adapter
    adapter_path: Optional[str] = Field(default=None, description="Path to LoRA adapter")


class InferenceResponse(BaseModel):
    """Inference response model"""
    success: bool
    response: Optional[str] = None
    tokens_generated: int = 0
    inference_time_ms: int = 0
    model_loaded: str = ""
    error: Optional[str] = None


class ModelInfo(BaseModel):
    """Loaded model information"""
    model_path: str
    adapter_path: Optional[str] = None
    loaded_at: datetime
    memory_used_gb: float = 0.0


class InferenceService:
    """
    Service for model inference with memory management.
    Uses USF BIOS custom transformers fork for model loading and inference.
    """
    
    def __init__(self):
        self._engine: Optional[Any] = None
        self._current_model_path: Optional[str] = None
        self._current_adapter_path: Optional[str] = None
        self._lock = asyncio.Lock()
        self._loaded_at: Optional[datetime] = None
    
    def _get_gpu_memory_used(self) -> float:
        """Get GPU memory usage in GB"""
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 ** 3)
        except Exception:
            pass
        return 0.0
    
    async def get_model_info(self) -> Optional[ModelInfo]:
        """Get information about currently loaded model"""
        if self._current_model_path is None:
            return None
        
        return ModelInfo(
            model_path=self._current_model_path,
            adapter_path=self._current_adapter_path,
            loaded_at=self._loaded_at or datetime.now(),
            memory_used_gb=self._get_gpu_memory_used()
        )
    
    async def clear_memory(self) -> Dict[str, Any]:
        """Clear model from memory and free GPU"""
        async with self._lock:
            try:
                memory_before = self._get_gpu_memory_used()
                
                # Delete engine (contains model and processor)
                if self._engine is not None:
                    del self._engine
                    self._engine = None
                
                self._current_model_path = None
                self._current_adapter_path = None
                self._loaded_at = None
                
                # Force garbage collection
                gc.collect()
                
                # Clear CUDA cache
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                memory_after = self._get_gpu_memory_used()
                
                return {
                    "success": True,
                    "memory_freed_gb": round(memory_before - memory_after, 2),
                    "memory_used_gb": round(memory_after, 2)
                }
            
            except Exception as e:
                # Sanitize error for user
                sanitized = sanitized_log_service.sanitize_error(str(e))
                return {
                    "success": False,
                    "error": sanitized['user_message']
                }
    
    async def load_model(self, model_path: str, adapter_path: Optional[str] = None) -> Dict[str, Any]:
        """Load a model into memory using USF BIOS TransformersEngine"""
        if not USF_BIOS_AVAILABLE:
            return {
                "success": False,
                "error": "USF BIOS package not available. Please install the usf_bios package from the project root."
            }
        
        if not TORCH_AVAILABLE:
            return {
                "success": False,
                "error": "PyTorch not installed. Please install PyTorch first."
            }
        
        async with self._lock:
            try:
                # Check if same model is already loaded
                if (self._current_model_path == model_path and 
                    self._current_adapter_path == adapter_path and 
                    self._engine is not None):
                    return {
                        "success": True,
                        "message": "Model already loaded",
                        "memory_used_gb": self._get_gpu_memory_used()
                    }
                
                # Clear existing model first
                if self._engine is not None:
                    del self._engine
                    self._engine = None
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Determine device and dtype
                device_map = "auto" if torch.cuda.is_available() else "cpu"
                torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
                
                # Load model using USF BIOS TransformersEngine
                adapters = [adapter_path] if adapter_path else None
                
                self._engine = TransformersEngine(
                    model=model_path,
                    adapters=adapters,
                    torch_dtype=torch_dtype,
                    device_map=device_map,
                )
                
                self._current_model_path = model_path
                self._current_adapter_path = adapter_path
                self._loaded_at = datetime.now()
                
                return {
                    "success": True,
                    "model_path": model_path,
                    "adapter_path": adapter_path,
                    "memory_used_gb": self._get_gpu_memory_used()
                }
            
            except Exception as e:
                # Clean up on failure
                self._engine = None
                self._current_model_path = None
                self._current_adapter_path = None
                gc.collect()
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Sanitize error for user
                sanitized = sanitized_log_service.sanitize_error(str(e))
                return {
                    "success": False,
                    "error": sanitized['user_message']
                }
    
    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Generate response from model using USF BIOS inference engine"""
        if not USF_BIOS_AVAILABLE:
            return InferenceResponse(
                success=False,
                error="USF BIOS package not available. Please install the usf_bios package."
            )
        
        try:
            # Load model if needed
            if (self._engine is None or 
                self._current_model_path != request.model_path or
                self._current_adapter_path != request.adapter_path):
                
                load_result = await self.load_model(request.model_path, request.adapter_path)
                if not load_result["success"]:
                    return InferenceResponse(
                        success=False,
                        error=load_result.get("error", "Failed to load model")
                    )
            
            if self._engine is None:
                return InferenceResponse(
                    success=False,
                    error="Model not loaded"
                )
            
            start_time = datetime.now()
            
            # Create inference request using USF BIOS protocol
            infer_request = InferRequest(messages=request.messages)
            request_config = RequestConfig(
                max_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                repetition_penalty=request.repetition_penalty,
            )
            
            # Run inference using USF BIOS engine
            response = self._engine.infer(
                infer_request=infer_request,
                request_config=request_config,
            )
            
            end_time = datetime.now()
            inference_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            # Extract response text
            response_text = ""
            tokens_generated = 0
            
            if hasattr(response, 'choices') and response.choices:
                choice = response.choices[0]
                if hasattr(choice, 'message') and choice.message:
                    response_text = choice.message.content or ""
                if hasattr(response, 'usage') and response.usage:
                    tokens_generated = response.usage.completion_tokens or 0
            elif isinstance(response, str):
                response_text = response
            
            return InferenceResponse(
                success=True,
                response=response_text,
                tokens_generated=tokens_generated,
                inference_time_ms=inference_time_ms,
                model_loaded=self._current_model_path or ""
            )
        
        except Exception as e:
            full_error = str(e)
            
            # Handle OOM gracefully
            if "OutOfMemoryError" in full_error or "CUDA out of memory" in full_error:
                gc.collect()
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            # Sanitize error for user
            sanitized = sanitized_log_service.sanitize_error(full_error)
            
            return InferenceResponse(
                success=False,
                error=sanitized['user_message']
            )


# Global instance
inference_service = InferenceService()
