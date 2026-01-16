# Copyright (c) US Inc. All rights reserved.
"""
Inference service for model testing
- Test models before fine-tuning
- Test intermediate checkpoints during fine-tuning
- Test final fine-tuned models
"""

import asyncio
import gc
import os
import sys
import torch
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, Field


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
    """Service for model inference with memory management"""
    
    def __init__(self):
        self._model = None
        self._tokenizer = None
        self._current_model_path: Optional[str] = None
        self._current_adapter_path: Optional[str] = None
        self._lock = asyncio.Lock()
        self._loaded_at: Optional[datetime] = None
    
    def _get_gpu_memory_used(self) -> float:
        """Get GPU memory usage in GB"""
        try:
            if torch.cuda.is_available():
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
                
                # Delete model and tokenizer
                if self._model is not None:
                    del self._model
                    self._model = None
                
                if self._tokenizer is not None:
                    del self._tokenizer
                    self._tokenizer = None
                
                self._current_model_path = None
                self._current_adapter_path = None
                self._loaded_at = None
                
                # Force garbage collection
                gc.collect()
                
                # Clear CUDA cache
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                memory_after = self._get_gpu_memory_used()
                
                return {
                    "success": True,
                    "memory_freed_gb": round(memory_before - memory_after, 2),
                    "memory_used_gb": round(memory_after, 2)
                }
            
            except Exception as e:
                return {
                    "success": False,
                    "error": str(e)
                }
    
    async def load_model(self, model_path: str, adapter_path: Optional[str] = None) -> Dict[str, Any]:
        """Load a model into memory"""
        async with self._lock:
            try:
                # Check if same model is already loaded
                if (self._current_model_path == model_path and 
                    self._current_adapter_path == adapter_path and 
                    self._model is not None):
                    return {
                        "success": True,
                        "message": "Model already loaded",
                        "memory_used_gb": self._get_gpu_memory_used()
                    }
                
                # Clear existing model first
                if self._model is not None:
                    await self.clear_memory()
                
                # Import here to avoid loading torch at startup
                from transformers import AutoModelForCausalLM, AutoTokenizer
                
                # Load tokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    model_path,
                    trust_remote_code=True
                )
                
                # Determine device and dtype
                device_map = "auto" if torch.cuda.is_available() else "cpu"
                torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
                
                # Load model
                self._model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    device_map=device_map,
                    torch_dtype=torch_dtype,
                    trust_remote_code=True
                )
                
                # Load LoRA adapter if specified
                if adapter_path:
                    try:
                        from peft import PeftModel
                        self._model = PeftModel.from_pretrained(self._model, adapter_path)
                        self._current_adapter_path = adapter_path
                    except Exception as e:
                        return {
                            "success": False,
                            "error": f"Failed to load adapter: {str(e)}"
                        }
                
                self._current_model_path = model_path
                self._loaded_at = datetime.now()
                
                return {
                    "success": True,
                    "model_path": model_path,
                    "adapter_path": adapter_path,
                    "memory_used_gb": self._get_gpu_memory_used()
                }
            
            except Exception as e:
                # Clean up on failure
                self._model = None
                self._tokenizer = None
                self._current_model_path = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                return {
                    "success": False,
                    "error": str(e)
                }
    
    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """Generate response from model"""
        try:
            # Load model if needed
            if (self._model is None or 
                self._current_model_path != request.model_path or
                self._current_adapter_path != request.adapter_path):
                
                load_result = await self.load_model(request.model_path, request.adapter_path)
                if not load_result["success"]:
                    return InferenceResponse(
                        success=False,
                        error=load_result.get("error", "Failed to load model")
                    )
            
            if self._model is None or self._tokenizer is None:
                return InferenceResponse(
                    success=False,
                    error="Model not loaded"
                )
            
            start_time = datetime.now()
            
            # Format messages for chat
            try:
                # Try chat template first
                if hasattr(self._tokenizer, 'apply_chat_template'):
                    input_text = self._tokenizer.apply_chat_template(
                        request.messages,
                        tokenize=False,
                        add_generation_prompt=True
                    )
                else:
                    # Fallback to simple formatting
                    input_text = ""
                    for msg in request.messages:
                        role = msg.get("role", "user")
                        content = msg.get("content", "")
                        if role == "system":
                            input_text += f"System: {content}\n\n"
                        elif role == "user":
                            input_text += f"User: {content}\n\n"
                        elif role == "assistant":
                            input_text += f"Assistant: {content}\n\n"
                    input_text += "Assistant: "
            except Exception as e:
                return InferenceResponse(
                    success=False,
                    error=f"Failed to format messages: {str(e)}"
                )
            
            # Tokenize
            inputs = self._tokenizer(input_text, return_tensors="pt")
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            input_length = inputs["input_ids"].shape[1]
            
            # Generate
            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=request.max_new_tokens,
                    temperature=request.temperature if request.do_sample else 1.0,
                    top_p=request.top_p if request.do_sample else 1.0,
                    top_k=request.top_k if request.do_sample else 0,
                    repetition_penalty=request.repetition_penalty,
                    do_sample=request.do_sample,
                    pad_token_id=self._tokenizer.pad_token_id or self._tokenizer.eos_token_id,
                )
            
            # Decode output
            generated_tokens = outputs[0][input_length:]
            response_text = self._tokenizer.decode(generated_tokens, skip_special_tokens=True)
            
            end_time = datetime.now()
            inference_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            return InferenceResponse(
                success=True,
                response=response_text,
                tokens_generated=len(generated_tokens),
                inference_time_ms=inference_time_ms,
                model_loaded=self._current_model_path or ""
            )
        
        except torch.cuda.OutOfMemoryError:
            # Handle OOM gracefully
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return InferenceResponse(
                success=False,
                error="GPU out of memory. Try reducing max_new_tokens or clearing memory first."
            )
        
        except Exception as e:
            return InferenceResponse(
                success=False,
                error=str(e)
            )


# Global instance
inference_service = InferenceService()
