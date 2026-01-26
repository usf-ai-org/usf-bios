# Copyright (c) US Inc. All rights reserved.
"""
Inference service for model testing using USF BIOS package
Supports multiple backends: Transformers, SGLang, vLLM

Features:
- Multiple inference backends (transformers, sglang, vllm)
- Streaming support for real-time responses
- Multimodal support (text, image, audio, video)
- Memory management and cleanup
- LoRA adapter hot-loading
- OpenAI-compatible API format

IMPORTANT: This service uses the USF BIOS custom transformers fork,
NOT the standard HuggingFace transformers library.
"""

import asyncio
import base64
import gc
import os
import sys
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from pydantic import BaseModel, Field

from .sanitized_log_service import sanitized_log_service
from .gpu_cleanup_service import gpu_cleanup_service

# Add the project root to path for usf_bios imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Check available backends
TORCH_AVAILABLE = False
USF_BIOS_AVAILABLE = False
SGLANG_AVAILABLE = False
VLLM_AVAILABLE = False

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

try:
    import sglang as sgl
    SGLANG_AVAILABLE = True
except ImportError:
    sgl = None

try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    LLM = None
    SamplingParams = None


class InferenceBackend(str, Enum):
    """Supported inference backends"""
    TRANSFORMERS = "transformers"
    SGLANG = "sglang"
    VLLM = "vllm"


class ModelCapabilities(BaseModel):
    """Model capabilities for dynamic UI"""
    supports_text: bool = True
    supports_image: bool = False
    supports_audio: bool = False
    supports_video: bool = False
    supports_streaming: bool = True
    supports_system_prompt: bool = True
    max_context_length: int = 4096
    model_type: str = "llm"  # llm, vlm, mllm, asr, tts


class ContentItem(BaseModel):
    """Multimodal content item (OpenAI format)"""
    type: str = Field(..., description="Content type: text, image_url, audio, video")
    text: Optional[str] = None
    image_url: Optional[Dict[str, str]] = None  # {"url": "data:image/..."}
    audio: Optional[Dict[str, str]] = None  # {"data": "base64...", "format": "wav"}
    video: Optional[Dict[str, str]] = None  # {"url": "..."}


class ChatMessage(BaseModel):
    """Chat message with multimodal support (OpenAI format)"""
    role: str = Field(..., description="Role: system, user, or assistant")
    content: Union[str, List[ContentItem]] = Field(..., description="Message content")


class InferenceRequest(BaseModel):
    """Inference request model with multimodal and backend support"""
    model_path: str = Field(..., description="Path to model (HF ID or local path)")
    messages: List[Dict[str, Any]] = Field(..., description="Chat messages (OpenAI format)")
    max_new_tokens: int = Field(default=512, ge=1, le=8192)
    temperature: float = Field(default=0.7, ge=0, le=2)
    top_p: float = Field(default=0.9, ge=0, le=1)
    top_k: int = Field(default=50, ge=0)
    repetition_penalty: float = Field(default=1.0, ge=1, le=2)
    do_sample: bool = Field(default=True)
    # Backend selection
    backend: InferenceBackend = Field(default=InferenceBackend.TRANSFORMERS)
    # LoRA adapter
    adapter_path: Optional[str] = Field(default=None, description="Path to LoRA adapter")
    # Streaming
    stream: bool = Field(default=False, description="Enable streaming response")


class InferenceResponse(BaseModel):
    """Inference response model"""
    success: bool
    response: Optional[str] = None
    tokens_generated: int = 0
    inference_time_ms: int = 0
    model_loaded: str = ""
    backend_used: str = "transformers"
    error: Optional[str] = None


class StreamChunk(BaseModel):
    """Streaming response chunk (OpenAI format)"""
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Dict[str, Any]]


class ModelInfo(BaseModel):
    """Loaded model information"""
    model_path: str
    adapter_path: Optional[str] = None
    loaded_at: datetime
    memory_used_gb: float = 0.0
    backend: str = "transformers"
    capabilities: Optional[ModelCapabilities] = None
    loaded_adapters: List[str] = []


class InferenceService:
    """
    Service for model inference with memory management.
    Supports multiple backends: Transformers, SGLang, vLLM
    
    Features:
    - Multiple inference backends with automatic fallback
    - Streaming support for SGLang and vLLM
    - Multimodal support (image, audio, video inputs)
    - Hot-loading of LoRA adapters
    - Aggressive memory cleanup for training/inference switching
    """
    
    def __init__(self):
        self._engine: Optional[Any] = None
        self._vllm_engine: Optional[Any] = None
        self._sglang_engine: Optional[Any] = None
        self._current_model_path: Optional[str] = None
        self._current_adapter_path: Optional[str] = None
        self._current_backend: InferenceBackend = InferenceBackend.TRANSFORMERS
        self._lock = asyncio.Lock()
        self._loaded_at: Optional[datetime] = None
        self._model_capabilities: Optional[ModelCapabilities] = None
        self._loaded_adapters: List[str] = []
    
    def _get_gpu_memory_used(self) -> float:
        """Get GPU memory usage in GB using pynvml for accurate readings.
        
        NOTE: torch.cuda.memory_allocated() only shows PyTorch's allocation,
        not actual GPU memory usage. We use pynvml for driver-level accuracy.
        """
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            pynvml.nvmlShutdown()
            return mem_info.used / (1024 ** 3)
        except Exception:
            # Fallback to torch if pynvml fails
            try:
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    return torch.cuda.memory_allocated() / (1024 ** 3)
            except Exception:
                pass
        return 0.0
    
    def _get_total_gpu_memory(self) -> float:
        """Get total GPU memory in GB using pynvml for accuracy."""
        try:
            import pynvml
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
            pynvml.nvmlShutdown()
            return mem_info.total / (1024 ** 3)
        except Exception:
            # Fallback to torch if pynvml fails
            try:
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            except Exception:
                pass
        return 0.0
    
    def _detect_model_capabilities(self, model_path: str) -> ModelCapabilities:
        """Detect model capabilities based on model architecture"""
        model_path_lower = model_path.lower()
        
        # Vision-Language Models
        if any(x in model_path_lower for x in ['qwen2-vl', 'qwen-vl', 'llava', 'cogvlm', 'internvl', 'minicpm-v']):
            return ModelCapabilities(
                supports_text=True,
                supports_image=True,
                supports_video='video' in model_path_lower or 'qwen2-vl' in model_path_lower,
                supports_audio=False,
                supports_streaming=True,
                model_type="vlm",
                max_context_length=32768
            )
        
        # Audio/Speech Models
        if any(x in model_path_lower for x in ['whisper', 'qwen2-audio', 'seamless', 'mms']):
            return ModelCapabilities(
                supports_text=True,
                supports_image=False,
                supports_audio=True,
                supports_video=False,
                supports_streaming=True,
                model_type="asr",
                max_context_length=4096
            )
        
        # Multimodal LLMs (text + image + audio)
        if any(x in model_path_lower for x in ['qwen2.5-omni', 'gemini', 'gpt-4o']):
            return ModelCapabilities(
                supports_text=True,
                supports_image=True,
                supports_audio=True,
                supports_video=True,
                supports_streaming=True,
                model_type="mllm",
                max_context_length=128000
            )
        
        # Default: Text-only LLM
        return ModelCapabilities(
            supports_text=True,
            supports_image=False,
            supports_audio=False,
            supports_video=False,
            supports_streaming=True,
            model_type="llm",
            max_context_length=8192
        )
    
    def get_available_backends(self) -> Dict[str, bool]:
        """Get available inference backends"""
        return {
            "transformers": USF_BIOS_AVAILABLE,
            "sglang": SGLANG_AVAILABLE,
            "vllm": VLLM_AVAILABLE
        }
    
    async def get_model_info(self) -> Optional[ModelInfo]:
        """Get information about currently loaded model"""
        if self._current_model_path is None:
            return None
        
        return ModelInfo(
            model_path=self._current_model_path,
            adapter_path=self._current_adapter_path,
            loaded_at=self._loaded_at or datetime.now(),
            memory_used_gb=self._get_gpu_memory_used(),
            backend=self._current_backend.value,
            capabilities=self._model_capabilities,
            loaded_adapters=self._loaded_adapters
        )
    
    async def deep_clear_memory(self) -> Dict[str, Any]:
        """
        Aggressive memory cleanup - use before training or loading new model.
        Clears ALL GPU memory, Python garbage, and resets CUDA state.
        
        This performs:
        1. Delete all model/engine instances
        2. Clear Python references and garbage collect
        3. Empty CUDA cache on all GPUs
        4. Reset memory statistics
        5. Force synchronization
        6. Kill orphaned CUDA processes (via gpu_cleanup_service)
        """
        async with self._lock:
            try:
                memory_before = self._get_gpu_memory_used()
                total_memory = self._get_total_gpu_memory()
                
                # Step 1: Clear all inference engines
                engines_cleared = []
                
                if self._engine is not None:
                    # Clear model components first
                    if hasattr(self._engine, 'model'):
                        if hasattr(self._engine.model, 'cpu'):
                            try:
                                self._engine.model.cpu()  # Move to CPU first
                            except:
                                pass
                        del self._engine.model
                    if hasattr(self._engine, 'tokenizer'):
                        del self._engine.tokenizer
                    if hasattr(self._engine, 'processor'):
                        del self._engine.processor
                    del self._engine
                    self._engine = None
                    engines_cleared.append("transformers")
                
                if self._vllm_engine is not None:
                    if hasattr(self._vllm_engine, 'shutdown'):
                        try:
                            self._vllm_engine.shutdown()
                        except:
                            pass
                    del self._vllm_engine
                    self._vllm_engine = None
                    engines_cleared.append("vllm")
                
                if self._sglang_engine is not None:
                    if hasattr(self._sglang_engine, 'shutdown'):
                        try:
                            self._sglang_engine.shutdown()
                        except:
                            pass
                    del self._sglang_engine
                    self._sglang_engine = None
                    engines_cleared.append("sglang")
                
                # Step 2: Clear state
                self._current_model_path = None
                self._current_adapter_path = None
                self._loaded_at = None
                self._model_capabilities = None
                self._loaded_adapters = []
                
                # Step 3: Use centralized GPU cleanup service for comprehensive cleanup
                # This handles GC, CUDA cache, orphaned processes, and multi-GPU
                cleanup_result = await gpu_cleanup_service.async_deep_cleanup(
                    kill_orphans=True,
                    exclude_pids=[os.getpid()]
                )
                
                # Measure final memory
                memory_after = self._get_gpu_memory_used()
                memory_freed = memory_before - memory_after
                
                return {
                    "success": True,
                    "memory_freed_gb": round(max(0, memory_freed), 2),
                    "memory_used_gb": round(memory_after, 2),
                    "total_memory_gb": round(total_memory, 2),
                    "cleanup_type": "deep",
                    "engines_cleared": engines_cleared,
                    "cuda_cleanup": cleanup_result.get("success", False),
                    "cleanup_details": cleanup_result.get("steps", [])
                }
            
            except Exception as e:
                sanitized = sanitized_log_service.sanitize_error(str(e))
                return {
                    "success": False,
                    "memory_used_gb": round(self._get_gpu_memory_used(), 2),
                    "total_memory_gb": round(self._get_total_gpu_memory(), 2),
                    "error": sanitized['user_message']
                }
    
    async def clear_memory(self) -> Dict[str, Any]:
        """Clear model from memory and free GPU (standard cleanup)"""
        return await self.deep_clear_memory()
    
    async def load_model(
        self, 
        model_path: str, 
        adapter_path: Optional[str] = None,
        backend: InferenceBackend = InferenceBackend.TRANSFORMERS
    ) -> Dict[str, Any]:
        """
        Load a model into memory using selected backend.
        
        Backends:
        - transformers: Default, uses USF BIOS TransformersEngine
        - sglang: High-performance serving with SGLang
        - vllm: Optimized serving with vLLM
        
        SECURITY: Model is validated against allowed list before loading.
        Users cannot load unauthorized models even for inference.
        """
        if not TORCH_AVAILABLE:
            return {
                "success": False,
                "error": "PyTorch not installed. Please install PyTorch first."
            }
        
        # =====================================================================
        # CRITICAL: Validate model before loading - CANNOT be bypassed
        # This ensures only authorized models can be used for inference
        # =====================================================================
        try:
            from ..core.capabilities import get_validator
            validator = get_validator()
            
            # Determine model source
            model_source = "local" if model_path.startswith('/') else "huggingface"
            
            # Validate model path against allowed list
            is_valid, error_msg = validator.validate_for_inference(model_path, model_source)
            if not is_valid:
                return {
                    "success": False,
                    "error": f"Model not authorized: {error_msg}"
                }
        except Exception as e:
            # If validation fails, block loading for security
            return {
                "success": False,
                "error": f"Model validation failed: {str(e)}"
            }
        
        # Validate backend availability
        if backend == InferenceBackend.TRANSFORMERS and not USF_BIOS_AVAILABLE:
            return {
                "success": False,
                "error": "USF BIOS package not available. Please install the usf_bios package."
            }
        if backend == InferenceBackend.SGLANG and not SGLANG_AVAILABLE:
            return {
                "success": False,
                "error": "SGLang not available. Please install sglang package."
            }
        if backend == InferenceBackend.VLLM and not VLLM_AVAILABLE:
            return {
                "success": False,
                "error": "vLLM not available. Please install vllm package."
            }
        
        async with self._lock:
            try:
                # Check if same model is already loaded with same backend
                if (self._current_model_path == model_path and 
                    self._current_adapter_path == adapter_path and 
                    self._current_backend == backend and
                    self._engine is not None):
                    return {
                        "success": True,
                        "message": "Model already loaded",
                        "backend": backend.value,
                        "memory_used_gb": self._get_gpu_memory_used(),
                        "capabilities": self._model_capabilities.model_dump() if self._model_capabilities else None
                    }
                
                # Clear existing models from ALL backends
                await self._clear_all_engines()
                
                # Detect model capabilities
                self._model_capabilities = self._detect_model_capabilities(model_path)
                
                # Determine device and dtype
                device_map = "auto" if torch.cuda.is_available() else "cpu"
                torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
                
                # Validate adapter path if provided
                if adapter_path:
                    adapter_dir = Path(adapter_path)
                    if not adapter_dir.exists():
                        return {
                            "success": False,
                            "error": f"Adapter path does not exist: {adapter_path}"
                        }
                    adapter_config = adapter_dir / "adapter_config.json"
                    if not adapter_config.exists():
                        return {
                            "success": False,
                            "error": f"Not a valid LoRA adapter: missing adapter_config.json in {adapter_path}"
                        }
                
                # Load model using selected backend
                if backend == InferenceBackend.TRANSFORMERS:
                    adapters = [adapter_path] if adapter_path else None
                    self._engine = TransformersEngine(
                        model=model_path,
                        adapters=adapters,
                        torch_dtype=torch_dtype,
                        device_map=device_map,
                    )
                    self._loaded_adapters = [adapter_path] if adapter_path else []
                
                elif backend == InferenceBackend.VLLM:
                    # vLLM loading
                    self._vllm_engine = LLM(
                        model=model_path,
                        dtype="bfloat16" if torch.cuda.is_available() else "float32",
                        trust_remote_code=True,
                        max_model_len=self._model_capabilities.max_context_length,
                    )
                    self._engine = self._vllm_engine  # Use unified reference
                    if adapter_path:
                        self._loaded_adapters = [adapter_path]
                
                elif backend == InferenceBackend.SGLANG:
                    # SGLang loading - starts a runtime
                    self._sglang_engine = sgl.Runtime(model_path=model_path)
                    sgl.set_default_backend(self._sglang_engine)
                    self._engine = self._sglang_engine
                    if adapter_path:
                        self._loaded_adapters = [adapter_path]
                
                self._current_model_path = model_path
                self._current_adapter_path = adapter_path
                self._current_backend = backend
                self._loaded_at = datetime.now()
                
                return {
                    "success": True,
                    "model_path": model_path,
                    "adapter_path": adapter_path,
                    "backend": backend.value,
                    "memory_used_gb": self._get_gpu_memory_used(),
                    "capabilities": self._model_capabilities.model_dump() if self._model_capabilities else None
                }
            
            except Exception as e:
                # Clean up on failure
                await self._clear_all_engines()
                
                # Sanitize error for user
                sanitized = sanitized_log_service.sanitize_error(str(e))
                return {
                    "success": False,
                    "error": sanitized['user_message']
                }
    
    async def load_adapter(self, adapter_path: str) -> Dict[str, Any]:
        """
        Load a LoRA adapter onto the currently loaded model.
        Model must already be loaded.
        
        Validates adapter path contains required LoRA files before loading.
        """
        if self._engine is None or self._current_model_path is None:
            return {
                "success": False,
                "error": "No model loaded. Please load a base model first."
            }
        
        if self._current_backend != InferenceBackend.TRANSFORMERS:
            return {
                "success": False,
                "error": "Adapter loading only supported with Transformers backend"
            }
        
        # Validate adapter path exists and contains LoRA files
        adapter_dir = Path(adapter_path)
        if not adapter_dir.exists():
            return {
                "success": False,
                "error": f"Adapter path does not exist: {adapter_path}"
            }
        
        if not adapter_dir.is_dir():
            return {
                "success": False,
                "error": f"Adapter path is not a directory: {adapter_path}"
            }
        
        # Check for required LoRA adapter files
        adapter_config = adapter_dir / "adapter_config.json"
        adapter_model = adapter_dir / "adapter_model.safetensors"
        adapter_model_bin = adapter_dir / "adapter_model.bin"
        
        if not adapter_config.exists():
            return {
                "success": False,
                "error": f"Not a valid LoRA adapter: missing adapter_config.json in {adapter_path}"
            }
        
        if not (adapter_model.exists() or adapter_model_bin.exists()):
            return {
                "success": False,
                "error": f"Not a valid LoRA adapter: missing adapter weights in {adapter_path}"
            }
        
        try:
            import logging
            logging.info(f"Loading adapter from: {adapter_path} onto model: {self._current_model_path}")
            
            # Reload model with the new adapter
            result = await self.load_model(
                self._current_model_path,
                adapter_path,
                self._current_backend
            )
            
            if result.get("success"):
                logging.info(f"Adapter loaded successfully: {adapter_path}")
            else:
                logging.warning(f"Adapter load returned failure: {result.get('error')}")
            
            return result
        except Exception as e:
            import logging
            import traceback
            error_msg = str(e)
            tb = traceback.format_exc()
            logging.error(f"Adapter loading error: {error_msg}")
            logging.error(f"Traceback: {tb}")
            
            # Provide more specific error messages for common adapter issues
            if "config" in error_msg.lower() and "mismatch" in error_msg.lower():
                return {
                    "success": False,
                    "error": "Adapter configuration mismatch. The adapter was trained with a different model architecture."
                }
            elif "shape" in error_msg.lower() or "size mismatch" in error_msg.lower():
                return {
                    "success": False,
                    "error": "Adapter weight shape mismatch. The adapter is not compatible with this base model."
                }
            elif "not found" in error_msg.lower() or "no such file" in error_msg.lower():
                return {
                    "success": False,
                    "error": f"Adapter files not found at: {adapter_path}"
                }
            
            sanitized = sanitized_log_service.sanitize_error(str(e))
            return {
                "success": False,
                "error": sanitized['user_message']
            }
    
    async def switch_adapter(self, adapter_path: str) -> Dict[str, Any]:
        """
        Switch to a different LoRA adapter.
        This reloads the model with the new adapter.
        """
        return await self.load_adapter(adapter_path)
    
    async def _clear_all_engines(self):
        """Internal helper to clear all engine instances"""
        if self._engine is not None:
            if hasattr(self._engine, 'model'):
                del self._engine.model
            del self._engine
            self._engine = None
        
        if self._vllm_engine is not None:
            del self._vllm_engine
            self._vllm_engine = None
        
        if self._sglang_engine is not None:
            if hasattr(self._sglang_engine, 'shutdown'):
                self._sglang_engine.shutdown()
            del self._sglang_engine
            self._sglang_engine = None
        
        self._current_model_path = None
        self._current_adapter_path = None
        self._loaded_adapters = []
        
        gc.collect()
        if TORCH_AVAILABLE and torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    async def generate(self, request: InferenceRequest) -> InferenceResponse:
        """
        Generate response from model using selected backend.
        Supports Transformers, SGLang, and vLLM backends.
        """
        backend = request.backend if hasattr(request, 'backend') else InferenceBackend.TRANSFORMERS
        
        try:
            # Load model if needed (with selected backend)
            if (self._engine is None or 
                self._current_model_path != request.model_path or
                self._current_adapter_path != request.adapter_path or
                self._current_backend != backend):
                
                load_result = await self.load_model(
                    request.model_path, 
                    request.adapter_path,
                    backend
                )
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
            
            # Route to appropriate backend
            if self._current_backend == InferenceBackend.VLLM:
                response_text, tokens_generated = await self._generate_vllm(request)
            elif self._current_backend == InferenceBackend.SGLANG:
                response_text, tokens_generated = await self._generate_sglang(request)
            else:
                response_text, tokens_generated = await self._generate_transformers(request)
            
            end_time = datetime.now()
            inference_time_ms = int((end_time - start_time).total_seconds() * 1000)
            
            return InferenceResponse(
                success=True,
                response=response_text,
                tokens_generated=tokens_generated,
                inference_time_ms=inference_time_ms,
                model_loaded=self._current_model_path or "",
                backend_used=self._current_backend.value
            )
        
        except Exception as e:
            full_error = str(e)
            
            # Handle OOM gracefully
            if "OutOfMemoryError" in full_error or "CUDA out of memory" in full_error:
                gc.collect()
                if TORCH_AVAILABLE and torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            sanitized = sanitized_log_service.sanitize_error(full_error)
            return InferenceResponse(
                success=False,
                error=sanitized['user_message']
            )
    
    async def _generate_transformers(self, request: InferenceRequest) -> tuple:
        """Generate using Transformers/USF BIOS backend"""
        if not USF_BIOS_AVAILABLE:
            raise RuntimeError("USF BIOS not available")
        
        infer_request = InferRequest(messages=request.messages)
        request_config = RequestConfig(
            max_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
        )
        
        response = self._engine.infer(
            infer_request=infer_request,
            request_config=request_config,
        )
        
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
        
        return response_text, tokens_generated
    
    async def _generate_vllm(self, request: InferenceRequest) -> tuple:
        """Generate using vLLM backend with streaming support"""
        if not VLLM_AVAILABLE or self._vllm_engine is None:
            raise RuntimeError("vLLM not available")
        
        # Build prompt from messages
        prompt = self._messages_to_prompt(request.messages)
        
        sampling_params = SamplingParams(
            max_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
        )
        
        outputs = self._vllm_engine.generate([prompt], sampling_params)
        
        response_text = outputs[0].outputs[0].text if outputs else ""
        tokens_generated = len(outputs[0].outputs[0].token_ids) if outputs else 0
        
        return response_text, tokens_generated
    
    async def _generate_sglang(self, request: InferenceRequest) -> tuple:
        """Generate using SGLang backend with streaming support"""
        if not SGLANG_AVAILABLE:
            raise RuntimeError("SGLang not available")
        
        # Build prompt from messages
        prompt = self._messages_to_prompt(request.messages)
        
        @sgl.function
        def chat_completion(s):
            s += prompt
            s += sgl.gen(
                "response",
                max_tokens=request.max_new_tokens,
                temperature=request.temperature,
                top_p=request.top_p,
            )
        
        state = chat_completion.run()
        response_text = state["response"]
        tokens_generated = len(response_text.split())  # Approximate
        
        return response_text, tokens_generated
    
    def _messages_to_prompt(self, messages: List[Dict[str, Any]]) -> str:
        """Convert OpenAI-format messages to a single prompt string"""
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Handle multimodal content
            if isinstance(content, list):
                text_parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        text_parts.append(item.get("text", ""))
                content = " ".join(text_parts)
            
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        prompt_parts.append("Assistant:")
        return "\n".join(prompt_parts)
    
    async def generate_stream(self, request: InferenceRequest) -> AsyncGenerator[str, None]:
        """
        Stream generation - yields chunks as they're generated.
        Best supported with vLLM and SGLang backends.
        """
        backend = request.backend if hasattr(request, 'backend') else InferenceBackend.TRANSFORMERS
        
        # Load model if needed
        if (self._engine is None or 
            self._current_model_path != request.model_path or
            self._current_backend != backend):
            
            load_result = await self.load_model(request.model_path, request.adapter_path, backend)
            if not load_result["success"]:
                yield f"Error: {load_result.get('error', 'Failed to load model')}"
                return
        
        try:
            if self._current_backend == InferenceBackend.VLLM and VLLM_AVAILABLE:
                # vLLM streaming
                prompt = self._messages_to_prompt(request.messages)
                sampling_params = SamplingParams(
                    max_tokens=request.max_new_tokens,
                    temperature=request.temperature,
                    top_p=request.top_p,
                )
                
                async for output in self._vllm_engine.generate([prompt], sampling_params, stream=True):
                    if output.outputs:
                        yield output.outputs[0].text
            
            elif self._current_backend == InferenceBackend.SGLANG and SGLANG_AVAILABLE:
                # SGLang streaming
                prompt = self._messages_to_prompt(request.messages)
                
                async for chunk in sgl.gen_stream(prompt, max_tokens=request.max_new_tokens):
                    yield chunk
            
            else:
                # Transformers - non-streaming fallback
                response_text, _ = await self._generate_transformers(request)
                yield response_text
        
        except Exception as e:
            sanitized = sanitized_log_service.sanitize_error(str(e))
            yield f"Error: {sanitized['user_message']}"


# Global instance
inference_service = InferenceService()
