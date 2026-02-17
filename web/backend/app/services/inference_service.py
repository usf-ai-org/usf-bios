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
import json
import logging
import os
import re
import sys
import traceback
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional, Union

from pydantic import BaseModel, Field

# Optional imports
try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    pynvml = None
    PYNVML_AVAILABLE = False

from .sanitized_log_service import sanitized_log_service
from .gpu_cleanup_service import gpu_cleanup_service
from .system_encrypted_log_service import system_encrypted_log

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
    """
    Model capabilities for dynamic UI.
    
    Defines what inputs and outputs a model supports, and which backends can run it.
    
    Model Types:
    - llm: Text-to-text (LLaMA, Qwen, Mistral, etc.)
    - vlm: Vision-Language (Qwen2-VL, LLaVA, etc.) - image+text input, text output
    - mllm: Multimodal LLM (Qwen2.5-Omni) - any input, any output
    - asr: Audio-to-text (Whisper, Qwen2-Audio)
    - tts: Text-to-audio (SpeechT5, XTTS, Bark)
    - t2i: Text-to-image (Stable Diffusion, SDXL, Flux)
    - i2i: Image-to-image (Stable Diffusion img2img)
    - t2v: Text-to-video (CogVideoX, etc.)
    """
    # Input capabilities
    supports_text_input: bool = True
    supports_image_input: bool = False
    supports_audio_input: bool = False
    supports_video_input: bool = False
    
    # Output capabilities  
    supports_text_output: bool = True
    supports_image_output: bool = False
    supports_audio_output: bool = False
    supports_video_output: bool = False
    
    # Backend compatibility
    supported_backends: List[str] = ["transformers"]  # Which backends can run this model
    
    # Other capabilities
    supports_streaming: bool = True
    supports_system_prompt: bool = True
    supports_tool_calls: bool = False
    max_context_length: int = 4096
    
    # Model classification
    model_type: str = "llm"  # llm, vlm, mllm, asr, tts, t2i, i2i, t2v
    
    # Legacy compatibility aliases
    @property
    def supports_image(self) -> bool:
        return self.supports_image_input
    
    @property
    def supports_audio(self) -> bool:
        return self.supports_audio_input
    
    @property
    def supports_video(self) -> bool:
        return self.supports_video_input


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
    """
    Inference response model with multimodal output support.
    
    Supports:
    - Text output (response field)
    - Image output (images field - list of base64 encoded images)
    - Audio output (audio field - base64 encoded audio with format)
    - Video output (video field - base64 encoded video or URL)
    """
    success: bool
    response: Optional[str] = None
    # Multimodal outputs
    images: Optional[List[Dict[str, str]]] = None  # [{"data": "base64...", "format": "png"}]
    audio: Optional[Dict[str, str]] = None  # {"data": "base64...", "format": "wav"}
    video: Optional[Dict[str, str]] = None  # {"data": "base64...", "format": "mp4"} or {"url": "..."}
    # Metadata
    tokens_generated: int = 0
    inference_time_ms: int = 0
    model_loaded: str = ""
    backend_used: str = "transformers"
    output_type: str = "text"  # text, image, audio, video, multimodal
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
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                pynvml.nvmlShutdown()
                return mem_info.used / (1024 ** 3)
            except Exception:
                pass
        # Fallback to torch if pynvml fails
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return torch.cuda.memory_allocated() / (1024 ** 3)
        except Exception:
            pass
        return 0.0
    
    def _get_total_gpu_memory(self) -> float:
        """Get total GPU memory in GB using pynvml for accuracy."""
        if PYNVML_AVAILABLE:
            try:
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                pynvml.nvmlShutdown()
                return mem_info.total / (1024 ** 3)
            except Exception:
                pass
        # Fallback to torch if pynvml fails
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                return torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        except Exception:
            pass
        return 0.0
    
    def _resolve_adapter_path(self, adapter_path: str) -> Optional[str]:
        """
        Resolve adapter path from training output directory.
        
        USF BIOS saves adapters in various locations within the output directory:
        - Directly in the output directory (adapter_config.json at root)
        - In checkpoint subdirectories (checkpoint-*/adapter_config.json)
        
        Returns the resolved path or None if no valid adapter found.
        """
        adapter_dir = Path(adapter_path)
        
        # Check if path exists
        if not adapter_dir.exists():
            logging.warning(f"Adapter path does not exist: {adapter_path}")
            return None
        
        # Check if adapter_config.json exists directly
        if (adapter_dir / "adapter_config.json").exists():
            logging.info(f"Found adapter at: {adapter_path}")
            return adapter_path
        
        # Search recursively for adapter in subdirectories (handles nested v0-*/checkpoint-*/...)
        # Priority: checkpoint-final > highest checkpoint number > any checkpoint
        checkpoint_dirs = []
        for adapter_config in adapter_dir.glob("**/adapter_config.json"):
            checkpoint_dirs.append(adapter_config.parent)
        
        if not checkpoint_dirs:
            logging.warning(f"No adapter_config.json found in {adapter_path} or its subdirectories")
            return None
        
        # Sort to find best checkpoint (prefer 'checkpoint-final' or highest number)
        def checkpoint_priority(p: Path) -> tuple:
            name = p.name.lower()
            if 'final' in name:
                return (0, 0)  # Highest priority
            # Extract number if present
            match = re.search(r'(\d+)', name)
            if match:
                return (1, -int(match.group(1)))  # Higher numbers first
            return (2, 0)  # Default
        
        checkpoint_dirs.sort(key=checkpoint_priority)
        best_checkpoint = checkpoint_dirs[0]
        logging.info(f"Found adapter in checkpoint: {best_checkpoint}")
        return str(best_checkpoint)
    
    def _detect_model_capabilities(self, model_path: str) -> ModelCapabilities:
        """
        Detect model capabilities based on model architecture and config.json.
        
        Returns detailed input/output capabilities and supported backends.
        
        Model Types Detected:
        - llm: Text-to-text (all backends)
        - vlm: Image+text to text (transformers, some vllm)
        - mllm: Any modality (transformers only)
        - asr: Audio to text (transformers only)
        - tts: Text to audio (transformers only)
        - t2i: Text to image (transformers only - diffusers)
        - i2i: Image to image (transformers only - diffusers)
        - t2v: Text to video (transformers only)
        """
        model_path_lower = model_path.lower()
        
        # Pattern definitions for each model type
        vlm_patterns = ['qwen2-vl', 'qwen-vl', 'qwen2_vl', 'llava', 'cogvlm', 'internvl', 
                        'minicpm-v', 'minicpmv', 'phi3v', 'phi-3-vision', 'idefics', 
                        'paligemma', 'pixtral', 'molmo', 'mllama', 'llama3.2-vision',
                        'florence', 'blip', 'git-base', 'kosmos']
        
        asr_patterns = ['whisper', 'qwen2-audio', 'qwen2_audio', 'seamless', 'mms', 
                        'wav2vec', 'hubert', 'speech2text', 'canary']
        
        tts_patterns = ['speecht5', 'xtts', 'bark', 'vall-e', 'voicecraft', 
                        'parler', 'tortoise', 'coqui', 'styletts']
        
        mllm_patterns = ['qwen2.5-omni', 'qwen2_5_omni', 'gemini', 'gpt-4o', 'any-to-any']
        
        t2i_patterns = ['stable-diffusion', 'sdxl', 'flux', 'dalle', 'midjourney',
                        'kandinsky', 'deepfloyd', 'playground', 'pixart', 'hunyuan']
        
        i2i_patterns = ['controlnet', 'ip-adapter', 'instruct-pix2pix', 'img2img']
        
        t2v_patterns = ['cogvideo', 'text2video', 'modelscope-video', 'zeroscope',
                        'animatediff', 'videocrafter', 'lavie', 'show-1']
        
        # Try to read config.json for more accurate detection
        config = {}
        config_path = Path(model_path) / "config.json" if Path(model_path).is_dir() else None
        if config_path and config_path.exists():
            try:
                with open(config_path) as f:
                    config = json.load(f)
            except Exception:
                pass
        
        # Build detection string from config and path
        model_type_cfg = config.get("model_type", "").lower()
        architectures = [a.lower() for a in config.get("architectures", [])]
        arch_str = " ".join(architectures) + " " + model_type_cfg + " " + model_path_lower
        
        # Get context length from config
        context_length = 8192
        for key in ["max_position_embeddings", "max_seq_length", "seq_length", 
                    "n_positions", "sliding_window", "max_sequence_length"]:
            if key in config and isinstance(config[key], int) and config[key] > 0:
                context_length = config[key]
                break
        
        # Check for tool call support
        tool_patterns = ['qwen2', 'qwen2.5', 'llama3', 'mistral', 'deepseek', 'yi', 'glm4']
        supports_tools = any(tc in arch_str for tc in tool_patterns)
        
        # ============================================================
        # TEXT-TO-IMAGE MODELS (Diffusion models)
        # ============================================================
        if any(x in arch_str for x in t2i_patterns):
            return ModelCapabilities(
                supports_text_input=True,
                supports_image_input=False,
                supports_audio_input=False,
                supports_video_input=False,
                supports_text_output=False,
                supports_image_output=True,
                supports_audio_output=False,
                supports_video_output=False,
                supported_backends=["transformers"],  # Diffusers only
                supports_streaming=False,
                supports_system_prompt=False,
                model_type="t2i",
                max_context_length=77  # CLIP token limit
            )
        
        # ============================================================
        # IMAGE-TO-IMAGE MODELS
        # ============================================================
        if any(x in arch_str for x in i2i_patterns):
            return ModelCapabilities(
                supports_text_input=True,
                supports_image_input=True,
                supports_audio_input=False,
                supports_video_input=False,
                supports_text_output=False,
                supports_image_output=True,
                supports_audio_output=False,
                supports_video_output=False,
                supported_backends=["transformers"],
                supports_streaming=False,
                supports_system_prompt=False,
                model_type="i2i",
                max_context_length=77
            )
        
        # ============================================================
        # TEXT-TO-VIDEO MODELS
        # ============================================================
        if any(x in arch_str for x in t2v_patterns):
            return ModelCapabilities(
                supports_text_input=True,
                supports_image_input='img' in arch_str,  # Some support image conditioning
                supports_audio_input=False,
                supports_video_input=False,
                supports_text_output=False,
                supports_image_output=False,
                supports_audio_output=False,
                supports_video_output=True,
                supported_backends=["transformers"],
                supports_streaming=False,
                supports_system_prompt=False,
                model_type="t2v",
                max_context_length=256
            )
        
        # ============================================================
        # TTS MODELS (Text-to-Speech)
        # ============================================================
        if any(x in arch_str for x in tts_patterns):
            return ModelCapabilities(
                supports_text_input=True,
                supports_image_input=False,
                supports_audio_input='voice' in arch_str or 'clone' in arch_str,  # Voice cloning
                supports_video_input=False,
                supports_text_output=False,
                supports_image_output=False,
                supports_audio_output=True,
                supports_video_output=False,
                supported_backends=["transformers"],
                supports_streaming=True,
                supports_system_prompt=False,
                model_type="tts",
                max_context_length=4096
            )
        
        # ============================================================
        # ASR MODELS (Speech-to-Text)
        # ============================================================
        if any(x in arch_str for x in asr_patterns):
            return ModelCapabilities(
                supports_text_input=False,
                supports_image_input=False,
                supports_audio_input=True,
                supports_video_input=False,
                supports_text_output=True,
                supports_image_output=False,
                supports_audio_output=False,
                supports_video_output=False,
                supported_backends=["transformers"],
                supports_streaming=True,
                supports_system_prompt=False,
                model_type="asr",
                max_context_length=4096
            )
        
        # ============================================================
        # MULTIMODAL LLMs (Omni models - any input, any output)
        # ============================================================
        if any(x in arch_str for x in mllm_patterns):
            return ModelCapabilities(
                supports_text_input=True,
                supports_image_input=True,
                supports_audio_input=True,
                supports_video_input=True,
                supports_text_output=True,
                supports_image_output=True,
                supports_audio_output=True,
                supports_video_output=True,
                supported_backends=["transformers"],  # Only transformers for full omni
                supports_streaming=True,
                supports_system_prompt=True,
                supports_tool_calls=True,
                model_type="mllm",
                max_context_length=128000
            )
        
        # ============================================================
        # VISION-LANGUAGE MODELS (Image+text input, text output)
        # ============================================================
        if any(x in arch_str for x in vlm_patterns):
            # VLMs: vLLM supports some (Qwen2-VL, LLaVA), SGLang supports some
            vlm_vllm_supported = any(v in arch_str for v in ['qwen2-vl', 'qwen2_vl', 'llava', 'pixtral'])
            vlm_sglang_supported = any(v in arch_str for v in ['qwen2-vl', 'qwen2_vl', 'llava'])
            
            backends = ["transformers"]
            if vlm_vllm_supported and VLLM_AVAILABLE:
                backends.append("vllm")
            if vlm_sglang_supported and SGLANG_AVAILABLE:
                backends.append("sglang")
            
            return ModelCapabilities(
                supports_text_input=True,
                supports_image_input=True,
                supports_audio_input=False,
                supports_video_input='video' in arch_str or 'qwen2_vl' in arch_str or 'qwen2-vl' in arch_str,
                supports_text_output=True,
                supports_image_output=False,
                supports_audio_output=False,
                supports_video_output=False,
                supported_backends=backends,
                supports_streaming=True,
                supports_system_prompt=True,
                supports_tool_calls=supports_tools,
                model_type="vlm",
                max_context_length=max(context_length, 32768)
            )
        
        # ============================================================
        # DEFAULT: Text-only LLM (all backends supported)
        # ============================================================
        backends = ["transformers"]
        if VLLM_AVAILABLE:
            backends.append("vllm")
        if SGLANG_AVAILABLE:
            backends.append("sglang")
        
        return ModelCapabilities(
            supports_text_input=True,
            supports_image_input=False,
            supports_audio_input=False,
            supports_video_input=False,
            supports_text_output=True,
            supports_image_output=False,
            supports_audio_output=False,
            supports_video_output=False,
            supported_backends=backends,
            supports_streaming=True,
            supports_system_prompt=True,
            supports_tool_calls=supports_tools,
            model_type="llm",
            max_context_length=context_length
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
                
                # Log successful memory cleanup (encrypted only)
                system_encrypted_log.log_inference_memory_cleanup(
                    cleanup_type="deep",
                    success=True,
                    memory_freed_gb=round(max(0, memory_freed), 2),
                    engines_cleared=engines_cleared
                )
                
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
                # Log failed memory cleanup (encrypted only)
                system_encrypted_log.log_inference_memory_cleanup(
                    cleanup_type="deep",
                    success=False,
                    error=str(e)
                )
                
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
                
                # Validate and resolve adapter path if provided
                if adapter_path:
                    adapter_path = self._resolve_adapter_path(adapter_path)
                    if adapter_path is None:
                        return {
                            "success": False,
                            "error": f"Failed to load model with adapter: Adapter path does not exist or contains no valid adapter"
                        }
                    adapter_dir = Path(adapter_path)
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
                
                memory_used = self._get_gpu_memory_used()
                
                # Log successful model load (encrypted only)
                system_encrypted_log.log_inference_model_load(
                    model_path=model_path,
                    adapter_path=adapter_path,
                    backend=backend.value,
                    success=True,
                    memory_used_gb=memory_used
                )
                
                return {
                    "success": True,
                    "model_path": model_path,
                    "adapter_path": adapter_path,
                    "backend": backend.value,
                    "memory_used_gb": memory_used,
                    "capabilities": self._model_capabilities.model_dump() if self._model_capabilities else None
                }
            
            except Exception as e:
                # Clean up on failure
                await self._clear_all_engines()
                
                # Log failed model load (encrypted only)
                system_encrypted_log.log_inference_model_load(
                    model_path=model_path,
                    adapter_path=adapter_path,
                    backend=backend.value,
                    success=False,
                    error=str(e)
                )
                
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
        Performs VRAM cleanup before reloading to ensure clean state.
        """
        logging.info(f"=== ADAPTER LOAD REQUEST ===")
        logging.info(f"Adapter path: {adapter_path}")
        logging.info(f"Current model: {self._current_model_path}")
        logging.info(f"Current backend: {self._current_backend}")
        
        if self._engine is None or self._current_model_path is None:
            return {
                "success": False,
                "error": "No model loaded. Please load a base model first."
            }
        
        if self._current_backend != InferenceBackend.TRANSFORMERS:
            return {
                "success": False,
                "error": f"Adapter loading only supported with Transformers backend. Current backend: {self._current_backend.value}"
            }
        
        # Resolve and validate adapter path
        resolved_adapter_path = self._resolve_adapter_path(adapter_path)
        if resolved_adapter_path is None:
            return {
                "success": False,
                "error": f"Adapter path does not exist or contains no valid adapter: {adapter_path}"
            }
        adapter_path = resolved_adapter_path
        adapter_dir = Path(adapter_path)
        
        # Check for required LoRA adapter files
        adapter_config = adapter_dir / "adapter_config.json"
        adapter_model = adapter_dir / "adapter_model.safetensors"
        adapter_model_bin = adapter_dir / "adapter_model.bin"
        
        if not (adapter_model.exists() or adapter_model_bin.exists()):
            return {
                "success": False,
                "error": f"Not a valid LoRA adapter: missing adapter weights in {adapter_path}"
            }
        
        # Validate adapter config matches base model
        try:
            with open(adapter_config, 'r') as f:
                config = json.load(f)
            base_model_name = config.get('base_model_name_or_path', '')
            logging.info(f"Adapter was trained on base model: {base_model_name}")
            
            # Check if adapter's base model matches current model (warning only)
            if base_model_name and self._current_model_path:
                current_model_name = self._current_model_path.split('/')[-1].lower()
                adapter_base_name = base_model_name.split('/')[-1].lower()
                if current_model_name != adapter_base_name:
                    logging.warning(f"Adapter base model ({adapter_base_name}) may not match current model ({current_model_name})")
        except Exception as e:
            logging.warning(f"Could not validate adapter config: {e}")
        
        try:
            logging.info(f"Loading adapter from: {adapter_path} onto model: {self._current_model_path}")
            
            # Clear existing engine before reload to free VRAM
            logging.info("Clearing VRAM before adapter reload...")
            await self._clear_all_engines()
            
            # Reload model with the new adapter
            result = await self.load_model(
                self._current_model_path,
                adapter_path,
                self._current_backend
            )
            
            if result.get("success"):
                logging.info(f"Adapter loaded successfully: {adapter_path}")
                # Log successful adapter load (encrypted only)
                system_encrypted_log.log_inference_adapter_load(
                    adapter_path=adapter_path,
                    base_model=self._current_model_path or "",
                    success=True
                )
            else:
                logging.warning(f"Adapter load returned failure: {result.get('error')}")
                # Log failed adapter load (encrypted only)
                system_encrypted_log.log_inference_adapter_load(
                    adapter_path=adapter_path,
                    base_model=self._current_model_path or "",
                    success=False,
                    error=result.get('error')
                )
            
            return result
            
        except Exception as e:
            error_msg = str(e)
            tb = traceback.format_exc()
            logging.error(f"Adapter loading error: {error_msg}")
            logging.error(f"Traceback: {tb}")
            
            # Log adapter load exception (encrypted only)
            system_encrypted_log.log_inference_adapter_load(
                adapter_path=adapter_path,
                base_model=self._current_model_path or "",
                success=False,
                error=f"{error_msg}\n{tb}"
            )
            
            # Provide more specific error messages for common adapter issues
            error_lower = error_msg.lower()
            
            if "config" in error_lower and "mismatch" in error_lower:
                return {
                    "success": False,
                    "error": "Adapter configuration mismatch. The adapter was trained with a different model architecture."
                }
            elif "shape" in error_lower or "size mismatch" in error_lower:
                return {
                    "success": False,
                    "error": "Adapter weight shape mismatch. The adapter is not compatible with this base model."
                }
            elif "not found" in error_lower or "no such file" in error_lower:
                return {
                    "success": False,
                    "error": f"Adapter files not found at: {adapter_path}"
                }
            elif "target_modules" in error_lower:
                return {
                    "success": False,
                    "error": "Adapter target modules not found in base model. The model architecture may not be compatible."
                }
            elif "peft" in error_lower:
                return {
                    "success": False,
                    "error": f"PEFT adapter error: {error_msg[:200]}"
                }
            elif "cuda" in error_lower or "memory" in error_lower:
                return {
                    "success": False,
                    "error": "GPU memory error while loading adapter. Try clearing memory first."
                }
            
            # For unknown errors, include more context instead of generic message
            sanitized = sanitized_log_service.sanitize_error(error_msg)
            # If sanitization made it too generic, provide the first part of the actual error
            if sanitized['user_message'] == 'An unexpected error occurred.':
                # Extract first meaningful part of error (up to 200 chars, no sensitive paths)
                safe_error = error_msg.split('\n')[0][:200]
                return {
                    "success": False,
                    "error": f"Adapter load failed: {safe_error}"
                }
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
        """
        Internal helper to clear all engine instances and free VRAM.
        
        Performs:
        1. Delete all engine references (Transformers, vLLM, SGLang)
        2. Clear model from memory
        3. Multiple garbage collection passes
        4. Clear CUDA cache on all GPUs
        5. Use GPU cleanup service for thorough cleanup
        """
        logging.info("Clearing all inference engines...")
        
        # Step 1: Clear Transformers engine
        if self._engine is not None:
            try:
                if hasattr(self._engine, 'model'):
                    if hasattr(self._engine.model, 'cpu'):
                        # Move to CPU first to help release GPU memory
                        try:
                            self._engine.model.cpu()
                        except:
                            pass
                    del self._engine.model
                if hasattr(self._engine, 'tokenizer'):
                    del self._engine.tokenizer
                del self._engine
            except Exception as e:
                logging.warning(f"Error clearing transformers engine: {e}")
            self._engine = None
        
        # Step 2: Clear vLLM engine
        if self._vllm_engine is not None:
            try:
                del self._vllm_engine
            except Exception as e:
                logging.warning(f"Error clearing vLLM engine: {e}")
            self._vllm_engine = None
        
        # Step 3: Clear SGLang engine
        if self._sglang_engine is not None:
            try:
                if hasattr(self._sglang_engine, 'shutdown'):
                    self._sglang_engine.shutdown()
                del self._sglang_engine
            except Exception as e:
                logging.warning(f"Error clearing SGLang engine: {e}")
            self._sglang_engine = None
        
        # Step 4: Reset state
        self._current_model_path = None
        self._current_adapter_path = None
        self._loaded_adapters = []
        
        # Step 5: Aggressive memory cleanup
        # Multiple GC passes to handle circular references
        for _ in range(3):
            gc.collect()
        
        # Step 6: Clear CUDA cache
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                # Reset memory stats for accurate tracking
                if hasattr(torch.cuda, 'reset_peak_memory_stats'):
                    torch.cuda.reset_peak_memory_stats()
            except Exception as e:
                logging.warning(f"Error clearing CUDA cache: {e}")
        
        # Step 7: Use GPU cleanup service for thorough cleanup
        try:
            cleanup_result = gpu_cleanup_service.clear_torch_cuda_cache()
            if cleanup_result.get("success"):
                logging.info("GPU cleanup service completed successfully")
        except Exception as e:
            logging.warning(f"GPU cleanup service error: {e}")
        
        logging.info("All engines cleared")
    
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
            
            # Log successful chat/generation (encrypted only)
            system_encrypted_log.log_inference_chat(
                model_path=self._current_model_path or "",
                backend=self._current_backend.value,
                message_count=len(request.messages) if hasattr(request, 'messages') else 0,
                success=True,
                tokens_generated=tokens_generated,
                inference_time_ms=inference_time_ms
            )
            
            # Determine output type based on model capabilities
            output_type = "text"
            images = None
            audio = None
            video = None
            
            if self._model_capabilities:
                if self._model_capabilities.supports_image_output:
                    output_type = "image"
                elif self._model_capabilities.supports_audio_output:
                    output_type = "audio"
                elif self._model_capabilities.supports_video_output:
                    output_type = "video"
                
                # For multimodal models, check if response contains special markers
                if self._model_capabilities.model_type == "mllm":
                    output_type = "multimodal"
            
            return InferenceResponse(
                success=True,
                response=response_text,
                images=images,
                audio=audio,
                video=video,
                tokens_generated=tokens_generated,
                inference_time_ms=inference_time_ms,
                model_loaded=self._current_model_path or "",
                backend_used=self._current_backend.value,
                output_type=output_type
            )
        
        except Exception as e:
            full_error = str(e)
            
            # Log failed chat/generation (encrypted only)
            system_encrypted_log.log_inference_chat(
                model_path=self._current_model_path or "",
                backend=self._current_backend.value if self._current_backend else "unknown",
                message_count=len(request.messages) if hasattr(request, 'messages') else 0,
                success=False,
                error=full_error
            )
            
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
        """
        Generate using vLLM backend.
        
        Supports multimodal input for VLM models (Qwen2-VL, LLaVA, etc.)
        by passing images via multi_modal_data parameter.
        """
        if not VLLM_AVAILABLE or self._vllm_engine is None:
            raise RuntimeError("vLLM not available")
        
        # Build prompt from messages
        prompt = self._messages_to_prompt(request.messages)
        
        # Extract multimodal data for VLM models
        multimodal_data = None
        if self._model_capabilities and self._model_capabilities.supports_image_input:
            mm_data = self._extract_multimodal_data(request.messages)
            if mm_data["images"]:
                # vLLM expects images in multi_modal_data format
                # Process base64 images if needed
                processed_images = []
                for img in mm_data["images"]:
                    if img.startswith("data:"):
                        # Extract base64 data
                        import base64
                        try:
                            # Format: data:image/png;base64,<data>
                            header, b64_data = img.split(",", 1)
                            img_bytes = base64.b64decode(b64_data)
                            processed_images.append(img_bytes)
                        except Exception:
                            processed_images.append(img)
                    else:
                        processed_images.append(img)
                
                if processed_images:
                    multimodal_data = {"image": processed_images[0] if len(processed_images) == 1 else processed_images}
        
        sampling_params = SamplingParams(
            max_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
        )
        
        # Generate with or without multimodal data
        if multimodal_data:
            outputs = self._vllm_engine.generate(
                [{"prompt": prompt, "multi_modal_data": multimodal_data}],
                sampling_params
            )
        else:
            outputs = self._vllm_engine.generate([prompt], sampling_params)
        
        response_text = outputs[0].outputs[0].text if outputs else ""
        tokens_generated = len(outputs[0].outputs[0].token_ids) if outputs else 0
        
        return response_text, tokens_generated
    
    async def _generate_sglang(self, request: InferenceRequest) -> tuple:
        """
        Generate using SGLang backend.
        
        Supports multimodal input for VLM models (Qwen2-VL, LLaVA, etc.)
        by passing images in the prompt.
        """
        if not SGLANG_AVAILABLE:
            raise RuntimeError("SGLang not available")
        
        # Build prompt from messages
        prompt = self._messages_to_prompt(request.messages)
        
        # Extract multimodal data for VLM models
        images = None
        if self._model_capabilities and self._model_capabilities.supports_image_input:
            mm_data = self._extract_multimodal_data(request.messages)
            if mm_data["images"]:
                images = mm_data["images"]
        
        @sgl.function
        def chat_completion(s):
            # Add images if present (SGLang VLM support)
            if images:
                for img in images:
                    if hasattr(sgl, 'image'):
                        s += sgl.image(img)
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
        """
        Convert OpenAI-format messages to a single prompt string.
        
        NOTE: This is used by vLLM and SGLang backends.
        For multimodal content (images/audio/video), only text is extracted here.
        Multimodal content for these backends requires special handling at the
        engine level (e.g., vLLM's multi_modal_data parameter).
        """
        prompt_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            # Handle multimodal content - extract text only for prompt
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
    
    def _extract_multimodal_data(self, messages: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Extract multimodal data (images, audio, video) from messages.
        
        Returns a dict with:
        - images: list of image data (base64 or URLs)
        - audio: list of audio data
        - video: list of video data
        
        Used by backends that support multimodal input.
        """
        multimodal_data = {
            "images": [],
            "audio": [],
            "video": []
        }
        
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if not isinstance(item, dict):
                        continue
                    
                    item_type = item.get("type", "")
                    
                    if item_type == "image_url":
                        image_url = item.get("image_url", {})
                        if isinstance(image_url, dict) and image_url.get("url"):
                            multimodal_data["images"].append(image_url["url"])
                        elif isinstance(image_url, str):
                            multimodal_data["images"].append(image_url)
                    
                    elif item_type == "audio":
                        audio_data = item.get("audio", {})
                        if audio_data:
                            multimodal_data["audio"].append(audio_data)
                    
                    elif item_type == "video":
                        video_data = item.get("video", {})
                        if video_data:
                            multimodal_data["video"].append(video_data)
        
        return multimodal_data
    
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
