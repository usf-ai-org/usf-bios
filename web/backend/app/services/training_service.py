# Copyright (c) US Inc. All rights reserved.
"""Training service - executes fine-tuning jobs"""

import asyncio
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
import traceback
from datetime import datetime
from pathlib import Path
from typing import Optional

# Optional imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

from ..core.config import settings
from ..core.capabilities import get_system_settings
from ..core.database import SessionLocal
from ..models.schemas import JobInfo, JobStatus, TrainingConfig
from .job_manager import job_manager
from .websocket_manager import ws_manager
from .sanitized_log_service import sanitized_log_service
from .encrypted_log_service import encrypted_log_service
from .job_service import JobService
from .gpu_cleanup_service import gpu_cleanup_service


def _sync_job_to_database(job_id: str, status: str, error_message: str = None, 
                          started_at: datetime = None, completed_at: datetime = None,
                          output_dir: str = None):
    """Sync job status to database for persistent history.
    
    This ensures training history survives server restarts.
    Called after job status changes (started, completed, failed, stopped).
    """
    try:
        from ..core.database import SessionLocal
        from ..models.db_models import TrainingJob as DBTrainingJob
        
        db = SessionLocal()
        try:
            db_job = db.query(DBTrainingJob).filter(DBTrainingJob.id == job_id).first()
            if db_job:
                db_job.status = status
                if error_message:
                    db_job.error_message = error_message
                if started_at:
                    db_job.started_at = started_at
                if completed_at:
                    db_job.completed_at = completed_at
                    # Calculate duration
                    if db_job.started_at:
                        duration = (completed_at - db_job.started_at).total_seconds()
                        db_job.duration_seconds = int(duration)
                if output_dir:
                    db_job.output_dir = output_dir
                db.commit()
        finally:
            db.close()
    except Exception as e:
        _debug_log(job_id, f"Failed to sync job to database: {e}", "WARNING")


def _debug_log(job_id: str, message: str, level: str = "DEBUG"):
    """Write debug log to ENCRYPTED log file (only US Inc can read)."""
    encrypted_log_service.encrypt_and_format(f"[{level}] {message}", job_id, level)


# ============================================================
# HELPER FUNCTIONS FOR MODEL SIZE CALCULATION
# ============================================================


def _get_model_size_from_config(model_path: str) -> Optional[float]:
    """Try to read actual model size in GB from model's config.json.
    
    Detects dtype and calculates accurate size based on:
    - Parameter count from architecture
    - Bytes per parameter based on dtype (float32=4, bfloat16/float16=2, int8=1, int4=0.5)
    
    Returns size in GB if calculable, None otherwise.
    """
    config_path = os.path.join(model_path, "config.json")
    if not os.path.exists(config_path):
        return None
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Detect dtype - check multiple possible field names
        torch_dtype = config.get('torch_dtype', None)
        dtype = config.get('dtype', None)
        
        # Determine bytes per parameter based on dtype
        bytes_per_param = 2  # Default: bfloat16/float16
        
        dtype_str = (torch_dtype or dtype or '').lower()
        if 'float32' in dtype_str or 'fp32' in dtype_str:
            bytes_per_param = 4
        elif 'float16' in dtype_str or 'fp16' in dtype_str or 'bfloat16' in dtype_str or 'bf16' in dtype_str:
            bytes_per_param = 2
        elif 'int8' in dtype_str or 'q8' in dtype_str:
            bytes_per_param = 1
        elif 'int4' in dtype_str or 'q4' in dtype_str or 'nf4' in dtype_str or 'fp4' in dtype_str:
            bytes_per_param = 0.5
        else:
            # Check for quantization config
            quant_config = config.get('quantization_config', {})
            if quant_config:
                quant_method = quant_config.get('quant_method', '').lower()
                bits = quant_config.get('bits', quant_config.get('load_in_4bit', quant_config.get('load_in_8bit', None)))
                
                if bits == 4 or quant_config.get('load_in_4bit'):
                    bytes_per_param = 0.5
                elif bits == 8 or quant_config.get('load_in_8bit'):
                    bytes_per_param = 1
                elif 'gptq' in quant_method or 'awq' in quant_method:
                    # GPTQ/AWQ typically use 4-bit
                    bits = quant_config.get('bits', 4)
                    bytes_per_param = bits / 8
        
        # Get parameter count
        total_params = None
        
        # Some models have explicit parameter count
        if 'num_parameters' in config:
            total_params = config['num_parameters']
        else:
            # Calculate from architecture
            hidden_size = config.get('hidden_size', config.get('d_model', 0))
            num_layers = config.get('num_hidden_layers', config.get('n_layer', config.get('num_layers', 0)))
            vocab_size = config.get('vocab_size', 0)
            intermediate_size = config.get('intermediate_size', config.get('ffn_dim', hidden_size * 4 if hidden_size else 0))
            num_attention_heads = config.get('num_attention_heads', config.get('n_head', 0))
            num_kv_heads = config.get('num_key_value_heads', num_attention_heads)
            
            if hidden_size and num_layers and vocab_size:
                # Embedding parameters
                embedding_params = vocab_size * hidden_size * 2  # input + output embeddings
                
                # Per-layer parameters (attention + FFN)
                head_dim = hidden_size // num_attention_heads if num_attention_heads else hidden_size
                q_params = hidden_size * hidden_size
                k_params = hidden_size * (num_kv_heads * head_dim) if num_kv_heads else hidden_size * hidden_size
                v_params = k_params
                o_params = hidden_size * hidden_size
                attention_params = q_params + k_params + v_params + o_params
                
                # FFN: gate, up, down projections (for LLaMA-style)
                ffn_params = hidden_size * intermediate_size * 3
                
                # Layer norms
                ln_params = hidden_size * 4
                
                layer_params = attention_params + ffn_params + ln_params
                total_layer_params = layer_params * num_layers
                
                total_params = embedding_params + total_layer_params
        
        if total_params:
            # Calculate size in GB
            size_gb = (total_params * bytes_per_param) / (1024 ** 3)
            return max(size_gb, 0.5)
            
    except Exception:
        pass
    
    return None


def _get_model_size_from_directory(model_path: str, job_id: str = None) -> Optional[dict]:
    """
    Get model size by measuring ACTUAL directory size on disk.
    
    This approach works for ALL model types:
    - Text-to-text (LLMs like Llama, Mistral, Qwen)
    - Image-to-text (Vision models like LLaVA, BLIP)
    - Text-to-image (Diffusion models like Stable Diffusion)
    - Audio models (Whisper, etc.)
    - Video models
    - Multi-modal models (any combination)
    
    Returns dict with:
        - size_gb: Actual size on disk in GB
        - source: 'directory'
        - hidden_size: From config.json if available (for LoRA calculation)
        - num_layers: From config.json if available (for LoRA calculation)
    
    Returns None if path doesn't exist or size cannot be determined.
    """
    if not os.path.isdir(model_path):
        return None
    
    # Get actual directory size - works for ANY model type
    size_gb = _get_dir_size_gb_fast(model_path, job_id)
    
    if size_gb <= 0:
        return None
    
    # Try to get hidden_size and num_layers from config.json for LoRA calculations
    # These are optional - if not found, we'll use defaults
    hidden_size = 4096  # Default
    num_layers = 32     # Default
    
    config_path = os.path.join(model_path, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Try various config keys used by different model architectures
            hidden_size = config.get('hidden_size', 
                          config.get('d_model',
                          config.get('n_embd',
                          config.get('dim', 4096))))
            
            num_layers = config.get('num_hidden_layers',
                         config.get('n_layer',
                         config.get('num_layers',
                         config.get('n_layers', 32))))
        except Exception:
            pass
    
    return {
        'size_gb': size_gb,
        'hidden_size': hidden_size,
        'num_layers': num_layers,
        'source': 'directory'
    }


def _get_model_cache_path(model_source: str) -> str:
    """
    Get the cache directory path for a specific model source.
    
    IMPORTANT: HuggingFace and ModelScope have SEPARATE cache directories
    to prevent any conflicts or model replacement issues.
    
    Structure (using short names for cleaner paths):
    ~/.cache/usf_bios/
    ├── hf/     <- All HuggingFace models (short for HuggingFace)
    │   └── meta-llama_Llama-2-7b/
    │   └── mistralai_Mistral-7B/
    └── ms/     <- All ModelScope models (short for ModelScope)
        └── qwen_Qwen-7B/
        └── ZhipuAI_chatglm3-6b/
    """
    from pathlib import Path
    base_cache = Path.home() / ".cache" / "usf_bios"
    
    if model_source == "huggingface":
        return str(base_cache / "hf")  # Short: HuggingFace
    elif model_source == "modelscope":
        return str(base_cache / "ms")  # Short: ModelScope
    else:
        return str(base_cache / "models")


def _download_remote_model(model_id: str, model_source: str, job_id: str = None) -> Optional[str]:
    """
    Download a remote model (HuggingFace/ModelScope) to local cache.
    
    CRITICAL DESIGN:
    1. HuggingFace and ModelScope have SEPARATE cache directories
       - ~/.cache/usf_bios/hf/  for HuggingFace
       - ~/.cache/usf_bios/ms/  for ModelScope
    2. Model is downloaded ONCE - same path used for:
       - Size calculation
       - Storage validation  
       - ALL training types (SFT, RLHF, PT)
       - ALL algorithms (DPO, PPO, GRPO, KTO, ORPO, etc.)
       - ALL quantization methods (LoRA, QLoRA, Full)
       - Adapter merging
    3. If same model_id exists in both sources, they are stored separately
    4. No conflicts, no replacements, no duplicates within same source
    
    Returns the local path where model is downloaded, or None if failed.
    """
    try:
        # Get source-specific cache directory (SEPARATE for HF and MS)
        cache_dir = _get_model_cache_path(model_source)
        os.makedirs(cache_dir, exist_ok=True)
        
        # Create safe directory name from model_id
        safe_model_name = model_id.replace("/", "_").replace("\\", "_")
        local_model_dir = os.path.join(cache_dir, safe_model_name)
        
        if model_source == "huggingface":
            from huggingface_hub import snapshot_download
            
            # Download to source-specific directory
            # Using local_dir ensures files are in a predictable location
            local_path = snapshot_download(
                repo_id=model_id,
                local_dir=local_model_dir,
                local_dir_use_symlinks=False  # Actual files, not symlinks
            )
            
            if job_id:
                _debug_log(job_id, f"HuggingFace model downloaded to: {local_path}")
            
            return local_path
            
        elif model_source == "modelscope":
            from modelscope import snapshot_download as ms_snapshot_download
            
            # ModelScope downloads to its own structure
            local_path = ms_snapshot_download(
                model_id=model_id,
                cache_dir=cache_dir,
                local_dir=local_model_dir
            )
            
            if job_id:
                _debug_log(job_id, f"ModelScope model downloaded to: {local_path}")
            
            return local_path
            
    except Exception as e:
        if job_id:
            _debug_log(job_id, f"Failed to download model {model_id} from {model_source}: {e}")
    
    return None


def _get_model_info_universal(model_path: str, model_source: str, job_id: str = None, 
                               download_if_remote: bool = True) -> dict:
    """
    Universal model info function that works for ALL model types and sources.
    
    Supports:
    - LOCAL models: Measures actual directory size (works for any modality)
    - HuggingFace models: Downloads first, then measures local size
    - ModelScope models: Downloads first, then measures local size
    - Adapters: Measures adapter directory size
    
    Works for ALL modalities:
    - Text-to-text, Image-to-text, Text-to-image
    - Audio-to-text, Text-to-audio, Audio-to-audio
    - Video models, Multi-modal models
    
    Returns dict with:
        - size_gb: Model size in GB
        - hidden_size: For LoRA calculations
        - num_layers: For LoRA calculations
        - source: 'directory' or 'downloaded'
        - local_path: Actual local path to use for training (important for remote models!)
        - error: Error message if failed
    """
    result = {
        'size_gb': 0,
        'hidden_size': 4096,
        'num_layers': 32,
        'source': None,
        'local_path': model_path,  # Default: use original path
        'error': None
    }
    
    # Case 1: LOCAL model - just measure directory size
    if model_source == "local" and os.path.isdir(model_path):
        model_info = _get_model_size_from_directory(model_path, job_id)
        
        if model_info:
            result['size_gb'] = model_info['size_gb']
            result['hidden_size'] = model_info['hidden_size']
            result['num_layers'] = model_info['num_layers']
            result['source'] = 'directory'
            result['local_path'] = model_path
            return result
        else:
            result['error'] = f"Cannot determine size of local model at: {model_path}"
            return result
    
    # Case 2: Remote model (HuggingFace/ModelScope)
    if model_source in ("huggingface", "modelscope"):
        if not download_if_remote:
            # Skip download - return placeholder for remote
            result['source'] = 'remote_pending'
            result['error'] = None  # Not an error, just pending
            return result
        
        # Download model first
        if job_id:
            _debug_log(job_id, f"Downloading {model_source} model: {model_path}")
        
        local_path = _download_remote_model(model_path, model_source, job_id=job_id)
        
        if local_path and os.path.isdir(local_path):
            # Now measure the downloaded model's size
            model_info = _get_model_size_from_directory(local_path, job_id)
            
            if model_info:
                result['size_gb'] = model_info['size_gb']
                result['hidden_size'] = model_info['hidden_size']
                result['num_layers'] = model_info['num_layers']
                result['source'] = 'downloaded'
                result['local_path'] = local_path  # Use downloaded path for training!
                return result
        
        result['error'] = f"Failed to download model from {model_source}: {model_path}"
        return result
    
    # Unknown source
    result['error'] = f"Unknown model source: {model_source}"
    return result


def _get_dir_size_gb_fast(path: str, job_id: str = None, timeout_seconds: int = 10) -> float:
    """Get directory size in GB - tries multiple methods for reliability.
    
    Methods (in order of preference):
    1. 'du' command - fast, accurate on Unix systems
    2. Python os.walk() - slower but always works, cross-platform
    
    Returns size in GB, or 0 if cannot determine.
    """
    if not os.path.exists(path):
        return 0
    
    # Method 1: Try 'du' command first (fast)
    try:
        if platform.system() == 'Darwin':  # macOS
            result = subprocess.run(
                ['du', '-sk', path],
                capture_output=True, text=True, timeout=timeout_seconds
            )
            if result.returncode == 0:
                size_kb = int(result.stdout.split()[0])
                return size_kb / (1024 * 1024)
        else:  # Linux
            result = subprocess.run(
                ['du', '-sb', path],
                capture_output=True, text=True, timeout=timeout_seconds
            )
            if result.returncode == 0:
                size_bytes = int(result.stdout.split()[0])
                return size_bytes / (1024 ** 3)
    except subprocess.TimeoutExpired:
        if job_id:
            _debug_log(job_id, f"du command timed out, falling back to Python method")
    except Exception as e:
        if job_id:
            _debug_log(job_id, f"du command failed: {e}, falling back to Python method")
    
    # Method 2: Fallback to Python os.walk() - reliable but slower
    try:
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except (OSError, IOError):
                    pass  # Skip files we can't access
        return total_size / (1024 ** 3)
    except Exception as e:
        if job_id:
            _debug_log(job_id, f"Python size calculation also failed: {e}")
    
    return 0  # Return 0 if all methods fail


async def _get_dir_size_gb_async(path: str, job_id: str = None, timeout_seconds: int = 5) -> float:
    """Async version: Get directory size in GB without blocking the event loop.
    
    Uses asyncio.to_thread() to run the blocking 'du' command in a thread pool.
    Falls back to estimation from model name if du fails or times out.
    """
    try:
        # Run blocking du command in thread pool with overall timeout
        size = await asyncio.wait_for(
            asyncio.to_thread(_get_dir_size_gb_fast, path, job_id, timeout_seconds),
            timeout=timeout_seconds + 2  # Extra buffer for thread overhead
        )
        return size
    except asyncio.TimeoutError:
        if job_id:
            _debug_log(job_id, f"Async du timed out for {path}")
        return 0  # Return 0 if cannot determine size
    except Exception as e:
        if job_id:
            _debug_log(job_id, f"Async du failed: {e}")
        return 0  # Return 0 if cannot determine size


def _get_available_space_gb(path: str) -> float:
    """Get available disk space in GB for a path (instant, uses shutil.disk_usage).
    
    SAFETY: Uses iteration limit to prevent infinite loops in edge cases.
    """
    try:
        check_path = path
        max_iterations = 100  # Safety limit to prevent infinite loops
        iterations = 0
        
        while not os.path.exists(check_path) and check_path != '/' and iterations < max_iterations:
            parent = os.path.dirname(check_path)
            if parent == check_path:  # Reached root or stuck
                break
            check_path = parent
            iterations += 1
        
        if not check_path or not os.path.exists(check_path):
            check_path = '/'
        
        usage = shutil.disk_usage(check_path)
        return usage.free / (1024 ** 3)
    except Exception:
        return -1




def _calculate_lora_adapter_size(hidden_size: int, num_layers: int, lora_rank: int, 
                                  lora_alpha: int = None, target_modules: list = None) -> float:
    """
    Calculate LoRA adapter size in GB.
    
    LoRA adds low-rank matrices A and B to target modules.
    Size = 2 * rank * hidden_size * num_target_modules * num_layers * bytes_per_param
    
    Default target modules: q_proj, k_proj, v_proj, o_proj (4 modules per layer)
    """
    if not hidden_size or not num_layers or not lora_rank:
        return 0.5  # Default minimum
    
    # Default: 4 attention projections per layer
    num_target_modules = len(target_modules) if target_modules else 4
    
    # LoRA matrices: A (hidden x rank) and B (rank x hidden) for each target
    params_per_layer = 2 * lora_rank * hidden_size * num_target_modules
    total_params = params_per_layer * num_layers
    
    # LoRA adapters are typically saved in fp16/bf16 (2 bytes)
    size_gb = (total_params * 2) / (1024 ** 3)
    
    return max(size_gb, 0.01)  # Minimum 10MB


def _calculate_storage_requirements(config, model_path: str, model_source: str,
                                    merge_enabled: bool, adapter_base_path: str = None,
                                    job_id: str = None) -> dict:
    """
    Calculate accurate storage requirements for training.
    
    Uses ACTUAL directory size for local models (works for ALL model types):
    - Text-to-text, Image-to-text, Text-to-image
    - Audio models, Video models, Multi-modal models
    
    For remote models (HuggingFace/ModelScope): Downloads first, then calculates.
    
    Returns dict with:
        - temp_required_gb: space needed in temp dir (only for merge)
        - output_required_gb: space needed in output dir
        - model_info: dict with model size details
        - checkpoint_info: dict with checkpoint calculation details
        - breakdown: list of human-readable breakdown
        - local_model_path: Actual local path to use for training (important for remote!)
        - error: error message if failed
    """
    result = {
        'temp_required_gb': 0,
        'output_required_gb': 0,
        'model_info': None,
        'checkpoint_info': {},
        'breakdown': [],
        'local_model_path': model_path,  # Default to original path
        'error': None
    }
    
    # Get training parameters from config
    train_type = getattr(config, 'train_type', None)
    train_type_value = train_type.value if hasattr(train_type, 'value') else str(train_type)
    output_dir = getattr(config, 'output_dir', 'output') or 'output'
    
    # LoRA parameters
    lora_rank = getattr(config, 'lora_rank', 8) or 8
    lora_alpha = getattr(config, 'lora_alpha', 16) or 16
    lora_target = getattr(config, 'lora_target', None)
    
    # Checkpoint parameters
    save_steps = getattr(config, 'save_steps', None)
    save_total_limit = getattr(config, 'save_total_limit', None)
    num_train_epochs = getattr(config, 'num_train_epochs', 1) or 1
    
    # =====================================================
    # 1. GET MODEL SIZE (works for ALL model types/modalities)
    # Uses actual directory size - no architecture-specific parsing
    # =====================================================
    model_info = None
    check_path = None
    
    if merge_enabled and adapter_base_path:
        # For merge: use base model path
        check_path = adapter_base_path
        model_info = _get_model_size_from_directory(adapter_base_path, job_id)
    elif model_source == 'local' and os.path.isdir(model_path):
        # LOCAL model: measure actual directory size
        check_path = model_path
        model_info = _get_model_size_from_directory(model_path, job_id)
    elif model_source in ('huggingface', 'modelscope'):
        # REMOTE model: download first, then measure
        # This ensures NO duplicate downloads - same path used for training
        universal_info = _get_model_info_universal(
            model_path, model_source, job_id=job_id, download_if_remote=True
        )
        
        if universal_info.get('error'):
            result['error'] = universal_info['error']
            return result
        
        if universal_info.get('source') == 'downloaded':
            model_info = {
                'size_gb': universal_info['size_gb'],
                'hidden_size': universal_info['hidden_size'],
                'num_layers': universal_info['num_layers'],
                'source': 'downloaded'
            }
            # IMPORTANT: Update local_model_path for training to use!
            result['local_model_path'] = universal_info['local_path']
            check_path = universal_info['local_path']
        else:
            # Download pending or failed
            result['model_info'] = {
                'size_gb': 0,
                'hidden_size': 0,
                'num_layers': 0,
                'source': 'remote_skip'
            }
            result['breakdown'].append("Model: Remote model - storage validation skipped")
            result['output_required_gb'] = 0
            return result
    else:
        result['error'] = f"Unknown model source or invalid path: {model_source}"
        return result
    
    # Check if we got model info
    if model_info is None or model_info.get('size_gb', 0) <= 0:
        result['error'] = (
            f"Cannot determine model size.\n"
            f"Please ensure the model directory exists and is accessible."
        )
        return result
    
    result['model_info'] = model_info
    result['breakdown'].append(f"Model: {model_info['size_gb']:.1f}GB (actual size on disk)")
    
    # =====================================================
    # 2. TEMP DIRECTORY (ONLY FOR MERGE)
    # =====================================================
    if merge_enabled and adapter_base_path:
        # Merged model will be saved in fp16/bf16
        merge_size = model_info['size_gb'] * 1.1  # 10% buffer
        result['temp_required_gb'] = merge_size
        result['breakdown'].append(f"Temp (merge): {merge_size:.1f}GB (merged model + buffer)")
    
    # =====================================================
    # 3. OUTPUT DIRECTORY
    # =====================================================
    
    # Estimate number of checkpoints
    # Default assumption: 3 checkpoints during training + final
    if save_total_limit:
        num_checkpoints = save_total_limit
    else:
        num_checkpoints = 4  # Conservative default
    
    result['checkpoint_info'] = {
        'num_checkpoints': num_checkpoints,
        'save_total_limit': save_total_limit,
        'train_type': train_type_value
    }
    
    if train_type_value in ['lora', 'qlora', 'adalora']:
        # LoRA: small adapter checkpoints
        adapter_size = _calculate_lora_adapter_size(
            hidden_size=model_info.get('hidden_size', 4096),
            num_layers=model_info.get('num_layers', 32),
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            target_modules=lora_target.split(',') if isinstance(lora_target, str) else lora_target
        )
        output_size = adapter_size * num_checkpoints
        result['checkpoint_info']['checkpoint_size_gb'] = adapter_size
        result['breakdown'].append(f"Output (LoRA r={lora_rank}): {num_checkpoints} checkpoints × {adapter_size:.2f}GB = {output_size:.1f}GB")
    else:
        # Full fine-tuning: full model checkpoints
        checkpoint_size = model_info['size_gb']
        output_size = checkpoint_size * num_checkpoints
        result['checkpoint_info']['checkpoint_size_gb'] = checkpoint_size
        result['breakdown'].append(f"Output (Full): {num_checkpoints} checkpoints × {checkpoint_size:.1f}GB = {output_size:.1f}GB")
    
    # Add buffer for checkpoint overhead (full training only)
    if train_type_value == 'full':
        # Optimizer states are in GPU VRAM during training, NOT saved to disk by default.
        # Only add a small buffer for training state files (scheduler, rng, trainer_state.json)
        checkpoint_overhead = model_info['size_gb'] * 0.05  # 5% buffer
        output_size += checkpoint_overhead
        result['breakdown'].append(f"Checkpoint overhead: ~{checkpoint_overhead:.1f}GB")
    
    # Logs and misc
    output_size += 0.5  # 500MB for logs, metrics, etc.
    result['breakdown'].append("Logs & metrics: ~0.5GB")
    
    result['output_required_gb'] = output_size
    
    return result


def _log_step(job_id: str, step_name: str, details: dict = None, level: str = "INFO"):
    """
    Log a training step with full details to encrypted log.
    
    COMPLETE LOGGING: Every step is logged with all details.
    This ensures full traceability for debugging.
    """
    detail_str = ""
    if details:
        try:
            detail_str = f" | Details: {json.dumps(details, default=str)}"
        except:
            detail_str = f" | Details: {str(details)}"
    
    message = f"[STEP:{step_name}]{detail_str}"
    encrypted_log_service.encrypt_and_format(message, job_id, level)


class TrainingService:
    """Service for running training jobs"""
    
    def __init__(self):
        self._running_tasks: dict = {}
        # Track merged model temp directories for cleanup
        self._merged_model_dirs: dict = {}  # job_id -> temp_dir_path
    
    def _validate_feature_flags(self, config: TrainingConfig) -> tuple[bool, str]:
        """
        Validate that requested features are enabled in this build.
        
        SECURITY: This validation uses compiled flags from system_guard.
        Users CANNOT bypass this even by modifying frontend or version.py.
        
        Returns: (is_valid, error_message)
        """
        try:
            from usf_bios.system_guard import (
                validate_train_type,
                validate_rlhf_algorithm,
                validate_vllm_mode,
                validate_training_method,
                FeatureDisabledError
            )
            
            # Validate training method (sft, rlhf, pt)
            training_method = getattr(config, 'training_method', None)
            if training_method:
                method_value = training_method.value if hasattr(training_method, 'value') else str(training_method)
                try:
                    validate_training_method(method_value)
                except FeatureDisabledError as e:
                    return False, str(e)
            
            # Validate train type (lora, qlora, full, adalora)
            train_type = getattr(config, 'train_type', None)
            if train_type:
                type_value = train_type.value if hasattr(train_type, 'value') else str(train_type)
                try:
                    validate_train_type(type_value)
                except FeatureDisabledError as e:
                    return False, str(e)
            
            # Validate RLHF algorithm if applicable
            if training_method:
                method_value = training_method.value if hasattr(training_method, 'value') else str(training_method)
                if method_value == "rlhf":
                    rlhf_type = getattr(config, 'rlhf_type', None)
                    if rlhf_type:
                        try:
                            validate_rlhf_algorithm(rlhf_type)
                        except FeatureDisabledError as e:
                            return False, str(e)
                    
                    # Validate vLLM mode for online RL
                    online_algos = ["grpo", "ppo", "gkd"]
                    if rlhf_type in online_algos:
                        vllm_mode = getattr(config, 'vllm_mode', None)
                        if vllm_mode:
                            try:
                                validate_vllm_mode(vllm_mode)
                            except FeatureDisabledError as e:
                                return False, str(e)
            
            return True, ""
            
        except ImportError:
            # system_guard not available (development mode) - allow everything
            return True, ""
    
    def _validate_online_rl_config(self, config: TrainingConfig) -> tuple[bool, str]:
        """
        Validate online RL configuration before training starts.
        
        SECURITY: This validation is enforced by backend - frontend cannot bypass.
        
        Checks:
        1. Server mode requires verified vLLM server endpoint
        2. Colocate mode requires sufficient GPU resources (2+ GPUs)
        3. Hash verification to prevent tampering with verified state
        
        Returns: (is_valid, error_message)
        """
        training_method = getattr(config, 'training_method', None)
        if training_method:
            method_value = training_method.value if hasattr(training_method, 'value') else str(training_method)
        else:
            return True, ""
        
        if method_value != "rlhf":
            return True, ""
        
        rlhf_type = getattr(config, 'rlhf_type', None)
        online_rl_algorithms = ["grpo", "ppo", "gkd"]
        
        if rlhf_type not in online_rl_algorithms:
            return True, ""  # Offline RL doesn't need validation
        
        use_vllm = getattr(config, 'use_vllm', True)
        vllm_mode = getattr(config, 'vllm_mode', None)
        
        if not use_vllm:
            return True, ""  # Not using vLLM
        
        # Server mode validation
        if vllm_mode == "server":
            host = getattr(config, 'vllm_server_host', None)
            port = getattr(config, 'vllm_server_port', 8000)
            verified = getattr(config, 'vllm_server_verified', False)
            verified_hash = getattr(config, 'vllm_server_verified_hash', None)
            
            if not host:
                return False, "Server mode requires vllm_server_host to be specified"
            
            if not verified:
                return False, "vLLM server endpoint must be verified before training. Click 'Test Connection' in the UI."
            
            # Verify hash to prevent tampering
            expected_hash = hashlib.sha256(f"{host}:{port}".encode()).hexdigest()[:16]
            if verified_hash != expected_hash:
                return False, "vLLM server verification mismatch. Please re-verify the server endpoint."
        
        # Colocate mode validation
        elif vllm_mode == "colocate":
            if TORCH_AVAILABLE and torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                if gpu_count < 2:
                    return False, "Colocate mode requires at least 2 GPUs. Use server mode instead."
            else:
                return False, "No GPU available for colocate mode"
        
        return True, ""
    
    def _build_command(self, config: TrainingConfig, job_id: str, resume_from_checkpoint: str = None) -> list:
        """Build the training command based on training method"""
        # SECURITY: Validate feature flags first (compiled, cannot bypass)
        is_valid, error = self._validate_feature_flags(config)
        if not is_valid:
            raise ValueError(f"Feature validation failed: {error}")
        
        # SECURITY: Validate online RL config before building command
        is_valid, error = self._validate_online_rl_config(config)
        if not is_valid:
            raise ValueError(f"Online RL validation failed: {error}")
        
        # Use locked output path from capabilities (ignores user-provided path when locked)
        from ..core.capabilities import get_validator
        validator = get_validator()
        user_output_dir = getattr(config, 'output_dir', '') or ''
        output_dir = validator.get_output_path(job_id, user_output_dir)
        
        # Determine CLI command based on training method
        training_method = getattr(config, 'training_method', None)
        if training_method:
            method_value = training_method.value if hasattr(training_method, 'value') else str(training_method)
        else:
            method_value = "sft"  # Default to SFT
        
        # Map train_type to USF BIOS compatible value
        # QLoRA is LoRA with quantization - USF BIOS doesn't have "qlora" train_type
        train_type_value = config.train_type.value
        if train_type_value == "qlora":
            train_type_value = "lora"  # QLoRA is LoRA + quant_bits
        
        # Handle multiple datasets - split comma-separated paths
        # USF BIOS expects each dataset as separate --dataset argument
        dataset_paths = [p.strip() for p in config.dataset_path.split(',') if p.strip()]
        
        cmd = [
            sys.executable, "-m", "usf_bios", method_value,
            "--model", config.model_path,
            "--train_type", train_type_value,
            "--dataset", dataset_paths[0] if dataset_paths else config.dataset_path,
            "--output_dir", output_dir,
            "--num_train_epochs", str(config.num_train_epochs),
            "--learning_rate", str(config.learning_rate),
            "--per_device_train_batch_size", str(config.per_device_train_batch_size),
            "--gradient_accumulation_steps", str(config.gradient_accumulation_steps),
            "--max_length", str(config.max_length),
            "--torch_dtype", config.torch_dtype,
            "--logging_steps", "1",
            "--report_to", "tensorboard",
        ]
        
        # Add additional datasets if multiple were provided
        # Each additional dataset gets its own --dataset argument
        for additional_ds in dataset_paths[1:]:
            cmd.extend(["--dataset", additional_ds])
        
        # RLHF specific parameters
        if method_value == "rlhf":
            rlhf_type = getattr(config, 'rlhf_type', None)
            if rlhf_type:
                cmd.extend(["--rlhf_type", rlhf_type])
            
            # Column mapping for offline RLHF datasets (prompt/chosen/rejected format)
            # Maps: chosen -> response, rejected -> rejected_response
            # This is required for USF BIOS to properly process RLHF preference datasets
            offline_rlhf_algorithms = ["dpo", "orpo", "simpo", "cpo", "rm", "kto"]
            if rlhf_type in offline_rlhf_algorithms:
                import json
                columns_mapping = {
                    "chosen": "response",
                    "rejected": "rejected_response"
                }
                cmd.extend(["--columns", json.dumps(columns_mapping)])
            
            # Beta parameter
            if hasattr(config, 'beta') and config.beta is not None:
                cmd.extend(["--beta", str(config.beta)])
            
            # Max completion length for GRPO/PPO/GKD
            if hasattr(config, 'max_completion_length') and config.max_completion_length:
                cmd.extend(["--max_completion_length", str(config.max_completion_length)])
            
            # DPO specific
            if rlhf_type == "dpo":
                if hasattr(config, 'label_smoothing') and config.label_smoothing > 0:
                    cmd.extend(["--label_smoothing", str(config.label_smoothing)])
                if hasattr(config, 'rpo_alpha') and config.rpo_alpha is not None:
                    cmd.extend(["--rpo_alpha", str(config.rpo_alpha)])
            
            # SimPO specific
            elif rlhf_type == "simpo":
                if hasattr(config, 'simpo_gamma') and config.simpo_gamma:
                    cmd.extend(["--simpo_gamma", str(config.simpo_gamma)])
            
            # KTO specific
            elif rlhf_type == "kto":
                if hasattr(config, 'desirable_weight') and config.desirable_weight != 1.0:
                    cmd.extend(["--desirable_weight", str(config.desirable_weight)])
                if hasattr(config, 'undesirable_weight') and config.undesirable_weight != 1.0:
                    cmd.extend(["--undesirable_weight", str(config.undesirable_weight)])
            
            # PPO specific
            elif rlhf_type == "ppo":
                if hasattr(config, 'num_ppo_epochs') and config.num_ppo_epochs:
                    cmd.extend(["--num_ppo_epochs", str(config.num_ppo_epochs)])
                if hasattr(config, 'kl_coef') and config.kl_coef:
                    cmd.extend(["--kl_coef", str(config.kl_coef)])
                if hasattr(config, 'cliprange') and config.cliprange:
                    cmd.extend(["--cliprange", str(config.cliprange)])
            
            # GRPO specific
            elif rlhf_type == "grpo":
                if hasattr(config, 'num_generations') and config.num_generations:
                    cmd.extend(["--num_generations", str(config.num_generations)])
            
            # Online RL vLLM configuration (GRPO/PPO/GKD)
            if rlhf_type in ["grpo", "ppo", "gkd"]:
                # vLLM usage
                if hasattr(config, 'use_vllm') and config.use_vllm:
                    cmd.extend(["--use_vllm", "true"])
                    
                    # vLLM mode (colocate or server)
                    if hasattr(config, 'vllm_mode') and config.vllm_mode:
                        cmd.extend(["--vllm_mode", config.vllm_mode])
                        
                        # Server mode specific
                        if config.vllm_mode == "server":
                            if hasattr(config, 'vllm_server_host') and config.vllm_server_host:
                                cmd.extend(["--vllm_server_host", config.vllm_server_host])
                            if hasattr(config, 'vllm_server_port') and config.vllm_server_port:
                                cmd.extend(["--vllm_server_port", str(config.vllm_server_port)])
                        
                        # Colocate mode specific
                        elif config.vllm_mode == "colocate":
                            if hasattr(config, 'vllm_tensor_parallel_size') and config.vllm_tensor_parallel_size > 1:
                                cmd.extend(["--vllm_tensor_parallel_size", str(config.vllm_tensor_parallel_size)])
                            if hasattr(config, 'vllm_gpu_memory_utilization') and config.vllm_gpu_memory_utilization:
                                cmd.extend(["--vllm_gpu_memory_utilization", str(config.vllm_gpu_memory_utilization)])
                            
                            # Memory optimization options
                            if hasattr(config, 'offload_model') and config.offload_model:
                                cmd.extend(["--offload_model", "true"])
                            if hasattr(config, 'offload_optimizer') and config.offload_optimizer:
                                cmd.extend(["--offload_optimizer", "true"])
                            if hasattr(config, 'sleep_level') and config.sleep_level > 0:
                                cmd.extend(["--sleep_level", str(config.sleep_level)])
                
                # Reward functions for GRPO
                if rlhf_type == "grpo" and hasattr(config, 'reward_funcs') and config.reward_funcs:
                    cmd.extend(["--reward_funcs", ",".join(config.reward_funcs)])
        
        # Warmup ratio
        if hasattr(config, 'warmup_ratio') and config.warmup_ratio is not None:
            cmd.extend(["--warmup_ratio", str(config.warmup_ratio)])
        
        # Resume from checkpoint
        if resume_from_checkpoint:
            cmd.extend(["--resume_from_checkpoint", resume_from_checkpoint])
        
        # Existing adapter - continue training on an existing LoRA/QLoRA adapter
        # USF BIOS uses --adapters argument to load existing adapters for continued training
        existing_adapter = getattr(config, 'existing_adapter_path', None)
        if existing_adapter and train_type_value in ["lora", "qlora", "adalora"]:
            cmd.extend(["--adapters", existing_adapter])
        
        # LoRA parameters (LoRA, QLoRA, AdaLoRA all use lora params)
        if train_type_value in ["lora", "adalora"]:
            cmd.extend([
                "--lora_rank", str(config.lora_rank),
                "--lora_alpha", str(config.lora_alpha),
                "--lora_dropout", str(config.lora_dropout),
                "--target_modules", config.target_modules,
            ])
            
            # ============================================================
            # ADVANCED LORA OPTIONS (from USF-BIOS tuner_args.py)
            # ============================================================
            # RSLoRA - Rank-Stabilized LoRA for better training stability
            if hasattr(config, 'use_rslora') and config.use_rslora:
                cmd.extend(["--use_rslora", "true"])
            
            # DoRA - Weight-Decomposed LoRA for improved performance
            if hasattr(config, 'use_dora') and config.use_dora:
                cmd.extend(["--use_dora", "true"])
            
            # LoRA bias training
            if hasattr(config, 'lora_bias') and config.lora_bias and config.lora_bias != "none":
                cmd.extend(["--lora_bias", config.lora_bias])
            
            # LoRA initialization method (PiSSA, OLoRA, LoftQ, etc.)
            if hasattr(config, 'init_weights') and config.init_weights and config.init_weights != "true":
                cmd.extend(["--init_weights", config.init_weights])
        
        # Quantization (QLoRA uses quant_bits=4 by default)
        # If train_type was "qlora", ensure quant_bits is set
        if config.train_type.value == "qlora":
            quant_bits = config.quant_bits if config.quant_bits else 4
            cmd.extend(["--quant_bits", str(quant_bits)])
        elif config.quant_bits:
            cmd.extend(["--quant_bits", str(config.quant_bits)])
        
        # ============================================================
        # MULTIMODAL FREEZE OPTIONS (from USF-BIOS tuner_args.py)
        # For VLMs (vision-language models) like LLaVA, Qwen-VL
        # ============================================================
        modality = getattr(config, 'modality', None)
        modality_value = modality.value if hasattr(modality, 'value') else str(modality) if modality else "text"
        
        if modality_value in ["vision", "audio", "video"]:
            # Freeze LLM weights (train only vision/aligner)
            if hasattr(config, 'freeze_llm') and config.freeze_llm:
                cmd.extend(["--freeze_llm", "true"])
            
            # Freeze Vision Transformer weights
            if hasattr(config, 'freeze_vit') and config.freeze_vit:
                cmd.extend(["--freeze_vit", "true"])
            
            # Freeze aligner/projector weights
            if hasattr(config, 'freeze_aligner') and config.freeze_aligner:
                cmd.extend(["--freeze_aligner", "true"])
        
        # ============================================================
        # LONG CONTEXT SUPPORT (from USF-BIOS model_args.py)
        # RoPE scaling for extended context
        # ============================================================
        if hasattr(config, 'rope_scaling') and config.rope_scaling:
            cmd.extend(["--rope_scaling", config.rope_scaling])
        
        if hasattr(config, 'max_model_len') and config.max_model_len:
            cmd.extend(["--max_model_len", str(config.max_model_len)])
        
        # ============================================================
        # DATASET SPLIT (from USF-BIOS data_args.py)
        # Auto train/val split
        # ============================================================
        if hasattr(config, 'split_dataset_ratio') and config.split_dataset_ratio and config.split_dataset_ratio > 0:
            cmd.extend(["--split_dataset_ratio", str(config.split_dataset_ratio)])
        
        # Attention Implementation (Flash Attention, SDPA, etc.)
        if config.attn_impl:
            cmd.extend(["--attn_impl", config.attn_impl])
        
        # Gradient Checkpointing
        if hasattr(config, 'gradient_checkpointing') and config.gradient_checkpointing:
            cmd.extend(["--gradient_checkpointing", "true"])
        
        # Liger Kernel (triton-based optimizations)
        if hasattr(config, 'use_liger_kernel') and config.use_liger_kernel:
            cmd.extend(["--use_liger_kernel", "true"])
        
        # Packing (combine short sequences)
        if hasattr(config, 'packing') and config.packing:
            cmd.extend(["--packing", "true"])
        
        # Sequence Parallelism
        if hasattr(config, 'sequence_parallel_size') and config.sequence_parallel_size > 1:
            cmd.extend(["--sequence_parallel_size", str(config.sequence_parallel_size)])
        
        # Learning Rate Scheduler
        if hasattr(config, 'lr_scheduler_type') and config.lr_scheduler_type:
            cmd.extend(["--lr_scheduler_type", config.lr_scheduler_type])
        
        # Optimizer parameters
        if hasattr(config, 'weight_decay') and config.weight_decay is not None:
            cmd.extend(["--weight_decay", str(config.weight_decay)])
        if hasattr(config, 'adam_beta1') and config.adam_beta1 is not None:
            cmd.extend(["--adam_beta1", str(config.adam_beta1)])
        if hasattr(config, 'adam_beta2') and config.adam_beta2 is not None:
            cmd.extend(["--adam_beta2", str(config.adam_beta2)])
        if hasattr(config, 'max_grad_norm') and config.max_grad_norm is not None and config.max_grad_norm > 0:
            cmd.extend(["--max_grad_norm", str(config.max_grad_norm)])
        
        # DeepSpeed (cannot be used with FSDP)
        if config.deepspeed:
            cmd.extend(["--deepspeed", config.deepspeed])
        
        # FSDP (cannot be used with DeepSpeed)
        elif config.fsdp:
            cmd.extend(["--fsdp", config.fsdp])
        
        # Early stopping
        if config.early_stop_interval:
            cmd.extend(["--early_stop_interval", str(config.early_stop_interval)])
        
        # ============================================================
        # STREAMING OPTIONS FOR LARGE DATASETS
        # Verified against USF-BIOS get_dataset_kwargs() in data_args.py
        # ============================================================
        if getattr(config, 'streaming', False):
            cmd.extend(["--streaming", "true"])
            
            # Shuffle buffer size for streaming
            if hasattr(config, 'shuffle_buffer_size') and config.shuffle_buffer_size:
                cmd.extend(["--shuffle_buffer_size", str(config.shuffle_buffer_size)])
            
            # max_steps is REQUIRED for streaming mode (dataset length unknown)
            if hasattr(config, 'max_steps') and config.max_steps:
                cmd.extend(["--max_steps", str(config.max_steps)])
        
        # ============================================================
        # MULTIPLE DATASET MIXING OPTIONS
        # Verified against USF-BIOS get_dataset_kwargs() in data_args.py
        # ============================================================
        # Interleave probabilities for multiple datasets
        if hasattr(config, 'interleave_prob') and config.interleave_prob:
            import json
            cmd.extend(["--interleave_prob", json.dumps(config.interleave_prob)])
        
        # Stopping strategy for multiple datasets
        if hasattr(config, 'stopping_strategy') and config.stopping_strategy:
            cmd.extend(["--stopping_strategy", config.stopping_strategy])
        
        # HuggingFace source flag - required when model/dataset is from HuggingFace
        model_source = getattr(config, 'model_source', None)
        if model_source:
            source_value = model_source.value if hasattr(model_source, 'value') else str(model_source)
            if source_value == 'huggingface':
                cmd.extend(["--use_hf", "true"])
        
        # Note: API tokens (hf_token, ms_token) are passed via environment variables
        # (HF_TOKEN, MODELSCOPE_API_TOKEN) for security - see _run_job method
        
        # Evaluation
        if config.eval_strategy:
            cmd.extend(["--eval_strategy", config.eval_strategy])
        if config.eval_steps:
            cmd.extend(["--eval_steps", str(config.eval_steps)])
        if config.save_steps:
            cmd.extend(["--save_steps", str(config.save_steps)])
        if hasattr(config, 'save_total_limit') and config.save_total_limit is not None:
            cmd.extend(["--save_total_limit", str(config.save_total_limit)])
        
        return cmd, output_dir
    
    def _parse_log_line(self, line: str, job_id: str) -> dict:
        """Parse a log line to extract training metrics.
        
        Supports ALL training algorithms:
        - SFT: loss, learning_rate, grad_norm, epoch
        - RLHF/PPO: reward, kl_divergence, policy_loss, value_loss, entropy
        - DPO: chosen_rewards, rejected_rewards, reward_margin
        - GRPO: reward, kl_penalty, policy_gradient_loss
        - GKD: distillation_loss, student_loss, teacher_loss
        - Pre-training: loss, perplexity, tokens_per_second
        - ASR: wer, cer, loss
        - TTS: mel_loss, duration_loss, pitch_loss
        - Multimodal: image_loss, text_loss, contrastive_loss
        """
        metrics = {}
        
        # ============================================================
        # COMMON METRICS (all training types)
        # ============================================================
        
        # Parse loss (multiple formats)
        loss_match = re.search(r"'loss':\s*([\d.]+)", line)
        if loss_match:
            metrics["loss"] = float(loss_match.group(1))
        else:
            loss_alt = re.search(r"(?:loss|Loss)[=:\s]+([\d.]+)", line)
            if loss_alt:
                metrics["loss"] = float(loss_alt.group(1))
        
        # Parse learning rate
        lr_match = re.search(r"'learning_rate':\s*([\d.e\-+]+)", line)
        if lr_match:
            metrics["learning_rate"] = float(lr_match.group(1))
        else:
            lr_alt = re.search(r"(?:lr|LR|learning_rate)[=:\s]+([\d.e\-+]+)", line)
            if lr_alt:
                metrics["learning_rate"] = float(lr_alt.group(1))
        
        # Parse step - supports multiple formats
        # Format 1: 'global_step/max_steps': '2/3' (USF BIOS format)
        step_max_match = re.search(r"'global_step/max_steps':\s*'(\d+)/(\d+)'", line)
        if step_max_match:
            metrics["step"] = int(step_max_match.group(1))
            metrics["total_steps"] = int(step_max_match.group(2))
        else:
            # Format 2: 'step': 2 or 'global_step': 2
            step_match = re.search(r"'(global_)?step':\s*(\d+)", line)
            if step_match:
                metrics["step"] = int(step_match.group(2))
            else:
                # Format 3: Step 2 or step=2
                step_alt = re.search(r"(?:Step|step)[=:\s]+(\d+)", line)
                if step_alt:
                    metrics["step"] = int(step_alt.group(1))
        
        # Parse epoch
        epoch_match = re.search(r"'epoch':\s*([\d.]+)", line)
        if epoch_match:
            metrics["epoch"] = float(epoch_match.group(1))
        else:
            epoch_alt = re.search(r"(?:Epoch|epoch)[=:\s]+([\d.]+)", line)
            if epoch_alt:
                metrics["epoch"] = float(epoch_alt.group(1))
        
        # Parse grad_norm
        grad_match = re.search(r"'grad_norm':\s*([\d.]+)", line)
        if grad_match:
            metrics["grad_norm"] = float(grad_match.group(1))
        
        # Parse eval_loss
        eval_loss_match = re.search(r"'eval_loss':\s*([\d.]+)", line)
        if eval_loss_match:
            metrics["eval_loss"] = float(eval_loss_match.group(1))
        
        # ============================================================
        # RLHF / PPO / GRPO METRICS
        # ============================================================
        
        # Reward
        reward_match = re.search(r"'reward':\s*([\d.\-e+]+)", line)
        if reward_match:
            metrics["reward"] = float(reward_match.group(1))
        
        # KL divergence
        kl_match = re.search(r"'kl(?:_divergence)?':\s*([\d.\-e+]+)", line)
        if kl_match:
            metrics["kl_divergence"] = float(kl_match.group(1))
        
        # Policy loss
        policy_loss_match = re.search(r"'policy_loss':\s*([\d.\-e+]+)", line)
        if policy_loss_match:
            metrics["policy_loss"] = float(policy_loss_match.group(1))
        
        # Value loss
        value_loss_match = re.search(r"'value_loss':\s*([\d.\-e+]+)", line)
        if value_loss_match:
            metrics["value_loss"] = float(value_loss_match.group(1))
        
        # Entropy
        entropy_match = re.search(r"'entropy':\s*([\d.\-e+]+)", line)
        if entropy_match:
            metrics["entropy"] = float(entropy_match.group(1))
        
        # ============================================================
        # DPO METRICS
        # ============================================================
        
        # Chosen rewards
        chosen_match = re.search(r"'(?:chosen_rewards|rewards/chosen)':\s*([\d.\-e+]+)", line)
        if chosen_match:
            metrics["chosen_rewards"] = float(chosen_match.group(1))
        
        # Rejected rewards
        rejected_match = re.search(r"'(?:rejected_rewards|rewards/rejected)':\s*([\d.\-e+]+)", line)
        if rejected_match:
            metrics["rejected_rewards"] = float(rejected_match.group(1))
        
        # Reward margin
        margin_match = re.search(r"'(?:reward_margin|rewards/margin)':\s*([\d.\-e+]+)", line)
        if margin_match:
            metrics["reward_margin"] = float(margin_match.group(1))
        
        # ============================================================
        # EXTRA METRICS (stored in JSON column)
        # ============================================================
        extra = {}
        
        # Perplexity (pre-training)
        ppl_match = re.search(r"'perplexity':\s*([\d.]+)", line)
        if ppl_match:
            extra["perplexity"] = float(ppl_match.group(1))
        
        # WER/CER (ASR)
        wer_match = re.search(r"'wer':\s*([\d.]+)", line)
        if wer_match:
            extra["wer"] = float(wer_match.group(1))
        cer_match = re.search(r"'cer':\s*([\d.]+)", line)
        if cer_match:
            extra["cer"] = float(cer_match.group(1))
        
        # Mel loss (TTS)
        mel_match = re.search(r"'mel_loss':\s*([\d.]+)", line)
        if mel_match:
            extra["mel_loss"] = float(mel_match.group(1))
        
        # Contrastive loss (multimodal)
        contrastive_match = re.search(r"'contrastive_loss':\s*([\d.]+)", line)
        if contrastive_match:
            extra["contrastive_loss"] = float(contrastive_match.group(1))
        
        # Approx KL (PPO)
        approx_kl_match = re.search(r"'approx_kl':\s*([\d.\-e+]+)", line)
        if approx_kl_match:
            extra["approx_kl"] = float(approx_kl_match.group(1))
        
        # Clip fraction (PPO)
        clip_match = re.search(r"'clip(?:_fraction)?':\s*([\d.]+)", line)
        if clip_match:
            extra["clip_fraction"] = float(clip_match.group(1))
        
        if extra:
            metrics["extra_metrics"] = extra
        
        # ============================================================
        # PROGRESS BAR FORMAT
        # ============================================================
        progress_match = re.search(r"(\d+)/(\d+)\s*\[", line)
        if progress_match:
            metrics["step"] = int(progress_match.group(1))
            metrics["total_steps"] = int(progress_match.group(2))
        
        return metrics
    
    async def run_training(self, job_id: str) -> None:
        """Run the training process with comprehensive step-by-step logging."""
        _log_step(job_id, "TRAINING_START", {"action": "run_training called"})
        
        # STEP 1: IMMEDIATELY write to terminal log - this should ALWAYS appear
        sanitized_log_service.create_terminal_log(job_id, "[Step 1/12] Training task started - preparing environment...", "INFO")
        
        # STEP 2: Load job from job manager
        sanitized_log_service.create_terminal_log(job_id, "[Step 2/12] Loading job configuration...", "INFO")
        _log_step(job_id, "LOAD_JOB", {"action": "fetching job from job_manager"})
        job = await job_manager.get_job(job_id)
        if not job:
            _log_step(job_id, "LOAD_JOB_FAILED", {"error": "Job not found in job_manager"}, "ERROR")
            sanitized_log_service.create_terminal_log(job_id, "ERROR: Job not found in system - training cannot proceed. Job may have been deleted.", "ERROR")
            return
        sanitized_log_service.create_terminal_log(job_id, f"[Step 2/12] ✓ Job loaded: {job.name}", "INFO")
        
        # Log complete job configuration
        _log_step(job_id, "JOB_CONFIG", {
            "job_id": job_id,
            "name": job.name,
            "model_path": job.config.model_path,
            "model_source": str(getattr(job.config, 'model_source', 'local')),
            "dataset_path": job.config.dataset_path,
            "train_type": str(job.config.train_type),
            "training_method": str(getattr(job.config, 'training_method', 'sft')),
            "num_epochs": job.config.num_train_epochs,
            "learning_rate": job.config.learning_rate,
            "batch_size": job.config.per_device_train_batch_size,
            "max_length": job.config.max_length,
            "lora_rank": getattr(job.config, 'lora_rank', None),
            "lora_alpha": getattr(job.config, 'lora_alpha', None),
            "deepspeed": getattr(job.config, 'deepspeed', None),
            "fsdp": getattr(job.config, 'fsdp', None),
        })
        
        # ============================================================
        # STEP 3: PRE-TRAINING GPU CLEANUP
        # ============================================================
        sanitized_log_service.create_terminal_log(job_id, "[Step 3/12] Cleaning GPU memory from previous runs...", "INFO")
        _log_step(job_id, "GPU_CLEANUP_START", {"phase": "pre-training"})
        try:
            cleanup_result = await gpu_cleanup_service.async_deep_cleanup(
                kill_orphans=True,
                exclude_pids=[os.getpid()]
            )
            _log_step(job_id, "GPU_CLEANUP_COMPLETE", {
                "success": cleanup_result.get("success"),
                "memory_freed_gb": cleanup_result.get("memory_freed_gb", 0),
                "memory_before": cleanup_result.get("memory_before", {}),
                "memory_after": cleanup_result.get("memory_after", {}),
            })
            freed_gb = cleanup_result.get("memory_freed_gb", 0)
            if freed_gb > 0:
                sanitized_log_service.create_terminal_log(job_id, f"[Step 3/12] ✓ GPU cleanup freed {freed_gb:.1f}GB VRAM", "INFO")
            else:
                sanitized_log_service.create_terminal_log(job_id, "[Step 3/12] ✓ GPU memory is clean", "INFO")
        except Exception as cleanup_err:
            _log_step(job_id, "GPU_CLEANUP_ERROR", {"error": str(cleanup_err)}, "WARN")
            sanitized_log_service.create_terminal_log(job_id, f"[Step 3/12] Warning: GPU cleanup failed ({str(cleanup_err)[:50]}), continuing anyway...", "WARN")
        
        try:
            # ============================================================
            # STEP 4: UPDATE JOB STATUS TO INITIALIZING
            # ============================================================
            sanitized_log_service.create_terminal_log(job_id, "[Step 4/12] Setting job status to INITIALIZING...", "INFO")
            _log_step(job_id, "STATUS_UPDATE", {"status": "INITIALIZING"})
            await job_manager.update_job(job_id, 
                status=JobStatus.INITIALIZING,
                started_at=datetime.now()
            )
            await ws_manager.send_status(job_id, "initializing")
            sanitized_log_service.create_terminal_log(job_id, "[Step 4/12] \u2713 Job status updated", "INFO")
            
            # ============================================================
            # STEP 5: VALIDATE MODEL AND DATASET PATHS
            # ============================================================
            sanitized_log_service.create_terminal_log(job_id, "[Step 5/12] Validating model and dataset paths...", "INFO")
            _log_step(job_id, "VALIDATION_START", {"phase": "pre-training validation"})
            validation_errors = []
            
            # Check model path exists (for local models)
            model_source = getattr(job.config.model_source, 'value', str(job.config.model_source))
            _log_step(job_id, "VALIDATION_MODEL", {
                "model_source": model_source,
                "model_path": job.config.model_path,
            })
            model_path = job.config.model_path
            if model_source == 'local':
                if not os.path.exists(model_path):
                    validation_errors.append(f"Model path does not exist: {model_path}")
                elif not os.path.isdir(model_path):
                    validation_errors.append(f"Model path is not a directory: {model_path}")
                else:
                    # Check for config.json which indicates a valid model directory
                    config_path = os.path.join(model_path, "config.json")
                    if not os.path.exists(config_path):
                        sanitized_log_service.create_terminal_log(job_id, "Warning: config.json not found in model directory", "WARN")
                        await ws_manager.send_log(job_id, f"Warning: config.json not found in model directory")
                        _debug_log(job_id, f"Warning: config.json not found at {config_path}")
            
            # Check dataset paths exist
            dataset_paths = job.config.dataset_path.split(',') if job.config.dataset_path else []
            for ds_path in dataset_paths:
                ds_path = ds_path.strip()
                if ds_path and not ds_path.upper().startswith(('HF::', 'MS::')):
                    if not os.path.exists(ds_path):
                        validation_errors.append(f"Dataset path does not exist: {ds_path}")
            
            # ============================================================
            # OPTIMIZATION COMPATIBILITY VALIDATION
            # Check for incompatible combinations before training starts
            # ============================================================
            config = job.config
            
            # DeepSpeed + FSDP conflict
            if getattr(config, 'deepspeed', None) and getattr(config, 'fsdp', None):
                validation_errors.append(
                    "DeepSpeed and FSDP cannot be used together. "
                    "They are both distributed training frameworks. Please disable one."
                )
            
            # Packing requires Flash Attention
            if getattr(config, 'packing', False):
                attn_impl = getattr(config, 'attn_impl', None)
                flash_attn_values = ['flash_attn', 'flash_attention_2', 'flash_attention_3']
                if attn_impl and attn_impl not in flash_attn_values:
                    validation_errors.append(
                        f"Sequence Packing requires Flash Attention but '{attn_impl}' is selected. "
                        "Please enable Flash Attention 2 or 3, or disable Packing."
                    )
            
            # Flash Attention 3 requires Hopper GPU
            attn_impl = getattr(config, 'attn_impl', None)
            if attn_impl == 'flash_attention_3':
                try:
                    import torch
                    if torch.cuda.is_available():
                        props = torch.cuda.get_device_properties(0)
                        if props.major < 9:  # Hopper is sm_90 (major=9)
                            validation_errors.append(
                                f"Flash Attention 3 requires Hopper GPU (H100/H200) but detected "
                                f"{props.name} (compute capability {props.major}.{props.minor}). "
                                "Please use Flash Attention 2 or SDPA instead."
                            )
                except Exception as e:
                    _debug_log(job_id, f"GPU detection error: {e}", "WARN")
            
            # Liger Kernel + Packing warning (not error, just log)
            if getattr(config, 'use_liger_kernel', False) and getattr(config, 'packing', False):
                sanitized_log_service.create_terminal_log(
                    job_id, 
                    "Warning: Liger Kernel may have issues with Sequence Packing. Proceeding anyway.", 
                    "WARN"
                )
                await ws_manager.send_log(job_id, "Warning: Liger Kernel + Packing combination may be unstable")
            
            # Report validation errors
            if validation_errors:
                error_msg = "Pre-training validation failed:\n" + "\n".join(f"  - {e}" for e in validation_errors)
                _log_step(job_id, "VALIDATION_FAILED", {
                    "errors": validation_errors,
                    "error_count": len(validation_errors),
                }, "ERROR")
                sanitized_log_service.create_terminal_log(job_id, f"[Step 5/12] FAILED - Validation errors found", "ERROR")
                sanitized_log_service.create_terminal_log(job_id, f"ERROR: {error_msg}", "ERROR")
                await ws_manager.send_log(job_id, f"ERROR: {error_msg}")
                await job_manager.add_log(job_id, f"ERROR: {error_msg}")
                
                await job_manager.update_job(job_id, 
                    status=JobStatus.FAILED,
                    error=error_msg,
                    completed_at=datetime.now()
                )
                await ws_manager.send_status(job_id, "failed")
                return
            
            _log_step(job_id, "VALIDATION_PASSED", {"phase": "pre-training validation complete"})
            sanitized_log_service.create_terminal_log(job_id, "[Step 5/12] \u2713 Model and dataset paths validated", "INFO")
            
            # ============================================================
            # STEP 6: STORAGE SPACE VALIDATION (Accurate, Parameter-Based)
            # ============================================================
            sanitized_log_service.create_terminal_log(job_id, "[Step 6/12] Calculating storage requirements...", "INFO")
            _log_step(job_id, "STORAGE_VALIDATION_START", {"phase": "accurate storage calculation"})
            
            try:
                import tempfile
                import traceback as tb
                
                merge_enabled = getattr(config, 'merge_adapter_before_training', False)
                adapter_base_path = getattr(config, 'adapter_base_model_path', None)
                adapter_base_source = getattr(config, 'adapter_base_model_source', None)
                if adapter_base_source and hasattr(adapter_base_source, 'value'):
                    adapter_base_source = adapter_base_source.value
                else:
                    adapter_base_source = str(adapter_base_source) if adapter_base_source else 'local'
                output_dir = getattr(config, 'output_dir', 'output') or 'output'
                
                # Get existing adapter settings (for continuing training on adapter)
                existing_adapter_path = getattr(config, 'existing_adapter_path', None)
                existing_adapter_source = getattr(config, 'existing_adapter_source', None)
                if existing_adapter_source and hasattr(existing_adapter_source, 'value'):
                    existing_adapter_source = existing_adapter_source.value
                else:
                    existing_adapter_source = str(existing_adapter_source) if existing_adapter_source else 'local'
                
                # ============================================================
                # VALIDATE EXISTING ADAPTER (if provided)
                # Only allowed for LoRA/QLoRA training on supported models
                # ============================================================
                if existing_adapter_path:
                    train_type_val = config.train_type.value if hasattr(config.train_type, 'value') else str(config.train_type)
                    
                    # Block adapter for full fine-tuning (doesn't make sense)
                    if train_type_val == 'full':
                        error_msg = "Cannot use existing adapter with full fine-tuning. Use LoRA or QLoRA instead."
                        _log_step(job_id, "ADAPTER_VALIDATION_FAILED", {"error": error_msg}, "ERROR")
                        sanitized_log_service.create_terminal_log(job_id, f"[Step 6/12] FAILED - {error_msg}", "ERROR")
                        await job_manager.update_job(job_id, status=JobStatus.FAILED, error=error_msg, completed_at=datetime.now())
                        await ws_manager.send_status(job_id, "failed")
                        return
                    
                    # Validate adapter path exists (for local)
                    if existing_adapter_source == 'local':
                        if not os.path.isdir(existing_adapter_path):
                            error_msg = f"Existing adapter path does not exist: {existing_adapter_path}"
                            _log_step(job_id, "ADAPTER_NOT_FOUND", {"error": error_msg}, "ERROR")
                            sanitized_log_service.create_terminal_log(job_id, f"[Step 6/12] FAILED - {error_msg}", "ERROR")
                            await job_manager.update_job(job_id, status=JobStatus.FAILED, error=error_msg, completed_at=datetime.now())
                            await ws_manager.send_status(job_id, "failed")
                            return
                        
                        # Check adapter_config.json exists
                        adapter_config_file = os.path.join(existing_adapter_path, 'adapter_config.json')
                        if not os.path.isfile(adapter_config_file):
                            error_msg = f"Not a valid adapter: missing adapter_config.json in {existing_adapter_path}"
                            _log_step(job_id, "INVALID_ADAPTER", {"error": error_msg}, "ERROR")
                            sanitized_log_service.create_terminal_log(job_id, f"[Step 6/12] FAILED - {error_msg}", "ERROR")
                            await job_manager.update_job(job_id, status=JobStatus.FAILED, error=error_msg, completed_at=datetime.now())
                            await ws_manager.send_status(job_id, "failed")
                            return
                        
                        # Check adapter has content (size > 0)
                        adapter_size = _get_dir_size_gb_fast(existing_adapter_path, job_id, timeout_seconds=5)
                        if adapter_size <= 0:
                            error_msg = f"Adapter appears to be empty or inaccessible: {existing_adapter_path}"
                            _log_step(job_id, "EMPTY_ADAPTER", {"error": error_msg}, "ERROR")
                            sanitized_log_service.create_terminal_log(job_id, f"[Step 6/12] FAILED - {error_msg}", "ERROR")
                            await job_manager.update_job(job_id, status=JobStatus.FAILED, error=error_msg, completed_at=datetime.now())
                            await ws_manager.send_status(job_id, "failed")
                            return
                        
                        _log_step(job_id, "ADAPTER_VALIDATED", {
                            "adapter_path": existing_adapter_path,
                            "size_gb": adapter_size,
                        })
                        sanitized_log_service.create_terminal_log(job_id, f"[Step 6/12] ✓ Existing adapter validated", "INFO")
                    
                    elif existing_adapter_source in ('huggingface', 'modelscope'):
                        # Download adapter from remote
                        sanitized_log_service.create_terminal_log(
                            job_id, f"[Step 6/12] Downloading existing adapter from {existing_adapter_source}...", "INFO"
                        )
                        adapter_local_path = _download_remote_model(
                            existing_adapter_path, existing_adapter_source, job_id=job_id
                        )
                        if adapter_local_path and os.path.isdir(adapter_local_path):
                            existing_adapter_path = adapter_local_path
                            config.existing_adapter_path = adapter_local_path
                            _log_step(job_id, "ADAPTER_DOWNLOADED", {"local_path": adapter_local_path})
                            sanitized_log_service.create_terminal_log(job_id, f"[Step 6/12] ✓ Adapter downloaded", "INFO")
                        else:
                            error_msg = f"Failed to download adapter from {existing_adapter_source}"
                            _log_step(job_id, "ADAPTER_DOWNLOAD_FAILED", {"error": error_msg}, "ERROR")
                            sanitized_log_service.create_terminal_log(job_id, f"[Step 6/12] FAILED - {error_msg}", "ERROR")
                            await job_manager.update_job(job_id, status=JobStatus.FAILED, error=error_msg, completed_at=datetime.now())
                            await ws_manager.send_status(job_id, "failed")
                            return
                
                # ============================================================
                # HANDLE BASE MODEL FOR ADAPTER MERGE (can be different source)
                # Base model and adapter can be from different sources:
                # - Adapter (model_path): local, HuggingFace, or ModelScope
                # - Base model (adapter_base_path): local, HuggingFace, or ModelScope
                # ============================================================
                if merge_enabled and adapter_base_path:
                    if adapter_base_source in ('huggingface', 'modelscope'):
                        # Download base model if it's remote
                        sanitized_log_service.create_terminal_log(
                            job_id, f"[Step 6/12] Downloading base model from {adapter_base_source}...", "INFO"
                        )
                        _log_step(job_id, "DOWNLOADING_BASE_MODEL", {
                            "base_model_id": adapter_base_path,
                            "source": adapter_base_source,
                        })
                        
                        base_local_path = _download_remote_model(
                            adapter_base_path, adapter_base_source, job_id=job_id
                        )
                        
                        if base_local_path and os.path.isdir(base_local_path):
                            _log_step(job_id, "BASE_MODEL_DOWNLOADED", {
                                "original_id": adapter_base_path,
                                "local_path": base_local_path,
                            })
                            sanitized_log_service.create_terminal_log(
                                job_id, f"[Step 6/12] Base model downloaded successfully", "INFO"
                            )
                            # Update to use local path
                            adapter_base_path = base_local_path
                            job.config.adapter_base_model_path = base_local_path
                        else:
                            error_msg = f"Failed to download base model from {adapter_base_source}: {adapter_base_path}"
                            _log_step(job_id, "BASE_MODEL_DOWNLOAD_FAILED", {
                                "error": error_msg,
                            }, "ERROR")
                            sanitized_log_service.create_terminal_log(job_id, f"[Step 6/12] FAILED - {error_msg}", "ERROR")
                            await job_manager.update_job(job_id,
                                status=JobStatus.FAILED,
                                error=error_msg,
                                completed_at=datetime.now()
                            )
                            await ws_manager.send_status(job_id, "failed")
                            return
                    elif adapter_base_source == 'local':
                        # Validate local base model path exists
                        if not os.path.isdir(adapter_base_path):
                            error_msg = f"Base model path does not exist: {adapter_base_path}"
                            _log_step(job_id, "BASE_MODEL_NOT_FOUND", {
                                "error": error_msg,
                                "path": adapter_base_path,
                            }, "ERROR")
                            sanitized_log_service.create_terminal_log(job_id, f"[Step 6/12] FAILED - {error_msg}", "ERROR")
                            await job_manager.update_job(job_id,
                                status=JobStatus.FAILED,
                                error=error_msg,
                                completed_at=datetime.now()
                            )
                            await ws_manager.send_status(job_id, "failed")
                            return
                
                # Calculate accurate storage requirements based on actual model size
                sanitized_log_service.create_terminal_log(job_id, "[Step 6/12] Calculating model size...", "INFO")
                storage_req = _calculate_storage_requirements(
                    config=config,
                    model_path=model_path,
                    model_source=model_source,
                    merge_enabled=merge_enabled,
                    adapter_base_path=adapter_base_path,
                    job_id=job_id
                )
                
                # IMPORTANT: If remote model was downloaded, use the local path for training
                # This prevents duplicate downloads - same path used for size calc AND training
                if storage_req.get('local_model_path') and storage_req['local_model_path'] != model_path:
                    downloaded_path = storage_req['local_model_path']
                    original_source = model_source  # Save for logging
                    _log_step(job_id, "REMOTE_MODEL_DOWNLOADED", {
                        "original_id": model_path,
                        "original_source": original_source,
                        "local_path": downloaded_path,
                    })
                    sanitized_log_service.create_terminal_log(
                        job_id, f"[Step 6/12] Model downloaded and ready for training", "INFO"
                    )
                    # Update ALL references to ensure training uses local path
                    # This is CRITICAL to prevent duplicate downloads during actual training
                    # 1. Update local variables
                    model_path = downloaded_path
                    model_source = "local"
                    # 2. Update config for training command (IMPORTANT!)
                    job.config.model_path = downloaded_path
                    # 3. Update model_source in config to LOCAL - prevents --use_hf flag
                    #    This ensures _build_command doesn't add HuggingFace/ModelScope flags
                    from ..models.schemas import ModelSource
                    job.config.model_source = ModelSource.LOCAL
                
                # Check if storage calculation returned an error
                if storage_req.get('error'):
                    error_msg = storage_req['error']
                    _log_step(job_id, "STORAGE_CALCULATION_ERROR", {
                        "error": error_msg,
                        "model_path": model_path,
                        "model_source": model_source,
                    }, "ERROR")
                    
                    sanitized_log_service.create_terminal_log(job_id, "[Step 6/12] FAILED - Cannot calculate storage requirements", "ERROR")
                    sanitized_log_service.create_terminal_log(job_id, "", "ERROR")
                    sanitized_log_service.create_terminal_log(job_id, "============================================================", "ERROR")
                    sanitized_log_service.create_terminal_log(job_id, "  CANNOT DETERMINE MODEL SIZE", "ERROR")
                    sanitized_log_service.create_terminal_log(job_id, "============================================================", "ERROR")
                    sanitized_log_service.create_terminal_log(job_id, "", "ERROR")
                    sanitized_log_service.create_terminal_log(job_id, f"  {error_msg}", "ERROR")
                    sanitized_log_service.create_terminal_log(job_id, "", "ERROR")
                    sanitized_log_service.create_terminal_log(job_id, "  HOW TO FIX:", "ERROR")
                    sanitized_log_service.create_terminal_log(job_id, "  \u2022 Ensure the model path is correct and accessible", "ERROR")
                    sanitized_log_service.create_terminal_log(job_id, "  • If using a HuggingFace model, download it first", "ERROR")
                    sanitized_log_service.create_terminal_log(job_id, "============================================================", "ERROR")
                    
                    await job_manager.update_job(job_id,
                        status=JobStatus.FAILED,
                        error="Cannot determine model size - storage validation failed",
                        completed_at=datetime.now()
                    )
                    await ws_manager.send_status(job_id, "failed")
                    return
                
                # Check if this is a remote model (storage validation skipped)
                model_info = storage_req['model_info']
                if model_info.get('source') == 'remote_skip':
                    sanitized_log_service.create_terminal_log(
                        job_id,
                        f"[Step 6/12] Remote model detected - storage validation skipped (model will be downloaded during training)",
                        "INFO"
                    )
                    sanitized_log_service.create_terminal_log(job_id, "[Step 6/12] ✓ Skipping storage check for remote model", "INFO")
                    # Skip to next step - no storage validation for remote models
                else:
                    # Log detailed breakdown for local models
                    _log_step(job_id, "STORAGE_REQUIREMENTS_CALCULATED", {
                        "model_info": storage_req['model_info'],
                        "checkpoint_info": storage_req['checkpoint_info'],
                        "temp_required_gb": storage_req['temp_required_gb'],
                        "output_required_gb": storage_req['output_required_gb'],
                        "breakdown": storage_req['breakdown']
                    })
                    
                    # Show breakdown in terminal
                    sanitized_log_service.create_terminal_log(
                        job_id,
                        f"[Step 6/12] Model size: {model_info['size_gb']:.1f}GB (actual size on disk)",
                        "INFO"
                    )
                    for line in storage_req['breakdown'][1:]:  # Skip first line (model info already shown)
                        sanitized_log_service.create_terminal_log(job_id, f"[Step 6/12]   {line}", "INFO")
                
                storage_errors = []
                
                # Only run storage checks for local models (not remote)
                if model_info.get('source') != 'remote_skip':
                    # ============================================================
                    # CHECK 1: TEMP DIRECTORY (only needed for adapter merge)
                    # ============================================================
                    if storage_req['temp_required_gb'] > 0:
                        temp_dir = tempfile.gettempdir()
                        temp_available = _get_available_space_gb(temp_dir)
                        temp_required = storage_req['temp_required_gb']
                        
                        sanitized_log_service.create_terminal_log(
                            job_id,
                            f"[Step 6/12] Checking TEMPORARY storage for adapter merge...",
                            "INFO"
                        )
                        sanitized_log_service.create_terminal_log(
                            job_id,
                            f"[Step 6/12]   Location: {temp_dir}",
                            "INFO"
                        )
                        sanitized_log_service.create_terminal_log(
                            job_id,
                            f"[Step 6/12]   Available: {temp_available:.1f}GB | Required: {temp_required:.1f}GB",
                            "INFO"
                        )
                        
                        if temp_available >= 0 and temp_available < temp_required:
                            shortfall = temp_required - temp_available
                            storage_errors.append({
                                'type': 'TEMPORARY STORAGE (for Adapter Merge)',
                                'location': temp_dir,
                                'available': temp_available,
                                'required': temp_required,
                                'shortfall': shortfall,
                                'why_needed': 'When merging a LoRA adapter with a base model, the system creates a temporary merged model file before training.',
                                'how_to_fix': [
                                    f'Add at least {shortfall:.1f}GB more storage at {temp_dir}',
                                    f'Or free up {shortfall:.1f}GB by removing unused files from this location',
                                ]
                            })
                        else:
                            sanitized_log_service.create_terminal_log(job_id, f"[Step 6/12] \u2713 Temporary storage OK", "INFO")
                    
                    # ============================================================
                    # CHECK 2: OUTPUT DIRECTORY (always required for local models)
                    # ============================================================
                    output_available = _get_available_space_gb(output_dir)
                    output_required = storage_req['output_required_gb']
                    checkpoint_info = storage_req['checkpoint_info']
                    
                    sanitized_log_service.create_terminal_log(
                        job_id,
                        f"[Step 6/12] Checking OUTPUT storage for training checkpoints...",
                        "INFO"
                    )
                    sanitized_log_service.create_terminal_log(
                        job_id,
                        f"[Step 6/12]   Location: {output_dir}",
                        "INFO"
                    )
                    sanitized_log_service.create_terminal_log(
                        job_id,
                        f"[Step 6/12]   Available: {output_available:.1f}GB | Required: {output_required:.1f}GB",
                        "INFO"
                    )
                    sanitized_log_service.create_terminal_log(
                        job_id,
                        f"[Step 6/12]   ({checkpoint_info['num_checkpoints']} checkpoints \u00d7 {checkpoint_info.get('checkpoint_size_gb', 0):.1f}GB each)",
                        "INFO"
                    )
                    
                    if output_available >= 0 and output_available < output_required:
                        shortfall = output_required - output_available
                        train_type = checkpoint_info['train_type']
                        
                        if train_type in ['lora', 'qlora', 'adalora']:
                            why_needed = 'LoRA training saves adapter checkpoint files during training. Each checkpoint contains the trained adapter weights.'
                        else:
                            why_needed = 'Full fine-tuning saves complete model checkpoints during training. Each checkpoint contains the entire model weights.'
                        
                        storage_errors.append({
                            'type': 'OUTPUT STORAGE (for Training Checkpoints)',
                            'location': output_dir,
                            'available': output_available,
                            'required': output_required,
                            'shortfall': shortfall,
                            'why_needed': why_needed,
                            'how_to_fix': [
                                f'Add at least {shortfall:.1f}GB more storage at {output_dir}',
                                f'Or free up {shortfall:.1f}GB by removing unused files from this location',
                                f'Or reduce save_total_limit to save fewer checkpoints (currently: {checkpoint_info["num_checkpoints"]})',
                            ]
                        })
                    else:
                        sanitized_log_service.create_terminal_log(job_id, f"[Step 6/12] \u2713 Output storage OK", "INFO")
                    
                    # ============================================================
                    # BLOCK IF INSUFFICIENT STORAGE
                    # ============================================================
                    if storage_errors:
                        _log_step(job_id, "STORAGE_VALIDATION_FAILED", {
                            "errors": [e['type'] for e in storage_errors],
                            "temp_required": storage_req['temp_required_gb'],
                            "output_required": storage_req['output_required_gb'],
                            "model_size_gb": model_info['size_gb'],
                        }, "ERROR")
                        
                        # Display user-friendly error in terminal
                        sanitized_log_service.create_terminal_log(job_id, "", "ERROR")
                        sanitized_log_service.create_terminal_log(job_id, "============================================================", "ERROR")
                        sanitized_log_service.create_terminal_log(job_id, "  TRAINING CANNOT START - INSUFFICIENT DISK SPACE", "ERROR")
                        sanitized_log_service.create_terminal_log(job_id, "============================================================", "ERROR")
                        
                        for i, err in enumerate(storage_errors, 1):
                            sanitized_log_service.create_terminal_log(job_id, "", "ERROR")
                            sanitized_log_service.create_terminal_log(job_id, f"  [{i}] {err['type']}", "ERROR")
                            sanitized_log_service.create_terminal_log(job_id, f"      Location: {err['location']}", "ERROR")
                            sanitized_log_service.create_terminal_log(job_id, f"      Available: {err['available']:.1f}GB", "ERROR")
                            sanitized_log_service.create_terminal_log(job_id, f"      Required:  {err['required']:.1f}GB", "ERROR")
                            sanitized_log_service.create_terminal_log(job_id, f"      Shortfall: {err['shortfall']:.1f}GB", "ERROR")
                            sanitized_log_service.create_terminal_log(job_id, "", "ERROR")
                            sanitized_log_service.create_terminal_log(job_id, f"      WHY THIS IS NEEDED:", "ERROR")
                            sanitized_log_service.create_terminal_log(job_id, f"      {err['why_needed']}", "ERROR")
                            sanitized_log_service.create_terminal_log(job_id, "", "ERROR")
                            sanitized_log_service.create_terminal_log(job_id, f"      HOW TO FIX:", "ERROR")
                            for fix in err['how_to_fix']:
                                sanitized_log_service.create_terminal_log(job_id, f"      \u2022 {fix}", "ERROR")
                        
                        sanitized_log_service.create_terminal_log(job_id, "", "ERROR")
                        sanitized_log_service.create_terminal_log(job_id, "============================================================", "ERROR")
                        
                        # Create short summary for job status
                        error_summary = "Insufficient disk space: "
                        error_parts = []
                        for err in storage_errors:
                            error_parts.append(f"{err['type'].split('(')[0].strip()} needs {err['shortfall']:.1f}GB more")
                        error_summary += "; ".join(error_parts)
                        
                        await ws_manager.send_log(job_id, f"ERROR: {error_summary}")
                        await job_manager.add_log(job_id, f"ERROR: {error_summary}")
                        
                        await job_manager.update_job(job_id,
                            status=JobStatus.FAILED,
                            error=error_summary,
                            completed_at=datetime.now()
                        )
                        await ws_manager.send_status(job_id, "failed")
                        return
                
                # Storage validation passed
                total_required = storage_req['temp_required_gb'] + storage_req['output_required_gb']
                _log_step(job_id, "STORAGE_VALIDATION_PASSED", {
                    "total_required_gb": total_required,
                    "temp_required_gb": storage_req['temp_required_gb'],
                    "output_required_gb": storage_req['output_required_gb'],
                })
                sanitized_log_service.create_terminal_log(
                    job_id,
                    f"[Step 6/12] \u2713 Storage check passed. Total required: ~{total_required:.1f}GB",
                    "INFO"
                )
                await ws_manager.send_log(job_id, f"\u2713 Storage check passed (~{total_required:.1f}GB needed)")
                
            except Exception as storage_err:
                _log_step(job_id, "STORAGE_VALIDATION_ERROR", {
                    "error": str(storage_err),
                    "error_type": type(storage_err).__name__,
                    "traceback": traceback.format_exc(),
                }, "ERROR")
                sanitized_log_service.create_terminal_log(job_id, "[Step 6/12] FAILED - Storage validation error", "ERROR")
                sanitized_log_service.create_terminal_log(
                    job_id,
                    f"ERROR: {type(storage_err).__name__}: {str(storage_err)[:150]}",
                    "ERROR"
                )
                raise
            
            # ============================================================
            # STEP 7: ADAPTER MERGE (if enabled)
            # ============================================================
            
            if merge_enabled and adapter_base_path:
                sanitized_log_service.create_terminal_log(job_id, "[Step 7/12] Merging adapter with base model...", "INFO")
                _log_step(job_id, "ADAPTER_MERGE_START", {
                    "adapter_path": model_path,
                    "base_model_path": adapter_base_path,
                })
                await ws_manager.send_log(job_id, "Merging adapter with base model...")
                
                try:
                    # Import merge utilities
                    from peft import PeftModel, PeftConfig
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                    import torch
                    import tempfile
                    import shutil
                    
                    # Create temporary directory for merged model
                    merge_output_dir = tempfile.mkdtemp(prefix="merged_model_")
                    _debug_log(job_id, f"Merge output directory: {merge_output_dir}")
                    
                    sanitized_log_service.create_terminal_log(
                        job_id, 
                        f"Loading base model: {adapter_base_path}", 
                        "INFO"
                    )
                    await ws_manager.send_log(job_id, f"Loading base model for merge...")
                    
                    # Load base model
                    base_model = AutoModelForCausalLM.from_pretrained(
                        adapter_base_path,
                        torch_dtype=torch.bfloat16,
                        device_map="auto",
                        trust_remote_code=True
                    )
                    
                    sanitized_log_service.create_terminal_log(
                        job_id, 
                        f"Loading adapter: {model_path}", 
                        "INFO"
                    )
                    await ws_manager.send_log(job_id, f"Loading adapter for merge...")
                    
                    # Load adapter onto base model
                    model = PeftModel.from_pretrained(base_model, model_path)
                    
                    sanitized_log_service.create_terminal_log(
                        job_id, 
                        "Merging adapter weights into base model...", 
                        "INFO"
                    )
                    await ws_manager.send_log(job_id, "Merging adapter weights...")
                    
                    # Merge and unload
                    merged_model = model.merge_and_unload()
                    
                    # Save merged model
                    sanitized_log_service.create_terminal_log(
                        job_id, 
                        f"Saving merged model...", 
                        "INFO"
                    )
                    await ws_manager.send_log(job_id, "Saving merged model...")
                    
                    merged_model.save_pretrained(merge_output_dir)
                    
                    # Copy tokenizer from base model
                    tokenizer = AutoTokenizer.from_pretrained(adapter_base_path, trust_remote_code=True)
                    tokenizer.save_pretrained(merge_output_dir)
                    
                    # Update model path to use merged model
                    job.config.model_path = merge_output_dir
                    model_path = merge_output_dir
                    
                    # Track merged model directory for cleanup after training
                    self._merged_model_dirs[job_id] = merge_output_dir
                    
                    # Clean up GPU memory after merge
                    del base_model, model, merged_model
                    torch.cuda.empty_cache()
                    
                    _log_step(job_id, "ADAPTER_MERGE_COMPLETE", {
                        "merged_model_path": merge_output_dir,
                        "success": True,
                    })
                    sanitized_log_service.create_terminal_log(
                        job_id, 
                        "[Step 7/12] \u2713 Adapter merged successfully", 
                        "INFO"
                    )
                    await ws_manager.send_log(job_id, "\u2713 Adapter merged successfully")
                    
                except Exception as merge_err:
                    error_msg = f"Adapter merge failed: {str(merge_err)}"
                    _log_step(job_id, "ADAPTER_MERGE_FAILED", {
                        "error": str(merge_err),
                        "error_type": type(merge_err).__name__,
                    }, "ERROR")
                    sanitized_log_service.create_terminal_log(job_id, "[Step 7/12] FAILED - Adapter merge failed", "ERROR")
                    sanitized_log_service.create_terminal_log(job_id, f"ERROR: {error_msg}", "ERROR")
                    await ws_manager.send_log(job_id, f"ERROR: {error_msg}")
                    
                    await job_manager.update_job(job_id, 
                        status=JobStatus.FAILED,
                        error=error_msg,
                        completed_at=datetime.now()
                    )
                    await ws_manager.send_status(job_id, "failed")
                    return
            
            # ============================================================
            # STEP 8: BUILD TRAINING COMMAND
            # ============================================================
            sanitized_log_service.create_terminal_log(job_id, "[Step 8/12] Building training command...", "INFO")
            
            resume_checkpoint = getattr(job, 'resume_from_checkpoint', None)
            if resume_checkpoint:
                sanitized_log_service.create_terminal_log(job_id, f"[Step 8/12] Resuming from checkpoint: {os.path.basename(resume_checkpoint)}", "INFO")
                await ws_manager.send_log(job_id, f"Resuming from checkpoint")
                _debug_log(job_id, f"Resuming from checkpoint: {resume_checkpoint}")
            
            try:
                cmd, actual_output_dir = self._build_command(job.config, job_id, resume_from_checkpoint=resume_checkpoint)
                cmd_str = " ".join(cmd)
                _log_step(job_id, "COMMAND_BUILD", {
                    "command": cmd_str,
                    "command_args_count": len(cmd),
                    "resume_from_checkpoint": resume_checkpoint,
                })
                sanitized_log_service.create_terminal_log(job_id, "[Step 8/12] \u2713 Training command ready", "INFO")
            except Exception as cmd_err:
                _log_step(job_id, "COMMAND_BUILD_FAILED", {"error": str(cmd_err)}, "ERROR")
                sanitized_log_service.create_terminal_log(job_id, "[Step 8/12] FAILED - Could not build training command", "ERROR")
                sanitized_log_service.create_terminal_log(job_id, f"ERROR: {str(cmd_err)}", "ERROR")
                raise
            
            # ============================================================
            # SHOW USER-FRIENDLY CONFIG SUMMARY (not raw command)
            # ============================================================
            training_method = getattr(job.config, 'training_method', None)
            method_value = training_method.value if hasattr(training_method, 'value') else str(training_method) if training_method else 'sft'
            method_display = method_value.upper()
            train_type_display = job.config.train_type.value.upper() if hasattr(job.config.train_type, 'value') else 'LORA'
            
            # For RLHF, show the specific algorithm type (DPO, GRPO, etc.)
            if method_value == 'rlhf':
                rlhf_type = getattr(job.config, 'rlhf_type', None)
                if rlhf_type:
                    rlhf_type_str = rlhf_type.upper() if isinstance(rlhf_type, str) else rlhf_type.value.upper() if hasattr(rlhf_type, 'value') else str(rlhf_type).upper()
                    method_display = f"RLHF ({rlhf_type_str})"
            
            # Build config summary for user
            config_lines = [
                f"Training Configuration:",
                f"  • Method: {method_display} ({train_type_display})",
                f"  • Epochs: {job.config.num_train_epochs}",
                f"  • Learning Rate: {job.config.learning_rate}",
                f"  • Batch Size: {job.config.per_device_train_batch_size}",
                f"  • Max Length: {job.config.max_length}",
            ]
            
            # Add LoRA params if applicable
            if train_type_display in ['LORA', 'QLORA', 'ADALORA']:
                config_lines.append(f"  • LoRA Rank: {job.config.lora_rank}")
                config_lines.append(f"  • LoRA Alpha: {job.config.lora_alpha}")
            
            # Add dataset info (just filename, not full path)
            dataset_name = os.path.basename(job.config.dataset_path) if job.config.dataset_path else 'dataset'
            config_lines.append(f"  • Dataset: {dataset_name}")
            
            config_summary = "\n".join(config_lines)
            sanitized_log_service.create_terminal_log(job_id, config_summary, "INFO")
            await ws_manager.send_log(job_id, config_summary)
            
            # ============================================================
            # STEP 9: CONFIGURE TRAINING ENVIRONMENT
            # ============================================================
            sanitized_log_service.create_terminal_log(job_id, "[Step 9/12] Configuring training environment...", "INFO")
            _debug_log(job_id, "Creating subprocess environment...")
            training_env = {
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                # DeepSpeed: Disable JIT compilation of CUDA ops
                # This prevents "CUDA_HOME does not exist" errors
                "DS_BUILD_OPS": "0",
                "DS_BUILD_AIO": "0",
                "DS_BUILD_SPARSE_ATTN": "0",
                "DS_BUILD_TRANSFORMER": "0",
                "DS_BUILD_TRANSFORMER_INFERENCE": "0",
                "DS_BUILD_STOCHASTIC_TRANSFORMER": "0",
                "DS_BUILD_UTILS": "0",
                "DS_BUILD_FUSED_ADAM": "0",
                "DS_BUILD_FUSED_LAMB": "0",
                "DS_BUILD_CPU_ADAM": "0",
                "DS_BUILD_CPU_LION": "0",
                "DS_BUILD_EVOFORMER_ATTN": "0",
                "DS_BUILD_INFERENCE_CORE_OPS": "0",
                "DS_BUILD_CUTLASS_OPS": "0",
                "DS_BUILD_RAGGED_DEVICE_OPS": "0",
            }
            
            # API Tokens - pass via environment variable for security (not visible in process list)
            if hasattr(job.config, 'hf_token') and job.config.hf_token:
                training_env["HF_TOKEN"] = job.config.hf_token
            if hasattr(job.config, 'ms_token') and job.config.ms_token:
                training_env["MODELSCOPE_API_TOKEN"] = job.config.ms_token
            
            # ============================================================
            # STEP 10: GPU DETECTION AND SELECTION
            # ============================================================
            sanitized_log_service.create_terminal_log(job_id, "[Step 10/12] Detecting available GPUs...", "INFO")
            num_gpus_to_use = 1  # Default to 1 GPU
            
            # Get available GPU count from system
            try:
                import torch
                available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
                if available_gpus > 0:
                    gpu_name = torch.cuda.get_device_name(0)
                    sanitized_log_service.create_terminal_log(job_id, f"[Step 10/12] Found {available_gpus} GPU(s): {gpu_name}", "INFO")
                else:
                    sanitized_log_service.create_terminal_log(job_id, "[Step 10/12] No CUDA GPUs detected", "WARN")
            except Exception as gpu_detect_err:
                available_gpus = 0
                sanitized_log_service.create_terminal_log(job_id, f"[Step 10/12] GPU detection failed: {str(gpu_detect_err)[:50]}", "WARN")
            
            if available_gpus > 0:
                # Determine which GPUs to use
                if hasattr(job.config, 'gpu_ids') and job.config.gpu_ids:
                    # User specified specific GPUs
                    gpu_ids = job.config.gpu_ids
                    # Validate GPU IDs don't exceed available GPUs
                    valid_gpu_ids = [g for g in gpu_ids if g < available_gpus]
                    if len(valid_gpu_ids) != len(gpu_ids):
                        sanitized_log_service.create_terminal_log(job_id, f"Warning: Some GPU IDs exceed available GPUs ({available_gpus}). Using valid GPUs only.", "WARN")
                        await ws_manager.send_log(job_id, f"Warning: Some GPU IDs exceed available GPUs ({available_gpus}). Using valid GPUs only.")
                    if valid_gpu_ids:
                        training_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in valid_gpu_ids)
                        num_gpus_to_use = len(valid_gpu_ids)
                        sanitized_log_service.create_terminal_log(job_id, f"Using specified GPUs: {valid_gpu_ids}", "INFO")
                        await ws_manager.send_log(job_id, f"Using specified GPUs: {valid_gpu_ids}")
                elif hasattr(job.config, 'num_gpus') and job.config.num_gpus:
                    # User specified number of GPUs
                    num_gpus_to_use = min(job.config.num_gpus, available_gpus)
                    training_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus_to_use))
                    sanitized_log_service.create_terminal_log(job_id, f"Using {num_gpus_to_use} GPUs (0-{num_gpus_to_use-1})", "INFO")
                    await ws_manager.send_log(job_id, f"Using {num_gpus_to_use} GPUs (0-{num_gpus_to_use-1})")
                else:
                    # Auto: Use all available GPUs
                    num_gpus_to_use = available_gpus
                    sanitized_log_service.create_terminal_log(job_id, f"Auto-detected {available_gpus} GPU(s). Using all.", "INFO")
                    await ws_manager.send_log(job_id, f"Auto-detected {available_gpus} GPU(s). Using all.")
                
                # Enable multi-GPU training with torchrun if more than 1 GPU
                if num_gpus_to_use > 1:
                    training_env["NPROC_PER_NODE"] = str(num_gpus_to_use)
                    sanitized_log_service.create_terminal_log(job_id, f"Multi-GPU training enabled: NPROC_PER_NODE={num_gpus_to_use}", "INFO")
                    await ws_manager.send_log(job_id, f"Multi-GPU training enabled: NPROC_PER_NODE={num_gpus_to_use}")
            else:
                sanitized_log_service.create_terminal_log(job_id, "No GPUs detected. Training will use CPU.", "WARN")
                await ws_manager.send_log(job_id, "No GPUs detected. Training will use CPU.")
            
            _log_step(job_id, "GPU_SELECTION", {
                "available_gpus": available_gpus,
                "num_gpus_to_use": num_gpus_to_use,
                "cuda_visible_devices": training_env.get("CUDA_VISIBLE_DEVICES", "all"),
            })
            sanitized_log_service.create_terminal_log(job_id, f"[Step 10/12] \u2713 GPU configuration complete", "INFO")
            
            # ============================================================
            # STEP 11: LAUNCH TRAINING SUBPROCESS
            # ============================================================
            sanitized_log_service.create_terminal_log(job_id, "[Step 11/12] Launching training process...", "INFO")
            _log_step(job_id, "SUBPROCESS_CREATE", {
                "command_length": len(cmd),
                "env_keys": list(training_env.keys())[:10],  # First 10 env keys
            })
            
            try:
                process = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.STDOUT,
                    env=training_env,
                )
                _log_step(job_id, "SUBPROCESS_STARTED", {
                    "pid": process.pid,
                    "status": "running"
                })
                sanitized_log_service.create_terminal_log(job_id, f"[Step 11/12] \u2713 Training process launched (PID: {process.pid})", "INFO")
            except Exception as proc_err:
                _log_step(job_id, "SUBPROCESS_FAILED", {"error": str(proc_err)}, "ERROR")
                sanitized_log_service.create_terminal_log(job_id, "[Step 11/12] FAILED - Could not start training process", "ERROR")
                sanitized_log_service.create_terminal_log(job_id, f"ERROR: {type(proc_err).__name__}: {str(proc_err)[:100]}", "ERROR")
                raise
            
            await job_manager.set_process(job_id, process)
            started_time = datetime.now()
            await job_manager.update_job(job_id, status=JobStatus.RUNNING, output_dir=actual_output_dir)
            
            # PERSIST TO DATABASE - mark as running with start time and output dir
            _sync_job_to_database(job_id, "running", started_at=started_time, output_dir=actual_output_dir)
            
            # ============================================================
            # STEP 12: TRAINING RUNNING - STREAMING OUTPUT
            # ============================================================
            _log_step(job_id, "STATUS_UPDATE", {"status": "RUNNING", "pid": process.pid})
            await ws_manager.send_status(job_id, "running")
            sanitized_log_service.create_terminal_log(job_id, "[Step 12/12] \u2713 Training is now RUNNING", "INFO")
            sanitized_log_service.create_terminal_log(job_id, "Training output will appear below...", "INFO")
            await ws_manager.send_log(job_id, "Training started...")
            
            # CRITICAL: Update global training status service
            from .training_status_service import training_status_service
            job = await job_manager.get_job(job_id)
            model_name = job.config.model_path.split('/')[-1] if job and job.config else None
            await training_status_service.set_training_started(
                job_id=job_id,
                job_name=job.name if job else job_id,
                model_name=model_name,
            )
            await training_status_service.set_training_running(job_id)
            
            total_steps = 0
            current_step = 0
            current_loss = None
            checkpoint_count = 0  # Track number of checkpoints saved
            has_received_training_output = False  # Track if we've seen actual training steps
            
            # Get job timeout from system settings
            from ..core.capabilities import get_system_settings
            job_timeout_hours = get_system_settings().JOB_TIMEOUT_HOURS
            job_start_time = datetime.now()
            last_timeout_check = job_start_time
            last_output_time = datetime.now()
            loading_msg_count = 0
            
            # Stream output with periodic "still loading" messages
            while True:
                try:
                    line = await asyncio.wait_for(process.stdout.readline(), timeout=60.0)
                    if not line:
                        break
                    last_output_time = datetime.now()
                except asyncio.TimeoutError:
                    # No output for 60 seconds - emit a progress message if model is still loading
                    if process.returncode is not None:
                        break
                    if not has_received_training_output:
                        loading_msg_count += 1
                        elapsed = int((datetime.now() - job_start_time).total_seconds())
                        loading_msg = f"Loading model into GPU memory... ({elapsed}s elapsed, large models can take 3-5 minutes)"
                        sanitized_log_service.create_terminal_log(job_id, loading_msg, "INFO")
                        await ws_manager.send_log(job_id, loading_msg)
                    continue
                
                # ============================================================
                # TIMEOUT CHECK: Auto-stop jobs that run too long
                # Check every 5 minutes to avoid overhead
                # ============================================================
                now = datetime.now()
                if (now - last_timeout_check).total_seconds() > 300:  # Check every 5 minutes
                    last_timeout_check = now
                    elapsed_hours = (now - job_start_time).total_seconds() / 3600
                    if elapsed_hours >= job_timeout_hours:
                        timeout_msg = f"Job exceeded maximum runtime ({job_timeout_hours} hours). Auto-stopping."
                        _log_step(job_id, "TRAINING_TIMEOUT", {
                            "elapsed_hours": elapsed_hours,
                            "timeout_hours": job_timeout_hours,
                        }, "ERROR")
                        sanitized_log_service.create_terminal_log(job_id, f"TIMEOUT: {timeout_msg}", "ERROR")
                        await ws_manager.send_log(job_id, f"TIMEOUT: {timeout_msg}", "error")
                        process.terminate()
                        await job_manager.update_job(job_id, 
                            status=JobStatus.FAILED,
                            error=timeout_msg,
                            completed_at=datetime.now()
                        )
                        await ws_manager.send_status(job_id, "failed", timeout_msg)
                        return
                
                line_str = line.decode("utf-8", errors="ignore").strip()
                if not line_str:
                    continue
                
                # ============================================================
                # ENCRYPTED LOG: Full output (only US Inc can read)
                # Contains ALL details including file paths, library names, etc.
                # ============================================================
                encrypted_log_service.encrypt_and_format(line_str, job_id)
                
                # ============================================================
                # SANITIZE OUTPUT FOR USER: Hide internal details
                # - File paths (/app/..., /usr/...)
                # - Library names (transformers, torch, huggingface, etc.)
                # - Code references (line numbers, function names)
                # User gets clear, understandable messages
                # ============================================================
                sanitized_line = sanitized_log_service.sanitize_for_display(line_str)
                
                # Only log if there's something to show (sanitize_for_display returns None for filtered messages)
                if sanitized_line:
                    # ============================================================
                    # TERMINAL LOG: Write sanitized output to terminal log file
                    # This ensures logs persist even after page refresh/backend restart
                    # ============================================================
                    sanitized_log_service.create_terminal_log(job_id, sanitized_line, "OUTPUT")
                    
                    # ============================================================
                    # WEBSOCKET & IN-MEMORY: Send SANITIZED output only
                    # User sees clean, understandable logs without internal details
                    # ============================================================
                    await job_manager.add_log(job_id, sanitized_line)
                    await ws_manager.send_log(job_id, sanitized_line)
                
                # Parse metrics from output
                metrics = self._parse_log_line(line_str, job_id)
                
                if "total_steps" in metrics:
                    total_steps = metrics["total_steps"]
                if "step" in metrics:
                    current_step = metrics["step"]
                    has_received_training_output = True
                if "loss" in metrics:
                    current_loss = metrics["loss"]
                    has_received_training_output = True
                
                # ============================================================
                # CHECKPOINT DETECTION: Multiple methods for robustness
                # 1. HuggingFace Trainer: "Saving model checkpoint to ..."
                # 2. Trainer logging: checkpoint-XXX patterns
                # 3. Progress output: "Saving checkpoint" messages
                # Note: USF BIOS info_debug is suppressed in production, so we
                # also detect from HuggingFace Trainer's standard logging
                # ============================================================
                checkpoint_detected = False
                checkpoint_path = None
                
                # Method 1: Direct checkpoint message (may appear from HuggingFace Trainer)
                if "Saving model checkpoint" in line_str or "Saving checkpoint" in line_str:
                    checkpoint_detected = True
                    if " to " in line_str:
                        checkpoint_path = line_str.split(" to ")[-1].strip()
                
                # Method 2: Checkpoint directory pattern in output
                # HuggingFace Trainer outputs paths like "/path/checkpoint-500"
                elif "checkpoint-" in line_str and current_step > 0:
                    import re
                    ckpt_match = re.search(r'(checkpoint-\d+)', line_str)
                    if ckpt_match:
                        # Only count if this looks like a save operation (not just a reference)
                        if any(kw in line_str.lower() for kw in ['saving', 'saved', 'save', 'output', 'write']):
                            checkpoint_detected = True
                            checkpoint_path = ckpt_match.group(1)
                
                if checkpoint_detected:
                    checkpoint_count += 1
                    sanitized_log_service.create_terminal_log(job_id, f"✓ Checkpoint {checkpoint_count} saved: {checkpoint_path or f'step-{current_step}'}", "INFO")
                    _debug_log(job_id, f"Checkpoint {checkpoint_count} detected: {line_str}")
                    # Send checkpoint event to frontend via WebSocket
                    await ws_manager.broadcast(job_id, {
                        "type": "checkpoint",
                        "count": checkpoint_count,
                        "path": checkpoint_path,
                        "step": current_step
                    })
                
                # Update job state
                update_fields = {}
                if current_step > 0:
                    update_fields["current_step"] = current_step
                if total_steps > 0:
                    update_fields["total_steps"] = total_steps
                if current_loss is not None:
                    update_fields["current_loss"] = current_loss
                if "epoch" in metrics:
                    update_fields["current_epoch"] = int(metrics["epoch"])
                if "learning_rate" in metrics:
                    update_fields["learning_rate"] = metrics["learning_rate"]
                
                if update_fields:
                    await job_manager.update_job(job_id, **update_fields)
                    
                    # Update global training status service with progress
                    await training_status_service.update_progress(
                        job_id=job_id,
                        current_step=update_fields.get("current_step"),
                        total_steps=update_fields.get("total_steps"),
                        current_loss=update_fields.get("current_loss"),
                        learning_rate=update_fields.get("learning_rate"),
                        current_epoch=update_fields.get("current_epoch"),
                    )
                
                # ============================================================
                # DATABASE: Save metrics for graphs (every step with metrics)
                # Metrics are stored PER JOB_ID - no mixing between trainings
                # Supports ALL training algorithms: SFT, RLHF, PPO, DPO, GRPO, etc.
                # ============================================================
                if "step" in metrics and len(metrics) > 1:
                    try:
                        db = SessionLocal()
                        job_service = JobService(db)
                        job_service.save_training_metric(
                            job_id=job_id,  # Ensures data isolation per training job
                            step=metrics["step"],
                            epoch=metrics.get("epoch"),
                            # Common metrics
                            loss=metrics.get("loss"),
                            learning_rate=metrics.get("learning_rate"),
                            grad_norm=metrics.get("grad_norm"),
                            eval_loss=metrics.get("eval_loss"),
                            # RLHF/PPO metrics
                            reward=metrics.get("reward"),
                            kl_divergence=metrics.get("kl_divergence"),
                            policy_loss=metrics.get("policy_loss"),
                            value_loss=metrics.get("value_loss"),
                            entropy=metrics.get("entropy"),
                            # DPO metrics
                            chosen_rewards=metrics.get("chosen_rewards"),
                            rejected_rewards=metrics.get("rejected_rewards"),
                            reward_margin=metrics.get("reward_margin"),
                            # Extra metrics (algorithm-specific in JSON)
                            extra_metrics=metrics.get("extra_metrics"),
                        )
                        db.close()
                    except Exception as e:
                        _debug_log(job_id, f"Failed to save metric: {e}", "WARNING")
                
                # ============================================================
                # TERMINAL LOG: Also save formatted progress metrics to sanitized log
                # ============================================================
                if metrics:
                    # Build minimal progress string for terminal
                    progress_parts = []
                    if "step" in metrics:
                        step_str = f"Step {metrics['step']}"
                        if total_steps > 0:
                            pct = (metrics['step'] / total_steps) * 100
                            step_str += f"/{total_steps} ({pct:.1f}%)"
                        progress_parts.append(step_str)
                    if "epoch" in metrics:
                        progress_parts.append(f"Epoch {metrics['epoch']:.2f}")
                    if "loss" in metrics:
                        progress_parts.append(f"Loss: {metrics['loss']:.4f}")
                    if "learning_rate" in metrics:
                        progress_parts.append(f"LR: {metrics['learning_rate']:.2e}")
                    
                    if progress_parts:
                        terminal_line = " | ".join(progress_parts)
                        # Only save to sanitized log file (not websocket - raw output already sent)
                        sanitized_log_service.create_terminal_log(job_id, terminal_line, "INFO")
                
                # Send progress update to frontend
                if current_step > 0 and total_steps > 0:
                    await ws_manager.send_progress(
                        job_id, current_step, total_steps,
                        loss=current_loss,
                        lr=metrics.get("learning_rate")
                    )
            
            # Wait for process to complete
            return_code = await process.wait()
            _log_step(job_id, "SUBPROCESS_COMPLETE", {
                "return_code": return_code,
                "success": return_code == 0,
            })
            
            if return_code == 0:
                _log_step(job_id, "TRAINING_COMPLETED", {
                    "status": "success",
                    "return_code": return_code,
                    "total_steps": total_steps,
                    "checkpoint_count": checkpoint_count,
                })
                completed_time = datetime.now()
                duration = (completed_time - started_time).total_seconds()
                duration_str = f"{int(duration // 3600)}h {int((duration % 3600) // 60)}m {int(duration % 60)}s"
                
                await job_manager.update_job(job_id,
                    status=JobStatus.COMPLETED,
                    completed_at=completed_time
                )
                
                # PERSIST TO DATABASE for training history
                _sync_job_to_database(job_id, "completed", completed_at=completed_time)
                
                await ws_manager.send_status(job_id, "completed")
                sanitized_log_service.create_terminal_log(job_id, "========================================", "INFO")
                sanitized_log_service.create_terminal_log(job_id, f"\u2713 TRAINING COMPLETED SUCCESSFULLY", "INFO")
                sanitized_log_service.create_terminal_log(job_id, f"  Duration: {duration_str}", "INFO")
                sanitized_log_service.create_terminal_log(job_id, f"  Total Steps: {total_steps}", "INFO")
                sanitized_log_service.create_terminal_log(job_id, f"  Checkpoints Saved: {checkpoint_count}", "INFO")
                sanitized_log_service.create_terminal_log(job_id, "========================================", "INFO")
                await ws_manager.send_log(job_id, "Training completed successfully!", "success")
                sanitized_log_service.log_session_end(job_id, "COMPLETED")
                
                # CRITICAL: Update global training status service
                from .training_status_service import training_status_service
                await training_status_service.set_training_completed(job_id)
                
                # POST-TRAINING CLEANUP: Release GPU memory after successful completion
                await self._cleanup_after_training(job_id, "completed")
            else:
                # Log failure with exit code
                _log_step(job_id, "TRAINING_FAILED", {
                    "status": "failed",
                    "return_code": return_code,
                    "total_steps": total_steps,
                    "checkpoint_count": checkpoint_count,
                }, "ERROR")
                error_msg = f"Training failed with exit code {return_code}"
                sanitized_log_service.create_terminal_log(job_id, "========================================", "ERROR")
                sanitized_log_service.create_terminal_log(job_id, "TRAINING FAILED", "ERROR")
                sanitized_log_service.create_terminal_log(job_id, f"  Exit Code: {return_code}", "ERROR")
                sanitized_log_service.create_terminal_log(job_id, f"  Steps Completed: {current_step}/{total_steps}", "ERROR")
                
                # Try to get last few lines of output for error context
                logs = await job_manager.get_logs(job_id, last_n=10)
                if logs:
                    last_log = logs[-1] if logs else ""
                    if "No code object" in last_log or "ModuleNotFoundError" in last_log:
                        error_msg = f"Module loading error. Please rebuild the container. ({last_log[:100]})"
                    elif "CUDA" in last_log or "GPU" in last_log:
                        error_msg = "GPU/CUDA error occurred. Check GPU availability."
                    elif "out of memory" in last_log.lower():
                        error_msg = "Out of memory. Try reducing batch size."
                
                completed_time = datetime.now()
                await job_manager.update_job(job_id,
                    status=JobStatus.FAILED,
                    error=error_msg,
                    completed_at=completed_time
                )
                
                # PERSIST TO DATABASE for training history
                _sync_job_to_database(job_id, "failed", error_message=error_msg, completed_at=completed_time)
                
                await ws_manager.send_status(job_id, "failed", error_msg)
                sanitized_log_service.create_terminal_log(job_id, error_msg, "ERROR")
                await ws_manager.send_log(job_id, error_msg, "error")
                sanitized_log_service.log_session_end(job_id, "FAILED", error_message=error_msg)
                
                # CRITICAL: Update global training status service
                from .training_status_service import training_status_service
                await training_status_service.set_training_failed(job_id, error_msg)
                
                # POST-TRAINING CLEANUP: Release GPU memory after failure
                await self._cleanup_after_training(job_id, "failed")
        
        except asyncio.CancelledError:
            _log_step(job_id, "TRAINING_CANCELLED", {
                "status": "cancelled",
                "reason": "user_requested",
            }, "WARN")
            completed_time = datetime.now()
            await job_manager.update_job(job_id, status=JobStatus.STOPPED)
            
            # PERSIST TO DATABASE for training history
            _sync_job_to_database(job_id, "cancelled", completed_at=completed_time)
            
            await ws_manager.send_status(job_id, "stopped")
            sanitized_log_service.create_terminal_log(job_id, "Training stopped by user", "WARN")
            await ws_manager.send_log(job_id, "Training stopped by user", "warning")
            sanitized_log_service.log_session_end(job_id, "CANCELLED")
            
            # CRITICAL: Update global training status service
            from .training_status_service import training_status_service
            await training_status_service.set_training_stopped(job_id)
            
            # POST-TRAINING CLEANUP: Release GPU memory after cancellation
            await self._cleanup_after_training(job_id, "cancelled")
        
        except Exception as e:
            # ============================================================
            # ENCRYPTED LOG: Full error details (only US Inc can read)
            # ============================================================
            full_error = str(e)
            full_traceback = traceback.format_exc()
            _log_step(job_id, "TRAINING_EXCEPTION", {
                "status": "exception",
                "error_type": type(e).__name__,
                "error_message": full_error,
                "traceback": full_traceback,
            }, "ERROR")
            
            # ============================================================
            # TERMINAL LOG: User-friendly error with context
            # ============================================================
            # Get minimal user-friendly error from sanitized service
            sanitized = sanitized_log_service.sanitize_error(full_error)
            minimal_error = sanitized['user_message']
            error_type = type(e).__name__
            
            sanitized_log_service.create_terminal_log(job_id, "========================================", "ERROR")
            sanitized_log_service.create_terminal_log(job_id, "TRAINING FAILED - UNEXPECTED ERROR", "ERROR")
            sanitized_log_service.create_terminal_log(job_id, f"  Error Type: {error_type}", "ERROR")
            sanitized_log_service.create_terminal_log(job_id, f"  Message: {minimal_error}", "ERROR")
            sanitized_log_service.create_terminal_log(job_id, "  Check encrypted logs for full details", "ERROR")
            sanitized_log_service.create_terminal_log(job_id, "========================================", "ERROR")
            
            completed_time = datetime.now()
            await job_manager.update_job(job_id,
                status=JobStatus.FAILED,
                error=minimal_error  # Store minimal error only
            )
            
            # PERSIST TO DATABASE for training history
            _sync_job_to_database(job_id, "failed", error_message=minimal_error, completed_at=completed_time)
            
            await ws_manager.send_status(job_id, "failed", minimal_error)
            await ws_manager.send_log(job_id, f"Error: {minimal_error}", "error")
            sanitized_log_service.log_session_end(job_id, "FAILED", error_message=full_error)
            
            # CRITICAL: Update global training status service
            from .training_status_service import training_status_service
            await training_status_service.set_training_failed(job_id, minimal_error)
            
            # POST-TRAINING CLEANUP: Release GPU memory after exception
            await self._cleanup_after_training(job_id, "exception")
    
    async def _cleanup_after_training(self, job_id: str, reason: str) -> None:
        """
        Perform GPU memory cleanup and temp file cleanup after training ends.
        
        This is critical for:
        1. Preventing VRAM leaks between training sessions
        2. Cleaning up merged model temp directories to free disk space
        
        Called after training completes, fails, is cancelled, or throws an exception.
        """
        try:
            _log_step(job_id, "POST_TRAINING_CLEANUP_START", {"reason": reason})
            
            # ============================================================
            # CLEANUP MERGED MODEL TEMP DIRECTORY
            # This frees up significant disk space (can be 10-100+ GB)
            # ============================================================
            if job_id in self._merged_model_dirs:
                merge_dir = self._merged_model_dirs[job_id]
                try:
                    import shutil
                    if os.path.exists(merge_dir):
                        # Use fast 'du' command to get size (avoids slow os.walk on large models)
                        dir_size_gb = _get_dir_size_gb_fast(merge_dir, job_id, timeout_seconds=5)
                        
                        # Delete the merged model directory
                        shutil.rmtree(merge_dir)
                        
                        _debug_log(job_id, f"Cleaned up merged model temp dir: {merge_dir} ({dir_size_gb:.1f}GB freed)")
                        sanitized_log_service.create_terminal_log(
                            job_id,
                            f"Cleaned up temporary merged model ({dir_size_gb:.1f}GB freed)",
                            "INFO"
                        )
                except Exception as cleanup_err:
                    _debug_log(job_id, f"Failed to cleanup merged model dir {merge_dir}: {cleanup_err}", "WARN")
                finally:
                    # Remove from tracking dict regardless of success
                    del self._merged_model_dirs[job_id]
            
            # Small delay to allow subprocess to fully terminate
            await asyncio.sleep(0.5)
            
            # ============================================================
            # GPU MEMORY CLEANUP
            # ============================================================
            # Deep cleanup with orphan killing enabled
            cleanup_result = await gpu_cleanup_service.async_deep_cleanup(
                kill_orphans=True,
                exclude_pids=[os.getpid()]
            )
            
            _log_step(job_id, "POST_TRAINING_CLEANUP_COMPLETE", {
                "reason": reason,
                "cleanup_result": cleanup_result,
            })
            
            if cleanup_result.get("success"):
                freed_gb = cleanup_result.get("memory_freed_gb", 0)
                mem_after = cleanup_result.get("memory_after", {})
                used_gb = mem_after.get("total_used_gb", 0)
                total_gb = mem_after.get("total_memory_gb", 0)
                
                if freed_gb > 1.0:  # Only log if significant memory was freed
                    sanitized_log_service.create_terminal_log(
                        job_id,
                        f"GPU cleanup: freed {freed_gb}GB VRAM",
                        "INFO"
                    )
                
        except Exception as e:
            _log_step(job_id, "POST_TRAINING_CLEANUP_ERROR", {"error": str(e)}, "WARN")
    
    async def start_training(self, job_id: str) -> bool:
        """Start training in background"""
        _debug_log(job_id, "start_training called")
        
        job = await job_manager.get_job(job_id)
        if not job:
            _debug_log(job_id, "Job not found", "ERROR")
            return False
        if job.status == JobStatus.RUNNING:
            _debug_log(job_id, "Job already running", "WARNING")
            return False
        
        # IMMEDIATELY write to terminal log so frontend sees something
        sanitized_log_service.create_terminal_log(job_id, "Starting training job...", "INFO")
        
        # =====================================================================
        # CRITICAL SECURITY: Validate model and output path before training
        # This ensures only authorized models/paths can be used - CANNOT be bypassed
        # Even if someone modifies the frontend, this backend check blocks them
        # =====================================================================
        try:
            from ..core.capabilities import get_validator
            validator = get_validator()
            
            # Validate model path
            model_path = job.config.model_path
            model_source = getattr(job.config, 'model_source', 'local')
            if model_path.startswith('/') or model_path.startswith('./'):
                model_source = 'local'
            
            is_valid, error_msg = validator.validate_model_path(model_path, model_source)
            if not is_valid:
                _debug_log(job_id, f"Model validation failed: {error_msg}", "ERROR")
                sanitized_log_service.create_terminal_log(job_id, f"ERROR: Model validation failed - {error_msg}", "ERROR")
                await job_manager.fail_job(job_id, f"Model validation failed: {error_msg}")
                return False
            
            _debug_log(job_id, f"Model validation passed: {model_path}")
            sanitized_log_service.create_terminal_log(job_id, "Model validation passed", "INFO")
            
            # Validate output path (when locked, user cannot customize)
            user_output_dir = getattr(job.config, 'output_dir', '') or ''
            is_valid, error_msg = validator.validate_output_path(user_output_dir)
            if not is_valid:
                _debug_log(job_id, f"Output path validation failed: {error_msg}", "ERROR")
                sanitized_log_service.create_terminal_log(job_id, f"ERROR: Output path validation failed - {error_msg}", "ERROR")
                await job_manager.fail_job(job_id, f"Output path validation failed: {error_msg}")
                return False
            
            _debug_log(job_id, f"Output path validation passed")
            
            # =====================================================================
            # DATASET TYPE VALIDATION: Ensure algorithm is compatible with dataset
            # Also validates that multiple datasets have compatible types
            # =====================================================================
            try:
                from .algorithm_compatibility import algorithm_compatibility_service
                from .dataset_type_service import dataset_type_service
                
                # Get all dataset paths (comma-separated or list)
                dataset_path_str = getattr(job.config, 'dataset_path', '')
                dataset_paths = [p.strip() for p in dataset_path_str.split(',') if p.strip()]
                
                if dataset_paths and len(dataset_paths) > 0:
                    # =========================================================
                    # MULTIPLE DATASET TYPE COMPATIBILITY CHECK
                    # USF-BIOS requires all datasets to be the same type
                    # =========================================================
                    dataset_types = []
                    for ds_path in dataset_paths:
                        try:
                            # Skip HuggingFace/ModelScope prefixes for type detection
                            clean_path = ds_path
                            if clean_path.upper().startswith('HF::'):
                                clean_path = clean_path[4:]
                            elif clean_path.upper().startswith('MS::'):
                                clean_path = clean_path[4:]
                            
                            dataset_result = dataset_type_service.detect_dataset_type(ds_path)
                            dataset_types.append({
                                'path': ds_path,
                                'type': dataset_result.dataset_type.value,
                                'confidence': dataset_result.confidence
                            })
                        except Exception as e:
                            _debug_log(job_id, f"Could not detect type for {ds_path}: {e}", "WARNING")
                            dataset_types.append({
                                'path': ds_path,
                                'type': 'unknown',
                                'confidence': 0.0
                            })
                    
                    # Check if all datasets have the same type
                    if len(dataset_types) > 1:
                        unique_types = set(dt['type'] for dt in dataset_types if dt['type'] != 'unknown')
                        if len(unique_types) > 1:
                            # STRICT: Multiple different types detected - REJECT the job
                            type_list = ", ".join([f"{Path(dt['path']).name}: {dt['type']}" for dt in dataset_types])
                            error_msg = f"Cannot train with mixed dataset types: {type_list}. All datasets must be the same type (e.g., all SFT or all RLHF). Please select datasets of only one type."
                            sanitized_log_service.create_terminal_log(job_id, f"ERROR: {error_msg}", "ERROR")
                            _debug_log(job_id, error_msg, "ERROR")
                            raise ValueError(error_msg)
                        else:
                            sanitized_log_service.create_terminal_log(
                                job_id, 
                                f"Multiple datasets detected ({len(dataset_paths)}), all same type: {list(unique_types)[0] if unique_types else 'unknown'}", 
                                "INFO"
                            )
                    
                    # STRICT: Check for unknown type datasets - reject them
                    unknown_datasets = [dt for dt in dataset_types if dt['type'] == 'unknown']
                    if unknown_datasets:
                        unknown_names = ", ".join([Path(dt['path']).name for dt in unknown_datasets])
                        error_msg = f"Cannot train with datasets of unknown type: {unknown_names}. Please ensure all datasets follow a supported format (SFT, RLHF, Pre-training, or KTO)."
                        sanitized_log_service.create_terminal_log(job_id, f"ERROR: {error_msg}", "ERROR")
                        _debug_log(job_id, error_msg, "ERROR")
                        raise ValueError(error_msg)
                    
                    # Use first dataset's type for compatibility check
                    dataset_type = dataset_types[0]['type'] if dataset_types else 'unknown'
                    
                    # Get training configuration
                    training_method = getattr(job.config, 'training_method', None)
                    method_value = training_method.value if hasattr(training_method, 'value') else str(training_method) if training_method else 'sft'
                    rlhf_algorithm = getattr(job.config, 'rlhf_type', None)
                    train_type = getattr(job.config, 'train_type', None)
                    train_type_value = train_type.value if hasattr(train_type, 'value') else str(train_type) if train_type else 'lora'
                    
                    # Validate configuration compatibility
                    validation_result = algorithm_compatibility_service.validate_training_config(
                        dataset_type=dataset_type,
                        training_method=method_value,
                        rlhf_algorithm=rlhf_algorithm,
                        training_type=train_type_value,
                        quantization="4bit" if train_type_value == "qlora" else "none",
                    )
                    
                    if not validation_result["valid"]:
                        error_msg = "Training configuration validation failed:\n" + "\n".join(validation_result["errors"])
                        _debug_log(job_id, f"Config validation failed: {error_msg}", "ERROR")
                        sanitized_log_service.create_terminal_log(job_id, f"ERROR: {error_msg}", "ERROR")
                        await job_manager.fail_job(job_id, error_msg)
                        return False
                    
                    # Log any warnings
                    for warning in validation_result.get("warnings", []):
                        sanitized_log_service.create_terminal_log(job_id, f"WARNING: {warning}", "WARNING")
                    
                    _debug_log(job_id, f"Dataset type validation passed: {dataset_type} with {method_value}")
                    sanitized_log_service.create_terminal_log(job_id, f"Dataset type validation passed ({dataset_type})", "INFO")
            except ImportError:
                _debug_log(job_id, "Algorithm compatibility service not available - skipping dataset type validation")
            except Exception as e:
                _debug_log(job_id, f"Dataset type validation error (non-blocking): {e}", "WARNING")
            
        except Exception as e:
            _debug_log(job_id, f"Validation error: {e}", "ERROR")
            sanitized_log_service.create_terminal_log(job_id, f"ERROR: Configuration validation failed - {str(e)}", "ERROR")
            await job_manager.fail_job(job_id, f"Configuration validation failed: {str(e)}")
            return False
        
        _debug_log(job_id, "Creating background task...")
        sanitized_log_service.create_terminal_log(job_id, "Launching training process...", "INFO")
        
        # Create task with exception callback to catch silent failures
        task = asyncio.create_task(self.run_training(job_id))
        
        # Add callback to handle task exceptions (prevents silent failures)
        def handle_task_exception(t):
            try:
                exc = t.exception()
                if exc:
                    error_msg = f"Training task failed: {str(exc)}"
                    _debug_log(job_id, error_msg, "ERROR")
                    sanitized_log_service.create_terminal_log(job_id, f"ERROR: {error_msg}", "ERROR")
                    # Update job status synchronously
                    import asyncio
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        asyncio.create_task(job_manager.fail_job(job_id, error_msg))
            except asyncio.CancelledError:
                pass
            except Exception as cb_err:
                _debug_log(job_id, f"Task callback error: {cb_err}", "ERROR")
        
        task.add_done_callback(handle_task_exception)
        self._running_tasks[job_id] = task
        _debug_log(job_id, "Background task created")
        return True
    
    async def stop_training(self, job_id: str) -> bool:
        """Stop a running training"""
        if job_id in self._running_tasks:
            self._running_tasks[job_id].cancel()
            del self._running_tasks[job_id]
        
        return await job_manager.stop_job(job_id)


# Global instance
training_service = TrainingService()
