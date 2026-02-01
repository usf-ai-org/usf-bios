"""System metrics and status endpoints"""
import httpx
import importlib.util
import json
import logging
import os
import shutil
import socket
import subprocess
from typing import List, Literal, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Optional imports - may not be available
try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None
    PSUTIL_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available() if hasattr(torch, 'cuda') else False
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    import pynvml
    PYNVML_AVAILABLE = True
except ImportError:
    pynvml = None
    PYNVML_AVAILABLE = False

from ...core.config import settings
from ...core.capabilities import get_validator, is_system_expired, SystemExpiredError
from ...services.gpu_cleanup_service import gpu_cleanup_service
from ...services.system_encrypted_log_service import system_encrypted_log

router = APIRouter(prefix="/system", tags=["system"])


class SystemStatus(BaseModel):
    """System status response model"""
    status: Literal["live", "starting", "degraded", "offline", "error"]
    message: str
    gpu_available: bool
    gpu_name: Optional[str] = None
    cuda_available: bool
    bios_installed: bool
    backend_ready: bool
    details: dict


def check_bios_installation() -> tuple[bool, str]:
    """Check if USF BIOS training system is properly installed"""
    try:
        # Check if core packages are available
        import transformers
        import peft
        import trl
        import accelerate
        import datasets
        
        # CRITICAL: Check if usf_bios module can be imported
        # This catches missing dependencies like json_repair
        try:
            import usf_bios
        except ImportError as e:
            return False, f"USF BIOS module error: {str(e)}"
        except Exception as e:
            return False, f"USF BIOS initialization error: {str(e)}"
        
        return True, "BIOS training packages installed"
    except ImportError as e:
        return False, f"Missing required packages: {str(e)}"
    except Exception as e:
        return False, f"Installation check failed: {str(e)}"


def check_gpu_availability() -> tuple[bool, str, Optional[str]]:
    """Check if GPU is available and functional"""
    gpu_name = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            # Test GPU is actually usable
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            return True, f"GPU ready: {gpu_name}", gpu_name
        else:
            return False, "CUDA not available", None
    except Exception as e:
        return False, "GPU check failed", None


def check_cuda_available() -> bool:
    """Check if CUDA is available"""
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False


def get_gpu_metrics():
    """
    Get AGGREGATED GPU metrics across ALL GPUs using pynvml (NVIDIA Management Library).
    Returns system-wide GPU metrics matching what cloud providers (RunPod, etc.) show.
    
    Returns real-time data - no caching since GPU metrics change every second.
    """
    result = {
        "gpu_utilization": None,
        "gpu_memory_used": None,
        "gpu_memory_total": None,
        "gpu_temperature": None,
        "gpu_available": False,
        "metrics_source": "none",
        "device_count": 0
    }
    
    try:
        import pynvml
        pynvml.nvmlInit()
        
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count == 0:
            pynvml.nvmlShutdown()
            return result
        
        result["gpu_available"] = True
        result["metrics_source"] = "pynvml"
        result["device_count"] = device_count
        
        total_memory_used = 0
        total_memory_total = 0
        total_utilization = 0
        valid_util_count = 0
        max_temperature = 0
        valid_temp_count = 0
        
        for i in range(device_count):
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                
                # GPU Memory - sum across all GPUs
                try:
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                    if mem_info.used >= 0 and mem_info.total > 0:
                        total_memory_used += mem_info.used
                        total_memory_total += mem_info.total
                except pynvml.NVMLError:
                    pass
                
                # GPU Utilization - average across all GPUs
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    if 0 <= util.gpu <= 100:
                        total_utilization += util.gpu
                        valid_util_count += 1
                except pynvml.NVMLError:
                    pass
                
                # GPU Temperature - take maximum (hottest GPU)
                try:
                    temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    if 0 <= temperature <= 120:
                        max_temperature = max(max_temperature, temperature)
                        valid_temp_count += 1
                except pynvml.NVMLError:
                    pass
                    
            except pynvml.NVMLError:
                continue
        
        # Set aggregated values
        if total_memory_total > 0:
            result["gpu_memory_used"] = round(total_memory_used / (1024**3), 2)
            result["gpu_memory_total"] = round(total_memory_total / (1024**3), 2)
            if result["gpu_memory_used"] > result["gpu_memory_total"]:
                result["gpu_memory_used"] = None
                result["gpu_memory_total"] = None
        
        if valid_util_count > 0:
            result["gpu_utilization"] = int(round(total_utilization / valid_util_count))
        
        if valid_temp_count > 0:
            result["gpu_temperature"] = int(max_temperature)
        
        pynvml.nvmlShutdown()
        return result
        
    except ImportError:
        # pynvml not available - try nvidia-smi as fallback with ALL GPUs
        try:
            cmd = "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits"
            output = subprocess.check_output(cmd.split(), timeout=5).decode().strip()
            lines = output.strip().split('\n')
            
            if lines:
                result["gpu_available"] = True
                result["metrics_source"] = "nvidia-smi"
                result["device_count"] = len(lines)
                
                total_memory_used = 0
                total_memory_total = 0
                total_utilization = 0
                valid_util_count = 0
                max_temperature = 0
                valid_temp_count = 0
                
                for line in lines:
                    values = line.split(',')
                    if len(values) >= 4:
                        # Parse utilization
                        try:
                            util = int(values[0].strip())
                            if 0 <= util <= 100:
                                total_utilization += util
                                valid_util_count += 1
                        except ValueError:
                            pass
                        
                        # Parse memory used (MiB to bytes for summing)
                        try:
                            mem_used = float(values[1].strip()) * 1024 * 1024
                            if mem_used >= 0:
                                total_memory_used += mem_used
                        except ValueError:
                            pass
                        
                        # Parse memory total (MiB to bytes for summing)
                        try:
                            mem_total = float(values[2].strip()) * 1024 * 1024
                            if mem_total > 0:
                                total_memory_total += mem_total
                        except ValueError:
                            pass
                        
                        # Parse temperature (take max)
                        try:
                            temp = int(values[3].strip())
                            if 0 <= temp <= 120:
                                max_temperature = max(max_temperature, temp)
                                valid_temp_count += 1
                        except ValueError:
                            pass
                
                # Set aggregated values
                if total_memory_total > 0:
                    result["gpu_memory_used"] = round(total_memory_used / (1024**3), 2)
                    result["gpu_memory_total"] = round(total_memory_total / (1024**3), 2)
                
                if valid_util_count > 0:
                    result["gpu_utilization"] = int(round(total_utilization / valid_util_count))
                
                if valid_temp_count > 0:
                    result["gpu_temperature"] = int(max_temperature)
                
                return result
        except (subprocess.SubprocessError, FileNotFoundError, subprocess.TimeoutExpired):
            pass
        
        # Last resort: try torch for basic info (limited accuracy)
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                result["gpu_available"] = True
                result["metrics_source"] = "torch"
                result["device_count"] = device_count
                
                total_memory = 0
                for i in range(device_count):
                    total_memory += torch.cuda.get_device_properties(i).total_memory
                
                result["gpu_memory_total"] = round(total_memory / (1024**3), 2)
        except:
            pass
        
        return result
        
    except Exception as e:
        logging.warning(f"GPU metrics error: {e}")
        return result


def get_cpu_metrics():
    """
    Get CPU and RAM metrics using psutil.
    Returns None for unavailable metrics to avoid showing wrong data.
    """
    result = {
        "cpu_percent": None,
        "ram_used": None,
        "ram_total": None,
        "cpu_available": False
    }
    
    if PSUTIL_AVAILABLE:
        result["cpu_available"] = True
        
        # CPU Utilization (0-100%)
        # interval=0.1 gives a quick snapshot, but may be less accurate
        # For better accuracy, consider using interval=1.0 but it blocks
        try:
            cpu_percent = psutil.cpu_percent(interval=0.1)
            # Validate: should be 0-100
            if 0 <= cpu_percent <= 100:
                result["cpu_percent"] = round(cpu_percent, 1)
        except Exception:
            pass
        
        # RAM Usage
        try:
            mem = psutil.virtual_memory()
            # Validate: memory values should be positive
            if mem.used >= 0 and mem.total > 0:
                ram_used = mem.used / (1024**3)  # GB
                ram_total = mem.total / (1024**3)  # GB
                # Sanity check: used should not exceed total
                if ram_used <= ram_total:
                    result["ram_used"] = round(ram_used, 2)
                    result["ram_total"] = round(ram_total, 2)
        except Exception:
            pass
        
        return result
    
    return result


def get_storage_metrics():
    """
    Get storage metrics for the output directory.
    
    IMPORTANT: Does NOT expose the actual path - only shows storage metrics.
    This is intentional for security - users should not see internal paths.
    
    Returns:
        - storage_total_gb: Total storage capacity
        - storage_used_gb: Used storage
        - storage_free_gb: Free/available storage
        - storage_available: Whether storage info could be determined
    
    Returns None values if storage cannot be determined accurately.
    We only show data if we're 100% confident it's correct.
    """
    result = {
        "storage_total_gb": None,
        "storage_used_gb": None,
        "storage_free_gb": None,
        "storage_available": False
    }
    
    try:
        from pathlib import Path
        
        # Get the output directory from settings
        # We check the output storage path, NOT model/dataset paths
        output_dir = None
        try:
            from ...core.capabilities import get_system_settings
            system_settings = get_system_settings()
            if hasattr(system_settings, 'OUTPUT_DIR'):
                output_dir = system_settings.OUTPUT_DIR
            elif hasattr(system_settings, 'output_dir'):
                output_dir = system_settings.output_dir
        except Exception:
            pass
        
        # Fallback: try settings.OUTPUT_DIR
        if not output_dir:
            try:
                if hasattr(settings, 'OUTPUT_DIR'):
                    output_dir = settings.OUTPUT_DIR
                elif hasattr(settings, 'output_dir'):
                    output_dir = settings.output_dir
            except Exception:
                pass
        
        # Final fallback: use current working directory
        if not output_dir:
            output_dir = Path.cwd()
        
        # Ensure it's a Path object
        if isinstance(output_dir, str):
            output_dir = Path(output_dir)
        
        # Find an existing path to check storage
        check_path = output_dir
        while not check_path.exists() and str(check_path) != '/':
            check_path = check_path.parent
        if not check_path.exists():
            check_path = Path('/')
        
        # Get disk usage using shutil (most reliable cross-platform method)
        usage = shutil.disk_usage(str(check_path))
        
        # Validate the data before returning
        if usage.total > 0 and usage.free >= 0 and usage.used >= 0:
            total_gb = usage.total / (1024 ** 3)
            free_gb = usage.free / (1024 ** 3)
            used_gb = usage.used / (1024 ** 3)
            
            # Sanity checks - only return if data makes sense
            # Used + Free should approximately equal Total (allow 1% tolerance for filesystem overhead)
            expected_total = used_gb + free_gb
            tolerance = total_gb * 0.01  # 1% tolerance
            
            if abs(expected_total - total_gb) <= tolerance and used_gb <= total_gb and free_gb <= total_gb:
                result["storage_total_gb"] = round(total_gb, 1)
                result["storage_used_gb"] = round(used_gb, 1)
                result["storage_free_gb"] = round(free_gb, 1)
                result["storage_available"] = True
        
        return result
        
    except Exception as e:
        logging.warning(f"Storage metrics error: {e}")
        return result


@router.get("/metrics")
async def get_system_metrics():
    """Get comprehensive system metrics including GPU, CPU, and Storage"""
    gpu_metrics = get_gpu_metrics()
    cpu_metrics = get_cpu_metrics()
    storage_metrics = get_storage_metrics()
    
    return {
        **gpu_metrics,
        **cpu_metrics,
        **storage_metrics
    }


@router.get("/storage")
async def get_storage_info():
    """
    Get storage information for the system.
    
    Returns storage metrics WITHOUT exposing any paths.
    This is intentional for security - internal paths should not be visible.
    
    Response includes:
    - storage_total_gb: Total storage capacity in GB
    - storage_used_gb: Used storage in GB
    - storage_free_gb: Free/available storage in GB
    - storage_available: Whether storage info could be determined
    
    Note: Returns None values if storage cannot be determined with 100% accuracy.
    We prioritize showing correct data over showing any data.
    """
    return get_storage_metrics()


@router.get("/gpu")
async def get_gpu_info():
    """Get GPU-specific information"""
    return get_gpu_metrics()


@router.get("/gpus")
async def get_all_gpus():
    """
    Get list of all available GPUs with their details.
    Used by frontend to show available GPU options for selection.
    Returns device_count and list of GPUs with id, name, memory info.
    """
    result = {
        "available": False,
        "device_count": 0,
        "gpus": []
    }
    
    try:
        import pynvml
        pynvml.nvmlInit()
        
        device_count = pynvml.nvmlDeviceGetCount()
        if device_count == 0:
            pynvml.nvmlShutdown()
            return result
        
        result["available"] = True
        result["device_count"] = device_count
        
        for i in range(device_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            # Get GPU name
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode('utf-8')
            
            # Get memory info
            try:
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                memory_total_gb = round(mem_info.total / (1024**3), 1)
                memory_used_gb = round(mem_info.used / (1024**3), 1)
                memory_free_gb = round(mem_info.free / (1024**3), 1)
            except:
                memory_total_gb = None
                memory_used_gb = None
                memory_free_gb = None
            
            # Get utilization
            try:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                utilization = util.gpu
            except:
                utilization = None
            
            result["gpus"].append({
                "id": i,
                "name": name,
                "memory_total_gb": memory_total_gb,
                "memory_used_gb": memory_used_gb,
                "memory_free_gb": memory_free_gb,
                "utilization": utilization
            })
        
        pynvml.nvmlShutdown()
        return result
        
    except ImportError:
        # pynvml not available - try torch
        try:
            import torch
            if torch.cuda.is_available():
                device_count = torch.cuda.device_count()
                result["available"] = True
                result["device_count"] = device_count
                
                for i in range(device_count):
                    props = torch.cuda.get_device_properties(i)
                    result["gpus"].append({
                        "id": i,
                        "name": props.name,
                        "memory_total_gb": round(props.total_memory / (1024**3), 1),
                        "memory_used_gb": None,
                        "memory_free_gb": None,
                        "utilization": None
                    })
        except:
            pass
        
        return result
    except Exception:
        return result


@router.get("/cpu")
async def get_cpu_info():
    """Get CPU and memory information"""
    return get_cpu_metrics()


@router.get("/status", response_model=SystemStatus)
async def get_system_status():
    """
    Get comprehensive system status for frontend display.
    Returns status: live, starting, degraded, offline, error
    Frontend should block job submission unless status is 'live'
    """
    details = {}
    
    # Check system expiration first
    expired, exp_msg = is_system_expired()
    if expired:
        return SystemStatus(
            status="error",
            message=exp_msg,
            gpu_available=False,
            gpu_name=None,
            cuda_available=False,
            bios_installed=False,
            backend_ready=False,
            details={"error": exp_msg}
        )
    
    # Check BIOS installation
    bios_ok, bios_msg = check_bios_installation()
    details["bios"] = bios_msg
    
    # Check GPU
    gpu_ok, gpu_msg, gpu_name = check_gpu_availability()
    details["gpu"] = gpu_msg
    
    # Check CUDA
    cuda_ok = check_cuda_available()
    details["cuda"] = "Available" if cuda_ok else "Not available"
    
    # Backend is ready if we got this far
    backend_ready = True
    details["backend"] = "Running"
    
    # Determine overall status
    if bios_ok and gpu_ok and cuda_ok:
        status = "live"
        message = "System fully operational - Ready for training"
    elif bios_ok and not gpu_ok:
        status = "degraded"
        message = "GPU not available - Training will fail"
    elif not bios_ok and gpu_ok:
        status = "degraded"
        message = "BIOS packages not installed - Training will fail"
    elif not bios_ok and not gpu_ok:
        status = "offline"
        message = "System not ready - Missing GPU and BIOS packages"
    else:
        status = "error"
        message = "Unknown system state"
    
    # Log system health check (encrypted only)
    system_encrypted_log.log_system_health({
        "status": status,
        "gpu_available": gpu_ok,
        "gpu_name": gpu_name,
        "cuda_available": cuda_ok,
        "bios_installed": bios_ok,
        "backend_ready": backend_ready
    })
    
    return SystemStatus(
        status=status,
        message=message,
        gpu_available=gpu_ok,
        gpu_name=gpu_name,
        cuda_available=cuda_ok,
        bios_installed=bios_ok,
        backend_ready=backend_ready,
        details=details
    )


@router.get("/ready")
async def readiness_check():
    """
    Simple readiness check for job submission.
    Returns {"ready": true/false, "reason": "..."}
    """
    # Check expiration first
    expired, exp_msg = is_system_expired()
    if expired:
        return {"ready": False, "reason": exp_msg}
    
    bios_ok, bios_msg = check_bios_installation()
    gpu_ok, gpu_msg, _ = check_gpu_availability()
    
    if bios_ok and gpu_ok:
        return {"ready": True, "reason": "System ready for training"}
    
    reasons = []
    if not bios_ok:
        reasons.append(f"BIOS: {bios_msg}")
    if not gpu_ok:
        reasons.append(f"GPU: {gpu_msg}")
    
    return {"ready": False, "reason": "; ".join(reasons)}


@router.get("/hardware-requirements")
async def get_hardware_requirements():
    """
    Get hardware requirements and validation status.
    
    USF BIOS Requirements:
    - NVIDIA GPU with CUDA support (required for training)
    - Minimum 16GB VRAM recommended for most models
    - AMD GPUs are NOT supported (no ROCm support in this Docker image)
    - CPU-only training is NOT supported (impractical for LLMs)
    
    Returns detailed hardware status and clear error messages for users.
    """
    result = {
        "hardware_supported": False,
        "can_train": False,
        "can_inference": False,
        "gpu_vendor": None,
        "gpu_name": None,
        "gpu_memory_gb": None,
        "cuda_available": False,
        "requirements": {
            "gpu": "NVIDIA GPU with CUDA support",
            "min_vram": "16GB recommended (8GB minimum for small models)",
            "supported_architectures": ["Turing (RTX 20xx)", "Ampere (RTX 30xx, A100)", "Ada Lovelace (RTX 40xx)", "Hopper (H100, H200)"],
            "not_supported": ["AMD GPUs (ROCm)", "Intel GPUs", "CPU-only"]
        },
        "errors": [],
        "warnings": []
    }
    
    # Check CUDA availability
    try:
        import torch
        result["cuda_available"] = torch.cuda.is_available()
        
        if not result["cuda_available"]:
            result["errors"].append({
                "code": "NO_CUDA",
                "title": "CUDA Not Available",
                "message": "No NVIDIA GPU with CUDA support detected. USF BIOS requires an NVIDIA GPU for training.",
                "suggestions": [
                    "Ensure you have an NVIDIA GPU installed",
                    "Install NVIDIA drivers and CUDA toolkit",
                    "If using cloud GPU, ensure GPU is attached to instance",
                    "AMD and Intel GPUs are not supported"
                ]
            })
            return result
        
        # GPU is available - get details
        device_count = torch.cuda.device_count()
        if device_count == 0:
            result["errors"].append({
                "code": "NO_GPU_DEVICE",
                "title": "No GPU Device Found",
                "message": "CUDA is available but no GPU devices were found.",
                "suggestions": ["Check GPU driver installation", "Restart the container"]
            })
            return result
        
        # Get GPU info
        props = torch.cuda.get_device_properties(0)
        gpu_name = props.name
        gpu_memory_gb = round(props.total_memory / (1024**3), 1)
        compute_capability = f"{props.major}.{props.minor}"
        
        result["gpu_name"] = gpu_name
        result["gpu_memory_gb"] = gpu_memory_gb
        result["gpu_vendor"] = "NVIDIA"
        result["hardware_supported"] = True
        result["can_train"] = True
        result["can_inference"] = True
        
        # Check for AMD GPU (shouldn't happen in CUDA context, but just in case)
        if "AMD" in gpu_name.upper() or "RADEON" in gpu_name.upper():
            result["hardware_supported"] = False
            result["can_train"] = False
            result["gpu_vendor"] = "AMD"
            result["errors"].append({
                "code": "AMD_NOT_SUPPORTED",
                "title": "AMD GPU Not Supported",
                "message": "AMD GPUs are not supported in this Docker image. USF BIOS requires NVIDIA GPUs with CUDA.",
                "suggestions": [
                    "Use an NVIDIA GPU (RTX 20xx/30xx/40xx, A100, H100, etc.)",
                    "AMD ROCm support may be added in future versions"
                ]
            })
            return result
        
        # Check VRAM warnings
        if gpu_memory_gb < 8:
            result["warnings"].append({
                "code": "LOW_VRAM",
                "title": "Low GPU Memory",
                "message": f"Your GPU has only {gpu_memory_gb}GB VRAM. You may only be able to train very small models with QLoRA.",
                "suggestions": ["Use QLoRA with 4-bit quantization", "Reduce batch size to 1", "Reduce max sequence length"]
            })
        elif gpu_memory_gb < 16:
            result["warnings"].append({
                "code": "LIMITED_VRAM",
                "title": "Limited GPU Memory",
                "message": f"Your GPU has {gpu_memory_gb}GB VRAM. Some larger models may require QLoRA or reduced batch sizes.",
                "suggestions": ["Consider QLoRA for 7B+ models", "Monitor VRAM usage during training"]
            })
        
        # Check compute capability
        major = props.major
        if major < 7:
            result["warnings"].append({
                "code": "OLD_GPU_ARCH",
                "title": "Older GPU Architecture",
                "message": f"Your GPU (compute capability {compute_capability}) is an older architecture. Some optimizations may not be available.",
                "suggestions": ["Flash Attention may not work optimally", "Consider upgrading to Ampere or newer"]
            })
        
        # Add GPU architecture info
        if major >= 9:
            result["gpu_architecture"] = "Hopper"
        elif major >= 8:
            result["gpu_architecture"] = "Ampere"
        elif major >= 7:
            result["gpu_architecture"] = "Turing/Volta"
        else:
            result["gpu_architecture"] = "Legacy"
        
        return result
        
    except ImportError:
        result["errors"].append({
            "code": "NO_TORCH",
            "title": "PyTorch Not Installed",
            "message": "PyTorch is not installed. The training system cannot function.",
            "suggestions": ["Reinstall the Docker image", "Contact support"]
        })
        return result
    except Exception as e:
        result["errors"].append({
            "code": "HARDWARE_CHECK_FAILED",
            "title": "Hardware Check Failed",
            "message": "Unable to verify hardware status.",
            "suggestions": ["Restart the container", "Check GPU drivers"]
        })
        return result


# =============================================================================
# System Configuration Endpoints (Hidden from API docs)
# =============================================================================

class SystemInfo(BaseModel):
    """System information - minimal, does not expose restrictions"""
    ready: bool = True


class ValidationRequest(BaseModel):
    """Request to check if configuration works with this system"""
    model_path: str
    model_source: str = "local"
    architecture: Optional[str] = None


class ValidationResponse(BaseModel):
    """Response indicating if configuration is compatible"""
    is_supported: bool
    message: Optional[str] = None


@router.get("/info", response_model=SystemInfo, include_in_schema=False)
async def get_system_info():
    """Get minimal system info (hidden from docs)."""
    return SystemInfo(ready=True)


def check_external_storage() -> dict:
    """
    Check if external storage is mounted.
    Common mount points for cloud GPU providers:
    - RunPod: /runpod-volume
    - Lambda Labs: /home/ubuntu/data
    - Vast.ai: /workspace
    - Generic: /mnt/storage, /shared
    """
    EXTERNAL_STORAGE_PATHS = [
        "/runpod-volume",
        "/workspace", 
        "/mnt/storage",
        "/shared",
        "/data/external",
    ]
    
    # Also check env var for custom storage path
    custom_path = os.getenv("EXTERNAL_STORAGE_PATH")
    if custom_path:
        EXTERNAL_STORAGE_PATHS.insert(0, custom_path)
    
    for path in EXTERNAL_STORAGE_PATHS:
        if os.path.exists(path) and os.path.isdir(path):
            # Check if writable
            try:
                test_file = os.path.join(path, ".write_test")
                with open(test_file, "w") as f:
                    f.write("test")
                os.remove(test_file)
                return {
                    "has_external_storage": True,
                    "storage_path": path,
                    "storage_writable": True
                }
            except:
                return {
                    "has_external_storage": True,
                    "storage_path": path,
                    "storage_writable": False
                }
    
    return {
        "has_external_storage": False,
        "storage_path": None,
        "storage_writable": False
    }


@router.get("/capabilities", include_in_schema=False)
async def get_system_capabilities():
    """Capabilities including storage detection and source restrictions."""
    storage_info = check_external_storage()
    validator = get_validator()
    
    # Get supported sources - presented as "what the system supports"
    # NOT as "what is restricted" - stealth messaging
    info = validator.get_info()
    
    return {
        "ready": True,
        "supported_model_sources": info.get("supported_sources", ["local"]),
        "supported_dataset_sources": info.get("supported_dataset_sources", ["local"]),
        **storage_info
    }


@router.get("/locked-models", include_in_schema=False)
async def get_locked_models():
    """
    Get list of allowed/locked models for frontend.
    This endpoint provides the list of models that can be used for training/inference.
    Frontend should fetch this instead of hardcoding model list.
    
    Response format:
    {
        "is_locked": true/false,
        "models": [
            {
                "name": "USF Omega",
                "path": "/workspace/models/usf_omega",
                "source": "local",
                "modality": "text",
                "description": "USF Omega Model"
            }
        ]
    }
    """
    validator = get_validator()
    
    return {
        "is_locked": validator.is_model_locked(),
        "models": validator.get_locked_models()
    }


@router.get("/feature-flags")
async def get_feature_flags():
    """
    Get feature flags for training methods and capabilities.
    These flags are compiled into the system and CANNOT be changed at runtime.
    Frontend uses these to show/hide options accordingly.
    
    Response format:
    {
        "pretraining": false,      # Pre-training disabled
        "sft": true,               # SFT enabled
        "rlhf": true,              # RLHF enabled
        "rlhf_online": false,      # Online RL (GRPO, PPO, GKD) disabled
        "rlhf_offline": true,      # Offline RL (DPO, ORPO, etc.) enabled
        "vllm_colocate": false,    # Colocate mode disabled (requires online RL)
        "vllm_server": false,      # Server mode disabled (requires online RL)
        "lora": true,              # LoRA enabled
        "qlora": true,             # QLoRA enabled
        "adalora": true,           # AdaLoRA enabled
        "full": true,              # Full fine-tuning enabled
        "grpo": false,             # GRPO algorithm disabled
        "ppo": false,              # PPO algorithm disabled
        "gkd": false,              # GKD algorithm disabled
        "dpo": true,               # DPO algorithm enabled
        "orpo": true,              # ORPO algorithm enabled
        "simpo": true,             # SimPO algorithm enabled
        "kto": true,               # KTO algorithm enabled
        "cpo": true,               # CPO algorithm enabled
        "rm": true                 # Reward Modeling enabled
    }
    """
    try:
        from usf_bios.system_guard import get_feature_flags as get_flags
        return get_flags()
    except ImportError:
        # Fallback if system_guard is not available (development mode)
        return {
            "pretraining": True,
            "sft": True,
            "rlhf": True,
            "rlhf_online": True,
            "rlhf_offline": True,
            "vllm_colocate": True,
            "vllm_server": True,
            "lora": True,
            "qlora": True,
            "adalora": True,
            "full": True,
            "grpo": True,
            "ppo": True,
            "gkd": True,
            "dpo": True,
            "orpo": True,
            "simpo": True,
            "kto": True,
            "cpo": True,
            "rm": True,
        }


@router.get("/output-path-config", include_in_schema=False)
async def get_output_path_config():
    """
    Get output path configuration for frontend.
    Tells frontend whether output path is locked, base-locked, or free.
    
    Response format:
    {
        "mode": "locked" | "base_locked" | "free",
        "base_path": "/workspace/output",
        "is_locked": true/false,
        "user_can_customize": true/false,
        "user_can_add_path": true/false
    }
    
    When is_locked=true: Frontend should NOT show output path input
    When user_can_add_path=true: Show input for intermediate path only
    When user_can_customize=true: Show full path input
    """
    validator = get_validator()
    return validator.get_output_path_config()


@router.get("/model-lock", include_in_schema=False)
async def get_model_capabilities_legacy():
    """Legacy endpoint - hidden from docs."""
    validator = get_validator()
    return {
        "ready": True,
        "is_locked": validator.is_model_locked(),
        "models": validator.get_locked_models()
    }


@router.get("/api-tokens")
async def get_api_token_status():
    """
    Check if system has HuggingFace or ModelScope tokens configured.
    
    This allows the UI to show "Use system token" option when tokens
    are already configured via environment variables.
    
    Returns:
        - hf_token_available: True if HF_TOKEN env var is set
        - ms_token_available: True if MODELSCOPE_API_TOKEN env var is set
        - hf_token_masked: Masked version of token (first 4 chars) if available
        - ms_token_masked: Masked version of token (first 4 chars) if available
    """
    hf_token = os.environ.get("HF_TOKEN", "")
    ms_token = os.environ.get("MODELSCOPE_API_TOKEN", "")
    
    return {
        "hf_token_available": bool(hf_token),
        "ms_token_available": bool(ms_token),
        "hf_token_masked": f"{hf_token[:4]}..." if len(hf_token) > 4 else None,
        "ms_token_masked": f"{ms_token[:4]}..." if len(ms_token) > 4 else None,
    }


@router.post("/validate", response_model=ValidationResponse, include_in_schema=False)
async def validate_configuration(request: ValidationRequest):
    """Validate if configuration works with this system (hidden from docs)."""
    validator = get_validator()
    
    # Validate model path and source
    is_valid, message = validator.validate_model_path(request.model_path, request.model_source)
    if not is_valid:
        return ValidationResponse(is_supported=False, message=message)
    
    # Validate architecture if provided
    if request.architecture:
        is_valid, message = validator.validate_architecture(request.architecture)
        if not is_valid:
            return ValidationResponse(is_supported=False, message=message)
    
    return ValidationResponse(is_supported=True)


@router.post("/validate-config", include_in_schema=False)
async def validate_system_config(request: ValidationRequest):
    """Legacy endpoint - hidden from docs."""
    return await validate_configuration(request)


@router.post("/validate-model", include_in_schema=False)
async def validate_model_for_training(request: ValidationRequest):
    """Legacy endpoint - hidden from docs."""
    return await validate_configuration(request)


# =============================================================================
# Training Capabilities Detection - Dynamic UI Support
# =============================================================================

def detect_gpu_architecture() -> dict:
    """
    Detect GPU architecture to determine available optimizations.
    Returns architecture name and capabilities.
    
    GPU Architectures:
    - Hopper (sm_90): H100, H200, H20 - supports FA2 + FA3
    - Ada Lovelace (sm_89): RTX 40xx, L40, L4 - supports FA2
    - Ampere (sm_80/86): A100, A10, RTX 30xx - supports FA2
    - Turing (sm_75): RTX 20xx, T4 - limited FA support
    - Older: No Flash Attention support
    """
    result = {
        "architecture": "unknown",
        "compute_capability": None,
        "gpu_name": None,
        "supports_fa2": False,
        "supports_fa3": False,
        "is_hopper": False,
        "is_ampere_or_newer": False
    }
    
    try:
        import torch
        if not torch.cuda.is_available():
            return result
        
        # Get GPU properties
        props = torch.cuda.get_device_properties(0)
        result["gpu_name"] = props.name
        result["compute_capability"] = f"{props.major}.{props.minor}"
        
        major, minor = props.major, props.minor
        
        # Determine architecture
        if major >= 9:
            # Hopper (H100, H200) or newer
            result["architecture"] = "hopper"
            result["is_hopper"] = True
            result["is_ampere_or_newer"] = True
            result["supports_fa2"] = True
            result["supports_fa3"] = True
        elif major == 8 and minor >= 9:
            # Ada Lovelace (RTX 40xx, L40)
            result["architecture"] = "ada_lovelace"
            result["is_ampere_or_newer"] = True
            result["supports_fa2"] = True
            result["supports_fa3"] = False
        elif major == 8:
            # Ampere (A100, A10, RTX 30xx)
            result["architecture"] = "ampere"
            result["is_ampere_or_newer"] = True
            result["supports_fa2"] = True
            result["supports_fa3"] = False
        elif major == 7 and minor >= 5:
            # Turing (RTX 20xx, T4)
            result["architecture"] = "turing"
            result["supports_fa2"] = True  # Limited support
            result["supports_fa3"] = False
        else:
            # Older architectures
            result["architecture"] = "legacy"
            result["supports_fa2"] = False
            result["supports_fa3"] = False
        
        return result
        
    except Exception as e:
        logging.warning(f"GPU architecture detection error: {e}")
        return result


def detect_available_optimizations() -> dict:
    """
    Detect which optimization packages are installed and available.
    This is used to dynamically show/hide options in the frontend.
    """
    result = {
        "flash_attention_2": False,
        "flash_attention_3": False,
        "deepspeed": False,
        "fsdp": False,  # Always available with PyTorch 2.x
        "liger_kernel": False,
        "bitsandbytes": False,
        "xformers": False
    }
    
    # Check Flash Attention 2
    try:
        from usf_bios.utils import is_flash_attn_2_available
        result["flash_attention_2"] = is_flash_attn_2_available()
    except ImportError:
        try:
            import importlib.util
            result["flash_attention_2"] = importlib.util.find_spec('flash_attn') is not None
        except:
            pass
    
    # Check Flash Attention 3
    # FA3 installs as 'flash_attn_interface' or 'flashattn_hopper' module (not 'flash_attn_3')
    try:
        from usf_bios.utils import is_flash_attn_3_available
        result["flash_attention_3"] = is_flash_attn_3_available()
    except ImportError:
        try:
            import importlib.util
            # Check for flash_attn_interface (standard FA3 installation)
            if importlib.util.find_spec('flash_attn_interface') is not None:
                result["flash_attention_3"] = True
            # Check for flashattn_hopper (alternative build)
            elif importlib.util.find_spec('flashattn_hopper') is not None:
                result["flash_attention_3"] = True
            # Legacy check for flash_attn_3 module
            elif importlib.util.find_spec('flash_attn_3') is not None:
                result["flash_attention_3"] = True
        except:
            pass
    
    # Check DeepSpeed
    try:
        import importlib.util
        result["deepspeed"] = importlib.util.find_spec('deepspeed') is not None
    except:
        pass
    
    # FSDP is always available with PyTorch 2.x
    try:
        import torch
        result["fsdp"] = hasattr(torch.distributed, 'fsdp') or int(torch.__version__.split('.')[0]) >= 2
    except:
        pass
    
    # Check Liger Kernel
    try:
        from usf_bios.utils import is_liger_available
        result["liger_kernel"] = is_liger_available()
    except ImportError:
        try:
            import importlib.util
            result["liger_kernel"] = importlib.util.find_spec('liger_kernel') is not None
        except:
            pass
    
    # Check bitsandbytes
    try:
        import importlib.util
        result["bitsandbytes"] = importlib.util.find_spec('bitsandbytes') is not None
    except:
        pass
    
    # Check xformers
    try:
        import importlib.util
        result["xformers"] = importlib.util.find_spec('xformers') is not None
    except:
        pass
    
    return result


class TrainingCapabilitiesResponse(BaseModel):
    """Response model for training capabilities detection"""
    # GPU Architecture
    gpu_architecture: str
    gpu_name: Optional[str]
    compute_capability: Optional[str]
    is_hopper: bool
    is_ampere_or_newer: bool
    
    # Available Attention Implementations
    attention_implementations: List[dict]
    default_attention: str
    
    # Distributed Training Options
    deepspeed_available: bool
    deepspeed_options: List[dict]
    fsdp_available: bool
    fsdp_options: List[dict]
    
    # Other Optimizations
    liger_kernel_available: bool
    bitsandbytes_available: bool
    xformers_available: bool
    
    # Incompatible Combinations (for frontend validation)
    incompatible_combinations: List[dict]


@router.get("/training-capabilities")
async def get_training_capabilities():
    """
    Get training optimization capabilities based on installed packages and GPU.
    
    Frontend should call this endpoint on load and use the response to:
    1. Show/hide attention implementation options based on what's installed
    2. Show/hide FA3 based on GPU architecture (Hopper only)
    3. Enforce mutual exclusion between DeepSpeed and FSDP
    4. Show warnings for incompatible combinations
    5. Set sensible defaults based on hardware
    """
    gpu_info = detect_gpu_architecture()
    optimizations = detect_available_optimizations()
    
    # Build attention implementation options based on what's available
    attention_implementations = [
        {"value": None, "label": "Auto (Recommended)", "desc": "Automatically selects best available", "available": True}
    ]
    
    # FA3 - only show if installed AND Hopper GPU
    if optimizations["flash_attention_3"] and gpu_info["is_hopper"]:
        attention_implementations.append({
            "value": "flash_attention_3",
            "label": "Flash Attention 3",
            "desc": "Latest, fastest (Hopper GPU)",
            "available": True
        })
    
    # FA2 - show if installed AND supported GPU
    if optimizations["flash_attention_2"] and gpu_info["supports_fa2"]:
        attention_implementations.append({
            "value": "flash_attention_2",
            "label": "Flash Attention 2",
            "desc": "Fast, memory efficient (Ampere+ GPU)",
            "available": True
        })
    
    # SDPA - always available with PyTorch 2.x
    attention_implementations.append({
        "value": "sdpa",
        "label": "SDPA (PyTorch)",
        "desc": "PyTorch native, good compatibility",
        "available": True
    })
    
    # Eager - always available
    attention_implementations.append({
        "value": "eager",
        "label": "Eager",
        "desc": "Standard attention, most compatible",
        "available": True
    })
    
    # Determine best default attention
    if optimizations["flash_attention_3"] and gpu_info["is_hopper"]:
        default_attention = "flash_attention_3"
    elif optimizations["flash_attention_2"] and gpu_info["supports_fa2"]:
        default_attention = "flash_attention_2"
    else:
        default_attention = "sdpa"
    
    # Build DeepSpeed options
    deepspeed_options = [
        {"value": None, "label": "Disabled", "desc": "No distributed optimization"}
    ]
    if optimizations["deepspeed"]:
        deepspeed_options.extend([
            {"value": "zero0", "label": "ZeRO-0", "desc": "DDP only, no memory optimization"},
            {"value": "zero1", "label": "ZeRO-1", "desc": "Optimizer state partitioning"},
            {"value": "zero2", "label": "ZeRO-2", "desc": "+ Gradient partitioning (recommended)"},
            {"value": "zero2_offload", "label": "ZeRO-2 + Offload", "desc": "+ CPU offload for large models"},
            {"value": "zero3", "label": "ZeRO-3", "desc": "+ Parameter partitioning (70B+ models)"},
            {"value": "zero3_offload", "label": "ZeRO-3 + Offload", "desc": "Maximum memory savings"}
        ])
    
    # Build FSDP options
    fsdp_options = [
        {"value": None, "label": "Disabled", "desc": "No FSDP"}
    ]
    if optimizations["fsdp"]:
        fsdp_options.extend([
            {"value": "full_shard", "label": "Full Shard", "desc": "Shard parameters, gradients, optimizer"},
            {"value": "shard_grad_op", "label": "Shard Grad/Op", "desc": "Shard gradients and optimizer only"},
            {"value": "fsdp2", "label": "FSDP2", "desc": "PyTorch FSDP2 (newer, recommended)"}
        ])
    
    # Define incompatible combinations for frontend validation
    incompatible_combinations = [
        {
            "id": "deepspeed_fsdp",
            "condition": "deepspeed AND fsdp",
            "message": "DeepSpeed and FSDP cannot be used together. They are both distributed training frameworks.",
            "severity": "error"
        },
        {
            "id": "packing_no_flash",
            "condition": "packing AND NOT (flash_attention_2 OR flash_attention_3)",
            "message": "Sequence Packing requires Flash Attention. Please enable Flash Attention 2 or 3.",
            "severity": "error"
        },
        {
            "id": "liger_packing",
            "condition": "liger_kernel AND packing",
            "message": "Liger Kernel may have issues with Sequence Packing. Use with caution.",
            "severity": "warning"
        },
        {
            "id": "fa3_non_hopper",
            "condition": "flash_attention_3 AND NOT is_hopper",
            "message": "Flash Attention 3 requires Hopper GPU (H100/H200). Your GPU does not support it.",
            "severity": "error"
        },
        {
            "id": "fa2_legacy_gpu",
            "condition": "flash_attention_2 AND NOT is_ampere_or_newer",
            "message": "Flash Attention 2 works best on Ampere+ GPUs. Performance may be limited.",
            "severity": "warning"
        }
    ]
    
    return {
        # GPU Info
        "gpu_architecture": gpu_info["architecture"],
        "gpu_name": gpu_info["gpu_name"],
        "compute_capability": gpu_info["compute_capability"],
        "is_hopper": gpu_info["is_hopper"],
        "is_ampere_or_newer": gpu_info["is_ampere_or_newer"],
        
        # Attention
        "attention_implementations": attention_implementations,
        "default_attention": default_attention,
        
        # Distributed Training
        "deepspeed_available": optimizations["deepspeed"],
        "deepspeed_options": deepspeed_options,
        "fsdp_available": optimizations["fsdp"],
        "fsdp_options": fsdp_options,
        
        # Other Optimizations
        "liger_kernel_available": optimizations["liger_kernel"],
        "bitsandbytes_available": optimizations["bitsandbytes"],
        "xformers_available": optimizations["xformers"],
        
        # Validation Rules
        "incompatible_combinations": incompatible_combinations
    }


@router.post("/optimize-config")
async def optimize_training_config(
    model_size_gb: float = 0,
    model_params_b: float = 0,  # In billions (e.g., 7 for 7B model)
    gpu_memory_gb: float = 0,
    training_method: str = "sft",  # sft, pt, rlhf
    train_type: str = "lora",  # lora, qlora, adalora, full
    rlhf_type: str = None,  # dpo, ppo, grpo, etc.
    dataset_samples: int = 1000,
    max_seq_length: int = 2048
):
    """
    Smart training configuration optimizer.
    
    Analyzes model size, GPU memory, training type, and dataset to suggest
    the best optimized training settings.
    
    Returns recommended:
    - batch_size
    - gradient_accumulation
    - learning_rate
    - epochs
    - lora_rank (if applicable)
    - lora_alpha (if applicable)
    - quantization settings
    - deepspeed/fsdp settings
    - attention implementation
    """
    
    # Get GPU info if not provided
    if gpu_memory_gb <= 0:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except:
            gpu_memory_gb = 24  # Default assumption
    
    # Estimate model params from size if not provided
    if model_params_b <= 0 and model_size_gb > 0:
        # Rough estimate: 1B params ≈ 2GB in fp16
        model_params_b = model_size_gb / 2
    
    # If model_params not known, assume from common sizes
    if model_params_b <= 0:
        model_params_b = 7  # Default 7B
    
    # ========================================
    # MEMORY ESTIMATION
    # ========================================
    # Model memory = params * bytes_per_param
    # - fp32: 4 bytes, fp16/bf16: 2 bytes, int8: 1 byte, int4: 0.5 bytes
    # - Optimizer states: 8-12 bytes per param (AdamW)
    # - Gradients: 2-4 bytes per param
    # - Activations: varies with batch size and seq length
    
    bytes_per_param = {
        "full": 2,      # bf16/fp16
        "lora": 2,      # bf16/fp16 for base, lora adds small overhead
        "qlora": 0.5,   # 4-bit quantized
        "adalora": 2,   # Similar to lora
    }
    
    base_model_memory_gb = model_params_b * bytes_per_param.get(train_type, 2)
    
    # For QLoRA, base model is quantized
    if train_type == "qlora":
        base_model_memory_gb = model_params_b * 0.5  # 4-bit
    
    # Available memory for training (leave ~20% buffer)
    available_memory_gb = gpu_memory_gb * 0.8 - base_model_memory_gb
    
    # ========================================
    # DETERMINE OPTIMAL SETTINGS
    # ========================================
    
    result = {
        "recommended": {},
        "reasoning": [],
        "warnings": [],
        "gpu_utilization_estimate": 0,
        "training_time_estimate": None
    }
    
    # --- Training Type Recommendations ---
    recommended_train_type = train_type
    
    # If full fine-tuning but not enough memory, suggest LoRA/QLoRA
    if train_type == "full":
        full_memory_needed = model_params_b * 16  # fp16 + optimizer + gradients
        if full_memory_needed > gpu_memory_gb * 0.9:
            if gpu_memory_gb >= model_params_b * 2:
                recommended_train_type = "lora"
                result["warnings"].append(f"Full fine-tuning needs ~{full_memory_needed:.0f}GB. Recommending LoRA instead.")
            else:
                recommended_train_type = "qlora"
                result["warnings"].append(f"Full fine-tuning needs ~{full_memory_needed:.0f}GB. Recommending QLoRA (4-bit) instead.")
    
    result["recommended"]["train_type"] = recommended_train_type
    
    # --- Batch Size Calculation ---
    # Memory per sample ≈ seq_length * hidden_dim * 4 bytes * num_layers / 1024^3 GB
    # Simplified: ~0.5-2MB per sample for 7B model at 2048 seq length
    memory_per_sample_mb = (max_seq_length / 2048) * (model_params_b / 7) * 1.5
    
    if recommended_train_type == "qlora":
        memory_per_sample_mb *= 0.6  # QLoRA is more memory efficient
    elif recommended_train_type == "full":
        memory_per_sample_mb *= 2.5  # Full fine-tuning needs more
    
    max_batch_size = max(1, int((available_memory_gb * 1024) / memory_per_sample_mb))
    
    # Cap batch size at reasonable values
    if max_batch_size > 32:
        max_batch_size = 32
    elif max_batch_size > 16:
        max_batch_size = 16
    elif max_batch_size > 8:
        max_batch_size = 8
    elif max_batch_size > 4:
        max_batch_size = 4
    elif max_batch_size > 2:
        max_batch_size = 2
    else:
        max_batch_size = 1
    
    result["recommended"]["batch_size"] = max_batch_size
    result["reasoning"].append(f"Batch size {max_batch_size} based on ~{available_memory_gb:.1f}GB available memory")
    
    # --- Gradient Accumulation ---
    # Target effective batch size: 16-64 for most cases
    target_effective_batch = 32 if dataset_samples > 5000 else 16
    
    if training_method == "rlhf" and rlhf_type in ["ppo", "grpo"]:
        target_effective_batch = 8  # RLHF methods work better with smaller batches
    
    grad_accum = max(1, target_effective_batch // max_batch_size)
    if grad_accum > 32:
        grad_accum = 32
    
    result["recommended"]["gradient_accumulation"] = grad_accum
    result["recommended"]["effective_batch_size"] = max_batch_size * grad_accum
    result["reasoning"].append(f"Gradient accumulation {grad_accum} for effective batch size {max_batch_size * grad_accum}")
    
    # --- Learning Rate ---
    # Base LR depends on training type and model size
    if recommended_train_type == "full":
        base_lr = 2e-5 if model_params_b < 13 else 1e-5
    elif recommended_train_type in ["lora", "adalora"]:
        base_lr = 2e-4 if model_params_b < 13 else 1e-4
    else:  # qlora
        base_lr = 2e-4
    
    # Adjust for RLHF
    if training_method == "rlhf":
        if rlhf_type == "dpo":
            base_lr = 5e-7 if recommended_train_type == "full" else 1e-5
        elif rlhf_type == "ppo":
            base_lr = 1e-6 if recommended_train_type == "full" else 5e-6
        elif rlhf_type == "grpo":
            base_lr = 1e-6 if recommended_train_type == "full" else 1e-5
    
    result["recommended"]["learning_rate"] = base_lr
    result["reasoning"].append(f"Learning rate {base_lr:.0e} optimized for {training_method}/{recommended_train_type}")
    
    # --- Epochs ---
    # Smaller datasets need more epochs, larger need fewer
    if dataset_samples < 500:
        epochs = 5
    elif dataset_samples < 2000:
        epochs = 3
    elif dataset_samples < 10000:
        epochs = 2
    else:
        epochs = 1
    
    # RLHF typically needs more epochs
    if training_method == "rlhf":
        epochs = min(5, epochs + 1)
    
    result["recommended"]["epochs"] = epochs
    result["reasoning"].append(f"{epochs} epochs for {dataset_samples} samples")
    
    # --- LoRA Settings ---
    if recommended_train_type in ["lora", "qlora", "adalora"]:
        # LoRA rank: larger models can use lower ranks
        if model_params_b >= 70:
            lora_rank = 16
            lora_alpha = 32
        elif model_params_b >= 30:
            lora_rank = 32
            lora_alpha = 64
        elif model_params_b >= 13:
            lora_rank = 32
            lora_alpha = 64
        else:  # 7B and smaller
            lora_rank = 64
            lora_alpha = 128
        
        # For QLoRA, can use slightly lower ranks
        if recommended_train_type == "qlora":
            lora_rank = max(8, lora_rank // 2)
            lora_alpha = lora_rank * 2
        
        result["recommended"]["lora_rank"] = lora_rank
        result["recommended"]["lora_alpha"] = lora_alpha
        result["recommended"]["lora_dropout"] = 0.05
        result["recommended"]["target_modules"] = "all-linear"
        result["reasoning"].append(f"LoRA rank {lora_rank}, alpha {lora_alpha} for {model_params_b:.0f}B model")
    
    # --- Quantization (QLoRA) ---
    if recommended_train_type == "qlora":
        result["recommended"]["quant_bits"] = 4
        result["reasoning"].append("4-bit quantization for memory efficiency")
    
    # --- Max Sequence Length ---
    # Adjust based on available memory
    if available_memory_gb < 4:
        recommended_seq_length = min(max_seq_length, 1024)
    elif available_memory_gb < 8:
        recommended_seq_length = min(max_seq_length, 2048)
    elif available_memory_gb < 16:
        recommended_seq_length = min(max_seq_length, 4096)
    else:
        recommended_seq_length = max_seq_length
    
    result["recommended"]["max_length"] = recommended_seq_length
    
    # --- DeepSpeed / FSDP ---
    if model_params_b >= 70:
        result["recommended"]["deepspeed"] = "zero3_offload"
        result["reasoning"].append("ZeRO-3 + Offload recommended for 70B+ models")
    elif model_params_b >= 30:
        result["recommended"]["deepspeed"] = "zero3"
        result["reasoning"].append("ZeRO-3 recommended for 30B+ models")
    elif model_params_b >= 13 and recommended_train_type == "full":
        result["recommended"]["deepspeed"] = "zero2"
        result["reasoning"].append("ZeRO-2 recommended for full fine-tuning of 13B+ models")
    else:
        result["recommended"]["deepspeed"] = None
    
    # --- Attention Implementation ---
    # Check GPU capabilities
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            compute_cap = props.major * 10 + props.minor
            
            if compute_cap >= 90:  # Hopper
                result["recommended"]["attention"] = "flash_attention_3"
                result["reasoning"].append("Flash Attention 3 for Hopper GPU")
            elif compute_cap >= 80:  # Ampere
                result["recommended"]["attention"] = "flash_attention_2"
                result["reasoning"].append("Flash Attention 2 for Ampere+ GPU")
            else:
                result["recommended"]["attention"] = "sdpa"
                result["reasoning"].append("SDPA attention for older GPU")
    except:
        result["recommended"]["attention"] = "sdpa"
    
    # --- Warmup Ratio ---
    result["recommended"]["warmup_ratio"] = 0.03 if dataset_samples > 1000 else 0.1
    
    # --- RLHF-specific settings ---
    if training_method == "rlhf" and rlhf_type:
        if rlhf_type == "dpo":
            result["recommended"]["beta"] = 0.1
            result["reasoning"].append("DPO beta=0.1 (standard)")
        elif rlhf_type == "kto":
            result["recommended"]["beta"] = 0.1
            result["recommended"]["desirable_weight"] = 1.0
            result["recommended"]["undesirable_weight"] = 1.0
        elif rlhf_type == "ppo":
            result["recommended"]["num_ppo_epochs"] = 4
            result["recommended"]["kl_coef"] = 0.05
            result["recommended"]["cliprange"] = 0.2
        elif rlhf_type == "grpo":
            result["recommended"]["num_generations"] = 8 if gpu_memory_gb > 40 else 4
            result["reasoning"].append(f"GRPO generations={result['recommended']['num_generations']} based on memory")
    
    # --- Estimated GPU Utilization ---
    estimated_memory_used = base_model_memory_gb + (max_batch_size * memory_per_sample_mb / 1024)
    result["gpu_utilization_estimate"] = min(95, int((estimated_memory_used / gpu_memory_gb) * 100))
    
    # --- Training Time Estimate ---
    # Rough estimate: 1 sample/sec for 7B, scales with model size
    samples_per_sec = 1.0 / (model_params_b / 7) * max_batch_size
    total_samples = dataset_samples * epochs
    total_steps = total_samples / (max_batch_size * grad_accum)
    estimated_seconds = total_steps / samples_per_sec * (max_batch_size * grad_accum)
    
    if estimated_seconds < 3600:
        result["training_time_estimate"] = f"~{int(estimated_seconds / 60)} minutes"
    else:
        result["training_time_estimate"] = f"~{estimated_seconds / 3600:.1f} hours"
    
    return result


# =============================================================================
# GPU Memory Cleanup Endpoints
# =============================================================================

class GPUCleanupResponse(BaseModel):
    """Response model for GPU cleanup operations"""
    success: bool
    memory_freed_gb: float = 0.0
    memory_used_gb: float = 0.0
    total_memory_gb: float = 0.0
    cleanup_type: str = "unknown"
    message: Optional[str] = None
    error: Optional[str] = None
    details: Optional[dict] = None


@router.post("/gpu/cleanup", response_model=GPUCleanupResponse)
async def cleanup_gpu_memory(
    deep: bool = True,
    kill_orphans: bool = True
):
    """
    Manually trigger GPU memory cleanup.
    
    Use this endpoint when:
    - Training failed and GPU memory wasn't released
    - Before starting a new training to ensure clean state
    - After inference to free VRAM for training
    - When VRAM usage shows high even with no active jobs
    
    Args:
        deep: If True, performs comprehensive cleanup including orphan process killing.
              If False, performs quick cleanup (just GC and cache clear).
        kill_orphans: If True, kills orphaned CUDA processes consuming GPU memory.
    
    Returns:
        GPUCleanupResponse with memory freed and current state.
    """
    try:
        if deep:
            result = await gpu_cleanup_service.async_deep_cleanup(
                kill_orphans=kill_orphans,
                exclude_pids=[os.getpid()]
            )
            
            if result.get("success"):
                mem_after = result.get("memory_after", {})
                return GPUCleanupResponse(
                    success=True,
                    memory_freed_gb=result.get("memory_freed_gb", 0.0),
                    memory_used_gb=mem_after.get("total_used_gb", 0.0),
                    total_memory_gb=mem_after.get("total_memory_gb", 0.0),
                    cleanup_type="deep",
                    message=f"Successfully freed {result.get('memory_freed_gb', 0):.2f}GB VRAM",
                    details={
                        "steps": result.get("steps", []),
                        "gpus": mem_after.get("gpus", [])
                    }
                )
            else:
                return GPUCleanupResponse(
                    success=False,
                    cleanup_type="deep",
                    error=result.get("error", "Unknown cleanup error")
                )
        else:
            result = await gpu_cleanup_service.async_quick_cleanup()
            
            if result.get("success"):
                mem_info = result.get("memory_info", {})
                return GPUCleanupResponse(
                    success=True,
                    memory_used_gb=mem_info.get("total_used_gb", 0.0),
                    total_memory_gb=mem_info.get("total_memory_gb", 0.0),
                    cleanup_type="quick",
                    message="Quick cleanup completed"
                )
            else:
                return GPUCleanupResponse(
                    success=False,
                    cleanup_type="quick",
                    error=result.get("error", "Unknown cleanup error")
                )
    
    except Exception as e:
        return GPUCleanupResponse(
            success=False,
            cleanup_type="deep" if deep else "quick",
            error=str(e)
        )


@router.get("/gpu/memory")
async def get_gpu_memory_status():
    """
    Get detailed GPU memory status for all GPUs.
    
    Returns:
        - available: Whether GPU is available
        - device_count: Number of GPUs
        - gpus: List of per-GPU memory info
        - total_used_gb: Total VRAM used across all GPUs
        - total_free_gb: Total VRAM free across all GPUs
        - total_memory_gb: Total VRAM capacity
    """
    return gpu_cleanup_service.get_gpu_memory_info()


@router.post("/gpu/cleanup/quick")
async def quick_cleanup_gpu():
    """
    Perform a quick GPU memory cleanup.
    
    This is a lightweight cleanup that:
    - Runs Python garbage collection
    - Clears PyTorch CUDA cache
    
    Use this for routine cleanup between operations.
    For more thorough cleanup (e.g., after training failures), use POST /gpu/cleanup with deep=True.
    """
    result = await gpu_cleanup_service.async_quick_cleanup()
    return result


@router.post("/gpu/cleanup/deep")
async def deep_cleanup_gpu():
    """
    Perform a deep GPU memory cleanup.
    
    This is a comprehensive cleanup that:
    - Runs multiple passes of garbage collection
    - Clears PyTorch CUDA cache on all GPUs
    - Kills orphaned CUDA processes
    - Resets CUDA memory statistics
    - Performs IPC cleanup for multi-process scenarios
    
    Use this after training failures or when VRAM is stuck at high usage.
    """
    result = await gpu_cleanup_service.async_deep_cleanup(
        kill_orphans=True,
        exclude_pids=[os.getpid()]
    )
    return result


# ============================================================================
# vLLM Server Validation for Online RL (GRPO/PPO/GKD)
# ============================================================================

class VLLMServerTestRequest(BaseModel):
    """Request to test vLLM server connectivity for RL training"""
    host: str
    port: int = 8000
    # NCCL group port for weight sync (required for online RL)
    group_port: int = 51216


class VLLMServerTestResponse(BaseModel):
    """Response from vLLM server test"""
    success: bool
    verified: bool = False
    message: str
    error: Optional[str] = None
    # What was tested and results
    tests_passed: List[str] = []
    tests_failed: List[str] = []
    # Sample output showing the required keys
    sample_output: Optional[dict] = None
    # Server capabilities
    server_type: Optional[str] = None  # "standard_vllm" or "rl_rollout"
    supports_weight_sync: bool = False


@router.post("/vllm/test", response_model=VLLMServerTestResponse)
async def test_vllm_server(request: VLLMServerTestRequest):
    """
    Test vLLM server for RL training compatibility.
    
    Simple test: Send sample input → Check output has required keys (logprobs).
    If it works with sample, it will work during training.
    
    SECURITY: Must pass before training can start in server mode.
    """
    host = request.host.strip()
    port = request.port
    base_url = f"http://{host}:{port}"
    
    tests_passed = []
    tests_failed = []
    sample_output = {}
    
    # Validate host format
    if not host or len(host) > 255:
        return VLLMServerTestResponse(
            success=False,
            verified=False,
            message="Invalid host format",
            error="Host must be a valid IP address or hostname"
        )
    
    try:
        # Step 1: Can we connect?
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(5)
            result = sock.connect_ex((host, port))
            sock.close()
            if result != 0:
                return VLLMServerTestResponse(
                    success=False,
                    verified=False,
                    message=f"Cannot connect to {host}:{port}",
                    error="Server not reachable. Check IP/port and ensure vLLM is running.",
                    tests_failed=["connection"]
                )
            tests_passed.append("connection")
        except socket.gaierror:
            return VLLMServerTestResponse(
                success=False,
                verified=False,
                message=f"Cannot resolve hostname: {host}",
                error="DNS resolution failed. Check the hostname.",
                tests_failed=["connection"]
            )
        except socket.timeout:
            return VLLMServerTestResponse(
                success=False,
                verified=False,
                message=f"Connection timeout to {host}:{port}",
                error="Server not responding.",
                tests_failed=["connection"]
            )
        
        async with httpx.AsyncClient(timeout=30.0) as client:
            # Step 2: Get model name
            try:
                models_resp = await client.get(f"{base_url}/v1/models")
                if models_resp.status_code != 200:
                    return VLLMServerTestResponse(
                        success=False,
                        verified=False,
                        message="Cannot get models from server",
                        error=f"/v1/models returned status {models_resp.status_code}",
                        tests_passed=tests_passed,
                        tests_failed=["models_endpoint"]
                    )
                models_data = models_resp.json()
                if "data" not in models_data or len(models_data["data"]) == 0:
                    return VLLMServerTestResponse(
                        success=False,
                        verified=False,
                        message="No models loaded on server",
                        error="Start vLLM with a model loaded.",
                        tests_passed=tests_passed,
                        tests_failed=["no_models"]
                    )
                model_id = models_data["data"][0].get("id", "unknown")
                sample_output["model"] = model_id
                tests_passed.append("model_available")
            except Exception as e:
                return VLLMServerTestResponse(
                    success=False,
                    verified=False,
                    message="Cannot reach vLLM API",
                    error=str(e),
                    tests_passed=tests_passed,
                    tests_failed=["api_unreachable"]
                )
            
            # Step 3: THE MAIN TEST - Send sample input, check output has logprobs
            # This is what matters for RL training
            try:
                test_payload = {
                    "model": model_id,
                    "messages": [
                        {"role": "user", "content": "What is 2+2? Answer with just the number."}
                    ],
                    "max_tokens": 10,
                    "temperature": 0,
                    "logprobs": True,  # REQUIRED for RL training
                    "top_logprobs": 5
                }
                
                response = await client.post(
                    f"{base_url}/v1/chat/completions",
                    json=test_payload,
                    timeout=60.0
                )
                
                if response.status_code != 200:
                    return VLLMServerTestResponse(
                        success=False,
                        verified=False,
                        message="Generation request failed",
                        error=f"Server returned status {response.status_code}. Check server logs.",
                        tests_passed=tests_passed,
                        tests_failed=["generation_failed"]
                    )
                
                data = response.json()
                
                # Check required keys exist
                missing_keys = []
                
                # Must have choices
                if "choices" not in data or len(data["choices"]) == 0:
                    missing_keys.append("choices")
                
                # Must have usage (token counts)
                if "usage" not in data:
                    missing_keys.append("usage")
                else:
                    usage = data["usage"]
                    sample_output["usage"] = usage
                    for key in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                        if key not in usage:
                            missing_keys.append(f"usage.{key}")
                
                if missing_keys:
                    return VLLMServerTestResponse(
                        success=False,
                        verified=False,
                        message="Response missing required keys",
                        error=f"Missing: {', '.join(missing_keys)}",
                        tests_passed=tests_passed,
                        tests_failed=["missing_keys"],
                        sample_output=sample_output
                    )
                tests_passed.append("response_format")
                
                # Check logprobs - CRITICAL for RL training
                choice = data["choices"][0]
                sample_output["content"] = choice.get("message", {}).get("content", "")[:100]
                
                if "logprobs" not in choice or choice["logprobs"] is None:
                    return VLLMServerTestResponse(
                        success=False,
                        verified=False,
                        message="Server does not return logprobs",
                        error="logprobs is required for RL training. The server returned None for logprobs. Check vLLM configuration.",
                        tests_passed=tests_passed,
                        tests_failed=["no_logprobs"],
                        sample_output=sample_output
                    )
                
                logprobs = choice["logprobs"]
                if "content" not in logprobs or logprobs["content"] is None or len(logprobs["content"]) == 0:
                    return VLLMServerTestResponse(
                        success=False,
                        verified=False,
                        message="logprobs.content is empty",
                        error="Server returned logprobs but content is empty. This won't work for RL training.",
                        tests_passed=tests_passed,
                        tests_failed=["empty_logprobs"],
                        sample_output=sample_output
                    )
                
                # Check logprobs structure has required keys
                token_info = logprobs["content"][0]
                logprobs_required = ["token", "logprob"]
                logprobs_missing = [k for k in logprobs_required if k not in token_info]
                
                if logprobs_missing:
                    return VLLMServerTestResponse(
                        success=False,
                        verified=False,
                        message="logprobs missing required keys",
                        error=f"Each token needs: {', '.join(logprobs_required)}. Missing: {', '.join(logprobs_missing)}",
                        tests_passed=tests_passed,
                        tests_failed=["logprobs_keys"],
                        sample_output=sample_output
                    )
                
                # SUCCESS - logprobs work!
                tests_passed.append("logprobs")
                sample_output["logprobs_sample"] = {
                    "token": token_info["token"],
                    "logprob": token_info["logprob"],
                    "top_logprobs_count": len(token_info.get("top_logprobs", []))
                }
                
            except httpx.TimeoutException:
                return VLLMServerTestResponse(
                    success=False,
                    verified=False,
                    message="Request timed out",
                    error="Server took too long to respond (>60s). May be overloaded.",
                    tests_passed=tests_passed,
                    tests_failed=["timeout"]
                )
            except Exception as e:
                return VLLMServerTestResponse(
                    success=False,
                    verified=False,
                    message="Generation test failed",
                    error=str(e),
                    tests_passed=tests_passed,
                    tests_failed=["generation_error"]
                )
        
            # Step 4: Check for RL-specific endpoints (weight sync capability)
            # These are required for online RL training
            server_type = "standard_vllm"
            supports_weight_sync = False
            
            try:
                # Check if this is an RL rollout server with weight sync endpoints
                # Try /health/ (RL rollout server uses trailing slash)
                rl_health = await client.get(f"{base_url}/health/", timeout=5.0)
                if rl_health.status_code == 200:
                    # Check for get_engine_type (RL-specific endpoint)
                    engine_type_resp = await client.post(f"{base_url}/get_engine_type/", timeout=5.0)
                    if engine_type_resp.status_code == 200:
                        server_type = "rl_rollout"
                        supports_weight_sync = True
                        tests_passed.append("rl_endpoints")
                        sample_output["server_type"] = "rl_rollout"
                        sample_output["weight_sync_available"] = True
            except Exception:
                # Standard vLLM server without RL extensions
                pass
            
            if not supports_weight_sync:
                sample_output["server_type"] = "standard_vllm"
                sample_output["weight_sync_available"] = False
                sample_output["warning"] = (
                    "Server does not have RL weight sync endpoints. "
                    "You must start vLLM with: usf-bios rollout --vllm_mode server"
                )
                tests_failed.append("no_weight_sync")
                return VLLMServerTestResponse(
                    success=False,
                    verified=False,
                    message="Server missing RL weight sync endpoints",
                    error=(
                        "This vLLM server does not support weight updates. "
                        "For online RL, start the server with: "
                        "usf-bios rollout --model <model> --vllm_mode server --port 8000"
                    ),
                    tests_passed=tests_passed,
                    tests_failed=tests_failed,
                    sample_output=sample_output,
                    server_type=server_type,
                    supports_weight_sync=False
                )
        
        # All tests passed!
        return VLLMServerTestResponse(
            success=True,
            verified=True,
            message=f"Server ready for RL training with weight sync",
            tests_passed=tests_passed,
            sample_output=sample_output,
            server_type=server_type,
            supports_weight_sync=supports_weight_sync
        )
        
    except Exception as e:
        return VLLMServerTestResponse(
            success=False,
            verified=False,
            message="Test failed",
            error=str(e),
            tests_failed=["unknown_error"]
        )


class GPUResourceCheckRequest(BaseModel):
    """Request to check if GPU resources are sufficient for colocate mode"""
    model_path: str
    train_type: str = "lora"  # lora, qlora, full
    rlhf_type: str = "grpo"
    max_length: int = 2048
    per_device_train_batch_size: int = 1
    vllm_gpu_memory_utilization: float = 0.9


class GPUResourceCheckResponse(BaseModel):
    """Response from GPU resource check"""
    sufficient: bool
    can_use_colocate: bool
    message: str
    details: dict = {}
    recommendations: List[str] = []


@router.post("/gpu/resource-check", response_model=GPUResourceCheckResponse)
async def check_gpu_resources_for_colocate(request: GPUResourceCheckRequest):
    """
    Check if GPU resources are sufficient for colocate mode (training + vLLM on same GPUs).
    
    This validates:
    1. Available GPU memory
    2. Estimated model memory requirements
    3. vLLM KV cache requirements
    4. Training overhead (gradients, optimizer states)
    
    SECURITY: This check is enforced by backend. Frontend cannot bypass.
    """
    import torch
    
    details = {
        "gpu_count": 0,
        "total_gpu_memory_gb": 0,
        "available_gpu_memory_gb": 0,
        "estimated_model_memory_gb": 0,
        "estimated_training_overhead_gb": 0,
        "estimated_vllm_memory_gb": 0,
        "total_required_gb": 0,
    }
    recommendations = []
    
    try:
        if not torch.cuda.is_available():
            return GPUResourceCheckResponse(
                sufficient=False,
                can_use_colocate=False,
                message="No GPU available",
                details=details,
                recommendations=["Install NVIDIA drivers and CUDA toolkit"]
            )
        
        gpu_count = torch.cuda.device_count()
        details["gpu_count"] = gpu_count
        
        # Get total GPU memory
        total_memory_gb = 0
        available_memory_gb = 0
        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            total_memory_gb += props.total_memory / (1024**3)
            # Get current free memory
            free_mem, total_mem = torch.cuda.mem_get_info(i)
            available_memory_gb += free_mem / (1024**3)
        
        details["total_gpu_memory_gb"] = round(total_memory_gb, 2)
        details["available_gpu_memory_gb"] = round(available_memory_gb, 2)
        
        # Estimate model memory (rough heuristics based on model path)
        model_path_lower = request.model_path.lower()
        estimated_params_b = 7  # Default 7B
        
        # Parse model size from path
        if "70b" in model_path_lower or "72b" in model_path_lower:
            estimated_params_b = 70
        elif "32b" in model_path_lower or "34b" in model_path_lower:
            estimated_params_b = 32
        elif "14b" in model_path_lower or "13b" in model_path_lower:
            estimated_params_b = 14
        elif "8b" in model_path_lower or "7b" in model_path_lower:
            estimated_params_b = 7
        elif "3b" in model_path_lower or "4b" in model_path_lower:
            estimated_params_b = 4
        elif "1b" in model_path_lower or "2b" in model_path_lower:
            estimated_params_b = 2
        elif "0.5b" in model_path_lower or "500m" in model_path_lower:
            estimated_params_b = 0.5
        
        # Memory per parameter depends on dtype and training type
        if request.train_type == "qlora":
            bytes_per_param = 0.5  # 4-bit quantized
            training_multiplier = 1.5  # Lower overhead for QLoRA
        elif request.train_type == "lora":
            bytes_per_param = 2  # bfloat16
            training_multiplier = 2  # LoRA adapters + activations
        else:  # full
            bytes_per_param = 2  # bfloat16
            training_multiplier = 4  # Full gradients + optimizer states
        
        model_memory_gb = (estimated_params_b * bytes_per_param)
        training_overhead_gb = model_memory_gb * (training_multiplier - 1)
        
        # vLLM memory estimate (KV cache + model)
        vllm_memory_gb = model_memory_gb * request.vllm_gpu_memory_utilization
        
        details["estimated_model_memory_gb"] = round(model_memory_gb, 2)
        details["estimated_training_overhead_gb"] = round(training_overhead_gb, 2)
        details["estimated_vllm_memory_gb"] = round(vllm_memory_gb, 2)
        
        # Total required for colocate mode
        # Note: In colocate mode, model is shared, but we need memory for both purposes
        # With sleep_level, vLLM releases memory during training and vice versa
        total_required_gb = model_memory_gb + max(training_overhead_gb, vllm_memory_gb)
        details["total_required_gb"] = round(total_required_gb, 2)
        
        # Check if sufficient
        # We need at least 2 GPUs for colocate mode
        if gpu_count < 2:
            return GPUResourceCheckResponse(
                sufficient=False,
                can_use_colocate=False,
                message="Colocate mode requires at least 2 GPUs",
                details=details,
                recommendations=[
                    "Use server mode with external vLLM server",
                    "Add more GPUs to your system"
                ]
            )
        
        # Check memory
        if available_memory_gb < total_required_gb:
            recommendations.append(f"Estimated {total_required_gb:.1f}GB required, only {available_memory_gb:.1f}GB available")
            recommendations.append("Use QLoRA instead of LoRA to reduce memory")
            recommendations.append("Enable offload_model and offload_optimizer")
            recommendations.append("Set sleep_level to 1 or 2")
            recommendations.append("Or use server mode with dedicated inference GPU")
            
            return GPUResourceCheckResponse(
                sufficient=False,
                can_use_colocate=False,
                message="Insufficient GPU memory for colocate mode",
                details=details,
                recommendations=recommendations
            )
        
        # Memory is sufficient
        if available_memory_gb < total_required_gb * 1.2:
            recommendations.append("Memory is tight - consider enabling memory optimization options")
        
        return GPUResourceCheckResponse(
            sufficient=True,
            can_use_colocate=True,
            message=f"GPU resources sufficient for colocate mode ({gpu_count} GPUs, {available_memory_gb:.1f}GB available)",
            details=details,
            recommendations=recommendations
        )
        
    except Exception as e:
        return GPUResourceCheckResponse(
            sufficient=False,
            can_use_colocate=False,
            message=f"GPU resource check failed: {str(e)}",
            details=details,
            recommendations=["Check GPU drivers and CUDA installation"]
        )


# =============================================================================
# TRAINING STATUS ENDPOINT - Real-time training state management
# =============================================================================

@router.get("/training-status")
async def get_training_status():
    """
    Get comprehensive training status for the entire system.
    
    This is the SINGLE SOURCE OF TRUTH for training state. Frontend should:
    1. Call this on page load to detect active training
    2. Poll this during training for real-time updates
    3. Use can_create_job/can_load_inference to control UI
    
    Response:
    {
        "is_training_active": true/false,
        "phase": "idle" | "initializing" | "running" | "completing" | "completed" | "failed" | "stopped",
        "job_id": "abc123" | null,
        "job_name": "swift-phoenix-42" | null,
        "model_name": "llama-3-8b" | null,
        "started_at": "2024-01-15T10:30:00Z" | null,
        "progress": {
            "current_step": 100,
            "total_steps": 1000,
            "current_epoch": 1.5,
            "total_epochs": 3,
            "current_loss": 0.5,
            "learning_rate": 0.0001,
            "samples_per_second": 2.5,
            "eta_seconds": 3600,
            "progress_percent": 10.0
        },
        "error_message": null | "Out of memory",
        "can_create_job": true/false,
        "can_load_inference": true/false,
        "can_start_training": true/false,
        "process_running": true/false,
        "process_pid": 12345 | null,
        "last_updated": "2024-01-15T10:35:00Z",
        "status_message": "Training in progress: 10.0% (100/1000)",
        "status_color": "blue"
    }
    """
    from ...services.training_status_service import training_status_service
    
    return await training_status_service.get_status_dict()


@router.get("/training-status/can-create-job")
async def can_create_job():
    """
    Check if a new training job can be created.
    
    Returns:
    {
        "allowed": true/false,
        "reason": "Ready to create job" | "Training is in progress..."
    }
    """
    from ...services.training_status_service import training_status_service
    
    can_create, reason = await training_status_service.can_create_job()
    return {"allowed": can_create, "reason": reason}


@router.get("/training-status/can-load-inference")
async def can_load_inference():
    """
    Check if inference model can be loaded.
    
    Returns:
    {
        "allowed": true/false,
        "reason": "Ready to load inference model" | "Cannot load while training..."
    }
    """
    from ...services.training_status_service import training_status_service
    
    can_load, reason = await training_status_service.can_load_inference()
    return {"allowed": can_load, "reason": reason}
