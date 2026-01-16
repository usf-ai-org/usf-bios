"""System metrics and status endpoints"""
from fastapi import APIRouter
from pydantic import BaseModel
from typing import Literal, Optional
import os
import subprocess

router = APIRouter(prefix="/api/system", tags=["system"])


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
        return True, "BIOS training packages installed"
    except ImportError as e:
        return False, f"Missing package: {str(e)}"
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
        return False, f"GPU check failed: {str(e)}", None


def check_cuda_available() -> bool:
    """Check if CUDA is available"""
    try:
        import torch
        return torch.cuda.is_available()
    except:
        return False


def get_gpu_metrics():
    """Get GPU metrics using pynvml or fallback"""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        # Utilization
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        gpu_utilization = util.gpu
        
        # Memory
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_memory_used = mem_info.used / (1024**3)  # GB
        gpu_memory_total = mem_info.total / (1024**3)  # GB
        
        # Temperature
        try:
            temperature = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except:
            temperature = 0
        
        pynvml.nvmlShutdown()
        
        return {
            "gpu_utilization": gpu_utilization,
            "gpu_memory_used": round(gpu_memory_used, 2),
            "gpu_memory_total": round(gpu_memory_total, 2),
            "gpu_temperature": temperature
        }
    except ImportError:
        # pynvml not available, try torch
        try:
            import torch
            if torch.cuda.is_available():
                gpu_memory_used = torch.cuda.memory_allocated() / (1024**3)
                gpu_memory_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                return {
                    "gpu_utilization": 0,  # Not available without pynvml
                    "gpu_memory_used": round(gpu_memory_used, 2),
                    "gpu_memory_total": round(gpu_memory_total, 2),
                    "gpu_temperature": 0
                }
        except:
            pass
        
        return {
            "gpu_utilization": 0,
            "gpu_memory_used": 0,
            "gpu_memory_total": 0,
            "gpu_temperature": 0
        }
    except Exception as e:
        return {
            "gpu_utilization": 0,
            "gpu_memory_used": 0,
            "gpu_memory_total": 0,
            "gpu_temperature": 0,
            "error": str(e)
        }


def get_cpu_metrics():
    """Get CPU and RAM metrics"""
    try:
        import psutil
        
        cpu_percent = psutil.cpu_percent(interval=0.1)
        
        mem = psutil.virtual_memory()
        ram_used = mem.used / (1024**3)  # GB
        ram_total = mem.total / (1024**3)  # GB
        
        return {
            "cpu_percent": round(cpu_percent, 1),
            "ram_used": round(ram_used, 2),
            "ram_total": round(ram_total, 2)
        }
    except ImportError:
        return {
            "cpu_percent": 0,
            "ram_used": 0,
            "ram_total": 0
        }
    except Exception as e:
        return {
            "cpu_percent": 0,
            "ram_used": 0,
            "ram_total": 0,
            "error": str(e)
        }


@router.get("/metrics")
async def get_system_metrics():
    """Get comprehensive system metrics including GPU and CPU"""
    gpu_metrics = get_gpu_metrics()
    cpu_metrics = get_cpu_metrics()
    
    return {
        **gpu_metrics,
        **cpu_metrics
    }


@router.get("/gpu")
async def get_gpu_info():
    """Get GPU-specific information"""
    return get_gpu_metrics()


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
