import gc
import os
import shutil
from typing import Optional, Dict, Any
from datetime import datetime

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except:
    NVML_AVAILABLE = False


class GPUService:
    
    @staticmethod
    def get_gpu_info() -> Dict[str, Any]:
        if not NVML_AVAILABLE:
            return {"available": False, "error": "NVML not available"}
        
        try:
            device_count = pynvml.nvmlDeviceGetCount()
            gpus = []
            
            for i in range(device_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                name = pynvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode('utf-8')
                
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                
                gpus.append({
                    "index": i,
                    "name": name,
                    "memory_total_mb": memory.total / (1024 * 1024),
                    "memory_used_mb": memory.used / (1024 * 1024),
                    "memory_free_mb": memory.free / (1024 * 1024),
                    "memory_utilization_pct": (memory.used / memory.total) * 100,
                    "gpu_utilization_pct": utilization.gpu,
                })
            
            return {
                "available": True,
                "device_count": device_count,
                "gpus": gpus,
                "timestamp": datetime.utcnow().isoformat(),
            }
        except Exception as e:
            # Don't expose internal error details
            return {"available": False, "error": "Failed to get GPU information"}
    
    @staticmethod
    def get_current_memory_usage() -> Dict[str, float]:
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return {"allocated_mb": 0, "reserved_mb": 0, "max_allocated_mb": 0}
        
        return {
            "allocated_mb": torch.cuda.memory_allocated() / (1024 * 1024),
            "reserved_mb": torch.cuda.memory_reserved() / (1024 * 1024),
            "max_allocated_mb": torch.cuda.max_memory_allocated() / (1024 * 1024),
        }
    
    @staticmethod
    def cleanup_gpu_memory() -> Dict[str, Any]:
        result = {
            "success": True,
            "before": GPUService.get_current_memory_usage(),
            "actions": [],
        }
        
        try:
            if TORCH_AVAILABLE and torch.cuda.is_available():
                torch.cuda.empty_cache()
                result["actions"].append("torch.cuda.empty_cache()")
                
                torch.cuda.reset_peak_memory_stats()
                result["actions"].append("torch.cuda.reset_peak_memory_stats()")
                
                torch.cuda.synchronize()
                result["actions"].append("torch.cuda.synchronize()")
            
            gc.collect()
            result["actions"].append("gc.collect()")
            
            result["after"] = GPUService.get_current_memory_usage()
            result["memory_freed_mb"] = result["before"]["allocated_mb"] - result["after"]["allocated_mb"]
            
        except Exception as e:
            result["success"] = False
            result["error"] = "GPU cleanup failed"
        
        return result
    
    @staticmethod
    def cleanup_cache_directories(cache_dirs: Optional[list] = None) -> Dict[str, Any]:
        if cache_dirs is None:
            cache_dirs = [
                os.path.expanduser("~/.cache/huggingface/hub"),
                "/tmp/torch_cache",
                "/tmp/transformers_cache",
            ]
        
        result = {"cleaned": [], "errors": [], "total_freed_mb": 0}
        
        for cache_dir in cache_dirs:
            if os.path.exists(cache_dir):
                try:
                    size_before = GPUService._get_dir_size(cache_dir)
                    shutil.rmtree(cache_dir, ignore_errors=True)
                    os.makedirs(cache_dir, exist_ok=True)
                    result["cleaned"].append(cache_dir)
                    result["total_freed_mb"] += size_before / (1024 * 1024)
                except Exception as e:
                    result["errors"].append({"dir": "cache", "error": "cleanup failed"})
        
        return result
    
    @staticmethod
    def _get_dir_size(path: str) -> int:
        total = 0
        try:
            for dirpath, dirnames, filenames in os.walk(path):
                for f in filenames:
                    fp = os.path.join(dirpath, f)
                    if os.path.exists(fp):
                        total += os.path.getsize(fp)
        except:
            pass
        return total
    
    @staticmethod
    def full_cleanup() -> Dict[str, Any]:
        return {
            "gpu_cleanup": GPUService.cleanup_gpu_memory(),
            "cache_cleanup": GPUService.cleanup_cache_directories(),
            "timestamp": datetime.utcnow().isoformat(),
        }


gpu_service = GPUService()
