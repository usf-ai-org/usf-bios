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
        """Aggressive GPU memory cleanup using both PyTorch and pynvml."""
        result = {
            "success": True,
            "before": GPUService.get_current_memory_usage(),
            "before_nvml": GPUService.get_nvml_memory_usage(),
            "actions": [],
        }
        
        try:
            # Multiple garbage collection passes first
            for _ in range(3):
                gc.collect()
            result["actions"].append("gc.collect() x3")
            
            if TORCH_AVAILABLE and torch.cuda.is_available():
                # Synchronize before cleanup
                torch.cuda.synchronize()
                result["actions"].append("torch.cuda.synchronize()")
                
                # Empty cache on all devices
                device_count = torch.cuda.device_count()
                for i in range(device_count):
                    with torch.cuda.device(i):
                        torch.cuda.empty_cache()
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        torch.cuda.reset_accumulated_memory_stats()
                result["actions"].append(f"torch.cuda.empty_cache() on {device_count} GPUs")
                
                # IPC collect for multi-process scenarios
                if hasattr(torch.cuda, 'ipc_collect'):
                    torch.cuda.ipc_collect()
                    result["actions"].append("torch.cuda.ipc_collect()")
            
            # Final garbage collection
            for _ in range(2):
                gc.collect()
            result["actions"].append("gc.collect() x2 (final)")
            
            result["after"] = GPUService.get_current_memory_usage()
            result["after_nvml"] = GPUService.get_nvml_memory_usage()
            result["memory_freed_mb"] = result["before"]["allocated_mb"] - result["after"]["allocated_mb"]
            
            # Also report actual GPU memory freed (from driver level)
            if result["before_nvml"]["used_mb"] > 0 and result["after_nvml"]["used_mb"] >= 0:
                result["actual_memory_freed_mb"] = result["before_nvml"]["used_mb"] - result["after_nvml"]["used_mb"]
            
        except Exception as e:
            result["success"] = False
            result["error"] = "GPU cleanup failed"
        
        return result
    
    @staticmethod
    def get_nvml_memory_usage() -> Dict[str, float]:
        """Get actual GPU memory usage from driver (not PyTorch allocation)."""
        try:
            if NVML_AVAILABLE:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                return {
                    "used_mb": mem_info.used / (1024 * 1024),
                    "total_mb": mem_info.total / (1024 * 1024),
                    "free_mb": mem_info.free / (1024 * 1024),
                }
        except Exception:
            pass
        return {"used_mb": 0, "total_mb": 0, "free_mb": 0}
    
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
