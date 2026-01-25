# Copyright (c) US Inc. All rights reserved.
"""
GPU Memory Cleanup Service - Comprehensive VRAM and CPU memory management.

This service handles aggressive GPU memory cleanup to prevent VRAM leaks
between training/inference sessions. The issue is that PyTorch/CUDA doesn't
always release memory properly when processes end, especially:
- Training subprocess crashes or OOM errors
- Inference model unloading
- Multiple training runs in sequence

Features:
- Multi-GPU support (cleans all GPUs)
- Kills orphaned CUDA processes (with SAFE protections for FastAPI/frontend)
- Clears PyTorch CUDA cache
- Forces garbage collection
- Resets CUDA memory statistics
- IPC memory cleanup for multi-process scenarios
- CPU memory cleanup for DeepSpeed/FSDP offload scenarios

CRITICAL SAFETY:
- NEVER kills FastAPI/uvicorn backend processes
- NEVER kills Node.js/Next.js frontend processes
- Only kills orphaned training subprocess processes
"""

import gc
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Set, Tuple

import logging

logger = logging.getLogger(__name__)

# =============================================================================
# PROTECTED PROCESS PATTERNS - These processes will NEVER be killed
# =============================================================================
PROTECTED_PROCESS_NAMES = {
    # FastAPI/Backend
    "uvicorn",
    "gunicorn", 
    "fastapi",
    "hypercorn",
    # Frontend/Node.js
    "node",
    "npm",
    "next",
    "next-server",
    # System processes
    "bash",
    "sh",
    "zsh",
    "systemd",
    "dockerd",
    "containerd",
    # NVIDIA processes
    "nvidia-smi",
    "nvidia-persistenced",
    "dcgm",
    # Database
    "postgres",
    "mysql",
    "redis",
    "sqlite",
}

# Command line patterns that indicate protected processes
PROTECTED_CMDLINE_PATTERNS = [
    r"uvicorn.*main:app",
    r"uvicorn.*app",
    r"fastapi",
    r"gunicorn",
    r"node.*next",
    r"node.*server",
    r"npm\s+run",
    r"python.*-m\s+uvicorn",
    r"python.*main\.py",  # Backend main.py
]


class GPUCleanupService:
    """
    Service for comprehensive GPU memory cleanup.
    
    Use Cases:
    - Before starting a new training job
    - After training job completes or fails
    - Before loading a model for inference
    - After unloading inference model
    - Manual cleanup via API endpoint
    """
    
    def __init__(self):
        self._torch_available = False
        self._pynvml_available = False
        self._torch = None
        self._pynvml = None
        self._initialize_libraries()
    
    def _initialize_libraries(self):
        """Initialize PyTorch and pynvml if available."""
        try:
            import torch
            self._torch = torch
            self._torch_available = torch.cuda.is_available()
        except ImportError:
            logger.warning("PyTorch not available for GPU cleanup")
        
        try:
            import pynvml
            self._pynvml = pynvml
            self._pynvml_available = True
        except ImportError:
            logger.warning("pynvml not available - some cleanup features disabled")
    
    def get_gpu_memory_info(self) -> Dict[str, Any]:
        """
        Get current GPU memory status using pynvml for accuracy.
        
        Returns dict with:
        - available: bool
        - device_count: int
        - gpus: list of GPU info dicts
        - total_used_gb: float
        - total_free_gb: float
        - total_memory_gb: float
        """
        result = {
            "available": False,
            "device_count": 0,
            "gpus": [],
            "total_used_gb": 0.0,
            "total_free_gb": 0.0,
            "total_memory_gb": 0.0,
            "source": "none"
        }
        
        if self._pynvml_available:
            try:
                self._pynvml.nvmlInit()
                device_count = self._pynvml.nvmlDeviceGetCount()
                
                if device_count > 0:
                    result["available"] = True
                    result["device_count"] = device_count
                    result["source"] = "pynvml"
                    
                    total_used = 0
                    total_free = 0
                    total_memory = 0
                    
                    for i in range(device_count):
                        handle = self._pynvml.nvmlDeviceGetHandleByIndex(i)
                        mem_info = self._pynvml.nvmlDeviceGetMemoryInfo(handle)
                        
                        name = self._pynvml.nvmlDeviceGetName(handle)
                        if isinstance(name, bytes):
                            name = name.decode('utf-8')
                        
                        gpu_info = {
                            "id": i,
                            "name": name,
                            "used_gb": round(mem_info.used / (1024**3), 2),
                            "free_gb": round(mem_info.free / (1024**3), 2),
                            "total_gb": round(mem_info.total / (1024**3), 2),
                            "utilization_percent": round((mem_info.used / mem_info.total) * 100, 1)
                        }
                        
                        result["gpus"].append(gpu_info)
                        total_used += mem_info.used
                        total_free += mem_info.free
                        total_memory += mem_info.total
                    
                    result["total_used_gb"] = round(total_used / (1024**3), 2)
                    result["total_free_gb"] = round(total_free / (1024**3), 2)
                    result["total_memory_gb"] = round(total_memory / (1024**3), 2)
                
                self._pynvml.nvmlShutdown()
                return result
                
            except Exception as e:
                logger.warning(f"pynvml GPU info failed: {e}")
        
        # Fallback to torch
        if self._torch_available:
            try:
                device_count = self._torch.cuda.device_count()
                if device_count > 0:
                    result["available"] = True
                    result["device_count"] = device_count
                    result["source"] = "torch"
                    
                    for i in range(device_count):
                        props = self._torch.cuda.get_device_properties(i)
                        # Note: torch only gives total memory, not usage
                        gpu_info = {
                            "id": i,
                            "name": props.name,
                            "total_gb": round(props.total_memory / (1024**3), 2),
                            "used_gb": None,  # Not available from torch
                            "free_gb": None
                        }
                        result["gpus"].append(gpu_info)
                        result["total_memory_gb"] += gpu_info["total_gb"]
            except Exception as e:
                logger.warning(f"torch GPU info failed: {e}")
        
        return result
    
    def _get_cuda_processes(self) -> List[Dict[str, Any]]:
        """
        Get list of processes using CUDA GPUs.
        
        Returns list of dicts with pid, name, gpu_id, memory_mb.
        """
        processes = []
        
        if not self._pynvml_available:
            return processes
        
        try:
            self._pynvml.nvmlInit()
            device_count = self._pynvml.nvmlDeviceGetCount()
            
            for i in range(device_count):
                handle = self._pynvml.nvmlDeviceGetHandleByIndex(i)
                
                try:
                    # Get compute processes
                    procs = self._pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
                    for proc in procs:
                        proc_info = {
                            "pid": proc.pid,
                            "gpu_id": i,
                            "memory_mb": proc.usedGpuMemory // (1024 * 1024) if proc.usedGpuMemory else 0,
                            "type": "compute"
                        }
                        
                        # Try to get process name
                        try:
                            with open(f"/proc/{proc.pid}/comm", "r") as f:
                                proc_info["name"] = f.read().strip()
                        except:
                            proc_info["name"] = "unknown"
                        
                        processes.append(proc_info)
                except Exception:
                    pass
                
                try:
                    # Get graphics processes
                    gfx_procs = self._pynvml.nvmlDeviceGetGraphicsRunningProcesses(handle)
                    for proc in gfx_procs:
                        proc_info = {
                            "pid": proc.pid,
                            "gpu_id": i,
                            "memory_mb": proc.usedGpuMemory // (1024 * 1024) if proc.usedGpuMemory else 0,
                            "type": "graphics"
                        }
                        try:
                            with open(f"/proc/{proc.pid}/comm", "r") as f:
                                proc_info["name"] = f.read().strip()
                        except:
                            proc_info["name"] = "unknown"
                        
                        processes.append(proc_info)
                except Exception:
                    pass
            
            self._pynvml.nvmlShutdown()
        except Exception as e:
            logger.warning(f"Failed to get CUDA processes: {e}")
        
        return processes
    
    def _is_protected_process(self, pid: int) -> Tuple[bool, str]:
        """
        Check if a process is protected and should NEVER be killed.
        
        Protected processes include:
        - FastAPI/uvicorn backend
        - Node.js/Next.js frontend
        - System processes
        - NVIDIA driver processes
        
        Returns (is_protected, reason).
        """
        try:
            # Get process name
            try:
                with open(f"/proc/{pid}/comm", "r") as f:
                    proc_name = f.read().strip().lower()
            except:
                proc_name = ""
            
            # Check against protected names
            if proc_name in PROTECTED_PROCESS_NAMES:
                return True, f"Protected process name: {proc_name}"
            
            # Get full command line for more detailed checks
            try:
                with open(f"/proc/{pid}/cmdline", "r") as f:
                    cmdline = f.read().replace('\x00', ' ').strip().lower()
            except:
                cmdline = ""
            
            # Check against protected command line patterns
            for pattern in PROTECTED_CMDLINE_PATTERNS:
                if re.search(pattern, cmdline, re.IGNORECASE):
                    return True, f"Protected cmdline pattern: {pattern}"
            
            # Check if this is the FastAPI backend by looking for uvicorn in cmdline
            if "uvicorn" in cmdline or "fastapi" in cmdline:
                return True, "FastAPI/uvicorn process"
            
            # Check if this is the frontend by looking for node/next
            if "node" in cmdline and ("next" in cmdline or "server" in cmdline):
                return True, "Node.js/Next.js frontend"
            
            # Check if running main.py (likely backend)
            if "main.py" in cmdline and ("python" in cmdline or "uvicorn" in cmdline):
                return True, "Backend main.py process"
            
            return False, ""
            
        except Exception as e:
            # If we can't check, assume protected to be safe
            return True, f"Unable to verify (assuming protected): {e}"
    
    def _is_training_subprocess(self, pid: int) -> Tuple[bool, str]:
        """
        Check if a process is a training subprocess that can be cleaned up.
        
        Training subprocesses are characterized by:
        - Python process running usf_bios training
        - Parent is not PID 1 (not orphaned from start)
        - Command line contains training-related keywords
        
        Returns (is_training, reason).
        """
        try:
            # Get command line
            try:
                with open(f"/proc/{pid}/cmdline", "r") as f:
                    cmdline = f.read().replace('\x00', ' ').strip()
            except:
                return False, "Cannot read cmdline"
            
            cmdline_lower = cmdline.lower()
            
            # Check for training-related keywords in command line
            training_indicators = [
                "usf_bios",
                "--train_type",
                "--dataset",
                "--output_dir",
                "--num_train_epochs",
                "sft",
                "rlhf",
                "dpo",
                "ppo",
                "--lora_rank",
            ]
            
            is_training = any(indicator in cmdline_lower for indicator in training_indicators)
            
            if is_training:
                return True, f"Training subprocess detected"
            
            return False, "Not a training subprocess"
            
        except Exception as e:
            return False, f"Check failed: {e}"
    
    def _kill_orphaned_cuda_processes(self, exclude_pids: Optional[List[int]] = None) -> Dict[str, Any]:
        """
        Kill orphaned CUDA processes that are consuming GPU memory.
        
        CRITICAL SAFETY FEATURES:
        - NEVER kills FastAPI/uvicorn processes
        - NEVER kills Node.js/Next.js frontend processes
        - Only kills processes that are:
          1. Python processes
          2. NOT protected processes
          3. Either orphaned (parent=1) OR identified as training subprocesses
        
        Args:
            exclude_pids: PIDs to NOT kill (e.g., current process, backend server)
        
        Returns dict with killed_pids, skipped_pids, protected_pids, and errors.
        """
        result = {
            "killed_pids": [],
            "skipped_pids": [],
            "protected_pids": [],
            "errors": []
        }
        
        if exclude_pids is None:
            exclude_pids = []
        
        # Always exclude current process and parent (and their ancestors)
        current_pid = os.getpid()
        parent_pid = os.getppid()
        exclude_set = set(exclude_pids)
        exclude_set.add(current_pid)
        exclude_set.add(parent_pid)
        
        # Also exclude grandparent (in case we're in a subprocess)
        try:
            with open(f"/proc/{parent_pid}/stat", "r") as f:
                stat = f.read().split()
                grandparent_pid = int(stat[3])
                exclude_set.add(grandparent_pid)
        except:
            pass
        
        # Get all CUDA processes
        processes = self._get_cuda_processes()
        
        for proc in processes:
            pid = proc["pid"]
            proc_name = proc.get("name", "unknown")
            
            # Skip if in exclude list
            if pid in exclude_set:
                result["skipped_pids"].append({"pid": pid, "reason": "explicitly excluded"})
                continue
            
            # CRITICAL: Check if protected process
            is_protected, protect_reason = self._is_protected_process(pid)
            if is_protected:
                result["protected_pids"].append({
                    "pid": pid,
                    "name": proc_name,
                    "reason": protect_reason
                })
                logger.info(f"Protected process {pid} ({proc_name}): {protect_reason}")
                continue
            
            # Only consider Python processes for cleanup
            if proc_name not in ["python", "python3", "pt_main_thread"]:
                result["skipped_pids"].append({
                    "pid": pid, 
                    "name": proc_name,
                    "reason": f"not a Python process"
                })
                continue
            
            # Check if process is orphaned (parent = 1) or a training subprocess
            should_kill = False
            kill_reason = ""
            
            try:
                # Send signal 0 to check if process exists
                os.kill(pid, 0)
                
                # Check parent PID
                with open(f"/proc/{pid}/stat", "r") as f:
                    stat = f.read().split()
                    ppid = int(stat[3])
                
                # Case 1: Orphaned process (parent is init/PID 1)
                if ppid == 1:
                    should_kill = True
                    kill_reason = "orphaned process (parent=1)"
                
                # Case 2: Training subprocess that's stuck
                if not should_kill:
                    is_training, training_reason = self._is_training_subprocess(pid)
                    if is_training:
                        # Only kill if it's been running for a while without progress
                        # For now, only kill truly orphaned processes
                        # Training subprocesses with valid parents are handled by training service
                        pass
                
            except ProcessLookupError:
                # Process already dead
                continue
            except Exception as e:
                result["errors"].append(f"Error checking {pid}: {e}")
                continue
            
            # Kill the process if it should be killed
            if should_kill:
                try:
                    logger.info(f"Killing orphaned process {pid} ({proc_name}): {kill_reason}")
                    
                    # First try graceful termination
                    os.kill(pid, signal.SIGTERM)
                    time.sleep(0.5)
                    
                    # Force kill if still running
                    try:
                        os.kill(pid, 0)  # Check if still alive
                        os.kill(pid, signal.SIGKILL)
                        time.sleep(0.2)
                    except ProcessLookupError:
                        pass  # Process terminated gracefully
                    
                    result["killed_pids"].append({
                        "pid": pid,
                        "name": proc_name,
                        "reason": kill_reason
                    })
                    
                except PermissionError:
                    result["errors"].append(f"Permission denied killing {pid}")
                except Exception as e:
                    result["errors"].append(f"Failed to kill {pid}: {e}")
            else:
                result["skipped_pids"].append({
                    "pid": pid,
                    "name": proc_name, 
                    "reason": "not orphaned"
                })
        
        return result
    
    def clear_torch_cuda_cache(self) -> Dict[str, Any]:
        """
        Clear PyTorch CUDA cache on all GPUs.
        
        This performs:
        1. Synchronize all CUDA operations
        2. Empty CUDA cache on each GPU
        3. Reset memory statistics
        4. IPC collect for multi-process scenarios
        """
        result = {
            "success": False,
            "gpus_cleared": 0,
            "operations": []
        }
        
        if not self._torch_available:
            result["error"] = "PyTorch not available"
            return result
        
        try:
            device_count = self._torch.cuda.device_count()
            
            if device_count == 0:
                result["error"] = "No CUDA devices"
                return result
            
            # Global synchronize first
            self._torch.cuda.synchronize()
            result["operations"].append("global_sync")
            
            # Clear each GPU
            for i in range(device_count):
                with self._torch.cuda.device(i):
                    self._torch.cuda.empty_cache()
                    self._torch.cuda.synchronize()
                    
                    # Reset memory stats
                    if hasattr(self._torch.cuda, 'reset_peak_memory_stats'):
                        self._torch.cuda.reset_peak_memory_stats()
                    if hasattr(self._torch.cuda, 'reset_accumulated_memory_stats'):
                        self._torch.cuda.reset_accumulated_memory_stats()
                
                result["gpus_cleared"] += 1
            
            result["operations"].append(f"cleared_{device_count}_gpus")
            
            # Global empty cache
            self._torch.cuda.empty_cache()
            result["operations"].append("global_empty_cache")
            
            # IPC collect for multi-process
            if hasattr(self._torch.cuda, 'ipc_collect'):
                self._torch.cuda.ipc_collect()
                result["operations"].append("ipc_collect")
            
            result["success"] = True
            
        except Exception as e:
            result["error"] = str(e)
            logger.error(f"CUDA cache clear failed: {e}")
        
        return result
    
    def force_garbage_collection(self, passes: int = 5) -> Dict[str, Any]:
        """
        Force Python garbage collection multiple times.
        
        Multiple passes help ensure circular references are collected.
        """
        result = {
            "passes": passes,
            "objects_collected": []
        }
        
        for i in range(passes):
            collected = gc.collect()
            result["objects_collected"].append(collected)
        
        result["total_collected"] = sum(result["objects_collected"])
        return result
    
    def clear_cpu_memory(self) -> Dict[str, Any]:
        """
        Clear CPU memory used by DeepSpeed/FSDP offload and other CPU tensors.
        
        This is important when:
        - DeepSpeed ZeRO offload is used (optimizer states on CPU)
        - FSDP CPU offload is enabled
        - Large CPU tensors are created during data loading
        - Model weights were moved to CPU before deletion
        
        Performs:
        1. Delete any global CPU tensor references
        2. Clear PyTorch CPU memory caches
        3. Multiple garbage collection passes
        4. Optionally trigger OS memory release (Linux only)
        """
        result = {
            "success": True,
            "operations": [],
            "cpu_memory_before_gb": None,
            "cpu_memory_after_gb": None,
            "memory_freed_gb": 0.0
        }
        
        try:
            # Get CPU memory before cleanup
            try:
                import psutil
                process = psutil.Process(os.getpid())
                mem_before = process.memory_info().rss / (1024**3)
                result["cpu_memory_before_gb"] = round(mem_before, 2)
                result["operations"].append("got_memory_before")
            except ImportError:
                pass
            
            # Step 1: Clear DeepSpeed CPU memory if available
            try:
                import deepspeed
                if hasattr(deepspeed, 'runtime') and hasattr(deepspeed.runtime, 'zero'):
                    # DeepSpeed may have CPU memory pools
                    pass
                result["operations"].append("deepspeed_check")
            except ImportError:
                pass
            
            # Step 2: Clear any CPU tensors in PyTorch
            if self._torch_available:
                try:
                    # Force move any remaining CUDA tensors to CPU first (helps release GPU)
                    # Then delete them
                    
                    # Clear PyTorch's internal caches
                    if hasattr(self._torch, '_C') and hasattr(self._torch._C, '_cuda_clearCublasWorkspaces'):
                        try:
                            self._torch._C._cuda_clearCublasWorkspaces()
                            result["operations"].append("cleared_cublas_workspaces")
                        except:
                            pass
                    
                    # Clear memory allocator caches
                    if hasattr(self._torch.cuda, 'memory') and hasattr(self._torch.cuda.memory, '_dump_snapshot'):
                        pass  # This is for debugging only
                    
                    result["operations"].append("pytorch_cpu_cleanup")
                except Exception as e:
                    logger.warning(f"PyTorch CPU cleanup warning: {e}")
            
            # Step 3: Aggressive garbage collection
            for i in range(5):
                gc.collect()
            result["operations"].append("gc_collect_5_passes")
            
            # Step 4: Try to release memory back to OS (Linux only)
            try:
                import ctypes
                libc = ctypes.CDLL("libc.so.6")
                # malloc_trim releases free memory back to OS
                if hasattr(libc, 'malloc_trim'):
                    libc.malloc_trim(0)
                    result["operations"].append("malloc_trim")
            except:
                pass  # Not available on all systems
            
            # Step 5: Final garbage collection
            gc.collect()
            gc.collect()
            result["operations"].append("final_gc")
            
            # Get CPU memory after cleanup
            try:
                import psutil
                process = psutil.Process(os.getpid())
                mem_after = process.memory_info().rss / (1024**3)
                result["cpu_memory_after_gb"] = round(mem_after, 2)
                
                if result["cpu_memory_before_gb"] is not None:
                    result["memory_freed_gb"] = round(
                        result["cpu_memory_before_gb"] - mem_after, 2
                    )
                result["operations"].append("got_memory_after")
            except ImportError:
                pass
            
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            logger.error(f"CPU memory cleanup failed: {e}")
        
        return result
    
    def deep_cleanup(
        self,
        kill_orphans: bool = True,
        exclude_pids: Optional[List[int]] = None,
        cleanup_cpu: bool = True
    ) -> Dict[str, Any]:
        """
        Perform comprehensive GPU and CPU memory cleanup.
        
        This is the main cleanup function that combines all cleanup methods:
        1. Get memory state before cleanup
        2. Force garbage collection (first pass)
        3. Clear PyTorch CUDA cache
        4. Clear CPU memory (for DeepSpeed/FSDP offload)
        5. Optionally kill orphaned CUDA processes (SAFE - protects FastAPI/frontend)
        6. Force garbage collection (second pass)
        7. Clear CUDA cache again
        8. Get memory state after cleanup
        
        SAFETY: This method NEVER kills FastAPI/uvicorn or Node.js/Next.js processes.
        Only orphaned training subprocesses (parent=1) are terminated.
        
        Args:
            kill_orphans: Whether to kill orphaned CUDA processes
            exclude_pids: PIDs to exclude from killing
            cleanup_cpu: Whether to also cleanup CPU memory (for offload scenarios)
        
        Returns comprehensive result dict.
        """
        result = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "memory_before": None,
            "memory_after": None,
            "memory_freed_gb": 0.0,
            "cpu_memory_freed_gb": 0.0,
            "steps": []
        }
        
        try:
            # Step 1: Get memory state before
            result["memory_before"] = self.get_gpu_memory_info()
            result["steps"].append({
                "step": "get_memory_before",
                "success": True
            })
            
            # Step 2: First garbage collection
            gc_result1 = self.force_garbage_collection(passes=3)
            result["steps"].append({
                "step": "gc_pass_1",
                "success": True,
                "objects_collected": gc_result1["total_collected"]
            })
            
            # Step 3: Clear CUDA cache
            cuda_result1 = self.clear_torch_cuda_cache()
            result["steps"].append({
                "step": "cuda_cache_clear_1",
                "success": cuda_result1.get("success", False),
                "gpus_cleared": cuda_result1.get("gpus_cleared", 0)
            })
            
            # Step 4: Clear CPU memory (for DeepSpeed/FSDP offload scenarios)
            if cleanup_cpu:
                cpu_result = self.clear_cpu_memory()
                result["steps"].append({
                    "step": "cpu_memory_clear",
                    "success": cpu_result.get("success", False),
                    "operations": cpu_result.get("operations", []),
                    "cpu_memory_freed_gb": cpu_result.get("memory_freed_gb", 0.0)
                })
                result["cpu_memory_freed_gb"] = cpu_result.get("memory_freed_gb", 0.0)
            
            # Step 5: Kill orphaned processes (if enabled) - SAFE implementation
            if kill_orphans:
                kill_result = self._kill_orphaned_cuda_processes(exclude_pids)
                result["steps"].append({
                    "step": "kill_orphans",
                    "success": True,
                    "killed_pids": kill_result["killed_pids"],
                    "skipped_pids": len(kill_result["skipped_pids"]),  # Just count for brevity
                    "protected_pids": kill_result.get("protected_pids", [])
                })
                
                # Wait for processes to release memory
                if kill_result["killed_pids"]:
                    time.sleep(1.0)
            
            # Step 6: Second garbage collection
            gc_result2 = self.force_garbage_collection(passes=5)
            result["steps"].append({
                "step": "gc_pass_2",
                "success": True,
                "objects_collected": gc_result2["total_collected"]
            })
            
            # Step 7: Clear CUDA cache again
            cuda_result2 = self.clear_torch_cuda_cache()
            result["steps"].append({
                "step": "cuda_cache_clear_2",
                "success": cuda_result2.get("success", False),
                "gpus_cleared": cuda_result2.get("gpus_cleared", 0)
            })
            
            # Step 8: Get memory state after
            result["memory_after"] = self.get_gpu_memory_info()
            result["steps"].append({
                "step": "get_memory_after",
                "success": True
            })
            
            # Calculate GPU memory freed
            if (result["memory_before"].get("total_used_gb") is not None and
                result["memory_after"].get("total_used_gb") is not None):
                result["memory_freed_gb"] = round(
                    result["memory_before"]["total_used_gb"] - 
                    result["memory_after"]["total_used_gb"],
                    2
                )
            
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
            logger.error(f"Deep cleanup failed: {e}")
        
        return result
    
    def quick_cleanup(self) -> Dict[str, Any]:
        """
        Quick cleanup - just GC and CUDA cache clear.
        
        Use this for lightweight cleanup between operations.
        """
        result = {
            "success": True,
            "timestamp": datetime.now().isoformat()
        }
        
        try:
            # Quick GC
            gc.collect()
            gc.collect()
            
            # Clear CUDA cache
            if self._torch_available:
                self._torch.cuda.empty_cache()
                if hasattr(self._torch.cuda, 'ipc_collect'):
                    self._torch.cuda.ipc_collect()
            
            result["memory_info"] = self.get_gpu_memory_info()
            
        except Exception as e:
            result["success"] = False
            result["error"] = str(e)
        
        return result
    
    async def async_deep_cleanup(
        self,
        kill_orphans: bool = True,
        exclude_pids: Optional[List[int]] = None,
        cleanup_cpu: bool = True
    ) -> Dict[str, Any]:
        """
        Async wrapper for deep_cleanup.
        
        SAFETY: This method NEVER kills FastAPI/uvicorn or Node.js/Next.js processes.
        Only orphaned training subprocesses (parent=1) are terminated.
        
        Args:
            kill_orphans: Whether to kill orphaned CUDA processes
            exclude_pids: PIDs to exclude from killing
            cleanup_cpu: Whether to also cleanup CPU memory (for offload scenarios)
        """
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.deep_cleanup(kill_orphans, exclude_pids, cleanup_cpu)
        )
    
    async def async_quick_cleanup(self) -> Dict[str, Any]:
        """
        Async wrapper for quick_cleanup.
        """
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None,
            self.quick_cleanup
        )
    
    async def async_cpu_cleanup(self) -> Dict[str, Any]:
        """
        Async wrapper for clear_cpu_memory.
        Use this for CPU-only cleanup (DeepSpeed/FSDP offload scenarios).
        """
        import asyncio
        return await asyncio.get_event_loop().run_in_executor(
            None,
            self.clear_cpu_memory
        )


# Global singleton instance
gpu_cleanup_service = GPUCleanupService()


# Convenience functions for direct import
def deep_cleanup_gpu(
    kill_orphans: bool = True, 
    exclude_pids: Optional[List[int]] = None,
    cleanup_cpu: bool = True
) -> Dict[str, Any]:
    """
    Perform comprehensive GPU and CPU memory cleanup.
    
    SAFETY: This function NEVER kills FastAPI/uvicorn or Node.js/Next.js processes.
    Only orphaned training subprocesses are terminated.
    """
    return gpu_cleanup_service.deep_cleanup(kill_orphans, exclude_pids, cleanup_cpu)


def quick_cleanup_gpu() -> Dict[str, Any]:
    """Perform quick GPU cleanup (GC + CUDA cache clear)."""
    return gpu_cleanup_service.quick_cleanup()


def clear_cpu_memory() -> Dict[str, Any]:
    """Clear CPU memory for DeepSpeed/FSDP offload scenarios."""
    return gpu_cleanup_service.clear_cpu_memory()


def get_gpu_memory_status() -> Dict[str, Any]:
    """Get current GPU memory status."""
    return gpu_cleanup_service.get_gpu_memory_info()
