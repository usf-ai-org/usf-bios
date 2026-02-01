# Copyright (c) US Inc. All rights reserved.
"""Job manager for tracking training jobs"""

import asyncio
import os
import random
import signal
import subprocess
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from ..models.schemas import JobInfo, JobStatus, TrainingConfig


# Word lists for generating meaningful training names
_ADJECTIVES = [
    "swift", "bright", "cosmic", "stellar", "quantum", "neural", "rapid", "smart",
    "deep", "prime", "alpha", "omega", "ultra", "mega", "hyper", "super",
    "golden", "silver", "crystal", "atomic", "dynamic", "agile", "bold", "vivid"
]
_NOUNS = [
    "phoenix", "falcon", "eagle", "dragon", "titan", "atlas", "nova", "pulse",
    "spark", "flame", "wave", "storm", "bolt", "flash", "beam", "core",
    "nexus", "apex", "zenith", "vertex", "matrix", "cipher", "vector", "orbit"
]


class JobManager:
    """Manages training jobs"""
    
    def __init__(self):
        self._jobs: Dict[str, JobInfo] = {}
        self._logs: Dict[str, List[str]] = {}
        self._processes: Dict[str, asyncio.subprocess.Process] = {}
        self._lock = asyncio.Lock()
    
    def is_training_process_running(self) -> bool:
        """Check if any usf_bios training process is running on the system.
        
        This is a fallback check that works even if the in-memory state is lost.
        Checks ALL training types supported by USF BIOS.
        """
        # All training method patterns - from schemas.py TrainingMethod + RLHFType
        # Main methods: sft, pt, rlhf
        # RLHF types: dpo, orpo, simpo, kto, cpo, rm, ppo, grpo, gkd
        # Train types: lora, qlora, adalora, full
        patterns = [
            "usf_bios.*sft",
            "usf_bios.*pt",
            "usf_bios.*rlhf",
            "usf_bios.*dpo",
            "usf_bios.*orpo",
            "usf_bios.*simpo",
            "usf_bios.*kto",
            "usf_bios.*cpo",
            "usf_bios.*rm",
            "usf_bios.*ppo",
            "usf_bios.*grpo",
            "usf_bios.*gkd",
            "usf_bios.*lora",
            "usf_bios.*qlora",
            "usf_bios.*adalora",
            "usf_bios.*full",
            "usf_bios train",
            "python.*-m usf_bios",
        ]
        
        for pattern in patterns:
            try:
                result = subprocess.run(
                    ["pgrep", "-f", pattern],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0 and result.stdout.strip():
                    return True
            except Exception:
                continue
        return False
    
    def get_running_training_pid(self) -> Optional[int]:
        """Get the PID of running usf_bios training process if any."""
        # All training method patterns - from schemas.py TrainingMethod + RLHFType
        patterns = [
            "usf_bios.*sft",
            "usf_bios.*pt",
            "usf_bios.*rlhf",
            "usf_bios.*dpo",
            "usf_bios.*orpo",
            "usf_bios.*simpo",
            "usf_bios.*kto",
            "usf_bios.*cpo",
            "usf_bios.*rm",
            "usf_bios.*ppo",
            "usf_bios.*grpo",
            "usf_bios.*gkd",
            "usf_bios.*lora",
            "usf_bios.*qlora",
            "usf_bios.*adalora",
            "usf_bios.*full",
            "usf_bios train",
            "python.*-m usf_bios",
        ]
        
        for pattern in patterns:
            try:
                result = subprocess.run(
                    ["pgrep", "-f", pattern],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0 and result.stdout.strip():
                    pids = result.stdout.strip().split('\n')
                    if pids:
                        return int(pids[0])
            except Exception:
                continue
        return None
    
    def _generate_name(self) -> str:
        """Generate a unique, meaningful training name."""
        for _ in range(100):
            adj = random.choice(_ADJECTIVES)
            noun = random.choice(_NOUNS)
            num = random.randint(10, 99)
            name = f"{adj}-{noun}-{num}"
            # Check if name already exists
            if not any(j.name == name for j in self._jobs.values()):
                return name
        # Fallback with timestamp
        return f"training-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    
    async def create_job(self, config: TrainingConfig) -> JobInfo:
        """Create a new training job"""
        job_id = str(uuid.uuid4())[:8]
        
        # Generate name if not provided
        job_name = config.name.strip() if config.name else None
        if not job_name:
            job_name = self._generate_name()
        else:
            # Check if name is already used
            async with self._lock:
                if any(j.name == job_name for j in self._jobs.values()):
                    raise ValueError(f"Training name '{job_name}' is already in use")
        
        job = JobInfo(
            job_id=job_id,
            name=job_name,
            status=JobStatus.PENDING,
            config=config,
            created_at=datetime.now(),
            total_epochs=config.num_train_epochs,
        )
        
        async with self._lock:
            self._jobs[job_id] = job
            self._logs[job_id] = []
        
        return job
    
    async def get_job(self, job_id: str) -> Optional[JobInfo]:
        """Get job by ID"""
        return self._jobs.get(job_id)
    
    async def get_all_jobs(self) -> List[JobInfo]:
        """Get all jobs"""
        return list(self._jobs.values())
    
    async def update_job(self, job_id: str, **kwargs) -> Optional[JobInfo]:
        """Update job fields"""
        async with self._lock:
            if job_id not in self._jobs:
                return None
            
            job = self._jobs[job_id]
            for key, value in kwargs.items():
                if hasattr(job, key):
                    setattr(job, key, value)
            
            return job
    
    async def add_log(self, job_id: str, log_line: str) -> None:
        """Add a log line to job"""
        async with self._lock:
            if job_id in self._logs:
                self._logs[job_id].append(log_line)
                # Keep only last 1000 lines
                if len(self._logs[job_id]) > 1000:
                    self._logs[job_id] = self._logs[job_id][-1000:]
    
    async def get_logs(self, job_id: str, last_n: int = 100) -> List[str]:
        """Get job logs"""
        logs = self._logs.get(job_id, [])
        return logs[-last_n:]
    
    async def set_process(self, job_id: str, process: asyncio.subprocess.Process) -> None:
        """Store process reference for a job"""
        async with self._lock:
            self._processes[job_id] = process
    
    async def stop_job(self, job_id: str) -> bool:
        """Stop a running job"""
        async with self._lock:
            if job_id in self._processes:
                process = self._processes[job_id]
                try:
                    process.terminate()
                    await asyncio.wait_for(process.wait(), timeout=10)
                except asyncio.TimeoutError:
                    process.kill()
                except Exception:
                    pass
                
                del self._processes[job_id]
            
            if job_id in self._jobs:
                self._jobs[job_id].status = JobStatus.STOPPED
                return True
        
        return False
    
    async def fail_job(self, job_id: str, error_message: str) -> bool:
        """Mark a job as failed with an error message"""
        async with self._lock:
            if job_id in self._jobs:
                self._jobs[job_id].status = JobStatus.FAILED
                self._jobs[job_id].error = error_message
                self._jobs[job_id].completed_at = datetime.now()
                return True
        return False
    
    def stop_training_process_by_pid(self, pid: int) -> bool:
        """Stop a training process by its PID (fallback when job state is lost)."""
        try:
            os.kill(pid, signal.SIGTERM)
            return True
        except Exception:
            return False
    
    def stop_all_training_processes(self) -> bool:
        """Stop all usf_bios training processes (emergency stop)."""
        try:
            result = subprocess.run(
                ["pkill", "-f", "usf_bios.*sft"],
                capture_output=True,
                timeout=10
            )
            return result.returncode == 0
        except Exception:
            return False
    
    async def delete_job(self, job_id: str) -> bool:
        """Delete a job (only if not running)
        
        Also deletes associated encrypted log file.
        """
        from .encrypted_log_service import encrypted_log_service
        from .sanitized_log_service import sanitized_log_service
        
        async with self._lock:
            if job_id in self._jobs:
                if self._jobs[job_id].status == JobStatus.RUNNING:
                    return False
                del self._jobs[job_id]
                self._logs.pop(job_id, None)
                
                # Delete encrypted log file
                try:
                    encrypted_log_path = encrypted_log_service.get_encrypted_log_path(job_id)
                    if os.path.exists(encrypted_log_path):
                        os.remove(encrypted_log_path)
                except Exception:
                    pass  # Non-fatal - don't fail deletion for log cleanup
                
                # Delete terminal log file
                try:
                    sanitized_log_service.delete_terminal_log(job_id)
                except Exception:
                    pass  # Non-fatal
                
                return True
        return False
    
    async def clear_logs(self, job_id: str) -> None:
        """Clear in-memory logs for a job (used when restarting)"""
        async with self._lock:
            if job_id in self._logs:
                self._logs[job_id] = []


# Global instance
job_manager = JobManager()
