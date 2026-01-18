# Copyright (c) US Inc. All rights reserved.
"""Job manager for tracking training jobs"""

import asyncio
import uuid
import random
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
    
    async def delete_job(self, job_id: str) -> bool:
        """Delete a job (only if not running)"""
        async with self._lock:
            if job_id in self._jobs:
                if self._jobs[job_id].status == JobStatus.RUNNING:
                    return False
                del self._jobs[job_id]
                self._logs.pop(job_id, None)
                return True
        return False


# Global instance
job_manager = JobManager()
