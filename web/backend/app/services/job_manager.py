# Copyright (c) US Inc. All rights reserved.
"""Job manager for tracking training jobs"""

import asyncio
import uuid
from datetime import datetime
from typing import Dict, List, Optional

from ..models.schemas import JobInfo, JobStatus, TrainingConfig


class JobManager:
    """Manages training jobs"""
    
    def __init__(self):
        self._jobs: Dict[str, JobInfo] = {}
        self._logs: Dict[str, List[str]] = {}
        self._processes: Dict[str, asyncio.subprocess.Process] = {}
        self._lock = asyncio.Lock()
    
    async def create_job(self, config: TrainingConfig) -> JobInfo:
        """Create a new training job"""
        job_id = str(uuid.uuid4())[:8]
        
        job = JobInfo(
            job_id=job_id,
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
