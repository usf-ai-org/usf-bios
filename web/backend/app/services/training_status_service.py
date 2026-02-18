# Copyright (c) US Inc. All rights reserved.
"""
Training Status Service - Centralized training state management

This service provides a single source of truth for training status across the system.
It tracks:
- Whether training is active
- Current job details (id, name, status, progress)
- Whether new jobs can be created
- Whether inference can be loaded
- Real-time progress updates
"""

import asyncio
import os
import subprocess
from datetime import datetime
from enum import Enum
from typing import Any, Dict, Optional
from dataclasses import dataclass, field

from .job_manager import job_manager
from ..models.schemas import JobStatus


class TrainingPhase(str, Enum):
    """Training lifecycle phases"""
    IDLE = "idle"                    # No training running
    INITIALIZING = "initializing"   # Training starting up
    RUNNING = "running"             # Training in progress
    COMPLETING = "completing"       # Training finishing up
    COMPLETED = "completed"         # Training finished successfully
    FAILED = "failed"               # Training failed
    STOPPED = "stopped"             # Training manually stopped


@dataclass
class TrainingProgress:
    """Training progress details"""
    current_step: int = 0
    total_steps: int = 0
    current_epoch: float = 0
    total_epochs: int = 0
    current_loss: Optional[float] = None
    learning_rate: Optional[float] = None
    samples_per_second: Optional[float] = None
    eta_seconds: Optional[int] = None
    progress_percent: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_step": self.current_step,
            "total_steps": self.total_steps,
            "current_epoch": self.current_epoch,
            "total_epochs": self.total_epochs,
            "current_loss": self.current_loss,
            "learning_rate": self.learning_rate,
            "samples_per_second": self.samples_per_second,
            "eta_seconds": self.eta_seconds,
            "progress_percent": self.progress_percent,
        }


@dataclass
class TrainingStatus:
    """Complete training status snapshot"""
    # Core status
    is_training_active: bool = False
    phase: TrainingPhase = TrainingPhase.IDLE
    
    # Job details
    job_id: Optional[str] = None
    job_name: Optional[str] = None
    model_name: Optional[str] = None
    started_at: Optional[datetime] = None
    
    # Progress
    progress: TrainingProgress = field(default_factory=TrainingProgress)
    
    # Error info (if failed)
    error_message: Optional[str] = None
    
    # System restrictions
    can_create_job: bool = True
    can_load_inference: bool = True
    can_start_training: bool = True
    
    # Process-level detection (fallback)
    process_running: bool = False
    process_pid: Optional[int] = None
    
    # Timestamps
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_training_active": self.is_training_active,
            "phase": self.phase.value,
            "job_id": self.job_id,
            "job_name": self.job_name,
            "model_name": self.model_name,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "progress": self.progress.to_dict(),
            "error_message": self.error_message,
            "can_create_job": self.can_create_job,
            "can_load_inference": self.can_load_inference,
            "can_start_training": self.can_start_training,
            "process_running": self.process_running,
            "process_pid": self.process_pid,
            "last_updated": self.last_updated.isoformat(),
            # Computed fields for UI convenience
            "status_message": self._get_status_message(),
            "status_color": self._get_status_color(),
        }
    
    def _get_status_message(self) -> str:
        """Human-readable status message for UI"""
        if self.phase == TrainingPhase.IDLE:
            return "Ready to train"
        elif self.phase == TrainingPhase.INITIALIZING:
            return f"Initializing training: {self.job_name or 'Unknown'}"
        elif self.phase == TrainingPhase.RUNNING:
            if self.progress.total_steps > 0:
                return f"Training in progress: {self.progress.progress_percent:.1f}% ({self.progress.current_step}/{self.progress.total_steps})"
            return f"Training in progress: {self.job_name or 'Unknown'}"
        elif self.phase == TrainingPhase.COMPLETING:
            return "Completing training..."
        elif self.phase == TrainingPhase.COMPLETED:
            return f"Training completed: {self.job_name or 'Unknown'}"
        elif self.phase == TrainingPhase.FAILED:
            return f"Training failed: {self.error_message or 'Unknown error'}"
        elif self.phase == TrainingPhase.STOPPED:
            return f"Training stopped: {self.job_name or 'Unknown'}"
        return "Unknown status"
    
    def _get_status_color(self) -> str:
        """Status color for UI (tailwind classes)"""
        color_map = {
            TrainingPhase.IDLE: "slate",
            TrainingPhase.INITIALIZING: "blue",
            TrainingPhase.RUNNING: "blue",
            TrainingPhase.COMPLETING: "blue",
            TrainingPhase.COMPLETED: "green",
            TrainingPhase.FAILED: "red",
            TrainingPhase.STOPPED: "amber",
        }
        return color_map.get(self.phase, "slate")


class TrainingStatusService:
    """
    Centralized service for managing training status.
    
    This service provides:
    1. Single source of truth for training state
    2. Real-time status updates
    3. System restrictions (block inference during training)
    4. Process-level fallback detection
    5. Staleness detection for stuck training jobs
    """
    
    # Staleness thresholds (in seconds)
    INITIALIZING_STALE_THRESHOLD = 300  # 5 minutes in initializing with no logs = stale
    NO_PROGRESS_STALE_THRESHOLD = 600   # 10 minutes with no progress updates = stale
    
    def __init__(self):
        self._status = TrainingStatus()
        self._lock = asyncio.Lock()
        self._last_progress_update: Optional[datetime] = None
    
    async def get_status(self) -> TrainingStatus:
        """
        Get current training status.
        
        This method checks multiple sources to ensure accurate status:
        1. In-memory job manager state
        2. OS process detection (fallback)
        """
        async with self._lock:
            await self._refresh_status()
            return self._status
    
    async def get_status_dict(self) -> Dict[str, Any]:
        """Get status as dictionary for API response"""
        status = await self.get_status()
        return status.to_dict()
    
    async def _refresh_status(self) -> None:
        """Refresh status from all sources"""
        from .sanitized_log_service import sanitized_log_service
        
        # Check job manager for active jobs
        jobs = await job_manager.get_all_jobs()
        active_job = None
        
        for job in jobs:
            if job.status in [JobStatus.RUNNING, JobStatus.INITIALIZING]:
                active_job = job
                break
        
        # Check OS process as fallback
        process_running = job_manager.is_training_process_running()
        process_pid = job_manager.get_running_training_pid() if process_running else None
        
        # STALENESS DETECTION: Check if job appears stuck
        is_stale = False
        stale_reason = None
        
        if active_job and not process_running:
            # Job says it's running but no process found via pgrep.
            # This can happen in two cases:
            # 1. Process finished normally — completion handler is about to update status
            # 2. Process actually crashed
            
            # Check if the stored subprocess has a return code (finished naturally)
            stored_process = job_manager._processes.get(active_job.job_id)
            process_finished_cleanly = (
                stored_process is not None 
                and stored_process.returncode is not None
                and stored_process.returncode == 0
            )
            
            if process_finished_cleanly:
                # Process exited with code 0 — completion handler will update status shortly
                # Do NOT mark as stale, this is normal end-of-training
                pass
            else:
                # No stored process or non-zero exit — check grace period
                time_since_start = 0
                if active_job.started_at:
                    time_since_start = (datetime.utcnow() - active_job.started_at).total_seconds()
                
                # Give 120 seconds grace from start (covers model loading + short training)
                # and additional 30 seconds after process disappears for cleanup
                if time_since_start > 120:
                    is_stale = True
                    stale_reason = "Training process not found - job may have crashed"
        elif active_job and active_job.started_at:
            time_since_start = (datetime.utcnow() - active_job.started_at).total_seconds()
            
            # Check if stuck in INITIALIZING too long
            if active_job.status == JobStatus.INITIALIZING:
                # Check if there are any terminal logs
                try:
                    logs = sanitized_log_service.get_terminal_logs(active_job.job_id, lines=10)
                    has_logs = len(logs) > 0
                except Exception:
                    has_logs = False
                
                if time_since_start > self.INITIALIZING_STALE_THRESHOLD and not has_logs:
                    is_stale = True
                    stale_reason = f"Stuck in initializing for {int(time_since_start/60)} minutes with no output"
            
            # Check if running but no progress for too long
            elif active_job.status == JobStatus.RUNNING:
                if self._last_progress_update:
                    time_since_progress = (datetime.utcnow() - self._last_progress_update).total_seconds()
                    if time_since_progress > self.NO_PROGRESS_STALE_THRESHOLD:
                        is_stale = True
                        stale_reason = f"No progress updates for {int(time_since_progress/60)} minutes"
        
        # If stale, mark job as failed
        if is_stale and active_job:
            await job_manager.update_job(
                active_job.job_id,
                status=JobStatus.FAILED,
                error=f"Training appears stuck: {stale_reason}",
                completed_at=datetime.utcnow()
            )
            sanitized_log_service.create_terminal_log(
                active_job.job_id,
                f"ERROR: Training detected as stale - {stale_reason}",
                "ERROR"
            )
            # Clear the active job reference
            active_job = None
        
        # Determine training state
        if active_job:
            # We have job info - use it
            self._status.is_training_active = True
            self._status.job_id = active_job.job_id
            self._status.job_name = active_job.name
            self._status.model_name = active_job.config.model_path.split('/')[-1] if active_job.config else None
            self._status.started_at = active_job.started_at
            
            # Determine phase
            if active_job.status == JobStatus.INITIALIZING:
                self._status.phase = TrainingPhase.INITIALIZING
            else:
                self._status.phase = TrainingPhase.RUNNING
            
            # Update progress
            self._status.progress = TrainingProgress(
                current_step=active_job.current_step or 0,
                total_steps=active_job.total_steps or 0,
                current_epoch=active_job.current_epoch or 0,
                total_epochs=active_job.total_epochs or 0,
                current_loss=active_job.current_loss,
                learning_rate=active_job.learning_rate,
                samples_per_second=getattr(active_job, 'samples_per_second', None),
                eta_seconds=getattr(active_job, 'eta_seconds', None),
                progress_percent=(active_job.current_step / active_job.total_steps * 100) if active_job.total_steps else 0,
            )
            
            self._status.error_message = None
            
        elif process_running:
            # Process running but no job info (state lost)
            self._status.is_training_active = True
            self._status.phase = TrainingPhase.RUNNING
            self._status.job_id = f"pid-{process_pid}" if process_pid else "unknown"
            self._status.job_name = "Training in Progress"
            self._status.model_name = None
            self._status.started_at = None
            self._status.progress = TrainingProgress()
            self._status.error_message = None
            
        else:
            # No training active
            self._status.is_training_active = False
            self._status.phase = TrainingPhase.IDLE
            self._status.job_id = None
            self._status.job_name = None
            self._status.model_name = None
            self._status.started_at = None
            self._status.progress = TrainingProgress()
            self._status.error_message = None
        
        # Update process info
        self._status.process_running = process_running
        self._status.process_pid = process_pid
        
        # Update restrictions based on training state
        self._status.can_create_job = not self._status.is_training_active
        self._status.can_load_inference = not self._status.is_training_active
        self._status.can_start_training = not self._status.is_training_active
        
        # Update timestamp
        self._status.last_updated = datetime.utcnow()
    
    async def update_progress(
        self,
        job_id: str,
        current_step: int = None,
        total_steps: int = None,
        current_loss: float = None,
        learning_rate: float = None,
        current_epoch: float = None,
        samples_per_second: float = None,
        eta_seconds: int = None,
    ) -> None:
        """Update training progress (called from training service)"""
        async with self._lock:
            if self._status.job_id == job_id:
                if current_step is not None:
                    self._status.progress.current_step = current_step
                if total_steps is not None:
                    self._status.progress.total_steps = total_steps
                if current_loss is not None:
                    self._status.progress.current_loss = current_loss
                if learning_rate is not None:
                    self._status.progress.learning_rate = learning_rate
                if current_epoch is not None:
                    self._status.progress.current_epoch = current_epoch
                if samples_per_second is not None:
                    self._status.progress.samples_per_second = samples_per_second
                if eta_seconds is not None:
                    self._status.progress.eta_seconds = eta_seconds
                
                # Update progress percent
                if self._status.progress.total_steps > 0:
                    self._status.progress.progress_percent = (
                        self._status.progress.current_step / self._status.progress.total_steps * 100
                    )
                
                self._status.last_updated = datetime.utcnow()
                # Track last progress update for staleness detection
                self._last_progress_update = datetime.utcnow()
    
    async def set_training_started(
        self,
        job_id: str,
        job_name: str,
        model_name: str = None,
        total_steps: int = 0,
        total_epochs: int = 0,
    ) -> None:
        """Mark training as started"""
        async with self._lock:
            self._status.is_training_active = True
            self._status.phase = TrainingPhase.INITIALIZING
            self._status.job_id = job_id
            self._status.job_name = job_name
            self._status.model_name = model_name
            self._status.started_at = datetime.utcnow()
            self._status.progress = TrainingProgress(
                total_steps=total_steps,
                total_epochs=total_epochs,
            )
            self._status.error_message = None
            self._status.can_create_job = False
            self._status.can_load_inference = False
            self._status.can_start_training = False
            self._status.last_updated = datetime.utcnow()
            # Reset staleness tracker so previous job's timestamp
            # doesn't cause the new job to be immediately marked stale
            self._last_progress_update = datetime.utcnow()
    
    async def set_training_running(self, job_id: str) -> None:
        """Mark training as actively running"""
        async with self._lock:
            if self._status.job_id == job_id:
                self._status.phase = TrainingPhase.RUNNING
                self._status.last_updated = datetime.utcnow()
    
    async def set_training_completed(self, job_id: str) -> None:
        """Mark training as completed"""
        async with self._lock:
            if self._status.job_id == job_id or self._status.is_training_active:
                self._status.is_training_active = False
                self._status.phase = TrainingPhase.COMPLETED
                self._status.can_create_job = True
                self._status.can_load_inference = True
                self._status.can_start_training = True
                self._status.last_updated = datetime.utcnow()
                self._last_progress_update = None
    
    async def set_training_failed(self, job_id: str, error_message: str = None) -> None:
        """Mark training as failed"""
        async with self._lock:
            if self._status.job_id == job_id or self._status.is_training_active:
                self._status.is_training_active = False
                self._status.phase = TrainingPhase.FAILED
                self._status.error_message = error_message
                self._status.can_create_job = True
                self._status.can_load_inference = True
                self._status.can_start_training = True
                self._status.last_updated = datetime.utcnow()
                self._last_progress_update = None
    
    async def set_training_stopped(self, job_id: str) -> None:
        """Mark training as manually stopped"""
        async with self._lock:
            if self._status.job_id == job_id or self._status.is_training_active:
                self._status.is_training_active = False
                self._status.phase = TrainingPhase.STOPPED
                self._status.can_create_job = True
                self._status.can_load_inference = True
                self._status.can_start_training = True
                self._status.last_updated = datetime.utcnow()
                self._last_progress_update = None
    
    def is_training_active_sync(self) -> bool:
        """
        Synchronous check if training is active.
        
        This is used by middleware and other sync code that needs
        to check training status without async.
        """
        # Quick check of in-memory state
        if self._status.is_training_active:
            return True
        
        # Fallback to OS process check
        return job_manager.is_training_process_running()
    
    async def can_create_job(self) -> tuple[bool, str]:
        """
        Check if a new job can be created.
        
        Returns: (can_create, reason)
        """
        status = await self.get_status()
        if status.is_training_active:
            return False, f"Training is in progress: {status.job_name or 'Unknown'}. Please wait for it to complete or stop it first."
        return True, "Ready to create job"
    
    async def can_load_inference(self) -> tuple[bool, str]:
        """
        Check if inference model can be loaded.
        
        Returns: (can_load, reason)
        """
        status = await self.get_status()
        if status.is_training_active:
            return False, f"Cannot load inference model while training is in progress: {status.job_name or 'Unknown'}. Please wait for training to complete or stop it first."
        return True, "Ready to load inference model"
    
    async def can_start_training(self) -> tuple[bool, str]:
        """
        Check if training can be started.
        
        Returns: (can_start, reason)
        """
        status = await self.get_status()
        if status.is_training_active:
            return False, f"Another training is already in progress: {status.job_name or 'Unknown'}. Please wait for it to complete or stop it first."
        return True, "Ready to start training"


    async def force_reset_stuck_training(self, job_id: str = None) -> Dict[str, Any]:
        """
        Force reset a stuck training job.
        
        This is used when a training job appears stuck and needs manual intervention.
        It will:
        1. Kill all training processes (ALL types: SFT, DPO, PPO, GRPO, etc.)
        2. Clear GPU VRAM and CPU memory
        3. Mark the job as failed
        4. Clear training state
        5. Allow new training to start
        
        Returns status of the reset operation.
        """
        from .sanitized_log_service import sanitized_log_service
        from .gpu_cleanup_service import gpu_cleanup_service
        
        errors = []
        target_job_id = job_id or self._status.job_id
        
        # Allow reset even if no active training detected (process might be orphaned)
        if not target_job_id:
            target_job_id = "unknown"
        
        # Step 1: Kill ALL training processes (force kill with SIGKILL)
        try:
            killed = job_manager.stop_all_training_processes()
            if killed:
                # Wait for processes to terminate and release resources
                await asyncio.sleep(2.0)
        except Exception as e:
            errors.append(f"Process kill warning: {str(e)}")
        
        # Step 2: Force GPU VRAM cleanup (critical for allowing new training)
        gpu_cleanup_result = None
        try:
            gpu_cleanup_result = gpu_cleanup_service.deep_cleanup(
                kill_orphans=True,  # Kill any orphaned CUDA processes
                cleanup_cpu=True    # Also cleanup CPU memory (DeepSpeed offload)
            )
            if not gpu_cleanup_result.get("success"):
                errors.append(f"GPU cleanup warning: {gpu_cleanup_result.get('error', 'unknown')}")
        except Exception as e:
            errors.append(f"GPU cleanup exception: {str(e)}")
        
        # Step 3: Mark job as failed in job manager
        async with self._lock:
            if target_job_id and target_job_id != "unknown" and not target_job_id.startswith("pid-"):
                try:
                    await job_manager.update_job(
                        target_job_id,
                        status=JobStatus.FAILED,
                        error="Training forcefully reset by user - job appeared stuck",
                        completed_at=datetime.utcnow()
                    )
                    sanitized_log_service.create_terminal_log(
                        target_job_id,
                        "Training forcefully reset by user - VRAM cleared",
                        "WARN"
                    )
                except Exception as e:
                    errors.append(f"Job update warning: {str(e)}")
            
            # Step 4: Clear training state completely
            self._status.is_training_active = False
            self._status.phase = TrainingPhase.FAILED
            self._status.error_message = "Training forcefully reset"
            self._status.job_id = None
            self._status.job_name = None
            self._status.model_name = None
            self._status.started_at = None
            self._status.progress = TrainingProgress()
            self._status.process_running = False
            self._status.process_pid = None
            self._status.can_create_job = True
            self._status.can_load_inference = True
            self._status.can_start_training = True
            self._status.last_updated = datetime.utcnow()
            self._last_progress_update = None
        
        # Build result with detailed info
        result = {
            "success": True,
            "message": f"Training reset successfully. Job {target_job_id} marked as failed. VRAM cleared.",
            "job_id": target_job_id,
            "gpu_cleanup": {
                "memory_freed_gb": gpu_cleanup_result.get("memory_freed_gb", 0) if gpu_cleanup_result else 0,
                "cpu_memory_freed_gb": gpu_cleanup_result.get("cpu_memory_freed_gb", 0) if gpu_cleanup_result else 0,
            } if gpu_cleanup_result else None
        }
        
        if errors:
            result["warnings"] = errors
        
        return result


# Global instance
training_status_service = TrainingStatusService()
