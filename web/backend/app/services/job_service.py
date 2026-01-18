import os
import gc
import json
import hashlib
import shutil
import random
from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session

from app.models.db_models import (
    TrainingJob, TrainingMetric, Checkpoint, TrainingLog,
    SystemState, JobStatus, Dataset
)
from app.services.gpu_service import gpu_service
from app.services.encrypted_log_service import encrypted_log_service
from app.services.sanitized_log_service import sanitized_log_service, CrashReason


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


class JobServiceError(Exception):
    def __init__(self, message: str, error_code: str, user_message: str):
        super().__init__(message)
        self.error_code = error_code
        self.user_message = user_message


class JobService:
    
    OUTPUT_BASE_PATH = os.getenv("OUTPUT_PATH", "/app/data/output")
    CHECKPOINT_BASE_PATH = os.getenv("CHECKPOINT_PATH", "/app/data/checkpoints")
    LOG_BASE_PATH = os.getenv("LOG_PATH", "/app/data/logs")
    
    VALID_ACTIONS = {
        "start": ["pending"],
        "stop": ["running"],
        "resume": ["failed", "cancelled"],
        "restart_fresh": ["failed", "cancelled", "completed"],
        "delete": ["pending", "failed", "cancelled", "completed"],
        "clone": ["pending", "failed", "cancelled", "completed"],
        "update_config": ["pending"],
    }
    
    def __init__(self, db: Session):
        self.db = db
        os.makedirs(self.OUTPUT_BASE_PATH, exist_ok=True)
        os.makedirs(self.CHECKPOINT_BASE_PATH, exist_ok=True)
        os.makedirs(self.LOG_BASE_PATH, exist_ok=True)
    
    def generate_training_name(self, model_path: Optional[str] = None) -> str:
        """Generate a unique, meaningful training name.
        
        Format: {adjective}-{noun}-{number} e.g. "swift-phoenix-42"
        Ensures uniqueness by checking existing names.
        """
        max_attempts = 100
        for _ in range(max_attempts):
            adj = random.choice(_ADJECTIVES)
            noun = random.choice(_NOUNS)
            num = random.randint(10, 99)
            name = f"{adj}-{noun}-{num}"
            
            # Check if name already exists
            existing = self.db.query(TrainingJob).filter(TrainingJob.name == name).first()
            if not existing:
                return name
        
        # Fallback with timestamp if all attempts fail
        ts = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        return f"training-{ts}"
    
    def is_name_available(self, name: str, exclude_job_id: Optional[str] = None) -> bool:
        """Check if a training name is available (not used by another job)."""
        query = self.db.query(TrainingJob).filter(TrainingJob.name == name)
        if exclude_job_id:
            query = query.filter(TrainingJob.id != exclude_job_id)
        return query.first() is None
    
    def update_job_name(self, job_id: str, new_name: str) -> Dict[str, Any]:
        """Update the name of a training job.
        
        Args:
            job_id: The job ID
            new_name: New name for the job
            
        Returns:
            Updated job info
        """
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")
        
        # Validate new name
        new_name = new_name.strip()
        if not new_name:
            raise ValueError("Job name cannot be empty")
        if len(new_name) > 255:
            raise ValueError("Job name cannot exceed 255 characters")
        
        # Check if name is already used by another job
        if not self.is_name_available(new_name, exclude_job_id=job_id):
            raise ValueError(f"Name '{new_name}' is already used by another training")
        
        old_name = job.name
        job.name = new_name
        self.db.commit()
        
        return {
            "job_id": job_id,
            "old_name": old_name,
            "new_name": new_name,
            "success": True
        }
    
    def validate_action(self, job_id: str, action: str) -> Dict[str, Any]:
        if action not in self.VALID_ACTIONS:
            return {
                "allowed": False,
                "error_code": "INVALID_ACTION",
                "user_message": f"Action '{action}' is not supported."
            }
        
        job = self.get_job(job_id)
        if not job:
            return {
                "allowed": False,
                "error_code": "JOB_NOT_FOUND",
                "user_message": "Training job not found."
            }
        
        allowed_statuses = self.VALID_ACTIONS[action]
        if job.status not in allowed_statuses:
            status_names = {
                "pending": "Pending",
                "running": "Running",
                "completed": "Completed",
                "failed": "Failed",
                "cancelled": "Cancelled"
            }
            return {
                "allowed": False,
                "error_code": "INVALID_STATUS",
                "user_message": f"Cannot {action} a job that is {status_names.get(job.status, job.status)}."
            }
        
        if action in ["start", "resume", "restart_fresh"] and self.is_training_active():
            return {
                "allowed": False,
                "error_code": "TRAINING_ACTIVE",
                "user_message": "Another training is in progress. Please wait for it to complete or stop it first."
            }
        
        if action == "resume":
            checkpoints = self.db.query(Checkpoint).filter(Checkpoint.job_id == job_id).all()
            has_valid_checkpoint = any(os.path.exists(cp.path) for cp in checkpoints)
            if not has_valid_checkpoint:
                return {
                    "allowed": False,
                    "error_code": "NO_CHECKPOINT",
                    "user_message": "No checkpoint available to resume from. Use 'Start Fresh' to restart training."
                }
        
        return {
            "allowed": True,
            "job_id": job_id,
            "action": action,
            "current_status": job.status
        }
    
    def execute_action(self, job_id: str, action: str, **kwargs) -> Dict[str, Any]:
        validation = self.validate_action(job_id, action)
        if not validation["allowed"]:
            raise JobServiceError(
                message=validation["user_message"],
                error_code=validation["error_code"],
                user_message=validation["user_message"]
            )
        
        action_map = {
            "start": lambda: self.start_job(job_id),
            "stop": lambda: self.cancel_job(job_id),
            "resume": lambda: self.resume_job_from_latest(job_id) if not kwargs.get("checkpoint_id") else self.resume_job(job_id, kwargs["checkpoint_id"]),
            "restart_fresh": lambda: self.restart_job_fresh(job_id),
            "delete": lambda: self.delete_job(job_id),
            "clone": lambda: self.clone_job(job_id, kwargs.get("new_name")),
        }
        
        if action not in action_map:
            raise JobServiceError(
                message=f"Action {action} not implemented",
                error_code="NOT_IMPLEMENTED",
                user_message="This action is not available yet."
            )
        
        try:
            result = action_map[action]()
            return {
                "success": True,
                "action": action,
                "job_id": job_id,
                "result": result
            }
        except Exception as e:
            # Sanitize error for user
            sanitized = sanitized_log_service.sanitize_error(str(e))
            raise JobServiceError(
                message=str(e),  # Internal message (not exposed)
                error_code="ACTION_FAILED",
                user_message=f"Failed to {action}: {sanitized['user_message']}"
            )
    
    def get_allowed_actions(self, job_id: str) -> Dict[str, Any]:
        job = self.get_job(job_id)
        if not job:
            return {"error": "Job not found", "actions": []}
        
        training_active = self.is_training_active()
        checkpoints = self.db.query(Checkpoint).filter(Checkpoint.job_id == job_id).all()
        has_checkpoint = any(os.path.exists(cp.path) for cp in checkpoints)
        
        allowed = []
        blocked = []
        
        for action, valid_statuses in self.VALID_ACTIONS.items():
            if job.status not in valid_statuses:
                blocked.append({
                    "action": action,
                    "reason": f"Not available when job is {job.status}"
                })
                continue
            
            if action in ["start", "resume", "restart_fresh"] and training_active:
                blocked.append({
                    "action": action,
                    "reason": "Another training is running"
                })
                continue
            
            if action == "resume" and not has_checkpoint:
                blocked.append({
                    "action": action,
                    "reason": "No checkpoint available"
                })
                continue
            
            allowed.append(action)
        
        return {
            "job_id": job_id,
            "status": job.status,
            "allowed_actions": allowed,
            "blocked_actions": blocked,
            "has_checkpoint": has_checkpoint,
            "training_active": training_active
        }
    
    @staticmethod
    def _compute_config_hash(config: Dict[str, Any]) -> str:
        config_str = json.dumps(config, sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()
    
    def _full_memory_cleanup(self) -> Dict[str, Any]:
        gc.collect()
        
        gpu_result = gpu_service.full_cleanup()
        
        gc.collect()
        
        return {
            "gc_collected": True,
            "gpu_cleanup": gpu_result,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def recover_interrupted_jobs(self) -> List[Dict[str, Any]]:
        recovered = []
        
        running_jobs = self.db.query(TrainingJob).filter(
            TrainingJob.status == JobStatus.RUNNING.value
        ).all()
        
        for job in running_jobs:
            crash_info = sanitized_log_service.detect_crash_reason(
                error_message=job.error_message,
                last_gpu_info=self.get_system_state().gpu_info
            )
            
            job.status = JobStatus.FAILED.value
            job.error_message = "Server restart detected. Training was interrupted."
            job.completed_at = datetime.utcnow()
            if job.started_at:
                job.duration_seconds = int((job.completed_at - job.started_at).total_seconds())
            
            sanitized_log_service.create_terminal_log(
                job.id, 
                f"Training interrupted due to server restart. Reason: {crash_info['reason_display']}",
                "WARNING"
            )
            
            checkpoints = self.db.query(Checkpoint).filter(
                Checkpoint.job_id == job.id
            ).all()
            has_checkpoints = any(os.path.exists(cp.path) for cp in checkpoints)
            latest_checkpoint = None
            if has_checkpoints:
                valid_cps = [cp for cp in checkpoints if os.path.exists(cp.path)]
                if valid_cps:
                    latest_checkpoint = max(valid_cps, key=lambda x: x.step)
            
            recovered.append({
                "job_id": job.id,
                "name": job.name,
                "was_status": "RUNNING",
                "new_status": "FAILED",
                "crash_reason": crash_info['reason'].value,
                "crash_reason_display": crash_info['reason_display'],
                "suggestion": crash_info['suggestion'],
                "has_checkpoints": has_checkpoints,
                "latest_checkpoint_step": latest_checkpoint.step if latest_checkpoint else None,
                "can_resume": has_checkpoints and crash_info['can_resume'],
                "can_restart_fresh": True,
                "message": f"Training was interrupted ({crash_info['reason_display']}). " + (
                    f"You can resume from step {latest_checkpoint.step}." if latest_checkpoint 
                    else "No checkpoints available. You can restart training from scratch."
                )
            })
        
        state = self.get_system_state()
        state.is_training_active = False
        state.current_job_id = None
        state.last_cleanup_at = datetime.utcnow()
        
        self.db.commit()
        
        self._full_memory_cleanup()
        
        return recovered
    
    def get_interrupted_jobs(self) -> List[Dict[str, Any]]:
        interrupted = self.db.query(TrainingJob).filter(
            TrainingJob.status.in_([JobStatus.FAILED.value, JobStatus.CANCELLED.value])
        ).order_by(TrainingJob.updated_at.desc()).all()
        
        result = []
        for job in interrupted:
            checkpoints = self.db.query(Checkpoint).filter(
                Checkpoint.job_id == job.id
            ).all()
            
            valid_checkpoints = [cp for cp in checkpoints if os.path.exists(cp.path)]
            has_checkpoints = len(valid_checkpoints) > 0
            latest_checkpoint = max(valid_checkpoints, key=lambda x: x.step) if valid_checkpoints else None
            
            result.append({
                "job_id": job.id,
                "name": job.name,
                "status": job.status,
                "error_message": job.error_message,
                "created_at": job.created_at.isoformat() if job.created_at else None,
                "stopped_at": job.completed_at.isoformat() if job.completed_at else None,
                "has_checkpoints": has_checkpoints,
                "checkpoint_count": len(valid_checkpoints),
                "latest_checkpoint_step": latest_checkpoint.step if latest_checkpoint else None,
                "can_resume": has_checkpoints,
                "can_restart_fresh": True,
                "model_name": job.model_name,
                "dataset_id": job.dataset_id,
                "ui_actions": self._get_ui_actions_for_job(job, has_checkpoints)
            })
        
        return result
    
    def _get_ui_actions_for_job(self, job: TrainingJob, has_checkpoints: bool) -> List[Dict[str, Any]]:
        actions = []
        
        if has_checkpoints:
            actions.append({
                "action": "resume",
                "label": "Resume Training",
                "description": "Continue training from the last checkpoint",
                "primary": True,
                "icon": "play",
                "method": "resume_job_from_latest"
            })
        
        actions.append({
            "action": "restart_fresh",
            "label": "Start Fresh" if has_checkpoints else "Restart Training",
            "description": "Delete all progress and start training from scratch with the same settings",
            "primary": not has_checkpoints,
            "icon": "refresh",
            "method": "restart_job_fresh",
            "confirm_required": True,
            "confirm_message": "This will delete all checkpoints and logs. Are you sure?"
        })
        
        actions.append({
            "action": "delete",
            "label": "Delete Job",
            "description": "Delete this job and all associated data",
            "primary": False,
            "icon": "trash",
            "method": "delete_job",
            "confirm_required": True,
            "confirm_message": "This will permanently delete all data. Are you sure?"
        })
        
        return actions
    
    def restart_job_fresh(self, job_id: str) -> Dict[str, Any]:
        job = self.get_job(job_id)
        if not job:
            return {"success": False, "error": "Job not found"}
        
        if job.status == JobStatus.RUNNING.value:
            raise RuntimeError("Cannot restart a running job. Stop it first.")
        
        if self.is_training_active():
            raise RuntimeError("Another training job is running. Please wait for it to complete.")
        
        saved_config = {
            "name": job.name,
            "model_source": job.model_source,
            "model_path": job.model_path,
            "model_name": job.model_name,
            "training_config": job.training_config,
            "dataset_id": job.dataset_id,
        }
        
        delete_result = self.delete_job(job_id, delete_outputs=True)
        
        new_job = self.create_job(
            name=saved_config["name"],
            model_source=saved_config["model_source"],
            model_path=saved_config["model_path"],
            training_config=saved_config["training_config"],
            dataset_id=saved_config["dataset_id"],
            model_name=saved_config["model_name"],
        )
        
        self._add_log(new_job.id, "INFO", f"Fresh start - previous job {job_id} was deleted")
        self.db.commit()
        
        start_result = self.start_job(new_job.id)
        
        return {
            "success": True,
            "old_job_deleted": delete_result,
            "new_job": {
                "job_id": new_job.id,
                "name": new_job.name,
                "status": new_job.status,
                "output_dir": new_job.output_dir,
                "checkpoint_dir": new_job.checkpoint_dir,
                "log_file_path": new_job.log_file_path
            },
            "started": start_result,
            "message": "Old job deleted and new training started successfully"
        }
    
    def get_dashboard_status(self) -> Dict[str, Any]:
        state = self.get_system_state()
        current_job = self.get_current_job() if state.current_job_id else None
        interrupted_jobs = self.get_interrupted_jobs()
        
        return {
            "is_training_active": state.is_training_active,
            "current_job": {
                "job_id": current_job.id,
                "name": current_job.name,
                "status": current_job.status,
                "started_at": current_job.started_at.isoformat() if current_job.started_at else None,
                "model_name": current_job.model_name,
            } if current_job else None,
            "interrupted_jobs_count": len(interrupted_jobs),
            "interrupted_jobs": interrupted_jobs[:5],
            "total_jobs": self.db.query(TrainingJob).count(),
            "completed_jobs": self.db.query(TrainingJob).filter(
                TrainingJob.status == JobStatus.COMPLETED.value
            ).count(),
            "gpu_info": state.gpu_info,
            "last_cleanup_at": state.last_cleanup_at.isoformat() if state.last_cleanup_at else None,
            "ui_message": self._get_dashboard_message(state, current_job, interrupted_jobs)
        }
    
    def _get_dashboard_message(self, state: SystemState, current_job: Optional[TrainingJob], interrupted_jobs: List) -> Dict[str, Any]:
        if state.is_training_active and current_job:
            return {
                "type": "info",
                "title": "Training in Progress",
                "message": f"'{current_job.name}' is currently training. You can monitor progress or stop the training.",
                "actions": ["monitor", "stop"]
            }
        elif interrupted_jobs:
            resumable = [j for j in interrupted_jobs if j["can_resume"]]
            if resumable:
                return {
                    "type": "warning",
                    "title": "Interrupted Training Found",
                    "message": f"You have {len(resumable)} training job(s) that can be resumed from checkpoint.",
                    "actions": ["resume", "restart_fresh", "delete"]
                }
            else:
                return {
                    "type": "warning",
                    "title": "Failed Training Found",
                    "message": f"You have {len(interrupted_jobs)} failed job(s). You can restart training with the same settings.",
                    "actions": ["restart_fresh", "delete"]
                }
        else:
            return {
                "type": "success",
                "title": "Ready to Train",
                "message": "No training is running. You can start a new training job.",
                "actions": ["create_new"]
            }
    
    def get_system_state(self) -> SystemState:
        state = self.db.query(SystemState).first()
        if not state:
            state = SystemState(id=1, is_training_active=False)
            self.db.add(state)
            self.db.commit()
        return state
    
    def is_training_active(self) -> bool:
        state = self.get_system_state()
        return state.is_training_active
    
    def get_current_job(self) -> Optional[TrainingJob]:
        state = self.get_system_state()
        if state.current_job_id:
            return self.get_job(state.current_job_id)
        return None
    
    def list_jobs(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[TrainingJob]:
        query = self.db.query(TrainingJob)
        
        if status:
            query = query.filter(TrainingJob.status == status)
        
        return query.order_by(TrainingJob.created_at.desc()).offset(offset).limit(limit).all()
    
    def get_job(self, job_id: str) -> Optional[TrainingJob]:
        return self.db.query(TrainingJob).filter(TrainingJob.id == job_id).first()
    
    def update_job_config(self, job_id: str, training_config: Dict[str, Any]) -> TrainingJob:
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")
        
        if job.config_finalized:
            raise RuntimeError("Training configuration cannot be modified after job has started.")
        
        if job.status not in [JobStatus.PENDING.value]:
            raise RuntimeError(f"Training config can only be modified when job is PENDING. Current status: {job.status}")
        
        job.training_config = training_config
        job.updated_at = datetime.utcnow()
        self.db.commit()
        return job
    
    def create_job(
        self,
        name: Optional[str],
        model_source: str,
        model_path: str,
        training_config: Dict[str, Any],
        dataset_id: Optional[str] = None,
        model_name: Optional[str] = None,
    ) -> TrainingJob:
        """Create a new training job.
        
        Args:
            name: Optional custom name. If None or empty, auto-generates a unique name.
            model_source: Source of the model (huggingface, modelscope, local)
            model_path: Path or ID of the model
            training_config: Training configuration dict
            dataset_id: Optional dataset ID
            model_name: Optional display name for the model
            
        Returns:
            Created TrainingJob instance
        """
        if self.is_training_active():
            raise RuntimeError("A training job is already running. Please wait for it to complete.")
        
        if dataset_id:
            dataset = self.db.query(Dataset).filter(Dataset.id == dataset_id).first()
            if not dataset:
                raise ValueError(f"Dataset not found: {dataset_id}")
        
        # Auto-generate name if not provided or empty
        if not name or not name.strip():
            name = self.generate_training_name(model_path)
        else:
            name = name.strip()
            # Validate name length
            if len(name) > 255:
                raise ValueError("Training name cannot exceed 255 characters")
            # Check if name is already used
            if not self.is_name_available(name):
                raise ValueError(f"Training name '{name}' is already in use. Please choose a different name.")
        
        job = TrainingJob(
            name=name,
            model_source=model_source,
            model_path=model_path,
            model_name=model_name or model_path.split("/")[-1],
            training_config=training_config,
            dataset_id=dataset_id,
            status=JobStatus.PENDING.value,
        )
        
        self.db.add(job)
        self.db.flush()
        
        job.output_dir = os.path.join(self.OUTPUT_BASE_PATH, job.id)
        job.checkpoint_dir = os.path.join(self.CHECKPOINT_BASE_PATH, job.id)
        job.log_file_path = os.path.join(self.LOG_BASE_PATH, f"{job.id}.log")
        
        job.original_config_hash = self._compute_config_hash(training_config)
        
        os.makedirs(job.output_dir, exist_ok=True)
        os.makedirs(job.checkpoint_dir, exist_ok=True)
        
        self.db.commit()
        return job
    
    def start_job(self, job_id: str) -> Dict[str, Any]:
        if self.is_training_active():
            raise RuntimeError("A training job is already running. Please wait for it to complete.")
        
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")
        
        if job.status not in [JobStatus.PENDING.value]:
            raise ValueError(f"Job cannot be started from status: {job.status}. Use resume for FAILED/CANCELLED jobs.")
        
        cleanup_result = self._full_memory_cleanup()
        self._add_log(job_id, "INFO", f"Pre-training full cleanup completed: {cleanup_result}")
        
        job.status = JobStatus.RUNNING.value
        job.started_at = datetime.utcnow()
        job.error_message = None
        job.config_finalized = True
        
        state = self.get_system_state()
        state.current_job_id = job_id
        state.is_training_active = True
        state.gpu_info = gpu_service.get_gpu_info()
        
        self._add_log(job_id, "INFO", f"Training started for job {job_id}")
        self._add_log(job_id, "INFO", f"Config hash: {job.original_config_hash}")
        self._add_log(job_id, "INFO", f"Dataset ID: {job.dataset_id}")
        self._add_log(job_id, "INFO", f"Model: {job.model_path}")
        
        sanitized_log_service.log_session_start(
            job_id, 
            session_type="START",
            config_summary=job.training_config
        )
        sanitized_log_service.log_gpu_status(job_id, state.gpu_info)
        
        self.db.commit()
        
        return {
            "job_id": job.id,
            "status": "started",
            "config_hash": job.original_config_hash,
            "output_dir": job.output_dir,
            "checkpoint_dir": job.checkpoint_dir,
            "log_file_path": job.log_file_path,
            "terminal_log_path": sanitized_log_service.get_terminal_log_path(job_id),
            "training_config": job.training_config,
            "dataset_id": job.dataset_id,
            "model_path": job.model_path
        }
    
    def complete_job(self, job_id: str, success: bool, error_message: Optional[str] = None,
                     final_loss: float = None) -> TrainingJob:
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")
        
        job.status = JobStatus.COMPLETED.value if success else JobStatus.FAILED.value
        job.completed_at = datetime.utcnow()
        job.error_message = error_message
        
        if job.started_at:
            job.duration_seconds = int((job.completed_at - job.started_at).total_seconds())
        
        job.status = JobStatus.CLEANUP.value
        self.db.commit()
        
        cleanup_result = gpu_service.full_cleanup()
        self._add_log(job_id, "INFO", f"Post-training cleanup completed: {cleanup_result}")
        
        job.status = JobStatus.COMPLETED.value if success else JobStatus.FAILED.value
        
        sanitized_log_service.log_session_end(
            job_id,
            status="COMPLETED" if success else "FAILED",
            duration_seconds=job.duration_seconds,
            final_loss=final_loss,
            error_message=error_message
        )
        
        state = self.get_system_state()
        state.current_job_id = None
        state.is_training_active = False
        state.last_cleanup_at = datetime.utcnow()
        
        self.db.commit()
        return job
    
    def cancel_job(self, job_id: str) -> TrainingJob:
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")
        
        if job.status not in [JobStatus.PENDING.value, JobStatus.QUEUED.value, JobStatus.RUNNING.value]:
            raise ValueError(f"Job cannot be cancelled from status: {job.status}")
        
        job.status = JobStatus.CANCELLED.value
        job.completed_at = datetime.utcnow()
        
        if job.started_at:
            job.duration_seconds = int((job.completed_at - job.started_at).total_seconds())
        
        cleanup_result = gpu_service.full_cleanup()
        self._add_log(job_id, "INFO", f"Job cancelled, cleanup completed: {cleanup_result}")
        
        sanitized_log_service.log_session_end(
            job_id,
            status="CANCELLED",
            duration_seconds=job.duration_seconds
        )
        
        state = self.get_system_state()
        if state.current_job_id == job_id:
            state.current_job_id = None
            state.is_training_active = False
            state.last_cleanup_at = datetime.utcnow()
        
        self.db.commit()
        return job
    
    def resume_job(self, job_id: str, checkpoint_id: str) -> Dict[str, Any]:
        if self.is_training_active():
            raise RuntimeError("A training job is already running. Please wait for it to complete.")
        
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")
        
        if job.status not in [JobStatus.FAILED.value, JobStatus.CANCELLED.value]:
            raise ValueError(f"Job cannot be resumed from status: {job.status}. Only FAILED or CANCELLED jobs can be resumed.")
        
        checkpoint = self.db.query(Checkpoint).filter(
            Checkpoint.id == checkpoint_id,
            Checkpoint.job_id == job_id
        ).first()
        
        if not checkpoint:
            raise ValueError(f"Checkpoint not found: {checkpoint_id}")
        
        if checkpoint.job_id != job_id:
            raise ValueError(f"Checkpoint {checkpoint_id} does not belong to job {job_id}. Possible mixing detected.")
        
        if not os.path.exists(checkpoint.path):
            raise ValueError(f"Checkpoint files not found at: {checkpoint.path}")
        
        cleanup_result = self._full_memory_cleanup()
        self._add_log(job_id, "INFO", f"Pre-resume full cleanup completed: {cleanup_result}")
        
        job.status = JobStatus.RUNNING.value
        job.started_at = datetime.utcnow()
        job.completed_at = None
        job.error_message = None
        job.resume_from_checkpoint_id = checkpoint_id
        job.resume_from_step = checkpoint.step
        job.resume_count = (job.resume_count or 0) + 1
        
        state = self.get_system_state()
        state.current_job_id = job_id
        state.is_training_active = True
        state.gpu_info = gpu_service.get_gpu_info()
        
        self._add_log(job_id, "INFO", f"Resuming job {job_id} from checkpoint at step {checkpoint.step}")
        self._add_log(job_id, "INFO", f"Resume count: {job.resume_count}")
        self._add_log(job_id, "INFO", f"Config hash (unchanged): {job.original_config_hash}")
        self._add_log(job_id, "INFO", f"Checkpoint path: {checkpoint.path}")
        
        sanitized_log_service.log_session_start(
            job_id,
            session_type="RESUME",
            resume_from_step=checkpoint.step,
            config_summary=job.training_config
        )
        sanitized_log_service.log_gpu_status(job_id, state.gpu_info)
        
        self.db.commit()
        
        return {
            "job_id": job_id,
            "checkpoint_id": checkpoint_id,
            "resume_from_step": checkpoint.step,
            "checkpoint_path": checkpoint.path,
            "status": "resumed",
            "resume_count": job.resume_count,
            "config_hash": job.original_config_hash,
            "training_config": job.training_config,
            "dataset_id": job.dataset_id,
            "model_path": job.model_path,
            "output_dir": job.output_dir,
            "log_file_path": job.log_file_path,
            "terminal_log_path": sanitized_log_service.get_terminal_log_path(job_id)
        }
    
    def get_resumable_checkpoints(self, job_id: str) -> Dict[str, Any]:
        job = self.get_job(job_id)
        if not job:
            return {"error": "Job not found", "checkpoints": [], "has_checkpoints": False}
        
        if job.status not in [JobStatus.FAILED.value, JobStatus.CANCELLED.value]:
            return {
                "error": f"Job status is {job.status}. Only FAILED or CANCELLED jobs can be resumed.",
                "checkpoints": [],
                "has_checkpoints": False
            }
        
        checkpoints = self.get_job_checkpoints(job_id)
        resumable = []
        latest_checkpoint = None
        
        for cp in checkpoints:
            if os.path.exists(cp.path):
                cp_data = {
                    "id": cp.id,
                    "step": cp.step,
                    "epoch": cp.epoch,
                    "path": cp.path,
                    "size_mb": cp.size_mb,
                    "is_best": cp.is_best,
                    "is_final": cp.is_final,
                    "created_at": cp.created_at.isoformat() if cp.created_at else None,
                    "metrics": cp.metrics_snapshot,
                }
                resumable.append(cp_data)
                
                if latest_checkpoint is None or cp.step > latest_checkpoint["step"]:
                    latest_checkpoint = cp_data
        
        return {
            "job_id": job_id,
            "job_name": job.name,
            "job_status": job.status,
            "checkpoints": resumable,
            "has_checkpoints": len(resumable) > 0,
            "latest_checkpoint": latest_checkpoint,
            "recommended_action": "resume" if len(resumable) > 0 else "clone_and_start_fresh"
        }
    
    def resume_job_from_latest(self, job_id: str) -> Dict[str, Any]:
        checkpoint_info = self.get_resumable_checkpoints(job_id)
        
        if not checkpoint_info.get("has_checkpoints"):
            raise ValueError(f"No checkpoints available for job {job_id}. Use clone_job() to start fresh with same config.")
        
        latest = checkpoint_info["latest_checkpoint"]
        return self.resume_job(job_id, latest["id"])
    
    def clone_job(self, job_id: str, new_name: Optional[str] = None) -> TrainingJob:
        original_job = self.get_job(job_id)
        if not original_job:
            raise ValueError(f"Job not found: {job_id}")
        
        if self.is_training_active():
            raise RuntimeError("A training job is already running. Please wait for it to complete.")
        
        new_job = self.create_job(
            name=new_name or f"{original_job.name} (copy)",
            model_source=original_job.model_source,
            model_path=original_job.model_path,
            training_config=original_job.training_config,
            dataset_id=original_job.dataset_id,
            model_name=original_job.model_name,
        )
        
        self._add_log(new_job.id, "INFO", f"Job cloned from original job {job_id}")
        self._add_log(new_job.id, "INFO", f"Using same config hash: {new_job.original_config_hash}")
        self.db.commit()
        
        return new_job
    
    def get_job_status_for_action(self, job_id: str) -> Dict[str, Any]:
        job = self.get_job(job_id)
        if not job:
            return {"error": "Job not found"}
        
        checkpoint_info = self.get_resumable_checkpoints(job_id) if job.status in [JobStatus.FAILED.value, JobStatus.CANCELLED.value] else {"has_checkpoints": False}
        
        can_start = job.status == JobStatus.PENDING.value and not self.is_training_active()
        can_resume = job.status in [JobStatus.FAILED.value, JobStatus.CANCELLED.value] and checkpoint_info.get("has_checkpoints", False) and not self.is_training_active()
        can_stop = job.status == JobStatus.RUNNING.value
        can_clone = job.status not in [JobStatus.RUNNING.value]
        can_delete = job.status not in [JobStatus.RUNNING.value]
        
        return {
            "job_id": job_id,
            "job_name": job.name,
            "status": job.status,
            "can_start": can_start,
            "can_resume": can_resume,
            "can_stop": can_stop,
            "can_clone": can_clone,
            "can_delete": can_delete,
            "has_checkpoints": checkpoint_info.get("has_checkpoints", False),
            "latest_checkpoint": checkpoint_info.get("latest_checkpoint"),
            "resume_count": job.resume_count or 0,
            "recommended_action": self._get_recommended_action(job, checkpoint_info)
        }
    
    def _get_recommended_action(self, job: TrainingJob, checkpoint_info: Dict) -> str:
        if job.status == JobStatus.PENDING.value:
            return "start"
        elif job.status == JobStatus.RUNNING.value:
            return "wait_or_stop"
        elif job.status in [JobStatus.FAILED.value, JobStatus.CANCELLED.value]:
            if checkpoint_info.get("has_checkpoints"):
                return "resume_from_latest"
            else:
                return "clone_and_start_fresh"
        elif job.status == JobStatus.COMPLETED.value:
            return "completed_can_clone_or_delete"
        return "unknown"
    
    def add_metric(
        self,
        job_id: str,
        step: int,
        epoch: Optional[float] = None,
        loss: Optional[float] = None,
        learning_rate: Optional[float] = None,
        grad_norm: Optional[float] = None,
        eval_loss: Optional[float] = None,
        eval_accuracy: Optional[float] = None,
        extra_metrics: Optional[Dict] = None,
    ) -> TrainingMetric:
        existing = self.db.query(TrainingMetric).filter(
            TrainingMetric.job_id == job_id,
            TrainingMetric.step == step
        ).first()
        
        if existing:
            existing.loss = loss if loss is not None else existing.loss
            existing.learning_rate = learning_rate if learning_rate is not None else existing.learning_rate
            existing.eval_loss = eval_loss if eval_loss is not None else existing.eval_loss
            existing.eval_accuracy = eval_accuracy if eval_accuracy is not None else existing.eval_accuracy
            if extra_metrics:
                existing.extra_metrics = {**(existing.extra_metrics or {}), **extra_metrics}
            self.db.commit()
            return existing
        
        gpu_info = gpu_service.get_current_memory_usage()
        
        metric = TrainingMetric(
            job_id=job_id,
            step=step,
            epoch=epoch,
            loss=loss,
            learning_rate=learning_rate,
            grad_norm=grad_norm,
            eval_loss=eval_loss,
            eval_accuracy=eval_accuracy,
            extra_metrics=extra_metrics,
            gpu_memory_mb=gpu_info.get("allocated_mb"),
        )
        
        self.db.add(metric)
        self.db.commit()
        return metric
    
    def get_job_metrics(self, job_id: str, limit: int = 1000) -> List[TrainingMetric]:
        return self.db.query(TrainingMetric).filter(
            TrainingMetric.job_id == job_id
        ).order_by(TrainingMetric.step.asc()).limit(limit).all()
    
    def add_checkpoint(
        self,
        job_id: str,
        step: int,
        path: str,
        epoch: Optional[float] = None,
        is_best: bool = False,
        is_final: bool = False,
        metrics_snapshot: Optional[Dict] = None,
    ) -> Checkpoint:
        size_mb = 0
        if os.path.exists(path):
            if os.path.isdir(path):
                for root, dirs, files in os.walk(path):
                    for f in files:
                        size_mb += os.path.getsize(os.path.join(root, f))
                size_mb = size_mb / (1024 * 1024)
            else:
                size_mb = os.path.getsize(path) / (1024 * 1024)
        
        checkpoint = Checkpoint(
            job_id=job_id,
            step=step,
            epoch=epoch,
            path=path,
            size_mb=size_mb,
            is_best=is_best,
            is_final=is_final,
            metrics_snapshot=metrics_snapshot,
        )
        
        self.db.add(checkpoint)
        self.db.commit()
        return checkpoint
    
    def get_job_checkpoints(self, job_id: str) -> List[Checkpoint]:
        return self.db.query(Checkpoint).filter(
            Checkpoint.job_id == job_id
        ).order_by(Checkpoint.step.asc()).all()
    
    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        checkpoint = self.db.query(Checkpoint).filter(Checkpoint.id == checkpoint_id).first()
        if not checkpoint:
            return False
        
        if os.path.exists(checkpoint.path):
            if os.path.isdir(checkpoint.path):
                shutil.rmtree(checkpoint.path, ignore_errors=True)
            else:
                os.remove(checkpoint.path)
        
        self.db.delete(checkpoint)
        self.db.commit()
        return True
    
    def delete_job_outputs(self, job_id: str, keep_final: bool = True) -> Dict[str, Any]:
        job = self.get_job(job_id)
        if not job:
            return {"error": "Job not found"}
        
        result = {"deleted_checkpoints": 0, "freed_mb": 0}
        
        checkpoints = self.get_job_checkpoints(job_id)
        for cp in checkpoints:
            if keep_final and cp.is_final:
                continue
            
            result["freed_mb"] += cp.size_mb or 0
            if os.path.exists(cp.path):
                if os.path.isdir(cp.path):
                    shutil.rmtree(cp.path, ignore_errors=True)
                else:
                    os.remove(cp.path)
            
            self.db.delete(cp)
            result["deleted_checkpoints"] += 1
        
        self.db.commit()
        return result
    
    def delete_job(self, job_id: str, delete_outputs: bool = True) -> Dict[str, Any]:
        job = self.get_job(job_id)
        if not job:
            return {"success": False, "error": "Job not found"}
        
        if job.status == JobStatus.RUNNING.value:
            raise RuntimeError("Cannot delete a running job. Cancel it first.")
        
        deleted_items = {
            "job_id": job_id,
            "job_name": job.name,
            "output_dir_deleted": False,
            "checkpoint_dir_deleted": False,
            "log_file_deleted": False,
            "metrics_deleted": 0,
            "checkpoints_deleted": 0,
            "logs_deleted": 0,
            "freed_space_mb": 0
        }
        
        if delete_outputs:
            if job.output_dir and os.path.exists(job.output_dir):
                deleted_items["freed_space_mb"] += self._get_dir_size_mb(job.output_dir)
                shutil.rmtree(job.output_dir, ignore_errors=True)
                deleted_items["output_dir_deleted"] = True
            
            if job.checkpoint_dir and os.path.exists(job.checkpoint_dir):
                deleted_items["freed_space_mb"] += self._get_dir_size_mb(job.checkpoint_dir)
                shutil.rmtree(job.checkpoint_dir, ignore_errors=True)
                deleted_items["checkpoint_dir_deleted"] = True
            
            if job.log_file_path and os.path.exists(job.log_file_path):
                deleted_items["freed_space_mb"] += os.path.getsize(job.log_file_path) / (1024 * 1024)
                os.remove(job.log_file_path)
                deleted_items["log_file_deleted"] = True
            
            # Delete terminal log (user-facing log)
            terminal_log_path = sanitized_log_service.get_terminal_log_path(job_id)
            if os.path.exists(terminal_log_path):
                deleted_items["freed_space_mb"] += os.path.getsize(terminal_log_path) / (1024 * 1024)
                sanitized_log_service.delete_terminal_log(job_id)
                deleted_items["terminal_log_deleted"] = True
            else:
                deleted_items["terminal_log_deleted"] = False
            
            # Delete final model path if it exists and is different from output_dir
            if job.final_model_path and os.path.exists(job.final_model_path):
                if job.final_model_path != job.output_dir:
                    deleted_items["freed_space_mb"] += self._get_dir_size_mb(job.final_model_path)
                    shutil.rmtree(job.final_model_path, ignore_errors=True)
                deleted_items["final_model_deleted"] = True
            else:
                deleted_items["final_model_deleted"] = False
        
        # NOTE: We do NOT delete the dataset or base model - they are GLOBAL resources
        # that can be used by other trainings. Only training-specific data is deleted.
        
        deleted_items["metrics_deleted"] = self.db.query(TrainingMetric).filter(
            TrainingMetric.job_id == job_id
        ).delete()
        
        deleted_items["checkpoints_deleted"] = self.db.query(Checkpoint).filter(
            Checkpoint.job_id == job_id
        ).delete()
        
        deleted_items["logs_deleted"] = self.db.query(TrainingLog).filter(
            TrainingLog.job_id == job_id
        ).delete()
        
        self.db.delete(job)
        self.db.commit()
        
        deleted_items["success"] = True
        deleted_items["message"] = f"Job {job_id} and all related data deleted successfully"
        
        return deleted_items
    
    def _get_dir_size_mb(self, path: str) -> float:
        total = 0
        if os.path.exists(path):
            for root, dirs, files in os.walk(path):
                for f in files:
                    try:
                        total += os.path.getsize(os.path.join(root, f))
                    except Exception:
                        pass
        return total / (1024 * 1024)
    
    def _add_log(self, job_id: str, level: str, message: str):
        encrypted_message = encrypted_log_service.format_log_entry(level, message, job_id)
        log = TrainingLog(job_id=job_id, level=level, message_encrypted=encrypted_message)
        self.db.add(log)
        
        job = self.get_job(job_id)
        if job and job.log_file_path:
            try:
                with open(job.log_file_path, 'a') as f:
                    f.write(encrypted_message + '\n')
            except Exception:
                pass
    
    def get_job_logs(self, job_id: str, limit: int = 500, level: Optional[str] = None) -> List[TrainingLog]:
        query = self.db.query(TrainingLog).filter(TrainingLog.job_id == job_id)
        if level:
            query = query.filter(TrainingLog.level == level)
        return query.order_by(TrainingLog.timestamp.desc()).limit(limit).all()
    
    def export_job_logs(self, job_id: str) -> Dict[str, Any]:
        job = self.get_job(job_id)
        if not job:
            return {"error": "Job not found"}
        
        logs = self.db.query(TrainingLog).filter(
            TrainingLog.job_id == job_id
        ).order_by(TrainingLog.timestamp.asc()).all()
        
        export_data = {
            "job_id": job_id,
            "job_name": job.name,
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "status": job.status,
            "total_logs": len(logs),
            "log_file_path": job.log_file_path,
            "encrypted_logs": [
                {
                    "id": log.id,
                    "level": log.level,
                    "message_encrypted": log.message_encrypted,
                    "timestamp": log.timestamp.isoformat() if log.timestamp else None
                }
                for log in logs
            ]
        }
        
        return export_data
    
    def get_job_log_file_path(self, job_id: str) -> Optional[str]:
        job = self.get_job(job_id)
        if not job:
            return None
        return job.log_file_path if job.log_file_path and os.path.exists(job.log_file_path) else None
    
    def get_storage_stats(self) -> Dict[str, Any]:
        def get_dir_size(path):
            total = 0
            if os.path.exists(path):
                for root, dirs, files in os.walk(path):
                    for f in files:
                        total += os.path.getsize(os.path.join(root, f))
            return total / (1024 * 1024)
        
        return {
            "output_path": self.OUTPUT_BASE_PATH,
            "output_size_mb": get_dir_size(self.OUTPUT_BASE_PATH),
            "checkpoint_path": self.CHECKPOINT_BASE_PATH,
            "checkpoint_size_mb": get_dir_size(self.CHECKPOINT_BASE_PATH),
            "total_jobs": self.db.query(TrainingJob).count(),
            "total_checkpoints": self.db.query(Checkpoint).count(),
        }
    
    def add_terminal_log(self, job_id: str, message: str, level: str = "INFO") -> str:
        return sanitized_log_service.create_terminal_log(job_id, message, level)
    
    def get_terminal_logs(self, job_id: str, lines: int = 100) -> List[str]:
        return sanitized_log_service.get_terminal_logs(job_id, lines)
    
    def get_terminal_log_path(self, job_id: str) -> Optional[str]:
        path = sanitized_log_service.get_terminal_log_path(job_id)
        return path if os.path.exists(path) else None
    
    def log_training_progress(self, job_id: str, step: int, total_steps: int, 
                               loss: float, lr: float = None, epoch: float = None,
                               gpu_memory_mb: float = None) -> str:
        message = sanitized_log_service.format_training_progress(
            step, total_steps, loss, lr, epoch, gpu_memory_mb
        )
        return self.add_terminal_log(job_id, message, "INFO")
    
    def log_training_error(self, job_id: str, error_message: str) -> Dict[str, Any]:
        sanitized = sanitized_log_service.sanitize_error(error_message)
        
        self.add_terminal_log(job_id, sanitized['user_message'], "ERROR")
        
        self._add_log(job_id, "ERROR", error_message)
        
        return {
            "logged": True,
            "terminal_message": sanitized['user_message'],
            "reason": sanitized['reason'].value,
            "severity": sanitized['severity'].value,
            "safe_to_display": sanitized['safe_to_display']
        }
    
    def complete_job_with_error(self, job_id: str, error_message: str) -> Dict[str, Any]:
        sanitized = sanitized_log_service.sanitize_error(error_message)
        crash_info = sanitized_log_service.detect_crash_reason(error_message)
        
        job = self.get_job(job_id)
        if not job:
            return {"error": "Job not found"}
        
        job.status = JobStatus.FAILED.value
        job.error_message = sanitized['user_message']
        job.completed_at = datetime.utcnow()
        if job.started_at:
            job.duration_seconds = int((job.completed_at - job.started_at).total_seconds())
        
        self.add_terminal_log(job_id, f"Training failed: {sanitized['user_message']}", "ERROR")
        self.add_terminal_log(job_id, f"Suggestion: {crash_info['suggestion']}", "INFO")
        
        self._add_log(job_id, "ERROR", error_message)
        
        cleanup_result = self._full_memory_cleanup()
        
        state = self.get_system_state()
        state.current_job_id = None
        state.is_training_active = False
        state.last_cleanup_at = datetime.utcnow()
        
        self.db.commit()
        
        checkpoints = self.db.query(Checkpoint).filter(Checkpoint.job_id == job_id).all()
        has_checkpoint = any(os.path.exists(cp.path) for cp in checkpoints)
        
        return {
            "job_id": job_id,
            "status": "failed",
            "error_display": sanitized['user_message'],
            "crash_reason": crash_info['reason'].value,
            "crash_reason_display": crash_info['reason_display'],
            "suggestion": crash_info['suggestion'],
            "can_resume": has_checkpoint and crash_info['can_resume'],
            "can_restart_fresh": True,
            "cleanup_completed": True
        }
