# Copyright (c) US Inc. All rights reserved.
"""Job management endpoints"""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ...models.schemas import JobInfo, JobResponse, JobStatus, TrainingConfig
from ...services.job_manager import job_manager
from ...services.training_service import training_service
from ...services.websocket_manager import ws_manager
from ...services.job_service import JobService
from ...services.encrypted_log_service import encrypted_log_service
from ...core.database import get_db
from ...core.config import settings
from ...core.capabilities import get_validator, is_system_expired


def _debug_log(message: str, job_id: str = "system", level: str = "DEBUG"):
    """Write debug log to ENCRYPTED log file (only US Inc can read)."""
    encrypted_log_service.encrypt_and_format(f"[JOBS_API] {message}", job_id, level)


router = APIRouter()


class JobNameUpdate(BaseModel):
    """Request model for updating job name"""
    name: str


@router.post("/create", response_model=JobInfo)
async def create_job(config: TrainingConfig):
    """Create a new training job"""
    _debug_log(f"create_job called with config: {config}")
    try:
        # Check system expiration first
        expired, exp_msg = is_system_expired()
        if expired:
            _debug_log(f"System expired: {exp_msg}", level="ERROR")
            raise HTTPException(status_code=403, detail=exp_msg)
        
        validator = get_validator()
        model_source = config.model_source.value if hasattr(config.model_source, 'value') else str(config.model_source)
        _debug_log(f"Model source: {model_source}, Model path: {config.model_path}")
        
        # Validate model path against system restrictions
        is_supported, message = validator.validate_model_path(config.model_path, model_source)
        if not is_supported:
            _debug_log(f"Model validation failed: {message}", level="ERROR")
            raise HTTPException(status_code=403, detail=message)
        
        # Validate dataset paths
        if hasattr(config, 'dataset_path') and config.dataset_path:
            _debug_log(f"Dataset path: {config.dataset_path}")
            dataset_paths = config.dataset_path.split(',') if isinstance(config.dataset_path, str) else config.dataset_path
            for ds_path in dataset_paths:
                ds_path = ds_path.strip()
                if ds_path.upper().startswith('HF::'):
                    is_valid, ds_msg = validator.validate_dataset_source('huggingface')
                    if not is_valid:
                        _debug_log(f"Dataset validation failed: {ds_msg}", level="ERROR")
                        raise HTTPException(status_code=403, detail=ds_msg)
                elif ds_path.upper().startswith('MS::'):
                    is_valid, ds_msg = validator.validate_dataset_source('modelscope')
                    if not is_valid:
                        _debug_log(f"Dataset validation failed: {ds_msg}", level="ERROR")
                        raise HTTPException(status_code=403, detail=ds_msg)
        
        # Architecture validation happens when model is loaded
        _debug_log("Creating job in job_manager...")
        job = await job_manager.create_job(config)
        _debug_log(f"Job created: {job.job_id}")
        return job
    except HTTPException:
        raise
    except ValueError as e:
        _debug_log(f"ValueError: {e}", level="ERROR")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        _debug_log(f"Unexpected error: {e}", level="ERROR")
        raise HTTPException(status_code=500, detail=f"Failed to create training job: {str(e)}")


@router.post("/{job_id}/start")
async def start_job(job_id: str):
    """Start a training job"""
    _debug_log(f"start_job called for job_id: {job_id}", job_id)
    
    job = await job_manager.get_job(job_id)
    if not job:
        _debug_log(f"Job {job_id} not found", job_id, "ERROR")
        raise HTTPException(status_code=404, detail="Job not found")
    
    _debug_log(f"Job {job_id} found, status: {job.status}", job_id)
    
    if job.status == JobStatus.RUNNING:
        _debug_log(f"Job {job_id} is already running", job_id, "WARNING")
        raise HTTPException(status_code=400, detail="Job is already running")
    
    if job.status == JobStatus.COMPLETED:
        _debug_log(f"Job {job_id} has already completed", job_id, "WARNING")
        raise HTTPException(status_code=400, detail="Job has already completed")
    
    _debug_log(f"Calling training_service.start_training for {job_id}", job_id)
    success = await training_service.start_training(job_id)
    _debug_log(f"start_training returned: {success}", job_id)
    
    if not success:
        _debug_log(f"Failed to start training for {job_id}", job_id, "ERROR")
        raise HTTPException(status_code=500, detail="Failed to start training")
    
    return {"job_id": job_id, "status": "starting"}


@router.post("/{job_id}/stop")
async def stop_job(job_id: str):
    """Stop a running job"""
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != JobStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Job is not running")
    
    success = await training_service.stop_training(job_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to stop training")
    
    return {"job_id": job_id, "status": "stopped"}


@router.get("/{job_id}", response_model=JobResponse)
async def get_job(job_id: str, logs_limit: int = Query(100, ge=1, le=1000)):
    """Get job status and logs"""
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    logs = await job_manager.get_logs(job_id, last_n=logs_limit)
    
    return JobResponse(job=job, logs=logs)


@router.get("/", response_model=List[JobInfo])
async def list_jobs(
    status: Optional[JobStatus] = Query(None, description="Filter by status"),
    limit: int = Query(50, ge=1, le=100)
):
    """List all jobs"""
    jobs = await job_manager.get_all_jobs()
    
    if status:
        jobs = [j for j in jobs if j.status == status]
    
    # Sort by created_at descending
    jobs.sort(key=lambda x: x.created_at, reverse=True)
    
    return jobs[:limit]


@router.patch("/{job_id}/name")
async def update_job_name(job_id: str, update: JobNameUpdate, db: Session = Depends(get_db)):
    """Update the name of a training job"""
    try:
        service = JobService(db)
        result = service.update_job_name(job_id, update.name)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to update job name")


@router.get("/{job_id}/delete-info")
async def get_delete_info(job_id: str):
    """Get information needed for delete confirmation (returns job name)"""
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    return {
        "job_id": job_id,
        "job_name": job.name,
        "status": job.status,
        "can_delete": job.status != JobStatus.RUNNING,
        "confirm_text": job.name  # User must type this to confirm deletion
    }


@router.delete("/{job_id}")
async def delete_job(
    job_id: str,
    confirm: str = Query(..., description="Type the job NAME to confirm deletion")
):
    """Delete a job (only if not running). User must type the job name to confirm."""
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status == JobStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Cannot delete a running job. Stop it first.")
    
    # Verify confirmation matches the job name
    if confirm != job.name:
        raise HTTPException(
            status_code=400, 
            detail=f"Confirmation failed. You must type '{job.name}' to delete this training."
        )
    
    success = await job_manager.delete_job(job_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to delete job")
    
    return {"job_id": job_id, "job_name": job.name, "deleted": True}


@router.get("/debug/all")
async def debug_all_jobs():
    """Debug endpoint - get all jobs with full details"""
    try:
        jobs = await job_manager.get_all_jobs()
        return {
            "total_jobs": len(jobs),
            "jobs": [
                {
                    "job_id": j.job_id,
                    "name": j.name,
                    "status": j.status.value if hasattr(j.status, 'value') else str(j.status),
                    "model_path": j.config.model_path if j.config else None,
                    "dataset_path": j.config.dataset_path if j.config else None,
                    "created_at": str(j.created_at) if j.created_at else None,
                    "started_at": str(j.started_at) if j.started_at else None,
                    "error": j.error,
                    "current_step": j.current_step,
                    "total_steps": j.total_steps,
                }
                for j in jobs
            ]
        }
    except Exception as e:
        return {"error": str(e)}


@router.get("/debug/{job_id}")
async def debug_job(job_id: str):
    """Debug endpoint - get full job details and logs"""
    try:
        job = await job_manager.get_job(job_id)
        if not job:
            return {"error": "Job not found"}
        
        logs = await job_manager.get_logs(job_id, last_n=200)
        
        return {
            "job_id": job.job_id,
            "name": job.name,
            "status": job.status.value if hasattr(job.status, 'value') else str(job.status),
            "config": job.config.model_dump() if job.config else None,
            "created_at": str(job.created_at) if job.created_at else None,
            "started_at": str(job.started_at) if job.started_at else None,
            "completed_at": str(job.completed_at) if job.completed_at else None,
            "error": job.error,
            "current_step": job.current_step,
            "total_steps": job.total_steps,
            "current_loss": job.current_loss,
            "logs_count": len(logs),
            "last_logs": logs[-20:] if logs else [],
        }
    except Exception as e:
        return {"error": str(e), "traceback": str(e.__traceback__)}


@router.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time job updates"""
    # Verify job exists
    job = await job_manager.get_job(job_id)
    if not job:
        await websocket.close(code=4004, reason="Job not found")
        return
    
    await ws_manager.connect(websocket, job_id)
    
    try:
        # Send initial state
        logs = await job_manager.get_logs(job_id, last_n=50)
        await websocket.send_json({
            "type": "init",
            "job": job.model_dump(mode="json"),
            "logs": logs,
        })
        
        # Keep connection alive
        while True:
            try:
                # Wait for messages from client (ping/pong or commands)
                data = await websocket.receive_text()
                
                if data == "ping":
                    await websocket.send_text("pong")
                elif data == "status":
                    current_job = await job_manager.get_job(job_id)
                    if current_job:
                        await websocket.send_json({
                            "type": "status",
                            "job": current_job.model_dump(mode="json"),
                        })
                elif data == "logs":
                    logs = await job_manager.get_logs(job_id, last_n=100)
                    await websocket.send_json({
                        "type": "logs",
                        "logs": logs,
                    })
            except WebSocketDisconnect:
                break
    
    finally:
        await ws_manager.disconnect(websocket, job_id)
