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
from ...core.database import get_db
from ...core.config import settings

router = APIRouter()


class JobNameUpdate(BaseModel):
    """Request model for updating job name"""
    name: str


@router.post("/create", response_model=JobInfo)
async def create_job(config: TrainingConfig):
    """Create a new training job"""
    try:
        model_source = config.model_source.value if hasattr(config.model_source, 'value') else str(config.model_source)
        modality = config.modality.value if hasattr(config.modality, 'value') else str(config.modality)
        
        # Validate model path against system capabilities
        is_supported, message = settings.validate_model_path(config.model_path, model_source)
        if not is_supported:
            raise HTTPException(status_code=403, detail=message)
        
        # Validate modality against system capabilities
        is_supported, message = settings.validate_modality(modality)
        if not is_supported:
            raise HTTPException(status_code=403, detail=message)
        
        job = await job_manager.create_job(config)
        return job
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to create training job")


@router.post("/{job_id}/start")
async def start_job(job_id: str):
    """Start a training job"""
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status == JobStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Job is already running")
    
    if job.status == JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job has already completed")
    
    success = await training_service.start_training(job_id)
    if not success:
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
