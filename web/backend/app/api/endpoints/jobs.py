# Copyright (c) US Inc. All rights reserved.
"""Job management endpoints"""

import asyncio
import json
import math
import os
import shutil
import traceback
from datetime import datetime, timedelta
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


@router.post("/preflight", response_model=None)
async def preflight_validation(config: TrainingConfig):
    """Comprehensive pre-flight validation before training starts.
    
    Checks EVERYTHING that could cause a runtime failure:
    1. System expiration / subscription status
    2. GPU availability and VRAM
    3. Model path exists and is valid (config.json, model files)
    4. Dataset paths exist and format is valid
    5. Dataset type compatible with training method
    6. Feature flags (training method enabled in this build)
    7. Config compatibility (DeepSpeed+FSDP, Flash Attention, etc.)
    8. Storage space for output
    9. Core dependencies (torch, transformers, peft, trl, etc.)
    10. No other training already running
    
    Returns detailed validation results so the frontend can show
    exactly what's wrong before the user even clicks Start.
    """
    checks = []
    all_passed = True
    
    def add_check(name: str, passed: bool, message: str, severity: str = "error"):
        nonlocal all_passed
        if not passed and severity == "error":
            all_passed = False
        checks.append({
            "name": name,
            "passed": passed,
            "message": message,
            "severity": severity  # "error", "warning", "info"
        })
    
    # ── 1. System expiration ──────────────────────────────────────
    try:
        expired, exp_msg = is_system_expired()
        if expired:
            add_check("system_status", False, exp_msg)
        else:
            add_check("system_status", True, "System is active and operational")
    except Exception as e:
        add_check("system_status", False, f"System check failed: {str(e)}")
    
    # ── 2. GPU availability ───────────────────────────────────────
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_properties(0).name
            gpu_mem = torch.cuda.get_device_properties(0).total_mem / (1024**3)
            gpu_count = torch.cuda.device_count()
            add_check("gpu", True, f"{gpu_name} ({gpu_mem:.1f}GB){f' x{gpu_count}' if gpu_count > 1 else ''}")
        else:
            add_check("gpu", False, "No CUDA GPU detected. Training requires an NVIDIA GPU.")
    except ImportError:
        add_check("gpu", False, "PyTorch not installed. Cannot check GPU.")
    except Exception as e:
        add_check("gpu", False, f"GPU check failed: {str(e)}")
    
    # ── 3. No other training running ──────────────────────────────
    try:
        from ...services.training_status_service import training_status_service
        can_create, reason = await training_status_service.can_create_job()
        if can_create:
            add_check("no_active_training", True, "No other training in progress")
        else:
            add_check("no_active_training", False, reason)
    except Exception as e:
        add_check("no_active_training", True, "Could not check active training status", "warning")
    
    # ── 4. Model path validation ──────────────────────────────────
    model_source = config.model_source.value if hasattr(config.model_source, 'value') else str(config.model_source)
    model_path = config.model_path
    
    if not model_path:
        add_check("model_path", False, "No model path specified")
    elif model_source == 'local':
        if not os.path.exists(model_path):
            add_check("model_path", False, f"Model path does not exist: {model_path}")
        elif not os.path.isdir(model_path):
            add_check("model_path", False, f"Model path is not a directory: {model_path}")
        else:
            config_json = os.path.join(model_path, "config.json")
            has_config = os.path.exists(config_json)
            # Check for model weight files
            model_files = [f for f in os.listdir(model_path) 
                          if f.endswith(('.safetensors', '.bin', '.pt', '.pth', '.gguf'))]
            if has_config and model_files:
                add_check("model_path", True, f"Model directory valid ({len(model_files)} weight files)")
            elif has_config and not model_files:
                add_check("model_path", False, "Model directory has config.json but no weight files (.safetensors, .bin)")
            elif not has_config and model_files:
                add_check("model_path", True, f"Model directory has {len(model_files)} weight files (no config.json)", "warning")
            else:
                add_check("model_path", False, "Model directory is empty or has no recognizable model files")
    else:
        # Remote model (HuggingFace/ModelScope) - validate format
        if '/' in model_path or model_path.count('/') >= 1:
            add_check("model_path", True, f"Remote model ID: {model_path}")
        else:
            add_check("model_path", True, f"Model: {model_path}", "warning")
    
    # ── 5. Model compatibility (architecture, source restrictions) ─
    try:
        validator = get_validator()
        is_supported, msg = validator.validate_model_path(model_path, model_source)
        if is_supported:
            add_check("model_compatibility", True, "Model is compatible with this system")
        else:
            add_check("model_compatibility", False, msg)
    except Exception as e:
        add_check("model_compatibility", False, f"Model compatibility check failed: {str(e)}")
    
    # ── 6. Dataset validation ─────────────────────────────────────
    dataset_path_str = config.dataset_path if hasattr(config, 'dataset_path') and config.dataset_path else ''
    dataset_paths = [p.strip() for p in dataset_path_str.split(',') if p.strip()] if dataset_path_str else []
    
    if not dataset_paths:
        add_check("dataset", False, "No dataset specified")
    else:
        all_datasets_valid = True
        dataset_messages = []
        for ds_path in dataset_paths:
            if ds_path.upper().startswith(('HF::', 'MS::')):
                dataset_messages.append(f"Remote dataset: {ds_path}")
                continue
            if not os.path.exists(ds_path):
                all_datasets_valid = False
                dataset_messages.append(f"NOT FOUND: {ds_path}")
            else:
                # Check file is readable and has content
                try:
                    file_size = os.path.getsize(ds_path)
                    if file_size == 0:
                        all_datasets_valid = False
                        dataset_messages.append(f"EMPTY FILE: {os.path.basename(ds_path)}")
                    else:
                        size_str = f"{file_size/1024:.1f}KB" if file_size < 1024*1024 else f"{file_size/(1024*1024):.1f}MB"
                        dataset_messages.append(f"{os.path.basename(ds_path)} ({size_str})")
                except Exception:
                    dataset_messages.append(f"{os.path.basename(ds_path)}")
        
        if all_datasets_valid:
            add_check("dataset", True, f"{len(dataset_paths)} dataset(s): {'; '.join(dataset_messages)}")
        else:
            add_check("dataset", False, f"Dataset issues: {'; '.join(dataset_messages)}")
    
    # ── 7. Dataset format validation ──────────────────────────────
    if dataset_paths:
        try:
            from ...services.dataset_type_service import dataset_type_service
            from pathlib import Path
            
            format_issues = []
            for ds_path in dataset_paths:
                ds_path = ds_path.strip()
                if ds_path.upper().startswith(('HF::', 'MS::')):
                    continue
                if not os.path.exists(ds_path):
                    continue
                try:
                    type_info = dataset_type_service.detect_dataset_type(ds_path)
                    if type_info.dataset_type.value == 'unknown':
                        format_issues.append(f"{Path(ds_path).name}: unknown format")
                except Exception as e:
                    format_issues.append(f"{Path(ds_path).name}: detection failed ({str(e)[:50]})")
            
            if format_issues:
                add_check("dataset_format", False, f"Format issues: {'; '.join(format_issues)}")
            else:
                add_check("dataset_format", True, "All dataset formats recognized")
        except Exception as e:
            add_check("dataset_format", True, "Dataset format check skipped", "warning")
    
    # ── 8. Dataset-training method compatibility ──────────────────
    dataset_type = getattr(config, 'dataset_type', None)
    compatible_methods = getattr(config, 'compatible_training_methods', None)
    training_method = config.training_method.value if hasattr(config.training_method, 'value') else str(config.training_method) if config.training_method else 'sft'
    
    if dataset_type and dataset_type != 'unknown' and compatible_methods:
        if training_method in compatible_methods:
            add_check("method_compatibility", True, f"Dataset type '{dataset_type}' is compatible with {training_method.upper()}")
        else:
            method_names = {'sft': 'Supervised Fine-Tuning (SFT)', 'rlhf': 'RLHF', 'pt': 'Pre-Training'}
            add_check("method_compatibility", False, 
                f"Dataset type '{dataset_type}' is not compatible with {method_names.get(training_method, training_method.upper())}. "
                f"Compatible: {', '.join(m.upper() for m in compatible_methods)}")
    else:
        add_check("method_compatibility", True, "Training method compatibility OK", "info")
    
    # ── 9. Feature flags (training method enabled) ────────────────
    try:
        from ...services.training_service import training_service as ts
        is_valid, error = ts._validate_feature_flags(config)
        if is_valid:
            add_check("feature_flags", True, "Training method is enabled in this build")
        else:
            add_check("feature_flags", False, error)
    except Exception as e:
        add_check("feature_flags", True, "Feature flag check skipped", "warning")
    
    # ── 10. Config compatibility ──────────────────────────────────
    config_issues = []
    
    # DeepSpeed + FSDP conflict
    if getattr(config, 'deepspeed', None) and getattr(config, 'fsdp', None):
        config_issues.append("DeepSpeed and FSDP cannot be used together")
    
    # Packing requires Flash Attention
    if getattr(config, 'packing', False):
        attn_impl = getattr(config, 'attn_impl', None)
        flash_attn_values = ['flash_attn', 'flash_attention_2', 'flash_attention_3']
        if attn_impl and attn_impl not in flash_attn_values:
            config_issues.append(f"Sequence Packing requires Flash Attention but '{attn_impl}' is selected")
    
    # Flash Attention 3 requires Hopper GPU
    attn_impl = getattr(config, 'attn_impl', None)
    if attn_impl == 'flash_attention_3':
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                if props.major < 9:
                    config_issues.append(f"Flash Attention 3 requires Hopper GPU (H100/H200) but detected {props.name}")
        except Exception:
            pass
    
    if config_issues:
        add_check("config_compatibility", False, "; ".join(config_issues))
    else:
        add_check("config_compatibility", True, "Configuration is valid")
    
    # ── 11. Core dependencies ─────────────────────────────────────
    dep_issues = []
    for pkg_name in ['torch', 'transformers', 'peft', 'trl', 'datasets', 'accelerate']:
        try:
            mod = __import__(pkg_name)
            ver = getattr(mod, '__version__', 'unknown')
        except ImportError:
            dep_issues.append(f"{pkg_name} not installed")
    
    if dep_issues:
        add_check("dependencies", False, f"Missing: {', '.join(dep_issues)}")
    else:
        add_check("dependencies", True, "All core ML dependencies installed")
    
    # ── 12. Storage space ─────────────────────────────────────────
    try:
        output_dir = getattr(config, 'output_dir', '') or '/workspace'
        # Check the parent directory that exists
        check_path = output_dir
        while check_path and not os.path.exists(check_path):
            check_path = os.path.dirname(check_path)
        if check_path:
            stat = shutil.disk_usage(check_path)
            free_gb = stat.free / (1024**3)
            if free_gb < 5:
                add_check("storage", False, f"Only {free_gb:.1f}GB free disk space. Training may need more.")
            elif free_gb < 20:
                add_check("storage", True, f"{free_gb:.1f}GB free disk space (may be tight for large models)", "warning")
            else:
                add_check("storage", True, f"{free_gb:.1f}GB free disk space")
        else:
            add_check("storage", True, "Could not check storage", "warning")
    except Exception as e:
        add_check("storage", True, f"Storage check skipped: {str(e)[:50]}", "warning")
    
    return {
        "success": all_passed,
        "checks": checks,
        "total_checks": len(checks),
        "passed": sum(1 for c in checks if c["passed"]),
        "failed": sum(1 for c in checks if not c["passed"] and c["severity"] == "error"),
        "warnings": sum(1 for c in checks if not c["passed"] and c["severity"] == "warning"),
    }


@router.post("/create", response_model=JobInfo)
async def create_job(config: TrainingConfig, db: Session = Depends(get_db)):
    """Create a new training job
    
    BLOCKED if another training is already in progress.
    
    IMPORTANT: Saves to both in-memory job_manager AND database for persistence.
    """
    _debug_log(f"create_job called with config: {config}")
    try:
        # CRITICAL: Check if training is already active - block new job creation
        from ...services.training_status_service import training_status_service
        can_create, reason = await training_status_service.can_create_job()
        if not can_create:
            _debug_log(f"Job creation blocked: {reason}", level="WARNING")
            raise HTTPException(
                status_code=409,  # Conflict
                detail=reason
            )
        
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
            
            # ============================================================
            # STRICT VALIDATION: Check dataset types - reject mixed/unknown
            # This prevents runtime failures by catching issues early
            # ============================================================
            from ...services.dataset_type_service import dataset_type_service
            from pathlib import Path
            
            detected_types = []
            for ds_path in dataset_paths:
                ds_path = ds_path.strip()
                # Skip HuggingFace/ModelScope paths for now (can't detect locally)
                if ds_path.upper().startswith('HF::') or ds_path.upper().startswith('MS::'):
                    continue
                
                # Detect dataset type
                if Path(ds_path).exists():
                    try:
                        type_info = dataset_type_service.detect_dataset_type(ds_path)
                        detected_types.append({
                            'path': ds_path,
                            'name': Path(ds_path).name,
                            'type': type_info.dataset_type.value
                        })
                    except Exception as e:
                        _debug_log(f"Dataset type detection failed for {ds_path}: {e}", level="WARNING")
                        detected_types.append({'path': ds_path, 'name': Path(ds_path).name, 'type': 'unknown'})
            
            # STRICT: Reject unknown type datasets
            unknown_datasets = [d for d in detected_types if d['type'] == 'unknown']
            if unknown_datasets:
                unknown_names = ', '.join([d['name'] for d in unknown_datasets])
                error_msg = f"Cannot create job: Dataset(s) with unknown format detected: {unknown_names}. Please ensure all datasets follow a supported format (SFT, RLHF, Pre-training, or KTO)."
                _debug_log(error_msg, level="ERROR")
                raise HTTPException(status_code=400, detail=error_msg)
            
            # STRICT: Reject mixed dataset types
            if len(detected_types) > 1:
                unique_types = set(d['type'] for d in detected_types)
                if len(unique_types) > 1:
                    type_list = ', '.join([f"{d['name']}: {d['type']}" for d in detected_types])
                    error_msg = f"Cannot create job: Mixed dataset types detected: {type_list}. All datasets must be the same type. Please select datasets of only one type."
                    _debug_log(error_msg, level="ERROR")
                    raise HTTPException(status_code=400, detail=error_msg)
        
        # ============================================================
        # VALIDATE: Dataset type must be compatible with training method
        # This prevents mismatches like using SFT dataset for DPO training
        # ============================================================
        dataset_type = getattr(config, 'dataset_type', None)
        compatible_methods = getattr(config, 'compatible_training_methods', None)
        compatible_rlhf = getattr(config, 'compatible_rlhf_types', None)
        training_method = config.training_method.value if hasattr(config.training_method, 'value') else str(config.training_method) if config.training_method else 'sft'
        rlhf_type = getattr(config, 'rlhf_type', None)
        
        if dataset_type and dataset_type != 'unknown' and compatible_methods:
            # Check if training_method is compatible with dataset type
            if training_method not in compatible_methods:
                dataset_display = getattr(config, 'dataset_type_display', dataset_type) or dataset_type
                method_names = {'sft': 'Supervised Fine-Tuning (SFT)', 'rlhf': 'RLHF', 'pt': 'Pre-Training'}
                method_display = method_names.get(training_method, training_method.upper())
                _debug_log(f"Training method mismatch: {training_method} not in {compatible_methods}", level="ERROR")
                raise HTTPException(
                    status_code=400, 
                    detail=f"Dataset type '{dataset_display}' is not compatible with {method_display}. "
                           f"Compatible methods: {', '.join(compatible_methods).upper()}. "
                           f"Please select a compatible training method or use a different dataset."
                )
            
            # For RLHF, also validate rlhf_type is compatible
            if training_method == 'rlhf' and rlhf_type and compatible_rlhf:
                if rlhf_type not in compatible_rlhf:
                    dataset_display = getattr(config, 'dataset_type_display', dataset_type) or dataset_type
                    _debug_log(f"RLHF type mismatch: {rlhf_type} not in {compatible_rlhf}", level="ERROR")
                    raise HTTPException(
                        status_code=400,
                        detail=f"RLHF algorithm '{rlhf_type.upper()}' is not compatible with dataset type '{dataset_display}'. "
                               f"Compatible algorithms: {', '.join(compatible_rlhf).upper()}. "
                               f"Please select a compatible RLHF algorithm."
                    )
        
        # Architecture validation happens when model is loaded
        _debug_log("Creating job in job_manager...")
        job = await job_manager.create_job(config)
        _debug_log(f"Job created in memory: {job.job_id}")
        
        # Write initial log entry so frontend has something to show immediately
        from ...services.sanitized_log_service import sanitized_log_service
        sanitized_log_service.create_terminal_log(job.job_id, "Job created - ready to start training", "INFO")
        
        # ============================================================
        # PERSIST TO DATABASE for training history
        # This ensures history survives server restarts
        # ============================================================
        try:
            from ...models.db_models import TrainingJob as DBTrainingJob
            
            # Convert config to dict for JSON storage
            # Include rlhf_type for proper metric selection in RLHF training
            rlhf_type_val = None
            if hasattr(config, 'rlhf_type') and config.rlhf_type:
                rlhf_type_val = config.rlhf_type.value if hasattr(config.rlhf_type, 'value') else str(config.rlhf_type)
            
            config_dict = {
                "train_type": config.train_type.value if hasattr(config.train_type, 'value') else str(config.train_type),
                "training_method": config.training_method.value if hasattr(config.training_method, 'value') else 'sft',
                "rlhf_type": rlhf_type_val,
                "model_path": config.model_path,
                "dataset_path": config.dataset_path,
                "num_train_epochs": config.num_train_epochs,
                "learning_rate": config.learning_rate,
                "per_device_train_batch_size": config.per_device_train_batch_size,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "max_length": config.max_length,
                "lora_rank": config.lora_rank,
                "lora_alpha": config.lora_alpha,
                "output_dir": config.output_dir,
            }
            
            db_job = DBTrainingJob(
                id=job.job_id,
                name=job.name,
                model_source=model_source,
                model_path=config.model_path,
                model_name=config.model_path.split('/')[-1],
                training_config=config_dict,
                status="pending",
                created_at=job.created_at,
            )
            db.add(db_job)
            db.commit()
            _debug_log(f"Job persisted to database: {job.job_id}")
        except Exception as db_err:
            _debug_log(f"Failed to persist job to database: {db_err}", level="WARNING")
            # Don't fail the request - in-memory job is still valid
        
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
    
    # CRITICAL: Check if ANY training is already running (prevents parallel training)
    jobs = await job_manager.get_all_jobs()
    for existing_job in jobs:
        if existing_job.job_id != job_id and existing_job.status in [JobStatus.RUNNING, JobStatus.INITIALIZING]:
            _debug_log(f"Cannot start {job_id}: another training ({existing_job.job_id}) is already running", job_id, "WARNING")
            raise HTTPException(
                status_code=409,  # Conflict
                detail=f"Another training is already in progress ('{existing_job.name}'). Please wait for it to complete or stop it first."
            )
    
    # Also check at OS level (fallback in case in-memory state is lost)
    if job_manager.is_training_process_running():
        _debug_log(f"Cannot start {job_id}: training process detected at OS level", job_id, "WARNING")
        raise HTTPException(
            status_code=409,
            detail="A training process is already running on the system. Please wait for it to complete or stop it first."
        )
    
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
    
    # Write to terminal log immediately so frontend sees activity
    from ...services.sanitized_log_service import sanitized_log_service
    sanitized_log_service.create_terminal_log(job_id, "Received start command - initializing...", "INFO")
    
    _debug_log(f"Calling training_service.start_training for {job_id}", job_id)
    success = await training_service.start_training(job_id)
    _debug_log(f"start_training returned: {success}", job_id)
    
    if not success:
        _debug_log(f"Failed to start training for {job_id}", job_id, "ERROR")
        sanitized_log_service.create_terminal_log(job_id, "ERROR: Failed to start training - check configuration", "ERROR")
        raise HTTPException(status_code=500, detail="Failed to start training")
    
    return {"job_id": job_id, "status": "starting"}


@router.post("/{job_id}/stop")
async def stop_job(job_id: str):
    """Stop a running job"""
    job = await job_manager.get_job(job_id)
    
    # Handle case where job state is lost but we have a PID
    if not job and job_id.startswith("pid-"):
        try:
            pid = int(job_id.replace("pid-", ""))
            success = job_manager.stop_training_process_by_pid(pid)
            if success:
                return {"job_id": job_id, "status": "stopped", "message": "Process terminated by PID"}
            else:
                raise HTTPException(status_code=500, detail="Failed to stop process")
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid PID format")
    
    if not job:
        # Fallback: Try to stop any running training process
        if job_manager.is_training_process_running():
            success = job_manager.stop_all_training_processes()
            if success:
                return {"job_id": job_id, "status": "stopped", "message": "All training processes stopped"}
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != JobStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Job is not running")
    
    success = await training_service.stop_training(job_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to stop training")
    
    return {"job_id": job_id, "status": "stopped"}


@router.post("/force-reset")
async def force_reset_stuck_training():
    """Force reset a stuck training job.
    
    This endpoint is used when training appears stuck (no progress for extended time).
    It will:
    1. Kill ALL training processes (SFT, DPO, PPO, GRPO, RLHF, etc.)
    2. Clear GPU VRAM and CPU memory
    3. Mark the current job as failed
    4. Clear training state to allow new training
    
    Returns:
        - success: True if reset completed
        - message: Human-readable status
        - job_id: The job that was reset
        - gpu_cleanup: Memory freed info (if available)
        - warnings: Non-fatal issues encountered (if any)
    """
    from ...services.training_status_service import training_status_service
    
    try:
        result = await training_status_service.force_reset_stuck_training()
        
        if not result["success"]:
            raise HTTPException(status_code=400, detail=result["message"])
        
        return result
    except HTTPException:
        raise
    except Exception as e:
        _debug_log(f"Force reset failed with exception: {e}", level="ERROR")
        raise HTTPException(
            status_code=500, 
            detail=f"Force reset failed: {str(e)}. Try stopping training manually or restart the server."
        )


@router.get("/current")
async def get_current_job():
    """Get the currently active/running training job if any.
    
    This endpoint allows the frontend to restore training state after page refresh.
    Returns the active job or null if no training is running.
    
    It checks both:
    1. In-memory job state (for normal operation)
    2. OS process check (fallback if in-memory state is lost)
    3. Recently failed jobs (within last 2 minutes) - so user sees validation errors
    """
    from ...services.sanitized_log_service import sanitized_log_service
    
    try:
        jobs = await job_manager.get_all_jobs()
        
        # Find running job first in memory
        for job in jobs:
            if job.status == JobStatus.RUNNING:
                logs = await job_manager.get_logs(job.job_id, last_n=100)
                # Fall back to file-based logs if in-memory is empty
                if not logs:
                    try:
                        logs = sanitized_log_service.get_terminal_logs(job.job_id, lines=100)
                    except Exception:
                        logs = []
                return {
                    "has_active_job": True,
                    "job": job.model_dump(mode="json"),
                    "logs": logs,
                    "process_running": True,
                }
        
        # Check for initializing jobs
        for job in jobs:
            if job.status == JobStatus.INITIALIZING:
                logs = await job_manager.get_logs(job.job_id, last_n=100)
                # Fall back to file-based logs if in-memory is empty
                if not logs:
                    try:
                        logs = sanitized_log_service.get_terminal_logs(job.job_id, lines=100)
                    except Exception:
                        logs = []
                return {
                    "has_active_job": True,
                    "job": job.model_dump(mode="json"),
                    "logs": logs,
                    "process_running": True,
                }
        
        # Check for recently failed/completed/stopped jobs (within last 2 minutes)
        # This ensures users see validation errors even after quick failures
        recent_cutoff = datetime.now() - timedelta(minutes=2)
        recent_terminal_jobs = []
        for job in jobs:
            if job.status in [JobStatus.FAILED, JobStatus.COMPLETED, JobStatus.STOPPED]:
                # Check if completed_at is recent
                if job.completed_at and job.completed_at > recent_cutoff:
                    recent_terminal_jobs.append(job)
                # Fallback: check started_at if completed_at not set
                elif job.started_at and job.started_at > recent_cutoff:
                    recent_terminal_jobs.append(job)
        
        # Return the most recent terminal job if any
        if recent_terminal_jobs:
            # Sort by completed_at or started_at, most recent first
            recent_terminal_jobs.sort(
                key=lambda j: j.completed_at or j.started_at or datetime.min, 
                reverse=True
            )
            job = recent_terminal_jobs[0]
            logs = await job_manager.get_logs(job.job_id, last_n=100)
            if not logs:
                try:
                    logs = sanitized_log_service.get_terminal_logs(job.job_id, lines=100)
                except Exception:
                    logs = []
            return {
                "has_active_job": True,  # True so frontend shows the result
                "job": job.model_dump(mode="json"),
                "logs": logs,
                "process_running": False,
                "is_recent_terminal": True,  # Indicate this is a recently finished job
            }
        
        # Fallback: Check if training process is running at OS level
        # This handles cases where in-memory state was lost but process continues
        if job_manager.is_training_process_running():
            pid = job_manager.get_running_training_pid()
            return {
                "has_active_job": True,
                "job": None,  # No job info available
                "logs": [],
                "process_running": True,
                "process_pid": pid,
                "message": "Training process detected but job state was lost. Training continues in background."
            }
        
        return {
            "has_active_job": False,
            "job": None,
            "logs": [],
            "process_running": False,
        }
    except Exception as e:
        _debug_log(f"Error getting current job: {e}", level="ERROR")
        return {
            "has_active_job": False,
            "job": None,
            "logs": [],
            "process_running": False,
            "error": str(e)
        }


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


@router.get("/history/all")
async def get_training_history(
    limit: int = Query(50, ge=1, le=200),
    include_metrics: bool = Query(True),
    db: Session = Depends(get_db)
):
    """Get training history with all past trainings.
    
    Returns a list of all training jobs with their final metrics,
    output paths, and status. Ordered by creation date (newest first).
    
    IMPORTANT: Uses DATABASE for persistence, not in-memory job_manager.
    This ensures training history survives server restarts.
    """
    from pathlib import Path
    from ...core.capabilities import get_system_settings
    from ...models.db_models import TrainingJob as DBTrainingJob
    
    try:
        service = JobService(db)
        
        # ============================================================
        # QUERY DATABASE for persistent training history
        # This ensures history survives server restarts
        # ============================================================
        db_jobs = db.query(DBTrainingJob).order_by(
            DBTrainingJob.created_at.desc()
        ).limit(limit).all()
        
        # Also get in-memory jobs for currently running trainings
        memory_jobs = await job_manager.get_all_jobs()
        memory_job_ids = {j.job_id for j in memory_jobs}
        
        history = []
        
        # Process database jobs first (persistent history)
        for db_job in db_jobs:
            job_data = {
                "job_id": db_job.id,
                "job_name": db_job.name,
                "status": db_job.status,
                "created_at": db_job.created_at.isoformat() if db_job.created_at else None,
                "started_at": db_job.started_at.isoformat() if db_job.started_at else None,
                "completed_at": db_job.completed_at.isoformat() if db_job.completed_at else None,
                "current_step": None,
                "total_steps": None,
                "error": db_job.error_message,
            }
            
            # Add training config info from JSON
            if db_job.training_config:
                config = db_job.training_config
                job_data["config"] = {
                    "train_type": config.get("train_type", "lora"),
                    "training_method": config.get("training_method", "sft"),
                    "num_epochs": config.get("num_train_epochs", 1),
                    "learning_rate": config.get("learning_rate", 5e-5),
                    "batch_size": config.get("per_device_train_batch_size", 1),
                }
            
            # Add output path info
            output_dir = Path(db_job.output_dir) if db_job.output_dir else get_system_settings().OUTPUT_DIR / db_job.id
            job_data["output_path"] = str(output_dir)
            job_data["output_exists"] = output_dir.exists() if output_dir else False
            
            # Check for adapter/checkpoint/full-model files (recursive for nested v0-*/checkpoint-*)
            if output_dir and output_dir.exists():
                # LoRA/QLoRA/AdaLoRA adapters
                adapters = list(output_dir.glob("**/adapter_model.safetensors")) + list(output_dir.glob("**/adapter_model.bin"))
                job_data["has_adapter"] = len(adapters) > 0
                if adapters:
                    # Pick the best adapter: prefer non-checkpoint > highest checkpoint step
                    non_ckpt = [a for a in adapters if "checkpoint-" not in str(a)]
                    job_data["adapter_path"] = str(non_ckpt[0].parent) if non_ckpt else str(adapters[-1].parent)
                
                # Full fine-tuning model files
                full_models = list(output_dir.glob("**/model.safetensors")) + list(output_dir.glob("**/model.safetensors.index.json"))
                non_ckpt_models = [m for m in full_models if "checkpoint-" not in str(m)]
                ckpt_models = [m for m in full_models if "checkpoint-" in str(m)]
                job_data["has_full_model"] = len(full_models) > 0
                if non_ckpt_models:
                    job_data["model_path"] = str(non_ckpt_models[0].parent)
                elif ckpt_models:
                    job_data["model_path"] = str(ckpt_models[-1].parent)
                
                # Recursive checkpoint count (handles v0-*/checkpoint-* nesting)
                checkpoint_dirs = list(output_dir.glob("**/checkpoint-*"))
                checkpoints = [d for d in checkpoint_dirs if d.is_dir()]
                job_data["checkpoint_count"] = len(checkpoints)
            else:
                job_data["has_adapter"] = False
                job_data["has_full_model"] = False
                job_data["checkpoint_count"] = 0
            
            # Add final metrics if requested
            if include_metrics:
                try:
                    metrics = service.get_job_metrics(db_job.id, limit=1)
                    if metrics:
                        last_metric = metrics[-1]
                        job_data["final_metrics"] = {
                            "loss": last_metric.loss,
                            "learning_rate": last_metric.learning_rate,
                            "epoch": last_metric.epoch,
                            "step": last_metric.step,
                        }
                    else:
                        job_data["final_metrics"] = None
                except Exception:
                    job_data["final_metrics"] = None
            
            history.append(job_data)
        
        # Add in-memory jobs that aren't in DB yet (currently running)
        db_job_ids = {j.id for j in db_jobs}
        for mem_job in memory_jobs:
            if mem_job.job_id not in db_job_ids:
                job_data = {
                    "job_id": mem_job.job_id,
                    "job_name": mem_job.name,
                    "status": mem_job.status.value if hasattr(mem_job.status, 'value') else str(mem_job.status),
                    "created_at": mem_job.created_at.isoformat() if mem_job.created_at else None,
                    "started_at": mem_job.started_at.isoformat() if mem_job.started_at else None,
                    "completed_at": mem_job.completed_at.isoformat() if mem_job.completed_at else None,
                    "current_step": mem_job.current_step,
                    "total_steps": mem_job.total_steps,
                    "error": mem_job.error,
                    "config": None,
                    "output_path": str(get_system_settings().OUTPUT_DIR / mem_job.job_id),
                    "output_exists": False,
                    "has_adapter": False,
                    "checkpoint_count": 0,
                    "final_metrics": None,
                }
                
                if mem_job.config:
                    train_type = mem_job.config.train_type.value if hasattr(mem_job.config.train_type, 'value') else str(mem_job.config.train_type)
                    training_method = getattr(mem_job.config, 'training_method', None)
                    method_value = training_method.value if hasattr(training_method, 'value') else 'sft'
                    job_data["config"] = {
                        "train_type": train_type,
                        "training_method": method_value,
                        "num_epochs": mem_job.config.num_train_epochs,
                        "learning_rate": mem_job.config.learning_rate,
                        "batch_size": mem_job.config.per_device_train_batch_size,
                    }
                
                history.insert(0, job_data)  # Currently running jobs at top
        
        return {
            "count": len(history),
            "history": history
        }
    except Exception as e:
        return {
            "count": 0,
            "history": [],
            "error": str(e),
            "traceback": traceback.format_exc()
        }


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


@router.get("/{job_id}/terminal-logs")
async def get_terminal_logs(job_id: str, lines: int = 100):
    """Get terminal logs from file for a job.
    
    This endpoint retrieves logs from the persistent terminal log file,
    which is useful when in-memory logs are lost (e.g., after page refresh).
    """
    from ...services.sanitized_log_service import sanitized_log_service
    
    try:
        log_path = sanitized_log_service.get_terminal_log_path(job_id)
        file_exists = os.path.exists(log_path)
        file_size = os.path.getsize(log_path) if file_exists else 0
        
        logs = sanitized_log_service.get_terminal_logs(job_id, lines=lines)
        
        # Debug info for troubleshooting
        print(f"[TERMINAL-LOGS] job_id={job_id}, path={log_path}, exists={file_exists}, size={file_size}, lines={len(logs)}")
        
        return {
            "job_id": job_id,
            "logs": logs,
            "count": len(logs),
            "debug": {
                "path": log_path,
                "exists": file_exists,
                "size": file_size
            }
        }
    except Exception as e:
        print(f"[TERMINAL-LOGS] ERROR for job_id={job_id}: {e}")
        traceback.print_exc()
        return {
            "job_id": job_id,
            "logs": [],
            "count": 0,
            "error": str(e)
        }


@router.get("/{job_id}/stream/logs")
async def stream_logs(job_id: str):
    """HTTP Server-Sent Events (SSE) endpoint for real-time log streaming.
    
    This is the HTTP-based alternative to WebSocket for log streaming.
    Clients can use EventSource API to receive real-time log updates.
    
    Example client usage:
        const eventSource = new EventSource('/api/jobs/{job_id}/stream/logs');
        eventSource.onmessage = (event) => {
            const data = JSON.parse(event.data);
            console.log(data.logs);
        };
    """
    from fastapi.responses import StreamingResponse
    from ...services.sanitized_log_service import sanitized_log_service
    
    async def log_generator():
        """Generator that yields log updates as SSE events."""
        last_line_count = 0
        retry_count = 0
        max_retries = 3600  # 1 hour at 1 second intervals
        
        # Send initial connection event
        yield f"event: connected\ndata: {json.dumps({'job_id': job_id, 'status': 'connected'})}\n\n"
        
        while retry_count < max_retries:
            try:
                # Get current logs
                logs = sanitized_log_service.get_terminal_logs(job_id, lines=500)
                current_count = len(logs)
                
                # Check if job still exists
                job = await job_manager.get_job(job_id)
                if not job:
                    yield f"event: error\ndata: {json.dumps({'error': 'Job not found'})}\n\n"
                    break
                
                # Send new logs if any
                if current_count > last_line_count:
                    new_logs = logs[last_line_count:]
                    yield f"data: {json.dumps({'logs': new_logs, 'total': current_count})}\n\n"
                    last_line_count = current_count
                
                # Send job status update
                status = job.status.value if hasattr(job.status, 'value') else str(job.status)
                yield f"event: status\ndata: {json.dumps({'status': status, 'step': job.current_step, 'total': job.total_steps})}\n\n"
                
                # Stop streaming if job is complete
                if status in ['completed', 'failed', 'stopped', 'cancelled']:
                    yield f"event: complete\ndata: {json.dumps({'status': status})}\n\n"
                    break
                
                # Wait before next poll
                await asyncio.sleep(1)
                retry_count += 1
                
            except Exception as e:
                yield f"event: error\ndata: {json.dumps({'error': str(e)})}\n\n"
                await asyncio.sleep(5)
                retry_count += 1
    
    return StreamingResponse(
        log_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@router.get("/{job_id}/checkpoint-info")
async def get_checkpoint_info(job_id: str):
    """Get checkpoint information for a job.
    
    Returns available checkpoints that can be used to resume training.
    """
    from pathlib import Path
    from ...core.capabilities import get_system_settings
    
    try:
        job = await job_manager.get_job(job_id)
        output_dir = get_system_settings().OUTPUT_DIR / job_id
        
        checkpoints = []
        has_checkpoints = False
        latest_checkpoint = None
        latest_step = 0
        
        if output_dir.exists():
            # Find checkpoint directories (checkpoint-XXX format)
            for item in output_dir.iterdir():
                if item.is_dir() and item.name.startswith("checkpoint-"):
                    try:
                        step = int(item.name.replace("checkpoint-", ""))
                        checkpoints.append({
                            "path": str(item),
                            "step": step,
                            "name": item.name,
                        })
                        has_checkpoints = True
                        if step > latest_step:
                            latest_step = step
                            latest_checkpoint = str(item)
                    except ValueError:
                        continue
        
        # Sort checkpoints by step
        checkpoints.sort(key=lambda x: x["step"], reverse=True)
        
        return {
            "job_id": job_id,
            "output_dir": str(output_dir),
            "has_checkpoints": has_checkpoints,
            "checkpoints": checkpoints,
            "latest_checkpoint": latest_checkpoint,
            "latest_step": latest_step,
            "can_resume": has_checkpoints,
            "can_restart": True,  # Always can restart (will delete old data)
        }
    except Exception as e:
        return {
            "job_id": job_id,
            "error": str(e),
            "has_checkpoints": False,
            "can_resume": False,
            "can_restart": True,
        }


@router.post("/{job_id}/resume")
async def resume_job(job_id: str, checkpoint_path: str = None):
    """Resume a failed/stopped job from a checkpoint.
    
    If checkpoint_path is not provided, uses the latest checkpoint.
    """
    from pathlib import Path
    from ...core.capabilities import get_system_settings
    
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status == JobStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Job is already running")
    
    if job.status == JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="Job already completed. Use restart instead.")
    
    # Find checkpoint to resume from
    output_dir = get_system_settings().OUTPUT_DIR / job_id
    
    if not checkpoint_path:
        # Find latest checkpoint
        latest_step = 0
        if output_dir.exists():
            for item in output_dir.iterdir():
                if item.is_dir() and item.name.startswith("checkpoint-"):
                    try:
                        step = int(item.name.replace("checkpoint-", ""))
                        if step > latest_step:
                            latest_step = step
                            checkpoint_path = str(item)
                    except ValueError:
                        continue
    
    if not checkpoint_path:
        raise HTTPException(status_code=400, detail="No checkpoint found. Use restart instead.")
    
    if not Path(checkpoint_path).exists():
        raise HTTPException(status_code=400, detail=f"Checkpoint not found: {checkpoint_path}")
    
    _debug_log(f"Resuming job {job_id} from checkpoint: {checkpoint_path}", job_id)
    
    # Update job config to resume from checkpoint
    await job_manager.update_job(job_id, 
        status=JobStatus.PENDING,
        error=None,
        resume_from_checkpoint=checkpoint_path
    )
    
    # Start training
    success = await training_service.start_training(job_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to resume training")
    
    return {
        "job_id": job_id,
        "status": "resuming",
        "checkpoint": checkpoint_path,
    }


@router.post("/{job_id}/restart")
async def restart_job(job_id: str, delete_data: bool = True):
    """Restart a job from scratch.
    
    If delete_data is True, deletes the job's output folder (checkpoints, logs)
    but keeps the dataset and model intact.
    """
    from pathlib import Path
    from ...core.capabilities import get_system_settings
    
    job = await job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status == JobStatus.RUNNING:
        raise HTTPException(status_code=400, detail="Stop the job first before restarting")
    
    _debug_log(f"Restarting job {job_id}, delete_data={delete_data}", job_id)
    
    if delete_data:
        # Delete output directory (checkpoints, adapter weights, etc.)
        output_dir = get_system_settings().OUTPUT_DIR / job_id
        if output_dir.exists():
            try:
                shutil.rmtree(output_dir)
                _debug_log(f"Deleted output directory: {output_dir}", job_id)
            except Exception as e:
                _debug_log(f"Failed to delete output directory: {e}", job_id, "ERROR")
        
        # Delete terminal logs for this job
        from ...services.sanitized_log_service import sanitized_log_service
        terminal_log_path = sanitized_log_service.get_terminal_log_path(job_id)
        if Path(terminal_log_path).exists():
            try:
                Path(terminal_log_path).unlink()
                _debug_log(f"Deleted terminal log: {terminal_log_path}", job_id)
            except Exception as e:
                _debug_log(f"Failed to delete terminal log: {e}", job_id, "ERROR")
        
        # Delete encrypted logs for this job
        encrypted_log_path = encrypted_log_service.get_encrypted_log_path(job_id)
        if Path(encrypted_log_path).exists():
            try:
                Path(encrypted_log_path).unlink()
                _debug_log(f"Deleted encrypted log: {encrypted_log_path}", job_id)
            except Exception as e:
                _debug_log(f"Failed to delete encrypted log: {e}", job_id, "ERROR")
        
        # Clear in-memory logs
        await job_manager.clear_logs(job_id)
    
    # Reset job state
    await job_manager.update_job(job_id,
        status=JobStatus.PENDING,
        error=None,
        current_step=0,
        total_steps=0,
        current_loss=None,
        current_epoch=None,
        started_at=None,
        completed_at=None,
        resume_from_checkpoint=None,
    )
    
    # Start training
    success = await training_service.start_training(job_id)
    if not success:
        raise HTTPException(status_code=500, detail="Failed to restart training")
    
    return {
        "job_id": job_id,
        "status": "restarting",
        "data_deleted": delete_data,
    }


@router.get("/{job_id}/metrics")
async def get_job_metrics(job_id: str, limit: int = Query(500, ge=1, le=2000), db: Session = Depends(get_db)):
    """Get training metrics for a job (for graphs).
    
    Returns ONLY relevant metrics for the training type:
    - SFT/Pre-training: loss, learning_rate, grad_norm, eval_loss
    - RLHF/PPO: reward, kl_divergence, policy_loss, value_loss, entropy
    - DPO: chosen_rewards, rejected_rewards, reward_margin
    - GRPO: reward, kl_penalty, policy_gradient_loss
    
    Metrics are stored per job_id - no mixing between different trainings.
    """
    try:
        service = JobService(db)
        metrics = service.get_job_metrics(job_id, limit=limit)
        
        # Get job info to determine training type
        job = await job_manager.get_job(job_id)
        
        # Determine effective training type for metric selection
        # Priority: rlhf_type > training_method > train_type
        # This ensures RLHF methods (DPO, GRPO, PPO, etc.) show their specific metrics
        train_type = "sft"  # Default
        if job and job.config:
            training_method = getattr(job.config, 'training_method', None)
            method_value = training_method.value if hasattr(training_method, 'value') else str(training_method) if training_method else 'sft'
            
            if method_value == 'rlhf':
                # For RLHF, use the specific rlhf_type (dpo, grpo, ppo, etc.)
                rlhf_type = getattr(job.config, 'rlhf_type', None)
                if rlhf_type:
                    train_type = rlhf_type.lower() if isinstance(rlhf_type, str) else rlhf_type.value.lower() if hasattr(rlhf_type, 'value') else str(rlhf_type).lower()
                else:
                    train_type = 'rlhf'  # Generic RLHF if no specific type
            elif method_value == 'pt':
                train_type = 'pt'  # Pre-training
            else:
                # For SFT, use train_type (lora, qlora, full, adalora)
                train_type = job.config.train_type.value if hasattr(job.config.train_type, 'value') else 'lora'
        
        # Define which metrics are relevant for each training type
        METRIC_CONFIG = {
            # ============================================================
            # SUPERVISED FINE-TUNING (SFT)
            # ============================================================
            "sft": {
                "display_name": "Supervised Fine-Tuning",
                "primary_metrics": ["loss", "learning_rate", "grad_norm"],
                "eval_metrics": ["eval_loss", "eval_accuracy"],
                "graph_configs": [
                    {"key": "loss", "label": "Training Loss", "unit": "", "color": "#3B82F6"},
                    {"key": "learning_rate", "label": "Learning Rate", "unit": "", "color": "#10B981"},
                    {"key": "grad_norm", "label": "Gradient Norm", "unit": "", "color": "#F59E0B"},
                ],
                "show_reward": False,
                "show_dpo": False,
            },
            "full": {
                "display_name": "Full Fine-Tuning",
                "primary_metrics": ["loss", "learning_rate", "grad_norm"],
                "eval_metrics": ["eval_loss", "eval_accuracy"],
                "graph_configs": [
                    {"key": "loss", "label": "Training Loss", "unit": "", "color": "#3B82F6"},
                    {"key": "learning_rate", "label": "Learning Rate", "unit": "", "color": "#10B981"},
                    {"key": "grad_norm", "label": "Gradient Norm", "unit": "", "color": "#F59E0B"},
                ],
                "show_reward": False,
                "show_dpo": False,
            },
            "lora": {
                "display_name": "LoRA Fine-Tuning",
                "primary_metrics": ["loss", "learning_rate", "grad_norm"],
                "eval_metrics": ["eval_loss", "eval_accuracy"],
                "graph_configs": [
                    {"key": "loss", "label": "Training Loss", "unit": "", "color": "#3B82F6"},
                    {"key": "learning_rate", "label": "Learning Rate", "unit": "", "color": "#10B981"},
                    {"key": "grad_norm", "label": "Gradient Norm", "unit": "", "color": "#F59E0B"},
                ],
                "show_reward": False,
                "show_dpo": False,
            },
            "qlora": {
                "display_name": "QLoRA Fine-Tuning",
                "primary_metrics": ["loss", "learning_rate", "grad_norm"],
                "eval_metrics": ["eval_loss", "eval_accuracy"],
                "graph_configs": [
                    {"key": "loss", "label": "Training Loss", "unit": "", "color": "#3B82F6"},
                    {"key": "learning_rate", "label": "Learning Rate", "unit": "", "color": "#10B981"},
                    {"key": "grad_norm", "label": "Gradient Norm", "unit": "", "color": "#F59E0B"},
                ],
                "show_reward": False,
                "show_dpo": False,
            },
            "adalora": {
                "display_name": "AdaLoRA Fine-Tuning",
                "primary_metrics": ["loss", "learning_rate", "grad_norm"],
                "eval_metrics": ["eval_loss", "eval_accuracy"],
                "graph_configs": [
                    {"key": "loss", "label": "Training Loss", "unit": "", "color": "#3B82F6"},
                    {"key": "learning_rate", "label": "Learning Rate", "unit": "", "color": "#10B981"},
                    {"key": "grad_norm", "label": "Gradient Norm", "unit": "", "color": "#F59E0B"},
                ],
                "show_reward": False,
                "show_dpo": False,
            },
            # ============================================================
            # PRE-TRAINING
            # ============================================================
            "pretrain": {
                "display_name": "Pre-Training",
                "primary_metrics": ["loss", "learning_rate", "grad_norm", "perplexity"],
                "eval_metrics": ["eval_loss", "eval_perplexity"],
                "graph_configs": [
                    {"key": "loss", "label": "Training Loss", "unit": "", "color": "#3B82F6"},
                    {"key": "perplexity", "label": "Perplexity", "unit": "", "color": "#8B5CF6"},
                    {"key": "learning_rate", "label": "Learning Rate", "unit": "", "color": "#10B981"},
                    {"key": "grad_norm", "label": "Gradient Norm", "unit": "", "color": "#F59E0B"},
                ],
                "show_reward": False,
                "show_dpo": False,
            },
            "pt": {
                "display_name": "Pre-Training",
                "primary_metrics": ["loss", "learning_rate", "grad_norm", "perplexity"],
                "eval_metrics": ["eval_loss", "eval_perplexity"],
                "graph_configs": [
                    {"key": "loss", "label": "Training Loss", "unit": "", "color": "#3B82F6"},
                    {"key": "perplexity", "label": "Perplexity", "unit": "", "color": "#8B5CF6"},
                    {"key": "learning_rate", "label": "Learning Rate", "unit": "", "color": "#10B981"},
                ],
                "show_reward": False,
                "show_dpo": False,
            },
            # ============================================================
            # REINFORCEMENT LEARNING FROM HUMAN FEEDBACK (RLHF)
            # ============================================================
            "rlhf": {
                "display_name": "RLHF",
                "primary_metrics": ["reward", "kl_divergence", "policy_loss", "value_loss", "entropy"],
                "eval_metrics": ["approx_kl", "clip_fraction"],
                "graph_configs": [
                    {"key": "reward", "label": "Reward", "unit": "", "color": "#10B981"},
                    {"key": "kl_divergence", "label": "KL Divergence", "unit": "", "color": "#F59E0B"},
                    {"key": "policy_loss", "label": "Policy Loss", "unit": "", "color": "#3B82F6"},
                    {"key": "value_loss", "label": "Value Loss", "unit": "", "color": "#EF4444"},
                    {"key": "entropy", "label": "Entropy", "unit": "", "color": "#8B5CF6"},
                ],
                "show_reward": True,
                "show_dpo": False,
            },
            "ppo": {
                "display_name": "PPO (Proximal Policy Optimization)",
                "primary_metrics": ["reward", "policy_loss", "value_loss", "entropy", "kl_divergence"],
                "eval_metrics": ["approx_kl", "clip_fraction"],
                "graph_configs": [
                    {"key": "reward", "label": "Reward", "unit": "", "color": "#10B981"},
                    {"key": "policy_loss", "label": "Policy Loss", "unit": "", "color": "#3B82F6"},
                    {"key": "value_loss", "label": "Value Loss", "unit": "", "color": "#EF4444"},
                    {"key": "entropy", "label": "Entropy", "unit": "nats", "color": "#8B5CF6"},
                    {"key": "kl_divergence", "label": "KL Divergence", "unit": "", "color": "#F59E0B"},
                ],
                "show_reward": True,
                "show_dpo": False,
            },
            # ============================================================
            # DIRECT PREFERENCE OPTIMIZATION (DPO) & VARIANTS
            # ============================================================
            "dpo": {
                "display_name": "DPO (Direct Preference Optimization)",
                "primary_metrics": ["loss", "chosen_rewards", "rejected_rewards", "reward_margin"],
                "eval_metrics": ["eval_loss", "accuracy"],
                "graph_configs": [
                    {"key": "loss", "label": "DPO Loss", "unit": "", "color": "#3B82F6"},
                    {"key": "chosen_rewards", "label": "Chosen Rewards", "unit": "", "color": "#10B981"},
                    {"key": "rejected_rewards", "label": "Rejected Rewards", "unit": "", "color": "#EF4444"},
                    {"key": "reward_margin", "label": "Reward Margin", "unit": "", "color": "#F59E0B"},
                ],
                "show_reward": False,
                "show_dpo": True,
            },
            "kto": {
                "display_name": "KTO (Kahneman-Tversky Optimization)",
                "primary_metrics": ["loss", "chosen_rewards", "rejected_rewards", "kl_divergence"],
                "eval_metrics": ["eval_loss"],
                "graph_configs": [
                    {"key": "loss", "label": "KTO Loss", "unit": "", "color": "#3B82F6"},
                    {"key": "chosen_rewards", "label": "Desirable Rewards", "unit": "", "color": "#10B981"},
                    {"key": "rejected_rewards", "label": "Undesirable Rewards", "unit": "", "color": "#EF4444"},
                    {"key": "kl_divergence", "label": "KL Divergence", "unit": "", "color": "#F59E0B"},
                ],
                "show_reward": False,
                "show_dpo": True,
            },
            "simpo": {
                "display_name": "SimPO (Simple Preference Optimization)",
                "primary_metrics": ["loss", "chosen_rewards", "rejected_rewards", "reward_margin"],
                "eval_metrics": ["eval_loss"],
                "graph_configs": [
                    {"key": "loss", "label": "SimPO Loss", "unit": "", "color": "#3B82F6"},
                    {"key": "chosen_rewards", "label": "Chosen Rewards", "unit": "", "color": "#10B981"},
                    {"key": "rejected_rewards", "label": "Rejected Rewards", "unit": "", "color": "#EF4444"},
                    {"key": "reward_margin", "label": "Reward Margin", "unit": "", "color": "#F59E0B"},
                ],
                "show_reward": False,
                "show_dpo": True,
            },
            "orpo": {
                "display_name": "ORPO (Odds Ratio Preference Optimization)",
                "primary_metrics": ["loss", "chosen_rewards", "rejected_rewards", "log_odds"],
                "eval_metrics": ["eval_loss", "accuracy"],
                "graph_configs": [
                    {"key": "loss", "label": "ORPO Loss", "unit": "", "color": "#3B82F6"},
                    {"key": "chosen_rewards", "label": "Chosen Log Probs", "unit": "", "color": "#10B981"},
                    {"key": "rejected_rewards", "label": "Rejected Log Probs", "unit": "", "color": "#EF4444"},
                ],
                "show_reward": False,
                "show_dpo": True,
            },
            "rpo": {
                "display_name": "RPO (Relative Preference Optimization)",
                "primary_metrics": ["loss", "chosen_rewards", "rejected_rewards", "reward_margin"],
                "eval_metrics": ["eval_loss"],
                "graph_configs": [
                    {"key": "loss", "label": "RPO Loss", "unit": "", "color": "#3B82F6"},
                    {"key": "chosen_rewards", "label": "Chosen Rewards", "unit": "", "color": "#10B981"},
                    {"key": "rejected_rewards", "label": "Rejected Rewards", "unit": "", "color": "#EF4444"},
                ],
                "show_reward": False,
                "show_dpo": True,
            },
            # ============================================================
            # GROUP RELATIVE POLICY OPTIMIZATION (GRPO)
            # ============================================================
            "grpo": {
                "display_name": "GRPO (Group Relative Policy Optimization)",
                "primary_metrics": ["reward", "loss", "kl_divergence", "policy_loss"],
                "eval_metrics": ["entropy", "approx_kl"],
                "graph_configs": [
                    {"key": "reward", "label": "Group Reward", "unit": "", "color": "#10B981"},
                    {"key": "loss", "label": "GRPO Loss", "unit": "", "color": "#3B82F6"},
                    {"key": "kl_divergence", "label": "KL Penalty", "unit": "", "color": "#F59E0B"},
                    {"key": "policy_loss", "label": "Policy Loss", "unit": "", "color": "#EF4444"},
                ],
                "show_reward": True,
                "show_dpo": False,
            },
            # ============================================================
            # REWARD MODEL TRAINING
            # ============================================================
            "rm": {
                "display_name": "Reward Model Training",
                "primary_metrics": ["loss", "accuracy", "chosen_rewards", "rejected_rewards"],
                "eval_metrics": ["eval_loss", "eval_accuracy"],
                "graph_configs": [
                    {"key": "loss", "label": "RM Loss", "unit": "", "color": "#3B82F6"},
                    {"key": "accuracy", "label": "Accuracy", "unit": "%", "color": "#10B981"},
                    {"key": "chosen_rewards", "label": "Chosen Score", "unit": "", "color": "#8B5CF6"},
                    {"key": "rejected_rewards", "label": "Rejected Score", "unit": "", "color": "#EF4444"},
                ],
                "show_reward": False,
                "show_dpo": True,
            },
        }
        
        config = METRIC_CONFIG.get(train_type, METRIC_CONFIG["sft"])
        
        # Build metrics response with ONLY relevant fields for this training type
        formatted_metrics = []
        for m in metrics:
            metric_data = {
                "step": m.step,
                "epoch": m.epoch,
                "timestamp": m.timestamp.isoformat() if m.timestamp else None,
            }
            
            # Common metrics (always included if present)
            if m.loss is not None:
                metric_data["loss"] = m.loss
            if m.learning_rate is not None:
                metric_data["learning_rate"] = m.learning_rate
            if m.grad_norm is not None:
                metric_data["grad_norm"] = m.grad_norm
            if m.eval_loss is not None:
                metric_data["eval_loss"] = m.eval_loss
            
            # RLHF/PPO metrics (only if relevant)
            if config["show_reward"]:
                if m.reward is not None:
                    metric_data["reward"] = m.reward
                if m.kl_divergence is not None:
                    metric_data["kl_divergence"] = m.kl_divergence
                if m.policy_loss is not None:
                    metric_data["policy_loss"] = m.policy_loss
                if m.value_loss is not None:
                    metric_data["value_loss"] = m.value_loss
                if m.entropy is not None:
                    metric_data["entropy"] = m.entropy
            
            # DPO metrics (only if relevant)
            if config["show_dpo"]:
                if m.chosen_rewards is not None:
                    metric_data["chosen_rewards"] = m.chosen_rewards
                if m.rejected_rewards is not None:
                    metric_data["rejected_rewards"] = m.rejected_rewards
                if m.reward_margin is not None:
                    metric_data["reward_margin"] = m.reward_margin
            
            # Extra metrics (always included if present)
            if m.extra_metrics:
                metric_data["extra_metrics"] = m.extra_metrics
            
            # System metrics (always included)
            if m.gpu_memory_mb is not None:
                metric_data["gpu_memory_mb"] = m.gpu_memory_mb
            if m.gpu_utilization_pct is not None:
                metric_data["gpu_utilization_pct"] = m.gpu_utilization_pct
            if m.throughput_samples_sec is not None:
                metric_data["throughput_samples_sec"] = m.throughput_samples_sec
            
            formatted_metrics.append(metric_data)
        
        return {
            "job_id": job_id,
            "train_type": train_type,
            "train_type_display": config["display_name"],
            "primary_metrics": config["primary_metrics"],
            "eval_metrics": config["eval_metrics"],
            "graph_configs": config.get("graph_configs", []),
            "count": len(formatted_metrics),
            "metrics": formatted_metrics
        }
    except Exception as e:
        return {
            "job_id": job_id,
            "count": 0,
            "metrics": [],
            "error": str(e)
        }


@router.get("/{job_id}/tensorboard")
async def get_tensorboard_data(job_id: str):
    """Get TensorBoard data for a job with data validation.
    
    Reads TensorBoard event files from the job's output directory.
    USF BIOS writes TensorBoard data to: {output_dir}/runs/
    
    IMPORTANT: Only returns validated data. If data appears corrupt or
    inconsistent, it is excluded to maintain user trust.
    """
    from pathlib import Path
    from ...core.capabilities import get_system_settings
    
    def is_valid_metric_value(value: float) -> bool:
        """Validate that a metric value is reasonable and not corrupt."""
        if value is None:
            return False
        if math.isnan(value) or math.isinf(value):
            return False
        # Reject extremely large values that indicate corruption
        if abs(value) > 1e10:
            return False
        return True
    
    def validate_metric_series(values: list) -> list:
        """Validate a series of metric values, removing invalid entries."""
        validated = []
        for v in values:
            if is_valid_metric_value(v.get("value")):
                validated.append(v)
        return validated
    
    try:
        output_dir = get_system_settings().OUTPUT_DIR / job_id
        
        # USF BIOS writes TensorBoard to {output_dir}/runs/ directory
        # Also check the main output_dir for compatibility
        search_dirs = [
            output_dir / "runs",  # Primary: USF BIOS default location
            output_dir,           # Fallback: direct in output_dir
            output_dir / "logs",  # Alternative location
        ]
        
        # Find TensorBoard event files
        tb_files = []
        for search_dir in search_dirs:
            if search_dir.exists():
                for root, dirs, files in os.walk(str(search_dir)):
                    for f in files:
                        if f.startswith('events.out.tfevents.'):
                            tb_files.append(os.path.join(root, f))
        
        if not tb_files:
            return {
                "job_id": job_id,
                "has_tensorboard": False,
                "data_source": "none",
                "metrics": {},
                "message": "No TensorBoard data found yet"
            }
        
        # Read TensorBoard data with validation
        try:
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
            
            all_metrics = {}
            total_valid_points = 0
            total_invalid_points = 0
            
            for tb_file in tb_files:
                try:
                    ea = EventAccumulator(tb_file)
                    ea.Reload()
                    tags = ea.Tags().get('scalars', [])
                    
                    for tag in tags:
                        values = ea.Scalars(tag)
                        # Normalize tag name (TensorBoard uses various formats)
                        normalized_tag = tag.replace('train/', '').replace('eval/', 'eval_').replace('/', '_').lower()
                        
                        if normalized_tag not in all_metrics:
                            all_metrics[normalized_tag] = []
                        
                        for v in values:
                            if is_valid_metric_value(v.value):
                                all_metrics[normalized_tag].append({
                                    "step": v.step,
                                    "value": round(v.value, 6),  # Limit precision
                                    "wall_time": v.wall_time
                                })
                                total_valid_points += 1
                            else:
                                total_invalid_points += 1
                except Exception as file_error:
                    # Skip corrupted files silently
                    continue
            
            # Remove empty metrics
            all_metrics = {k: v for k, v in all_metrics.items() if len(v) > 0}
            
            # Sort each metric by step
            for key in all_metrics:
                all_metrics[key].sort(key=lambda x: x["step"])
            
            if not all_metrics:
                return {
                    "job_id": job_id,
                    "has_tensorboard": False,
                    "data_source": "tensorboard_empty",
                    "metrics": {},
                    "message": "TensorBoard files found but no valid data yet"
                }
            
            return {
                "job_id": job_id,
                "has_tensorboard": True,
                "data_source": "tensorboard",
                "data_quality": {
                    "valid_points": total_valid_points,
                    "invalid_points": total_invalid_points,
                    "quality_score": round(total_valid_points / max(1, total_valid_points + total_invalid_points), 2)
                },
                "available_metrics": list(all_metrics.keys()),
                "metrics": all_metrics
            }
        except ImportError:
            return {
                "job_id": job_id,
                "has_tensorboard": False,
                "data_source": "unavailable",
                "error": "TensorBoard library not installed"
            }
            
    except Exception as e:
        return {
            "job_id": job_id,
            "has_tensorboard": False,
            "data_source": "error",
            "error": str(e)
        }


@router.get("/{job_id}/metrics/unified")
async def get_unified_metrics(
    job_id: str, 
    limit: int = Query(500, ge=1, le=2000),
    prefer_tensorboard: bool = Query(True),
    db: Session = Depends(get_db)
):
    """Get unified training metrics combining TensorBoard and parsed log data.
    
    This endpoint provides the most accurate real-time metrics by:
    1. Preferring TensorBoard data when available (more accurate)
    2. Falling back to parsed log data if TensorBoard unavailable
    3. Validating all data points before returning
    4. NEVER returning potentially incorrect data
    
    Data Sources Priority:
    - tensorboard: Highest accuracy, direct from training framework
    - database: Good accuracy, parsed from training logs
    - none: No data available yet
    """
    def is_valid_value(val) -> bool:
        """Check if a value is valid for display."""
        if val is None:
            return False
        if isinstance(val, float):
            if math.isnan(val) or math.isinf(val):
                return False
            if abs(val) > 1e10:
                return False
        return True
    
    try:
        # Get job info for training type
        job = await job_manager.get_job(job_id)
        if not job:
            return {"job_id": job_id, "error": "Job not found", "metrics": []}
        
        train_type = job.config.train_type.value if job and job.config else "sft"
        training_method = getattr(job.config, 'training_method', None)
        method_value = training_method.value if hasattr(training_method, 'value') else 'sft'
        
        # Determine effective training type for metric selection
        # RLHF methods override train_type for metric purposes
        effective_type = train_type
        if method_value == 'rlhf' and job.config.rlhf_type:
            rlhf_type = job.config.rlhf_type.value if hasattr(job.config.rlhf_type, 'value') else job.config.rlhf_type
            effective_type = rlhf_type  # Use specific RLHF type (dpo, ppo, kto, etc.)
        
        # Define metric configurations for each training type (comprehensive)
        TRAINING_TYPE_METRICS = {
            # SFT and variants
            "sft": ["loss", "learning_rate", "grad_norm", "epoch", "token_acc", "seq_acc", "eval_loss", "MFU"],
            "lora": ["loss", "learning_rate", "grad_norm", "epoch", "token_acc", "seq_acc", "eval_loss", "MFU"],
            "qlora": ["loss", "learning_rate", "grad_norm", "epoch", "token_acc", "seq_acc", "eval_loss", "MFU"],
            "adalora": ["loss", "learning_rate", "grad_norm", "epoch", "token_acc", "seq_acc", "eval_loss"],
            "dora": ["loss", "learning_rate", "grad_norm", "epoch", "token_acc", "seq_acc", "eval_loss"],
            "full": ["loss", "learning_rate", "grad_norm", "epoch", "token_acc", "seq_acc", "eval_loss", "MFU"],
            # Pre-training / Megatron
            "pt": ["loss", "learning_rate", "grad_norm", "perplexity", "epoch", "throughput", "iteration_time", 
                   "loss_scale", "params_norm", "batch_size", "num_zeros", "world_size"],
            "pretrain": ["loss", "learning_rate", "grad_norm", "perplexity", "epoch", "throughput", "iteration_time",
                        "loss_scale", "params_norm", "batch_size"],
            # DPO and variants (Offline RLHF)
            "dpo": ["loss", "learning_rate", "chosen_rewards", "rejected_rewards", "reward_margin", "reward_accuracy",
                   "logps_chosen", "logps_rejected", "nll_loss", "mean_chosen_logits", "mean_rejected_logits"],
            "kto": ["loss", "learning_rate", "chosen_rewards", "rejected_rewards", "kl_divergence", "reward_accuracy",
                   "logps_chosen", "logps_rejected", "nll_loss"],
            "simpo": ["loss", "learning_rate", "chosen_rewards", "rejected_rewards", "reward_margin", "reward_accuracy"],
            "orpo": ["loss", "learning_rate", "chosen_rewards", "rejected_rewards", "reward_accuracy"],
            "cpo": ["loss", "learning_rate", "chosen_rewards", "rejected_rewards", "reward_margin", "nll_loss"],
            "rpo": ["loss", "learning_rate", "chosen_rewards", "rejected_rewards", "reward_margin"],
            # PPO (Online RLHF)
            "ppo": ["reward", "policy_loss", "value_loss", "entropy", "kl_divergence", "learning_rate",
                   "clip_fraction", "approx_kl", "reward_std"],
            # GRPO (Online RLHF)
            "grpo": ["reward", "loss", "kl_divergence", "learning_rate", "reward_std", "frac_reward_zero_std",
                    "entropy_mean", "entropy_min", "entropy_max", "entropy_threshold", "policy_loss", "value_loss",
                    "clip_ratio_region_mean", "clip_ratio_low_min", "clip_ratio_high_max", "cispo_clip_ratio",
                    "completions_mean_length", "completions_min_length", "completions_max_length", 
                    "completions_clipped_ratio", "num_turns"],
            # GKD (Generative Knowledge Distillation)
            "gkd": ["loss", "learning_rate", "reward", "kl_divergence"],
            # Generic RLHF
            "rlhf": ["reward", "policy_loss", "value_loss", "entropy", "kl_divergence", "learning_rate"],
            # Reward Model
            "rm": ["loss", "accuracy", "chosen_rewards", "rejected_rewards", "reward_margin", "center_rewards_loss"],
            # MoE (Mixture of Experts)
            "moe": ["loss", "learning_rate", "grad_norm", "aux_loss", "load_balancing_loss", "z_loss"],
        }
        
        relevant_metrics = TRAINING_TYPE_METRICS.get(effective_type, TRAINING_TYPE_METRICS["sft"])
        
        unified_data = []
        data_source = "none"
        data_quality = 1.0
        
        # Try TensorBoard first if preferred
        if prefer_tensorboard:
            tb_response = await get_tensorboard_data(job_id)
            if tb_response.get("has_tensorboard") and tb_response.get("metrics"):
                tb_metrics = tb_response["metrics"]
                data_source = "tensorboard"
                data_quality = tb_response.get("data_quality", {}).get("quality_score", 1.0)
                
                # Find the metric with most data points to use as reference for steps
                max_len = 0
                reference_metric = None
                for key in ["loss", "train_loss", "reward"]:
                    if key in tb_metrics and len(tb_metrics[key]) > max_len:
                        max_len = len(tb_metrics[key])
                        reference_metric = key
                
                if reference_metric:
                    # Build unified data from TensorBoard
                    for i, ref_point in enumerate(tb_metrics[reference_metric]):
                        step = ref_point["step"]
                        data_point = {
                            "step": step,
                            "timestamp": ref_point.get("wall_time"),
                            "data_source": "tensorboard"
                        }
                        
                        # Add all available metrics for this step
                        for metric_key in relevant_metrics:
                            # Try various naming conventions
                            for tb_key in [metric_key, f"train_{metric_key}", f"train/{metric_key}"]:
                                if tb_key in tb_metrics:
                                    # Find value at this step
                                    for m in tb_metrics[tb_key]:
                                        if m["step"] == step and is_valid_value(m["value"]):
                                            data_point[metric_key] = m["value"]
                                            break
                        
                        unified_data.append(data_point)
        
        # If no TensorBoard data, use database metrics
        if not unified_data:
            service = JobService(db)
            db_metrics = service.get_job_metrics(job_id, limit=limit)
            
            if db_metrics:
                data_source = "database"
                for m in db_metrics:
                    data_point = {
                        "step": m.step,
                        "epoch": m.epoch,
                        "timestamp": m.timestamp.isoformat() if m.timestamp else None,
                        "data_source": "database"
                    }
                    
                    # Add validated metrics only
                    if is_valid_value(m.loss):
                        data_point["loss"] = round(m.loss, 6)
                    if is_valid_value(m.learning_rate):
                        data_point["learning_rate"] = m.learning_rate
                    if is_valid_value(m.grad_norm):
                        data_point["grad_norm"] = round(m.grad_norm, 4)
                    if is_valid_value(m.eval_loss):
                        data_point["eval_loss"] = round(m.eval_loss, 6)
                    if is_valid_value(m.reward):
                        data_point["reward"] = round(m.reward, 4)
                    if is_valid_value(m.kl_divergence):
                        data_point["kl_divergence"] = round(m.kl_divergence, 6)
                    if is_valid_value(m.policy_loss):
                        data_point["policy_loss"] = round(m.policy_loss, 6)
                    if is_valid_value(m.value_loss):
                        data_point["value_loss"] = round(m.value_loss, 6)
                    if is_valid_value(m.entropy):
                        data_point["entropy"] = round(m.entropy, 6)
                    if is_valid_value(m.chosen_rewards):
                        data_point["chosen_rewards"] = round(m.chosen_rewards, 4)
                    if is_valid_value(m.rejected_rewards):
                        data_point["rejected_rewards"] = round(m.rejected_rewards, 4)
                    if is_valid_value(m.reward_margin):
                        data_point["reward_margin"] = round(m.reward_margin, 4)
                    
                    # Extra metrics
                    if m.extra_metrics:
                        for key, val in m.extra_metrics.items():
                            if is_valid_value(val):
                                data_point[key] = round(val, 6) if isinstance(val, float) else val
                    
                    unified_data.append(data_point)
        
        # Filter to only include data points with at least one relevant metric
        filtered_data = []
        for point in unified_data:
            has_relevant = any(key in point and is_valid_value(point.get(key)) for key in relevant_metrics)
            if has_relevant:
                filtered_data.append(point)
        
        return {
            "job_id": job_id,
            "training_type": effective_type,
            "training_method": method_value,
            "data_source": data_source,
            "data_quality": data_quality,
            "relevant_metrics": relevant_metrics,
            "count": len(filtered_data),
            "metrics": filtered_data[-limit:] if len(filtered_data) > limit else filtered_data
        }
        
    except Exception as e:
        return {
            "job_id": job_id,
            "data_source": "error",
            "error": str(e),
            "metrics": []
        }


@router.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time job updates.
    
    DEPRECATED: This WebSocket endpoint is maintained for backward compatibility.
    For new integrations, prefer the HTTP SSE endpoint: GET /{job_id}/stream/logs
    
    The SSE endpoint provides the same real-time log streaming via standard HTTP,
    which is more reliable across proxies and doesn't require WebSocket support.
    """
    from ...services.sanitized_log_service import sanitized_log_service
    
    # Verify job exists
    job = await job_manager.get_job(job_id)
    if not job:
        await websocket.close(code=4004, reason="Job not found")
        return
    
    await ws_manager.connect(websocket, job_id)
    
    try:
        # Send initial state - try in-memory logs first, fall back to file-based logs
        logs = await job_manager.get_logs(job_id, last_n=50)
        
        # If no in-memory logs, try to get from terminal log file
        if not logs:
            try:
                logs = sanitized_log_service.get_terminal_logs(job_id, lines=50)
            except Exception:
                logs = []
        
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
