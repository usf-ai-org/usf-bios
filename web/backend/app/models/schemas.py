# Copyright (c) US Inc. All rights reserved.
"""Pydantic schemas for API models"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class JobStatus(str, Enum):
    """Training job status"""
    PENDING = "pending"
    INITIALIZING = "initializing"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    STOPPED = "stopped"


class TrainType(str, Enum):
    """Training type"""
    FULL = "full"
    LORA = "lora"
    QLORA = "qlora"
    ADALORA = "adalora"


class ModelSource(str, Enum):
    """Model source"""
    HUGGINGFACE = "huggingface"
    MODELSCOPE = "modelscope"
    LOCAL = "local"


class Modality(str, Enum):
    """Model modality"""
    TEXT = "text"
    VISION = "vision"
    AUDIO = "audio"
    VIDEO = "video"


class TrainingConfig(BaseModel):
    """Training configuration"""
    # Model settings
    model_path: str = Field(..., description="Model path (HF, MS, or local)")
    model_source: ModelSource = ModelSource.HUGGINGFACE
    modality: Modality = Modality.TEXT
    
    # Training type
    train_type: TrainType = TrainType.LORA
    
    # Dataset
    dataset_path: str = Field(..., description="Dataset path")
    
    # Output
    output_dir: str = Field(default="output", description="Output directory")
    
    # Hyperparameters
    num_train_epochs: int = Field(default=3, ge=1, le=100)
    learning_rate: float = Field(default=1e-4, gt=0)
    per_device_train_batch_size: int = Field(default=1, ge=1)
    gradient_accumulation_steps: int = Field(default=16, ge=1)
    max_length: int = Field(default=2048, ge=128)
    
    # LoRA specific
    lora_rank: int = Field(default=8, ge=1)
    lora_alpha: int = Field(default=32, ge=1)
    lora_dropout: float = Field(default=0.05, ge=0, le=1)
    target_modules: str = Field(default="all-linear")
    
    # QLoRA specific
    quant_bits: Optional[int] = Field(default=None, description="4 or 8 for QLoRA")
    
    # Advanced
    torch_dtype: str = Field(default="bfloat16")
    deepspeed: Optional[str] = Field(default=None)
    fsdp: Optional[str] = Field(default=None)
    early_stop_interval: Optional[int] = Field(default=None)
    
    # Evaluation
    eval_strategy: Optional[str] = Field(default=None)
    eval_steps: Optional[int] = Field(default=None)
    save_steps: Optional[int] = Field(default=500)


class JobCreate(BaseModel):
    """Request to create a job"""
    config: TrainingConfig


class JobInfo(BaseModel):
    """Job information"""
    job_id: str
    status: JobStatus
    config: TrainingConfig
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    current_step: int = 0
    total_steps: int = 0
    current_epoch: int = 0
    total_epochs: int = 0
    current_loss: Optional[float] = None
    best_loss: Optional[float] = None
    learning_rate: Optional[float] = None
    eta_seconds: Optional[int] = None
    gpu_memory_used: Optional[str] = None
    error: Optional[str] = None


class JobResponse(BaseModel):
    """Job response with logs"""
    job: JobInfo
    logs: List[str] = []


class DatasetValidation(BaseModel):
    """Dataset validation result"""
    valid: bool
    total_samples: int = 0
    format_detected: str = "unknown"
    columns: List[str] = []
    sample_preview: List[Dict[str, Any]] = []
    errors: List[str] = []


class ModelValidation(BaseModel):
    """Model validation result"""
    valid: bool
    model_type: Optional[str] = None
    model_size: Optional[str] = None
    error: Optional[str] = None
