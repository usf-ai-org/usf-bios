# Copyright (c) US Inc. All rights reserved.
"""Pydantic schemas for API models"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


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
    # Training name (optional - auto-generated if not provided)
    name: Optional[str] = Field(default=None, description="Custom training name (auto-generated if empty)")
    
    # Model settings
    model_path: str = Field("/path/to/local/model", description="Local model path")
    model_source: ModelSource = ModelSource.LOCAL
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
    torch_dtype: Literal["bfloat16", "float16", "float32"] = Field(default="bfloat16")
    warmup_ratio: float = Field(default=0.03, ge=0, le=1)
    
    # Optimization - Attention Implementation
    # Valid options from USF BIOS: None, 'sdpa', 'eager', 'flash_attn', 'flash_attention_2', 'flash_attention_3'
    attn_impl: Optional[Literal["sdpa", "eager", "flash_attn", "flash_attention_2", "flash_attention_3"]] = Field(
        default=None, description="Attention implementation"
    )
    
    # Optimization - DeepSpeed ZeRO
    # Valid options from USF BIOS config: 'zero0', 'zero1', 'zero2', 'zero2_offload', 'zero3', 'zero3_offload'
    deepspeed: Optional[Literal["zero0", "zero1", "zero2", "zero2_offload", "zero3", "zero3_offload"]] = Field(
        default=None, description="DeepSpeed ZeRO stage"
    )
    
    # Optimization - FSDP (cannot be used with DeepSpeed)
    # Valid options: 'full_shard', 'shard_grad_op', 'fsdp2'
    fsdp: Optional[Literal["full_shard", "shard_grad_op", "fsdp2"]] = Field(
        default=None, description="FSDP configuration"
    )
    
    # Optimization - Gradient Checkpointing
    gradient_checkpointing: bool = Field(default=True, description="Enable gradient checkpointing to save memory")
    
    early_stop_interval: Optional[int] = Field(default=None, ge=1)
    
    @model_validator(mode='after')
    def validate_optimization_combination(self):
        """Validate that DeepSpeed and FSDP are not used together"""
        if self.deepspeed and self.fsdp:
            raise ValueError("DeepSpeed and FSDP cannot be used together. Please choose one or the other.")
        return self
    
    @field_validator('quant_bits')
    @classmethod
    def validate_quant_bits(cls, v):
        """Validate quant_bits is 4 or 8 if provided"""
        if v is not None and v not in [4, 8]:
            raise ValueError("quant_bits must be 4 or 8")
        return v
    
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
    name: str = Field(..., description="User-friendly training name")
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
    resume_from_checkpoint: Optional[str] = Field(default=None, description="Checkpoint path to resume from")


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
