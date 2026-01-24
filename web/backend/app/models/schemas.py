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
    """Training type (parameter efficient or full)"""
    FULL = "full"
    LORA = "lora"
    QLORA = "qlora"
    ADALORA = "adalora"


class TrainingMethod(str, Enum):
    """Training method"""
    SFT = "sft"      # Supervised Fine-Tuning
    PT = "pt"        # Pre-Training
    RLHF = "rlhf"    # Reinforcement Learning from Human Feedback


class RLHFType(str, Enum):
    """RLHF algorithm type"""
    DPO = "dpo"      # Direct Preference Optimization
    ORPO = "orpo"    # Odds Ratio Preference Optimization
    SIMPO = "simpo"  # Simple Preference Optimization
    KTO = "kto"      # Kahneman-Tversky Optimization
    CPO = "cpo"      # Contrastive Preference Optimization
    RM = "rm"        # Reward Model training
    PPO = "ppo"      # Proximal Policy Optimization
    GRPO = "grpo"    # Group Relative Policy Optimization
    GKD = "gkd"      # Generalized Knowledge Distillation


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


# ============================================================================
# INCOMPATIBLE COMBINATIONS - Used for validation
# ============================================================================
INCOMPATIBLE_COMBINATIONS = {
    # DeepSpeed and FSDP cannot be used together
    ("deepspeed", "fsdp"): "DeepSpeed and FSDP cannot be used together",
    # Packing requires Flash Attention
    ("packing", "attn_impl"): {
        "condition": lambda config: config.packing and config.attn_impl not in [None, "flash_attn", "flash_attention_2", "flash_attention_3"],
        "message": "Packing requires Flash Attention (flash_attention_2 or flash_attention_3)"
    },
    # Liger kernel has limitations
    ("use_liger_kernel", "packing"): {
        "condition": lambda config: config.use_liger_kernel and config.packing,
        "message": "Liger kernel does not support packing yet"
    },
}

# RLHF algorithms that require reference model
RLHF_REQUIRES_REF_MODEL = ["dpo", "kto", "ppo", "grpo"]

# RLHF algorithms that support vLLM
RLHF_SUPPORTS_VLLM = ["grpo", "gkd"]


class TrainingConfig(BaseModel):
    """Training configuration"""
    # Training name (optional - auto-generated if not provided)
    name: Optional[str] = Field(default=None, description="Custom training name (auto-generated if empty)")
    
    # Model settings
    model_path: str = Field("/path/to/local/model", description="Local model path")
    model_source: ModelSource = ModelSource.LOCAL
    modality: Modality = Modality.TEXT
    
    # Training method (SFT, PT, RLHF)
    training_method: TrainingMethod = Field(default=TrainingMethod.SFT, description="Training method")
    
    # Training type (parameter efficient)
    train_type: TrainType = TrainType.LORA
    
    # RLHF specific - only used when training_method is RLHF
    rlhf_type: Optional[Literal["dpo", "orpo", "simpo", "kto", "cpo", "rm", "ppo", "grpo", "gkd"]] = Field(
        default=None, description="RLHF algorithm type"
    )
    
    # RLHF parameters
    beta: Optional[float] = Field(default=None, ge=0, description="Beta parameter for RLHF (deviation from ref model)")
    max_completion_length: int = Field(default=512, ge=1, description="Max generation length for GRPO/PPO/GKD")
    
    # DPO specific
    label_smoothing: float = Field(default=0.0, ge=0, le=1, description="Label smoothing for DPO")
    rpo_alpha: Optional[float] = Field(default=None, ge=0, description="RPO alpha (SFT loss weight)")
    
    # SimPO specific
    simpo_gamma: float = Field(default=1.0, ge=0, description="SimPO reward margin (0.5-1.5)")
    
    # KTO specific
    desirable_weight: float = Field(default=1.0, ge=0, description="KTO desirable loss weight")
    undesirable_weight: float = Field(default=1.0, ge=0, description="KTO undesirable loss weight")
    
    # PPO specific
    num_ppo_epochs: int = Field(default=4, ge=1, description="PPO epochs per update")
    kl_coef: float = Field(default=0.05, ge=0, description="KL divergence coefficient")
    cliprange: float = Field(default=0.2, ge=0, le=1, description="PPO clip range")
    
    # GRPO specific
    num_generations: int = Field(default=8, ge=1, description="GRPO generations per prompt")
    
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
    
    # Optimization - Liger Kernel (triton-based optimizations)
    use_liger_kernel: bool = Field(default=False, description="Use Liger kernel for faster training")
    
    # Optimization - Packing (combine short sequences)
    packing: bool = Field(default=False, description="Pack multiple sequences together to reduce padding waste")
    
    # Optimization - Sequence Parallelism
    sequence_parallel_size: int = Field(default=1, ge=1, description="Sequence parallel size (1 = disabled)")
    
    # Learning Rate Scheduler
    lr_scheduler_type: Literal["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "cosine_with_min_lr"] = Field(
        default="cosine", description="Learning rate scheduler type"
    )
    
    # Optimizer parameters
    weight_decay: float = Field(default=0.1, ge=0, le=1, description="Weight decay for optimizer")
    adam_beta2: float = Field(default=0.95, ge=0, le=1, description="Adam beta2 parameter")
    
    # GPU Selection
    # None or empty = use all available GPUs (auto-detect)
    # List of GPU indices like [0, 1, 2] = use only specified GPUs
    gpu_ids: Optional[List[int]] = Field(default=None, description="Specific GPU indices to use (None = all available)")
    
    # Number of GPUs to use (alternative to gpu_ids)
    # None = auto (use all or specified by gpu_ids)
    num_gpus: Optional[int] = Field(default=None, ge=1, description="Number of GPUs to use")
    
    early_stop_interval: Optional[int] = Field(default=None, ge=1)
    
    # API Tokens for private models/datasets (optional)
    hf_token: Optional[str] = Field(default=None, description="HuggingFace token for private models/datasets")
    ms_token: Optional[str] = Field(default=None, description="ModelScope token for private models/datasets")
    
    @field_validator('gpu_ids')
    @classmethod
    def validate_gpu_ids(cls, v):
        """Validate GPU IDs are non-negative and unique"""
        if v is not None:
            if len(v) == 0:
                return None  # Empty list means use all GPUs
            if any(gpu_id < 0 for gpu_id in v):
                raise ValueError("GPU IDs must be non-negative integers")
            if len(v) != len(set(v)):
                raise ValueError("GPU IDs must be unique")
        return v
    
    @model_validator(mode='after')
    def validate_all_combinations(self):
        """Comprehensive validation of all parameter combinations"""
        errors = []
        
        # DeepSpeed and FSDP cannot be used together
        if self.deepspeed and self.fsdp:
            errors.append("DeepSpeed and FSDP cannot be used together")
        
        # Packing requires Flash Attention
        if self.packing and self.attn_impl not in [None, "flash_attn", "flash_attention_2", "flash_attention_3"]:
            errors.append("Packing requires Flash Attention (set attn_impl to flash_attention_2 or flash_attention_3)")
        
        # Liger kernel limitations
        if self.use_liger_kernel and self.packing:
            errors.append("Liger kernel does not support packing yet")
        
        # RLHF validation
        if self.training_method.value == "rlhf":
            if not self.rlhf_type:
                errors.append("RLHF training requires rlhf_type to be specified")
            
            # Validate RLHF-specific combinations
            if self.rlhf_type in ["dpo", "kto", "ppo", "grpo"] and self.train_type.value == "full":
                pass  # ref_model is auto-set in backend
            
            # CPO/ORPO don't use ref_model with LoRA
            if self.rlhf_type in ["cpo", "orpo"] and self.train_type.value != "full":
                pass  # Valid - no ref_model needed
        
        # QLoRA requires quant_bits
        if self.train_type.value == "qlora" and not self.quant_bits:
            self.quant_bits = 4  # Default to 4-bit
        
        if errors:
            raise ValueError("; ".join(errors))
        
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
