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


class VLLMMode(str, Enum):
    """vLLM deployment mode for online RL (GRPO/PPO/GKD)"""
    COLOCATE = "colocate"  # Training and inference share GPUs (internal vLLM engine)
    SERVER = "server"      # External vLLM server (separate GPU cluster)


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
# FEATURE DEPENDENCIES & UI CONTROL
# ============================================================================
# This defines which features depend on which selections, used for:
# 1. Dynamic UI enable/disable
# 2. Backend validation
# 3. Auto-nullify when dependencies change
# ============================================================================

FEATURE_DEPENDENCIES = {
    # ========== TRAINING METHOD DEPENDENCIES ==========
    "rlhf_type": {
        "depends_on": {"training_method": "rlhf"},
        "nullify_when_invalid": True,
        "message": "RLHF type is only applicable when training method is RLHF"
    },
    "beta": {
        "depends_on": {"training_method": "rlhf", "rlhf_type": ["dpo", "kto", "simpo", "cpo"]},
        "nullify_when_invalid": True,
        "message": "Beta parameter is only for DPO/KTO/SimPO/CPO"
    },
    "use_vllm": {
        "depends_on": {"training_method": "rlhf", "rlhf_type": ["grpo", "ppo", "gkd"]},
        "nullify_when_invalid": True,
        "message": "vLLM is only for online RL (GRPO/PPO/GKD)"
    },
    "vllm_mode": {
        "depends_on": {"use_vllm": True},
        "nullify_when_invalid": True,
        "message": "vLLM mode requires vLLM to be enabled"
    },
    
    # ========== TRAIN TYPE DEPENDENCIES ==========
    "lora_rank": {
        "depends_on": {"train_type": ["lora", "qlora", "adalora"]},
        "message": "LoRA rank is only for LoRA/QLoRA/AdaLoRA"
    },
    "lora_alpha": {
        "depends_on": {"train_type": ["lora", "qlora", "adalora"]},
        "message": "LoRA alpha is only for LoRA/QLoRA/AdaLoRA"
    },
    "lora_dropout": {
        "depends_on": {"train_type": ["lora", "qlora", "adalora"]},
        "message": "LoRA dropout is only for LoRA/QLoRA/AdaLoRA"
    },
    "use_rslora": {
        "depends_on": {"train_type": ["lora", "qlora", "adalora"]},
        "nullify_when_invalid": True,
        "message": "RSLoRA is only for LoRA training"
    },
    "use_dora": {
        "depends_on": {"train_type": ["lora", "qlora", "adalora"]},
        "nullify_when_invalid": True,
        "message": "DoRA is only for LoRA training"
    },
    "quant_bits": {
        "depends_on": {"train_type": "qlora"},
        "nullify_when_invalid": True,
        "message": "Quantization bits is only for QLoRA"
    },
    
    # ========== MULTIMODAL DEPENDENCIES ==========
    "freeze_llm": {
        "depends_on": {"modality": ["vision", "audio", "video"]},
        "message": "Freeze LLM is only for multimodal models"
    },
    "freeze_vit": {
        "depends_on": {"modality": ["vision", "audio", "video"]},
        "message": "Freeze ViT is only for multimodal models"
    },
    "freeze_aligner": {
        "depends_on": {"modality": ["vision", "audio", "video"]},
        "message": "Freeze aligner is only for multimodal models"
    },
    
    # ========== STREAMING DEPENDENCIES ==========
    "max_steps": {
        "required_when": {"streaming": True},
        "message": "max_steps is REQUIRED when streaming is enabled"
    },
    "shuffle_buffer_size": {
        "depends_on": {"streaming": True},
        "message": "Shuffle buffer is only for streaming mode"
    },
    
    # ========== ROPE SCALING DEPENDENCIES ==========
    "max_model_len": {
        "depends_on": {"rope_scaling": ["linear", "dynamic", "yarn"]},
        "message": "Max model length is used with RoPE scaling"
    },
}

# ============================================================================
# INCOMPATIBLE COMBINATIONS - Mutually exclusive options
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
    # RSLoRA and DoRA cannot be used together
    ("use_rslora", "use_dora"): {
        "condition": lambda config: config.use_rslora and config.use_dora,
        "message": "RSLoRA and DoRA cannot be used together"
    },
}

# ============================================================================
# RLHF ALGORITHM METADATA - For dynamic UI
# ============================================================================
RLHF_ALGORITHM_METADATA = {
    "dpo": {
        "name": "DPO",
        "full_name": "Direct Preference Optimization",
        "category": "offline",
        "requires_ref_model": True,
        "requires_vllm": False,
        "dataset_type": "preference",  # messages + rejected_messages
        "description": "Simple and stable preference optimization without reward model"
    },
    "orpo": {
        "name": "ORPO",
        "full_name": "Odds Ratio Preference Optimization",
        "category": "offline",
        "requires_ref_model": False,
        "requires_vllm": False,
        "dataset_type": "preference",
        "description": "Combines SFT and preference optimization in one step"
    },
    "simpo": {
        "name": "SimPO",
        "full_name": "Simple Preference Optimization",
        "category": "offline",
        "requires_ref_model": False,
        "requires_vllm": False,
        "dataset_type": "preference",
        "description": "Simplified DPO with length-normalized rewards"
    },
    "kto": {
        "name": "KTO",
        "full_name": "Kahneman-Tversky Optimization",
        "category": "offline",
        "requires_ref_model": True,
        "requires_vllm": False,
        "dataset_type": "binary",  # messages + label (0/1)
        "description": "Works with binary feedback (thumbs up/down)"
    },
    "cpo": {
        "name": "CPO",
        "full_name": "Contrastive Preference Optimization",
        "category": "offline",
        "requires_ref_model": False,
        "requires_vllm": False,
        "dataset_type": "preference",
        "description": "Contrastive learning approach for preference"
    },
    "rm": {
        "name": "RM",
        "full_name": "Reward Model Training",
        "category": "offline",
        "requires_ref_model": False,
        "requires_vllm": False,
        "dataset_type": "preference",
        "description": "Train a reward model for scoring responses"
    },
    "ppo": {
        "name": "PPO",
        "full_name": "Proximal Policy Optimization",
        "category": "online",
        "requires_ref_model": True,
        "requires_vllm": True,
        "dataset_type": "prompt",  # messages only (model generates)
        "description": "Classic RLHF with reward model"
    },
    "grpo": {
        "name": "GRPO",
        "full_name": "Group Relative Policy Optimization",
        "category": "online",
        "requires_ref_model": True,
        "requires_vllm": True,
        "dataset_type": "prompt",
        "description": "Efficient online RL without reward model"
    },
    "gkd": {
        "name": "GKD",
        "full_name": "Generalized Knowledge Distillation",
        "category": "online",
        "requires_ref_model": False,
        "requires_vllm": True,
        "dataset_type": "prompt",
        "description": "Knowledge distillation with online sampling"
    },
}

# RLHF algorithms that require reference model
RLHF_REQUIRES_REF_MODEL = ["dpo", "kto", "ppo", "grpo"]

# RLHF algorithms that support vLLM (online RL)
RLHF_SUPPORTS_VLLM = ["grpo", "ppo", "gkd"]

# RLHF algorithms that are online (require model sampling during training)
RLHF_ONLINE_ALGORITHMS = ["grpo", "ppo", "gkd"]

# RLHF algorithms that are offline (use pre-collected preference data)
RLHF_OFFLINE_ALGORITHMS = ["dpo", "orpo", "simpo", "kto", "cpo", "rm"]


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
    
    # Online RL vLLM configuration (for GRPO/PPO/GKD)
    use_vllm: bool = Field(default=True, description="Use vLLM for fast inference during online RL")
    vllm_mode: Optional[Literal["colocate", "server"]] = Field(
        default=None, description="vLLM deployment mode: 'colocate' (same GPUs) or 'server' (external)"
    )
    vllm_server_host: Optional[str] = Field(
        default=None, description="vLLM server host IP (required for server mode)"
    )
    vllm_server_port: Optional[int] = Field(
        default=8000, ge=1, le=65535, description="vLLM server port (default: 8000)"
    )
    vllm_tensor_parallel_size: int = Field(
        default=1, ge=1, description="Tensor parallel size for vLLM (colocate mode)"
    )
    vllm_gpu_memory_utilization: float = Field(
        default=0.9, ge=0.1, le=0.99, description="GPU memory utilization for vLLM (colocate mode)"
    )
    
    # Memory optimization for online RL (colocate mode)
    offload_model: bool = Field(default=False, description="Offload model to CPU during vLLM inference")
    offload_optimizer: bool = Field(default=False, description="Offload optimizer to CPU during vLLM inference")
    sleep_level: int = Field(default=0, ge=0, le=2, description="vLLM sleep level during training (0=off, 1=partial, 2=full)")
    
    # Reward functions for GRPO
    reward_funcs: Optional[List[str]] = Field(
        default=None, description="List of reward function names (e.g., ['accuracy', 'format'])"
    )
    
    # vLLM Server verification state (SECURITY: enforced by backend)
    # This MUST be True for server mode before training can start
    # Resets to False whenever host/port is changed
    vllm_server_verified: bool = Field(
        default=False, description="Whether vLLM server has been verified (required for server mode)"
    )
    # Hash of verified host:port - used by backend to detect tampering
    vllm_server_verified_hash: Optional[str] = Field(
        default=None, description="Hash of verified server config (backend security check)"
    )
    
    # Dataset
    dataset_path: str = Field(..., description="Dataset path")
    
    # Dataset type validation - sent by frontend for compatibility checks
    dataset_type: Optional[str] = Field(default=None, description="Detected dataset type (sft, rlhf_offline, rlhf_online, pt, kto)")
    dataset_type_display: Optional[str] = Field(default=None, description="Human-readable dataset type name")
    compatible_training_methods: Optional[List[str]] = Field(default=None, description="Compatible training methods")
    compatible_rlhf_types: Optional[List[str]] = Field(default=None, description="Compatible RLHF algorithms")
    
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
    
    # LoRA Advanced Options (from USF-BIOS tuner_args.py)
    use_rslora: bool = Field(default=False, description="Use Rank-Stabilized LoRA for better training stability")
    use_dora: bool = Field(default=False, description="Use DoRA (Weight-Decomposed LoRA) for improved performance")
    lora_bias: Literal["none", "all"] = Field(default="none", description="LoRA bias training: 'none' or 'all'")
    init_weights: Literal["true", "gaussian", "pissa", "olora", "loftq"] = Field(
        default="true", description="LoRA initialization method: true (default), gaussian, pissa, olora, loftq"
    )
    
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
    
    # Multimodal Model Options (from USF-BIOS tuner_args.py)
    # For VLMs like LLaVA, Qwen-VL, etc.
    freeze_llm: bool = Field(default=False, description="Freeze LLM weights for multimodal training (train only vision/aligner)")
    freeze_vit: bool = Field(default=True, description="Freeze Vision Transformer weights (default: True for VLMs)")
    freeze_aligner: bool = Field(default=True, description="Freeze aligner/projector weights (default: True)")
    
    # Long Context Support (from USF-BIOS model_args.py)
    rope_scaling: Optional[Literal["linear", "dynamic", "yarn"]] = Field(
        default=None, description="RoPE scaling for long context: 'linear', 'dynamic', or 'yarn'"
    )
    max_model_len: Optional[int] = Field(
        default=None, ge=1, description="Override max model length (for RoPE scaling)"
    )
    
    # Dataset Split (from USF-BIOS data_args.py)
    split_dataset_ratio: float = Field(
        default=0.0, ge=0, le=0.5, description="Auto split dataset into train/val (e.g., 0.1 = 10% for validation)"
    )
    
    # Learning Rate Scheduler
    lr_scheduler_type: Literal["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup", "cosine_with_min_lr"] = Field(
        default="cosine", description="Learning rate scheduler type"
    )
    
    # Optimizer parameters
    weight_decay: float = Field(default=0.1, ge=0, le=1, description="Weight decay for optimizer")
    adam_beta1: float = Field(default=0.9, ge=0, le=1, description="Adam beta1 parameter")
    adam_beta2: float = Field(default=0.95, ge=0, le=1, description="Adam beta2 parameter")
    max_grad_norm: float = Field(default=1.0, ge=0, description="Maximum gradient norm for clipping (0 = disabled)")
    
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
    
    # Dataset streaming for large datasets (billions of samples, 100TB+)
    # When enabled, dataset is read on-the-fly without loading into memory
    # IMPORTANT: --max_steps MUST be set when streaming=True (dataset length unknown)
    streaming: bool = Field(default=False, description="Enable streaming for large datasets (requires max_steps)")
    max_steps: Optional[int] = Field(default=None, ge=1, description="Max training steps (REQUIRED when streaming=True)")
    shuffle_buffer_size: int = Field(default=1000, ge=1, description="Buffer size for shuffling in streaming mode")
    
    # Multiple dataset mixing options (when using multiple datasets)
    # These are passed directly to USF-BIOS load_dataset() function
    interleave_prob: Optional[List[float]] = Field(
        default=None, 
        description="Probability weights for interleaving multiple datasets (e.g., [0.7, 0.3])"
    )
    stopping_strategy: Literal["first_exhausted", "all_exhausted"] = Field(
        default="first_exhausted",
        description="Strategy for multiple datasets: 'first_exhausted' or 'all_exhausted'"
    )
    
    # Adapter merge mode - merge LoRA/QLoRA adapter with base model before training
    # This enables full fine-tuning and RLHF on adapters by merging them first
    merge_adapter_before_training: bool = Field(
        default=False, description="Merge adapter with base model before training to unlock full fine-tuning"
    )
    adapter_base_model_path: Optional[str] = Field(
        default=None, description="Base model path for merging with adapter (required when merge_adapter_before_training=True)"
    )
    adapter_base_model_source: ModelSource = Field(
        default=ModelSource.LOCAL, 
        description="Source of the base model for adapter merging (can be different from main model source)"
    )
    
    # Existing adapter - continue training on an existing LoRA/QLoRA adapter
    existing_adapter_path: Optional[str] = Field(
        default=None, description="Path to existing adapter to continue training on (optional)"
    )
    existing_adapter_source: ModelSource = Field(
        default=ModelSource.LOCAL,
        description="Source of the existing adapter (local, huggingface, modelscope)"
    )
    
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
            
            # Online RL (GRPO/PPO/GKD) vLLM validation
            if self.rlhf_type in RLHF_ONLINE_ALGORITHMS:
                # Auto-set vllm_mode based on context if not specified
                if self.use_vllm and self.vllm_mode is None:
                    # Default to colocate for simplicity (will be overridden if 1 GPU)
                    self.vllm_mode = "colocate"
                
                # Server mode requires host
                if self.vllm_mode == "server" and not self.vllm_server_host:
                    errors.append("Server mode requires vllm_server_host to be specified")
            else:
                # Offline RL doesn't use vLLM
                self.use_vllm = False
                self.vllm_mode = None
        
        # QLoRA requires quant_bits
        if self.train_type.value == "qlora" and not self.quant_bits:
            self.quant_bits = 4  # Default to 4-bit
        
        # Streaming validation: max_steps is required when streaming is enabled
        if self.streaming and not self.max_steps:
            errors.append("Streaming mode requires max_steps to be specified (dataset length is unknown in streaming mode)")
        
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
    output_dir: Optional[str] = Field(default=None, description="Actual output directory for training artifacts")
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
