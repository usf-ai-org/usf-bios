# Copyright (c) US Inc. All rights reserved.
"""
Algorithm Compatibility Service - Enterprise Production Ready

This module defines the complete compatibility matrix between:
- Dataset types (SFT, RLHF Offline, RLHF Online, KTO, PT)
- Training methods (sft, rlhf, pt)
- RLHF algorithms (dpo, orpo, simpo, kto, cpo, rm, ppo, grpo, gkd)
- Quantization methods (none, 4bit, 8bit)
- Training types (full, lora, qlora)

Used by:
1. Frontend: To show/hide options dynamically based on dataset type
2. Backend: To validate training configuration before starting
3. Dataset detection: To tag datasets with compatible algorithms
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set


# ============================================================
# ENUMS - All possible values for each configuration option
# ============================================================

class DatasetType(str, Enum):
    """Dataset types based on format detection.
    
    Hierarchy:
    - SFT: Supervised Fine-Tuning
    - PT: Pre-Training
    - RLHF_OFFLINE_*: Offline RLHF (uses pre-collected data, no generation during training)
    - RLHF_ONLINE_*: Online RLHF (generates responses during training)
    
    This structure is scalable for future algorithms - just add new subtypes.
    """
    # ========== SUPERVISED FINE-TUNING ==========
    SFT = "sft"                                          # messages format (instruction-response pairs)
    
    # ========== PRE-TRAINING ==========
    PT = "pt"                                            # raw text for continued pre-training
    
    # ========== RLHF - OFFLINE (pre-collected data) ==========
    # Offline algorithms don't generate responses during training - they use pre-collected data
    RLHF_OFFLINE_PREFERENCE = "rlhf_offline_preference"  # prompt/chosen/rejected → DPO, ORPO, SimPO, CPO, RM
    RLHF_OFFLINE_BINARY = "rlhf_offline_binary"          # prompt/completion/label → KTO (binary good/bad feedback)
    # Future: RLHF_OFFLINE_MARGIN = "rlhf_offline_margin"  # For margin-based algorithms
    
    # ========== RLHF - ONLINE (generates responses) ==========
    # Online algorithms generate responses during training using reward signals
    RLHF_ONLINE = "rlhf_online"                          # prompt only → PPO, GRPO, GKD
    # Future: RLHF_ONLINE_REWARD = "rlhf_online_reward"    # For reward-model based online RL
    # Future: RLHF_ONLINE_DISTILL = "rlhf_online_distill"  # For knowledge distillation
    
    # ========== UNKNOWN ==========
    UNKNOWN = "unknown"                                  # Cannot determine type
    
    @classmethod
    def is_rlhf(cls, dataset_type: 'DatasetType') -> bool:
        """Check if dataset type is any RLHF variant"""
        return dataset_type.value.startswith('rlhf_')
    
    @classmethod
    def is_rlhf_offline(cls, dataset_type: 'DatasetType') -> bool:
        """Check if dataset type is offline RLHF"""
        return dataset_type.value.startswith('rlhf_offline_')
    
    @classmethod
    def is_rlhf_online(cls, dataset_type: 'DatasetType') -> bool:
        """Check if dataset type is online RLHF"""
        return dataset_type.value.startswith('rlhf_online')


class TrainingMethod(str, Enum):
    """High-level training methods"""
    SFT = "sft"      # Supervised Fine-Tuning
    RLHF = "rlhf"    # Reinforcement Learning from Human Feedback
    PT = "pt"        # Pre-Training


class RLHFAlgorithm(str, Enum):
    """RLHF algorithm types"""
    # Offline algorithms (use preference data)
    DPO = "dpo"      # Direct Preference Optimization
    ORPO = "orpo"    # Odds Ratio Preference Optimization
    SIMPO = "simpo"  # Simple Preference Optimization
    CPO = "cpo"      # Contrastive Preference Optimization
    RM = "rm"        # Reward Modeling
    KTO = "kto"      # Kahneman-Tversky Optimization
    
    # Online algorithms (generate responses)
    PPO = "ppo"      # Proximal Policy Optimization
    GRPO = "grpo"    # Group Relative Policy Optimization
    GKD = "gkd"      # Generalized Knowledge Distillation


class TrainingType(str, Enum):
    """Fine-tuning types"""
    FULL = "full"    # Full parameter fine-tuning
    LORA = "lora"    # Low-Rank Adaptation
    QLORA = "qlora"  # Quantized LoRA


class QuantizationType(str, Enum):
    """Quantization options"""
    NONE = "none"    # No quantization (FP16/BF16)
    INT8 = "8bit"    # 8-bit quantization
    INT4 = "4bit"    # 4-bit quantization


# ============================================================
# ALGORITHM DEFINITIONS - Complete configuration for each algorithm
# ============================================================

@dataclass
class AlgorithmConfig:
    """Configuration for a single algorithm"""
    id: str
    name: str
    description: str
    category: str  # "offline" or "online"
    
    # Dataset requirements
    required_fields: List[str]
    optional_fields: List[str] = field(default_factory=list)
    
    # Training compatibility
    supports_full: bool = True
    supports_lora: bool = True
    supports_qlora: bool = True
    
    # Special requirements
    requires_ref_model: bool = False
    requires_reward_model: bool = False
    requires_teacher_model: bool = False
    requires_vllm: bool = False
    
    # Default parameters
    default_beta: Optional[float] = None
    default_max_completion_length: Optional[int] = None
    
    # Sample format example
    sample_format: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "category": self.category,
            "required_fields": self.required_fields,
            "optional_fields": self.optional_fields,
            "supports_full": self.supports_full,
            "supports_lora": self.supports_lora,
            "supports_qlora": self.supports_qlora,
            "requires_ref_model": self.requires_ref_model,
            "requires_reward_model": self.requires_reward_model,
            "requires_teacher_model": self.requires_teacher_model,
            "requires_vllm": self.requires_vllm,
            "default_beta": self.default_beta,
            "default_max_completion_length": self.default_max_completion_length,
            "sample_format": self.sample_format,
        }


# ============================================================
# ALGORITHM REGISTRY - All supported algorithms
# ============================================================

ALGORITHM_REGISTRY: Dict[str, AlgorithmConfig] = {
    # ========== OFFLINE RLHF ALGORITHMS ==========
    "dpo": AlgorithmConfig(
        id="dpo",
        name="DPO (Direct Preference Optimization)",
        description="Learns from preference pairs without reward model. Most popular offline RLHF method.",
        category="offline",
        required_fields=["prompt", "chosen", "rejected"],
        optional_fields=["system", "margin"],
        supports_full=True,
        supports_lora=True,
        supports_qlora=True,
        requires_ref_model=True,  # For full training, ref model is needed
        default_beta=0.1,
        sample_format={
            "prompt": "What are the symptoms of diabetes?",
            "chosen": "The main symptoms of diabetes include increased thirst and frequent urination, unexplained weight loss, fatigue, blurred vision, slow-healing sores, and frequent infections.",
            "rejected": "Diabetes symptoms are thirst and peeing a lot. Just check your sugar levels at home."
        }
    ),
    
    "orpo": AlgorithmConfig(
        id="orpo",
        name="ORPO (Odds Ratio Preference Optimization)",
        description="Combines SFT and preference learning in one step. No reference model needed.",
        category="offline",
        required_fields=["prompt", "chosen", "rejected"],
        optional_fields=["system"],
        supports_full=True,
        supports_lora=True,
        supports_qlora=True,
        requires_ref_model=False,
        default_beta=0.1,
        sample_format={
            "prompt": "Explain quantum computing in simple terms.",
            "chosen": "Quantum computing uses quantum bits (qubits) that can exist in multiple states simultaneously, unlike classical bits which are either 0 or 1. This allows quantum computers to process many possibilities at once.",
            "rejected": "Quantum computing is just faster computers that use quantum stuff."
        }
    ),
    
    "simpo": AlgorithmConfig(
        id="simpo",
        name="SimPO (Simple Preference Optimization)",
        description="Simplified version of DPO with better training stability. No reference model needed.",
        category="offline",
        required_fields=["prompt", "chosen", "rejected"],
        optional_fields=["system"],
        supports_full=True,
        supports_lora=True,
        supports_qlora=True,
        requires_ref_model=False,
        default_beta=2.0,
        sample_format={
            "prompt": "How do I write a good cover letter?",
            "chosen": "A strong cover letter should: 1) Address the hiring manager by name, 2) Open with a compelling hook about your relevant experience, 3) Highlight 2-3 key achievements with metrics, 4) Show knowledge of the company, 5) End with a clear call to action.",
            "rejected": "Just write about yourself and why you want the job."
        }
    ),
    
    "cpo": AlgorithmConfig(
        id="cpo",
        name="CPO (Contrastive Preference Optimization)",
        description="Uses contrastive learning for preference optimization. Good for complex preferences.",
        category="offline",
        required_fields=["prompt", "chosen", "rejected"],
        optional_fields=["system"],
        supports_full=True,
        supports_lora=True,
        supports_qlora=True,
        requires_ref_model=False,
        default_beta=0.1,
        sample_format={
            "prompt": "What's the best way to learn programming?",
            "chosen": "The most effective approach combines: 1) Structured learning through courses or tutorials, 2) Hands-on practice with real projects, 3) Reading and understanding existing code, 4) Building a portfolio of projects, 5) Participating in coding communities for feedback.",
            "rejected": "Just watch YouTube tutorials and copy code."
        }
    ),
    
    "rm": AlgorithmConfig(
        id="rm",
        name="RM (Reward Modeling)",
        description="Trains a reward model to score responses. Used with PPO or other online methods.",
        category="offline",
        required_fields=["prompt", "chosen", "rejected"],
        optional_fields=["margin"],
        supports_full=True,
        supports_lora=True,
        supports_qlora=True,
        requires_ref_model=False,
        sample_format={
            "prompt": "Summarize the benefits of exercise.",
            "chosen": "Regular exercise improves cardiovascular health, strengthens muscles and bones, boosts mental health by releasing endorphins, helps maintain healthy weight, improves sleep quality, and reduces risk of chronic diseases like diabetes and heart disease.",
            "rejected": "Exercise is good for you and makes you healthy."
        }
    ),
    
    "kto": AlgorithmConfig(
        id="kto",
        name="KTO (Kahneman-Tversky Optimization)",
        description="Uses binary feedback (good/bad) instead of preference pairs. Simpler data collection.",
        category="offline",
        required_fields=["prompt", "completion", "label"],
        optional_fields=["system"],
        supports_full=True,
        supports_lora=True,
        supports_qlora=True,
        requires_ref_model=True,
        default_beta=0.1,
        sample_format={
            "prompt": "What is machine learning?",
            "completion": "Machine learning is a subset of artificial intelligence where computers learn patterns from data to make predictions or decisions without being explicitly programmed for each task.",
            "label": True  # True = desirable, False = undesirable
        }
    ),
    
    # ========== ONLINE RLHF ALGORITHMS ==========
    "ppo": AlgorithmConfig(
        id="ppo",
        name="PPO (Proximal Policy Optimization)",
        description="Classic online RL algorithm. Generates responses and learns from reward model feedback.",
        category="online",
        required_fields=["prompt"],
        optional_fields=["system"],
        supports_full=True,
        supports_lora=True,
        supports_qlora=False,  # PPO doesn't work well with QLoRA
        requires_ref_model=True,
        requires_reward_model=True,
        requires_vllm=True,
        default_beta=0.1,
        default_max_completion_length=512,
        sample_format={
            "prompt": "Write a professional email declining a meeting request.",
        }
    ),
    
    "grpo": AlgorithmConfig(
        id="grpo",
        name="GRPO (Group Relative Policy Optimization)",
        description="Generates multiple responses and optimizes based on relative rewards. Modern alternative to PPO.",
        category="online",
        required_fields=["prompt"],
        optional_fields=["system", "target"],
        supports_full=True,
        supports_lora=True,
        supports_qlora=True,
        requires_ref_model=True,
        requires_reward_model=False,  # Can use reward functions instead
        requires_vllm=True,
        default_beta=0.04,
        default_max_completion_length=1024,
        sample_format={
            "prompt": "Solve this math problem step by step: What is 15% of 240?",
        }
    ),
    
    "gkd": AlgorithmConfig(
        id="gkd",
        name="GKD (Generalized Knowledge Distillation)",
        description="Distills knowledge from a teacher model. Good for model compression.",
        category="online",
        required_fields=["prompt"],
        optional_fields=["system"],
        supports_full=True,
        supports_lora=True,
        supports_qlora=True,
        requires_ref_model=False,
        requires_teacher_model=True,
        requires_vllm=True,
        default_beta=0.5,
        default_max_completion_length=512,
        sample_format={
            "prompt": "Explain the water cycle.",
        }
    ),
}


# ============================================================
# DATASET TYPE CONFIGURATIONS - What each dataset type supports
# ============================================================

@dataclass
class DatasetTypeConfig:
    """Configuration for a dataset type"""
    id: str
    name: str
    display_name: str
    description: str
    
    # Compatible training methods
    compatible_methods: List[str]
    incompatible_methods: List[str]
    
    # Compatible RLHF algorithms (empty for non-RLHF types)
    compatible_algorithms: List[str]
    
    # Field patterns for detection
    required_fields: Set[str]
    alternative_field_patterns: List[Set[str]] = field(default_factory=list)
    
    # Sample dataset format
    sample_format: Dict[str, Any] = field(default_factory=dict)
    sample_format_description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "compatible_methods": self.compatible_methods,
            "incompatible_methods": self.incompatible_methods,
            "compatible_algorithms": self.compatible_algorithms,
            "sample_format": self.sample_format,
            "sample_format_description": self.sample_format_description,
        }


DATASET_TYPE_CONFIGS: Dict[str, DatasetTypeConfig] = {
    # ==========================================================================
    # KEY NAMES MUST MATCH DatasetType enum values from dataset_type_service.py
    # DatasetType.SFT = "sft", DatasetType.RLHF_PREF = "rlhf_pref", etc.
    # ==========================================================================
    "sft": DatasetTypeConfig(
        id="sft",
        name="SFT",
        display_name="SFT (Supervised Fine-Tuning)",
        description="Instruction-response pairs for teaching the model to follow instructions.",
        compatible_methods=["sft"],
        incompatible_methods=["pt", "rlhf"],
        compatible_algorithms=[],
        required_fields={"messages"},
        alternative_field_patterns=[
            {"instruction", "output"},
            {"instruction", "input", "output"},
            {"query", "response"},
        ],
        sample_format={
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "What is the capital of France?"},
                {"role": "assistant", "content": "The capital of France is Paris."}
            ]
        },
        sample_format_description="Messages array with role (system/user/assistant) and content pairs."
    ),
    
    # Key "rlhf_pref" matches DatasetType.RLHF_PREF from dataset_type_service.py
    "rlhf_pref": DatasetTypeConfig(
        id="rlhf_pref",
        name="RLHF_PREF",
        display_name="RLHF Preference (DPO/ORPO/SimPO/CPO/RM)",
        description="Preference pairs with chosen and rejected responses for offline RLHF training. Used by DPO, ORPO, SimPO, CPO, and Reward Model training.",
        compatible_methods=["rlhf"],
        incompatible_methods=["sft", "pt"],
        compatible_algorithms=["dpo", "orpo", "simpo", "cpo", "rm"],
        required_fields={"prompt", "chosen", "rejected"},
        alternative_field_patterns=[
            {"question", "chosen", "rejected"},
            {"messages", "rejected_response"},
            {"messages", "rejected_messages"},
        ],
        sample_format={
            "prompt": "What are the symptoms of diabetes?",
            "chosen": "The main symptoms of diabetes include increased thirst and frequent urination, unexplained weight loss, fatigue, blurred vision, slow-healing sores, and frequent infections. Type 1 diabetes symptoms often appear suddenly, while Type 2 symptoms develop gradually.",
            "rejected": "Diabetes symptoms are thirst and peeing a lot. You might lose weight too. Just check your sugar levels at home."
        },
        sample_format_description="Prompt with chosen (good) and rejected (bad) response pairs."
    ),
    
    "rlhf_online": DatasetTypeConfig(
        id="rlhf_online",
        name="RLHF_ONLINE",
        display_name="RLHF Online (Prompt Only)",
        description="Prompts only - the model generates responses during training using reward signals.",
        compatible_methods=["rlhf"],
        incompatible_methods=["sft", "pt"],
        compatible_algorithms=["ppo", "grpo", "gkd"],
        required_fields={"prompt"},
        alternative_field_patterns=[
            {"query"},
            {"messages"},  # Messages without assistant response
        ],
        sample_format={
            "prompt": "Write a professional email declining a meeting request politely."
        },
        sample_format_description="Prompt only - model generates responses during training."
    ),
    
    # Key "rlhf_binary" matches DatasetType.RLHF_BINARY from dataset_type_service.py
    "rlhf_binary": DatasetTypeConfig(
        id="rlhf_binary",
        name="RLHF_BINARY",
        display_name="RLHF Binary Feedback (KTO)",
        description="Binary feedback dataset with good/bad labels instead of preference pairs. Used by KTO (Kahneman-Tversky Optimization).",
        compatible_methods=["rlhf"],
        incompatible_methods=["sft", "pt"],
        compatible_algorithms=["kto"],
        required_fields={"prompt", "completion", "label"},
        alternative_field_patterns=[
            {"messages", "label"},
        ],
        sample_format={
            "prompt": "What is machine learning?",
            "completion": "Machine learning is a subset of artificial intelligence where computers learn patterns from data to make predictions or decisions without being explicitly programmed.",
            "label": True
        },
        sample_format_description="Prompt, completion, and boolean label (true=desirable, false=undesirable)."
    ),
    
    "pt": DatasetTypeConfig(
        id="pt",
        name="PT",
        display_name="Pre-Training (Raw Text)",
        description="Raw text data for continued pre-training or domain adaptation.",
        compatible_methods=["pt"],
        incompatible_methods=["sft", "rlhf"],
        compatible_algorithms=[],
        required_fields={"text"},
        alternative_field_patterns=[
            {"content"},
        ],
        sample_format={
            "text": "The mitochondria is the powerhouse of the cell. It produces ATP through cellular respiration, converting glucose and oxygen into energy that cells can use for various functions."
        },
        sample_format_description="Raw text for language modeling."
    ),
}


# ============================================================
# TRAINING TYPE CONFIGURATIONS
# ============================================================

@dataclass
class TrainingTypeConfig:
    """Configuration for a training type (full, lora, qlora)"""
    id: str
    name: str
    display_name: str
    description: str
    requires_quantization: bool
    default_quantization: Optional[str]
    compatible_quantizations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "requires_quantization": self.requires_quantization,
            "default_quantization": self.default_quantization,
            "compatible_quantizations": self.compatible_quantizations,
        }


TRAINING_TYPE_CONFIGS: Dict[str, TrainingTypeConfig] = {
    "full": TrainingTypeConfig(
        id="full",
        name="Full Fine-Tuning",
        display_name="Full Fine-Tuning",
        description="Updates all model parameters. Requires more memory but can achieve best results.",
        requires_quantization=False,
        default_quantization=None,
        compatible_quantizations=["none"],
    ),
    "lora": TrainingTypeConfig(
        id="lora",
        name="LoRA",
        display_name="LoRA (Low-Rank Adaptation)",
        description="Trains small adapter layers. Memory efficient with good results.",
        requires_quantization=False,
        default_quantization=None,
        compatible_quantizations=["none"],
    ),
    "qlora": TrainingTypeConfig(
        id="qlora",
        name="QLoRA",
        display_name="QLoRA (Quantized LoRA)",
        description="LoRA with quantized base model. Most memory efficient option.",
        requires_quantization=True,
        default_quantization="4bit",
        compatible_quantizations=["4bit", "8bit"],
    ),
}


# ============================================================
# QUANTIZATION CONFIGURATIONS
# ============================================================

@dataclass
class QuantizationConfig:
    """Configuration for quantization options"""
    id: str
    name: str
    display_name: str
    description: str
    bits: Optional[int]
    memory_reduction: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "bits": self.bits,
            "memory_reduction": self.memory_reduction,
        }


QUANTIZATION_CONFIGS: Dict[str, QuantizationConfig] = {
    "none": QuantizationConfig(
        id="none",
        name="None",
        display_name="No Quantization (FP16/BF16)",
        description="Full precision training. Best quality but highest memory usage.",
        bits=None,
        memory_reduction="0%",
    ),
    "8bit": QuantizationConfig(
        id="8bit",
        name="8-bit",
        display_name="8-bit Quantization",
        description="Reduces memory by ~50% with minimal quality loss.",
        bits=8,
        memory_reduction="~50%",
    ),
    "4bit": QuantizationConfig(
        id="4bit",
        name="4-bit",
        display_name="4-bit Quantization (NF4)",
        description="Reduces memory by ~75%. Best for limited GPU memory.",
        bits=4,
        memory_reduction="~75%",
    ),
}


# ============================================================
# COMPATIBILITY SERVICE
# ============================================================

class AlgorithmCompatibilityService:
    """Service for checking algorithm and training configuration compatibility"""
    
    @staticmethod
    def get_compatible_algorithms(dataset_type: str) -> List[Dict[str, Any]]:
        """Get list of compatible algorithms for a dataset type"""
        config = DATASET_TYPE_CONFIGS.get(dataset_type)
        if not config:
            return []
        
        algorithms = []
        for algo_id in config.compatible_algorithms:
            algo_config = ALGORITHM_REGISTRY.get(algo_id)
            if algo_config:
                algorithms.append(algo_config.to_dict())
        
        return algorithms
    
    @staticmethod
    def get_compatible_training_methods(dataset_type: str) -> List[str]:
        """Get compatible training methods for a dataset type"""
        config = DATASET_TYPE_CONFIGS.get(dataset_type)
        if not config:
            return ["sft", "pt", "rlhf"]  # Unknown type - allow all
        return config.compatible_methods
    
    @staticmethod
    def get_incompatible_training_methods(dataset_type: str) -> List[str]:
        """Get incompatible training methods for a dataset type.
        Derived from compatible methods (future-proof).
        """
        ALL_METHODS = {"sft", "pt", "rlhf"}
        config = DATASET_TYPE_CONFIGS.get(dataset_type)
        if not config:
            return list(ALL_METHODS)  # Unknown type - ALL incompatible
        # Derive incompatible from compatible
        return list(ALL_METHODS - set(config.compatible_methods))
    
    @staticmethod
    def is_algorithm_compatible(dataset_type: str, algorithm: str) -> bool:
        """Check if an algorithm is compatible with a dataset type"""
        config = DATASET_TYPE_CONFIGS.get(dataset_type)
        if not config:
            return True  # Unknown type - allow all
        return algorithm in config.compatible_algorithms
    
    @staticmethod
    def get_algorithm_config(algorithm: str) -> Optional[AlgorithmConfig]:
        """Get configuration for a specific algorithm"""
        return ALGORITHM_REGISTRY.get(algorithm)
    
    @staticmethod
    def get_all_algorithms() -> Dict[str, List[Dict[str, Any]]]:
        """Get all algorithms grouped by category"""
        offline = []
        online = []
        
        for algo_id, config in ALGORITHM_REGISTRY.items():
            algo_dict = config.to_dict()
            if config.category == "offline":
                offline.append(algo_dict)
            else:
                online.append(algo_dict)
        
        return {
            "offline": offline,
            "online": online,
        }
    
    @staticmethod
    def get_dataset_type_config(dataset_type: str) -> Optional[DatasetTypeConfig]:
        """Get configuration for a dataset type"""
        return DATASET_TYPE_CONFIGS.get(dataset_type)
    
    @staticmethod
    def get_all_dataset_types() -> List[Dict[str, Any]]:
        """Get all dataset type configurations"""
        return [config.to_dict() for config in DATASET_TYPE_CONFIGS.values()]
    
    @staticmethod
    def get_training_type_config(training_type: str) -> Optional[TrainingTypeConfig]:
        """Get configuration for a training type"""
        return TRAINING_TYPE_CONFIGS.get(training_type)
    
    @staticmethod
    def get_all_training_types() -> List[Dict[str, Any]]:
        """Get all training type configurations"""
        return [config.to_dict() for config in TRAINING_TYPE_CONFIGS.values()]
    
    @staticmethod
    def get_quantization_config(quantization: str) -> Optional[QuantizationConfig]:
        """Get configuration for a quantization type"""
        return QUANTIZATION_CONFIGS.get(quantization)
    
    @staticmethod
    def get_all_quantizations() -> List[Dict[str, Any]]:
        """Get all quantization configurations"""
        return [config.to_dict() for config in QUANTIZATION_CONFIGS.values()]
    
    @staticmethod
    def validate_training_config(
        dataset_type: str,
        training_method: str,
        rlhf_algorithm: Optional[str] = None,
        training_type: str = "lora",
        quantization: str = "none",
    ) -> Dict[str, Any]:
        """
        Validate a complete training configuration.
        
        Returns:
            Dict with:
            - valid: bool
            - errors: List[str] - blocking errors
            - warnings: List[str] - non-blocking warnings
            - suggestions: List[str] - improvement suggestions
        """
        errors = []
        warnings = []
        suggestions = []
        
        # Get dataset type config
        ds_config = DATASET_TYPE_CONFIGS.get(dataset_type)
        if not ds_config and dataset_type != "unknown":
            errors.append(f"Unknown dataset type: {dataset_type}")
            return {"valid": False, "errors": errors, "warnings": warnings, "suggestions": suggestions}
        
        # 1. Check training method compatibility with dataset type
        if ds_config:
            if training_method not in ds_config.compatible_methods:
                errors.append(
                    f"Training method '{training_method}' is not compatible with dataset type '{ds_config.display_name}'. "
                    f"Compatible methods: {', '.join(ds_config.compatible_methods)}"
                )
        
        # 2. Check RLHF algorithm compatibility
        if training_method == "rlhf":
            if not rlhf_algorithm:
                errors.append("RLHF training requires an algorithm selection (e.g., dpo, orpo, grpo)")
            elif ds_config and rlhf_algorithm not in ds_config.compatible_algorithms:
                errors.append(
                    f"RLHF algorithm '{rlhf_algorithm}' is not compatible with dataset type '{ds_config.display_name}'. "
                    f"Compatible algorithms: {', '.join(ds_config.compatible_algorithms)}"
                )
            
            # Check algorithm-specific requirements
            if rlhf_algorithm:
                algo_config = ALGORITHM_REGISTRY.get(rlhf_algorithm)
                if algo_config:
                    if algo_config.requires_vllm:
                        suggestions.append(f"{algo_config.name} performs best with vLLM enabled for fast generation.")
                    
                    # Check training type compatibility with algorithm
                    if training_type == "qlora" and not algo_config.supports_qlora:
                        errors.append(f"{algo_config.name} does not support QLoRA training.")
                    elif training_type == "lora" and not algo_config.supports_lora:
                        errors.append(f"{algo_config.name} does not support LoRA training.")
                    elif training_type == "full" and not algo_config.supports_full:
                        errors.append(f"{algo_config.name} does not support full fine-tuning.")
        
        # 3. Check training type and quantization compatibility
        tt_config = TRAINING_TYPE_CONFIGS.get(training_type)
        if tt_config:
            if quantization not in tt_config.compatible_quantizations:
                errors.append(
                    f"Quantization '{quantization}' is not compatible with training type '{tt_config.display_name}'. "
                    f"Compatible options: {', '.join(tt_config.compatible_quantizations)}"
                )
            
            if tt_config.requires_quantization and quantization == "none":
                errors.append(f"{tt_config.display_name} requires quantization (4bit or 8bit)")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
            "suggestions": suggestions,
        }
    
    @staticmethod
    def get_sample_dataset(dataset_type: str, algorithm: Optional[str] = None) -> Dict[str, Any]:
        """Get sample dataset format for a type/algorithm"""
        if algorithm:
            algo_config = ALGORITHM_REGISTRY.get(algorithm)
            if algo_config:
                return {
                    "algorithm": algorithm,
                    "algorithm_name": algo_config.name,
                    "description": algo_config.description,
                    "required_fields": algo_config.required_fields,
                    "optional_fields": algo_config.optional_fields,
                    "sample": algo_config.sample_format,
                }
        
        ds_config = DATASET_TYPE_CONFIGS.get(dataset_type)
        if ds_config:
            return {
                "dataset_type": dataset_type,
                "display_name": ds_config.display_name,
                "description": ds_config.description,
                "format_description": ds_config.sample_format_description,
                "sample": ds_config.sample_format,
                "compatible_algorithms": ds_config.compatible_algorithms,
            }
        
        return {"error": f"Unknown dataset type: {dataset_type}"}


# Singleton instance
algorithm_compatibility_service = AlgorithmCompatibilityService()
