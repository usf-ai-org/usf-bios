# Copyright (c) US Inc. All rights reserved.
"""Training service - executes fine-tuning jobs"""

import asyncio
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from ..core.config import settings
from ..core.capabilities import get_system_settings
from ..core.database import SessionLocal
from ..models.schemas import JobInfo, JobStatus, TrainingConfig
from .job_manager import job_manager
from .websocket_manager import ws_manager
from .sanitized_log_service import sanitized_log_service
from .encrypted_log_service import encrypted_log_service
from .job_service import JobService
from .gpu_cleanup_service import gpu_cleanup_service


def _debug_log(job_id: str, message: str, level: str = "DEBUG"):
    """Write debug log to ENCRYPTED log file (only US Inc can read)."""
    encrypted_log_service.encrypt_and_format(f"[{level}] {message}", job_id, level)


class TrainingService:
    """Service for running training jobs"""
    
    def __init__(self):
        self._running_tasks: dict = {}
    
    def _validate_feature_flags(self, config: TrainingConfig) -> tuple[bool, str]:
        """
        Validate that requested features are enabled in this build.
        
        SECURITY: This validation uses compiled flags from system_guard.
        Users CANNOT bypass this even by modifying frontend or version.py.
        
        Returns: (is_valid, error_message)
        """
        try:
            from usf_bios.system_guard import (
                validate_train_type,
                validate_rlhf_algorithm,
                validate_vllm_mode,
                validate_training_method,
                FeatureDisabledError
            )
            
            # Validate training method (sft, rlhf, pt)
            training_method = getattr(config, 'training_method', None)
            if training_method:
                method_value = training_method.value if hasattr(training_method, 'value') else str(training_method)
                try:
                    validate_train_type(method_value)
                except FeatureDisabledError as e:
                    return False, str(e)
            
            # Validate train type (lora, qlora, full, adalora)
            train_type = getattr(config, 'train_type', None)
            if train_type:
                type_value = train_type.value if hasattr(train_type, 'value') else str(train_type)
                try:
                    validate_training_method(type_value)
                except FeatureDisabledError as e:
                    return False, str(e)
            
            # Validate RLHF algorithm if applicable
            if training_method:
                method_value = training_method.value if hasattr(training_method, 'value') else str(training_method)
                if method_value == "rlhf":
                    rlhf_type = getattr(config, 'rlhf_type', None)
                    if rlhf_type:
                        try:
                            validate_rlhf_algorithm(rlhf_type)
                        except FeatureDisabledError as e:
                            return False, str(e)
                    
                    # Validate vLLM mode for online RL
                    online_algos = ["grpo", "ppo", "gkd"]
                    if rlhf_type in online_algos:
                        vllm_mode = getattr(config, 'vllm_mode', None)
                        if vllm_mode:
                            try:
                                validate_vllm_mode(vllm_mode)
                            except FeatureDisabledError as e:
                                return False, str(e)
            
            return True, ""
            
        except ImportError:
            # system_guard not available (development mode) - allow everything
            return True, ""
    
    def _validate_online_rl_config(self, config: TrainingConfig) -> tuple[bool, str]:
        """
        Validate online RL configuration before training starts.
        
        SECURITY: This validation is enforced by backend - frontend cannot bypass.
        
        Checks:
        1. Server mode requires verified vLLM server endpoint
        2. Colocate mode requires sufficient GPU resources (2+ GPUs)
        3. Hash verification to prevent tampering with verified state
        
        Returns: (is_valid, error_message)
        """
        import hashlib
        
        training_method = getattr(config, 'training_method', None)
        if training_method:
            method_value = training_method.value if hasattr(training_method, 'value') else str(training_method)
        else:
            return True, ""
        
        if method_value != "rlhf":
            return True, ""
        
        rlhf_type = getattr(config, 'rlhf_type', None)
        online_rl_algorithms = ["grpo", "ppo", "gkd"]
        
        if rlhf_type not in online_rl_algorithms:
            return True, ""  # Offline RL doesn't need validation
        
        use_vllm = getattr(config, 'use_vllm', True)
        vllm_mode = getattr(config, 'vllm_mode', None)
        
        if not use_vllm:
            return True, ""  # Not using vLLM
        
        # Server mode validation
        if vllm_mode == "server":
            host = getattr(config, 'vllm_server_host', None)
            port = getattr(config, 'vllm_server_port', 8000)
            verified = getattr(config, 'vllm_server_verified', False)
            verified_hash = getattr(config, 'vllm_server_verified_hash', None)
            
            if not host:
                return False, "Server mode requires vllm_server_host to be specified"
            
            if not verified:
                return False, "vLLM server endpoint must be verified before training. Click 'Test Connection' in the UI."
            
            # Verify hash to prevent tampering
            expected_hash = hashlib.sha256(f"{host}:{port}".encode()).hexdigest()[:16]
            if verified_hash != expected_hash:
                return False, "vLLM server verification mismatch. Please re-verify the server endpoint."
        
        # Colocate mode validation
        elif vllm_mode == "colocate":
            import torch
            if torch.cuda.is_available():
                gpu_count = torch.cuda.device_count()
                if gpu_count < 2:
                    return False, "Colocate mode requires at least 2 GPUs. Use server mode instead."
            else:
                return False, "No GPU available for colocate mode"
        
        return True, ""
    
    def _build_command(self, config: TrainingConfig, job_id: str, resume_from_checkpoint: str = None) -> list:
        """Build the training command based on training method"""
        # SECURITY: Validate feature flags first (compiled, cannot bypass)
        is_valid, error = self._validate_feature_flags(config)
        if not is_valid:
            raise ValueError(f"Feature validation failed: {error}")
        
        # SECURITY: Validate online RL config before building command
        is_valid, error = self._validate_online_rl_config(config)
        if not is_valid:
            raise ValueError(f"Online RL validation failed: {error}")
        
        # Use locked output path from capabilities (ignores user-provided path when locked)
        from ..core.capabilities import get_validator
        validator = get_validator()
        user_output_dir = getattr(config, 'output_dir', '') or ''
        output_dir = validator.get_output_path(job_id, user_output_dir)
        
        # Determine CLI command based on training method
        training_method = getattr(config, 'training_method', None)
        if training_method:
            method_value = training_method.value if hasattr(training_method, 'value') else str(training_method)
        else:
            method_value = "sft"  # Default to SFT
        
        # Map train_type to USF BIOS compatible value
        # QLoRA is LoRA with quantization - USF BIOS doesn't have "qlora" train_type
        train_type_value = config.train_type.value
        if train_type_value == "qlora":
            train_type_value = "lora"  # QLoRA is LoRA + quant_bits
        
        cmd = [
            sys.executable, "-m", "usf_bios", method_value,
            "--model", config.model_path,
            "--train_type", train_type_value,
            "--dataset", config.dataset_path,
            "--output_dir", output_dir,
            "--num_train_epochs", str(config.num_train_epochs),
            "--learning_rate", str(config.learning_rate),
            "--per_device_train_batch_size", str(config.per_device_train_batch_size),
            "--gradient_accumulation_steps", str(config.gradient_accumulation_steps),
            "--max_length", str(config.max_length),
            "--torch_dtype", config.torch_dtype,
            "--logging_steps", "1",
            "--report_to", "tensorboard",
        ]
        
        # RLHF specific parameters
        if method_value == "rlhf":
            rlhf_type = getattr(config, 'rlhf_type', None)
            if rlhf_type:
                cmd.extend(["--rlhf_type", rlhf_type])
            
            # Beta parameter
            if hasattr(config, 'beta') and config.beta is not None:
                cmd.extend(["--beta", str(config.beta)])
            
            # Max completion length for GRPO/PPO/GKD
            if hasattr(config, 'max_completion_length') and config.max_completion_length:
                cmd.extend(["--max_completion_length", str(config.max_completion_length)])
            
            # DPO specific
            if rlhf_type == "dpo":
                if hasattr(config, 'label_smoothing') and config.label_smoothing > 0:
                    cmd.extend(["--label_smoothing", str(config.label_smoothing)])
                if hasattr(config, 'rpo_alpha') and config.rpo_alpha is not None:
                    cmd.extend(["--rpo_alpha", str(config.rpo_alpha)])
            
            # SimPO specific
            elif rlhf_type == "simpo":
                if hasattr(config, 'simpo_gamma') and config.simpo_gamma:
                    cmd.extend(["--simpo_gamma", str(config.simpo_gamma)])
            
            # KTO specific
            elif rlhf_type == "kto":
                if hasattr(config, 'desirable_weight') and config.desirable_weight != 1.0:
                    cmd.extend(["--desirable_weight", str(config.desirable_weight)])
                if hasattr(config, 'undesirable_weight') and config.undesirable_weight != 1.0:
                    cmd.extend(["--undesirable_weight", str(config.undesirable_weight)])
            
            # PPO specific
            elif rlhf_type == "ppo":
                if hasattr(config, 'num_ppo_epochs') and config.num_ppo_epochs:
                    cmd.extend(["--num_ppo_epochs", str(config.num_ppo_epochs)])
                if hasattr(config, 'kl_coef') and config.kl_coef:
                    cmd.extend(["--kl_coef", str(config.kl_coef)])
                if hasattr(config, 'cliprange') and config.cliprange:
                    cmd.extend(["--cliprange", str(config.cliprange)])
            
            # GRPO specific
            elif rlhf_type == "grpo":
                if hasattr(config, 'num_generations') and config.num_generations:
                    cmd.extend(["--num_generations", str(config.num_generations)])
            
            # Online RL vLLM configuration (GRPO/PPO/GKD)
            if rlhf_type in ["grpo", "ppo", "gkd"]:
                # vLLM usage
                if hasattr(config, 'use_vllm') and config.use_vllm:
                    cmd.extend(["--use_vllm", "true"])
                    
                    # vLLM mode (colocate or server)
                    if hasattr(config, 'vllm_mode') and config.vllm_mode:
                        cmd.extend(["--vllm_mode", config.vllm_mode])
                        
                        # Server mode specific
                        if config.vllm_mode == "server":
                            if hasattr(config, 'vllm_server_host') and config.vllm_server_host:
                                cmd.extend(["--vllm_server_host", config.vllm_server_host])
                            if hasattr(config, 'vllm_server_port') and config.vllm_server_port:
                                cmd.extend(["--vllm_server_port", str(config.vllm_server_port)])
                        
                        # Colocate mode specific
                        elif config.vllm_mode == "colocate":
                            if hasattr(config, 'vllm_tensor_parallel_size') and config.vllm_tensor_parallel_size > 1:
                                cmd.extend(["--vllm_tensor_parallel_size", str(config.vllm_tensor_parallel_size)])
                            if hasattr(config, 'vllm_gpu_memory_utilization') and config.vllm_gpu_memory_utilization:
                                cmd.extend(["--vllm_gpu_memory_utilization", str(config.vllm_gpu_memory_utilization)])
                            
                            # Memory optimization options
                            if hasattr(config, 'offload_model') and config.offload_model:
                                cmd.extend(["--offload_model", "true"])
                            if hasattr(config, 'offload_optimizer') and config.offload_optimizer:
                                cmd.extend(["--offload_optimizer", "true"])
                            if hasattr(config, 'sleep_level') and config.sleep_level > 0:
                                cmd.extend(["--sleep_level", str(config.sleep_level)])
                
                # Reward functions for GRPO
                if rlhf_type == "grpo" and hasattr(config, 'reward_funcs') and config.reward_funcs:
                    cmd.extend(["--reward_funcs", ",".join(config.reward_funcs)])
        
        # Warmup ratio
        if hasattr(config, 'warmup_ratio') and config.warmup_ratio is not None:
            cmd.extend(["--warmup_ratio", str(config.warmup_ratio)])
        
        # Resume from checkpoint
        if resume_from_checkpoint:
            cmd.extend(["--resume_from_checkpoint", resume_from_checkpoint])
        
        # LoRA parameters (LoRA, QLoRA, AdaLoRA all use lora params)
        if train_type_value in ["lora", "adalora"]:
            cmd.extend([
                "--lora_rank", str(config.lora_rank),
                "--lora_alpha", str(config.lora_alpha),
                "--lora_dropout", str(config.lora_dropout),
                "--target_modules", config.target_modules,
            ])
        
        # Quantization (QLoRA uses quant_bits=4 by default)
        # If train_type was "qlora", ensure quant_bits is set
        if config.train_type.value == "qlora":
            quant_bits = config.quant_bits if config.quant_bits else 4
            cmd.extend(["--quant_bits", str(quant_bits)])
        elif config.quant_bits:
            cmd.extend(["--quant_bits", str(config.quant_bits)])
        
        # Attention Implementation (Flash Attention, SDPA, etc.)
        if config.attn_impl:
            cmd.extend(["--attn_impl", config.attn_impl])
        
        # Gradient Checkpointing
        if hasattr(config, 'gradient_checkpointing') and config.gradient_checkpointing:
            cmd.extend(["--gradient_checkpointing", "true"])
        
        # Liger Kernel (triton-based optimizations)
        if hasattr(config, 'use_liger_kernel') and config.use_liger_kernel:
            cmd.extend(["--use_liger_kernel", "true"])
        
        # Packing (combine short sequences)
        if hasattr(config, 'packing') and config.packing:
            cmd.extend(["--packing", "true"])
        
        # Sequence Parallelism
        if hasattr(config, 'sequence_parallel_size') and config.sequence_parallel_size > 1:
            cmd.extend(["--sequence_parallel_size", str(config.sequence_parallel_size)])
        
        # Learning Rate Scheduler
        if hasattr(config, 'lr_scheduler_type') and config.lr_scheduler_type:
            cmd.extend(["--lr_scheduler_type", config.lr_scheduler_type])
        
        # Optimizer parameters
        if hasattr(config, 'weight_decay') and config.weight_decay is not None:
            cmd.extend(["--weight_decay", str(config.weight_decay)])
        if hasattr(config, 'adam_beta2') and config.adam_beta2 is not None:
            cmd.extend(["--adam_beta2", str(config.adam_beta2)])
        
        # DeepSpeed (cannot be used with FSDP)
        if config.deepspeed:
            cmd.extend(["--deepspeed", config.deepspeed])
        
        # FSDP (cannot be used with DeepSpeed)
        elif config.fsdp:
            cmd.extend(["--fsdp", config.fsdp])
        
        # Early stopping
        if config.early_stop_interval:
            cmd.extend(["--early_stop_interval", str(config.early_stop_interval)])
        
        # HuggingFace source flag - required when model/dataset is from HuggingFace
        model_source = getattr(config, 'model_source', None)
        if model_source:
            source_value = model_source.value if hasattr(model_source, 'value') else str(model_source)
            if source_value == 'huggingface':
                cmd.extend(["--use_hf", "true"])
        
        # Note: API tokens (hf_token, ms_token) are passed via environment variables
        # (HF_TOKEN, MODELSCOPE_API_TOKEN) for security - see _run_job method
        
        # Evaluation
        if config.eval_strategy:
            cmd.extend(["--eval_strategy", config.eval_strategy])
        if config.eval_steps:
            cmd.extend(["--eval_steps", str(config.eval_steps)])
        if config.save_steps:
            cmd.extend(["--save_steps", str(config.save_steps)])
        
        return cmd
    
    def _parse_log_line(self, line: str, job_id: str) -> dict:
        """Parse a log line to extract training metrics.
        
        Supports ALL training algorithms:
        - SFT: loss, learning_rate, grad_norm, epoch
        - RLHF/PPO: reward, kl_divergence, policy_loss, value_loss, entropy
        - DPO: chosen_rewards, rejected_rewards, reward_margin
        - GRPO: reward, kl_penalty, policy_gradient_loss
        - GKD: distillation_loss, student_loss, teacher_loss
        - Pre-training: loss, perplexity, tokens_per_second
        - ASR: wer, cer, loss
        - TTS: mel_loss, duration_loss, pitch_loss
        - Multimodal: image_loss, text_loss, contrastive_loss
        """
        metrics = {}
        
        # ============================================================
        # COMMON METRICS (all training types)
        # ============================================================
        
        # Parse loss (multiple formats)
        loss_match = re.search(r"'loss':\s*([\d.]+)", line)
        if loss_match:
            metrics["loss"] = float(loss_match.group(1))
        else:
            loss_alt = re.search(r"(?:loss|Loss)[=:\s]+([\d.]+)", line)
            if loss_alt:
                metrics["loss"] = float(loss_alt.group(1))
        
        # Parse learning rate
        lr_match = re.search(r"'learning_rate':\s*([\d.e\-+]+)", line)
        if lr_match:
            metrics["learning_rate"] = float(lr_match.group(1))
        else:
            lr_alt = re.search(r"(?:lr|LR|learning_rate)[=:\s]+([\d.e\-+]+)", line)
            if lr_alt:
                metrics["learning_rate"] = float(lr_alt.group(1))
        
        # Parse step - supports multiple formats
        # Format 1: 'global_step/max_steps': '2/3' (USF BIOS format)
        step_max_match = re.search(r"'global_step/max_steps':\s*'(\d+)/(\d+)'", line)
        if step_max_match:
            metrics["step"] = int(step_max_match.group(1))
            metrics["total_steps"] = int(step_max_match.group(2))
        else:
            # Format 2: 'step': 2 or 'global_step': 2
            step_match = re.search(r"'(global_)?step':\s*(\d+)", line)
            if step_match:
                metrics["step"] = int(step_match.group(2))
            else:
                # Format 3: Step 2 or step=2
                step_alt = re.search(r"(?:Step|step)[=:\s]+(\d+)", line)
                if step_alt:
                    metrics["step"] = int(step_alt.group(1))
        
        # Parse epoch
        epoch_match = re.search(r"'epoch':\s*([\d.]+)", line)
        if epoch_match:
            metrics["epoch"] = float(epoch_match.group(1))
        else:
            epoch_alt = re.search(r"(?:Epoch|epoch)[=:\s]+([\d.]+)", line)
            if epoch_alt:
                metrics["epoch"] = float(epoch_alt.group(1))
        
        # Parse grad_norm
        grad_match = re.search(r"'grad_norm':\s*([\d.]+)", line)
        if grad_match:
            metrics["grad_norm"] = float(grad_match.group(1))
        
        # Parse eval_loss
        eval_loss_match = re.search(r"'eval_loss':\s*([\d.]+)", line)
        if eval_loss_match:
            metrics["eval_loss"] = float(eval_loss_match.group(1))
        
        # ============================================================
        # RLHF / PPO / GRPO METRICS
        # ============================================================
        
        # Reward
        reward_match = re.search(r"'reward':\s*([\d.\-e+]+)", line)
        if reward_match:
            metrics["reward"] = float(reward_match.group(1))
        
        # KL divergence
        kl_match = re.search(r"'kl(?:_divergence)?':\s*([\d.\-e+]+)", line)
        if kl_match:
            metrics["kl_divergence"] = float(kl_match.group(1))
        
        # Policy loss
        policy_loss_match = re.search(r"'policy_loss':\s*([\d.\-e+]+)", line)
        if policy_loss_match:
            metrics["policy_loss"] = float(policy_loss_match.group(1))
        
        # Value loss
        value_loss_match = re.search(r"'value_loss':\s*([\d.\-e+]+)", line)
        if value_loss_match:
            metrics["value_loss"] = float(value_loss_match.group(1))
        
        # Entropy
        entropy_match = re.search(r"'entropy':\s*([\d.\-e+]+)", line)
        if entropy_match:
            metrics["entropy"] = float(entropy_match.group(1))
        
        # ============================================================
        # DPO METRICS
        # ============================================================
        
        # Chosen rewards
        chosen_match = re.search(r"'(?:chosen_rewards|rewards/chosen)':\s*([\d.\-e+]+)", line)
        if chosen_match:
            metrics["chosen_rewards"] = float(chosen_match.group(1))
        
        # Rejected rewards
        rejected_match = re.search(r"'(?:rejected_rewards|rewards/rejected)':\s*([\d.\-e+]+)", line)
        if rejected_match:
            metrics["rejected_rewards"] = float(rejected_match.group(1))
        
        # Reward margin
        margin_match = re.search(r"'(?:reward_margin|rewards/margin)':\s*([\d.\-e+]+)", line)
        if margin_match:
            metrics["reward_margin"] = float(margin_match.group(1))
        
        # ============================================================
        # EXTRA METRICS (stored in JSON column)
        # ============================================================
        extra = {}
        
        # Perplexity (pre-training)
        ppl_match = re.search(r"'perplexity':\s*([\d.]+)", line)
        if ppl_match:
            extra["perplexity"] = float(ppl_match.group(1))
        
        # WER/CER (ASR)
        wer_match = re.search(r"'wer':\s*([\d.]+)", line)
        if wer_match:
            extra["wer"] = float(wer_match.group(1))
        cer_match = re.search(r"'cer':\s*([\d.]+)", line)
        if cer_match:
            extra["cer"] = float(cer_match.group(1))
        
        # Mel loss (TTS)
        mel_match = re.search(r"'mel_loss':\s*([\d.]+)", line)
        if mel_match:
            extra["mel_loss"] = float(mel_match.group(1))
        
        # Contrastive loss (multimodal)
        contrastive_match = re.search(r"'contrastive_loss':\s*([\d.]+)", line)
        if contrastive_match:
            extra["contrastive_loss"] = float(contrastive_match.group(1))
        
        # Approx KL (PPO)
        approx_kl_match = re.search(r"'approx_kl':\s*([\d.\-e+]+)", line)
        if approx_kl_match:
            extra["approx_kl"] = float(approx_kl_match.group(1))
        
        # Clip fraction (PPO)
        clip_match = re.search(r"'clip(?:_fraction)?':\s*([\d.]+)", line)
        if clip_match:
            extra["clip_fraction"] = float(clip_match.group(1))
        
        if extra:
            metrics["extra_metrics"] = extra
        
        # ============================================================
        # PROGRESS BAR FORMAT
        # ============================================================
        progress_match = re.search(r"(\d+)/(\d+)\s*\[", line)
        if progress_match:
            metrics["step"] = int(progress_match.group(1))
            metrics["total_steps"] = int(progress_match.group(2))
        
        return metrics
    
    async def run_training(self, job_id: str) -> None:
        """Run the training process"""
        _debug_log(job_id, "run_training called")
        
        job = await job_manager.get_job(job_id)
        if not job:
            _debug_log(job_id, "Job not found in job_manager", "ERROR")
            return
        
        _debug_log(job_id, f"Job found, config: {job.config}")
        
        # ============================================================
        # PRE-TRAINING GPU CLEANUP: Ensure GPU memory is clean
        # This prevents OOM errors from previous training runs
        # ============================================================
        try:
            _debug_log(job_id, "Performing pre-training GPU cleanup...")
            cleanup_result = await gpu_cleanup_service.async_deep_cleanup(
                kill_orphans=True,
                exclude_pids=[os.getpid()]
            )
            if cleanup_result.get("success"):
                freed_gb = cleanup_result.get("memory_freed_gb", 0)
                if freed_gb > 0:
                    _debug_log(job_id, f"Pre-training cleanup freed {freed_gb}GB VRAM")
                    sanitized_log_service.create_terminal_log(
                        job_id, 
                        f"GPU cleanup freed {freed_gb}GB VRAM", 
                        "INFO"
                    )
            else:
                _debug_log(job_id, f"Pre-training cleanup warning: {cleanup_result.get('error', 'unknown')}", "WARN")
        except Exception as cleanup_err:
            _debug_log(job_id, f"Pre-training cleanup error (non-fatal): {cleanup_err}", "WARN")
        
        try:
            # Update status
            await job_manager.update_job(job_id, 
                status=JobStatus.INITIALIZING,
                started_at=datetime.now()
            )
            await ws_manager.send_status(job_id, "initializing")
            
            # Write to terminal log file FIRST (this is what frontend polls)
            sanitized_log_service.create_terminal_log(job_id, "Initializing training environment...", "INFO")
            await ws_manager.send_log(job_id, "Initializing training environment...")
            _debug_log(job_id, "Status set to INITIALIZING")
            
            # ============================================================
            # PRE-TRAINING VALIDATION - Catch issues early
            # ============================================================
            validation_errors = []
            
            # Check model path exists (for local models)
            model_source = getattr(job.config.model_source, 'value', str(job.config.model_source))
            if model_source == 'local':
                model_path = job.config.model_path
                if not os.path.exists(model_path):
                    validation_errors.append(f"Model path does not exist: {model_path}")
                elif not os.path.isdir(model_path):
                    validation_errors.append(f"Model path is not a directory: {model_path}")
                else:
                    # Check for config.json which indicates a valid model directory
                    config_path = os.path.join(model_path, "config.json")
                    if not os.path.exists(config_path):
                        sanitized_log_service.create_terminal_log(job_id, "Warning: config.json not found in model directory", "WARN")
                        await ws_manager.send_log(job_id, f"Warning: config.json not found in model directory")
                        _debug_log(job_id, f"Warning: config.json not found at {config_path}")
            
            # Check dataset paths exist
            dataset_paths = job.config.dataset_path.split(',') if job.config.dataset_path else []
            for ds_path in dataset_paths:
                ds_path = ds_path.strip()
                if ds_path and not ds_path.upper().startswith(('HF::', 'MS::')):
                    if not os.path.exists(ds_path):
                        validation_errors.append(f"Dataset path does not exist: {ds_path}")
            
            # ============================================================
            # OPTIMIZATION COMPATIBILITY VALIDATION
            # Check for incompatible combinations before training starts
            # ============================================================
            config = job.config
            
            # DeepSpeed + FSDP conflict
            if getattr(config, 'deepspeed', None) and getattr(config, 'fsdp', None):
                validation_errors.append(
                    "DeepSpeed and FSDP cannot be used together. "
                    "They are both distributed training frameworks. Please disable one."
                )
            
            # Packing requires Flash Attention
            if getattr(config, 'packing', False):
                attn_impl = getattr(config, 'attn_impl', None)
                flash_attn_values = ['flash_attn', 'flash_attention_2', 'flash_attention_3']
                if attn_impl and attn_impl not in flash_attn_values:
                    validation_errors.append(
                        f"Sequence Packing requires Flash Attention but '{attn_impl}' is selected. "
                        "Please enable Flash Attention 2 or 3, or disable Packing."
                    )
            
            # Flash Attention 3 requires Hopper GPU
            attn_impl = getattr(config, 'attn_impl', None)
            if attn_impl == 'flash_attention_3':
                try:
                    import torch
                    if torch.cuda.is_available():
                        props = torch.cuda.get_device_properties(0)
                        if props.major < 9:  # Hopper is sm_90 (major=9)
                            validation_errors.append(
                                f"Flash Attention 3 requires Hopper GPU (H100/H200) but detected "
                                f"{props.name} (compute capability {props.major}.{props.minor}). "
                                "Please use Flash Attention 2 or SDPA instead."
                            )
                except Exception as e:
                    _debug_log(job_id, f"GPU detection error: {e}", "WARN")
            
            # Liger Kernel + Packing warning (not error, just log)
            if getattr(config, 'use_liger_kernel', False) and getattr(config, 'packing', False):
                sanitized_log_service.create_terminal_log(
                    job_id, 
                    "Warning: Liger Kernel may have issues with Sequence Packing. Proceeding anyway.", 
                    "WARN"
                )
                await ws_manager.send_log(job_id, "Warning: Liger Kernel + Packing combination may be unstable")
            
            # Report validation errors
            if validation_errors:
                error_msg = "Pre-training validation failed:\n" + "\n".join(f"  - {e}" for e in validation_errors)
                sanitized_log_service.create_terminal_log(job_id, f"ERROR: {error_msg}", "ERROR")
                await ws_manager.send_log(job_id, f"ERROR: {error_msg}")
                await job_manager.add_log(job_id, f"ERROR: {error_msg}")
                _debug_log(job_id, error_msg, "ERROR")
                
                await job_manager.update_job(job_id, 
                    status=JobStatus.FAILED,
                    error=error_msg,
                    completed_at=datetime.now()
                )
                await ws_manager.send_status(job_id, "failed")
                return
            
            # Note: Don't show validation message or command to user - internal details
            
            # Build command (with resume checkpoint if set)
            resume_checkpoint = getattr(job, 'resume_from_checkpoint', None)
            if resume_checkpoint:
                sanitized_log_service.create_terminal_log(job_id, f"Resuming from checkpoint: {resume_checkpoint}", "INFO")
                await ws_manager.send_log(job_id, f"Resuming from checkpoint: {resume_checkpoint}")
                _debug_log(job_id, f"Resuming from checkpoint: {resume_checkpoint}")
            
            cmd = self._build_command(job.config, job_id, resume_from_checkpoint=resume_checkpoint)
            cmd_str = " ".join(cmd)
            _debug_log(job_id, f"Built command: {cmd_str}")
            
            # Store full command in encrypted log only (for US Inc) - NOT shown to user
            encrypted_log_service.encrypt_and_format(f"Command: {cmd_str}", job_id)
            _debug_log(job_id, f"Full command: {cmd_str}")
            
            # ============================================================
            # SHOW USER-FRIENDLY CONFIG SUMMARY (not raw command)
            # ============================================================
            training_method = getattr(job.config, 'training_method', None)
            method_display = training_method.value.upper() if hasattr(training_method, 'value') else 'SFT'
            train_type_display = job.config.train_type.value.upper() if hasattr(job.config.train_type, 'value') else 'LORA'
            
            # Build config summary for user
            config_lines = [
                f"Training Configuration:",
                f"  • Method: {method_display} ({train_type_display})",
                f"  • Epochs: {job.config.num_train_epochs}",
                f"  • Learning Rate: {job.config.learning_rate}",
                f"  • Batch Size: {job.config.per_device_train_batch_size}",
                f"  • Max Length: {job.config.max_length}",
            ]
            
            # Add LoRA params if applicable
            if train_type_display in ['LORA', 'QLORA', 'ADALORA']:
                config_lines.append(f"  • LoRA Rank: {job.config.lora_rank}")
                config_lines.append(f"  • LoRA Alpha: {job.config.lora_alpha}")
            
            # Add dataset info (just filename, not full path)
            dataset_name = os.path.basename(job.config.dataset_path) if job.config.dataset_path else 'dataset'
            config_lines.append(f"  • Dataset: {dataset_name}")
            
            config_summary = "\n".join(config_lines)
            sanitized_log_service.create_terminal_log(job_id, config_summary, "INFO")
            await ws_manager.send_log(job_id, config_summary)
            
            sanitized_log_service.create_terminal_log(job_id, "Starting training...", "INFO")
            await ws_manager.send_log(job_id, "Starting training...")
            
            # Create process with proper environment
            # Include DeepSpeed build flags to prevent CUDA_HOME errors
            _debug_log(job_id, "Creating subprocess...")
            training_env = {
                **os.environ,
                "PYTHONUNBUFFERED": "1",
                # DeepSpeed: Disable JIT compilation of CUDA ops
                # This prevents "CUDA_HOME does not exist" errors
                "DS_BUILD_OPS": "0",
                "DS_BUILD_AIO": "0",
                "DS_BUILD_SPARSE_ATTN": "0",
                "DS_BUILD_TRANSFORMER": "0",
                "DS_BUILD_TRANSFORMER_INFERENCE": "0",
                "DS_BUILD_STOCHASTIC_TRANSFORMER": "0",
                "DS_BUILD_UTILS": "0",
                "DS_BUILD_FUSED_ADAM": "0",
                "DS_BUILD_FUSED_LAMB": "0",
                "DS_BUILD_CPU_ADAM": "0",
                "DS_BUILD_CPU_LION": "0",
                "DS_BUILD_EVOFORMER_ATTN": "0",
                "DS_BUILD_INFERENCE_CORE_OPS": "0",
                "DS_BUILD_CUTLASS_OPS": "0",
                "DS_BUILD_RAGGED_DEVICE_OPS": "0",
            }
            
            # API Tokens - pass via environment variable for security (not visible in process list)
            if hasattr(job.config, 'hf_token') and job.config.hf_token:
                training_env["HF_TOKEN"] = job.config.hf_token
            if hasattr(job.config, 'ms_token') and job.config.ms_token:
                training_env["MODELSCOPE_API_TOKEN"] = job.config.ms_token
            
            # ============================================================
            # GPU SELECTION: Configure which GPUs to use
            # ============================================================
            num_gpus_to_use = 1  # Default to 1 GPU
            
            # Get available GPU count from system
            try:
                import torch
                available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            except:
                available_gpus = 0
            
            if available_gpus > 0:
                # Determine which GPUs to use
                if hasattr(job.config, 'gpu_ids') and job.config.gpu_ids:
                    # User specified specific GPUs
                    gpu_ids = job.config.gpu_ids
                    # Validate GPU IDs don't exceed available GPUs
                    valid_gpu_ids = [g for g in gpu_ids if g < available_gpus]
                    if len(valid_gpu_ids) != len(gpu_ids):
                        sanitized_log_service.create_terminal_log(job_id, f"Warning: Some GPU IDs exceed available GPUs ({available_gpus}). Using valid GPUs only.", "WARN")
                        await ws_manager.send_log(job_id, f"Warning: Some GPU IDs exceed available GPUs ({available_gpus}). Using valid GPUs only.")
                    if valid_gpu_ids:
                        training_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in valid_gpu_ids)
                        num_gpus_to_use = len(valid_gpu_ids)
                        sanitized_log_service.create_terminal_log(job_id, f"Using specified GPUs: {valid_gpu_ids}", "INFO")
                        await ws_manager.send_log(job_id, f"Using specified GPUs: {valid_gpu_ids}")
                elif hasattr(job.config, 'num_gpus') and job.config.num_gpus:
                    # User specified number of GPUs
                    num_gpus_to_use = min(job.config.num_gpus, available_gpus)
                    training_env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus_to_use))
                    sanitized_log_service.create_terminal_log(job_id, f"Using {num_gpus_to_use} GPUs (0-{num_gpus_to_use-1})", "INFO")
                    await ws_manager.send_log(job_id, f"Using {num_gpus_to_use} GPUs (0-{num_gpus_to_use-1})")
                else:
                    # Auto: Use all available GPUs
                    num_gpus_to_use = available_gpus
                    sanitized_log_service.create_terminal_log(job_id, f"Auto-detected {available_gpus} GPU(s). Using all.", "INFO")
                    await ws_manager.send_log(job_id, f"Auto-detected {available_gpus} GPU(s). Using all.")
                
                # Enable multi-GPU training with torchrun if more than 1 GPU
                if num_gpus_to_use > 1:
                    training_env["NPROC_PER_NODE"] = str(num_gpus_to_use)
                    sanitized_log_service.create_terminal_log(job_id, f"Multi-GPU training enabled: NPROC_PER_NODE={num_gpus_to_use}", "INFO")
                    await ws_manager.send_log(job_id, f"Multi-GPU training enabled: NPROC_PER_NODE={num_gpus_to_use}")
            else:
                sanitized_log_service.create_terminal_log(job_id, "No GPUs detected. Training will use CPU.", "WARN")
                await ws_manager.send_log(job_id, "No GPUs detected. Training will use CPU.")
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env=training_env,
            )
            _debug_log(job_id, f"Subprocess created with PID: {process.pid}")
            
            await job_manager.set_process(job_id, process)
            await job_manager.update_job(job_id, status=JobStatus.RUNNING)
            await ws_manager.send_status(job_id, "running")
            sanitized_log_service.create_terminal_log(job_id, "Training started...", "INFO")
            await ws_manager.send_log(job_id, "Training started...")
            
            total_steps = 0
            current_step = 0
            current_loss = None
            checkpoint_count = 0  # Track number of checkpoints saved
            
            # Get job timeout from system settings
            from ..core.capabilities import get_system_settings
            job_timeout_hours = get_system_settings().JOB_TIMEOUT_HOURS
            job_start_time = datetime.now()
            last_timeout_check = job_start_time
            
            # Stream output
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                
                # ============================================================
                # TIMEOUT CHECK: Auto-stop jobs that run too long
                # Check every 5 minutes to avoid overhead
                # ============================================================
                now = datetime.now()
                if (now - last_timeout_check).total_seconds() > 300:  # Check every 5 minutes
                    last_timeout_check = now
                    elapsed_hours = (now - job_start_time).total_seconds() / 3600
                    if elapsed_hours >= job_timeout_hours:
                        timeout_msg = f"Job exceeded maximum runtime ({job_timeout_hours} hours). Auto-stopping."
                        sanitized_log_service.create_terminal_log(job_id, f"TIMEOUT: {timeout_msg}", "ERROR")
                        await ws_manager.send_log(job_id, f"TIMEOUT: {timeout_msg}", "error")
                        process.terminate()
                        await job_manager.update_job(job_id, 
                            status=JobStatus.FAILED,
                            error=timeout_msg,
                            completed_at=datetime.now()
                        )
                        await ws_manager.send_status(job_id, "failed", timeout_msg)
                        return
                
                line_str = line.decode("utf-8", errors="ignore").strip()
                if not line_str:
                    continue
                
                # ============================================================
                # ENCRYPTED LOG: Full output (only US Inc can read)
                # Contains ALL details including file paths, library names, etc.
                # ============================================================
                encrypted_log_service.encrypt_and_format(line_str, job_id)
                
                # ============================================================
                # SANITIZE OUTPUT FOR USER: Hide internal details
                # - File paths (/app/..., /usr/...)
                # - Library names (transformers, torch, huggingface, etc.)
                # - Code references (line numbers, function names)
                # User gets clear, understandable messages
                # ============================================================
                sanitized_line = sanitized_log_service.sanitize_for_display(line_str)
                
                # Only log if there's something to show (sanitize_for_display returns None for filtered messages)
                if sanitized_line:
                    # ============================================================
                    # TERMINAL LOG: Write sanitized output to terminal log file
                    # This ensures logs persist even after page refresh/backend restart
                    # ============================================================
                    sanitized_log_service.create_terminal_log(job_id, sanitized_line, "OUTPUT")
                    
                    # ============================================================
                    # WEBSOCKET & IN-MEMORY: Send SANITIZED output only
                    # User sees clean, understandable logs without internal details
                    # ============================================================
                    await job_manager.add_log(job_id, sanitized_line)
                    await ws_manager.send_log(job_id, sanitized_line)
                
                # Parse metrics from output
                metrics = self._parse_log_line(line_str, job_id)
                
                if "total_steps" in metrics:
                    total_steps = metrics["total_steps"]
                if "step" in metrics:
                    current_step = metrics["step"]
                if "loss" in metrics:
                    current_loss = metrics["loss"]
                
                # ============================================================
                # CHECKPOINT DETECTION: Multiple methods for robustness
                # 1. HuggingFace Trainer: "Saving model checkpoint to ..."
                # 2. Trainer logging: checkpoint-XXX patterns
                # 3. Progress output: "Saving checkpoint" messages
                # Note: USF BIOS info_debug is suppressed in production, so we
                # also detect from HuggingFace Trainer's standard logging
                # ============================================================
                checkpoint_detected = False
                checkpoint_path = None
                
                # Method 1: Direct checkpoint message (may appear from HuggingFace Trainer)
                if "Saving model checkpoint" in line_str or "Saving checkpoint" in line_str:
                    checkpoint_detected = True
                    if " to " in line_str:
                        checkpoint_path = line_str.split(" to ")[-1].strip()
                
                # Method 2: Checkpoint directory pattern in output
                # HuggingFace Trainer outputs paths like "/path/checkpoint-500"
                elif "checkpoint-" in line_str and current_step > 0:
                    import re
                    ckpt_match = re.search(r'(checkpoint-\d+)', line_str)
                    if ckpt_match:
                        # Only count if this looks like a save operation (not just a reference)
                        if any(kw in line_str.lower() for kw in ['saving', 'saved', 'save', 'output', 'write']):
                            checkpoint_detected = True
                            checkpoint_path = ckpt_match.group(1)
                
                if checkpoint_detected:
                    checkpoint_count += 1
                    sanitized_log_service.create_terminal_log(job_id, f"✓ Checkpoint {checkpoint_count} saved: {checkpoint_path or f'step-{current_step}'}", "INFO")
                    _debug_log(job_id, f"Checkpoint {checkpoint_count} detected: {line_str}")
                    # Send checkpoint event to frontend via WebSocket
                    await ws_manager.broadcast(job_id, {
                        "type": "checkpoint",
                        "count": checkpoint_count,
                        "path": checkpoint_path,
                        "step": current_step
                    })
                
                # Update job state
                update_fields = {}
                if current_step > 0:
                    update_fields["current_step"] = current_step
                if total_steps > 0:
                    update_fields["total_steps"] = total_steps
                if current_loss is not None:
                    update_fields["current_loss"] = current_loss
                if "epoch" in metrics:
                    update_fields["current_epoch"] = int(metrics["epoch"])
                if "learning_rate" in metrics:
                    update_fields["learning_rate"] = metrics["learning_rate"]
                
                if update_fields:
                    await job_manager.update_job(job_id, **update_fields)
                
                # ============================================================
                # DATABASE: Save metrics for graphs (every step with metrics)
                # Metrics are stored PER JOB_ID - no mixing between trainings
                # Supports ALL training algorithms: SFT, RLHF, PPO, DPO, GRPO, etc.
                # ============================================================
                if "step" in metrics and len(metrics) > 1:
                    try:
                        db = SessionLocal()
                        job_service = JobService(db)
                        job_service.save_training_metric(
                            job_id=job_id,  # Ensures data isolation per training job
                            step=metrics["step"],
                            epoch=metrics.get("epoch"),
                            # Common metrics
                            loss=metrics.get("loss"),
                            learning_rate=metrics.get("learning_rate"),
                            grad_norm=metrics.get("grad_norm"),
                            eval_loss=metrics.get("eval_loss"),
                            # RLHF/PPO metrics
                            reward=metrics.get("reward"),
                            kl_divergence=metrics.get("kl_divergence"),
                            policy_loss=metrics.get("policy_loss"),
                            value_loss=metrics.get("value_loss"),
                            entropy=metrics.get("entropy"),
                            # DPO metrics
                            chosen_rewards=metrics.get("chosen_rewards"),
                            rejected_rewards=metrics.get("rejected_rewards"),
                            reward_margin=metrics.get("reward_margin"),
                            # Extra metrics (algorithm-specific in JSON)
                            extra_metrics=metrics.get("extra_metrics"),
                        )
                        db.close()
                    except Exception as e:
                        _debug_log(job_id, f"Failed to save metric: {e}", "WARNING")
                
                # ============================================================
                # TERMINAL LOG: Also save formatted progress metrics to sanitized log
                # ============================================================
                if metrics:
                    # Build minimal progress string for terminal
                    progress_parts = []
                    if "step" in metrics:
                        step_str = f"Step {metrics['step']}"
                        if total_steps > 0:
                            pct = (metrics['step'] / total_steps) * 100
                            step_str += f"/{total_steps} ({pct:.1f}%)"
                        progress_parts.append(step_str)
                    if "epoch" in metrics:
                        progress_parts.append(f"Epoch {metrics['epoch']:.2f}")
                    if "loss" in metrics:
                        progress_parts.append(f"Loss: {metrics['loss']:.4f}")
                    if "learning_rate" in metrics:
                        progress_parts.append(f"LR: {metrics['learning_rate']:.2e}")
                    
                    if progress_parts:
                        terminal_line = " | ".join(progress_parts)
                        # Only save to sanitized log file (not websocket - raw output already sent)
                        sanitized_log_service.create_terminal_log(job_id, terminal_line, "INFO")
                
                # Send progress update to frontend
                if current_step > 0 and total_steps > 0:
                    await ws_manager.send_progress(
                        job_id, current_step, total_steps,
                        loss=current_loss,
                        lr=metrics.get("learning_rate")
                    )
            
            # Wait for process to complete
            return_code = await process.wait()
            
            if return_code == 0:
                await job_manager.update_job(job_id,
                    status=JobStatus.COMPLETED,
                    completed_at=datetime.now()
                )
                await ws_manager.send_status(job_id, "completed")
                sanitized_log_service.create_terminal_log(job_id, "Training completed successfully!", "INFO")
                await ws_manager.send_log(job_id, "Training completed successfully!", "success")
                sanitized_log_service.log_session_end(job_id, "COMPLETED")
                
                # POST-TRAINING CLEANUP: Release GPU memory after successful completion
                await self._cleanup_after_training(job_id, "completed")
            else:
                # Log failure with exit code
                error_msg = f"Training failed with exit code {return_code}"
                sanitized_log_service.create_terminal_log(job_id, error_msg, "ERROR")
                
                # Try to get last few lines of output for error context
                logs = await job_manager.get_logs(job_id, last_n=10)
                if logs:
                    last_log = logs[-1] if logs else ""
                    if "No code object" in last_log or "ModuleNotFoundError" in last_log:
                        error_msg = f"Module loading error. Please rebuild the container. ({last_log[:100]})"
                    elif "CUDA" in last_log or "GPU" in last_log:
                        error_msg = "GPU/CUDA error occurred. Check GPU availability."
                    elif "out of memory" in last_log.lower():
                        error_msg = "Out of memory. Try reducing batch size."
                
                await job_manager.update_job(job_id,
                    status=JobStatus.FAILED,
                    error=error_msg,
                    completed_at=datetime.now()
                )
                await ws_manager.send_status(job_id, "failed", error_msg)
                sanitized_log_service.create_terminal_log(job_id, error_msg, "ERROR")
                await ws_manager.send_log(job_id, error_msg, "error")
                sanitized_log_service.log_session_end(job_id, "FAILED", error_message=error_msg)
                
                # POST-TRAINING CLEANUP: Release GPU memory after failure
                await self._cleanup_after_training(job_id, "failed")
        
        except asyncio.CancelledError:
            await job_manager.update_job(job_id, status=JobStatus.STOPPED)
            await ws_manager.send_status(job_id, "stopped")
            sanitized_log_service.create_terminal_log(job_id, "Training stopped by user", "WARN")
            await ws_manager.send_log(job_id, "Training stopped by user", "warning")
            sanitized_log_service.log_session_end(job_id, "CANCELLED")
            
            # POST-TRAINING CLEANUP: Release GPU memory after cancellation
            await self._cleanup_after_training(job_id, "cancelled")
        
        except Exception as e:
            # ============================================================
            # ENCRYPTED LOG: Full error details (only US Inc can read)
            # ============================================================
            import traceback
            full_error = str(e)
            full_traceback = traceback.format_exc()
            _debug_log(job_id, f"Training error: {full_error}", "ERROR")
            _debug_log(job_id, f"Traceback: {full_traceback}", "ERROR")
            encrypted_log_service.encrypt_and_format(f"FULL ERROR: {full_error}", job_id, "ERROR")
            encrypted_log_service.encrypt_and_format(f"TRACEBACK: {full_traceback}", job_id, "ERROR")
            
            # ============================================================
            # TERMINAL LOG: Minimal error (NO file names, NO code details)
            # ============================================================
            # Get minimal user-friendly error from sanitized service
            sanitized = sanitized_log_service.sanitize_error(full_error)
            minimal_error = sanitized['user_message']
            
            await job_manager.update_job(job_id,
                status=JobStatus.FAILED,
                error=minimal_error  # Store minimal error only
            )
            await ws_manager.send_status(job_id, "failed", minimal_error)
            sanitized_log_service.create_terminal_log(job_id, f"Error: {minimal_error}", "ERROR")
            await ws_manager.send_log(job_id, f"Error: {minimal_error}", "error")
            sanitized_log_service.log_session_end(job_id, "FAILED", error_message=full_error)
            
            # POST-TRAINING CLEANUP: Release GPU memory after exception
            await self._cleanup_after_training(job_id, "exception")
    
    async def _cleanup_after_training(self, job_id: str, reason: str) -> None:
        """
        Perform GPU memory cleanup after training ends.
        
        This is critical for preventing VRAM leaks between training sessions.
        Called after training completes, fails, is cancelled, or throws an exception.
        """
        try:
            _debug_log(job_id, f"Performing post-training GPU cleanup (reason: {reason})...")
            
            # Small delay to allow subprocess to fully terminate
            await asyncio.sleep(0.5)
            
            # Deep cleanup with orphan killing enabled
            cleanup_result = await gpu_cleanup_service.async_deep_cleanup(
                kill_orphans=True,
                exclude_pids=[os.getpid()]
            )
            
            if cleanup_result.get("success"):
                freed_gb = cleanup_result.get("memory_freed_gb", 0)
                mem_after = cleanup_result.get("memory_after", {})
                used_gb = mem_after.get("total_used_gb", 0)
                total_gb = mem_after.get("total_memory_gb", 0)
                
                _debug_log(job_id, f"Post-training cleanup: freed {freed_gb}GB, now using {used_gb}/{total_gb}GB")
                
                if freed_gb > 1.0:  # Only log if significant memory was freed
                    sanitized_log_service.create_terminal_log(
                        job_id,
                        f"GPU cleanup: freed {freed_gb}GB VRAM",
                        "INFO"
                    )
            else:
                _debug_log(job_id, f"Post-training cleanup warning: {cleanup_result.get('error', 'unknown')}", "WARN")
                
        except Exception as e:
            _debug_log(job_id, f"Post-training cleanup error (non-fatal): {e}", "WARN")
    
    async def start_training(self, job_id: str) -> bool:
        """Start training in background"""
        _debug_log(job_id, "start_training called")
        
        job = await job_manager.get_job(job_id)
        if not job:
            _debug_log(job_id, "Job not found", "ERROR")
            return False
        if job.status == JobStatus.RUNNING:
            _debug_log(job_id, "Job already running", "WARNING")
            return False
        
        # =====================================================================
        # CRITICAL SECURITY: Validate model and output path before training
        # This ensures only authorized models/paths can be used - CANNOT be bypassed
        # Even if someone modifies the frontend, this backend check blocks them
        # =====================================================================
        try:
            from ..core.capabilities import get_validator
            validator = get_validator()
            
            # Validate model path
            model_path = job.config.model_path
            model_source = getattr(job.config, 'model_source', 'local')
            if model_path.startswith('/') or model_path.startswith('./'):
                model_source = 'local'
            
            is_valid, error_msg = validator.validate_model_path(model_path, model_source)
            if not is_valid:
                _debug_log(job_id, f"Model validation failed", "ERROR")
                await job_manager.fail_job(job_id, "Invalid configuration")
                return False
            
            _debug_log(job_id, f"Model validation passed: {model_path}")
            
            # Validate output path (when locked, user cannot customize)
            user_output_dir = getattr(job.config, 'output_dir', '') or ''
            is_valid, error_msg = validator.validate_output_path(user_output_dir)
            if not is_valid:
                _debug_log(job_id, f"Output path validation failed", "ERROR")
                await job_manager.fail_job(job_id, "Invalid configuration")
                return False
            
            _debug_log(job_id, f"Output path validation passed")
        except Exception as e:
            _debug_log(job_id, f"Validation error: {e}", "ERROR")
            await job_manager.fail_job(job_id, "Invalid configuration")
            return False
        
        _debug_log(job_id, "Creating background task...")
        task = asyncio.create_task(self.run_training(job_id))
        self._running_tasks[job_id] = task
        _debug_log(job_id, "Background task created")
        return True
    
    async def stop_training(self, job_id: str) -> bool:
        """Stop a running training"""
        if job_id in self._running_tasks:
            self._running_tasks[job_id].cancel()
            del self._running_tasks[job_id]
        
        return await job_manager.stop_job(job_id)


# Global instance
training_service = TrainingService()
