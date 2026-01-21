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


def _debug_log(job_id: str, message: str, level: str = "DEBUG"):
    """Write debug log to ENCRYPTED log file (only US Inc can read)."""
    encrypted_log_service.encrypt_and_format(f"[{level}] {message}", job_id, level)


class TrainingService:
    """Service for running training jobs"""
    
    def __init__(self):
        self._running_tasks: dict = {}
    
    def _build_command(self, config: TrainingConfig, job_id: str, resume_from_checkpoint: str = None) -> list:
        """Build the training command"""
        output_dir = str(get_system_settings().OUTPUT_DIR / job_id)
        
        cmd = [
            sys.executable, "-m", "usf_bios", "sft",
            "--model", config.model_path,
            "--train_type", config.train_type.value,
            "--dataset", config.dataset_path,
            "--output_dir", output_dir,
            "--num_train_epochs", str(config.num_train_epochs),
            "--learning_rate", str(config.learning_rate),
            "--per_device_train_batch_size", str(config.per_device_train_batch_size),
            "--gradient_accumulation_steps", str(config.gradient_accumulation_steps),
            "--max_length", str(config.max_length),
            "--torch_dtype", config.torch_dtype,
            "--logging_steps", "1",
            "--report_to", "tensorboard",  # Enable TensorBoard logging
        ]
        
        # Resume from checkpoint
        if resume_from_checkpoint:
            cmd.extend(["--resume_from_checkpoint", resume_from_checkpoint])
        
        # LoRA parameters
        if config.train_type.value in ["lora", "qlora", "adalora"]:
            cmd.extend([
                "--lora_rank", str(config.lora_rank),
                "--lora_alpha", str(config.lora_alpha),
                "--lora_dropout", str(config.lora_dropout),
                "--target_modules", config.target_modules,
            ])
        
        # QLoRA quantization
        if config.quant_bits:
            cmd.extend(["--quant_bits", str(config.quant_bits)])
        
        # DeepSpeed
        if config.deepspeed:
            cmd.extend(["--deepspeed", config.deepspeed])
        
        # FSDP
        if config.fsdp:
            cmd.extend(["--fsdp", config.fsdp])
        
        # Early stopping
        if config.early_stop_interval:
            cmd.extend(["--early_stop_interval", str(config.early_stop_interval)])
        
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
        
        # Parse step
        step_match = re.search(r"'(global_)?step':\s*(\d+)", line)
        if step_match:
            metrics["step"] = int(step_match.group(2))
        else:
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
            
            sanitized_log_service.create_terminal_log(job_id, "Validation passed. Building training command...", "INFO")
            await ws_manager.send_log(job_id, "Validation passed. Building training command...")
            
            # Build command (with resume checkpoint if set)
            resume_checkpoint = getattr(job, 'resume_from_checkpoint', None)
            if resume_checkpoint:
                sanitized_log_service.create_terminal_log(job_id, f"Resuming from checkpoint: {resume_checkpoint}", "INFO")
                await ws_manager.send_log(job_id, f"Resuming from checkpoint: {resume_checkpoint}")
                _debug_log(job_id, f"Resuming from checkpoint: {resume_checkpoint}")
            
            cmd = self._build_command(job.config, job_id, resume_from_checkpoint=resume_checkpoint)
            cmd_str = " ".join(cmd)
            _debug_log(job_id, f"Built command: {cmd_str}")
            
            # Store full command in encrypted log only (for US Inc)
            encrypted_log_service.encrypt_and_format(f"Command: {cmd_str}", job_id)
            
            # Send command info to WebSocket for debugging
            sanitized_log_service.create_terminal_log(job_id, f"Command: {cmd_str}", "INFO")
            await ws_manager.send_log(job_id, f"Command: {cmd_str}")
            sanitized_log_service.create_terminal_log(job_id, "Starting training process...", "INFO")
            await ws_manager.send_log(job_id, "Starting training process...")
            
            # Create process
            _debug_log(job_id, "Creating subprocess...")
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
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
            
            # Stream output
            while True:
                line = await process.stdout.readline()
                if not line:
                    break
                
                line_str = line.decode("utf-8", errors="ignore").strip()
                if not line_str:
                    continue
                
                # ============================================================
                # ENCRYPTED LOG: Full output (only US Inc can read)
                # ============================================================
                encrypted_log_service.encrypt_and_format(line_str, job_id)
                
                # ============================================================
                # TERMINAL LOG: Write raw output to terminal log file
                # This ensures logs persist even after page refresh/backend restart
                # ============================================================
                sanitized_log_service.create_terminal_log(job_id, line_str, "OUTPUT")
                
                # ============================================================
                # SHOW ALL OUTPUT IN TERMINAL (for debugging training issues)
                # This helps users see what's happening, especially errors
                # ============================================================
                # Always add raw output to job logs and send to websocket
                await job_manager.add_log(job_id, line_str)
                await ws_manager.send_log(job_id, line_str)
                
                # Parse metrics from output
                metrics = self._parse_log_line(line_str, job_id)
                
                if "total_steps" in metrics:
                    total_steps = metrics["total_steps"]
                if "step" in metrics:
                    current_step = metrics["step"]
                if "loss" in metrics:
                    current_loss = metrics["loss"]
                
                # ============================================================
                # CHECKPOINT DETECTION: Detect when checkpoints are saved
                # HuggingFace Trainer outputs: "Saving model checkpoint to ..."
                # ============================================================
                if "Saving model checkpoint" in line_str:
                    checkpoint_count += 1
                    # Extract checkpoint path from log line
                    # Format: "Saving model checkpoint to /path/to/checkpoint-XXX"
                    checkpoint_path = None
                    if " to " in line_str:
                        checkpoint_path = line_str.split(" to ")[-1].strip()
                    sanitized_log_service.create_terminal_log(job_id, f"✓ Checkpoint {checkpoint_count} saved: {checkpoint_path or 'unknown'}", "INFO")
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
        
        except asyncio.CancelledError:
            await job_manager.update_job(job_id, status=JobStatus.STOPPED)
            await ws_manager.send_status(job_id, "stopped")
            sanitized_log_service.create_terminal_log(job_id, "Training stopped by user", "WARN")
            await ws_manager.send_log(job_id, "Training stopped by user", "warning")
            sanitized_log_service.log_session_end(job_id, "CANCELLED")
        
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
