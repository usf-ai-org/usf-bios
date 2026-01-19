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
from ..models.schemas import JobInfo, JobStatus, TrainingConfig
from .job_manager import job_manager
from .websocket_manager import ws_manager
from .sanitized_log_service import sanitized_log_service
from .encrypted_log_service import encrypted_log_service


def _debug_log(job_id: str, message: str, level: str = "DEBUG"):
    """Write debug log to ENCRYPTED log file (only US Inc can read)."""
    encrypted_log_service.encrypt_and_format(f"[{level}] {message}", job_id, level)


class TrainingService:
    """Service for running training jobs"""
    
    def __init__(self):
        self._running_tasks: dict = {}
    
    def _build_command(self, config: TrainingConfig, job_id: str) -> list:
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
        ]
        
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
        """Parse a log line to extract training metrics"""
        metrics = {}
        
        # Parse loss
        loss_match = re.search(r"'loss':\s*([\d.]+)", line)
        if loss_match:
            metrics["loss"] = float(loss_match.group(1))
        
        # Parse learning rate
        lr_match = re.search(r"'learning_rate':\s*([\d.e-]+)", line)
        if lr_match:
            metrics["learning_rate"] = float(lr_match.group(1))
        
        # Parse step
        step_match = re.search(r"'(global_)?step':\s*(\d+)", line)
        if step_match:
            metrics["step"] = int(step_match.group(2))
        
        # Parse epoch
        epoch_match = re.search(r"'epoch':\s*([\d.]+)", line)
        if epoch_match:
            metrics["epoch"] = float(epoch_match.group(1))
        
        # Parse progress bar format: X/Y [time<eta]
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
            await ws_manager.send_log(job_id, "Initializing training environment...")
            _debug_log(job_id, "Status set to INITIALIZING")
            
            # Build command
            cmd = self._build_command(job.config, job_id)
            cmd_str = " ".join(cmd)
            _debug_log(job_id, f"Built command: {cmd_str}")
            
            # Store full command in encrypted log only (for US Inc)
            encrypted_log_service.encrypt_and_format(f"Command: {cmd_str}", job_id)
            
            # Send command info to WebSocket for debugging
            await ws_manager.send_log(job_id, f"Command: {cmd_str}")
            sanitized_log_service.create_terminal_log(job_id, f"Command: {cmd_str}", "INFO")
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
            await ws_manager.send_log(job_id, "Training started...")
            
            total_steps = 0
            current_step = 0
            current_loss = None
            
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
                
                # Parse metrics from output
                metrics = self._parse_log_line(line_str, job_id)
                
                if "total_steps" in metrics:
                    total_steps = metrics["total_steps"]
                if "step" in metrics:
                    current_step = metrics["step"]
                if "loss" in metrics:
                    current_loss = metrics["loss"]
                
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
                # TERMINAL LOG: Minimal - only progress metrics (no sensitive info)
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
                        await job_manager.add_log(job_id, terminal_line)
                        await ws_manager.send_log(job_id, terminal_line)
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
                await ws_manager.send_log(job_id, "Training completed successfully!", "success")
                
                # Don't expose full path to user
                await ws_manager.send_log(job_id, "Training completed! Output saved.", "success")
                sanitized_log_service.log_session_end(job_id, "COMPLETED")
            else:
                await job_manager.update_job(job_id,
                    status=JobStatus.FAILED,
                    error=f"Process exited with code {return_code}"
                )
                await ws_manager.send_status(job_id, "failed", f"Exit code: {return_code}")
                await ws_manager.send_log(job_id, f"Training failed with exit code {return_code}", "error")
        
        except asyncio.CancelledError:
            await job_manager.update_job(job_id, status=JobStatus.STOPPED)
            await ws_manager.send_status(job_id, "stopped")
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
