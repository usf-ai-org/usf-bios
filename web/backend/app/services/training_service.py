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
from ..models.schemas import JobInfo, JobStatus, TrainingConfig
from .job_manager import job_manager
from .websocket_manager import ws_manager


class TrainingService:
    """Service for running training jobs"""
    
    def __init__(self):
        self._running_tasks: dict = {}
    
    def _build_command(self, config: TrainingConfig, job_id: str) -> list:
        """Build the training command"""
        output_dir = str(settings.OUTPUT_DIR / job_id)
        
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
        job = await job_manager.get_job(job_id)
        if not job:
            return
        
        try:
            # Update status
            await job_manager.update_job(job_id, 
                status=JobStatus.INITIALIZING,
                started_at=datetime.now()
            )
            await ws_manager.send_status(job_id, "initializing")
            await ws_manager.send_log(job_id, "Initializing training environment...")
            
            # Build command
            cmd = self._build_command(job.config, job_id)
            cmd_str = " ".join(cmd)
            
            await job_manager.add_log(job_id, f"Command: {cmd_str}")
            await ws_manager.send_log(job_id, f"Starting training with command:")
            await ws_manager.send_log(job_id, cmd_str, "command")
            
            # Create process
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                env={**os.environ, "PYTHONUNBUFFERED": "1"},
            )
            
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
                
                # Store log
                await job_manager.add_log(job_id, line_str)
                
                # Parse metrics
                metrics = self._parse_log_line(line_str, job_id)
                
                if "total_steps" in metrics:
                    total_steps = metrics["total_steps"]
                if "step" in metrics:
                    current_step = metrics["step"]
                if "loss" in metrics:
                    current_loss = metrics["loss"]
                
                # Update job
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
                
                # Send real-time updates
                await ws_manager.send_log(job_id, line_str)
                
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
                
                output_dir = str(settings.OUTPUT_DIR / job_id)
                await ws_manager.send_log(job_id, f"Output saved to: {output_dir}", "success")
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
        
        except Exception as e:
            error_msg = str(e)
            await job_manager.update_job(job_id,
                status=JobStatus.FAILED,
                error=error_msg
            )
            await ws_manager.send_status(job_id, "failed", error_msg)
            await ws_manager.send_log(job_id, f"Error: {error_msg}", "error")
    
    async def start_training(self, job_id: str) -> bool:
        """Start training in background"""
        job = await job_manager.get_job(job_id)
        if not job or job.status == JobStatus.RUNNING:
            return False
        
        task = asyncio.create_task(self.run_training(job_id))
        self._running_tasks[job_id] = task
        return True
    
    async def stop_training(self, job_id: str) -> bool:
        """Stop a running training"""
        if job_id in self._running_tasks:
            self._running_tasks[job_id].cancel()
            del self._running_tasks[job_id]
        
        return await job_manager.stop_job(job_id)


# Global instance
training_service = TrainingService()
