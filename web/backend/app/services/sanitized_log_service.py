import re
import os
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum


class CrashReason(str, Enum):
    GPU_OUT_OF_MEMORY = "gpu_out_of_memory"
    GPU_ERROR = "gpu_error"
    MEMORY_ERROR = "memory_error"
    SERVER_RESTART = "server_restart"
    CONFIG_ERROR = "config_error"
    DATASET_ERROR = "dataset_error"
    MODEL_ERROR = "model_error"
    TIMEOUT = "timeout"
    USER_CANCELLED = "user_cancelled"
    UNKNOWN = "unknown"


class ErrorSeverity(str, Enum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


SENSITIVE_PATTERNS = [
    # File paths and line numbers
    r'File\s+"[^"]*\.py[c]?"',
    r'line\s+\d+',
    r'in\s+\w+\s*\(',
    r'/app/[^\s]+\.py[c]?',
    r'/usr/[^\s]+\.py[c]?',
    r'/home/[^\s]+',
    r'/workspace/[^\s]+\.py[c]?',
    r'/usr/local/lib/[^\s]+',
    r'/site-packages/[^\s]+',
    # Code patterns
    r'def\s+\w+',
    r'class\s+\w+',
    r'import\s+\w+',
    r'from\s+\w+\s+import',
    r'Traceback\s+\(most recent call last\)',
    r'raise\s+\w+',
    r'assert\s+',
    r'\.forward\(',
    r'\.backward\(',
    r'self\.\w+',
    r'__\w+__',
    r'0x[0-9a-fA-F]+',
    r'at\s+0x[0-9a-fA-F]+',
    # Library names to hide (don't expose what libraries we use)
    r'transformers\.[\w\.]+',
    r'torch\.[\w\.]+',
    r'huggingface[\w\.]*',
    r'peft\.[\w\.]+',
    r'trl\.[\w\.]+',
    r'deepspeed\.[\w\.]+',
    r'accelerate\.[\w\.]+',
    r'bitsandbytes\.[\w\.]+',
    r'datasets\.[\w\.]+',
    r'tokenizers\.[\w\.]+',
    r'safetensors\.[\w\.]+',
    r'usf_bios\.[\w\.]+',
    r'app\.[\w\.]+',
    # Internal module references
    r'HuggingFace',
    r'Hugging Face',
    r'HfArgumentParser',
    r'TrainingArguments',
    r'Seq2SeqTrainer',
    r'Trainer\.',
]

SAFE_ERROR_PATTERNS = {
    r'CUDA out of memory': {
        'reason': CrashReason.GPU_OUT_OF_MEMORY,
        'user_message': 'GPU ran out of memory. Try reducing batch size or using a smaller model.',
        'severity': ErrorSeverity.ERROR
    },
    r'CUDA error': {
        'reason': CrashReason.GPU_ERROR,
        'user_message': 'GPU error occurred. The GPU may be overloaded or unavailable.',
        'severity': ErrorSeverity.ERROR
    },
    r'out of memory': {
        'reason': CrashReason.MEMORY_ERROR,
        'user_message': 'System ran out of memory. Try reducing batch size.',
        'severity': ErrorSeverity.ERROR
    },
    r'OOM': {
        'reason': CrashReason.GPU_OUT_OF_MEMORY,
        'user_message': 'Out of memory error. Try reducing batch size or model size.',
        'severity': ErrorSeverity.ERROR
    },
    r'RuntimeError.*CUDA': {
        'reason': CrashReason.GPU_ERROR,
        'user_message': 'GPU runtime error. Please check GPU availability.',
        'severity': ErrorSeverity.ERROR
    },
    r'torch\.cuda\.OutOfMemoryError': {
        'reason': CrashReason.GPU_OUT_OF_MEMORY,
        'user_message': 'GPU memory exhausted. Reduce batch size or use gradient checkpointing.',
        'severity': ErrorSeverity.ERROR
    },
    r'Connection refused': {
        'reason': CrashReason.SERVER_RESTART,
        'user_message': 'Connection lost. Server may have restarted.',
        'severity': ErrorSeverity.WARNING
    },
    r'KeyError.*config': {
        'reason': CrashReason.CONFIG_ERROR,
        'user_message': 'Configuration error. Please check your training settings.',
        'severity': ErrorSeverity.ERROR
    },
    r'FileNotFoundError.*dataset': {
        'reason': CrashReason.DATASET_ERROR,
        'user_message': 'Dataset file not found. Please verify the dataset path.',
        'severity': ErrorSeverity.ERROR
    },
    r'Invalid.*format': {
        'reason': CrashReason.DATASET_ERROR,
        'user_message': 'Invalid data format. Please check your dataset format.',
        'severity': ErrorSeverity.ERROR
    },
    r'model.*not found': {
        'reason': CrashReason.MODEL_ERROR,
        'user_message': 'Model not found. Please verify the model path or name.',
        'severity': ErrorSeverity.ERROR
    },
    r'Timeout': {
        'reason': CrashReason.TIMEOUT,
        'user_message': 'Operation timed out. Please try again.',
        'severity': ErrorSeverity.WARNING
    },
    r'gradient.*nan': {
        'reason': CrashReason.CONFIG_ERROR,
        'user_message': 'Training became unstable (NaN gradients). Try lowering learning rate.',
        'severity': ErrorSeverity.ERROR
    },
    r'loss.*nan': {
        'reason': CrashReason.CONFIG_ERROR,
        'user_message': 'Training became unstable (NaN loss). Try lowering learning rate.',
        'severity': ErrorSeverity.ERROR
    },
    r'CUDA_HOME does not exist': {
        'reason': CrashReason.GPU_ERROR,
        'user_message': 'GPU configuration error. CUDA toolkit not properly configured.',
        'severity': ErrorSeverity.ERROR
    },
    r'unable to compile CUDA': {
        'reason': CrashReason.GPU_ERROR,
        'user_message': 'GPU compilation error. Please check GPU drivers.',
        'severity': ErrorSeverity.ERROR
    },
    r'MissingCUDAException': {
        'reason': CrashReason.GPU_ERROR,
        'user_message': 'CUDA not available. Please verify GPU setup.',
        'severity': ErrorSeverity.ERROR
    },
    r'libcudart.*not found': {
        'reason': CrashReason.GPU_ERROR,
        'user_message': 'CUDA runtime library not found. Check CUDA installation.',
        'severity': ErrorSeverity.ERROR
    },
    r'No CUDA GPUs are available': {
        'reason': CrashReason.GPU_ERROR,
        'user_message': 'No GPU available. Please check GPU status.',
        'severity': ErrorSeverity.ERROR
    },
    # Adapter loading errors
    r'adapter.*config.*mismatch': {
        'reason': CrashReason.CONFIG_ERROR,
        'user_message': 'Adapter configuration mismatch. The adapter was trained with a different model.',
        'severity': ErrorSeverity.ERROR
    },
    r'size mismatch': {
        'reason': CrashReason.CONFIG_ERROR,
        'user_message': 'Adapter weight size mismatch. The adapter is not compatible with this base model.',
        'severity': ErrorSeverity.ERROR
    },
    r'shape.*mismatch': {
        'reason': CrashReason.CONFIG_ERROR,
        'user_message': 'Adapter shape mismatch. The adapter is not compatible with this base model.',
        'severity': ErrorSeverity.ERROR
    },
    r'adapter.*not found': {
        'reason': CrashReason.MODEL_ERROR,
        'user_message': 'Adapter not found. Please verify the adapter path.',
        'severity': ErrorSeverity.ERROR
    },
    r'adapter_config\.json': {
        'reason': CrashReason.CONFIG_ERROR,
        'user_message': 'Invalid adapter: missing adapter_config.json file.',
        'severity': ErrorSeverity.ERROR
    },
    r'adapter_model': {
        'reason': CrashReason.MODEL_ERROR,
        'user_message': 'Invalid adapter: missing adapter weights file.',
        'severity': ErrorSeverity.ERROR
    },
    r'peft.*error': {
        'reason': CrashReason.CONFIG_ERROR,
        'user_message': 'PEFT adapter loading error. The adapter may be corrupted or incompatible.',
        'severity': ErrorSeverity.ERROR
    },
    r'target_modules': {
        'reason': CrashReason.CONFIG_ERROR,
        'user_message': 'Adapter target modules not found in base model. Model architecture mismatch.',
        'severity': ErrorSeverity.ERROR
    },
}

SAFE_KEYWORDS = [
    'epoch', 'step', 'loss', 'accuracy', 'learning_rate', 'lr',
    'batch', 'samples', 'training', 'evaluation', 'eval',
    'checkpoint', 'saving', 'loading', 'progress',
    'gpu', 'cuda', 'memory', 'vram', 'ram',
    'model', 'dataset', 'config', 'parameter',
    'started', 'completed', 'finished', 'stopped', 'cancelled',
    'resume', 'resuming', 'error', 'warning', 'info',
    'time', 'duration', 'eta', 'remaining',
    'gradient', 'optimizer', 'scheduler',
]


class SanitizedLogService:
    
    TERMINAL_LOG_BASE_PATH = os.getenv("TERMINAL_LOG_PATH", "/app/data/terminal_logs")
    
    def __init__(self):
        os.makedirs(self.TERMINAL_LOG_BASE_PATH, exist_ok=True)
    
    def sanitize_error(self, error_message: str) -> Dict[str, Any]:
        if not error_message:
            return {
                'original_contains_sensitive': False,
                'sanitized_message': 'An error occurred.',
                'reason': CrashReason.UNKNOWN,
                'user_message': 'An unexpected error occurred.',
                'severity': ErrorSeverity.ERROR,
                'safe_to_display': True
            }
        
        detected_reason = CrashReason.UNKNOWN
        user_message = None
        severity = ErrorSeverity.ERROR
        
        for pattern, info in SAFE_ERROR_PATTERNS.items():
            if re.search(pattern, error_message, re.IGNORECASE):
                detected_reason = info['reason']
                user_message = info['user_message']
                severity = info['severity']
                break
        
        contains_sensitive = self._contains_sensitive_info(error_message)
        
        if contains_sensitive:
            sanitized = self._sanitize_message(error_message)
        else:
            sanitized = self._extract_safe_parts(error_message)
        
        if not user_message:
            user_message = self._generate_generic_message(detected_reason)
        
        return {
            'original_contains_sensitive': contains_sensitive,
            'sanitized_message': sanitized,
            'reason': detected_reason,
            'user_message': user_message,
            'severity': severity,
            'safe_to_display': not contains_sensitive or sanitized != error_message
        }
    
    def _contains_sensitive_info(self, message: str) -> bool:
        for pattern in SENSITIVE_PATTERNS:
            if re.search(pattern, message, re.IGNORECASE):
                return True
        return False
    
    def _sanitize_message(self, message: str) -> str:
        sanitized = message
        
        for pattern in SENSITIVE_PATTERNS:
            sanitized = re.sub(pattern, '[REDACTED]', sanitized, flags=re.IGNORECASE)
        
        sanitized = re.sub(r'\[REDACTED\](\s*\[REDACTED\])+', '[REDACTED]', sanitized)
        
        sanitized = re.sub(r'^\s*\[REDACTED\]\s*$', '', sanitized, flags=re.MULTILINE)
        
        lines = [line.strip() for line in sanitized.split('\n') if line.strip() and line.strip() != '[REDACTED]']
        sanitized = '\n'.join(lines[:5])
        
        if not sanitized or sanitized == '[REDACTED]':
            return 'An internal error occurred.'
        
        return sanitized
    
    def sanitize_for_display(self, message: str) -> str:
        """Sanitize a log message for user display.
        
        This method:
        1. Keeps useful training info (loss, step, epoch, progress bars)
        2. Hides internal details (file paths, library names, code references)
        3. Converts errors to user-friendly messages
        
        Returns the sanitized message suitable for terminal display.
        """
        if not message:
            return message
        
        # Check if this is an error/traceback - convert to user-friendly message
        for pattern, info in SAFE_ERROR_PATTERNS.items():
            if re.search(pattern, message, re.IGNORECASE):
                return f"⚠ {info['user_message']}"
        
        # Check if message contains sensitive info that needs sanitization
        if self._contains_sensitive_info(message):
            # For traceback lines, just skip them entirely
            if 'Traceback' in message or 'File "' in message or 'line ' in message:
                return None  # Will be filtered out
            
            # For other sensitive messages, sanitize
            sanitized = self._sanitize_message(message)
            if sanitized and sanitized != 'An internal error occurred.':
                return sanitized
            return None  # Filter out completely redacted messages
        
        # Message is safe - return as-is
        return message
    
    def _extract_safe_parts(self, message: str) -> str:
        safe_parts = []
        
        for line in message.split('\n'):
            line_lower = line.lower()
            if any(keyword in line_lower for keyword in SAFE_KEYWORDS):
                if not self._contains_sensitive_info(line):
                    safe_parts.append(line.strip())
        
        if safe_parts:
            return '\n'.join(safe_parts[:10])
        
        return 'An error occurred during training.'
    
    def _generate_generic_message(self, reason: CrashReason) -> str:
        messages = {
            CrashReason.GPU_OUT_OF_MEMORY: 'GPU ran out of memory. Try reducing batch size.',
            CrashReason.GPU_ERROR: 'A GPU error occurred. Please check GPU status.',
            CrashReason.MEMORY_ERROR: 'System memory error. Try reducing batch size.',
            CrashReason.SERVER_RESTART: 'Server was restarted. You can resume training.',
            CrashReason.CONFIG_ERROR: 'Configuration error. Please check your settings.',
            CrashReason.DATASET_ERROR: 'Dataset error. Please verify your dataset.',
            CrashReason.MODEL_ERROR: 'Model error. Please verify the model.',
            CrashReason.TIMEOUT: 'Operation timed out.',
            CrashReason.USER_CANCELLED: 'Training was cancelled by user.',
            CrashReason.UNKNOWN: 'An unexpected error occurred.',
        }
        return messages.get(reason, 'An error occurred.')
    
    def detect_crash_reason(self, error_message: str = None, last_gpu_info: Dict = None) -> Dict[str, Any]:
        reason = CrashReason.UNKNOWN
        details = {}
        
        if error_message:
            for pattern, info in SAFE_ERROR_PATTERNS.items():
                if re.search(pattern, error_message, re.IGNORECASE):
                    reason = info['reason']
                    break
        
        if last_gpu_info:
            memory_used = last_gpu_info.get('memory_used_mb', 0)
            memory_total = last_gpu_info.get('memory_total_mb', 1)
            utilization = memory_used / memory_total if memory_total > 0 else 0
            
            if utilization > 0.95:
                reason = CrashReason.GPU_OUT_OF_MEMORY
                details['gpu_memory_utilization'] = f"{utilization * 100:.1f}%"
        
        return {
            'reason': reason,
            'reason_display': self._get_reason_display(reason),
            'can_resume': reason in [
                CrashReason.GPU_OUT_OF_MEMORY,
                CrashReason.GPU_ERROR,
                CrashReason.MEMORY_ERROR,
                CrashReason.SERVER_RESTART,
                CrashReason.USER_CANCELLED,
            ],
            'suggestion': self._get_suggestion(reason),
            'details': details
        }
    
    def _get_reason_display(self, reason: CrashReason) -> str:
        display_names = {
            CrashReason.GPU_OUT_OF_MEMORY: 'GPU Memory Full',
            CrashReason.GPU_ERROR: 'GPU Error',
            CrashReason.MEMORY_ERROR: 'Memory Error',
            CrashReason.SERVER_RESTART: 'Server Restarted',
            CrashReason.CONFIG_ERROR: 'Configuration Error',
            CrashReason.DATASET_ERROR: 'Dataset Error',
            CrashReason.MODEL_ERROR: 'Model Error',
            CrashReason.TIMEOUT: 'Timeout',
            CrashReason.USER_CANCELLED: 'Cancelled',
            CrashReason.UNKNOWN: 'Unknown Error',
        }
        return display_names.get(reason, 'Unknown')
    
    def _get_suggestion(self, reason: CrashReason) -> str:
        suggestions = {
            CrashReason.GPU_OUT_OF_MEMORY: 'Try reducing batch size or using gradient checkpointing.',
            CrashReason.GPU_ERROR: 'Check GPU availability and restart if needed.',
            CrashReason.MEMORY_ERROR: 'Reduce batch size or close other applications.',
            CrashReason.SERVER_RESTART: 'You can resume training from the last checkpoint.',
            CrashReason.CONFIG_ERROR: 'Review your training configuration settings.',
            CrashReason.DATASET_ERROR: 'Verify your dataset file exists and is valid.',
            CrashReason.MODEL_ERROR: 'Check the model name/path is correct.',
            CrashReason.TIMEOUT: 'Try again or check network connectivity.',
            CrashReason.USER_CANCELLED: 'You can resume or start fresh.',
            CrashReason.UNKNOWN: 'Try resuming from the last checkpoint.',
        }
        return suggestions.get(reason, 'Try again or contact support.')
    
    def create_terminal_log(self, job_id: str, message: str, level: str = "INFO") -> str:
        if self._contains_sensitive_info(message):
            message = self._sanitize_message(message)
        
        timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] [{level}] {message}"
        
        log_file = os.path.join(self.TERMINAL_LOG_BASE_PATH, f"{job_id}.terminal.log")
        
        try:
            with open(log_file, 'a') as f:
                f.write(log_line + '\n')
                f.flush()  # Force flush to disk for real-time reading
                os.fsync(f.fileno())  # Ensure OS writes to disk
        except Exception:
            pass
        
        return log_line
    
    def log_session_start(self, job_id: str, session_type: str = "START", 
                          resume_from_step: int = None, config_summary: Dict = None) -> None:
        separator = "=" * 60
        self.create_terminal_log(job_id, separator, "INFO")
        
        if session_type == "RESUME":
            self.create_terminal_log(job_id, f"SESSION: RESUME TRAINING (from step {resume_from_step})", "INFO")
        else:
            self.create_terminal_log(job_id, "SESSION: NEW TRAINING STARTED", "INFO")
        
        self.create_terminal_log(job_id, f"Time: {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}", "INFO")
        
        if config_summary:
            safe_config = {
                "model": config_summary.get("model_name", "N/A"),
                "batch_size": config_summary.get("batch_size", "N/A"),
                "learning_rate": config_summary.get("learning_rate", "N/A"),
                "epochs": config_summary.get("num_epochs", "N/A"),
                "max_steps": config_summary.get("max_steps", "N/A"),
            }
            self.create_terminal_log(job_id, f"Config: {safe_config}", "INFO")
        
        self.create_terminal_log(job_id, separator, "INFO")
    
    def log_session_end(self, job_id: str, status: str, duration_seconds: int = None,
                        final_loss: float = None, error_message: str = None) -> None:
        separator = "=" * 60
        self.create_terminal_log(job_id, separator, "INFO")
        self.create_terminal_log(job_id, f"SESSION ENDED: {status.upper()}", "INFO")
        
        if duration_seconds:
            hours, remainder = divmod(duration_seconds, 3600)
            minutes, seconds = divmod(remainder, 60)
            self.create_terminal_log(job_id, f"Duration: {int(hours)}h {int(minutes)}m {int(seconds)}s", "INFO")
        
        if final_loss is not None:
            self.create_terminal_log(job_id, f"Final Loss: {final_loss:.4f}", "INFO")
        
        if error_message:
            sanitized = self.sanitize_error(error_message)
            self.create_terminal_log(job_id, f"Reason: {sanitized['user_message']}", "ERROR")
            self.create_terminal_log(job_id, f"Suggestion: {self._get_suggestion(sanitized['reason'])}", "INFO")
        
        self.create_terminal_log(job_id, separator, "INFO")
    
    def log_gpu_status(self, job_id: str, gpu_info: Dict) -> None:
        if not gpu_info:
            return
        
        parts = []
        
        if "name" in gpu_info:
            parts.append(f"GPU: {gpu_info['name']}")
        
        if "memory_used_mb" in gpu_info and "memory_total_mb" in gpu_info:
            used = gpu_info["memory_used_mb"]
            total = gpu_info["memory_total_mb"]
            pct = (used / total * 100) if total > 0 else 0
            parts.append(f"VRAM: {used:.0f}/{total:.0f}MB ({pct:.1f}%)")
        
        if "temperature" in gpu_info:
            parts.append(f"Temp: {gpu_info['temperature']}°C")
        
        if "utilization" in gpu_info:
            parts.append(f"Util: {gpu_info['utilization']}%")
        
        if "power_draw" in gpu_info and "power_limit" in gpu_info:
            parts.append(f"Power: {gpu_info['power_draw']:.0f}/{gpu_info['power_limit']:.0f}W")
        
        if parts:
            self.create_terminal_log(job_id, " | ".join(parts), "INFO")
    
    def log_training_metrics(self, job_id: str, metrics: Dict) -> None:
        parts = []
        
        if "step" in metrics:
            total = metrics.get("total_steps", "?")
            parts.append(f"Step {metrics['step']}/{total}")
        
        if "epoch" in metrics:
            parts.append(f"Epoch {metrics['epoch']:.2f}")
        
        if "loss" in metrics:
            parts.append(f"Loss: {metrics['loss']:.4f}")
        
        if "eval_loss" in metrics:
            parts.append(f"Eval Loss: {metrics['eval_loss']:.4f}")
        
        if "accuracy" in metrics:
            parts.append(f"Acc: {metrics['accuracy']:.2%}")
        
        if "learning_rate" in metrics:
            parts.append(f"LR: {metrics['learning_rate']:.2e}")
        
        if "grad_norm" in metrics:
            parts.append(f"Grad: {metrics['grad_norm']:.4f}")
        
        if "throughput" in metrics:
            parts.append(f"Speed: {metrics['throughput']:.1f} samples/s")
        
        if "eta_seconds" in metrics:
            eta = metrics["eta_seconds"]
            hours, remainder = divmod(eta, 3600)
            minutes, _ = divmod(remainder, 60)
            parts.append(f"ETA: {int(hours)}h {int(minutes)}m")
        
        if parts:
            self.create_terminal_log(job_id, " | ".join(parts), "INFO")
    
    def log_checkpoint_saved(self, job_id: str, step: int, path: str = None, 
                             is_best: bool = False) -> None:
        if is_best:
            msg = f"Checkpoint saved at step {step} (BEST)"
        else:
            msg = f"Checkpoint saved at step {step}"
        self.create_terminal_log(job_id, msg, "INFO")
    
    def log_system_info(self, job_id: str) -> None:
        try:
            import psutil
            
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            parts = [
                f"CPU: {cpu_percent}%",
                f"RAM: {memory.used / (1024**3):.1f}/{memory.total / (1024**3):.1f}GB ({memory.percent}%)"
            ]
            
            self.create_terminal_log(job_id, " | ".join(parts), "INFO")
        except ImportError:
            pass
        except Exception:
            pass
    
    def get_terminal_logs(self, job_id: str, lines: int = 100) -> List[str]:
        log_file = os.path.join(self.TERMINAL_LOG_BASE_PATH, f"{job_id}.terminal.log")
        
        if not os.path.exists(log_file):
            return []
        
        try:
            with open(log_file, 'r') as f:
                all_lines = f.readlines()
                return [line.strip() for line in all_lines[-lines:]]
        except Exception:
            return []
    
    def get_all_terminal_logs(self, job_id: str) -> List[str]:
        log_file = os.path.join(self.TERMINAL_LOG_BASE_PATH, f"{job_id}.terminal.log")
        
        if not os.path.exists(log_file):
            return []
        
        try:
            with open(log_file, 'r') as f:
                return [line.strip() for line in f.readlines()]
        except Exception:
            return []
    
    def get_terminal_log_path(self, job_id: str) -> str:
        return os.path.join(self.TERMINAL_LOG_BASE_PATH, f"{job_id}.terminal.log")
    
    def get_terminal_log_size(self, job_id: str) -> int:
        log_file = self.get_terminal_log_path(job_id)
        if os.path.exists(log_file):
            return os.path.getsize(log_file)
        return 0
    
    def delete_terminal_log(self, job_id: str) -> bool:
        log_file = self.get_terminal_log_path(job_id)
        if os.path.exists(log_file):
            try:
                os.remove(log_file)
                return True
            except Exception:
                pass
        return False
    
    def format_training_progress(self, step: int, total_steps: int, loss: float, 
                                  lr: float = None, epoch: float = None,
                                  gpu_memory_mb: float = None, gpu_temp: float = None,
                                  throughput: float = None, grad_norm: float = None) -> str:
        parts = [f"Step {step}/{total_steps}"]
        
        if epoch is not None:
            parts.append(f"Epoch {epoch:.2f}")
        
        parts.append(f"Loss: {loss:.4f}")
        
        if lr is not None:
            parts.append(f"LR: {lr:.2e}")
        
        if grad_norm is not None:
            parts.append(f"Grad: {grad_norm:.4f}")
        
        if gpu_memory_mb is not None:
            parts.append(f"VRAM: {gpu_memory_mb:.0f}MB")
        
        if gpu_temp is not None:
            parts.append(f"Temp: {gpu_temp:.0f}°C")
        
        if throughput is not None:
            parts.append(f"Speed: {throughput:.1f} s/s")
        
        progress_pct = (step / total_steps * 100) if total_steps > 0 else 0
        parts.append(f"Progress: {progress_pct:.1f}%")
        
        return " | ".join(parts)
    
    def is_message_safe(self, message: str) -> bool:
        return not self._contains_sensitive_info(message)
    
    def get_safe_message(self, message: str) -> str:
        if self._contains_sensitive_info(message):
            return self._sanitize_message(message)
        return message


sanitized_log_service = SanitizedLogService()
