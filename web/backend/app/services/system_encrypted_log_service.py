"""
System-wide Encrypted Logging Service

This service handles encrypted logging for ALL system components EXCEPT training
(training has its own encrypted logs with job_id as filename).

Logs are encrypted with RSA public key - only US Inc can decrypt with private key.
Users CANNOT read these logs.

Structure:
  /app/data/encrypted_logs/system/
    └── YYYY-MM-DD/           (date folder)
        └── HH.enc.log        (hour-based log file, e.g., 14.enc.log for 2pm)

Features:
- Auto-cleanup: Removes folders older than 24 hours
- Background task runs hourly to cleanup old logs
- Logs inference, datasets, models, system operations
- Full error details, tracebacks, request/response data
- ONLY encrypted - nothing goes to terminal logs

Components Logged:
- Inference: model loading, chat, generation, adapters, memory cleanup
- Datasets: upload, validation, processing
- Models: validation, registry operations
- System: health checks, configuration, errors
"""

import asyncio
import base64
import json
import os
import shutil
import threading
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from cryptography.hazmat.primitives import serialization, hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


class SystemEncryptedLogService:
    """
    Comprehensive encrypted logging for all system components.
    
    Separate from training logs - uses date/hour folder structure.
    Auto-cleans logs older than 24 hours.
    """
    
    # Log categories
    CATEGORY_INFERENCE = "INFERENCE"
    CATEGORY_DATASET = "DATASET"
    CATEGORY_MODEL = "MODEL"
    CATEGORY_SYSTEM = "SYSTEM"
    CATEGORY_API = "API"
    CATEGORY_ERROR = "ERROR"
    
    # Try multiple paths for the public key
    PUBLIC_KEY_PATHS = [
        os.getenv("RSA_PUBLIC_KEY_PATH", ""),
        "/app/keys/usf_bios_public.pem",
        "/app/.k",
        "keys/usf_bios_public.pem",  # Development path
    ]
    
    # Base directory for system encrypted logs (separate from training logs)
    SYSTEM_LOG_DIR = os.getenv("SYSTEM_ENCRYPTED_LOG_PATH", "/app/data/encrypted_logs/system")
    
    # Retention period in hours
    RETENTION_HOURS = 24
    
    _public_key = None
    _key_loaded = False
    _cleanup_task = None
    _lock = threading.Lock()
    
    def __init__(self):
        """Initialize the service and ensure directories exist."""
        Path(self.SYSTEM_LOG_DIR).mkdir(parents=True, exist_ok=True)
        self._start_cleanup_scheduler()
    
    @classmethod
    def _load_public_key(cls):
        """Load RSA public key for encryption."""
        if cls._key_loaded:
            return cls._public_key
        
        cls._key_loaded = True
        
        if not CRYPTO_AVAILABLE:
            return None
        
        for key_path in cls.PUBLIC_KEY_PATHS:
            if not key_path or not os.path.exists(key_path):
                continue
            
            try:
                with open(key_path, 'rb') as f:
                    cls._public_key = serialization.load_pem_public_key(
                        f.read(),
                        backend=default_backend()
                    )
                return cls._public_key
            except Exception:
                continue
        
        return None
    
    def _encrypt_message(self, message: str) -> str:
        """Encrypt a message using RSA public key."""
        public_key = self._load_public_key()
        
        if public_key is None:
            # No encryption available - still encode but mark as unencrypted
            return base64.b64encode(f"[UNENCRYPTED]{message}".encode()).decode()
        
        try:
            # RSA can only encrypt small messages (~190 bytes with OAEP)
            # For longer messages, we chunk them
            max_chunk = 190
            message_bytes = message.encode('utf-8')
            
            if len(message_bytes) <= max_chunk:
                encrypted = public_key.encrypt(
                    message_bytes,
                    padding.OAEP(
                        mgf=padding.MGF1(algorithm=hashes.SHA256()),
                        algorithm=hashes.SHA256(),
                        label=None
                    )
                )
                return base64.b64encode(encrypted).decode()
            else:
                # For long messages, encrypt in chunks
                chunks = []
                for i in range(0, len(message_bytes), max_chunk):
                    chunk = message_bytes[i:i + max_chunk]
                    encrypted_chunk = public_key.encrypt(
                        chunk,
                        padding.OAEP(
                            mgf=padding.MGF1(algorithm=hashes.SHA256()),
                            algorithm=hashes.SHA256(),
                            label=None
                        )
                    )
                    chunks.append(base64.b64encode(encrypted_chunk).decode())
                
                # Join chunks with separator
                return "CHUNKED:" + "|".join(chunks)
        
        except Exception as e:
            return base64.b64encode(f"[ENCRYPT_ERROR:{str(e)[:50]}]{message[:100]}".encode()).decode()
    
    def _get_current_log_path(self) -> Path:
        """Get the current log file path based on date and hour."""
        now = datetime.utcnow()
        date_folder = now.strftime("%Y-%m-%d")
        hour_file = f"{now.strftime('%H')}.enc.log"
        
        log_dir = Path(self.SYSTEM_LOG_DIR) / date_folder
        log_dir.mkdir(parents=True, exist_ok=True)
        
        return log_dir / hour_file
    
    def _format_log_entry(
        self, 
        category: str, 
        level: str, 
        message: str,
        component: str = "",
        details: Optional[Dict[str, Any]] = None
    ) -> str:
        """Format a log entry with metadata."""
        timestamp = datetime.utcnow().isoformat()
        
        entry = {
            "ts": timestamp,
            "cat": category,
            "lvl": level,
            "cmp": component,
            "msg": message,
        }
        
        if details:
            entry["details"] = details
        
        return json.dumps(entry, default=str)
    
    def log(
        self,
        category: str,
        level: str,
        message: str,
        component: str = "",
        details: Optional[Dict[str, Any]] = None,
        error: Optional[Exception] = None
    ) -> None:
        """
        Log an encrypted message to the system log.
        
        Args:
            category: Log category (INFERENCE, DATASET, MODEL, SYSTEM, API, ERROR)
            level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
            message: Main log message
            component: Specific component (e.g., "load_model", "chat", "upload")
            details: Additional details as dict
            error: Exception object if logging an error
        """
        try:
            with self._lock:
                # Add error details if provided
                if error:
                    if details is None:
                        details = {}
                    details["error_type"] = type(error).__name__
                    details["error_message"] = str(error)
                    details["traceback"] = traceback.format_exc()
                
                # Format and encrypt
                formatted = self._format_log_entry(category, level, message, component, details)
                encrypted = self._encrypt_message(formatted)
                
                # Write to current hour's log file
                log_path = self._get_current_log_path()
                with open(log_path, 'a') as f:
                    f.write(encrypted + '\n')
        
        except Exception:
            # Silently fail - don't break the system for logging issues
            pass
    
    # =========================================================================
    # INFERENCE LOGGING
    # =========================================================================
    
    def log_inference_model_load(
        self,
        model_path: str,
        adapter_path: Optional[str],
        backend: str,
        success: bool,
        error: Optional[str] = None,
        memory_used_gb: float = 0,
        load_time_ms: int = 0
    ) -> None:
        """Log model loading for inference."""
        self.log(
            category=self.CATEGORY_INFERENCE,
            level="INFO" if success else "ERROR",
            message=f"Model load {'succeeded' if success else 'failed'}: {model_path}",
            component="load_model",
            details={
                "model_path": model_path,
                "adapter_path": adapter_path,
                "backend": backend,
                "success": success,
                "error": error,
                "memory_used_gb": memory_used_gb,
                "load_time_ms": load_time_ms
            }
        )
    
    def log_inference_adapter_load(
        self,
        adapter_path: str,
        base_model: str,
        success: bool,
        error: Optional[str] = None
    ) -> None:
        """Log adapter loading."""
        self.log(
            category=self.CATEGORY_INFERENCE,
            level="INFO" if success else "ERROR",
            message=f"Adapter load {'succeeded' if success else 'failed'}: {adapter_path}",
            component="load_adapter",
            details={
                "adapter_path": adapter_path,
                "base_model": base_model,
                "success": success,
                "error": error
            }
        )
    
    def log_inference_chat(
        self,
        model_path: str,
        backend: str,
        message_count: int,
        success: bool,
        tokens_generated: int = 0,
        inference_time_ms: int = 0,
        error: Optional[str] = None
    ) -> None:
        """Log chat inference request."""
        self.log(
            category=self.CATEGORY_INFERENCE,
            level="INFO" if success else "ERROR",
            message=f"Chat inference {'succeeded' if success else 'failed'}",
            component="chat",
            details={
                "model_path": model_path,
                "backend": backend,
                "message_count": message_count,
                "success": success,
                "tokens_generated": tokens_generated,
                "inference_time_ms": inference_time_ms,
                "error": error
            }
        )
    
    def log_inference_generate(
        self,
        model_path: str,
        backend: str,
        prompt_length: int,
        success: bool,
        tokens_generated: int = 0,
        inference_time_ms: int = 0,
        error: Optional[str] = None
    ) -> None:
        """Log text generation request."""
        self.log(
            category=self.CATEGORY_INFERENCE,
            level="INFO" if success else "ERROR",
            message=f"Generation {'succeeded' if success else 'failed'}",
            component="generate",
            details={
                "model_path": model_path,
                "backend": backend,
                "prompt_length": prompt_length,
                "success": success,
                "tokens_generated": tokens_generated,
                "inference_time_ms": inference_time_ms,
                "error": error
            }
        )
    
    def log_inference_memory_cleanup(
        self,
        cleanup_type: str,  # "clear" or "deep_clear"
        success: bool,
        memory_freed_gb: float = 0,
        engines_cleared: List[str] = None,
        error: Optional[str] = None
    ) -> None:
        """Log memory cleanup operations."""
        self.log(
            category=self.CATEGORY_INFERENCE,
            level="INFO" if success else "ERROR",
            message=f"Memory cleanup ({cleanup_type}) {'succeeded' if success else 'failed'}",
            component="memory_cleanup",
            details={
                "cleanup_type": cleanup_type,
                "success": success,
                "memory_freed_gb": memory_freed_gb,
                "engines_cleared": engines_cleared or [],
                "error": error
            }
        )
    
    def log_inference_status(self, status: Dict[str, Any]) -> None:
        """Log inference status check."""
        self.log(
            category=self.CATEGORY_INFERENCE,
            level="DEBUG",
            message="Inference status check",
            component="status",
            details=status
        )
    
    # =========================================================================
    # DATASET LOGGING
    # =========================================================================
    
    def log_dataset_upload(
        self,
        filename: str,
        file_size: int,
        success: bool,
        dataset_id: Optional[str] = None,
        error: Optional[str] = None
    ) -> None:
        """Log dataset upload."""
        self.log(
            category=self.CATEGORY_DATASET,
            level="INFO" if success else "ERROR",
            message=f"Dataset upload {'succeeded' if success else 'failed'}: {filename}",
            component="upload",
            details={
                "filename": filename,
                "file_size": file_size,
                "success": success,
                "dataset_id": dataset_id,
                "error": error
            }
        )
    
    def log_dataset_validation(
        self,
        dataset_path: str,
        success: bool,
        row_count: int = 0,
        format_type: str = "",
        error: Optional[str] = None
    ) -> None:
        """Log dataset validation."""
        self.log(
            category=self.CATEGORY_DATASET,
            level="INFO" if success else "ERROR",
            message=f"Dataset validation {'succeeded' if success else 'failed'}: {dataset_path}",
            component="validation",
            details={
                "dataset_path": dataset_path,
                "success": success,
                "row_count": row_count,
                "format_type": format_type,
                "error": error
            }
        )
    
    def log_dataset_list(self, count: int) -> None:
        """Log dataset listing."""
        self.log(
            category=self.CATEGORY_DATASET,
            level="DEBUG",
            message=f"Listed {count} datasets",
            component="list"
        )
    
    def log_dataset_delete(
        self,
        dataset_path: str,
        success: bool,
        error: Optional[str] = None
    ) -> None:
        """Log dataset deletion."""
        self.log(
            category=self.CATEGORY_DATASET,
            level="INFO" if success else "ERROR",
            message=f"Dataset delete {'succeeded' if success else 'failed'}: {dataset_path}",
            component="delete",
            details={
                "dataset_path": dataset_path,
                "success": success,
                "error": error
            }
        )
    
    # =========================================================================
    # MODEL LOGGING
    # =========================================================================
    
    def log_model_validation(
        self,
        model_path: str,
        model_source: str,
        valid: bool,
        reason: str = ""
    ) -> None:
        """Log model path validation."""
        self.log(
            category=self.CATEGORY_MODEL,
            level="INFO" if valid else "WARNING",
            message=f"Model validation {'passed' if valid else 'failed'}: {model_path}",
            component="validation",
            details={
                "model_path": model_path,
                "model_source": model_source,
                "valid": valid,
                "reason": reason
            }
        )
    
    def log_model_list(self, count: int, source: str = "all") -> None:
        """Log model listing."""
        self.log(
            category=self.CATEGORY_MODEL,
            level="DEBUG",
            message=f"Listed {count} models from {source}",
            component="list"
        )
    
    def log_model_info(
        self,
        model_path: str,
        success: bool,
        info: Optional[Dict] = None,
        error: Optional[str] = None
    ) -> None:
        """Log model info retrieval."""
        self.log(
            category=self.CATEGORY_MODEL,
            level="INFO" if success else "ERROR",
            message=f"Model info {'retrieved' if success else 'failed'}: {model_path}",
            component="info",
            details={
                "model_path": model_path,
                "success": success,
                "info": info,
                "error": error
            }
        )
    
    # =========================================================================
    # SYSTEM LOGGING
    # =========================================================================
    
    def log_system_health(self, status: Dict[str, Any]) -> None:
        """Log system health check."""
        self.log(
            category=self.CATEGORY_SYSTEM,
            level="DEBUG",
            message="System health check",
            component="health",
            details=status
        )
    
    def log_system_startup(self, details: Dict[str, Any]) -> None:
        """Log system startup."""
        self.log(
            category=self.CATEGORY_SYSTEM,
            level="INFO",
            message="System startup",
            component="startup",
            details=details
        )
    
    def log_system_shutdown(self, reason: str = "normal") -> None:
        """Log system shutdown."""
        self.log(
            category=self.CATEGORY_SYSTEM,
            level="INFO",
            message=f"System shutdown: {reason}",
            component="shutdown"
        )
    
    def log_system_error(
        self,
        component: str,
        message: str,
        error: Optional[Exception] = None,
        details: Optional[Dict] = None
    ) -> None:
        """Log system-level error."""
        self.log(
            category=self.CATEGORY_ERROR,
            level="ERROR",
            message=message,
            component=component,
            details=details,
            error=error
        )
    
    def log_gpu_status(self, gpu_info: Dict[str, Any]) -> None:
        """Log GPU status."""
        self.log(
            category=self.CATEGORY_SYSTEM,
            level="DEBUG",
            message="GPU status check",
            component="gpu",
            details=gpu_info
        )
    
    # =========================================================================
    # API LOGGING
    # =========================================================================
    
    def log_api_request(
        self,
        endpoint: str,
        method: str,
        success: bool,
        status_code: int = 200,
        duration_ms: int = 0,
        error: Optional[str] = None
    ) -> None:
        """Log API request."""
        self.log(
            category=self.CATEGORY_API,
            level="DEBUG" if success else "ERROR",
            message=f"{method} {endpoint} -> {status_code}",
            component="request",
            details={
                "endpoint": endpoint,
                "method": method,
                "success": success,
                "status_code": status_code,
                "duration_ms": duration_ms,
                "error": error
            }
        )
    
    # =========================================================================
    # CLEANUP OPERATIONS
    # =========================================================================
    
    def _cleanup_old_logs(self) -> Dict[str, Any]:
        """Remove log folders older than RETENTION_HOURS."""
        try:
            now = datetime.utcnow()
            cutoff = now - timedelta(hours=self.RETENTION_HOURS)
            
            removed_folders = []
            kept_folders = []
            
            log_dir = Path(self.SYSTEM_LOG_DIR)
            if not log_dir.exists():
                return {"removed": 0, "kept": 0}
            
            for folder in log_dir.iterdir():
                if not folder.is_dir():
                    continue
                
                try:
                    # Parse folder name as date
                    folder_date = datetime.strptime(folder.name, "%Y-%m-%d")
                    
                    # Check if folder is older than cutoff
                    # We keep folders from today and yesterday (within 24h)
                    if folder_date.date() < cutoff.date():
                        shutil.rmtree(folder)
                        removed_folders.append(folder.name)
                    else:
                        kept_folders.append(folder.name)
                
                except ValueError:
                    # Not a date folder, skip
                    continue
            
            # Log cleanup result
            if removed_folders:
                self.log(
                    category=self.CATEGORY_SYSTEM,
                    level="INFO",
                    message=f"Log cleanup: removed {len(removed_folders)} old folders",
                    component="log_cleanup",
                    details={
                        "removed_folders": removed_folders,
                        "kept_folders": kept_folders,
                        "retention_hours": self.RETENTION_HOURS
                    }
                )
            
            return {
                "removed": len(removed_folders),
                "kept": len(kept_folders),
                "removed_folders": removed_folders
            }
        
        except Exception as e:
            return {"error": str(e)}
    
    _cleanup_thread_started = False
    _cleanup_thread_lock = threading.Lock()
    
    def _start_cleanup_scheduler(self) -> None:
        """
        Start background task to cleanup old logs every hour.
        
        ROBUST GUARANTEES:
        - Runs cleanup immediately on startup (catches old logs from previous runs)
        - Continues running every hour in background
        - Survives all exceptions - never crashes
        - Works after restarts, deployments, container recreation
        - Thread is daemon - won't block shutdown
        - Singleton pattern - only one cleanup thread per process
        """
        with self._cleanup_thread_lock:
            # Prevent multiple cleanup threads (singleton pattern)
            if SystemEncryptedLogService._cleanup_thread_started:
                return
            SystemEncryptedLogService._cleanup_thread_started = True
        
        try:
            # Run cleanup IMMEDIATELY on startup to handle old logs from previous runs
            # This ensures logs are cleaned even if service was down for a while
            self._safe_cleanup()
            
            # Start background thread for hourly cleanup
            cleanup_thread = threading.Thread(
                target=self._hourly_cleanup_loop,
                daemon=True,  # Daemon thread - exits when main process exits
                name="EncryptedLogCleanup"
            )
            cleanup_thread.start()
        except Exception:
            # Reset flag so it can be retried
            SystemEncryptedLogService._cleanup_thread_started = False
    
    def _safe_cleanup(self) -> Dict[str, Any]:
        """
        Run cleanup with full exception handling.
        Never raises exceptions - always returns result dict.
        """
        try:
            return self._cleanup_old_logs()
        except Exception as e:
            try:
                # Try to log the error
                self.log(
                    category=self.CATEGORY_ERROR,
                    level="ERROR",
                    message="Cleanup task failed",
                    component="cleanup_scheduler",
                    details={"error": str(e), "traceback": traceback.format_exc()}
                )
            except Exception:
                pass
            return {"error": str(e), "removed": 0}
    
    def _hourly_cleanup_loop(self) -> None:
        """
        Background thread for hourly cleanup.
        
        NEVER exits, NEVER raises exceptions.
        Runs cleanup every hour indefinitely.
        """
        while True:
            try:
                # Sleep for 1 hour (3600 seconds)
                # Using a loop with shorter sleeps to handle interrupts gracefully
                sleep_interval = 60  # Check every minute
                sleep_total = 3600   # Total 1 hour
                
                for _ in range(sleep_total // sleep_interval):
                    try:
                        time.sleep(sleep_interval)
                    except Exception:
                        # Handle sleep interruption gracefully
                        pass
                
                # Run cleanup after sleeping
                self._safe_cleanup()
                
            except Exception:
                # NEVER let any exception escape the loop
                # If something goes wrong, just continue to next iteration
                try:
                    time.sleep(60)  # Brief pause before retrying
                except Exception:
                    pass
    
    def force_cleanup(self) -> Dict[str, Any]:
        """
        Manually trigger log cleanup.
        Safe to call anytime - handles all exceptions.
        """
        return self._safe_cleanup()
    
    def ensure_cleanup_running(self) -> bool:
        """
        Ensure the cleanup scheduler is running.
        Call this on startup to guarantee cleanup is active.
        Safe to call multiple times - uses singleton pattern.
        
        Returns:
            True if cleanup thread is running
        """
        try:
            self._start_cleanup_scheduler()
            return SystemEncryptedLogService._cleanup_thread_started
        except Exception:
            return False
    
    def get_log_stats(self) -> Dict[str, Any]:
        """Get statistics about current logs."""
        try:
            log_dir = Path(self.SYSTEM_LOG_DIR)
            if not log_dir.exists():
                return {"total_folders": 0, "total_files": 0, "total_size_mb": 0}
            
            total_files = 0
            total_size = 0
            folders = []
            
            for folder in log_dir.iterdir():
                if not folder.is_dir():
                    continue
                
                folder_files = list(folder.glob("*.enc.log"))
                folder_size = sum(f.stat().st_size for f in folder_files)
                
                folders.append({
                    "date": folder.name,
                    "files": len(folder_files),
                    "size_mb": round(folder_size / (1024 * 1024), 2)
                })
                
                total_files += len(folder_files)
                total_size += folder_size
            
            return {
                "total_folders": len(folders),
                "total_files": total_files,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "folders": folders,
                "retention_hours": self.RETENTION_HOURS,
                "log_directory": str(log_dir)
            }
        
        except Exception as e:
            return {"error": str(e)}


# Global singleton instance
system_encrypted_log = SystemEncryptedLogService()


# Convenience functions for direct import
def log_inference(
    component: str,
    message: str,
    success: bool = True,
    details: Optional[Dict] = None,
    error: Optional[Exception] = None
) -> None:
    """Quick log for inference operations."""
    system_encrypted_log.log(
        category=SystemEncryptedLogService.CATEGORY_INFERENCE,
        level="INFO" if success else "ERROR",
        message=message,
        component=component,
        details=details,
        error=error
    )


def log_dataset(
    component: str,
    message: str,
    success: bool = True,
    details: Optional[Dict] = None,
    error: Optional[Exception] = None
) -> None:
    """Quick log for dataset operations."""
    system_encrypted_log.log(
        category=SystemEncryptedLogService.CATEGORY_DATASET,
        level="INFO" if success else "ERROR",
        message=message,
        component=component,
        details=details,
        error=error
    )


def log_model(
    component: str,
    message: str,
    success: bool = True,
    details: Optional[Dict] = None,
    error: Optional[Exception] = None
) -> None:
    """Quick log for model operations."""
    system_encrypted_log.log(
        category=SystemEncryptedLogService.CATEGORY_MODEL,
        level="INFO" if success else "ERROR",
        message=message,
        component=component,
        details=details,
        error=error
    )


def log_system(
    component: str,
    message: str,
    level: str = "INFO",
    details: Optional[Dict] = None,
    error: Optional[Exception] = None
) -> None:
    """Quick log for system operations."""
    system_encrypted_log.log(
        category=SystemEncryptedLogService.CATEGORY_SYSTEM,
        level=level,
        message=message,
        component=component,
        details=details,
        error=error
    )


def log_error(
    component: str,
    message: str,
    error: Optional[Exception] = None,
    details: Optional[Dict] = None
) -> None:
    """Quick log for errors."""
    system_encrypted_log.log(
        category=SystemEncryptedLogService.CATEGORY_ERROR,
        level="ERROR",
        message=message,
        component=component,
        details=details,
        error=error
    )
