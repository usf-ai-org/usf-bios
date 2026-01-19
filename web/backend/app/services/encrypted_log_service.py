import base64
import os
from typing import Optional
from datetime import datetime
from pathlib import Path

try:
    from cryptography.hazmat.primitives import serialization, hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


class EncryptedLogService:
    """
    Encrypted logging service for sensitive data.
    - Uses RSA public key encryption (only US Inc can decrypt with private key)
    - Stores encrypted logs to files for later retrieval
    - User CANNOT decrypt these logs (no private key in container)
    """
    
    PUBLIC_KEY_PATH = os.getenv("RSA_PUBLIC_KEY_PATH", "/app/.k")
    ENCRYPTED_LOG_DIR = os.getenv("ENCRYPTED_LOG_PATH", "/app/data/encrypted_logs")
    _public_key = None
    
    def __init__(self):
        # Ensure encrypted log directory exists
        Path(self.ENCRYPTED_LOG_DIR).mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def _load_public_key(cls):
        if cls._public_key is not None:
            return cls._public_key
        
        if not CRYPTO_AVAILABLE:
            return None
        
        if not os.path.exists(cls.PUBLIC_KEY_PATH):
            return None
        
        try:
            with open(cls.PUBLIC_KEY_PATH, 'rb') as f:
                cls._public_key = serialization.load_pem_public_key(
                    f.read(),
                    backend=default_backend()
                )
            return cls._public_key
        except Exception:
            return None
    
    @classmethod
    def encrypt_message(cls, message: str) -> str:
        """Encrypt a message using RSA public key."""
        public_key = cls._load_public_key()
        
        if public_key is None:
            # No encryption available - still encode but mark as unencrypted
            return base64.b64encode(f"[UNENCRYPTED]{message}".encode()).decode()
        
        try:
            # RSA can only encrypt small messages, so we need to handle long messages
            # by chunking or using hybrid encryption. For simplicity, truncate long messages.
            max_length = 190  # RSA 2048-bit can encrypt ~190 bytes with OAEP
            if len(message.encode('utf-8')) > max_length:
                # For long messages, just encrypt a summary
                truncated = message[:max_length - 20] + "...[TRUNCATED]"
                message = truncated
            
            encrypted = public_key.encrypt(
                message.encode('utf-8'),
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None
                )
            )
            return base64.b64encode(encrypted).decode()
        except Exception as e:
            return base64.b64encode(f"[ENCRYPT_ERROR:{str(e)}]{message[:100]}".encode()).decode()
    
    @classmethod
    def format_log_entry(cls, level: str, message: str, job_id: str) -> str:
        """Format and encrypt a log entry."""
        timestamp = datetime.utcnow().isoformat()
        full_message = f"[{timestamp}][{level}][job:{job_id}] {message}"
        return cls.encrypt_message(full_message)
    
    def encrypt_and_format(self, message: str, job_id: str, level: str = "INFO") -> None:
        """
        Encrypt a message and write to the job's encrypted log file.
        This is the main method called by training_service.
        
        - Full errors and tracebacks go here (for US Inc)
        - Users CANNOT read these logs (encrypted with public key)
        """
        try:
            # Create encrypted entry
            encrypted_entry = self.format_log_entry(level, message, job_id)
            
            # Write to job-specific encrypted log file
            log_file = Path(self.ENCRYPTED_LOG_DIR) / f"{job_id}.enc.log"
            with open(log_file, 'a') as f:
                f.write(encrypted_entry + '\n')
        except Exception:
            # Silently fail - don't break training for logging issues
            pass
    
    def get_encrypted_log_path(self, job_id: str) -> str:
        """Get path to encrypted log file for a job."""
        return str(Path(self.ENCRYPTED_LOG_DIR) / f"{job_id}.enc.log")
    
    def get_encrypted_logs(self, job_id: str) -> list:
        """Get all encrypted log entries for a job (still encrypted)."""
        log_file = Path(self.ENCRYPTED_LOG_DIR) / f"{job_id}.enc.log"
        if not log_file.exists():
            return []
        try:
            with open(log_file, 'r') as f:
                return [line.strip() for line in f.readlines() if line.strip()]
        except Exception:
            return []


encrypted_log_service = EncryptedLogService()
