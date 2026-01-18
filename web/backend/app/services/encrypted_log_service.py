import base64
import os
from typing import Optional
from datetime import datetime

try:
    from cryptography.hazmat.primitives import serialization, hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False


class EncryptedLogService:
    
    PUBLIC_KEY_PATH = os.getenv("RSA_PUBLIC_KEY_PATH", "/app/.k")
    _public_key = None
    
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
        public_key = cls._load_public_key()
        
        if public_key is None:
            return base64.b64encode(f"[UNENCRYPTED]{message}".encode()).decode()
        
        try:
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
            return base64.b64encode(f"[ENCRYPT_ERROR:{str(e)}]{message}".encode()).decode()
    
    @classmethod
    def format_log_entry(cls, level: str, message: str, job_id: str) -> str:
        timestamp = datetime.utcnow().isoformat()
        full_message = f"[{timestamp}][{level}][job:{job_id}] {message}"
        return cls.encrypt_message(full_message)


encrypted_log_service = EncryptedLogService()
