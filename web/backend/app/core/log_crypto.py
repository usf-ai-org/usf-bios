# Copyright (c) US Inc. All rights reserved.
"""Encrypted logging utilities for IP protection"""

import base64
import hashlib
import os
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

# Internal encryption configuration
_ENCRYPTION_KEY = base64.b64decode(b"YXJwaXRzaDAxOA==").decode()
_SALT = b"usf_bios_secure_2024"


def _derive_key(key_material: str) -> bytes:
    """Derive encryption key from key material"""
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=_SALT,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(key_material.encode()))
    return key


def get_cipher():
    """Get Fernet cipher instance"""
    key = _derive_key(_ENCRYPTION_KEY)
    return Fernet(key)


def encrypt_data(data: str) -> bytes:
    """Encrypt string data"""
    cipher = get_cipher()
    return cipher.encrypt(data.encode())


def decrypt_data(encrypted_data: bytes) -> str:
    """Decrypt data back to string"""
    cipher = get_cipher()
    return cipher.decrypt(encrypted_data).decode()


class EncryptedLogWriter:
    """Write encrypted logs to file"""
    
    def __init__(self, filepath: str):
        self.filepath = filepath
        self.cipher = get_cipher()
    
    def write(self, message: str):
        """Encrypt and write log message"""
        encrypted = self.cipher.encrypt(message.encode())
        with open(self.filepath, 'ab') as f:
            f.write(encrypted + b'\n---ENC_LINE---\n')
    
    def flush(self):
        pass


class EncryptedLogReader:
    """Read and decrypt log files"""
    
    def __init__(self, decryption_key: str):
        key = _derive_key(decryption_key)
        self.cipher = Fernet(key)
    
    def read_file(self, filepath: str) -> list:
        """Read and decrypt all log entries from file"""
        entries = []
        with open(filepath, 'rb') as f:
            content = f.read()
        
        for encrypted_line in content.split(b'\n---ENC_LINE---\n'):
            if encrypted_line.strip():
                try:
                    decrypted = self.cipher.decrypt(encrypted_line)
                    entries.append(decrypted.decode())
                except Exception:
                    entries.append("[DECRYPTION_FAILED]")
        
        return entries
