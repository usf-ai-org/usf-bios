#!/usr/bin/env python3
# Copyright (c) US Inc. All rights reserved.
# USF BIOS - Log Decryption Module
# Decrypt encrypted log files using RSA private key.

import base64
import sys
import os

CRYPTO_AVAILABLE = False
serialization = None
hashes = None
padding = None
default_backend = None

try:
    from cryptography.hazmat.primitives import serialization as _serialization
    from cryptography.hazmat.primitives import hashes as _hashes
    from cryptography.hazmat.primitives.asymmetric import padding as _padding
    from cryptography.hazmat.backends import default_backend as _default_backend
    serialization = _serialization
    hashes = _hashes
    padding = _padding
    default_backend = _default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    pass


def check_crypto_available():
    """Check if cryptography library is available."""
    if not CRYPTO_AVAILABLE:
        print("ERROR: cryptography library not installed. Run: pip install cryptography")
        sys.exit(1)


def load_private_key(key_path: str):
    """Load RSA private key from PEM file."""
    check_crypto_available()
    try:
        with open(key_path, 'rb') as f:
            private_key = serialization.load_pem_private_key(
                f.read(),
                password=None,
                backend=default_backend()
            )
        return private_key
    except Exception as e:
        print(f"ERROR: Failed to load private key from {key_path}: {e}")
        sys.exit(1)


def decrypt_single_chunk(encrypted_b64: str, private_key) -> str:
    """Decrypt a single base64-encoded encrypted chunk."""
    check_crypto_available()
    encrypted_bytes = base64.b64decode(encrypted_b64)
    decrypted = private_key.decrypt(
        encrypted_bytes,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None
        )
    )
    return decrypted.decode('utf-8')


def decrypt_message(encrypted_b64: str, private_key) -> str:
    """Decrypt a base64-encoded encrypted message (handles CHUNKED format)."""
    check_crypto_available()
    try:
        if encrypted_b64.startswith("CHUNKED:"):
            chunks_str = encrypted_b64[8:]
            chunks = chunks_str.split("|")
            
            decrypted_parts = []
            for chunk in chunks:
                if chunk:
                    decrypted_parts.append(decrypt_single_chunk(chunk, private_key))
            
            return "".join(decrypted_parts)
        
        encrypted_bytes = base64.b64decode(encrypted_b64)
        
        try:
            decoded = encrypted_bytes.decode('utf-8')
            if decoded.startswith('[UNENCRYPTED]'):
                return decoded
            if decoded.startswith('[ENCRYPT_ERROR'):
                return decoded
        except UnicodeDecodeError:
            pass
        
        decrypted = private_key.decrypt(
            encrypted_bytes,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return decrypted.decode('utf-8')
    except Exception as e:
        return f"[DECRYPT_ERROR: {e}] {encrypted_b64[:50]}..."


def decrypt_log_file(log_path: str, key_path: str):
    """Decrypt all entries in a log file."""
    print(f"Loading private key from: {key_path}")
    private_key = load_private_key(key_path)
    
    print(f"\n{'='*80}")
    print(f"Decrypting: {log_path}")
    print(f"{'='*80}\n")
    
    decrypted_count = 0
    error_count = 0
    unencrypted_count = 0
    
    with open(log_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
                
            decrypted = decrypt_message(line, private_key)
            print(decrypted)
            
            if decrypted.startswith('[DECRYPT_ERROR'):
                error_count += 1
            elif decrypted.startswith('[UNENCRYPTED]'):
                unencrypted_count += 1
            else:
                decrypted_count += 1
    
    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"  - Total entries: {decrypted_count + error_count + unencrypted_count}")
    print(f"  - Decrypted: {decrypted_count}")
    print(f"  - Unencrypted (base64 only): {unencrypted_count}")
    print(f"  - Errors: {error_count}")
    print(f"{'='*80}\n")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python decrypt_logs.py <log_file> [--key <private_key_path>]")
        print("Default key: keys/usf_bios_private.pem")
        sys.exit(1)
    
    log_file = sys.argv[1]
    key_file = "keys/usf_bios_private.pem"
    
    if '--key' in sys.argv:
        key_idx = sys.argv.index('--key')
        if key_idx + 1 < len(sys.argv):
            key_file = sys.argv[key_idx + 1]
    
    if not os.path.exists(log_file):
        print(f"ERROR: Log file not found: {log_file}")
        sys.exit(1)
    
    if not os.path.exists(key_file):
        print(f"ERROR: Private key not found: {key_file}")
        sys.exit(1)
    
    decrypt_log_file(log_file, key_file)
