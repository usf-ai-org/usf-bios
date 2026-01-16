# Copyright (c) US Inc. All rights reserved.
"""Security utilities"""

from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from .config import settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def verify_api_key(api_key: str = Security(api_key_header)) -> bool:
    """Verify API key if configured"""
    if settings.API_KEY is None:
        return True
    if api_key != settings.API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API key")
    return True
