# Copyright (c) US Inc. All rights reserved.
"""Application configuration - Minimal public settings only."""

from typing import List, Optional
from pydantic_settings import BaseSettings, SettingsConfigDict

# Import version from usf_bios package
try:
    from usf_bios.version import __version__ as USF_VERSION
except ImportError:
    USF_VERSION = "2.0.06"


class Settings(BaseSettings):
    """Minimal application settings."""
    
    APP_NAME: str = "USF BIOS"
    APP_VERSION: str = USF_VERSION
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: List[str] = ["*"]
    API_KEY: Optional[str] = None
    DISABLE_CLI: bool = True
    
    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=True,
    )


settings = Settings()

# Initialize system (loads all other settings from compiled binary)
from .capabilities import init_validator, get_system_settings
init_validator()
_sys = get_system_settings()
