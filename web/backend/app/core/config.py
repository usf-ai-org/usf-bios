# Copyright (c) US Inc. All rights reserved.
"""Application configuration settings"""

import os
from pathlib import Path
from typing import List, Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables"""
    
    # API Settings
    APP_NAME: str = "USF BIOS API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Server Settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # CORS Settings
    CORS_ORIGINS: List[str] = ["*"]
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent.parent.parent.parent
    UPLOAD_DIR: Path = Path("/app/data/uploads") if os.path.exists("/app") else Path("./uploads")
    OUTPUT_DIR: Path = Path("/app/data/outputs") if os.path.exists("/app") else Path("./outputs")
    MODELS_DIR: Path = Path("/app/data/models") if os.path.exists("/app") else Path("./models")
    
    # Training Settings
    MAX_CONCURRENT_JOBS: int = 3
    JOB_TIMEOUT_HOURS: int = 72
    
    # Security
    API_KEY: Optional[str] = None
    DISABLE_CLI: bool = True  # In Docker, CLI is disabled
    
    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()

# Create directories if they don't exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
