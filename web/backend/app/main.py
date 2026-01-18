# Copyright (c) US Inc. All rights reserved.
"""
USF BIOS Web API - Main Application
Enterprise AI Training & Fine-tuning Platform
"""

import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import api_router
from .core.config import settings
from .core.database import init_db, engine

# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="Enterprise AI Training & Fine-tuning Platform API - Powered by US Inc",
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api")


@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration"""
    return {"status": "healthy", "service": "usf-bios-api"}


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    os.makedirs("/app/data", exist_ok=True)
    os.makedirs("/app/data/datasets", exist_ok=True)
    os.makedirs("/app/data/output", exist_ok=True)
    os.makedirs("/app/data/checkpoints", exist_ok=True)
    os.makedirs("/app/data/logs", exist_ok=True)
    os.makedirs("/app/data/terminal_logs", exist_ok=True)
    
    init_db()
    
    from .core.database import get_db_session
    from .services.job_service import JobService
    
    with get_db_session() as db:
        job_service = JobService(db)
        recovered = job_service.recover_interrupted_jobs()
        if recovered:
            print(f"  Recovered {len(recovered)} interrupted job(s)")
            for r in recovered:
                print(f"    - Job {r['job_id']}: {r['name']} -> {r['new_status']}")
    
    print("=" * 60)
    print(f"  {settings.APP_NAME} v{settings.APP_VERSION}")
    print("  Powered by US Inc")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    engine.dispose()
