# Copyright (c) US Inc. All rights reserved.
"""
USF BIOS Web API - Main Application
Enterprise AI Training & Fine-tuning Platform
"""

import os
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware

from .api import api_router
from .core.config import settings
from .core.database import init_db, engine

# No middleware needed - /config endpoint returns URL directly from request

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


@app.get("/config")
async def get_config(request: Request):
    """Return backend URL directly from request - works with ANY cloud provider"""
    # Get the URL from THIS request's headers - always correct
    host = request.headers.get("host", "")
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
    backend_url = f"{scheme}://{host}" if host else ""
    return {"backendUrl": backend_url}


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    data_dir = os.environ.get("DATA_DIR", "/app/data")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(f"{data_dir}/datasets", exist_ok=True)
    os.makedirs(f"{data_dir}/output", exist_ok=True)
    os.makedirs(f"{data_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{data_dir}/logs", exist_ok=True)
    os.makedirs(f"{data_dir}/terminal_logs", exist_ok=True)
    
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
