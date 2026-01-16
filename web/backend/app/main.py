# Copyright (c) US Inc. All rights reserved.
"""
USF BIOS Web API - Main Application
Enterprise AI Training & Fine-tuning Platform
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from .api import api_router
from .core.config import settings

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


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    print("=" * 60)
    print(f"  {settings.APP_NAME} v{settings.APP_VERSION}")
    print("  Powered by US Inc")
    print("=" * 60)
    print(f"  API: http://{settings.HOST}:{settings.PORT}")
    print(f"  Docs: http://{settings.HOST}:{settings.PORT}/docs")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    print("Shutting down USF BIOS API...")
