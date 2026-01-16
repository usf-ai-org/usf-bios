# Copyright (c) US Inc. All rights reserved.
"""API router - combines all endpoints"""

from fastapi import APIRouter
from .endpoints import jobs, models, datasets, health, inference

api_router = APIRouter()

api_router.include_router(health.router, tags=["Health"])
api_router.include_router(models.router, prefix="/models", tags=["Models"])
api_router.include_router(datasets.router, prefix="/datasets", tags=["Datasets"])
api_router.include_router(jobs.router, prefix="/jobs", tags=["Jobs"])
api_router.include_router(inference.router, prefix="/inference", tags=["Inference"])
