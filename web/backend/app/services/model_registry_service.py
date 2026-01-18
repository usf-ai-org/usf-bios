# Copyright (c) US Inc. All rights reserved.
"""Model Registry Service - Manages global model registry for training."""

import os
from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session

from app.models.db_models import RegisteredModel, ModelSource, TrainingJob


class ModelRegistryService:
    """Service for managing the global model registry.
    
    Models are GLOBAL resources that can be used across multiple trainings.
    Deleting a training does NOT delete the model - models persist independently.
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def register_model(
        self,
        name: str,
        source: str,
        source_id: str,
        description: Optional[str] = None,
        model_type: Optional[str] = None,
        model_size: Optional[str] = None
    ) -> RegisteredModel:
        """Register a model from HuggingFace, ModelScope, or local path.
        
        Args:
            name: Display name for the model
            source: Source type - 'huggingface', 'modelscope', or 'local'
            source_id: HF/MS model ID or local path
            description: Optional description
            model_type: Optional model type (llm, vlm, etc.)
            model_size: Optional model size (7B, 14B, 40B, etc.)
        
        Returns:
            RegisteredModel instance
        """
        if source not in [ModelSource.HUGGINGFACE.value, ModelSource.MODELSCOPE.value, ModelSource.LOCAL.value]:
            raise ValueError(f"Invalid source: {source}. Must be 'huggingface', 'modelscope', or 'local'")
        
        # For local paths, verify existence
        if source == ModelSource.LOCAL.value:
            if not os.path.exists(source_id):
                raise ValueError(f"Local path does not exist: {source_id}")
        
        # Check if model already registered
        existing = self.db.query(RegisteredModel).filter(
            RegisteredModel.source == source,
            RegisteredModel.source_id == source_id
        ).first()
        
        if existing:
            return existing
        
        model = RegisteredModel(
            name=name,
            description=description,
            source=source,
            source_id=source_id,
            model_type=model_type,
            model_size=model_size,
        )
        
        self.db.add(model)
        self.db.commit()
        return model
    
    def get_model(self, model_id: str) -> Optional[RegisteredModel]:
        """Get a registered model by ID."""
        return self.db.query(RegisteredModel).filter(RegisteredModel.id == model_id).first()
    
    def get_model_by_source(self, source: str, source_id: str) -> Optional[RegisteredModel]:
        """Get a registered model by source and source_id."""
        return self.db.query(RegisteredModel).filter(
            RegisteredModel.source == source,
            RegisteredModel.source_id == source_id
        ).first()
    
    def list_models(self, source: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all registered models, optionally filtered by source."""
        query = self.db.query(RegisteredModel)
        
        if source:
            query = query.filter(RegisteredModel.source == source)
        
        models = query.order_by(RegisteredModel.created_at.desc()).all()
        
        result = []
        for model in models:
            trainings_count = self.db.query(TrainingJob).filter(
                TrainingJob.registered_model_id == model.id
            ).count()
            
            result.append({
                "id": model.id,
                "name": model.name,
                "description": model.description,
                "source": model.source,
                "source_id": model.source_id,
                "model_type": model.model_type,
                "model_size": model.model_size,
                "times_used": model.times_used,
                "last_used_at": model.last_used_at.isoformat() if model.last_used_at else None,
                "created_at": model.created_at.isoformat() if model.created_at else None,
                "trainings_count": trainings_count
            })
        
        return result
    
    def update_model(
        self,
        model_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        model_type: Optional[str] = None,
        model_size: Optional[str] = None
    ) -> Optional[RegisteredModel]:
        """Update a registered model's metadata."""
        model = self.get_model(model_id)
        if not model:
            return None
        
        if name:
            model.name = name
        if description is not None:
            model.description = description
        if model_type:
            model.model_type = model_type
        if model_size:
            model.model_size = model_size
        
        model.updated_at = datetime.utcnow()
        self.db.commit()
        return model
    
    def delete_model(self, model_id: str, force: bool = False) -> Dict[str, Any]:
        """Delete a registered model.
        
        Args:
            model_id: ID of model to delete
            force: If True, delete even if model is used by trainings
        
        Returns:
            Dict with success status and details
        """
        model = self.get_model(model_id)
        if not model:
            return {"success": False, "error": "Model not found"}
        
        # Check if model is used by any trainings
        trainings_using = self.db.query(TrainingJob).filter(
            TrainingJob.registered_model_id == model_id
        ).count()
        
        if trainings_using > 0 and not force:
            return {
                "success": False,
                "error": f"Model is used by {trainings_using} training(s). Use force=True to delete anyway.",
                "trainings_using": trainings_using
            }
        
        result = {
            "success": True,
            "model_id": model_id,
            "model_name": model.name,
            "source": model.source,
            "trainings_affected": trainings_using
        }
        
        # Note: We don't delete any files - models are external references
        # For local models, the files remain on disk
        self.db.delete(model)
        self.db.commit()
        
        result["message"] = "Model unregistered successfully"
        return result
    
    def increment_usage(self, model_id: str) -> None:
        """Increment the usage counter for a model."""
        model = self.get_model(model_id)
        if model:
            model.times_used = (model.times_used or 0) + 1
            model.last_used_at = datetime.utcnow()
            self.db.commit()
    
    def get_popular_models(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get the most frequently used models."""
        models = self.db.query(RegisteredModel).order_by(
            RegisteredModel.times_used.desc()
        ).limit(limit).all()
        
        return [
            {
                "id": m.id,
                "name": m.name,
                "source": m.source,
                "source_id": m.source_id,
                "model_type": m.model_type,
                "model_size": m.model_size,
                "times_used": m.times_used
            }
            for m in models
        ]
