# Copyright (c) US Inc. All rights reserved.
"""Dataset-related endpoints"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel

from ...core.config import settings
from ...core.capabilities import get_system_settings
from ...models.schemas import DatasetValidation

router = APIRouter()

# In-memory registry for datasets (in production, use a database)
_dataset_registry: dict = {}


class DatasetRegistration(BaseModel):
    """Request model for registering a dataset"""
    name: str
    source: Literal["huggingface", "modelscope", "local_path"]
    dataset_id: str  # HF/MS dataset ID or local path
    subset: Optional[str] = None  # For HF datasets with subsets
    split: Optional[str] = "train"
    max_samples: Optional[int] = None  # None or 0 = use all samples


class RegisteredDataset(BaseModel):
    """Registered dataset info"""
    id: str
    name: str
    source: Literal["upload", "huggingface", "modelscope", "local_path"]
    path: str  # Actual path or dataset ID
    subset: Optional[str] = None
    split: Optional[str] = None
    total_samples: int = 0
    size_human: str = "Unknown"
    format: str = "unknown"
    created_at: float
    selected: bool = True
    max_samples: Optional[int] = None  # None or 0 = use all samples


@router.post("/validate", response_model=DatasetValidation)
async def validate_dataset(dataset_path: str = Query(..., description="Path to dataset file")):
    """Validate dataset format and structure"""
    try:
        path = Path(dataset_path)
        
        if not path.exists():
            return DatasetValidation(
                valid=False,
                errors=["Dataset path does not exist"]
            )
        
        # Detect format by extension
        suffix = path.suffix.lower()
        
        if suffix == ".jsonl":
            return await _validate_jsonl(path)
        elif suffix == ".json":
            return await _validate_json(path)
        elif suffix == ".csv":
            return await _validate_csv(path)
        else:
            return DatasetValidation(
                valid=False,
                format_detected="unknown",
                errors=[f"Unsupported format: {suffix}. Use .jsonl, .json, or .csv"]
            )
    
    except Exception as e:
        return DatasetValidation(
            valid=False,
            errors=["Failed to validate dataset"]
        )


async def _validate_jsonl(path: Path) -> DatasetValidation:
    """Validate JSONL dataset"""
    samples = []
    total = 0
    errors = []
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                total += 1
                if i < 5:  # Preview first 5
                    try:
                        sample = json.loads(line.strip())
                        samples.append(sample)
                    except json.JSONDecodeError:
                        errors.append(f"Invalid JSON on line {i+1}")
        
        if not samples:
            return DatasetValidation(
                valid=False,
                total_samples=0,
                format_detected="jsonl",
                errors=["Dataset is empty"]
            )
        
        columns = list(samples[0].keys())
        
        # Check for required fields
        has_messages = "messages" in columns
        has_instruction = "instruction" in columns
        has_query_response = "query" in columns and "response" in columns
        
        if not (has_messages or has_instruction or has_query_response):
            errors.append("Dataset must have one of: 'messages', 'instruction', or 'query'/'response' fields")
        
        # Validate messages format if present
        if has_messages and samples:
            messages = samples[0].get("messages", [])
            if not isinstance(messages, list):
                errors.append("'messages' field must be a list")
            elif messages:
                for msg in messages:
                    if not isinstance(msg, dict):
                        errors.append("Each message must be a dict with 'role' and 'content'")
                        break
                    if "role" not in msg or "content" not in msg:
                        errors.append("Each message must have 'role' and 'content' keys")
                        break
        
        return DatasetValidation(
            valid=len(errors) == 0,
            total_samples=total,
            format_detected="jsonl",
            columns=columns,
            sample_preview=samples,
            errors=errors
        )
    
    except Exception as e:
        return DatasetValidation(
            valid=False,
            format_detected="jsonl",
            errors=["Failed to read JSONL file"]
        )


async def _validate_json(path: Path) -> DatasetValidation:
    """Validate JSON dataset"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            samples = data[:5]
            total = len(data)
            columns = list(samples[0].keys()) if samples else []
        else:
            return DatasetValidation(
                valid=False,
                format_detected="json",
                errors=["JSON file must contain a list of samples"]
            )
        
        return DatasetValidation(
            valid=True,
            total_samples=total,
            format_detected="json",
            columns=columns,
            sample_preview=samples,
            errors=[]
        )
    
    except Exception as e:
        return DatasetValidation(
            valid=False,
            format_detected="json",
            errors=["Failed to read JSON file"]
        )


async def _validate_csv(path: Path) -> DatasetValidation:
    """Validate CSV dataset"""
    try:
        import csv
        
        samples = []
        total = 0
        
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            columns = reader.fieldnames or []
            
            for i, row in enumerate(reader):
                total += 1
                if i < 5:
                    samples.append(dict(row))
        
        return DatasetValidation(
            valid=True,
            total_samples=total,
            format_detected="csv",
            columns=list(columns),
            sample_preview=samples,
            errors=[]
        )
    
    except Exception as e:
        return DatasetValidation(
            valid=False,
            format_detected="csv",
            errors=["Failed to read CSV file"]
        )


def _normalize_name(name: str) -> str:
    """Normalize dataset name for comparison"""
    safe = "".join(c for c in name if c.isalnum() or c in "._- ")
    return safe.strip().replace(" ", "_").lower()


def _is_name_taken(name: str) -> bool:
    """Check if normalized name already exists in uploads or registry"""
    normalized = _normalize_name(name)
    
    # Check uploaded files
    get_system_settings().UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    for f in get_system_settings().UPLOAD_DIR.glob("*"):
        if f.suffix.lower() in [".jsonl", ".json", ".csv"]:
            if _normalize_name(f.stem) == normalized:
                return True
    
    # Check registered datasets
    for ds in _dataset_registry.values():
        if _normalize_name(ds.get("name", "")) == normalized:
            return True
    
    return False


@router.get("/check-name")
async def check_name_available(name: str = Query(..., description="Dataset name to check")):
    """Check if a dataset name is available"""
    normalized = _normalize_name(name)
    if not normalized:
        return {"available": False, "error": "Invalid name"}
    
    taken = _is_name_taken(name)
    return {"available": not taken, "normalized": normalized}


@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    dataset_name: str = Query(..., description="Name for the dataset")
):
    """Upload a dataset file with a custom name"""
    try:
        # Validate file extension
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        suffix = Path(file.filename).suffix.lower()
        if suffix not in [".jsonl", ".json", ".csv"]:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {suffix}. Supported: .jsonl, .json, .csv")
        
        # Sanitize dataset name (remove special characters)
        safe_name = "".join(c for c in dataset_name if c.isalnum() or c in "._- ")
        safe_name = safe_name.strip().replace(" ", "_")
        if not safe_name:
            raise HTTPException(status_code=400, detail="Invalid dataset name")
        
        # Check if name is already taken
        if _is_name_taken(dataset_name):
            raise HTTPException(status_code=409, detail=f"Dataset name '{dataset_name}' is already in use. Please choose a different name.")
        
        # Create filename with user's name
        final_filename = f"{safe_name}{suffix}"
        upload_path = get_system_settings().UPLOAD_DIR / final_filename
        
        # Ensure upload directory exists
        get_system_settings().UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        
        # Save file
        content = await file.read()
        with open(upload_path, "wb") as f:
            f.write(content)
        
        # Validate the uploaded file
        validation = await validate_dataset(str(upload_path))
        
        return {
            "success": True,
            "id": final_filename,
            "name": safe_name,
            "filename": final_filename,
            "path": str(upload_path),
            "size": len(content),
            "format": suffix[1:],  # Remove the dot
            "valid": validation.valid,
            "total_samples": validation.total_samples,
            "errors": validation.errors
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to upload dataset")


@router.get("/list")
async def list_datasets():
    """List all uploaded datasets with metadata"""
    try:
        # Ensure upload directory exists
        get_system_settings().UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        
        datasets = []
        for f in get_system_settings().UPLOAD_DIR.glob("*"):
            if f.suffix.lower() in [".jsonl", ".json", ".csv"]:
                # Get basic info
                stat = f.stat()
                
                # Try to get sample count
                total_samples = 0
                try:
                    if f.suffix.lower() == ".jsonl":
                        with open(f, 'r', encoding='utf-8') as fp:
                            total_samples = sum(1 for _ in fp)
                    elif f.suffix.lower() == ".json":
                        import json
                        with open(f, 'r', encoding='utf-8') as fp:
                            data = json.load(fp)
                            if isinstance(data, list):
                                total_samples = len(data)
                    elif f.suffix.lower() == ".csv":
                        with open(f, 'r', encoding='utf-8') as fp:
                            total_samples = sum(1 for _ in fp) - 1  # Minus header
                except Exception:
                    pass
                
                datasets.append({
                    "id": f.name,
                    "name": f.stem,  # Name without extension
                    "filename": f.name,
                    "path": str(f),
                    "size": stat.st_size,
                    "size_human": _format_size(stat.st_size),
                    "format": f.suffix[1:].lower(),
                    "total_samples": total_samples,
                    "created_at": stat.st_ctime,
                })
        
        # Sort by creation time (newest first)
        datasets.sort(key=lambda x: x["created_at"], reverse=True)
        
        return {"datasets": datasets, "total": len(datasets)}
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to list datasets")


@router.get("/delete-info/{dataset_id}")
async def get_dataset_delete_info(dataset_id: str):
    """Get information needed for delete confirmation (returns dataset name)"""
    # Check uploaded datasets
    dataset_path = get_system_settings().UPLOAD_DIR / dataset_id
    if dataset_path.exists() and dataset_path.is_file():
        dataset_name = dataset_path.stem  # Name without extension
        return {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "source": "upload",
            "confirm_text": dataset_name  # User must type this to confirm deletion
        }
    
    # Check registered datasets
    if dataset_id in _dataset_registry:
        ds = _dataset_registry[dataset_id]
        dataset_name = ds.get("name", dataset_id)
        return {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "source": ds.get("source", "unknown"),
            "confirm_text": dataset_name  # User must type this to confirm deletion
        }
    
    raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")


@router.delete("/delete/{dataset_id}")
async def delete_dataset(dataset_id: str, confirm: str = Query(..., description="Type the dataset NAME to confirm")):
    """Delete an uploaded dataset (requires typing the dataset name to confirm)"""
    try:
        # Find the dataset
        dataset_path = get_system_settings().UPLOAD_DIR / dataset_id
        
        if not dataset_path.exists():
            raise HTTPException(status_code=404, detail=f"Dataset '{dataset_id}' not found")
        
        if not dataset_path.is_file():
            raise HTTPException(status_code=400, detail="Invalid dataset path")
        
        # Get the dataset name for confirmation
        dataset_name = dataset_path.stem  # Name without extension
        
        # Validate confirmation matches the dataset name
        if confirm != dataset_name:
            raise HTTPException(
                status_code=400, 
                detail=f"Confirmation failed. You must type '{dataset_name}' to delete this dataset."
            )
        
        # Verify it's in the upload directory (security check)
        try:
            dataset_path.resolve().relative_to(get_system_settings().UPLOAD_DIR.resolve())
        except ValueError:
            raise HTTPException(status_code=403, detail="Access denied")
        
        # Delete the file
        dataset_path.unlink()
        
        return {
            "success": True,
            "message": f"Dataset '{dataset_name}' deleted successfully",
            "deleted_id": dataset_id,
            "deleted_name": dataset_name
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to delete dataset")


def _format_size(size_bytes: int) -> str:
    """Format file size in human readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


@router.post("/register")
async def register_dataset(registration: DatasetRegistration):
    """Register a dataset from HuggingFace, ModelScope, or local path"""
    try:
        # Check if name is already taken
        if _is_name_taken(registration.name):
            raise HTTPException(status_code=409, detail=f"Dataset name '{registration.name}' is already in use. Please choose a different name.")
        
        dataset_id = str(uuid.uuid4())[:8]
        
        # Validate based on source
        if registration.source == "local_path":
            path = Path(registration.dataset_id)
            if not path.exists():
                raise HTTPException(status_code=400, detail=f"Local path does not exist: {registration.dataset_id}")
            
            # Get file info
            if path.is_file():
                stat = path.stat()
                size_human = _format_size(stat.st_size)
                fmt = path.suffix[1:].lower() if path.suffix else "unknown"
            else:
                size_human = "Directory"
                fmt = "directory"
        else:
            # For HuggingFace/ModelScope, we just register the ID
            size_human = "Remote"
            fmt = "hub"
        
        # Create registered dataset entry
        registered = RegisteredDataset(
            id=dataset_id,
            name=registration.name,
            source=registration.source,
            path=registration.dataset_id,
            subset=registration.subset,
            split=registration.split,
            total_samples=0,  # Will be determined during training
            size_human=size_human,
            format=fmt,
            created_at=datetime.now().timestamp(),
            selected=True,
            max_samples=registration.max_samples if registration.max_samples and registration.max_samples > 0 else None
        )
        
        # Store in registry
        _dataset_registry[dataset_id] = registered.model_dump()
        
        return {
            "success": True,
            "dataset": registered.model_dump()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to register dataset")


@router.get("/list-all")
async def list_all_datasets():
    """List all datasets (uploaded + registered from HF/MS/local)"""
    try:
        get_system_settings().UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        
        all_datasets = []
        
        # 1. Get uploaded files
        for f in get_system_settings().UPLOAD_DIR.glob("*"):
            if f.suffix.lower() in [".jsonl", ".json", ".csv"]:
                stat = f.stat()
                total_samples = 0
                try:
                    if f.suffix.lower() == ".jsonl":
                        with open(f, 'r', encoding='utf-8') as fp:
                            total_samples = sum(1 for _ in fp)
                    elif f.suffix.lower() == ".json":
                        with open(f, 'r', encoding='utf-8') as fp:
                            data = json.load(fp)
                            if isinstance(data, list):
                                total_samples = len(data)
                    elif f.suffix.lower() == ".csv":
                        with open(f, 'r', encoding='utf-8') as fp:
                            total_samples = sum(1 for _ in fp) - 1
                except Exception:
                    pass
                
                all_datasets.append({
                    "id": f"upload_{f.name}",
                    "name": f.stem,
                    "source": "upload",
                    "path": str(f),
                    "subset": None,
                    "split": None,
                    "total_samples": total_samples,
                    "size_human": _format_size(stat.st_size),
                    "format": f.suffix[1:].lower(),
                    "created_at": stat.st_ctime,
                    "selected": True
                })
        
        # 2. Get registered datasets (HF/MS/local_path)
        for ds in _dataset_registry.values():
            all_datasets.append(ds)
        
        # Sort by creation time (newest first)
        all_datasets.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        
        return {"datasets": all_datasets, "total": len(all_datasets)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to list datasets")


@router.delete("/unregister/{dataset_id}")
async def unregister_dataset(dataset_id: str, confirm: str = Query(..., description="Type the dataset NAME to confirm")):
    """Unregister a dataset (for HF/MS/local_path datasets). User must type the dataset name to confirm."""
    try:
        dataset_name = None
        
        # Check if it's an uploaded file
        if dataset_id.startswith("upload_"):
            filename = dataset_id[7:]  # Remove "upload_" prefix
            file_path = get_system_settings().UPLOAD_DIR / filename
            if file_path.exists():
                dataset_name = file_path.stem
                # Validate confirmation matches the dataset name
                if confirm != dataset_name:
                    raise HTTPException(
                        status_code=400, 
                        detail=f"Confirmation failed. You must type '{dataset_name}' to delete this dataset."
                    )
                file_path.unlink()
                return {"success": True, "message": f"Deleted uploaded file: {filename}", "deleted_name": dataset_name}
        
        # Check registry
        if dataset_id in _dataset_registry:
            ds = _dataset_registry[dataset_id]
            dataset_name = ds.get("name", dataset_id)
            # Validate confirmation matches the dataset name
            if confirm != dataset_name:
                raise HTTPException(
                    status_code=400, 
                    detail=f"Confirmation failed. You must type '{dataset_name}' to delete this dataset."
                )
            del _dataset_registry[dataset_id]
            return {"success": True, "message": f"Unregistered dataset: {dataset_name}", "deleted_name": dataset_name}
        
        raise HTTPException(status_code=404, detail=f"Dataset not found: {dataset_id}")
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to unregister dataset")
