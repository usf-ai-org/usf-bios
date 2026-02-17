# Copyright (c) US Inc. All rights reserved.
"""Dataset-related endpoints"""

import csv
import json
import logging
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Literal, Optional

logger = logging.getLogger(__name__)

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from ...core.config import settings
from ...core.capabilities import get_system_settings, get_validator
from ...models.schemas import DatasetValidation
from ...services.dataset_type_service import dataset_type_service, DatasetType, ModelType, FileFormatConfig
from ...services.algorithm_compatibility import (
    algorithm_compatibility_service,
    ALGORITHM_REGISTRY,
    DATASET_TYPE_CONFIGS,
    TRAINING_TYPE_CONFIGS,
    QUANTIZATION_CONFIGS,
)

router = APIRouter()

# In-memory registry for datasets (in production, use a database)
_dataset_registry: dict = {}


class DatasetRegistration(BaseModel):
    """Request model for registering a dataset"""
    name: str
    source: str = "local_path"  # Dataset source
    dataset_id: str  # Local path to dataset directory or file
    subset: Optional[str] = None  # Dataset subset name
    split: Optional[str] = "train"
    max_samples: Optional[int] = None  # None or 0 = use all samples


class RegisteredDataset(BaseModel):
    """Registered dataset info"""
    id: str
    name: str
    source: str  # Dataset source
    path: str  # Actual path or dataset ID
    subset: Optional[str] = None
    split: Optional[str] = None
    total_samples: int = 0
    size_human: str = "Unknown"
    format: str = "unknown"
    created_at: float
    selected: bool = True
    max_samples: Optional[int] = None  # None or 0 = use all samples
    dataset_type: Optional[str] = None  # sft, rlhf_offline, rlhf_online, pt, kto, unknown
    dataset_type_display: Optional[str] = None  # Human-readable type name
    compatible_training_methods: Optional[List[str]] = None


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
        
        # Check for recognized training fields and validate data integrity
        errors.extend(_validate_training_fields(columns, samples))
        
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


def _validate_training_fields(columns: list, samples: list) -> list:
    """Validate that dataset has recognized training fields and data integrity.
    
    Checks:
    1. At least one recognized training field pattern exists
    2. Messages format is valid (if present)
    3. Required fields are not empty/null in samples
    
    Returns list of error strings (empty = valid).
    """
    errors = []
    columns_set = set(columns)
    
    # Recognized field patterns for training
    has_messages = "messages" in columns_set
    has_instruction = "instruction" in columns_set
    has_query_response = "query" in columns_set and "response" in columns_set
    has_text = "text" in columns_set
    has_prompt = "prompt" in columns_set
    has_chosen_rejected = "chosen" in columns_set and "rejected" in columns_set
    has_completion_label = "completion" in columns_set and "label" in columns_set
    has_input_output = "input" in columns_set and "output" in columns_set
    
    recognized = (
        has_messages or has_instruction or has_query_response or
        has_text or (has_prompt and has_chosen_rejected) or
        (has_prompt and has_completion_label) or has_input_output
    )
    
    if not recognized:
        errors.append(
            "No recognized training fields found. Dataset must contain one of: "
            "'messages' (SFT), 'instruction'+'output' (SFT), 'query'+'response' (SFT), "
            "'text' (pre-training), 'prompt'+'chosen'+'rejected' (RLHF/DPO), "
            "'prompt'+'completion'+'label' (KTO), or 'input'+'output' (SFT)"
        )
        return errors
    
    if not samples:
        return errors
    
    # Validate messages format across all preview samples
    if has_messages:
        for i, sample in enumerate(samples):
            messages = sample.get("messages")
            if messages is None or (isinstance(messages, str) and not messages.strip()):
                errors.append(f"Sample {i+1}: 'messages' field is empty or null")
                continue
            if not isinstance(messages, list):
                errors.append(f"Sample {i+1}: 'messages' must be a list, got {type(messages).__name__}")
                continue
            if len(messages) == 0:
                errors.append(f"Sample {i+1}: 'messages' list is empty")
                continue
            for j, msg in enumerate(messages):
                if not isinstance(msg, dict):
                    errors.append(f"Sample {i+1}, message {j+1}: must be an object with 'role' and 'content'")
                    break
                if "role" not in msg:
                    errors.append(f"Sample {i+1}, message {j+1}: missing 'role' key")
                    break
                if "content" not in msg:
                    errors.append(f"Sample {i+1}, message {j+1}: missing 'content' key")
                    break
                if not msg.get("role") or not isinstance(msg["role"], str):
                    errors.append(f"Sample {i+1}, message {j+1}: 'role' must be a non-empty string")
                    break
    
    # Validate required text fields are not empty/null
    required_text_fields = []
    if has_instruction:
        required_text_fields.append("instruction")
    if has_query_response:
        required_text_fields.extend(["query", "response"])
    if has_text:
        required_text_fields.append("text")
    if has_input_output:
        required_text_fields.extend(["input", "output"])
    if has_prompt and has_chosen_rejected:
        required_text_fields.extend(["prompt", "chosen", "rejected"])
    if has_prompt and has_completion_label:
        required_text_fields.extend(["prompt", "completion"])
    
    for field in required_text_fields:
        if field not in columns_set:
            continue
        empty_count = 0
        for sample in samples:
            val = sample.get(field)
            if val is None or (isinstance(val, str) and not val.strip()):
                empty_count += 1
        if empty_count == len(samples):
            errors.append(f"Field '{field}' is empty or null in all preview samples")
        elif empty_count > 0:
            errors.append(f"Field '{field}' is empty or null in {empty_count}/{len(samples)} preview samples")
    
    return errors


async def _validate_json(path: Path) -> DatasetValidation:
    """Validate JSON dataset"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        errors = []
        
        if not isinstance(data, list):
            return DatasetValidation(
                valid=False,
                format_detected="json",
                errors=["JSON file must contain a list of samples"]
            )
        
        if len(data) == 0:
            return DatasetValidation(
                valid=False,
                total_samples=0,
                format_detected="json",
                errors=["Dataset is empty (0 samples)"]
            )
        
        samples = data[:5]
        total = len(data)
        
        # Validate each sample is a dict
        for i, sample in enumerate(samples):
            if not isinstance(sample, dict):
                errors.append(f"Sample {i+1} is not a JSON object (got {type(sample).__name__})")
        
        if errors:
            return DatasetValidation(
                valid=False,
                total_samples=total,
                format_detected="json",
                errors=errors
            )
        
        columns = list(samples[0].keys()) if samples else []
        
        # Check for recognized training fields
        errors.extend(_validate_training_fields(columns, samples))
        
        return DatasetValidation(
            valid=len(errors) == 0,
            total_samples=total,
            format_detected="json",
            columns=columns,
            sample_preview=samples,
            errors=errors
        )
    
    except json.JSONDecodeError as e:
        return DatasetValidation(
            valid=False,
            format_detected="json",
            errors=[f"Invalid JSON: {str(e)}"]
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
        samples = []
        total = 0
        errors = []
        
        with open(path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            columns = list(reader.fieldnames or [])
            
            if not columns:
                return DatasetValidation(
                    valid=False,
                    total_samples=0,
                    format_detected="csv",
                    errors=["CSV file has no header row or columns"]
                )
            
            for i, row in enumerate(reader):
                total += 1
                if i < 5:
                    samples.append(dict(row))
        
        if total == 0:
            return DatasetValidation(
                valid=False,
                total_samples=0,
                format_detected="csv",
                columns=columns,
                errors=["CSV file has headers but no data rows"]
            )
        
        # Check for recognized training fields
        errors.extend(_validate_training_fields(columns, samples))
        
        return DatasetValidation(
            valid=len(errors) == 0,
            total_samples=total,
            format_detected="csv",
            columns=columns,
            sample_preview=samples,
            errors=errors
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


def _get_available_disk_space_gb(path: Path) -> float:
    """Get available disk space in GB for a path."""
    try:
        check_path = path
        while not check_path.exists() and str(check_path) != '/':
            check_path = check_path.parent
        if not check_path.exists():
            check_path = Path('/')
        usage = shutil.disk_usage(str(check_path))
        return usage.free / (1024 ** 3)
    except Exception:
        return -1


@router.post("/upload")
async def upload_dataset(
    file: UploadFile = File(...),
    dataset_name: str = Query(..., description="Name for the dataset")
):
    """
    Upload a dataset file with a custom name.
    
    Validates:
    1. File format is supported by USF BIOS (jsonl, json, csv, txt)
    2. File size is within limits for the format
    3. Dataset type is allowed by system feature flags
    4. Sufficient disk space available (with buffer)
    
    Supported formats:
    - JSONL: Unlimited size, recommended for large datasets
    - JSON: Max 2GB (must fit in memory)
    - CSV: Unlimited size, streaming supported
    - TXT: Unlimited size, for pre-training only
    
    NOT supported (will be rejected):
    - TSV, Parquet, Excel, Arrow
    """
    try:
        # === VALIDATION 1: Filename provided ===
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        suffix = Path(file.filename).suffix.lower()
        
        # === VALIDATION 2: Check for explicitly unsupported formats ===
        if FileFormatConfig.is_unsupported(suffix):
            error_msg = FileFormatConfig.get_unsupported_error(suffix)
            raise HTTPException(
                status_code=400, 
                detail=f"Format not supported: {error_msg}"
            )
        
        # === VALIDATION 3: Check if format is supported by USF BIOS ===
        file_format = FileFormatConfig.get_format_by_extension(suffix)
        if not file_format:
            supported = ", ".join(FileFormatConfig.get_all_extensions())
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported format: {suffix}. USF BIOS only supports: {supported}"
            )
        
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
        
        # === VALIDATION 4: Check disk space BEFORE reading/writing file ===
        # Get file size from Content-Length header if available
        file_size_bytes = 0
        if file.size:
            file_size_bytes = file.size
        elif hasattr(file, 'file') and hasattr(file.file, 'seek'):
            # Try to get size by seeking to end
            current_pos = file.file.tell()
            file.file.seek(0, 2)  # Seek to end
            file_size_bytes = file.file.tell()
            file.file.seek(current_pos)  # Seek back
        
        # Check available disk space
        available_gb = _get_available_disk_space_gb(get_system_settings().UPLOAD_DIR)
        
        if available_gb >= 0:  # -1 means couldn't determine
            # Calculate required space: file size + 10% buffer + 500MB minimum reserve
            file_size_gb = file_size_bytes / (1024 ** 3) if file_size_bytes > 0 else 0
            buffer_gb = max(file_size_gb * 0.1, 0.1)  # 10% buffer or 100MB minimum
            system_reserve_gb = 0.5  # 500MB reserved for system
            required_gb = file_size_gb + buffer_gb + system_reserve_gb
            
            if available_gb < required_gb:
                # Provide clear error message with details
                shortfall_gb = required_gb - available_gb
                error_detail = (
                    f"Insufficient disk space for upload. "
                    f"Available: {available_gb:.1f}GB, "
                    f"Required: {required_gb:.1f}GB (file: {file_size_gb:.2f}GB + buffer + reserve). "
                    f"Please free up at least {shortfall_gb:.1f}GB on this machine before uploading."
                )
                raise HTTPException(status_code=507, detail=error_detail)
            
            # Additional check: if very low space (< 1GB), warn even for small files
            if available_gb < 1.0:
                raise HTTPException(
                    status_code=507, 
                    detail=f"Critically low disk space: only {available_gb:.2f}GB available. "
                           f"Please free up disk space before uploading any files."
                )
        
        # Read file content
        content = await file.read()
        actual_size_gb = len(content) / (1024 ** 3)
        
        # Double-check space with actual file size (in case Content-Length was wrong)
        if available_gb >= 0:
            required_with_actual = actual_size_gb + max(actual_size_gb * 0.1, 0.1) + 0.5
            if available_gb < required_with_actual:
                shortfall = required_with_actual - available_gb
                raise HTTPException(
                    status_code=507,
                    detail=f"Insufficient disk space. File size: {actual_size_gb:.2f}GB, "
                           f"Available: {available_gb:.1f}GB. "
                           f"Please free up at least {shortfall:.1f}GB before uploading."
                )
        
        # Save file with error handling for disk full
        try:
            with open(upload_path, "wb") as f:
                f.write(content)
        except OSError as e:
            # Handle disk full error gracefully
            if e.errno == 28 or 'No space left' in str(e):  # ENOSPC
                # Clean up partial file if created
                try:
                    if upload_path.exists():
                        upload_path.unlink()
                except Exception:
                    pass
                raise HTTPException(
                    status_code=507,
                    detail=f"Disk full: Unable to save dataset. "
                           f"The file requires {actual_size_gb:.2f}GB but disk space ran out during write. "
                           f"Please free up disk space and try again."
                )
            raise
        
        # Validate the uploaded file content and structure
        validation = await validate_dataset(str(upload_path))
        
        # STRICT: Reject upload if content validation fails
        if not validation.valid:
            try:
                upload_path.unlink()
            except Exception:
                pass
            error_detail = "Dataset validation failed"
            if validation.errors:
                error_detail += ": " + "; ".join(validation.errors)
            raise HTTPException(status_code=400, detail=error_detail)
        
        # Detect dataset type
        type_info = dataset_type_service.detect_dataset_type(str(upload_path))
        
        # Get feature flags and validate if dataset type is allowed
        try:
            from ...core.capabilities import get_validator
            validator = get_validator()
            feature_flags = validator.get_feature_flags() if hasattr(validator, 'get_feature_flags') else {}
        except Exception:
            feature_flags = {}
        
        # Validate dataset type against feature flags
        is_valid, error_msg, _ = dataset_type_service.validate_dataset_for_upload(
            str(upload_path), feature_flags
        )
        
        if not is_valid:
            # Delete the uploaded file since it's not allowed
            try:
                upload_path.unlink()
            except Exception:
                pass
            raise HTTPException(status_code=403, detail=error_msg)
        
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
            "errors": validation.errors,
            "dataset_type": type_info.dataset_type.value,
            "dataset_type_display": type_info.display_name,
            "compatible_training_methods": type_info.compatible_training_methods,
            "compatible_rlhf_types": type_info.compatible_rlhf_types,
            "format_warning": type_info.format_warning,
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
        # Get all supported extensions from FileFormatConfig
        supported_extensions = FileFormatConfig.get_all_extensions()
        for f in get_system_settings().UPLOAD_DIR.glob("*"):
            if f.suffix.lower() in supported_extensions:
                # Get basic info
                stat = f.stat()
                
                # Try to get sample count based on format
                total_samples = 0
                try:
                    ext = f.suffix.lower()
                    if ext == ".jsonl":
                        with open(f, 'r', encoding='utf-8') as fp:
                            total_samples = sum(1 for _ in fp)
                    elif ext == ".json":
                        import json
                        with open(f, 'r', encoding='utf-8') as fp:
                            data = json.load(fp)
                            if isinstance(data, list):
                                total_samples = len(data)
                    elif ext == ".csv":
                        with open(f, 'r', encoding='utf-8') as fp:
                            total_samples = sum(1 for _ in fp) - 1  # Minus header
                    elif ext == ".txt":
                        with open(f, 'r', encoding='utf-8') as fp:
                            total_samples = sum(1 for _ in fp)  # Line count for text
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


# Valid source values (internal - not exposed in schema)
_VALID_DATASET_SOURCES = {"local_path", "local", "huggingface", "modelscope", "upload"}


@router.post("/register")
async def register_dataset(registration: DatasetRegistration):
    """Register a dataset from local path"""
    try:
        # Validate source value is valid
        if registration.source not in _VALID_DATASET_SOURCES:
            raise HTTPException(status_code=400, detail="Invalid source type")
        
        # Validate dataset is supported by this system configuration
        validator = get_validator()
        source_key = registration.source
        # Convert frontend source names to capability keys
        if source_key in ("local_path", "upload"):
            source_key = "local"
        
        is_valid, message = validator.validate_dataset_source(source_key)
        if not is_valid:
            raise HTTPException(status_code=403, detail=message)
        
        # Check if name is already taken
        if _is_name_taken(registration.name):
            raise HTTPException(status_code=409, detail=f"Dataset name '{registration.name}' is already in use. Please choose a different name.")
        
        dataset_id = str(uuid.uuid4())[:8]
        
        # Validate based on source
        dataset_type_str = None
        dataset_type_display = None
        compatible_methods = None
        total_samples = 0
        
        if registration.source in ("local_path", "local"):
            path = Path(registration.dataset_id)
            if not path.exists():
                raise HTTPException(status_code=400, detail=f"Local path does not exist: {registration.dataset_id}")
            
            # Get file info
            if path.is_file():
                stat = path.stat()
                size_human = _format_size(stat.st_size)
                fmt = path.suffix[1:].lower() if path.suffix else "unknown"
                
                # Validate file is not empty
                if stat.st_size == 0:
                    raise HTTPException(status_code=400, detail="Dataset file is empty (0 bytes)")
                
                # Validate file format is supported
                supported_exts = FileFormatConfig.get_all_extensions()
                if path.suffix.lower() not in supported_exts:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Unsupported file format: {path.suffix}. Supported: {', '.join(supported_exts)}"
                    )
                
                # Validate content structure for supported formats
                if path.suffix.lower() in ('.jsonl', '.json', '.csv'):
                    validation = await validate_dataset(str(path))
                    if not validation.valid:
                        error_detail = "Dataset content validation failed"
                        if validation.errors:
                            error_detail += ": " + "; ".join(validation.errors)
                        raise HTTPException(status_code=400, detail=error_detail)
                    total_samples = validation.total_samples or 0
                
                # Detect dataset type
                type_info = dataset_type_service.detect_dataset_type(str(path))
                if type_info.dataset_type.value == "unknown":
                    error_msg = (
                        "Unable to detect dataset type. Please ensure your dataset has recognized fields: "
                        "'messages' (SFT), 'instruction'+'output' (SFT), 'text' (pre-training), "
                        "'prompt'+'chosen'+'rejected' (RLHF/DPO), 'prompt'+'completion'+'label' (KTO)"
                    )
                    if type_info.validation_errors:
                        error_msg += ". Errors: " + "; ".join(type_info.validation_errors)
                    raise HTTPException(status_code=400, detail=error_msg)
                
                dataset_type_str = type_info.dataset_type.value
                dataset_type_display = type_info.display_name
                compatible_methods = type_info.compatible_training_methods
            else:
                size_human = "Directory"
                fmt = "directory"
        else:
            # For remote sources (huggingface, modelscope), detect type by streaming samples
            size_human = "Remote"
            fmt = "hub"
            try:
                type_info = dataset_type_service.detect_hub_dataset_type(
                    dataset_id=registration.dataset_id,
                    source=source_key,
                    subset=registration.subset,
                    split=registration.split or "train",
                    num_samples=10
                )
                if type_info.dataset_type.value == "unknown":
                    raise HTTPException(
                        status_code=400,
                        detail="Unable to detect dataset type from remote source. "
                               "Please verify the dataset ID, subset, and split are correct."
                    )
                dataset_type_str = type_info.dataset_type.value
                dataset_type_display = type_info.display_name
                compatible_methods = type_info.compatible_training_methods
            except HTTPException:
                raise
            except Exception as e:
                logger.warning(f"Could not detect type for remote dataset {registration.dataset_id}: {e}")
        
        # Create registered dataset entry
        registered = RegisteredDataset(
            id=dataset_id,
            name=registration.name,
            source=registration.source,
            path=registration.dataset_id,
            subset=registration.subset,
            split=registration.split,
            total_samples=total_samples,
            size_human=size_human,
            format=fmt,
            created_at=datetime.now().timestamp(),
            selected=True,
            max_samples=registration.max_samples if registration.max_samples and registration.max_samples > 0 else None,
            dataset_type=dataset_type_str,
            dataset_type_display=dataset_type_display,
            compatible_training_methods=compatible_methods,
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
    """List all datasets (uploaded + registered) with dataset type labels"""
    try:
        get_system_settings().UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        
        all_datasets = []
        
        # 1. Get uploaded files
        for f in get_system_settings().UPLOAD_DIR.glob("*"):
            if f.suffix.lower() in [".jsonl", ".json", ".csv"]:
                stat = f.stat()
                total_samples = 0
                
                # Detect dataset type
                try:
                    type_info = dataset_type_service.detect_dataset_type(str(f))
                    dataset_type = type_info.dataset_type.value
                    dataset_type_display = type_info.display_name
                    compatible_methods = type_info.compatible_training_methods
                    total_samples = type_info.sample_count
                except Exception as e:
                    # STRICT: Skip unknown datasets - they should not be listed
                    logger.warning(f"Skipping dataset {f.name}: type detection failed - {e}")
                    continue
                
                # STRICT: Skip datasets with unknown type
                if dataset_type == "unknown" or type_info.dataset_type.value == "unknown":
                    logger.warning(f"Skipping dataset {f.name}: unknown type not allowed")
                    continue
                
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
                    "selected": True,
                    "dataset_type": dataset_type,
                    "dataset_type_display": dataset_type_display,
                    "compatible_training_methods": compatible_methods,
                })
        
        # 2. Get registered datasets
        for ds in _dataset_registry.values():
            # Add dataset type if not already present
            if "dataset_type" not in ds:
                try:
                    type_info = dataset_type_service.detect_dataset_type(ds.get("path", ""))
                    ds["dataset_type"] = type_info.dataset_type.value
                    ds["dataset_type_display"] = type_info.display_name
                    ds["compatible_training_methods"] = type_info.compatible_training_methods
                except Exception as e:
                    # STRICT: Skip registered datasets with unknown type
                    logger.warning(f"Skipping registered dataset {ds.get('name', 'unknown')}: type detection failed - {e}")
                    continue
                
                # STRICT: Skip datasets with unknown type
                if ds.get("dataset_type") == "unknown":
                    logger.warning(f"Skipping registered dataset {ds.get('name', 'unknown')}: unknown type not allowed")
                    continue
            
            # Only add if type is valid
            if ds.get("dataset_type") != "unknown":
                all_datasets.append(ds)
        
        # Sort by creation time (newest first)
        all_datasets.sort(key=lambda x: x.get("created_at", 0), reverse=True)
        
        return {"datasets": all_datasets, "total": len(all_datasets)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail="Failed to list datasets")


@router.delete("/unregister/{dataset_id}")
async def unregister_dataset(dataset_id: str, confirm: str = Query(..., description="Type the dataset NAME to confirm")):
    """Unregister a dataset. User must type the dataset name to confirm."""
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


# ============================================================================
# Dataset Type Detection Endpoints
# ============================================================================

class DatasetTypeResponse(BaseModel):
    """Response model for dataset type detection"""
    dataset_type: str
    confidence: float
    detected_fields: List[str]
    sample_count: int
    compatible_training_methods: List[str]
    incompatible_training_methods: List[str]
    compatible_rlhf_types: List[str] = []
    display_name: str = ""
    message: str
    file_size_bytes: int = 0
    is_large_file: bool = False
    format_warning: Optional[str] = None
    file_format: Optional[str] = None
    supports_streaming: bool = False
    estimated_samples: bool = False
    validation_errors: List[str] = []
    validation_warnings: List[str] = []


class ModelTypeResponse(BaseModel):
    """Response model for model type detection"""
    model_type: str
    is_adapter: bool
    base_model_path: Optional[str]
    can_do_lora: bool
    can_do_qlora: bool
    can_do_full: bool
    can_do_rlhf: bool
    warnings: List[str]
    # New fields for runtime merge support
    can_merge_with_base: bool = False
    merge_unlocks_full: bool = False
    adapter_r: Optional[int] = None
    adapter_alpha: Optional[int] = None
    quantization_bits: Optional[int] = None


class AdapterValidationRequest(BaseModel):
    """Request for validating adapter + base model compatibility"""
    adapter_path: str
    base_model_path: str
    merge_before_training: bool = False  # If true, merge adapter into base at runtime


class AdapterValidationResponse(BaseModel):
    """Response for adapter validation"""
    valid: bool
    compatible: bool
    message: str
    adapter_base_model: Optional[str] = None  # Base model from adapter config
    provided_base_model: str
    can_merge: bool = False
    merge_warnings: List[str] = []
    # After merge, all options available
    after_merge_can_do_full: bool = True
    after_merge_can_do_rlhf: bool = True


class TrainingValidationRequest(BaseModel):
    """Request for validating training configuration"""
    dataset_path: str
    training_method: str  # sft, pt, rlhf
    train_type: str  # lora, qlora, full, adalora
    model_path: Optional[str] = None
    rlhf_type: Optional[str] = None  # dpo, orpo, etc.


class TrainingValidationResponse(BaseModel):
    """Response for training configuration validation"""
    valid: bool
    message: str
    dataset_type: str
    model_type: Optional[str] = None
    compatible_methods: List[str]
    incompatible_methods: List[str]
    warnings: List[str]


@router.get("/limits")
async def get_dataset_limits():
    """
    Get dataset size limits and recommendations for different data sources.
    
    Returns comprehensive information about:
    - Upload limits by method (direct, chunked, local path, hub)
    - Recommended approach by dataset size
    - Type detection method for each source
    - Training recommendations for massive datasets
    """
    return {
        "upload_limits": {
            "direct_upload": {
                "max_size_gb": 50,
                "description": "Standard file upload - loads entire file into memory",
                "recommended_for": "Datasets under 50GB",
                "type_detection": "Samples first 100 rows from file"
            },
            "chunked_upload": {
                "max_size_gb": 100,
                "description": "Streaming upload - writes chunks to disk",
                "recommended_for": "Datasets 50GB - 100GB",
                "type_detection": "Samples first 100 rows from file"
            },
            "local_path": {
                "max_size_gb": None,  # Unlimited
                "max_size_display": "UNLIMITED",
                "description": "Reference a local file path - no upload needed",
                "recommended_for": "Datasets > 100GB, including 100TB+ datasets",
                "type_detection": "Streams first 100 rows (memory-safe)"
            },
            "huggingface": {
                "max_size_gb": None,  # Unlimited
                "max_size_display": "UNLIMITED",
                "description": "Stream from HuggingFace Hub - no download needed",
                "recommended_for": "Any size - especially large public datasets",
                "type_detection": "Streams 10 samples only (fast)"
            },
            "modelscope": {
                "max_size_gb": None,  # Unlimited
                "max_size_display": "UNLIMITED",
                "description": "Stream from ModelScope - no download needed",
                "recommended_for": "Any size - especially Chinese/Asian datasets",
                "type_detection": "Streams 10 samples only (fast)"
            }
        },
        "recommendations_by_size": [
            {
                "size_range": "< 50GB",
                "recommended_method": "direct_upload",
                "reason": "Simple and fast for smaller datasets"
            },
            {
                "size_range": "50GB - 100GB",
                "recommended_method": "chunked_upload OR local_path",
                "reason": "Chunked upload streams to disk; local path avoids upload entirely"
            },
            {
                "size_range": "100GB - 1TB",
                "recommended_method": "local_path",
                "reason": "Too large for upload - reference local file directly"
            },
            {
                "size_range": "1TB - 100TB+",
                "recommended_method": "local_path OR huggingface/modelscope",
                "reason": "Massive datasets must use streaming - never fully loaded into memory"
            },
            {
                "size_range": "Billions of rows",
                "recommended_method": "local_path (JSONL) OR huggingface/modelscope",
                "reason": "Use --streaming=true flag during training, set --max_steps"
            }
        ],
        "training_recommendations": {
            "massive_datasets": {
                "format": "JSONL (required for streaming)",
                "flags": [
                    "--streaming true",
                    "--max_steps <number>",
                    "--shuffle_buffer_size 10000"
                ],
                "notes": [
                    "Dataset length unknown with streaming - must set max_steps",
                    "Shuffle buffer controls memory usage during randomization",
                    "Type detection always uses streaming (10-100 samples only)"
                ]
            }
        },
        "type_detection": {
            "method": "Sample-based (memory-safe)",
            "samples_for_local": 100,
            "samples_for_hub": 10,
            "note": "Type detection NEVER loads full dataset regardless of size"
        }
    }


@router.get("/detect-hub-type")
async def detect_hub_dataset_type(
    dataset_id: str = Query(..., description="HuggingFace or ModelScope dataset ID"),
    source: str = Query("huggingface", description="Source: 'huggingface' or 'modelscope'"),
    subset: Optional[str] = Query(None, description="Optional subset/config name"),
    split: str = Query("train", description="Split to sample from"),
    num_samples: int = Query(10, description="Number of samples to fetch for detection")
):
    """
    Detect dataset type from HuggingFace or ModelScope by streaming a small sample.
    
    Downloads only the first N samples (default: 10) to detect the dataset type
    WITHOUT downloading the entire dataset. This allows:
    1. Immediate type detection for showing available training options
    2. Validation before registering the dataset
    3. Fast feedback on dataset compatibility
    
    Example:
    - /datasets/detect-hub-type?dataset_id=tatsu-lab/alpaca&source=huggingface
    - /datasets/detect-hub-type?dataset_id=Anthropic/hh-rlhf&source=huggingface&subset=helpful-base
    """
    try:
        if source not in ["huggingface", "modelscope"]:
            raise HTTPException(status_code=400, detail="Source must be 'huggingface' or 'modelscope'")
        
        result = dataset_type_service.detect_hub_dataset_type(
            dataset_id=dataset_id,
            source=source,
            subset=subset,
            split=split,
            num_samples=num_samples
        )
        
        return {
            "success": True,
            "source": source,
            "dataset_id": dataset_id,
            "subset": subset,
            "split": split,
            "samples_fetched": result.sample_count,
            "dataset_type": result.dataset_type.value,
            "confidence": result.confidence,
            "detected_fields": result.detected_fields,
            "compatible_training_methods": result.compatible_training_methods,
            "incompatible_training_methods": result.incompatible_training_methods,
            "compatible_rlhf_types": result.compatible_rlhf_types,
            "display_name": result.display_name,
            "message": result.message,
        }
    except HTTPException:
        raise
    except Exception as e:
        # STRICT: Return error with no compatible methods - dataset cannot be used
        return {
            "success": False,
            "source": source,
            "dataset_id": dataset_id,
            "error": str(e),
            "dataset_type": "error",
            "confidence": 0.0,
            "detected_fields": [],
            "compatible_training_methods": [],  # NO methods allowed
            "incompatible_training_methods": list({"sft", "pt", "rlhf"}),  # ALL blocked (derived from empty compatible)
            "compatible_rlhf_types": [],
            "display_name": "Error - Cannot Use",
            "message": f"Dataset type detection failed. Cannot use this dataset: {str(e)}",
            "is_valid": False,
        }


@router.get("/detect-type", response_model=DatasetTypeResponse)
async def detect_dataset_type(dataset_path: str = Query(..., description="Path to dataset file")):
    """
    Detect the type of dataset (SFT, RLHF-Offline, RLHF-Online, PT).
    
    This is used by the frontend to:
    1. Restrict available training methods based on dataset type
    2. Auto-reset training method when dataset type changes
    3. Show appropriate warnings/info to users
    4. Label datasets with their type (SFT, RLHF-Online, RLHF-Offline, PT)
    """
    try:
        result = dataset_type_service.detect_dataset_type(dataset_path)
        return DatasetTypeResponse(
            dataset_type=result.dataset_type.value,
            confidence=result.confidence,
            detected_fields=result.detected_fields,
            sample_count=result.sample_count,
            compatible_training_methods=result.compatible_training_methods,
            incompatible_training_methods=result.incompatible_training_methods,
            compatible_rlhf_types=result.compatible_rlhf_types,
            display_name=result.display_name,
            message=result.message,
            file_size_bytes=result.file_size_bytes,
            is_large_file=result.is_large_file,
            format_warning=result.format_warning,
            file_format=result.file_format,
            supports_streaming=result.supports_streaming,
            estimated_samples=result.estimated_samples,
            validation_errors=result.validation_errors,
            validation_warnings=result.validation_warnings,
        )
    except Exception as e:
        # STRICT: Return error with no compatible methods - dataset cannot be used
        raise HTTPException(
            status_code=400,
            detail=f"Dataset type detection failed. This dataset cannot be used: {str(e)}"
        )


@router.get("/detect-model-type", response_model=ModelTypeResponse)
async def detect_model_type(model_path: str = Query(..., description="Path to model directory")):
    """
    Detect if a model is a full model or LoRA adapter.
    
    This is used to:
    1. Warn users if they try to do full fine-tuning on a LoRA adapter
    2. Ensure proper base model is available for LoRA adapter training
    3. Validate RLHF compatibility
    4. Check if adapter can be merged with base for full training options
    """
    try:
        result = dataset_type_service.detect_model_type(model_path)
        return ModelTypeResponse(
            model_type=result.model_type.value,
            is_adapter=result.is_adapter,
            base_model_path=result.base_model_path,
            can_do_lora=result.can_do_lora,
            can_do_qlora=result.can_do_qlora,
            can_do_full=result.can_do_full,
            can_do_rlhf=result.can_do_rlhf,
            warnings=result.warnings,
            can_merge_with_base=result.can_merge_with_base,
            merge_unlocks_full=result.merge_unlocks_full,
            adapter_r=result.adapter_r,
            adapter_alpha=result.adapter_alpha,
            quantization_bits=result.quantization_bits,
        )
    except Exception as e:
        return ModelTypeResponse(
            model_type="unknown",
            is_adapter=False,
            base_model_path=None,
            can_do_lora=True,
            can_do_qlora=True,
            can_do_full=True,
            can_do_rlhf=True,
            warnings=[f"Error detecting model type: {str(e)}"],
        )


@router.post("/validate-adapter-base", response_model=AdapterValidationResponse)
async def validate_adapter_base(request: AdapterValidationRequest):
    """
    Validate if an adapter is compatible with a provided base model.
    
    This is used when user wants to:
    1. Provide adapter + base model separately
    2. Merge adapter into base at runtime for full training options
    3. Validate compatibility before training starts
    """
    adapter_path = Path(request.adapter_path)
    base_path = Path(request.base_model_path)
    merge_warnings = []
    
    # Check adapter exists
    if not adapter_path.exists():
        return AdapterValidationResponse(
            valid=False,
            compatible=False,
            message=f"Adapter path does not exist: {request.adapter_path}",
            provided_base_model=request.base_model_path,
            can_merge=False,
        )
    
    # Check base model exists
    if not base_path.exists():
        return AdapterValidationResponse(
            valid=False,
            compatible=False,
            message=f"Base model path does not exist: {request.base_model_path}",
            provided_base_model=request.base_model_path,
            can_merge=False,
        )
    
    # Check adapter has adapter_config.json
    adapter_config_path = adapter_path / "adapter_config.json"
    if not adapter_config_path.exists():
        return AdapterValidationResponse(
            valid=False,
            compatible=False,
            message=f"Not a valid adapter: missing adapter_config.json in {request.adapter_path}",
            provided_base_model=request.base_model_path,
            can_merge=False,
        )
    
    # Check base has config.json
    base_config_path = base_path / "config.json"
    if not base_config_path.exists():
        return AdapterValidationResponse(
            valid=False,
            compatible=False,
            message=f"Not a valid base model: missing config.json in {request.base_model_path}",
            provided_base_model=request.base_model_path,
            can_merge=False,
        )
    
    # Read adapter config
    try:
        with open(adapter_config_path, 'r') as f:
            adapter_config = json.load(f)
    except Exception as e:
        return AdapterValidationResponse(
            valid=False,
            compatible=False,
            message=f"Could not read adapter config: {str(e)}",
            provided_base_model=request.base_model_path,
            can_merge=False,
        )
    
    adapter_base_model = adapter_config.get("base_model_name_or_path", "")
    
    # Check compatibility - compare model names (flexible matching)
    compatible = True
    if adapter_base_model:
        # Extract model name from full path (e.g., "meta-llama/Llama-3.1-8B" -> "llama-3.1-8b")
        adapter_base_name = adapter_base_model.split("/")[-1].lower().replace("_", "-")
        provided_base_name = request.base_model_path.split("/")[-1].lower().replace("_", "-")
        
        if adapter_base_name not in provided_base_name and provided_base_name not in adapter_base_name:
            compatible = False
            merge_warnings.append(
                f"Warning: Adapter was trained on '{adapter_base_model}' but you provided '{request.base_model_path}'. "
                "Merging may produce unexpected results if architectures differ."
            )
    else:
        merge_warnings.append(
            "Adapter config does not specify base model. Cannot verify compatibility automatically."
        )
    
    # Check for QLoRA - merging quantized adapters has limitations
    is_qlora = adapter_config.get("bits") in [4, 8] or adapter_config.get("load_in_4bit") or adapter_config.get("load_in_8bit")
    if is_qlora:
        merge_warnings.append(
            "This is a QLoRA adapter. Merging will produce a full-precision model. "
            "Ensure you have sufficient VRAM for the merged model."
        )
    
    message = "Adapter and base model are compatible for merging." if compatible else \
              "Adapter may not be compatible with the provided base model. Proceed with caution."
    
    return AdapterValidationResponse(
        valid=True,
        compatible=compatible,
        message=message,
        adapter_base_model=adapter_base_model,
        provided_base_model=request.base_model_path,
        can_merge=True,
        merge_warnings=merge_warnings,
        after_merge_can_do_full=True,
        after_merge_can_do_rlhf=True,
    )


@router.post("/validate-training-config", response_model=TrainingValidationResponse)
async def validate_training_config(request: TrainingValidationRequest):
    """
    Validate if the training configuration is compatible with the dataset and model.
    
    This provides a final validation before training starts, ensuring:
    1. Dataset type matches training method
    2. Model type supports the selected train type
    3. All compatibility requirements are met
    """
    try:
        # Detect dataset type
        dataset_result = dataset_type_service.detect_dataset_type(request.dataset_path)
        
        # Detect model type if path provided
        model_result = None
        model_type_str = None
        warnings = []
        
        if request.model_path:
            model_result = dataset_type_service.detect_model_type(request.model_path)
            model_type_str = model_result.model_type.value
            warnings.extend(model_result.warnings)
        
        # Validate configuration
        is_valid, message = dataset_type_service.validate_training_config(
            dataset_type=dataset_result.dataset_type,
            training_method=request.training_method,
            train_type=request.train_type,
            model_type=model_result.model_type if model_result else ModelType.FULL,
            rlhf_type=request.rlhf_type
        )
        
        return TrainingValidationResponse(
            valid=is_valid,
            message=message,
            dataset_type=dataset_result.dataset_type.value,
            model_type=model_type_str,
            compatible_methods=dataset_result.compatible_training_methods,
            incompatible_methods=dataset_result.incompatible_training_methods,
            warnings=warnings,
        )
    
    except Exception as e:
        # STRICT: Return error with no compatible methods
        return TrainingValidationResponse(
            valid=False,
            message=f"Validation error: {str(e)}. Dataset cannot be used.",
            dataset_type="error",
            model_type=None,
            compatible_methods=[],  # NO methods allowed
            incompatible_methods=["sft", "pt", "rlhf"],  # ALL blocked
            warnings=["Dataset type detection failed. Please use a supported format."],
        )


@router.get("/detect-type-bulk")
async def detect_dataset_types_bulk(
    paths: str = Query(..., description="Comma-separated list of dataset paths")
):
    """
    Detect types for multiple datasets at once.
    
    Returns the combined compatible training methods (intersection of all datasets).
    This is useful when multiple datasets are selected for training.
    """
    try:
        path_list = [p.strip() for p in paths.split(",") if p.strip()]
        
        if not path_list:
            # STRICT: No datasets = error, not allowed
            raise HTTPException(
                status_code=400,
                detail="No datasets provided. At least one valid dataset is required."
            )
        
        results = []
        types_found = set()
        compatible_sets = []
        
        for path in path_list:
            result = dataset_type_service.detect_dataset_type(path)
            results.append({
                "path": path,
                "type": result.dataset_type.value,
                "confidence": result.confidence,
                "compatible_methods": result.compatible_training_methods,
            })
            types_found.add(result.dataset_type.value)
            compatible_sets.append(set(result.compatible_training_methods))
        
        # Find intersection of compatible methods
        if compatible_sets:
            combined_compatible = set.intersection(*compatible_sets)
        else:
            combined_compatible = {"sft", "pt", "rlhf"}
        
        # Find union of incompatible methods
        all_incompatible = set()
        for result in results:
            ds_result = dataset_type_service.detect_dataset_type(result["path"])
            all_incompatible.update(ds_result.incompatible_training_methods)
        
        all_same_type = len(types_found) == 1
        detected_type = list(types_found)[0] if all_same_type else "mixed"
        
        message = ""
        if not all_same_type:
            message = f"Mixed dataset types detected: {', '.join(types_found)}. Only methods compatible with all datasets are available."
        elif detected_type != "unknown":
            type_display = {"sft": "SFT", "rlhf": "RLHF preference", "pt": "Pre-training", "kto": "KTO"}.get(detected_type, detected_type)
            message = f"All datasets are {type_display} format."
        
        return {
            "datasets": results,
            "combined_compatible_methods": list(combined_compatible),
            "combined_incompatible_methods": list(all_incompatible),
            "all_same_type": all_same_type,
            "detected_type": detected_type,
            "message": message
        }
    
    except HTTPException:
        raise
    except Exception as e:
        # STRICT: Return error with no compatible methods
        raise HTTPException(
            status_code=400,
            detail=f"Dataset type detection failed: {str(e)}. Please ensure all datasets have a supported format."
        )


@router.get("/supported-formats")
async def get_supported_formats():
    """
    Get information about supported dataset file formats.
    
    Returns format limits, streaming support, and recommendations.
    This endpoint helps the frontend display format requirements to users.
    
    Format Summary:
    - JSONL: UNLIMITED size, streaming, recommended for large datasets
    - JSON: 2GB max, no streaming, for smaller datasets only
    - CSV/TSV: UNLIMITED size, streaming, good for tabular data
    - TXT: UNLIMITED size, streaming, for pre-training text
    - Parquet: UNLIMITED size, chunked reading, efficient columnar format
    """
    return dataset_type_service.get_supported_formats()


# ============================================================
# ALGORITHM COMPATIBILITY ENDPOINTS
# ============================================================

@router.get("/algorithm-compatibility")
async def get_algorithm_compatibility():
    """
    Get complete algorithm compatibility information.
    
    Returns all dataset types, algorithms, training types, and quantization options
    with their compatibility relationships. Used by frontend for dynamic UI filtering.
    """
    return {
        "dataset_types": algorithm_compatibility_service.get_all_dataset_types(),
        "algorithms": algorithm_compatibility_service.get_all_algorithms(),
        "training_types": algorithm_compatibility_service.get_all_training_types(),
        "quantizations": algorithm_compatibility_service.get_all_quantizations(),
    }


@router.get("/compatible-algorithms/{dataset_type}")
async def get_compatible_algorithms(dataset_type: str):
    """
    Get algorithms compatible with a specific dataset type.
    
    Args:
        dataset_type: One of 'sft', 'rlhf_offline', 'rlhf_online', 'kto', 'pt'
    
    Returns:
        List of compatible algorithm configurations
    """
    algorithms = algorithm_compatibility_service.get_compatible_algorithms(dataset_type)
    methods = algorithm_compatibility_service.get_compatible_training_methods(dataset_type)
    
    ds_config = DATASET_TYPE_CONFIGS.get(dataset_type)
    
    return {
        "dataset_type": dataset_type,
        "display_name": ds_config.display_name if ds_config else dataset_type,
        "description": ds_config.description if ds_config else "",
        "compatible_training_methods": methods,
        "compatible_algorithms": algorithms,
        "is_rlhf": "rlhf" in methods,
    }


class TrainingConfigValidation(BaseModel):
    """Request model for training configuration validation"""
    dataset_type: str
    training_method: str
    rlhf_algorithm: Optional[str] = None
    training_type: str = "lora"
    quantization: str = "none"


@router.post("/validate-training-config")
async def validate_training_config(config: TrainingConfigValidation):
    """
    Validate a complete training configuration before starting training.
    
    Checks:
    1. Dataset type compatibility with training method
    2. RLHF algorithm compatibility with dataset type
    3. Training type compatibility with algorithm
    4. Quantization compatibility with training type
    
    Returns:
        - valid: Whether the configuration is valid
        - errors: List of blocking errors
        - warnings: List of non-blocking warnings
        - suggestions: List of improvement suggestions
    """
    result = algorithm_compatibility_service.validate_training_config(
        dataset_type=config.dataset_type,
        training_method=config.training_method,
        rlhf_algorithm=config.rlhf_algorithm,
        training_type=config.training_type,
        quantization=config.quantization,
    )
    
    return result


# ============================================================
# SAMPLE DATASET ENDPOINTS
# ============================================================

# Sample datasets directory - with fallback for Docker environments
def _get_sample_datasets_dir() -> Path:
    """Get sample datasets directory with multiple fallback paths."""
    # Try relative path from this file first
    relative_path = Path(__file__).parent.parent.parent.parent.parent.parent / "examples" / "sample_datasets"
    if relative_path.exists():
        return relative_path
    
    # Try environment variable
    import os
    env_path = os.environ.get("USF_SAMPLE_DATASETS_DIR")
    if env_path and Path(env_path).exists():
        return Path(env_path)
    
    # Try common Docker paths
    docker_paths = [
        Path("/app/examples/sample_datasets"),
        Path("/workspace/examples/sample_datasets"),
        Path("/usf-bios/examples/sample_datasets"),
    ]
    for path in docker_paths:
        if path.exists():
            return path
    
    # Fallback to relative path (may not exist but prevents crashes)
    return relative_path

SAMPLE_DATASETS_DIR = _get_sample_datasets_dir()

SAMPLE_DATASET_INFO = {
    # ========== SFT - MESSAGES FORMAT ==========
    "sft_messages": {
        "filename": "sft_messages.jsonl",
        "display_name": "SFT - Messages Format",
        "description": "Standard chat format with system/user/assistant messages. Most common format for instruction tuning.",
        "format_description": "Messages array with role (system/user/assistant) and content pairs.",
        "format_example": '{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}',
        "compatible_methods": ["sft"],
        "compatible_algorithms": [],
        "usf_bios_verified": True,
    },
    
    # ========== SFT - ALPACA FORMAT ==========
    "sft_alpaca": {
        "filename": "sft_alpaca.jsonl",
        "display_name": "SFT - Alpaca Format",
        "description": "Instruction/Input/Output format popularized by Stanford Alpaca. Good for task-specific training.",
        "format_description": "instruction (required), input (optional context), output (required response).",
        "format_example": '{"instruction": "Summarize this text", "input": "Long text...", "output": "Summary..."}',
        "compatible_methods": ["sft"],
        "compatible_algorithms": [],
        "usf_bios_verified": True,
    },
    
    # ========== PRE-TRAINING ==========
    "pt": {
        "filename": "pt_raw_text.jsonl",
        "display_name": "Pre-Training - Raw Text",
        "description": "Raw text data for continued pre-training or domain adaptation. No instruction format needed.",
        "format_description": "Single 'text' field containing raw text for language modeling.",
        "format_example": '{"text": "This is raw text for pre-training..."}',
        "compatible_methods": ["pt"],
        "compatible_algorithms": [],
        "usf_bios_verified": True,
    },
    
    # ========== RLHF - PREFERENCE (SIMPLE) ==========
    "rlhf_pref_simple": {
        "filename": "rlhf_offline_preference.jsonl",
        "display_name": "RLHF Preference - Simple (prompt/chosen/rejected)",
        "description": "Simple preference format with prompt and chosen/rejected response strings. Common HuggingFace DPO format.",
        "format_description": "prompt (question), chosen (good response), rejected (bad response).",
        "format_example": '{"prompt": "Question?", "chosen": "Good answer", "rejected": "Bad answer"}',
        "compatible_methods": ["rlhf"],
        "compatible_algorithms": ["dpo", "orpo", "simpo", "cpo", "rm"],
        "usf_bios_verified": True,
    },
    
    # ========== RLHF - PREFERENCE (MESSAGES) ==========
    "rlhf_pref_messages": {
        "filename": "rlhf_pref_messages.jsonl",
        "display_name": "RLHF Preference - Messages Format",
        "description": "Preference pairs with full message arrays. Better for multi-turn conversations.",
        "format_description": "messages (chosen conversation), rejected_messages (rejected conversation).",
        "format_example": '{"messages": [...], "rejected_messages": [...]}',
        "compatible_methods": ["rlhf"],
        "compatible_algorithms": ["dpo", "orpo", "simpo", "cpo", "rm"],
        "usf_bios_verified": True,
    },
    
    # ========== RLHF - KTO (BINARY) ==========
    "rlhf_kto": {
        "filename": "kto_messages.jsonl",
        "display_name": "RLHF KTO - Binary Feedback",
        "description": "Binary feedback with messages + label. Each sample is either desirable (true) or undesirable (false).",
        "format_description": "messages array + label (true/false or 1/0).",
        "format_example": '{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}], "label": true}',
        "compatible_methods": ["rlhf"],
        "compatible_algorithms": ["kto"],
        "usf_bios_verified": True,
    },
    
    # ========== RLHF - KTO (LEGACY) ==========
    "rlhf_kto_legacy": {
        "filename": "rlhf_binary_feedback.jsonl",
        "display_name": "RLHF KTO - Legacy Format",
        "description": "Legacy KTO format with prompt/completion/label. Older format still supported.",
        "format_description": "prompt, completion, and boolean label.",
        "format_example": '{"prompt": "...", "completion": "...", "label": true}',
        "compatible_methods": ["rlhf"],
        "compatible_algorithms": ["kto"],
        "usf_bios_verified": True,
    },
    
    # ========== RLHF - ONLINE ==========
    "rlhf_online": {
        "filename": "rlhf_online_prompt.jsonl",
        "display_name": "RLHF Online - Prompts Only (PPO/GRPO/GKD)",
        "description": "Prompts only - model generates responses during training. Used with reward models or functions.",
        "format_description": "Just prompts - model generates completions during training.",
        "format_example": '{"prompt": "Write a poem about..."} or {"messages": [{"role": "user", "content": "..."}]}',
        "compatible_methods": ["rlhf"],
        "compatible_algorithms": ["ppo", "grpo", "gkd"],
        "usf_bios_verified": True,
    },
    
    # ========== ITERATIVE SELF-TRAINING (ReST/STaR) ==========
    "iterative_prompts": {
        "filename": "iterative_prompts.jsonl",
        "display_name": "Iterative Training - Prompts with Metadata",
        "description": "Prompts for iterative self-training (ReST/STaR/Expert Iteration). Supports difficulty curriculum and verifiable answers.",
        "format_description": "Prompts with optional difficulty level, expected answer (for rule-based verification), and metadata.",
        "format_example": '{"prompt": "Solve: 2+2=?", "difficulty": "easy", "metadata": {"expected_answer": "4", "category": "math"}}',
        "compatible_methods": ["rlhf", "iterative"],
        "compatible_algorithms": ["rest", "star", "expert_iteration"],
        "usf_bios_verified": True,
    },
    
    "iterative_curriculum": {
        "filename": "iterative_curriculum.jsonl",
        "display_name": "Iterative Training - Curriculum Format",
        "description": "Multi-difficulty dataset for curriculum-based iterative training. Model progresses from easy to hard problems.",
        "format_description": "Prompts organized by difficulty with optional verification data for rule-based scoring.",
        "format_example": '{"prompt": "...", "difficulty": "easy|medium|hard|expert", "metadata": {"expected_answer": "...", "test_cases": [...]}}',
        "compatible_methods": ["rlhf", "iterative"],
        "compatible_algorithms": ["rest", "star", "expert_iteration"],
        "usf_bios_verified": True,
    },
}


# Documentation directory
SAMPLE_DOCS_DIR = SAMPLE_DATASETS_DIR / "docs"

# Documentation file mapping
SAMPLE_DOCS_MAP = {
    "sft_messages": "sft_messages.md",
    "sft_alpaca": "sft_alpaca.md",
    "pt": "pt_raw_text.md",
    "rlhf_pref_simple": "rlhf_pref_simple.md",
    "rlhf_pref_messages": "rlhf_pref_messages.md",
    "rlhf_kto": "rlhf_kto.md",
    "rlhf_kto_legacy": "rlhf_kto.md",  # Same doc for both KTO formats
    "rlhf_online": "rlhf_online.md",
    "iterative_prompts": "iterative_training.md",
    "iterative_curriculum": "iterative_training.md",  # Same doc for both iterative formats
}


@router.get("/sample-datasets")
async def get_sample_datasets():
    """
    Get information about available sample datasets.
    
    Returns metadata for all sample datasets including:
    - Dataset type and display name
    - Description and format
    - Compatible training methods and algorithms
    - Sample preview (first few rows)
    - Documentation availability
    """
    samples = []
    
    for dataset_type, info in SAMPLE_DATASET_INFO.items():
        sample_path = SAMPLE_DATASETS_DIR / info["filename"]
        
        # Get sample preview
        preview = []
        if sample_path.exists():
            try:
                with open(sample_path, 'r', encoding='utf-8') as f:
                    for i, line in enumerate(f):
                        if i >= 2:  # Only first 2 samples for preview
                            break
                        preview.append(json.loads(line.strip()))
            except Exception:
                pass
        
        # Check for documentation
        doc_file = SAMPLE_DOCS_MAP.get(dataset_type)
        has_docs = doc_file and (SAMPLE_DOCS_DIR / doc_file).exists()
        
        samples.append({
            "type": dataset_type,
            "filename": info["filename"],
            "display_name": info["display_name"],
            "description": info["description"],
            "format_description": info["format_description"],
            "format_example": info.get("format_example", ""),
            "compatible_methods": info["compatible_methods"],
            "compatible_algorithms": info["compatible_algorithms"],
            "usf_bios_verified": info.get("usf_bios_verified", False),
            "preview": preview,
            "download_available": sample_path.exists(),
            "documentation_available": has_docs,
        })
    
    return {
        "samples": samples,
        "total": len(samples),
    }


@router.get("/sample-datasets/{dataset_type}")
async def get_sample_dataset_info(dataset_type: str):
    """
    Get detailed information about a specific sample dataset.
    
    Args:
        dataset_type: One of 'sft', 'rlhf_offline', 'rlhf_online', 'kto', 'pt'
    """
    if dataset_type not in SAMPLE_DATASET_INFO:
        raise HTTPException(status_code=404, detail=f"Sample dataset not found for type: {dataset_type}")
    
    info = SAMPLE_DATASET_INFO[dataset_type]
    sample_path = SAMPLE_DATASETS_DIR / info["filename"]
    
    # Get full content preview (up to 10 samples)
    preview = []
    total_samples = 0
    if sample_path.exists():
        try:
            with open(sample_path, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    total_samples += 1
                    if i < 10:
                        preview.append(json.loads(line.strip()))
        except Exception:
            pass
    
    # Get algorithm details for RLHF types
    algorithm_details = []
    for algo_id in info["compatible_algorithms"]:
        algo_config = ALGORITHM_REGISTRY.get(algo_id)
        if algo_config:
            algorithm_details.append({
                "id": algo_id,
                "name": algo_config.name,
                "description": algo_config.description,
                "category": algo_config.category,
                "sample_format": algo_config.sample_format,
            })
    
    return {
        "id": dataset_type,
        "filename": info["filename"],
        "display_name": info["display_name"],
        "description": info["description"],
        "format_description": info["format_description"],
        "compatible_methods": info["compatible_methods"],
        "compatible_algorithms": info["compatible_algorithms"],
        "algorithm_details": algorithm_details,
        "preview": preview,
        "total_samples": total_samples,
        "download_available": sample_path.exists(),
    }


@router.get("/sample-datasets/{dataset_type}/download")
async def download_sample_dataset(dataset_type: str):
    """
    Download a sample dataset file.
    
    Args:
        dataset_type: One of 'sft', 'rlhf_offline', 'rlhf_online', 'kto', 'pt'
    """
    if dataset_type not in SAMPLE_DATASET_INFO:
        raise HTTPException(status_code=404, detail=f"Sample dataset not found for type: {dataset_type}")
    
    info = SAMPLE_DATASET_INFO[dataset_type]
    sample_path = SAMPLE_DATASETS_DIR / info["filename"]
    
    if not sample_path.exists():
        raise HTTPException(status_code=404, detail=f"Sample file not found: {info['filename']}")
    
    return FileResponse(
        path=str(sample_path),
        filename=info["filename"],
        media_type="application/jsonl",
    )


@router.get("/sample-datasets/{dataset_type}/docs")
async def get_sample_dataset_documentation(dataset_type: str):
    """
    Get the full markdown documentation for a sample dataset format.
    
    Returns comprehensive documentation including:
    - Format structure and required fields
    - Compatible algorithms and training methods
    - Use cases and recommendations
    - Example data
    - Training considerations
    - Troubleshooting guide
    
    Args:
        dataset_type: Dataset type identifier
    """
    doc_file = SAMPLE_DOCS_MAP.get(dataset_type)
    if not doc_file:
        raise HTTPException(status_code=404, detail=f"Documentation not found for type: {dataset_type}")
    
    doc_path = SAMPLE_DOCS_DIR / doc_file
    if not doc_path.exists():
        raise HTTPException(status_code=404, detail=f"Documentation file not found: {doc_file}")
    
    try:
        with open(doc_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading documentation: {str(e)}")
    
    # Get basic info for context
    info = SAMPLE_DATASET_INFO.get(dataset_type, {})
    
    return {
        "dataset_type": dataset_type,
        "display_name": info.get("display_name", dataset_type),
        "documentation": content,
        "format": "markdown",
        "version": "1.0.0",
    }


@router.get("/sample-datasets-docs/overview")
async def get_sample_datasets_overview_docs():
    """
    Get the overview/README documentation for all sample datasets.
    
    Returns the main README.md that explains all formats and the training pipeline.
    """
    readme_path = SAMPLE_DOCS_DIR / "README.md"
    if not readme_path.exists():
        raise HTTPException(status_code=404, detail="Overview documentation not found")
    
    try:
        with open(readme_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading documentation: {str(e)}")
    
    return {
        "title": "USF-BIOS Dataset Format Documentation",
        "documentation": content,
        "format": "markdown",
        "version": "1.0.0",
    }


@router.get("/sample-format/{algorithm}")
async def get_sample_format_for_algorithm(algorithm: str):
    """
    Get the expected dataset format for a specific RLHF algorithm.
    
    Args:
        algorithm: One of 'dpo', 'orpo', 'simpo', 'kto', 'cpo', 'rm', 'ppo', 'grpo', 'gkd'
    """
    algo_config = ALGORITHM_REGISTRY.get(algorithm)
    if not algo_config:
        raise HTTPException(status_code=404, detail=f"Algorithm not found: {algorithm}")
    
    return {
        "algorithm": algorithm,
        "name": algo_config.name,
        "description": algo_config.description,
        "category": algo_config.category,
        "required_fields": algo_config.required_fields,
        "optional_fields": algo_config.optional_fields,
        "sample_format": algo_config.sample_format,
        "supports_full": algo_config.supports_full,
        "supports_lora": algo_config.supports_lora,
        "supports_qlora": algo_config.supports_qlora,
        "requires_ref_model": algo_config.requires_ref_model,
        "requires_reward_model": algo_config.requires_reward_model,
        "requires_vllm": algo_config.requires_vllm,
    }
