# Copyright (c) US Inc. All rights reserved.
"""Dataset-related endpoints"""

import json
import os
from pathlib import Path
from typing import List

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from ...core.config import settings
from ...models.schemas import DatasetValidation

router = APIRouter()


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
            errors=[f"Error validating dataset: {str(e)}"]
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
            errors=[str(e)]
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
            errors=[str(e)]
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
            errors=[str(e)]
        )


@router.post("/upload")
async def upload_dataset(file: UploadFile = File(...)):
    """Upload a dataset file"""
    try:
        # Validate file extension
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        suffix = Path(file.filename).suffix.lower()
        if suffix not in [".jsonl", ".json", ".csv"]:
            raise HTTPException(status_code=400, detail=f"Unsupported format: {suffix}")
        
        # Save file
        upload_path = settings.UPLOAD_DIR / file.filename
        
        content = await file.read()
        with open(upload_path, "wb") as f:
            f.write(content)
        
        return {
            "success": True,
            "filename": file.filename,
            "path": str(upload_path),
            "size": len(content)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/list")
async def list_datasets():
    """List uploaded datasets"""
    try:
        datasets = []
        for f in settings.UPLOAD_DIR.glob("*"):
            if f.suffix.lower() in [".jsonl", ".json", ".csv"]:
                datasets.append({
                    "name": f.name,
                    "path": str(f),
                    "size": f.stat().st_size,
                })
        return {"datasets": datasets}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
