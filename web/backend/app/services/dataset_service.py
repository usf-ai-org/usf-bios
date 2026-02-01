import csv
import json
import os
import shutil
from datetime import datetime
from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.models.db_models import Dataset, DatasetStatus, DatasetSource, TrainingJob
from app.core.config import settings
from app.services.system_encrypted_log_service import system_encrypted_log


class DatasetService:
    
    STORAGE_PATH = os.getenv("DATASET_STORAGE_PATH", "/app/data/datasets")
    ALLOWED_FORMATS = {"json", "jsonl", "csv", "parquet", "txt"}
    MAX_FILE_SIZE_MB = 10000
    
    def __init__(self, db: Session):
        self.db = db
        os.makedirs(self.STORAGE_PATH, exist_ok=True)
    
    def list_datasets(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dataset]:
        query = self.db.query(Dataset)
        
        if status:
            query = query.filter(Dataset.status == status)
        else:
            query = query.filter(Dataset.status != DatasetStatus.DELETED.value)
        
        return query.order_by(Dataset.created_at.desc()).offset(offset).limit(limit).all()
    
    def get_dataset(self, dataset_id: str) -> Optional[Dataset]:
        return self.db.query(Dataset).filter(Dataset.id == dataset_id).first()
    
    def create_dataset(
        self,
        name: str,
        file_name: str,
        file_content: bytes,
        description: Optional[str] = None
    ) -> Dataset:
        file_ext = file_name.split(".")[-1].lower()
        if file_ext not in self.ALLOWED_FORMATS:
            raise ValueError(f"Unsupported format: {file_ext}. Allowed: {self.ALLOWED_FORMATS}")
        
        file_size = len(file_content)
        if file_size > self.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise ValueError(f"File too large. Max: {self.MAX_FILE_SIZE_MB}MB")
        
        dataset = Dataset(
            name=name,
            description=description,
            source=DatasetSource.UPLOAD.value,
            file_name=file_name,
            file_format=file_ext,
            file_size=file_size,
            file_path="",
            status=DatasetStatus.UPLOADING.value,
        )
        self.db.add(dataset)
        self.db.flush()
        
        dataset_dir = os.path.join(self.STORAGE_PATH, dataset.id)
        os.makedirs(dataset_dir, exist_ok=True)
        
        file_path = os.path.join(dataset_dir, file_name)
        with open(file_path, "wb") as f:
            f.write(file_content)
        
        dataset.file_path = file_path
        dataset.status = DatasetStatus.PROCESSING.value
        
        try:
            stats = self._analyze_dataset(file_path, file_ext)
            dataset.num_samples = stats.get("num_samples")
            dataset.num_columns = stats.get("num_columns")
            dataset.column_info = stats.get("column_info")
            dataset.status = DatasetStatus.READY.value
            
            # Log successful dataset upload (encrypted only)
            system_encrypted_log.log_dataset_upload(
                filename=file_name,
                file_size=file_size,
                success=True,
                dataset_id=dataset.id
            )
            system_encrypted_log.log_dataset_validation(
                dataset_path=file_path,
                success=True,
                row_count=stats.get("num_samples", 0),
                format_type=file_ext
            )
        except Exception as e:
            dataset.status = DatasetStatus.ERROR.value
            # Don't expose internal error details
            dataset.error_message = "Failed to process dataset. Please check the file format."
            
            # Log failed dataset processing (encrypted only)
            system_encrypted_log.log_dataset_upload(
                filename=file_name,
                file_size=file_size,
                success=False,
                dataset_id=dataset.id,
                error=str(e)
            )
        
        self.db.commit()
        return dataset
    
    def update_dataset(
        self,
        dataset_id: str,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> Optional[Dataset]:
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            return None
        
        if name:
            dataset.name = name
        if description is not None:
            dataset.description = description
        
        dataset.updated_at = datetime.utcnow()
        self.db.commit()
        return dataset
    
    def delete_dataset(self, dataset_id: str, hard_delete: bool = False, force: bool = False) -> Dict[str, Any]:
        """Delete a dataset.
        
        Args:
            dataset_id: ID of dataset to delete
            hard_delete: If True, permanently delete from DB and disk. If False, soft delete (mark as DELETED)
            force: If True, delete even if dataset is used by trainings (trainings will keep reference)
        
        Returns:
            Dict with success status and details
        """
        dataset = self.get_dataset(dataset_id)
        if not dataset:
            return {"success": False, "error": "Dataset not found"}
        
        # Check if dataset is used by any trainings
        trainings_using = self.db.query(TrainingJob).filter(TrainingJob.dataset_id == dataset_id).count()
        if trainings_using > 0 and not force:
            return {
                "success": False,
                "error": f"Dataset is used by {trainings_using} training(s). Use force=True to delete anyway.",
                "trainings_using": trainings_using
            }
        
        result = {
            "success": True,
            "dataset_id": dataset_id,
            "dataset_name": dataset.name,
            "source": dataset.source,
            "hard_delete": hard_delete,
            "trainings_affected": trainings_using
        }
        
        if hard_delete:
            # Only delete files for uploaded/local datasets
            if dataset.source in [DatasetSource.UPLOAD.value, DatasetSource.LOCAL.value]:
                if dataset.file_path and os.path.exists(dataset.file_path):
                    dataset_dir = os.path.dirname(dataset.file_path)
                    if os.path.exists(dataset_dir):
                        shutil.rmtree(dataset_dir, ignore_errors=True)
                        result["files_deleted"] = True
            else:
                result["files_deleted"] = False  # HF/MS datasets have no local files
            
            self.db.delete(dataset)
            result["message"] = "Dataset permanently deleted"
        else:
            dataset.status = DatasetStatus.DELETED.value
            dataset.updated_at = datetime.utcnow()
            result["message"] = "Dataset marked as deleted (soft delete)"
        
        self.db.commit()
        
        # Log dataset deletion (encrypted only)
        system_encrypted_log.log_dataset_delete(
            dataset_path=dataset.file_path or dataset_id,
            success=True
        )
        
        return result
    
    def get_dataset_preview(self, dataset_id: str, num_rows: int = 10) -> Dict[str, Any]:
        dataset = self.get_dataset(dataset_id)
        if not dataset or not os.path.exists(dataset.file_path):
            return {"error": "Dataset not found"}
        
        try:
            return self._read_preview(dataset.file_path, dataset.file_format, num_rows)
        except Exception as e:
            # Don't expose internal error details
            return {"error": "Failed to read dataset preview"}
    
    def register_external_dataset(
        self,
        name: str,
        source: str,
        source_id: str,
        subset: Optional[str] = None,
        split: Optional[str] = "train",
        description: Optional[str] = None,
        max_samples: Optional[int] = None
    ) -> Dataset:
        """Register a dataset from HuggingFace, ModelScope, or local path.
        
        Args:
            name: Display name for the dataset
            source: Source type - 'huggingface', 'modelscope', or 'local'
            source_id: HF/MS dataset ID or local path
            subset: Optional subset name (for HF datasets)
            split: Split to use (train/test/validation)
            description: Optional description
            max_samples: Optional max samples to use (0 = all)
        """
        if source not in [DatasetSource.HUGGINGFACE.value, DatasetSource.MODELSCOPE.value, DatasetSource.LOCAL.value]:
            raise ValueError(f"Invalid source: {source}. Must be 'huggingface', 'modelscope', or 'local'")
        
        # For local paths, verify existence
        if source == DatasetSource.LOCAL.value:
            if not os.path.exists(source_id):
                raise ValueError(f"Local path does not exist: {source_id}")
        
        dataset = Dataset(
            name=name,
            description=description,
            source=source,
            source_id=source_id,
            source_subset=subset,
            source_split=split,
            file_path=source_id if source == DatasetSource.LOCAL.value else None,
            file_format="hub" if source != DatasetSource.LOCAL.value else None,
            status=DatasetStatus.READY.value,
        )
        
        self.db.add(dataset)
        self.db.commit()
        return dataset
    
    def list_datasets_by_source(self, source: Optional[str] = None) -> List[Dict[str, Any]]:
        """List datasets grouped by source."""
        query = self.db.query(Dataset).filter(Dataset.status != DatasetStatus.DELETED.value)
        
        if source:
            query = query.filter(Dataset.source == source)
        
        datasets = query.order_by(Dataset.created_at.desc()).all()
        
        result = []
        for ds in datasets:
            result.append({
                "id": ds.id,
                "name": ds.name,
                "description": ds.description,
                "source": ds.source,
                "source_id": ds.source_id,
                "source_subset": ds.source_subset,
                "source_split": ds.source_split,
                "file_path": ds.file_path,
                "file_format": ds.file_format,
                "num_samples": ds.num_samples,
                "status": ds.status,
                "created_at": ds.created_at.isoformat() if ds.created_at else None,
                "is_external": ds.source in [DatasetSource.HUGGINGFACE.value, DatasetSource.MODELSCOPE.value],
                "trainings_count": len(ds.training_jobs) if ds.training_jobs else 0
            })
        
        return result
    
    def get_storage_stats(self) -> Dict[str, Any]:
        total_size = 0
        dataset_count = 0
        
        for root, dirs, files in os.walk(self.STORAGE_PATH):
            for f in files:
                total_size += os.path.getsize(os.path.join(root, f))
            dataset_count = len(dirs) if root == self.STORAGE_PATH else dataset_count
        
        return {
            "storage_path": self.STORAGE_PATH,
            "total_size_mb": total_size / (1024 * 1024),
            "dataset_count": dataset_count,
        }
    
    def _analyze_dataset(self, file_path: str, file_format: str) -> Dict[str, Any]:
        stats = {"num_samples": 0, "num_columns": 0, "column_info": None}
        
        if file_format == "json":
            with open(file_path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                stats["num_samples"] = len(data)
                if data and isinstance(data[0], dict):
                    stats["num_columns"] = len(data[0])
                    stats["column_info"] = list(data[0].keys())
        
        elif file_format == "jsonl":
            with open(file_path, "r") as f:
                lines = f.readlines()
            stats["num_samples"] = len(lines)
            if lines:
                first = json.loads(lines[0])
                if isinstance(first, dict):
                    stats["num_columns"] = len(first)
                    stats["column_info"] = list(first.keys())
        
        elif file_format == "csv":
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                header = next(reader, None)
                if header:
                    stats["column_info"] = header
                    stats["num_columns"] = len(header)
                    stats["num_samples"] = sum(1 for _ in reader)
        
        elif file_format == "txt":
            with open(file_path, "r") as f:
                lines = f.readlines()
            stats["num_samples"] = len(lines)
        
        return stats
    
    def _read_preview(self, file_path: str, file_format: str, num_rows: int) -> Dict[str, Any]:
        preview = {"rows": [], "columns": None}
        
        if file_format == "json":
            with open(file_path, "r") as f:
                data = json.load(f)
            if isinstance(data, list):
                preview["rows"] = data[:num_rows]
                if data and isinstance(data[0], dict):
                    preview["columns"] = list(data[0].keys())
        
        elif file_format == "jsonl":
            with open(file_path, "r") as f:
                for i, line in enumerate(f):
                    if i >= num_rows:
                        break
                    preview["rows"].append(json.loads(line))
            if preview["rows"] and isinstance(preview["rows"][0], dict):
                preview["columns"] = list(preview["rows"][0].keys())
        
        elif file_format == "csv":
            with open(file_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                preview["columns"] = reader.fieldnames or []
                for i, row in enumerate(reader):
                    if i >= num_rows:
                        break
                    preview["rows"].append(dict(row))
        
        elif file_format == "txt":
            with open(file_path, "r") as f:
                preview["rows"] = [{"text": line.strip()} for line in f.readlines()[:num_rows]]
            preview["columns"] = ["text"]
        
        return preview
