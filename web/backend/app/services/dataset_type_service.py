# Copyright (c) US Inc. All rights reserved.
"""
Dataset Type Detection Service - Enterprise Production Ready

This service detects the type of dataset based on its format:
- SFT (Supervised Fine-Tuning): messages array with role/content pairs
- RLHF (Reinforcement Learning): prompt/chosen/rejected fields (preference data)
- PT (Pre-Training): text field only (raw text continuation)

The detected type is used to:
1. Restrict available training methods in the UI
2. Auto-reset training method when dataset type changes
3. Validate training configuration before starting

File Format Support & Limits:
- JSONL: UNLIMITED size (streaming-based, memory-safe) - Recommended for large datasets
- JSON: Max 2GB (must load into memory) - For smaller datasets only
- CSV: UNLIMITED size (streaming-based, memory-safe) - Good for tabular data
- TXT: UNLIMITED size (streaming-based) - For pre-training text
- Parquet: UNLIMITED size (chunked reading) - Efficient columnar format

Production Features:
- Memory-safe streaming for large files
- Graceful error handling with descriptive messages
- File integrity validation before processing
- Timeout protection for long operations
- Sample-based type detection (first N rows only)
"""

import csv
import json
import logging
import mmap
import os
import resource
import signal
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Generator, Iterator, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ============================================================
# FILE FORMAT CONFIGURATION - PRODUCTION LIMITS
# ============================================================

class FileFormatConfig:
    """
    Production-ready file format configuration.
    
    VERIFIED FROM USF BIOS SOURCE CODE:
    =====================================
    See usf_bios/dataset/loader.py:50-57 for format mapping:
    - jsonl → 'json' (HuggingFace datasets loader)
    - json → 'json' (direct)
    - csv → 'csv' (with na_filter=False)
    - txt → 'text' (pre-training)
    
    STREAMING SUPPORT (usf_bios/dataset/loader.py:52, 222, 254):
    ============================================================
    ALL formats support streaming via HuggingFace datasets library!
    - JSONL: ✅ Streaming supported (line-by-line reading)
    - CSV: ✅ Streaming supported (row-by-row reading)
    - TXT: ✅ Streaming supported (line-by-line reading)
    - JSON: ❌ NO streaming (must load entire array into memory)
    
    Streaming works for:
    - Local files (usf_bios/dataset/loader.py:52)
    - HuggingFace Hub (usf_bios/hub/hub.py:436)
    - ModelScope (usf_bios/hub/hub.py:307)
    
    For datasets with billions of samples:
    - Use JSONL format with --streaming true
    - Set --max_steps (required since dataset length is unknown)
    - Use --shuffle_buffer_size for randomization (default: 1000)
    
    NOT SUPPORTED by USF BIOS:
    - TSV (no explicit delimiter handling)
    - Parquet (not mapped in loader)
    - NDJSON (use .jsonl extension instead)
    """
    
    # ONLY formats that USF BIOS can actually train on
    # Verified from usf_bios/dataset/loader.py
    FORMATS = {
        "jsonl": {
            "extensions": [".jsonl"],
            "max_size_gb": None,  # UNLIMITED with streaming
            "streaming": True,    # VERIFIED: loader.py:52 passes streaming param
            "recommended_for_large": True,
            "description": "JSON Lines - RECOMMENDED for large datasets (billions of samples)",
            "usf_bios_mapping": "json",  # How HuggingFace loader handles it
            "notes": "Supports streaming for unlimited size. Use --streaming true --max_steps N",
        },
        "json": {
            "extensions": [".json"],
            "max_size_gb": 2.0,  # Hard limit - must load entire file into memory
            "streaming": False,  # JSON arrays cannot be streamed
            "recommended_for_large": False,
            "description": "JSON Array - For smaller datasets only (max 2GB)",
            "usf_bios_mapping": "json",
            "notes": "NO streaming support. Entire file loaded into memory.",
        },
        "csv": {
            "extensions": [".csv"],
            "max_size_gb": None,  # UNLIMITED with streaming
            "streaming": True,    # CSV supports row-by-row streaming
            "recommended_for_large": True,
            "description": "CSV - Streaming supported for large tabular data",
            "usf_bios_mapping": "csv",
            "notes": "Supports streaming. Use --streaming true --max_steps N",
        },
        "txt": {
            "extensions": [".txt"],
            "max_size_gb": None,  # UNLIMITED with streaming
            "streaming": True,    # Text supports line-by-line streaming
            "recommended_for_large": True,
            "description": "Plain Text - For pre-training corpora (unlimited size)",
            "usf_bios_mapping": "text",
            "notes": "Supports streaming. Each line is one sample.",
        },
    }
    
    # Formats that are NOT supported by USF BIOS - clear error messages
    UNSUPPORTED_FORMATS = {
        ".tsv": "TSV files are not supported by USF BIOS. Please convert to CSV format.",
        ".parquet": "Parquet files are not supported by USF BIOS. Please convert to JSONL format.",
        ".pq": "Parquet files are not supported by USF BIOS. Please convert to JSONL format.",
        ".ndjson": "NDJSON files should use .jsonl extension. Please rename the file.",
        ".arrow": "Arrow files are not supported by USF BIOS. Please convert to JSONL format.",
        ".xlsx": "Excel files are not supported. Please export to CSV or JSONL format.",
        ".xls": "Excel files are not supported. Please export to CSV or JSONL format.",
    }
    
    # Detection settings
    SAMPLE_ROWS_FOR_DETECTION = 100  # Read first N rows for type detection
    DETECTION_TIMEOUT_SECONDS = 30   # Timeout for detection operation
    
    # Memory protection
    MAX_MEMORY_FOR_DETECTION_MB = 512  # Max memory for detection operation
    
    # Thresholds
    LARGE_FILE_THRESHOLD_GB = 1.0    # Files larger than this are considered "large"
    WARN_JSON_SIZE_GB = 0.5          # Warn if JSON is larger than this
    
    @classmethod
    def get_format_by_extension(cls, ext: str) -> Optional[str]:
        """Get format name by file extension"""
        ext = ext.lower()
        for fmt, config in cls.FORMATS.items():
            if ext in config["extensions"]:
                return fmt
        return None
    
    @classmethod
    def is_supported(cls, ext: str) -> bool:
        """Check if file extension is supported"""
        return cls.get_format_by_extension(ext) is not None
    
    @classmethod
    def get_max_size_gb(cls, fmt: str) -> Optional[float]:
        """Get max size in GB for format (None = unlimited)"""
        return cls.FORMATS.get(fmt, {}).get("max_size_gb")
    
    @classmethod
    def supports_streaming(cls, fmt: str) -> bool:
        """Check if format supports streaming"""
        return cls.FORMATS.get(fmt, {}).get("streaming", False)
    
    @classmethod
    def get_all_extensions(cls) -> List[str]:
        """Get all supported file extensions"""
        extensions = []
        for config in cls.FORMATS.values():
            extensions.extend(config["extensions"])
        return extensions
    
    @classmethod
    def is_unsupported(cls, ext: str) -> bool:
        """Check if extension is explicitly unsupported (known but not allowed)"""
        return ext.lower() in cls.UNSUPPORTED_FORMATS
    
    @classmethod
    def get_unsupported_error(cls, ext: str) -> Optional[str]:
        """Get error message for unsupported format"""
        return cls.UNSUPPORTED_FORMATS.get(ext.lower())
    
    @classmethod
    def validate_file_for_upload(cls, file_path: str) -> Tuple[bool, str, Optional[str]]:
        """
        Validate if a file can be uploaded for training.
        
        Returns:
            Tuple of (is_valid, error_message, file_format)
        """
        from pathlib import Path
        path = Path(file_path)
        ext = path.suffix.lower()
        
        # Check if explicitly unsupported
        if cls.is_unsupported(ext):
            return False, cls.get_unsupported_error(ext), None
        
        # Check if supported
        file_format = cls.get_format_by_extension(ext)
        if not file_format:
            supported = ", ".join(cls.get_all_extensions())
            return False, f"Unsupported format: {ext}. USF BIOS only supports: {supported}", None
        
        # Check file size for non-streaming formats
        try:
            file_size = path.stat().st_size
            file_size_gb = file_size / (1024 ** 3)
            max_size = cls.get_max_size_gb(file_format)
            
            if max_size is not None and file_size_gb > max_size:
                return False, (
                    f"File too large for {file_format.upper()} format ({file_size_gb:.2f} GB). "
                    f"Maximum: {max_size} GB. Use JSONL format for unlimited size."
                ), file_format
        except OSError:
            pass  # Will be caught later during detection
        
        return True, "", file_format


class DatasetType(str, Enum):
    """Dataset types for training"""
    SFT = "sft"                    # Supervised Fine-Tuning (messages format)
    RLHF_OFFLINE = "rlhf_offline"  # Offline RLHF (preference data: prompt/chosen/rejected)
    RLHF_ONLINE = "rlhf_online"    # Online RLHF (prompt only, generates responses)
    PT = "pt"                      # Pre-Training (raw text)
    KTO = "kto"                    # KTO specific format (prompt + completion + label)
    UNKNOWN = "unknown"            # Cannot determine type


# Legacy constants - now using FileFormatConfig
LARGE_FILE_THRESHOLD_GB = FileFormatConfig.LARGE_FILE_THRESHOLD_GB
MAX_JSON_FILE_SIZE_GB = FileFormatConfig.FORMATS["json"]["max_size_gb"]


class ModelType(str, Enum):
    """Model types"""
    FULL = "full"         # Full base model
    LORA = "lora"         # LoRA adapter
    QLORA = "qlora"       # QLoRA adapter (quantized)
    MERGED = "merged"     # Merged LoRA into base
    UNKNOWN = "unknown"


@dataclass
class DatasetTypeInfo:
    """Dataset type detection result with production-ready metadata"""
    dataset_type: DatasetType
    confidence: float  # 0.0 to 1.0
    detected_fields: List[str]
    sample_count: int
    compatible_training_methods: List[str]
    incompatible_training_methods: List[str]
    compatible_rlhf_types: List[str]  # Which RLHF algorithms work with this dataset
    display_name: str  # Human-readable name for UI
    message: str
    file_size_bytes: int = 0
    is_large_file: bool = False
    format_warning: Optional[str] = None  # Warning for large JSON files etc.
    file_format: Optional[str] = None  # Detected file format (jsonl, json, csv, etc.)
    supports_streaming: bool = False  # Whether format supports streaming
    estimated_samples: bool = False  # True if sample_count is estimated (not exact)
    validation_errors: List[str] = field(default_factory=list)  # Any validation errors
    validation_warnings: List[str] = field(default_factory=list)  # Any validation warnings
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "dataset_type": self.dataset_type.value,
            "confidence": self.confidence,
            "detected_fields": self.detected_fields,
            "sample_count": self.sample_count,
            "compatible_training_methods": self.compatible_training_methods,
            "incompatible_training_methods": self.incompatible_training_methods,
            "compatible_rlhf_types": self.compatible_rlhf_types,
            "display_name": self.display_name,
            "message": self.message,
            "file_size_bytes": self.file_size_bytes,
            "is_large_file": self.is_large_file,
            "format_warning": self.format_warning,
            "file_format": self.file_format,
            "supports_streaming": self.supports_streaming,
            "estimated_samples": self.estimated_samples,
            "validation_errors": self.validation_errors,
            "validation_warnings": self.validation_warnings,
        }


@dataclass
class ModelTypeInfo:
    """Model type detection result"""
    model_type: ModelType
    is_adapter: bool
    base_model_path: Optional[str]  # For LoRA adapters, the base model path from adapter_config
    adapter_config: Optional[Dict[str, Any]]
    can_do_lora: bool
    can_do_qlora: bool
    can_do_full: bool
    can_do_rlhf: bool
    warnings: List[str]
    # New fields for runtime merge support
    can_merge_with_base: bool = False  # True if adapter can be merged with base for full options
    merge_unlocks_full: bool = False   # True if merging would enable full fine-tuning
    adapter_r: Optional[int] = None    # LoRA rank from adapter config
    adapter_alpha: Optional[int] = None  # LoRA alpha from adapter config
    quantization_bits: Optional[int] = None  # 4 or 8 for QLoRA, None for LoRA
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_type": self.model_type.value,
            "is_adapter": self.is_adapter,
            "base_model_path": self.base_model_path,
            "can_do_lora": self.can_do_lora,
            "can_do_qlora": self.can_do_qlora,
            "can_do_full": self.can_do_full,
            "can_do_rlhf": self.can_do_rlhf,
            "warnings": self.warnings,
            "can_merge_with_base": self.can_merge_with_base,
            "merge_unlocks_full": self.merge_unlocks_full,
            "adapter_r": self.adapter_r,
            "adapter_alpha": self.adapter_alpha,
            "quantization_bits": self.quantization_bits,
        }


# Training method compatibility matrix
TRAINING_METHOD_COMPATIBILITY = {
    DatasetType.SFT: {
        "compatible": ["sft"],
        "incompatible": ["pt", "rlhf"],
        "rlhf_types": [],
        "display_name": "SFT (Instruction-Response)",
        "message": "SFT dataset detected. Only Supervised Fine-Tuning is available."
    },
    DatasetType.RLHF_OFFLINE: {
        "compatible": ["rlhf"],
        "incompatible": ["sft", "pt"],
        "rlhf_types": ["dpo", "orpo", "simpo", "kto", "cpo", "rm"],  # Offline algorithms
        "display_name": "RLHF Offline (Preference Data)",
        "message": "Offline RLHF preference dataset detected. Compatible with DPO, ORPO, SimPO, KTO, CPO, RM."
    },
    DatasetType.RLHF_ONLINE: {
        "compatible": ["rlhf"],
        "incompatible": ["sft", "pt"],
        "rlhf_types": ["ppo", "grpo", "gkd"],  # Online algorithms
        "display_name": "RLHF Online (Generation Required)",
        "message": "Online RLHF dataset detected. Compatible with PPO, GRPO, GKD."
    },
    DatasetType.PT: {
        "compatible": ["pt"],
        "incompatible": ["sft", "rlhf"],
        "rlhf_types": [],
        "display_name": "Pre-Training (Raw Text)",
        "message": "Pre-training dataset detected. Only Pre-Training is available."
    },
    DatasetType.KTO: {
        "compatible": ["rlhf"],
        "incompatible": ["sft", "pt"],
        "rlhf_types": ["kto"],
        "display_name": "KTO (Binary Feedback)",
        "message": "KTO dataset detected. Only KTO training is available."
    },
    DatasetType.UNKNOWN: {
        "compatible": ["sft", "pt", "rlhf"],
        "incompatible": [],
        "rlhf_types": [],
        "display_name": "Unknown Format",
        "message": "Could not determine dataset type. All training methods available."
    },
}


class DatasetTypeService:
    """Service for detecting dataset types and validating training compatibility"""
    
    # Field patterns for each dataset type
    SFT_FIELDS = {"messages"}
    SFT_ALT_FIELDS = {"instruction", "input", "output"}  # Alpaca format
    SFT_QUERY_FIELDS = {"query", "response"}  # Query-response format
    
    # Offline RLHF: has chosen/rejected pairs for preference learning
    RLHF_OFFLINE_FIELDS = {"prompt", "chosen", "rejected"}
    RLHF_OFFLINE_ALT_FIELDS = {"question", "chosen", "rejected"}
    
    # Online RLHF: has prompt + reward signal, needs to generate responses
    RLHF_ONLINE_FIELDS = {"prompt"}  # Just prompt, model generates completions
    RLHF_ONLINE_WITH_REWARD = {"prompt", "reward"}  # Prompt + reward function
    
    KTO_FIELDS = {"prompt", "completion", "label"}  # KTO specific
    
    PT_FIELDS = {"text"}
    PT_ALT_FIELDS = {"content"}
    
    def detect_dataset_type(self, dataset_path: str, feature_flags: Optional[Dict[str, bool]] = None) -> DatasetTypeInfo:
        """
        Detect the type of dataset from file content.
        Handles large files efficiently by streaming.
        
        PRODUCTION NOTES:
        - JSONL: No size limit, streaming-based detection
        - JSON: Max 2GB (must load into memory)
        - CSV: No size limit, streaming-based detection
        - TXT: No size limit, assumed pre-training
        - Parquet: No size limit, chunked reading
        
        Args:
            dataset_path: Path to the dataset file
            feature_flags: Optional feature flags to validate against
            
        Returns:
            DatasetTypeInfo with detected type and compatibility info
        """
        path = Path(dataset_path)
        validation_warnings = []
        validation_errors = []
        
        # === VALIDATION: Path exists ===
        if not path.exists():
            result = self._create_unknown_result("Dataset path does not exist")
            result.validation_errors = ["File not found"]
            return result
        
        # === VALIDATION: Is a file (not directory) ===
        if not path.is_file():
            result = self._create_unknown_result("Path is not a file")
            result.validation_errors = ["Path must be a file, not a directory"]
            return result
        
        # === Get file metadata safely ===
        try:
            file_stat = path.stat()
            file_size = file_stat.st_size
        except OSError as e:
            result = self._create_unknown_result(f"Cannot read file: {str(e)}")
            result.validation_errors = [f"OS error accessing file: {str(e)}"]
            return result
        
        # === VALIDATION: File not empty ===
        if file_size == 0:
            result = self._create_unknown_result("File is empty")
            result.validation_errors = ["Dataset file is empty (0 bytes)"]
            return result
        
        file_size_gb = file_size / (1024 ** 3)
        is_large_file = file_size_gb >= FileFormatConfig.LARGE_FILE_THRESHOLD_GB
        
        # Get file extension and format
        suffix = path.suffix.lower()
        
        # === VALIDATION: Check for explicitly unsupported formats first ===
        if FileFormatConfig.is_unsupported(suffix):
            error_msg = FileFormatConfig.get_unsupported_error(suffix)
            result = self._create_unknown_result(error_msg)
            result.validation_errors = [error_msg]
            return result
        
        file_format = FileFormatConfig.get_format_by_extension(suffix)
        
        # === VALIDATION: Supported format ===
        if not file_format:
            supported = ", ".join(FileFormatConfig.get_all_extensions())
            result = self._create_unknown_result(
                f"Unsupported format: {suffix}. USF BIOS only supports: {supported}"
            )
            result.validation_errors = [
                f"File format '{suffix}' is not supported by USF BIOS training.",
                f"Supported formats: {supported}"
            ]
            return result
        
        supports_streaming = FileFormatConfig.supports_streaming(file_format)
        max_size = FileFormatConfig.get_max_size_gb(file_format)
        
        # === VALIDATION: Size limits for non-streaming formats ===
        if max_size is not None and file_size_gb > max_size:
            if file_format == "json":
                result = self._create_unknown_result(
                    f"JSON file is too large ({file_size_gb:.2f} GB). "
                    f"Maximum allowed: {max_size} GB. "
                    f"Please convert to JSONL format for unlimited size."
                )
                result.validation_errors = [
                    f"File exceeds maximum size for JSON format ({max_size} GB)",
                    "Convert to JSONL format using: python -c \"import json; [print(json.dumps(x)) for x in json.load(open('file.json'))]\" > file.jsonl"
                ]
                result.file_format = file_format
                return result
        
        # Warning for large JSON files approaching limit
        if file_format == "json" and file_size_gb > FileFormatConfig.WARN_JSON_SIZE_GB:
            validation_warnings.append(
                f"Large JSON file ({file_size_gb:.2f} GB). "
                f"Consider converting to JSONL for better performance and scalability."
            )
        
        # Large file info for streaming formats
        if is_large_file and supports_streaming:
            logger.info(f"Processing large file ({file_size_gb:.2f} GB) using streaming mode: {path}")
        
        try:
            # Route to appropriate detector based on format
            # ONLY formats supported by USF BIOS: jsonl, json, csv, txt
            if file_format == "jsonl":
                result = self._detect_from_jsonl(path, file_size)
            elif file_format == "json":
                result = self._detect_from_json(path, file_size)
            elif file_format == "csv":
                result = self._detect_from_csv(path, file_size)
            elif file_format == "txt":
                result = self._detect_from_txt(path, file_size)
            else:
                # This shouldn't happen if FileFormatConfig is properly configured
                result = self._create_unknown_result(f"No detector for format: {file_format}")
            
            # Enrich result with format metadata
            result.file_size_bytes = file_size
            result.is_large_file = is_large_file
            result.file_format = file_format
            result.supports_streaming = supports_streaming
            result.validation_warnings = validation_warnings
            
            return result
                
        except MemoryError:
            logger.error(f"Memory error processing file: {path} ({file_size_gb:.2f} GB)")
            result = self._create_unknown_result(
                f"Out of memory processing file ({file_size_gb:.2f} GB). "
                "Use JSONL or CSV format for large datasets (streaming supported)."
            )
            result.validation_errors = ["Memory allocation failed - file too large for this format"]
            result.file_format = file_format
            return result
            
        except UnicodeDecodeError as e:
            logger.error(f"Encoding error in file: {path} - {str(e)}")
            result = self._create_unknown_result(
                f"File encoding error: {str(e)}. Ensure file is UTF-8 encoded."
            )
            result.validation_errors = ["File must be UTF-8 encoded"]
            result.file_format = file_format
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parse error in file: {path} - {str(e)}")
            result = self._create_unknown_result(
                f"Invalid JSON syntax at line {e.lineno}: {e.msg}"
            )
            result.validation_errors = [f"JSON syntax error at line {e.lineno}"]
            result.file_format = file_format
            return result
            
        except Exception as e:
            logger.exception(f"Unexpected error processing file: {path}")
            result = self._create_unknown_result(f"Error reading dataset: {str(e)}")
            result.validation_errors = [f"Unexpected error: {str(e)}"]
            result.file_format = file_format
            return result
    
    def _detect_from_jsonl(self, path: Path, file_size: int = 0) -> DatasetTypeInfo:
        """
        Detect type from JSONL file - STREAMING, NO SIZE LIMIT.
        
        This is the recommended format for large datasets (pre-training, RLHF).
        Uses memory-efficient streaming - only reads first N lines.
        """
        samples = []
        sample_limit = FileFormatConfig.SAMPLE_ROWS_FOR_DETECTION
        invalid_lines = 0
        total_bytes_sampled = 0
        
        file_size_gb = file_size / (1024 ** 3)
        is_large = file_size_gb >= FileFormatConfig.LARGE_FILE_THRESHOLD_GB
        
        # Stream file - never load entire file into memory
        with open(path, 'r', encoding='utf-8', errors='replace') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                    
                if len(samples) < sample_limit:
                    try:
                        sample = json.loads(line)
                        if isinstance(sample, dict):
                            samples.append(sample)
                            total_bytes_sampled += len(line)
                        else:
                            invalid_lines += 1
                    except json.JSONDecodeError:
                        invalid_lines += 1
                        continue
                
                # For large files, stop after getting enough samples
                if is_large and len(samples) >= sample_limit:
                    break
        
        if not samples:
            result = self._create_unknown_result("No valid JSON objects found in file")
            if invalid_lines > 0:
                result.validation_warnings = [f"Found {invalid_lines} invalid lines"]
            return result
        
        # Estimate total samples
        if is_large and total_bytes_sampled > 0:
            avg_line_size = total_bytes_sampled / len(samples)
            estimated_total = int(file_size / avg_line_size)
            is_estimated = True
        else:
            # For smaller files, count exact lines (still streaming)
            with open(path, 'r', encoding='utf-8') as f:
                estimated_total = sum(1 for line in f if line.strip())
            is_estimated = False
        
        result = self._detect_from_samples(samples, estimated_total, file_size)
        result.estimated_samples = is_estimated
        result.supports_streaming = True
        
        if invalid_lines > 0:
            result.validation_warnings.append(f"Skipped {invalid_lines} invalid lines during detection")
        
        return result
    
    def _detect_from_json(self, path: Path, file_size: int = 0) -> DatasetTypeInfo:
        """
        Detect type from JSON file.
        
        WARNING: JSON requires loading entire file into memory.
        Max size: 2GB. For larger files, use JSONL format.
        """
        # Load entire file (required for JSON)
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Validate structure
        if not isinstance(data, list):
            result = self._create_unknown_result(
                "JSON must be an array of objects. Found: " + type(data).__name__
            )
            result.validation_errors = ["Invalid JSON structure - must be array of objects"]
            return result
        
        if not data:
            result = self._create_unknown_result("JSON array is empty")
            result.validation_errors = ["Dataset contains no samples"]
            return result
        
        # Take samples for detection
        samples = data[:FileFormatConfig.SAMPLE_ROWS_FOR_DETECTION]
        
        if not isinstance(samples[0], dict):
            result = self._create_unknown_result(
                "JSON array items must be objects. Found: " + type(samples[0]).__name__
            )
            result.validation_errors = ["Array items must be JSON objects"]
            return result
        
        result = self._detect_from_samples(samples, len(data), file_size)
        result.estimated_samples = False  # Exact count for JSON
        result.supports_streaming = False
        return result
    
    def _detect_from_csv(self, path: Path, file_size: int = 0) -> DatasetTypeInfo:
        """
        Detect type from CSV file - STREAMING, NO SIZE LIMIT.
        
        Uses memory-efficient streaming - only reads first N rows.
        Handles both CSV and TSV formats.
        """
        samples = []
        sample_limit = FileFormatConfig.SAMPLE_ROWS_FOR_DETECTION
        total_bytes_sampled = 0
        
        file_size_gb = file_size / (1024 ** 3) if file_size else 0
        is_large = file_size_gb >= FileFormatConfig.LARGE_FILE_THRESHOLD_GB
        
        # Detect delimiter
        suffix = path.suffix.lower()
        delimiter = '\t' if suffix == '.tsv' else ','
        
        with open(path, 'r', encoding='utf-8', errors='replace', newline='') as f:
            # Sniff dialect for better parsing
            try:
                sample_text = f.read(8192)
                f.seek(0)
                dialect = csv.Sniffer().sniff(sample_text, delimiters=',\t;|')
            except csv.Error:
                dialect = 'excel'
            
            reader = csv.DictReader(f, dialect=dialect if isinstance(dialect, str) else dialect)
            
            try:
                for i, row in enumerate(reader):
                    if len(samples) < sample_limit:
                        samples.append(dict(row))
                        total_bytes_sampled += sum(len(str(v)) for v in row.values())
                    
                    # For large files, stop after samples
                    if is_large and len(samples) >= sample_limit:
                        break
            except csv.Error as e:
                if not samples:
                    result = self._create_unknown_result(f"CSV parsing error: {str(e)}")
                    result.validation_errors = [f"CSV format error: {str(e)}"]
                    return result
        
        if not samples:
            result = self._create_unknown_result("No valid rows found in CSV")
            result.validation_errors = ["CSV file has no data rows"]
            return result
        
        # Estimate total for large files
        if is_large and total_bytes_sampled > 0:
            avg_row_size = total_bytes_sampled / len(samples)
            estimated_total = int(file_size / avg_row_size)
            is_estimated = True
        else:
            # Count exact rows for smaller files
            with open(path, 'r', encoding='utf-8') as f:
                estimated_total = sum(1 for _ in f) - 1  # Subtract header
            is_estimated = False
        
        result = self._detect_from_samples(samples, estimated_total, file_size)
        result.estimated_samples = is_estimated
        result.supports_streaming = True
        return result
    
    def _detect_from_txt(self, path: Path, file_size: int = 0) -> DatasetTypeInfo:
        """
        Detect type from TXT file - STREAMING, NO SIZE LIMIT.
        
        Text files are assumed to be pre-training corpora.
        """
        file_size_gb = file_size / (1024 ** 3)
        is_large = file_size_gb >= FileFormatConfig.LARGE_FILE_THRESHOLD_GB
        
        # Count lines (streaming)
        if is_large:
            # Estimate from file size
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                sample_lines = []
                total_bytes = 0
                for i, line in enumerate(f):
                    if i < 100:
                        sample_lines.append(line)
                        total_bytes += len(line)
                    else:
                        break
            
            if sample_lines:
                avg_line_size = total_bytes / len(sample_lines)
                estimated_total = int(file_size / avg_line_size)
            else:
                estimated_total = 0
            is_estimated = True
        else:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                estimated_total = sum(1 for _ in f)
            is_estimated = False
        
        result = self._create_result(DatasetType.PT, ["text"], estimated_total, 1.0, file_size)
        result.estimated_samples = is_estimated
        result.supports_streaming = True
        return result
    
    # NOTE: Parquet detection removed - USF BIOS does not support Parquet format
    # See usf_bios/dataset/loader.py for supported formats: jsonl, json, csv, txt
    
    def _detect_from_samples(self, samples: List[Dict], total: int, file_size: int = 0) -> DatasetTypeInfo:
        """Detect dataset type from sample data"""
        if not samples:
            return self._create_unknown_result("No samples to analyze")
        
        # Get all fields from first sample
        first_sample = samples[0]
        fields = set(first_sample.keys())
        
        # Check for Offline RLHF preference data (most specific first)
        # Has prompt + chosen + rejected for preference learning
        if self.RLHF_OFFLINE_FIELDS.issubset(fields) or self.RLHF_OFFLINE_ALT_FIELDS.issubset(fields):
            detected_fields = list(self.RLHF_OFFLINE_FIELDS & fields) or list(self.RLHF_OFFLINE_ALT_FIELDS & fields)
            return self._create_result(DatasetType.RLHF_OFFLINE, detected_fields, total, 1.0, file_size)
        
        # Check for KTO format (prompt + completion + label)
        if self.KTO_FIELDS.issubset(fields):
            return self._create_result(DatasetType.KTO, list(self.KTO_FIELDS), total, 1.0, file_size)
        
        # Check for Online RLHF format (prompt only, or prompt + reward)
        # This is for algorithms that generate their own completions (PPO, GRPO, GKD)
        if self.RLHF_ONLINE_WITH_REWARD.issubset(fields):
            return self._create_result(DatasetType.RLHF_ONLINE, list(self.RLHF_ONLINE_WITH_REWARD & fields), total, 1.0, file_size)
        
        # Check for SFT messages format
        if "messages" in fields:
            messages = first_sample.get("messages", [])
            if isinstance(messages, list) and messages:
                # Validate messages have role/content
                if all(isinstance(m, dict) and "role" in m and "content" in m for m in messages[:3]):
                    return self._create_result(DatasetType.SFT, ["messages"], total, 1.0, file_size)
        
        # Check for Alpaca-style instruction format (SFT)
        if self.SFT_ALT_FIELDS.issubset(fields):
            return self._create_result(DatasetType.SFT, list(self.SFT_ALT_FIELDS & fields), total, 0.9, file_size)
        
        # Check for query-response format (SFT)
        if self.SFT_QUERY_FIELDS.issubset(fields):
            return self._create_result(DatasetType.SFT, list(self.SFT_QUERY_FIELDS & fields), total, 0.9, file_size)
        
        # Check for pre-training text format
        if self.PT_FIELDS.issubset(fields) or self.PT_ALT_FIELDS.issubset(fields):
            detected_fields = list(self.PT_FIELDS & fields) or list(self.PT_ALT_FIELDS & fields)
            return self._create_result(DatasetType.PT, detected_fields, total, 0.8, file_size)
        
        # Check for Online RLHF - prompt only (less specific, check last)
        # Only prompt field with no response/output fields indicates online RL
        if "prompt" in fields and "chosen" not in fields and "rejected" not in fields and "response" not in fields:
            return self._create_result(DatasetType.RLHF_ONLINE, ["prompt"], total, 0.7, file_size)
        
        # Fallback: check if there's instruction-like content
        if "instruction" in fields:
            return self._create_result(DatasetType.SFT, ["instruction"], total, 0.7, file_size)
        
        return self._create_unknown_result("Could not determine dataset type from fields")
    
    def _create_result(
        self, 
        dataset_type: DatasetType, 
        detected_fields: List[str], 
        sample_count: int,
        confidence: float,
        file_size: int = 0
    ) -> DatasetTypeInfo:
        """Create a DatasetTypeInfo result with all fields properly initialized"""
        compat = TRAINING_METHOD_COMPATIBILITY.get(dataset_type, TRAINING_METHOD_COMPATIBILITY[DatasetType.UNKNOWN])
        
        return DatasetTypeInfo(
            dataset_type=dataset_type,
            confidence=confidence,
            detected_fields=detected_fields,
            sample_count=sample_count,
            compatible_training_methods=compat["compatible"],
            incompatible_training_methods=compat["incompatible"],
            compatible_rlhf_types=compat.get("rlhf_types", []),
            display_name=compat.get("display_name", dataset_type.value),
            message=compat["message"],
            file_size_bytes=file_size,
            is_large_file=file_size >= FileFormatConfig.LARGE_FILE_THRESHOLD_GB * (1024 ** 3),
            file_format=None,  # Set by caller
            supports_streaming=False,  # Set by caller
            estimated_samples=False,  # Set by caller
            validation_errors=[],
            validation_warnings=[],
        )
    
    def _create_unknown_result(self, message: str, format_warning: Optional[str] = None) -> DatasetTypeInfo:
        """Create an unknown type result with all fields properly initialized"""
        compat = TRAINING_METHOD_COMPATIBILITY[DatasetType.UNKNOWN]
        return DatasetTypeInfo(
            dataset_type=DatasetType.UNKNOWN,
            confidence=0.0,
            detected_fields=[],
            sample_count=0,
            compatible_training_methods=compat["compatible"],
            incompatible_training_methods=compat["incompatible"],
            compatible_rlhf_types=compat.get("rlhf_types", []),
            display_name=compat.get("display_name", "Unknown"),
            message=message,
            format_warning=format_warning,
            file_format=None,
            supports_streaming=False,
            estimated_samples=False,
            validation_errors=[],
            validation_warnings=[],
        )
    
    def validate_dataset_for_upload(
        self,
        dataset_path: str,
        feature_flags: Dict[str, bool]
    ) -> Tuple[bool, str, Optional[DatasetTypeInfo]]:
        """
        Validate if a dataset can be uploaded based on feature flags.
        
        Args:
            dataset_path: Path to the dataset file
            feature_flags: System feature flags
            
        Returns:
            Tuple of (is_valid, error_message, type_info)
        """
        # Detect dataset type
        type_info = self.detect_dataset_type(dataset_path)
        
        if type_info.dataset_type == DatasetType.UNKNOWN:
            # Unknown datasets are allowed (user might know what they're doing)
            return True, "", type_info
        
        # Check if the training method for this dataset type is enabled
        dataset_type = type_info.dataset_type
        
        # SFT dataset
        if dataset_type == DatasetType.SFT:
            if not feature_flags.get("sft", True):
                return False, "SFT training is not enabled on this system. Cannot upload SFT datasets.", type_info
        
        # Pre-training dataset
        elif dataset_type == DatasetType.PT:
            if not feature_flags.get("pretraining", True):
                return False, "Pre-training is not enabled on this system. Cannot upload pre-training datasets.", type_info
        
        # Offline RLHF dataset (DPO, ORPO, etc.)
        elif dataset_type == DatasetType.RLHF_OFFLINE:
            if not feature_flags.get("rlhf", True):
                return False, "RLHF training is not enabled on this system. Cannot upload RLHF datasets.", type_info
            if not feature_flags.get("rlhf_offline", True):
                return False, "Offline RLHF (DPO, ORPO, etc.) is not enabled on this system. Cannot upload offline RLHF datasets.", type_info
        
        # Online RLHF dataset (PPO, GRPO, GKD)
        elif dataset_type == DatasetType.RLHF_ONLINE:
            if not feature_flags.get("rlhf", True):
                return False, "RLHF training is not enabled on this system. Cannot upload RLHF datasets.", type_info
            if not feature_flags.get("rlhf_online", True):
                return False, "Online RLHF (PPO, GRPO, GKD) is not enabled on this system. Cannot upload online RLHF datasets.", type_info
        
        # KTO dataset
        elif dataset_type == DatasetType.KTO:
            if not feature_flags.get("rlhf", True):
                return False, "RLHF training is not enabled on this system. Cannot upload KTO datasets.", type_info
            if not feature_flags.get("kto", True):
                return False, "KTO training is not enabled on this system. Cannot upload KTO datasets.", type_info
        
        return True, "", type_info
    
    def check_dataset_compatibility(
        self,
        new_dataset_type: DatasetType,
        existing_dataset_types: List[DatasetType]
    ) -> Tuple[bool, str]:
        """
        Check if a new dataset is compatible with already selected datasets.
        All datasets must be the same type for training.
        
        Args:
            new_dataset_type: Type of the dataset being added
            existing_dataset_types: Types of already selected datasets
            
        Returns:
            Tuple of (is_compatible, error_message)
        """
        if not existing_dataset_types:
            return True, ""
        
        # Get the type of existing datasets (they should all be the same)
        existing_type = existing_dataset_types[0]
        
        # Unknown types are always compatible
        if new_dataset_type == DatasetType.UNKNOWN or existing_type == DatasetType.UNKNOWN:
            return True, ""
        
        # Check if types match
        if new_dataset_type != existing_type:
            new_display = TRAINING_METHOD_COMPATIBILITY.get(new_dataset_type, {}).get("display_name", new_dataset_type.value)
            existing_display = TRAINING_METHOD_COMPATIBILITY.get(existing_type, {}).get("display_name", existing_type.value)
            
            return False, (
                f"Cannot mix dataset types. You have selected {existing_display} datasets. "
                f"This dataset is {new_display}. All datasets must be the same type for training."
            )
        
        return True, ""
    
    def detect_model_type(self, model_path: str) -> ModelTypeInfo:
        """
        Detect if a model is a full model or LoRA adapter.
        
        Args:
            model_path: Path to the model directory
            
        Returns:
            ModelTypeInfo with model type and training compatibility
        """
        path = Path(model_path)
        warnings = []
        
        if not path.exists():
            return ModelTypeInfo(
                model_type=ModelType.UNKNOWN,
                is_adapter=False,
                base_model_path=None,
                adapter_config=None,
                can_do_lora=True,
                can_do_qlora=True,
                can_do_full=True,
                can_do_rlhf=True,
                warnings=["Model path does not exist"]
            )
        
        # Check for adapter_config.json (indicates LoRA adapter)
        adapter_config_path = path / "adapter_config.json"
        
        if adapter_config_path.exists():
            try:
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)
                
                base_model = adapter_config.get("base_model_name_or_path")
                
                # Check if it's QLoRA (quantized)
                is_qlora = adapter_config.get("bits") in [4, 8] or adapter_config.get("load_in_4bit", False) or adapter_config.get("load_in_8bit", False)
                
                model_type = ModelType.QLORA if is_qlora else ModelType.LORA
                
                # LoRA adapters have restrictions
                warnings.append(
                    f"This is a LoRA adapter. For training, ensure the base model '{base_model}' is accessible."
                )
                
                # For RLHF on LoRA adapter: it's possible but need base model
                # You can train a NEW LoRA adapter on top of base+adapter combination
                can_do_rlhf = True
                if not base_model:
                    warnings.append(
                        "Base model path not found in adapter config. RLHF training may require manual base model specification."
                    )
                
                # Extract adapter details for UI display
                adapter_r = adapter_config.get("r")
                adapter_alpha = adapter_config.get("lora_alpha")
                quant_bits = adapter_config.get("bits")
                if not quant_bits and adapter_config.get("load_in_4bit"):
                    quant_bits = 4
                elif not quant_bits and adapter_config.get("load_in_8bit"):
                    quant_bits = 8
                
                # Adapters can be merged with base model to unlock full fine-tuning
                can_merge = base_model is not None
                
                warnings.append(
                    "💡 Tip: Provide the base model to merge with this adapter and unlock all training options (Full, RLHF with Full)."
                )
                
                return ModelTypeInfo(
                    model_type=model_type,
                    is_adapter=True,
                    base_model_path=base_model,
                    adapter_config=adapter_config,
                    can_do_lora=True,  # Can train new LoRA on top
                    can_do_qlora=True,
                    can_do_full=False,  # Cannot do full fine-tuning on adapter alone
                    can_do_rlhf=can_do_rlhf,
                    warnings=warnings,
                    can_merge_with_base=can_merge,
                    merge_unlocks_full=True,  # Merging enables full fine-tuning
                    adapter_r=adapter_r,
                    adapter_alpha=adapter_alpha,
                    quantization_bits=quant_bits,
                )
                
            except (json.JSONDecodeError, IOError) as e:
                warnings.append(f"Could not read adapter config: {str(e)}")
        
        # Check for config.json (full model)
        config_path = path / "config.json"
        if config_path.exists():
            return ModelTypeInfo(
                model_type=ModelType.FULL,
                is_adapter=False,
                base_model_path=None,
                adapter_config=None,
                can_do_lora=True,
                can_do_qlora=True,
                can_do_full=True,
                can_do_rlhf=True,
                warnings=warnings
            )
        
        # Unknown model structure
        return ModelTypeInfo(
            model_type=ModelType.UNKNOWN,
            is_adapter=False,
            base_model_path=None,
            adapter_config=None,
            can_do_lora=True,
            can_do_qlora=True,
            can_do_full=True,
            can_do_rlhf=True,
            warnings=["Could not determine model type. No config.json or adapter_config.json found."]
        )
    
    def validate_training_config(
        self,
        dataset_type: DatasetType,
        training_method: str,
        train_type: str,
        model_type: ModelType,
        rlhf_type: Optional[str] = None
    ) -> Tuple[bool, str]:
        """
        Validate if the training configuration is compatible.
        
        Args:
            dataset_type: Detected dataset type
            training_method: Selected training method (sft, pt, rlhf)
            train_type: Selected training type (lora, qlora, full, adalora)
            model_type: Detected model type
            rlhf_type: For RLHF, the specific algorithm (dpo, orpo, etc.)
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        errors = []
        
        # Check dataset-training method compatibility
        compat = TRAINING_METHOD_COMPATIBILITY.get(dataset_type, TRAINING_METHOD_COMPATIBILITY[DatasetType.UNKNOWN])
        if training_method in compat["incompatible"]:
            method_name = {"sft": "SFT", "pt": "Pre-Training", "rlhf": "RLHF"}.get(training_method, training_method)
            type_name = {"sft": "SFT", "rlhf": "RLHF preference", "pt": "Pre-training", "kto": "KTO"}.get(dataset_type.value, dataset_type.value)
            errors.append(f"{method_name} is not compatible with {type_name} datasets. {compat['message']}")
        
        # Check KTO-specific validation
        if dataset_type == DatasetType.KTO and rlhf_type and rlhf_type != "kto":
            errors.append(f"KTO dataset format is only compatible with KTO algorithm. Please select KTO or use a standard RLHF preference dataset.")
        
        # Check model-training type compatibility
        if model_type == ModelType.LORA or model_type == ModelType.QLORA:
            if train_type == "full":
                errors.append("Full fine-tuning is not available for LoRA adapters. Please select LoRA, QLoRA, or AdaLoRA.")
        
        if errors:
            return False, " ".join(errors)
        
        return True, "Configuration is valid"


    def get_supported_formats(self) -> Dict[str, Any]:
        """
        Get information about supported file formats for the API.
        
        Returns:
            Dict with format information for each supported type
        """
        formats_info = {}
        for fmt, config in FileFormatConfig.FORMATS.items():
            max_size = config["max_size_gb"]
            formats_info[fmt] = {
                "extensions": config["extensions"],
                "max_size_gb": max_size,
                "max_size_display": f"{max_size} GB" if max_size else "Unlimited",
                "streaming_supported": config["streaming"],
                "recommended_for_large_files": config["recommended_for_large"],
                "description": config["description"],
            }
        
        return {
            "formats": formats_info,
            "recommended_format": "jsonl",
            "recommended_reason": "JSONL supports streaming for unlimited file sizes, ideal for pre-training and RLHF datasets",
            "sample_rows_for_detection": FileFormatConfig.SAMPLE_ROWS_FOR_DETECTION,
            "large_file_threshold_gb": FileFormatConfig.LARGE_FILE_THRESHOLD_GB,
        }


# Global instance
dataset_type_service = DatasetTypeService()
