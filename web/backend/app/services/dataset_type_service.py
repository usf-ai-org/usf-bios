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
    
    =============================================================================
    SCALABILITY FOR MASSIVE DATASETS (100TB+, Billions of Rows)
    =============================================================================
    
    RECOMMENDED APPROACH BY DATA SIZE:
    ----------------------------------
    | Size        | Method              | Type Detection        | Training      |
    |-------------|---------------------|----------------------|---------------|
    | < 50GB      | Direct Upload       | Sample first 100 rows| Full loading  |
    | 50GB-1TB    | Local Path          | Stream first 100 rows| Streaming     |
    | 1TB-100TB+  | Local Path          | Stream first 100 rows| Streaming     |
    | Any size    | HuggingFace/MS Hub  | Stream 10 samples    | Streaming     |
    
    UPLOAD LIMITS:
    -------------
    - Direct Upload: 50GB max (memory constraint)
    - Chunked Upload: 100GB max (streaming to disk)
    - Local Path: UNLIMITED (just reference path)
    - HuggingFace/ModelScope: UNLIMITED (streaming from hub)
    
    TYPE DETECTION FOR MASSIVE FILES:
    ---------------------------------
    - Local files: Streams first 100 rows only (memory-safe)
    - HuggingFace/ModelScope: Streams 10 samples only
    - NEVER loads full dataset for type detection
    
    TRAINING WITH MASSIVE DATASETS:
    -------------------------------
    For datasets with billions of samples:
    - Use JSONL format with --streaming true
    - Set --max_steps (required since dataset length is unknown)
    - Use --shuffle_buffer_size for randomization (default: 1000)
    
    =============================================================================
    
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


# ==============================================================================
# VALID MESSAGE ROLES - FROM USF-BIOS preprocessor/core.py:80
# ==============================================================================
# USF-BIOS validates: role in {'system', 'user', 'tool_call', 'tool_response', 'tool', 'assistant'}
# Note: 'tool' and 'tool_response' are equivalent (compatibility alias)
# Note: 'developer' role is NOT directly supported - use 'system' with model_identity
# ==============================================================================
VALID_MESSAGE_ROLES = {
    "system",        # System prompt / instructions
    "user",          # User input
    "assistant",     # Model response
    "tool",          # Tool/function response (alias for tool_response)
    "tool_call",     # Model calling a tool/function
    "tool_response", # Response from tool execution
}

# Tool calling message structure (for function/tool calling datasets)
TOOL_CALL_STRUCTURE = {
    "role": "tool_call",
    "content": {
        "type": "object",
        "description": "Can be string (JSON) or object with function call details",
        "properties": {
            "name": {"type": "string", "description": "Function name"},
            "arguments": {"type": "object", "description": "Function arguments"}
        }
    }
}

TOOL_RESPONSE_STRUCTURE = {
    "role": "tool",  # or "tool_response"
    "content": {"type": "string", "description": "JSON string of tool result"}
}

# ==============================================================================
# STANDARD FORMAT SPECIFICATIONS
# ==============================================================================
# Naming Convention: {TRAINING_METHOD}_{MODALITY}_{VARIANT}
#   - TRAINING_METHOD: sft, pt, rlhf_pref (preference), rlhf_binary, rlhf_online
#   - MODALITY: text (default, omitted), vision, audio, video
#   - VARIANT: optional specifier (e.g., alpaca, sharegpt, tool_calling)
#
# These specifications are verified against USF-BIOS:
#   preprocessor/core.py: _pair_keys = ['messages', 'images', 'videos', 'audios', 'tools', 'objects']
#   preprocessor/core.py: standard_keys includes prefixes: 'rejected_', 'positive_', 'negative_'
#   preprocessor/core.py:80: Valid roles = {'system', 'user', 'tool_call', 'tool_response', 'tool', 'assistant'}
#   template/template_inputs.py: tools field for function definitions
#   agent_template/base.py: Tool calling/function calling support
# ==============================================================================

FORMAT_SPECIFICATIONS = {
    # ==========================================================================
    # SFT FORMATS (Supervised Fine-Tuning)
    # ==========================================================================
    "sft_messages": {
        "id": "sft_messages",
        "display_name": "SFT - Messages (ChatML/ShareGPT)",
        "description": "Standard chat format with role-based messages",
        "training_method": "sft",
        "modality": "text",
        "required_fields": ["messages"],
        "optional_fields": ["system"],
        "field_structure": {
            "messages": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["role", "content"],
                    "properties": {
                        "role": {"type": "string", "enum": ["system", "user", "assistant", "tool", "tool_call", "tool_response"]},
                        "content": {"type": "string"},
                        "loss": {"type": "float", "optional": True, "description": "Loss weight for this message"}
                    }
                },
                "min_items": 1
            }
        },
        "example": '{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}]}',
        "usf_bios_ref": "preprocessor/core.py:_check_messages()"
    },
    "sft_alpaca": {
        "id": "sft_alpaca",
        "display_name": "SFT - Alpaca (Instruction/Output)",
        "description": "Stanford Alpaca format with instruction and output",
        "training_method": "sft",
        "modality": "text",
        "required_fields": ["instruction", "output"],
        "optional_fields": ["input", "system"],
        "field_structure": {
            "instruction": {"type": "string", "min_length": 1},
            "output": {"type": "string", "min_length": 1},
            "input": {"type": "string", "optional": True}
        },
        "example": '{"instruction": "Explain AI", "input": "", "output": "AI is..."}',
        "usf_bios_ref": "preprocessor/core.py:ResponsePreprocessor"
    },
    "sft_query_response": {
        "id": "sft_query_response",
        "display_name": "SFT - Query/Response",
        "description": "Simple query and response pairs",
        "training_method": "sft",
        "modality": "text",
        "required_fields": ["query", "response"],
        "optional_fields": ["system"],
        "field_structure": {
            "query": {"type": "string", "min_length": 1},
            "response": {"type": "string", "min_length": 1}
        },
        "example": '{"query": "What is 2+2?", "response": "4"}',
        "usf_bios_ref": "preprocessor/core.py:ResponsePreprocessor"
    },
    
    # ==========================================================================
    # SFT TOOL CALLING / FUNCTION CALLING FORMAT
    # ==========================================================================
    "sft_tool_calling": {
        "id": "sft_tool_calling",
        "display_name": "SFT - Tool/Function Calling",
        "description": "Training data for tool/function calling with tool_call and tool roles",
        "training_method": "sft",
        "modality": "text",
        "required_fields": ["messages"],
        "optional_fields": ["tools", "system"],
        "field_structure": {
            "messages": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["role", "content"],
                    "properties": {
                        "role": {"type": "string", "enum": ["system", "user", "assistant", "tool", "tool_call", "tool_response"]},
                        "content": {"type": "string_or_object", "description": "String or JSON object for tool calls"}
                    }
                },
                "must_contain_roles": ["tool_call"],
                "description": "Must contain at least one tool_call message"
            },
            "tools": {
                "type": "array",
                "optional": True,
                "items": {
                    "type": "object",
                    "properties": {
                        "type": {"type": "string", "enum": ["function"]},
                        "function": {
                            "type": "object",
                            "properties": {
                                "name": {"type": "string"},
                                "description": {"type": "string"},
                                "parameters": {"type": "object"}
                            }
                        }
                    }
                },
                "description": "Function/tool definitions (OpenAI format)"
            }
        },
        "example": '{"messages": [{"role": "user", "content": "What is the weather?"}, {"role": "tool_call", "content": "{\\"name\\": \\"get_weather\\", \\"arguments\\": {\\"city\\": \\"NYC\\"}}"}, {"role": "tool", "content": "{\\"temp\\": 72}"}, {"role": "assistant", "content": "It is 72°F"}], "tools": [...]}',
        "usf_bios_ref": "agent_template/base.py, template/template_inputs.py:tools"
    },
    
    # ==========================================================================
    # SFT MULTIMODAL FORMATS
    # ==========================================================================
    "sft_vision": {
        "id": "sft_vision",
        "display_name": "SFT - Vision (Image + Text)",
        "description": "Multimodal SFT with images for VLMs like LLaVA, Qwen-VL",
        "training_method": "sft",
        "modality": "vision",
        "required_fields": ["messages", "images"],
        "optional_fields": ["objects"],
        "field_structure": {
            "messages": {"type": "array", "items": {"type": "object", "required": ["role", "content"]}},
            "images": {
                "type": "array_or_string",
                "items": {"type": "string_or_object"},
                "description": "Image paths, URLs, base64, or PIL-like objects"
            }
        },
        "example": '{"messages": [...], "images": ["path/to/image.jpg"]}',
        "usf_bios_ref": "preprocessor/core.py:_cast_mm_data()"
    },
    "sft_audio": {
        "id": "sft_audio",
        "display_name": "SFT - Audio (Audio + Text)",
        "description": "Multimodal SFT with audio for models like Qwen-Audio",
        "training_method": "sft",
        "modality": "audio",
        "required_fields": ["messages", "audios"],
        "optional_fields": [],
        "field_structure": {
            "messages": {"type": "array", "items": {"type": "object", "required": ["role", "content"]}},
            "audios": {"type": "array_or_string", "items": {"type": "string"}}
        },
        "example": '{"messages": [...], "audios": ["path/to/audio.wav"]}',
        "usf_bios_ref": "preprocessor/core.py:_cast_mm_data()"
    },
    "sft_video": {
        "id": "sft_video",
        "display_name": "SFT - Video (Video + Text)",
        "description": "Multimodal SFT with video",
        "training_method": "sft",
        "modality": "video",
        "required_fields": ["messages", "videos"],
        "optional_fields": [],
        "field_structure": {
            "messages": {"type": "array", "items": {"type": "object", "required": ["role", "content"]}},
            "videos": {"type": "array_or_string", "items": {"type": "string"}}
        },
        "example": '{"messages": [...], "videos": ["path/to/video.mp4"]}',
        "usf_bios_ref": "preprocessor/core.py:_cast_mm_data()"
    },
    
    # ==========================================================================
    # PRE-TRAINING FORMAT
    # ==========================================================================
    "pt_text": {
        "id": "pt_text",
        "display_name": "Pre-Training - Raw Text",
        "description": "Plain text for continued pre-training (NO multimodal support)",
        "training_method": "pt",
        "modality": "text",
        "required_fields": ["text"],
        "optional_fields": [],
        "field_structure": {
            "text": {"type": "string", "min_length": 1}
        },
        "example": '{"text": "This is a long document for pre-training..."}',
        "usf_bios_ref": "sft.py:342-345 (text only)",
        "restrictions": ["No multimodal support - text only"]
    },
    
    # ==========================================================================
    # RLHF PREFERENCE FORMATS (DPO, ORPO, SimPO, CPO, RM)
    # ==========================================================================
    "rlhf_pref_messages": {
        "id": "rlhf_pref_messages",
        "display_name": "RLHF Preference - Rejected Messages",
        "description": "Preference pairs with chosen and rejected message arrays",
        "training_method": "rlhf",
        "rlhf_types": ["dpo", "orpo", "simpo", "cpo", "rm"],
        "modality": "text",
        "required_fields": ["messages", "rejected_messages"],
        "optional_fields": [],
        "field_structure": {
            "messages": {"type": "array", "description": "Chosen (preferred) conversation"},
            "rejected_messages": {"type": "array", "description": "Rejected conversation"}
        },
        "example": '{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Good answer"}], "rejected_messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Bad answer"}]}',
        "usf_bios_ref": "preprocessor/core.py:standard_keys"
    },
    "rlhf_pref_response": {
        "id": "rlhf_pref_response",
        "display_name": "RLHF Preference - Rejected Response",
        "description": "Preference with chosen messages and rejected response string",
        "training_method": "rlhf",
        "rlhf_types": ["dpo", "orpo", "simpo", "cpo", "rm"],
        "modality": "text",
        "required_fields": ["messages", "rejected_response"],
        "optional_fields": [],
        "field_structure": {
            "messages": {"type": "array", "description": "Chosen conversation (last message is chosen response)"},
            "rejected_response": {"type": "string", "description": "Rejected response (must differ from last message)"}
        },
        "example": '{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Good"}], "rejected_response": "Bad"}',
        "usf_bios_ref": "preprocessor/core.py:_check_rejected_response()"
    },
    "rlhf_pref_simple": {
        "id": "rlhf_pref_simple",
        "display_name": "RLHF Preference - Simple (prompt/chosen/rejected)",
        "description": "Simple preference format with prompt and chosen/rejected response strings (HuggingFace DPO format)",
        "training_method": "rlhf",
        "rlhf_types": ["dpo", "orpo", "simpo", "cpo", "rm"],
        "modality": "text",
        "required_fields": ["prompt", "chosen", "rejected"],
        "optional_fields": [],
        "field_structure": {
            "prompt": {"type": "string", "description": "User prompt/question"},
            "chosen": {"type": "string", "description": "Preferred (chosen) response"},
            "rejected": {"type": "string", "description": "Non-preferred (rejected) response"}
        },
        "example": '{"prompt": "What is 2+2?", "chosen": "2+2 equals 4.", "rejected": "I don\'t know."}',
        "usf_bios_ref": "dataset_info.json: zhihu_rlhf_3k, orca_dpo_pairs, Human-Like-DPO-Dataset",
        "notes": "USF-BIOS auto-maps: prompt→query, chosen→response, rejected→rejected_response"
    },
    
    # ==========================================================================
    # RLHF PREFERENCE MULTIMODAL FORMATS
    # ==========================================================================
    "rlhf_pref_vision": {
        "id": "rlhf_pref_vision",
        "display_name": "RLHF Preference - Vision",
        "description": "Vision RLHF with images and rejected_images",
        "training_method": "rlhf",
        "rlhf_types": ["dpo", "orpo", "simpo", "cpo", "rm"],
        "modality": "vision",
        "required_fields": ["messages", "rejected_messages", "images"],
        "optional_fields": ["rejected_images"],
        "field_structure": {
            "messages": {"type": "array"},
            "rejected_messages": {"type": "array"},
            "images": {"type": "array_or_string"},
            "rejected_images": {"type": "array_or_string", "optional": True}
        },
        "example": '{"messages": [...], "rejected_messages": [...], "images": ["img.jpg"]}',
        "usf_bios_ref": "preprocessor/core.py:_cast_mm_data() handles rejected_images"
    },
    "rlhf_pref_audio": {
        "id": "rlhf_pref_audio",
        "display_name": "RLHF Preference - Audio",
        "description": "Audio RLHF with audios field",
        "training_method": "rlhf",
        "rlhf_types": ["dpo", "orpo", "simpo", "cpo", "rm"],
        "modality": "audio",
        "required_fields": ["messages", "rejected_messages", "audios"],
        "optional_fields": ["rejected_audios"],
        "field_structure": {
            "messages": {"type": "array"},
            "rejected_messages": {"type": "array"},
            "audios": {"type": "array_or_string"}
        },
        "example": '{"messages": [...], "rejected_messages": [...], "audios": ["audio.wav"]}',
        "usf_bios_ref": "preprocessor/core.py:standard_keys"
    },
    "rlhf_pref_video": {
        "id": "rlhf_pref_video",
        "display_name": "RLHF Preference - Video",
        "description": "Video RLHF with videos field",
        "training_method": "rlhf",
        "rlhf_types": ["dpo", "orpo", "simpo", "cpo", "rm"],
        "modality": "video",
        "required_fields": ["messages", "rejected_messages", "videos"],
        "optional_fields": ["rejected_videos"],
        "field_structure": {
            "messages": {"type": "array"},
            "rejected_messages": {"type": "array"},
            "videos": {"type": "array_or_string"}
        },
        "example": '{"messages": [...], "rejected_messages": [...], "videos": ["video.mp4"]}',
        "usf_bios_ref": "preprocessor/core.py:standard_keys"
    },
    
    # ==========================================================================
    # RLHF BINARY FORMAT (KTO)
    # ==========================================================================
    "rlhf_binary_kto": {
        "id": "rlhf_binary_kto",
        "display_name": "RLHF Binary - KTO (Messages + Label)",
        "description": "Binary feedback with label (0=bad, 1=good) for KTO",
        "training_method": "rlhf",
        "rlhf_types": ["kto"],
        "modality": "text",
        "required_fields": ["messages", "label"],
        "optional_fields": [],
        "field_structure": {
            "messages": {"type": "array", "items": {"type": "object", "required": ["role", "content"]}},
            "label": {"type": "integer_or_boolean", "enum": [0, 1, True, False], "description": "1/True=desirable, 0/False=undesirable"}
        },
        "example": '{"messages": [{"role": "user", "content": "Hi"}, {"role": "assistant", "content": "Hello!"}], "label": 1}',
        "usf_bios_ref": "pipelines/train/kto.py:KTOPreprocessor"
    },
    
    # ==========================================================================
    # RLHF ONLINE FORMAT (PPO, GRPO, GKD)
    # ==========================================================================
    "rlhf_online_prompt": {
        "id": "rlhf_online_prompt",
        "display_name": "RLHF Online - Prompt Only (PPO/GRPO/GKD)",
        "description": "Prompts for online RL - model generates responses during training",
        "training_method": "rlhf",
        "rlhf_types": ["ppo", "grpo", "gkd"],
        "modality": "text",
        "required_fields": ["messages"],
        "optional_fields": [],
        "field_structure": {
            "messages": {
                "type": "array",
                "description": "Prompt messages (no assistant response - model will generate)",
                "items": {"type": "object", "required": ["role", "content"]}
            }
        },
        "example": '{"messages": [{"role": "user", "content": "Solve this math problem..."}]}',
        "usf_bios_ref": "rlhf_args.py:rlhf_type in ['ppo', 'grpo', 'gkd']"
    },
}

# Valid roles for messages (from USF-BIOS preprocessor/core.py line 80)
VALID_MESSAGE_ROLES = {"system", "user", "assistant", "tool", "tool_call", "tool_response"}


class DatasetType(str, Enum):
    """Dataset types with standard naming convention.
    
    NAMING CONVENTION: {training_method}_{modality}_{variant}
    - Training Method: sft, pt, rlhf_pref, rlhf_binary, rlhf_online
    - Modality: (text is default/omitted), vision, audio, video
    - Variant: optional (alpaca, sharegpt, etc.)
    
    VERIFIED AGAINST USF-BIOS:
    - preprocessor/core.py: _pair_keys, standard_keys, _check_messages()
    - Valid roles: system, user, assistant, tool, tool_call, tool_response
    """
    # ========== SFT (Supervised Fine-Tuning) ==========
    SFT = "sft"                      # Text-only SFT (messages/alpaca/query-response)
    SFT_TOOL_CALLING = "sft_tool_calling"  # SFT with tool_call/tool roles
    SFT_VISION = "sft_vision"        # SFT + images
    SFT_AUDIO = "sft_audio"          # SFT + audios  
    SFT_VIDEO = "sft_video"          # SFT + videos
    
    # ========== PT (Pre-Training) ==========
    PT = "pt"                        # Raw text only (NO multimodal per USF-BIOS)
    
    # ========== RLHF Preference (DPO/ORPO/SimPO/CPO/RM) ==========
    RLHF_PREF = "rlhf_pref"          # Text preference (messages + rejected_messages/response)
    RLHF_PREF_VISION = "rlhf_pref_vision"  # Vision preference + images
    RLHF_PREF_AUDIO = "rlhf_pref_audio"    # Audio preference + audios
    RLHF_PREF_VIDEO = "rlhf_pref_video"    # Video preference + videos
    
    # ========== RLHF Binary (KTO) ==========
    RLHF_BINARY = "rlhf_binary"      # Binary feedback (messages + label)
    
    # ========== RLHF Online (PPO/GRPO/GKD) ==========
    RLHF_ONLINE = "rlhf_online"      # Prompt only (model generates responses)
    
    # ========== UNKNOWN ==========
    UNKNOWN = "unknown"              # Cannot determine format
    
    # Legacy aliases for backward compatibility
    RLHF_OFFLINE_PREFERENCE = "rlhf_pref"  # Alias
    RLHF_OFFLINE_BINARY = "rlhf_binary"    # Alias
    RLHF_OFFLINE_VISION = "rlhf_pref_vision"  # Alias
    RLHF_OFFLINE_AUDIO = "rlhf_pref_audio"    # Alias
    RLHF_OFFLINE_VIDEO = "rlhf_pref_video"    # Alias
    
    @classmethod
    def is_rlhf(cls, dataset_type: 'DatasetType') -> bool:
        """Check if dataset type is any RLHF variant"""
        return dataset_type.value.startswith('rlhf_')
    
    @classmethod
    def is_rlhf_offline(cls, dataset_type: 'DatasetType') -> bool:
        """Check if dataset type is offline RLHF"""
        return dataset_type.value.startswith('rlhf_offline_')
    
    @classmethod
    def is_rlhf_online(cls, dataset_type: 'DatasetType') -> bool:
        """Check if dataset type is online RLHF"""
        return dataset_type.value.startswith('rlhf_online')


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


# ============================================================================
# ALL TRAINING METHODS - Source of truth for what methods exist
# Incompatible = ALL_TRAINING_METHODS - compatible (derived automatically)
# ============================================================================
ALL_TRAINING_METHODS = {"sft", "pt", "rlhf"}

# All RLHF algorithms - source of truth
ALL_RLHF_ALGORITHMS = {"dpo", "orpo", "simpo", "kto", "cpo", "rm", "ppo", "grpo", "gkd"}

def get_incompatible_methods(compatible: list) -> list:
    """Derive incompatible methods from compatible list. Future-proof."""
    return list(ALL_TRAINING_METHODS - set(compatible))

def get_incompatible_rlhf(compatible_rlhf: list) -> list:
    """Derive incompatible RLHF algorithms from compatible list."""
    return list(ALL_RLHF_ALGORITHMS - set(compatible_rlhf))


# Training method compatibility matrix - Maps dataset types to compatible training methods
# NOTE: Only "compatible" is defined. Incompatible is derived: ALL - compatible
TRAINING_METHOD_COMPATIBILITY = {
    # ========== SFT (TEXT) ==========
    DatasetType.SFT: {
        "compatible": ["sft"],
        "rlhf_types": [],  # No RLHF algorithms compatible
        "display_name": "SFT (Instruction-Response)",
        "message": "SFT dataset detected. Only Supervised Fine-Tuning is available.",
        "modality": "text",
    },
    
    # ========== SFT TOOL CALLING ==========
    DatasetType.SFT_TOOL_CALLING: {
        "compatible": ["sft"],
        "rlhf_types": [],
        "display_name": "SFT Tool/Function Calling",
        "message": "Tool calling SFT dataset detected. Contains tool_call/tool roles for function calling training.",
        "modality": "text",
        "format_id": "sft_tool_calling",
    },
    
    # ========== SFT (MULTIMODAL) ==========
    DatasetType.SFT_VISION: {
        "compatible": ["sft"],
        "rlhf_types": [],
        "display_name": "SFT Vision (Image + Text)",
        "message": "Vision SFT dataset detected. Use with vision-language models (LLaVA, Qwen-VL, etc.).",
        "modality": "vision",
    },
    DatasetType.SFT_AUDIO: {
        "compatible": ["sft"],
        "rlhf_types": [],
        "display_name": "SFT Audio (Audio + Text)",
        "message": "Audio SFT dataset detected. Use with audio-language models (Qwen-Audio, etc.).",
        "modality": "audio",
    },
    DatasetType.SFT_VIDEO: {
        "compatible": ["sft"],
        "rlhf_types": [],
        "display_name": "SFT Video (Video + Text)",
        "message": "Video SFT dataset detected. Use with video-language models.",
        "modality": "video",
    },
    
    # ========== PRE-TRAINING ==========
    DatasetType.PT: {
        "compatible": ["pt"],
        "rlhf_types": [],
        "display_name": "Pre-Training (Raw Text)",
        "message": "Pre-training dataset detected. Only Pre-Training is available.",
        "modality": "text",
    },
    
    # ========== RLHF PREFERENCE (DPO/ORPO/SimPO/CPO/RM) ==========
    DatasetType.RLHF_PREF: {
        "compatible": ["rlhf"],
        "rlhf_types": ["dpo", "orpo", "simpo", "cpo", "rm"],  # Only these RLHF algorithms
        "display_name": "RLHF Preference (DPO/ORPO/SimPO/CPO/RM)",
        "message": "RLHF preference dataset detected. Compatible with DPO, ORPO, SimPO, CPO, RM.",
        "modality": "text",
        "format_id": "rlhf_pref_messages",
    },
    DatasetType.RLHF_PREF_VISION: {
        "compatible": ["rlhf"],
        "rlhf_types": ["dpo", "orpo", "simpo", "cpo", "rm"],
        "display_name": "RLHF Preference - Vision",
        "message": "Vision RLHF preference dataset. Use with VLMs (LLaVA, Qwen-VL, etc.).",
        "modality": "vision",
        "format_id": "rlhf_pref_vision",
    },
    DatasetType.RLHF_PREF_AUDIO: {
        "compatible": ["rlhf"],
        "rlhf_types": ["dpo", "orpo", "simpo", "cpo", "rm"],
        "display_name": "RLHF Preference - Audio",
        "message": "Audio RLHF preference dataset. Use with audio models (Qwen-Audio, etc.).",
        "modality": "audio",
        "format_id": "rlhf_pref_audio",
    },
    DatasetType.RLHF_PREF_VIDEO: {
        "compatible": ["rlhf"],
        "rlhf_types": ["dpo", "orpo", "simpo", "cpo", "rm"],
        "display_name": "RLHF Preference - Video",
        "message": "Video RLHF preference dataset. Use with video models.",
        "modality": "video",
        "format_id": "rlhf_pref_video",
    },
    
    # ========== RLHF BINARY (KTO) ==========
    DatasetType.RLHF_BINARY: {
        "compatible": ["rlhf"],
        "rlhf_types": ["kto"],  # ONLY KTO
        "display_name": "RLHF Binary Feedback (KTO)",
        "message": "Binary feedback dataset. Compatible with KTO (Kahneman-Tversky Optimization).",
        "modality": "text",
        "format_id": "rlhf_binary_kto",
    },
    
    # ========== RLHF - ONLINE ==========
    DatasetType.RLHF_ONLINE: {
        "compatible": ["rlhf"],
        "rlhf_types": ["ppo", "grpo", "gkd"],  # Only online algorithms
        "modality": "text",
        "display_name": "RLHF Online (Generation Required)",
        "message": "Online RLHF dataset detected. Compatible with PPO, GRPO, GKD."
    },
    
    # ========== UNKNOWN - STRICTLY REJECTED ==========
    # IMPORTANT: Unknown datasets are NOT allowed for upload or registration
    DatasetType.UNKNOWN: {
        "compatible": [],  # NO training methods allowed - ALL are incompatible
        "rlhf_types": [],  # NO RLHF algorithms allowed
        "display_name": "Unknown Format (REJECTED)",
        "message": "Dataset type could not be determined. Upload/registration rejected. Please use a supported format.",
        "is_valid": False,
    },
}


class DatasetTypeService:
    """Service for detecting dataset types and validating training compatibility"""
    
    # ========================================================================
    # FIELD PATTERNS - VERIFIED AGAINST USF-BIOS preprocessor/core.py
    # ========================================================================
    # USF-BIOS standard_keys: messages, images, videos, audios, tools, objects
    #   + prefixes: rejected_, positive_, negative_
    #   + special: rejected_response, label, channel, margin
    # ========================================================================
    
    # SFT formats (all converted to messages internally by USF-BIOS)
    SFT_FIELDS = {"messages"}                                    # Native messages format
    SFT_ALT_FIELDS = {"instruction", "output"}                   # Alpaca format (input optional)
    SFT_QUERY_FIELDS = {"query", "response"}                     # Query-response format
    
    # RLHF Preference: messages + rejected_messages OR messages + rejected_response
    # USF-BIOS: _check_rejected_response() validates rejected_response field
    RLHF_PREFERENCE_FIELDS = {"messages", "rejected_messages"}   # Primary preference format
    RLHF_PREFERENCE_ALT_FIELDS = {"messages", "rejected_response"}  # Alternative format
    
    # RLHF Simple Preference: prompt/chosen/rejected strings (common HuggingFace format)
    # USF-BIOS: Supported via column mapping (prompt→query, chosen→response, rejected→rejected_response)
    # See dataset_info.json examples: zhihu_rlhf_3k, orca_dpo_pairs, Human-Like-DPO-Dataset
    RLHF_PREF_SIMPLE_FIELDS = {"prompt", "chosen", "rejected"}   # HuggingFace DPO format
    
    # KTO: messages + label (binary: 0=undesirable, 1=desirable)
    # USF-BIOS: KTOPreprocessor in kto.py uses messages + label
    KTO_FIELDS = {"messages", "label"}                           # KTO binary feedback
    
    # Online RLHF: just messages (GRPO/PPO/GKD generate completions)
    RLHF_ONLINE_FIELDS = {"messages"}                            # Same as SFT, used for online RLHF
    
    # Pre-training: raw text
    PT_FIELDS = {"text"}                                         # Primary PT format
    PT_ALT_FIELDS = {"content"}                                  # Alternative PT format
    
    # ========== MULTIMODAL FIELDS ==========
    # Vision: image/images field with path, URL, base64, or PIL Image
    VISION_FIELDS = {"image", "images", "image_url", "image_path", "img", "picture", "photo"}
    # Audio: audio field with path, URL, or array
    AUDIO_FIELDS = {"audio", "audio_path", "audio_url", "speech", "wav", "sound"}
    # Video: video field with path or URL
    VIDEO_FIELDS = {"video", "videos", "video_path", "video_url", "clip"}
    
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
        """
        Detect dataset type from sample data with DEEP STRUCTURE VALIDATION.
        
        PRODUCTION-READY: Handles all dataset types including multimodal:
        - SFT: messages format (text or multimodal)
        - SFT Vision: messages + image/images field
        - SFT Audio: messages + audio field
        - SFT Video: messages + video field
        - PT: raw text for pre-training
        - RLHF Preference: prompt/chosen/rejected
        - RLHF Binary (KTO): prompt/completion/label
        - RLHF Online: prompt only
        - Alpaca: instruction/input/output
        """
        if not samples:
            return self._create_unknown_result("No samples to analyze")
        
        # Handle edge case: sample is not a dict
        if not isinstance(samples[0], dict):
            return self._create_unknown_result(
                f"Invalid sample format: expected dict, got {type(samples[0]).__name__}. "
                "Each row must be a JSON object."
            )
        
        # Get all fields from first sample
        first_sample = samples[0]
        fields = set(first_sample.keys())
        
        # Handle edge case: empty fields
        if not fields:
            return self._create_unknown_result("Sample has no fields. Each row must have at least one field.")
        
        # Collect validation warnings for structure issues
        validation_warnings = []
        
        # ================================================================
        # DETECT MODALITY (vision/audio/video/text)
        # ================================================================
        modality, media_fields = self._detect_modality(fields, samples)
        
        # ================================================================
        # CHECK RLHF PREFERENCE (messages + rejected_messages/rejected_response)
        # USF-BIOS: preprocessor/core.py standard_keys includes rejected_messages, rejected_response
        # ================================================================
        if self.RLHF_PREFERENCE_FIELDS.issubset(fields) or self.RLHF_PREFERENCE_ALT_FIELDS.issubset(fields):
            has_rejected_messages = "rejected_messages" in fields
            has_rejected_response = "rejected_response" in fields
            
            detected_fields = ["messages"]
            if has_rejected_messages:
                detected_fields.append("rejected_messages")
            if has_rejected_response:
                detected_fields.append("rejected_response")
            
            # DEEP VALIDATION: Check messages structure
            valid_structure = True
            for sample in samples[:5]:
                messages = sample.get("messages")
                
                # Validate messages is a list
                if not isinstance(messages, list) or len(messages) == 0:
                    valid_structure = False
                    validation_warnings.append("'messages' field should be a non-empty array")
                    break
                
                # Validate rejected_messages if present
                if has_rejected_messages:
                    rejected = sample.get("rejected_messages")
                    if not isinstance(rejected, list) or len(rejected) == 0:
                        valid_structure = False
                        validation_warnings.append("'rejected_messages' field should be a non-empty array")
                        break
                
                # Validate rejected_response if present
                if has_rejected_response:
                    rejected = sample.get("rejected_response")
                    if not isinstance(rejected, str) or not rejected:
                        valid_structure = False
                        validation_warnings.append("'rejected_response' field should be a non-empty string")
                        break
            
            confidence = 1.0 if valid_structure else 0.7
            
            # Check for multimodal RLHF (with images/audio/video)
            # USF-BIOS supports: rejected_images, rejected_videos, rejected_audios
            if modality == "vision":
                rlhf_type = DatasetType.RLHF_PREF_VISION
                detected_fields = detected_fields + media_fields
            elif modality == "audio":
                rlhf_type = DatasetType.RLHF_PREF_AUDIO
                detected_fields = detected_fields + media_fields
            elif modality == "video":
                rlhf_type = DatasetType.RLHF_PREF_VIDEO
                detected_fields = detected_fields + media_fields
            else:
                rlhf_type = DatasetType.RLHF_PREF
            
            result = self._create_result(rlhf_type, detected_fields, total, confidence, file_size)
            result.validation_warnings = validation_warnings
            return result
        
        # ================================================================
        # CHECK RLHF SIMPLE PREFERENCE (prompt/chosen/rejected strings)
        # Common HuggingFace DPO format - USF-BIOS supports via column mapping
        # See dataset_info.json: zhihu_rlhf_3k, orca_dpo_pairs, Human-Like-DPO-Dataset
        # ================================================================
        if self.RLHF_PREF_SIMPLE_FIELDS.issubset(fields):
            detected_fields = ["prompt", "chosen", "rejected"]
            
            # DEEP VALIDATION: Check that all fields are strings
            valid_structure = True
            for sample in samples[:5]:
                prompt = sample.get("prompt")
                chosen = sample.get("chosen")
                rejected = sample.get("rejected")
                
                # Validate prompt is a string
                if not isinstance(prompt, str) or not prompt.strip():
                    valid_structure = False
                    validation_warnings.append("'prompt' field should be a non-empty string")
                    break
                
                # Validate chosen is a string
                if not isinstance(chosen, str) or not chosen.strip():
                    valid_structure = False
                    validation_warnings.append("'chosen' field should be a non-empty string")
                    break
                
                # Validate rejected is a string
                if not isinstance(rejected, str) or not rejected.strip():
                    valid_structure = False
                    validation_warnings.append("'rejected' field should be a non-empty string")
                    break
                
                # Validate chosen != rejected (should be different)
                if chosen.strip() == rejected.strip():
                    validation_warnings.append("Warning: 'chosen' and 'rejected' are identical in some samples")
            
            confidence = 1.0 if valid_structure else 0.7
            
            # Check for multimodal (though simple format is usually text-only)
            if modality == "vision":
                rlhf_type = DatasetType.RLHF_PREF_VISION
                detected_fields = detected_fields + media_fields
            elif modality == "audio":
                rlhf_type = DatasetType.RLHF_PREF_AUDIO
                detected_fields = detected_fields + media_fields
            elif modality == "video":
                rlhf_type = DatasetType.RLHF_PREF_VIDEO
                detected_fields = detected_fields + media_fields
            else:
                rlhf_type = DatasetType.RLHF_PREF
            
            result = self._create_result(rlhf_type, detected_fields, total, confidence, file_size)
            result.validation_warnings = validation_warnings
            # Add note about USF-BIOS column mapping
            if not result.validation_warnings:
                result.validation_warnings = []
            result.validation_warnings.append(
                "Note: USF-BIOS will auto-map this format (prompt→query, chosen→response, rejected→rejected_response)"
            )
            return result
        
        # ================================================================
        # CHECK KTO (messages + label)
        # USF-BIOS: kto.py uses messages + label (binary: 0/1)
        # ================================================================
        if self.KTO_FIELDS.issubset(fields):
            valid_structure = True
            for sample in samples[:5]:
                messages = sample.get("messages")
                label = sample.get("label")
                
                # Validate messages is a list
                if not isinstance(messages, list) or len(messages) == 0:
                    valid_structure = False
                    validation_warnings.append("'messages' field should be a non-empty array")
                    break
                # Validate label is boolean or 0/1
                if not isinstance(label, (bool, int)) or (isinstance(label, int) and label not in [0, 1]):
                    valid_structure = False
                    validation_warnings.append("'label' field should be boolean (true/false) or 0/1")
                    break
            
            confidence = 1.0 if valid_structure else 0.7
            result = self._create_result(DatasetType.RLHF_BINARY, ["messages", "label"], total, confidence, file_size)
            result.validation_warnings = validation_warnings
            return result
        
        # NOTE: RLHF_ONLINE detection is done AFTER SFT check below
        # Online RLHF (GRPO/PPO/GKD) uses the same format as SFT (messages)
        # The training method selection determines if it's SFT or online RLHF
        
        # ================================================================
        # CHECK SFT MESSAGES FORMAT
        # ================================================================
        if "messages" in fields:
            messages = first_sample.get("messages", [])
            if isinstance(messages, list) and messages:
                # DEEP VALIDATION: Check messages structure across multiple samples
                valid_structure = True
                for sample in samples[:5]:
                    msgs = sample.get("messages", [])
                    if not isinstance(msgs, list):
                        valid_structure = False
                        validation_warnings.append("'messages' field should be an array")
                        break
                    
                    for msg in msgs[:10]:  # Check first 10 messages per sample
                        if not isinstance(msg, dict):
                            valid_structure = False
                            validation_warnings.append("Each message should be an object with 'role' and 'content'")
                            break
                        if "role" not in msg:
                            valid_structure = False
                            validation_warnings.append("Message missing 'role' field")
                            break
                        if "content" not in msg:
                            valid_structure = False
                            validation_warnings.append("Message missing 'content' field")
                            break
                        # Validate role is valid
                        role = msg.get("role", "")
                        valid_roles = {"system", "user", "assistant", "tool", "function", "tool_call", "tool_response"}
                        if role not in valid_roles:
                            validation_warnings.append(f"Unusual role '{role}' found (expected: system/user/assistant)")
                        # Validate content is string
                        if not isinstance(msg.get("content"), (str, list)):
                            validation_warnings.append("Message 'content' should be string or array")
                    
                    if not valid_structure:
                        break
                
                confidence = 1.0 if valid_structure and not validation_warnings else 0.9
                
                # Check for tool calling roles in messages
                has_tool_calling = False
                for sample in samples[:10]:
                    msgs = sample.get("messages", [])
                    if isinstance(msgs, list):
                        for msg in msgs:
                            if isinstance(msg, dict) and msg.get("role") in {"tool_call", "tool", "tool_response"}:
                                has_tool_calling = True
                                break
                    if has_tool_calling:
                        break
                
                # Determine SFT type based on modality and tool calling
                if has_tool_calling and modality == "text":
                    # Tool calling dataset (text only for now)
                    sft_type = DatasetType.SFT_TOOL_CALLING
                    detected_fields = ["messages", "tool_call"]
                    if "tools" in fields:
                        detected_fields.append("tools")
                elif modality == "vision":
                    sft_type = DatasetType.SFT_VISION
                    detected_fields = ["messages"] + media_fields
                elif modality == "audio":
                    sft_type = DatasetType.SFT_AUDIO
                    detected_fields = ["messages"] + media_fields
                elif modality == "video":
                    sft_type = DatasetType.SFT_VIDEO
                    detected_fields = ["messages"] + media_fields
                else:
                    sft_type = DatasetType.SFT
                    detected_fields = ["messages"]
                
                result = self._create_result(sft_type, detected_fields, total, confidence, file_size)
                result.validation_warnings = validation_warnings[:5]  # Limit warnings
                return result
        
        # ================================================================
        # CHECK ALPACA FORMAT (instruction/input/output)
        # ================================================================
        if self.SFT_ALT_FIELDS.issubset(fields):
            valid_structure = True
            for sample in samples[:5]:
                instruction = sample.get("instruction")
                output = sample.get("output")
                
                if not isinstance(instruction, str):
                    valid_structure = False
                    validation_warnings.append("'instruction' field should be a string")
                    break
                if not isinstance(output, str):
                    valid_structure = False
                    validation_warnings.append("'output' field should be a string")
                    break
            
            confidence = 0.9 if valid_structure else 0.7
            
            # Determine SFT type based on modality
            if modality == "vision":
                sft_type = DatasetType.SFT_VISION
                detected_fields = list(self.SFT_ALT_FIELDS & fields) + media_fields
            elif modality == "audio":
                sft_type = DatasetType.SFT_AUDIO
                detected_fields = list(self.SFT_ALT_FIELDS & fields) + media_fields
            elif modality == "video":
                sft_type = DatasetType.SFT_VIDEO
                detected_fields = list(self.SFT_ALT_FIELDS & fields) + media_fields
            else:
                sft_type = DatasetType.SFT
                detected_fields = list(self.SFT_ALT_FIELDS & fields)
            
            result = self._create_result(sft_type, detected_fields, total, confidence, file_size)
            result.validation_warnings = validation_warnings
            return result
        
        # ================================================================
        # CHECK QUERY-RESPONSE FORMAT (SFT)
        # ================================================================
        if self.SFT_QUERY_FIELDS.issubset(fields):
            valid_structure = True
            for sample in samples[:5]:
                query = sample.get("query")
                response = sample.get("response")
                
                if not isinstance(query, str):
                    valid_structure = False
                    validation_warnings.append("'query' field should be a string")
                    break
                if not isinstance(response, str):
                    valid_structure = False
                    validation_warnings.append("'response' field should be a string")
                    break
            
            confidence = 0.9 if valid_structure else 0.7
            
            # Determine SFT type based on modality
            if modality == "vision":
                sft_type = DatasetType.SFT_VISION
                detected_fields = list(self.SFT_QUERY_FIELDS & fields) + media_fields
            elif modality == "audio":
                sft_type = DatasetType.SFT_AUDIO
                detected_fields = list(self.SFT_QUERY_FIELDS & fields) + media_fields
            elif modality == "video":
                sft_type = DatasetType.SFT_VIDEO
                detected_fields = list(self.SFT_QUERY_FIELDS & fields) + media_fields
            else:
                sft_type = DatasetType.SFT
                detected_fields = list(self.SFT_QUERY_FIELDS & fields)
            
            result = self._create_result(sft_type, detected_fields, total, confidence, file_size)
            result.validation_warnings = validation_warnings
            return result
        
        # ================================================================
        # CHECK PRE-TRAINING TEXT FORMAT
        # ================================================================
        if self.PT_FIELDS.issubset(fields) or self.PT_ALT_FIELDS.issubset(fields):
            detected_fields = list(self.PT_FIELDS & fields) or list(self.PT_ALT_FIELDS & fields)
            text_field = "text" if "text" in fields else "content"
            
            valid_structure = True
            for sample in samples[:5]:
                text_value = sample.get(text_field)
                
                if not isinstance(text_value, str):
                    valid_structure = False
                    validation_warnings.append(f"'{text_field}' field should be a string")
                    break
                if len(text_value.strip()) == 0:
                    validation_warnings.append(f"'{text_field}' field contains empty text")
            
            confidence = 0.8 if valid_structure else 0.5
            result = self._create_result(DatasetType.PT, detected_fields, total, confidence, file_size)
            result.validation_warnings = validation_warnings
            return result
        
        # ================================================================
        # CHECK ONLINE RLHF - PROMPT ONLY (less specific, check last)
        # ================================================================
        if "prompt" in fields and "chosen" not in fields and "rejected" not in fields and "response" not in fields:
            valid_structure = True
            for sample in samples[:5]:
                prompt = sample.get("prompt")
                if not self._is_valid_text_or_messages(prompt):
                    valid_structure = False
                    validation_warnings.append("'prompt' field should be string or messages array")
                    break
            
            confidence = 0.7 if valid_structure else 0.5
            result = self._create_result(DatasetType.RLHF_ONLINE, ["prompt"], total, confidence, file_size)
            result.validation_warnings = validation_warnings
            return result
        
        # Fallback: check if there's instruction-like content
        if "instruction" in fields:
            instruction = first_sample.get("instruction")
            if isinstance(instruction, str):
                return self._create_result(DatasetType.SFT, ["instruction"], total, 0.7, file_size)
        
        return self._create_unknown_result("Could not determine dataset type from fields")
    
    def _is_valid_text_or_messages(self, value: Any) -> bool:
        """
        Check if value is valid text content (string) or messages array.
        
        Valid formats:
        - String: "Hello, how can I help?"
        - Messages array: [{"role": "user", "content": "..."}]
        """
        if isinstance(value, str):
            return True
        if isinstance(value, list):
            # Check if it's a messages array
            if all(isinstance(m, dict) and "role" in m and "content" in m for m in value[:3]):
                return True
        return False
    
    def _detect_modality(self, fields: set, samples: List[Dict]) -> Tuple[str, List[str]]:
        """
        Detect modality from fields and sample content.
        
        Returns:
            Tuple of (modality: str, detected_media_fields: List[str])
            modality is one of: 'text', 'vision', 'audio', 'video'
        """
        detected_media_fields = []
        
        # Check for vision fields
        vision_fields_found = self.VISION_FIELDS & fields
        if vision_fields_found:
            # Validate that vision field contains valid data
            for field in vision_fields_found:
                for sample in samples[:3]:
                    value = sample.get(field)
                    if self._is_valid_media_content(value, "vision"):
                        detected_media_fields.append(field)
                        break
            if detected_media_fields:
                return "vision", detected_media_fields
        
        # Check for audio fields
        audio_fields_found = self.AUDIO_FIELDS & fields
        if audio_fields_found:
            for field in audio_fields_found:
                for sample in samples[:3]:
                    value = sample.get(field)
                    if self._is_valid_media_content(value, "audio"):
                        detected_media_fields.append(field)
                        break
            if detected_media_fields:
                return "audio", detected_media_fields
        
        # Check for video fields
        video_fields_found = self.VIDEO_FIELDS & fields
        if video_fields_found:
            for field in video_fields_found:
                for sample in samples[:3]:
                    value = sample.get(field)
                    if self._is_valid_media_content(value, "video"):
                        detected_media_fields.append(field)
                        break
            if detected_media_fields:
                return "video", detected_media_fields
        
        return "text", []
    
    def _is_valid_media_content(self, value: Any, media_type: str) -> bool:
        """
        Check if value is valid media content (image/audio/video).
        
        Valid formats:
        - String: file path, URL, or base64 encoded data
        - List: array of paths/URLs (for multiple images)
        - Dict: with 'path' or 'bytes' key
        - None: skip this sample
        """
        if value is None:
            return False
        
        # String - path, URL, or base64
        if isinstance(value, str):
            # Check if it looks like a path, URL, or base64
            if value.startswith(('http://', 'https://', '/', './', 'data:')):
                return True
            # Check common image extensions
            if media_type == "vision" and any(value.lower().endswith(ext) for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp', '.bmp']):
                return True
            # Check common audio extensions
            if media_type == "audio" and any(value.lower().endswith(ext) for ext in ['.wav', '.mp3', '.flac', '.ogg', '.m4a']):
                return True
            # Check common video extensions
            if media_type == "video" and any(value.lower().endswith(ext) for ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm']):
                return True
            # Could be base64 or relative path
            if len(value) > 10:
                return True
        
        # List - multiple media items
        if isinstance(value, list) and len(value) > 0:
            # Check if items are valid
            return any(self._is_valid_media_content(item, media_type) for item in value[:3])
        
        # Dict with path or bytes
        if isinstance(value, dict):
            return 'path' in value or 'bytes' in value or 'url' in value
        
        return False
    
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
        
        # Derive incompatible from compatible (future-proof)
        compatible = compat["compatible"]
        incompatible = get_incompatible_methods(compatible)
        
        return DatasetTypeInfo(
            dataset_type=dataset_type,
            confidence=confidence,
            detected_fields=detected_fields,
            sample_count=sample_count,
            compatible_training_methods=compatible,
            incompatible_training_methods=incompatible,
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
        # UNKNOWN has empty compatible list, so ALL methods are incompatible
        compatible = compat["compatible"]
        incompatible = get_incompatible_methods(compatible)
        
        return DatasetTypeInfo(
            dataset_type=DatasetType.UNKNOWN,
            confidence=0.0,
            detected_fields=[],
            sample_count=0,
            compatible_training_methods=compatible,
            incompatible_training_methods=incompatible,
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
            # STRICT: Reject unknown dataset types - they cannot be used for training
            return False, (
                "Unable to detect dataset type. Your dataset format is not recognized. "
                "Please ensure your dataset follows one of the supported formats:\n"
                "• SFT: 'messages' array with role/content, or 'instruction'/'output', or 'query'/'response'\n"
                "• Pre-training: 'text' field only\n"
                "• RLHF Preference: 'prompt'/'chosen'/'rejected', or 'messages'/'rejected_messages'\n"
                "• RLHF Online: 'messages' array (prompts only, no assistant response)\n"
                "• KTO: 'messages' array with 'label' field (0 or 1)"
            ), type_info
        
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
        
        # RLHF Preference dataset (DPO, ORPO, SimPO, CPO, RM)
        elif dataset_type == DatasetType.RLHF_PREF:
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
        
        # RLHF Binary Feedback dataset (KTO)
        elif dataset_type == DatasetType.RLHF_BINARY:
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
        
        # STRICT: Unknown types are NOT allowed - reject them
        if new_dataset_type == DatasetType.UNKNOWN:
            return False, "Cannot add dataset with unknown type. Please ensure it follows a supported format."
        if existing_type == DatasetType.UNKNOWN:
            return False, "Selected datasets have unknown type. Please remove them and select valid datasets."
        
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
        # Incompatible = NOT in compatible list (derived, future-proof)
        compat = TRAINING_METHOD_COMPATIBILITY.get(dataset_type, TRAINING_METHOD_COMPATIBILITY[DatasetType.UNKNOWN])
        compatible = compat["compatible"]
        if training_method not in compatible:
            method_name = {"sft": "SFT", "pt": "Pre-Training", "rlhf": "RLHF"}.get(training_method, training_method)
            type_name = {"sft": "SFT", "rlhf": "RLHF preference", "pt": "Pre-training", "kto": "KTO"}.get(dataset_type.value, dataset_type.value)
            errors.append(f"{method_name} is not compatible with {type_name} datasets. {compat['message']}")
        
        # Check KTO-specific validation (RLHF_BINARY only works with KTO algorithm)
        if dataset_type == DatasetType.RLHF_BINARY and rlhf_type and rlhf_type != "kto":
            errors.append(f"Binary feedback dataset (messages+label) is only compatible with KTO algorithm.")
        
        # Check RLHF_PREF specific validation (doesn't work with KTO)
        if dataset_type == DatasetType.RLHF_PREF and rlhf_type == "kto":
            errors.append(f"Preference dataset (messages+rejected) is not compatible with KTO. Use binary feedback dataset (messages+label).")
        
        # Check RLHF_ONLINE specific validation
        if dataset_type == DatasetType.RLHF_ONLINE and rlhf_type and rlhf_type not in ["ppo", "grpo", "gkd"]:
            errors.append(f"Online RLHF dataset only works with PPO, GRPO, or GKD. Selected: {rlhf_type}.")
        
        # Check offline algorithms trying to use online dataset
        if dataset_type == DatasetType.RLHF_ONLINE and rlhf_type in ["dpo", "orpo", "simpo", "cpo", "rm", "kto"]:
            errors.append(f"{rlhf_type.upper()} requires pre-collected data. Use a preference or binary feedback dataset instead of prompt-only.")
        
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
    
    def detect_hub_dataset_type(
        self,
        dataset_id: str,
        source: str = "huggingface",
        subset: Optional[str] = None,
        split: str = "train",
        num_samples: int = 10,
        timeout_seconds: int = 30
    ) -> DatasetTypeInfo:
        """
        Detect dataset type from HuggingFace or ModelScope by streaming a small sample.
        
        Downloads only the first N samples to detect the dataset type without
        downloading the entire dataset. This allows immediate type detection
        for showing available training options in the UI.
        
        Args:
            dataset_id: HuggingFace or ModelScope dataset ID (e.g., "tatsu-lab/alpaca")
            source: "huggingface" or "modelscope"
            subset: Optional subset/config name
            split: Split to sample from (default: "train")
            num_samples: Number of samples to download for detection (default: 10)
            timeout_seconds: Timeout for the operation (default: 30)
            
        Returns:
            DatasetTypeInfo with detected type and compatibility information
        """
        try:
            samples = []
            
            if source == "huggingface":
                samples = self._sample_huggingface_dataset(
                    dataset_id, subset, split, num_samples, timeout_seconds
                )
            elif source == "modelscope":
                samples = self._sample_modelscope_dataset(
                    dataset_id, subset, split, num_samples, timeout_seconds
                )
            else:
                return self._create_unknown_result(f"Unsupported source: {source}")
            
            if not samples:
                return self._create_unknown_result(
                    f"Could not fetch samples from {source}:{dataset_id}. "
                    "The dataset may not exist or requires authentication."
                )
            
            # Detect type from samples
            return self._detect_from_samples(samples, len(samples), 0)
            
        except Exception as e:
            logger.error(f"Error detecting hub dataset type: {e}")
            return self._create_unknown_result(f"Error accessing {source} dataset: {str(e)}")
    
    def _sample_huggingface_dataset(
        self,
        dataset_id: str,
        subset: Optional[str],
        split: str,
        num_samples: int,
        timeout_seconds: int
    ) -> List[Dict[str, Any]]:
        """Stream samples from HuggingFace dataset."""
        samples = []
        try:
            from datasets import load_dataset
            
            # Use streaming to avoid downloading entire dataset
            load_kwargs = {
                "path": dataset_id,
                "split": split,
                "streaming": True,
                "trust_remote_code": True,
            }
            if subset:
                load_kwargs["name"] = subset
            
            dataset = load_dataset(**load_kwargs)
            
            # Take only num_samples
            for i, sample in enumerate(dataset):
                if i >= num_samples:
                    break
                samples.append(dict(sample))
            
            logger.info(f"Sampled {len(samples)} rows from HuggingFace:{dataset_id}")
            
        except ImportError:
            logger.error("datasets library not installed")
        except Exception as e:
            logger.error(f"Error sampling HuggingFace dataset {dataset_id}: {e}")
        
        return samples
    
    def _sample_modelscope_dataset(
        self,
        dataset_id: str,
        subset: Optional[str],
        split: str,
        num_samples: int,
        timeout_seconds: int
    ) -> List[Dict[str, Any]]:
        """Stream samples from ModelScope dataset."""
        samples = []
        try:
            from modelscope.msdatasets import MsDataset
            
            load_kwargs = {
                "dataset_name": dataset_id,
                "split": split,
            }
            if subset:
                load_kwargs["subset_name"] = subset
            
            dataset = MsDataset.load(**load_kwargs)
            
            # Take only num_samples
            for i, sample in enumerate(dataset):
                if i >= num_samples:
                    break
                samples.append(dict(sample))
            
            logger.info(f"Sampled {len(samples)} rows from ModelScope:{dataset_id}")
            
        except ImportError:
            logger.error("modelscope library not installed")
        except Exception as e:
            logger.error(f"Error sampling ModelScope dataset {dataset_id}: {e}")
        
        return samples


# Global instance
dataset_type_service = DatasetTypeService()
