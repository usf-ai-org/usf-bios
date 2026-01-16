# Copyright (c) Ultrasafe AI. All rights reserved.
# Locale strings for Training WebUI

LOCALE_EN = {
    'title': '🚀 USF BIOS Training Studio',
    'subtitle': 'Fine-tune LLMs with ease - Supports LoRA, QLoRA, Full Fine-tuning',
    'model_tab': '📦 Model Configuration',
    'dataset_tab': '📊 Dataset Configuration',
    'training_tab': '⚙️ Training Parameters',
    'advanced_tab': '🔧 Advanced Settings',
    'run_tab': '▶️ Run Training',
    'model_source': 'Model Source',
    'model_source_hf': 'HuggingFace Hub',
    'model_source_local': 'Local Path',
    'model_id': 'Model ID / Path',
    'model_id_placeholder': 'e.g., Qwen/Qwen2.5-7B-Instruct or /path/to/model',
    'use_hf': 'Use HuggingFace Hub',
    'dataset_source': 'Dataset Source',
    'dataset_source_hf': 'HuggingFace Dataset',
    'dataset_source_local': 'Local File (JSON/JSONL)',
    'dataset_id': 'Dataset ID / Path',
    'dataset_id_placeholder': 'e.g., HF::username/dataset or /path/to/data.jsonl',
    'dataset_format': 'Dataset Format',
    'train_type': 'Training Type',
    'torch_dtype': 'Torch Dtype',
    'num_epochs': 'Number of Epochs',
    'batch_size': 'Batch Size (per device)',
    'gradient_accumulation': 'Gradient Accumulation Steps',
    'learning_rate': 'Learning Rate',
    'max_length': 'Max Sequence Length',
    'output_dir': 'Output Directory',
    'lora_rank': 'LoRA Rank',
    'lora_alpha': 'LoRA Alpha',
    'lora_dropout': 'LoRA Dropout',
    'target_modules': 'Target Modules',
    'logging_steps': 'Logging Steps',
    'save_steps': 'Save Steps',
    'eval_steps': 'Eval Steps',
    'warmup_ratio': 'Warmup Ratio',
    'weight_decay': 'Weight Decay',
    'max_grad_norm': 'Max Gradient Norm',
    'deepspeed': 'DeepSpeed Config',
    'gradient_checkpointing': 'Gradient Checkpointing',
    'flash_attention': 'Flash Attention',
    'start_training': '🚀 Start Training',
    'stop_training': '🛑 Stop Training',
    'training_status': 'Training Status',
    'training_logs': 'Training Logs',
    'generate_command': '📋 Generate Command',
    'command_output': 'Training Command',
    'system_prompt': 'System Prompt (Optional)',
    'dataset_sample': 'Dataset Sample Size (0 = all)',
    'split_ratio': 'Train/Val Split Ratio',
    'preview_dataset': '👁️ Preview Dataset',
    'dataset_preview': 'Dataset Preview',
    'quantization': 'Quantization',
    'quant_bits': 'Quantization Bits',
}

LOCALE = {
    'en': LOCALE_EN,
}


def get_locale(lang: str, key: str) -> str:
    """Get localized string"""
    return LOCALE.get(lang, LOCALE['en']).get(key, key)
