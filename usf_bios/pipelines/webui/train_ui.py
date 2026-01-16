# Copyright (c) Ultrasafe AI. All rights reserved.
# Training WebUI - Gradio-based interface for fine-tuning LLMs

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

import gradio as gr

from usf_bios.utils import get_logger

logger = get_logger()

# Locale mappings for internationalization
LOCALE = {
    'en': {
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
        'dataset_format_info': '''**Supported Formats:**
- **OpenAI/Messages**: `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`
- **ShareGPT**: `{"conversations": [{"from": "human", "value": "..."}, {"from": "gpt", "value": "..."}]}`
- **Alpaca**: `{"instruction": "...", "input": "...", "output": "..."}`
- **Query-Response**: `{"query": "...", "response": "..."}`

**Supported Roles:** `system`, `user`, `assistant`, `tool_call`, `tool_response`
''',
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
}


@dataclass
class TrainingState:
    """Holds the current training state"""
    is_running: bool = False
    process: Optional[subprocess.Popen] = None
    logs: List[str] = field(default_factory=list)
    start_time: Optional[float] = None


# Global training state
_training_state = TrainingState()


def get_locale(lang: str, key: str) -> str:
    """Get localized string"""
    return LOCALE.get(lang, LOCALE['en']).get(key, key)


def generate_training_command(
    model_id: str,
    use_hf: bool,
    dataset_id: str,
    train_type: str,
    torch_dtype: str,
    num_epochs: float,
    batch_size: int,
    gradient_accumulation: int,
    learning_rate: float,
    max_length: int,
    output_dir: str,
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: str,
    logging_steps: int,
    save_steps: int,
    eval_steps: int,
    warmup_ratio: float,
    weight_decay: float,
    max_grad_norm: float,
    gradient_checkpointing: bool,
    system_prompt: str,
    dataset_sample: int,
    split_ratio: float,
    quantization: str,
    quant_bits: int,
) -> str:
    """Generate the training command based on UI inputs"""
    
    cmd_parts = ['PYTHONPATH=/workspace/usf-usf-bios-prod python usf_bios/cli/sft.py']
    
    # Model configuration
    cmd_parts.append(f'    --model "{model_id}"')
    if use_hf:
        cmd_parts.append('    --use_hf true')
    
    # Dataset configuration
    cmd_parts.append(f'    --dataset "{dataset_id}"')
    
    # Training type
    cmd_parts.append(f'    --train_type {train_type}')
    
    # Data type
    cmd_parts.append(f'    --torch_dtype {torch_dtype}')
    
    # Training parameters
    cmd_parts.append(f'    --num_train_epochs {num_epochs}')
    cmd_parts.append(f'    --per_device_train_batch_size {batch_size}')
    cmd_parts.append(f'    --gradient_accumulation_steps {gradient_accumulation}')
    cmd_parts.append(f'    --learning_rate {learning_rate}')
    cmd_parts.append(f'    --max_length {max_length}')
    cmd_parts.append(f'    --output_dir "{output_dir}"')
    
    # LoRA parameters (if applicable)
    if train_type in ['lora', 'qlora']:
        cmd_parts.append(f'    --lora_rank {lora_rank}')
        cmd_parts.append(f'    --lora_alpha {lora_alpha}')
        cmd_parts.append(f'    --lora_dropout {lora_dropout}')
        if target_modules:
            cmd_parts.append(f'    --target_modules {target_modules}')
    
    # Logging and saving
    cmd_parts.append(f'    --logging_steps {logging_steps}')
    cmd_parts.append(f'    --save_steps {save_steps}')
    
    # Evaluation
    if eval_steps > 0:
        cmd_parts.append(f'    --eval_steps {eval_steps}')
        if split_ratio > 0:
            cmd_parts.append(f'    --split_dataset_ratio {split_ratio}')
    
    # Optimization
    cmd_parts.append(f'    --warmup_ratio {warmup_ratio}')
    cmd_parts.append(f'    --weight_decay {weight_decay}')
    cmd_parts.append(f'    --max_grad_norm {max_grad_norm}')
    
    # Advanced options
    if gradient_checkpointing:
        cmd_parts.append('    --gradient_checkpointing true')
    
    # System prompt
    if system_prompt and system_prompt.strip():
        cmd_parts.append(f'    --system "{system_prompt}"')
    
    # Quantization
    if quantization != 'none':
        cmd_parts.append(f'    --quant_method {quantization}')
        cmd_parts.append(f'    --quant_bits {quant_bits}')
    
    return ' \\\n'.join(cmd_parts)


def preview_dataset(dataset_id: str, num_samples: int = 3) -> str:
    """Preview dataset samples"""
    try:
        if dataset_id.startswith('HF::'):
            return "Dataset preview requires loading from HuggingFace. Please start training to see actual data processing."
        
        if os.path.exists(dataset_id):
            samples = []
            with open(dataset_id, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= num_samples:
                        break
                    try:
                        data = json.loads(line.strip())
                        samples.append(json.dumps(data, indent=2, ensure_ascii=False))
                    except json.JSONDecodeError:
                        continue
            
            if samples:
                return '\n\n---\n\n'.join(samples)
            return "No valid JSON samples found in the file."
        
        return f"File not found: {dataset_id}"
    except Exception as e:
        return f"Error previewing dataset: {str(e)}"


def start_training(command: str) -> Tuple[str, str]:
    """Start the training process"""
    global _training_state
    
    if _training_state.is_running:
        return "⚠️ Training is already running!", ""
    
    try:
        # Convert the formatted command to a single line for execution
        cmd_single = command.replace(' \\\n', ' ')
        
        _training_state.is_running = True
        _training_state.logs = []
        _training_state.start_time = time.time()
        
        # Start the training process
        _training_state.process = subprocess.Popen(
            cmd_single,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd='/workspace/usf-usf-bios-prod'
        )
        
        # Start log reader thread
        def read_logs():
            if _training_state.process:
                for line in iter(_training_state.process.stdout.readline, ''):
                    if line:
                        _training_state.logs.append(line.strip())
                        # Keep only last 500 lines
                        if len(_training_state.logs) > 500:
                            _training_state.logs = _training_state.logs[-500:]
                
                _training_state.process.wait()
                _training_state.is_running = False
        
        thread = threading.Thread(target=read_logs, daemon=True)
        thread.start()
        
        return "✅ Training started successfully!", "Training in progress..."
    
    except Exception as e:
        _training_state.is_running = False
        return f"❌ Failed to start training: {str(e)}", ""


def stop_training() -> str:
    """Stop the training process"""
    global _training_state
    
    if not _training_state.is_running:
        return "⚠️ No training is currently running."
    
    try:
        if _training_state.process:
            _training_state.process.terminate()
            _training_state.process.wait(timeout=10)
        _training_state.is_running = False
        return "✅ Training stopped."
    except Exception as e:
        return f"❌ Error stopping training: {str(e)}"


def get_training_logs() -> str:
    """Get current training logs"""
    if not _training_state.logs:
        if _training_state.is_running:
            return "Waiting for logs..."
        return "No logs available."
    return '\n'.join(_training_state.logs[-100:])  # Return last 100 lines


def get_training_status() -> str:
    """Get training status"""
    if _training_state.is_running:
        elapsed = time.time() - (_training_state.start_time or time.time())
        hours, remainder = divmod(int(elapsed), 3600)
        minutes, seconds = divmod(remainder, 60)
        return f"🏃 Running... (Elapsed: {hours:02d}:{minutes:02d}:{seconds:02d})"
    return "⏹️ Stopped"


def build_train_ui(
    lang: Literal['en', 'zh'] = 'en',
    server_name: str = '0.0.0.0',
    server_port: int = 7861,
    share: bool = False,
) -> gr.Blocks:
    """Build the training WebUI"""
    
    L = lambda key: get_locale(lang, key)
    
    with gr.Blocks(
        title=L('title'),
        theme=gr.themes.Soft(),
        css="""
        .main-title { text-align: center; margin-bottom: 20px; }
        .info-box { background: #f0f9ff; padding: 15px; border-radius: 8px; margin: 10px 0; }
        .warning-box { background: #fff7ed; padding: 15px; border-radius: 8px; margin: 10px 0; }
        """
    ) as demo:
        
        # Header
        gr.Markdown(f"# {L('title')}")
        gr.Markdown(f"*{L('subtitle')}*")
        
        with gr.Tabs():
            # Tab 1: Model Configuration
            with gr.TabItem(L('model_tab')):
                with gr.Row():
                    with gr.Column(scale=2):
                        model_source = gr.Radio(
                            choices=[
                                (L('model_source_hf'), 'hf'),
                                (L('model_source_local'), 'local')
                            ],
                            value='hf',
                            label=L('model_source')
                        )
                        model_id = gr.Textbox(
                            label=L('model_id'),
                            placeholder=L('model_id_placeholder'),
                            value='Qwen/Qwen2.5-7B-Instruct'
                        )
                        use_hf = gr.Checkbox(
                            label=L('use_hf'),
                            value=True,
                            visible=True
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown("""
### Popular Models
- `Qwen/Qwen2.5-7B-Instruct`
- `Qwen/Qwen2.5-14B-Instruct`
- `meta-llama/Llama-3.1-8B-Instruct`
- `mistralai/Mistral-7B-Instruct-v0.3`
- `arpitsh018/usf-omega-40b-base`
                        """)
            
            # Tab 2: Dataset Configuration
            with gr.TabItem(L('dataset_tab')):
                with gr.Row():
                    with gr.Column(scale=2):
                        dataset_source = gr.Radio(
                            choices=[
                                (L('dataset_source_hf'), 'hf'),
                                (L('dataset_source_local'), 'local')
                            ],
                            value='hf',
                            label=L('dataset_source')
                        )
                        dataset_id = gr.Textbox(
                            label=L('dataset_id'),
                            placeholder=L('dataset_id_placeholder'),
                            value='HF::TeichAI/glm-4.7-2000x'
                        )
                        dataset_sample = gr.Number(
                            label=L('dataset_sample'),
                            value=0,
                            precision=0
                        )
                        system_prompt = gr.Textbox(
                            label=L('system_prompt'),
                            placeholder='You are a helpful assistant.',
                            lines=2
                        )
                    
                    with gr.Column(scale=1):
                        gr.Markdown(L('dataset_format_info'))
                
                with gr.Row():
                    preview_btn = gr.Button(L('preview_dataset'), variant='secondary')
                    dataset_preview = gr.Textbox(
                        label=L('dataset_preview'),
                        lines=10,
                        max_lines=20
                    )
            
            # Tab 3: Training Parameters
            with gr.TabItem(L('training_tab')):
                with gr.Row():
                    with gr.Column():
                        train_type = gr.Dropdown(
                            choices=['lora', 'qlora', 'full', 'adalora', 'dora'],
                            value='lora',
                            label=L('train_type')
                        )
                        torch_dtype = gr.Dropdown(
                            choices=['bfloat16', 'float16', 'float32'],
                            value='bfloat16',
                            label=L('torch_dtype')
                        )
                        num_epochs = gr.Number(
                            label=L('num_epochs'),
                            value=2,
                            minimum=1,
                            maximum=100
                        )
                        batch_size = gr.Number(
                            label=L('batch_size'),
                            value=1,
                            minimum=1,
                            maximum=64,
                            precision=0
                        )
                    
                    with gr.Column():
                        gradient_accumulation = gr.Number(
                            label=L('gradient_accumulation'),
                            value=16,
                            minimum=1,
                            maximum=128,
                            precision=0
                        )
                        learning_rate = gr.Number(
                            label=L('learning_rate'),
                            value=1e-4,
                            minimum=1e-7,
                            maximum=1e-1
                        )
                        max_length = gr.Number(
                            label=L('max_length'),
                            value=4096,
                            minimum=128,
                            maximum=131072,
                            precision=0
                        )
                        output_dir = gr.Textbox(
                            label=L('output_dir'),
                            value='output/sft-training'
                        )
                
                # LoRA specific parameters
                with gr.Accordion("LoRA Parameters", open=True):
                    with gr.Row():
                        lora_rank = gr.Number(
                            label=L('lora_rank'),
                            value=64,
                            minimum=1,
                            maximum=512,
                            precision=0
                        )
                        lora_alpha = gr.Number(
                            label=L('lora_alpha'),
                            value=128,
                            minimum=1,
                            maximum=1024,
                            precision=0
                        )
                        lora_dropout = gr.Number(
                            label=L('lora_dropout'),
                            value=0.05,
                            minimum=0,
                            maximum=0.5
                        )
                        target_modules = gr.Textbox(
                            label=L('target_modules'),
                            value='all-linear',
                            placeholder='all-linear, q_proj,v_proj, etc.'
                        )
            
            # Tab 4: Advanced Settings
            with gr.TabItem(L('advanced_tab')):
                with gr.Row():
                    with gr.Column():
                        logging_steps = gr.Number(
                            label=L('logging_steps'),
                            value=10,
                            minimum=1,
                            precision=0
                        )
                        save_steps = gr.Number(
                            label=L('save_steps'),
                            value=500,
                            minimum=1,
                            precision=0
                        )
                        eval_steps = gr.Number(
                            label=L('eval_steps'),
                            value=0,
                            minimum=0,
                            precision=0
                        )
                        split_ratio = gr.Number(
                            label=L('split_ratio'),
                            value=0.0,
                            minimum=0,
                            maximum=0.5
                        )
                    
                    with gr.Column():
                        warmup_ratio = gr.Number(
                            label=L('warmup_ratio'),
                            value=0.05,
                            minimum=0,
                            maximum=0.5
                        )
                        weight_decay = gr.Number(
                            label=L('weight_decay'),
                            value=0.1,
                            minimum=0,
                            maximum=1
                        )
                        max_grad_norm = gr.Number(
                            label=L('max_grad_norm'),
                            value=1.0,
                            minimum=0,
                            maximum=10
                        )
                        gradient_checkpointing = gr.Checkbox(
                            label=L('gradient_checkpointing'),
                            value=True
                        )
                
                with gr.Accordion("Quantization", open=False):
                    with gr.Row():
                        quantization = gr.Dropdown(
                            choices=['none', 'bnb', 'hqq', 'eetq'],
                            value='none',
                            label=L('quantization')
                        )
                        quant_bits = gr.Dropdown(
                            choices=[4, 8],
                            value=4,
                            label=L('quant_bits')
                        )
            
            # Tab 5: Run Training
            with gr.TabItem(L('run_tab')):
                with gr.Row():
                    generate_cmd_btn = gr.Button(L('generate_command'), variant='secondary')
                    start_btn = gr.Button(L('start_training'), variant='primary')
                    stop_btn = gr.Button(L('stop_training'), variant='stop')
                
                command_output = gr.Code(
                    label=L('command_output'),
                    language='bash',
                    lines=20
                )
                
                with gr.Row():
                    status_output = gr.Textbox(
                        label=L('training_status'),
                        value="⏹️ Ready",
                        interactive=False
                    )
                    refresh_btn = gr.Button("🔄 Refresh", variant='secondary')
                
                logs_output = gr.Textbox(
                    label=L('training_logs'),
                    lines=15,
                    max_lines=30,
                    interactive=False
                )
        
        # Event handlers
        def update_use_hf(source):
            return gr.update(value=(source == 'hf'), visible=(source == 'hf'))
        
        model_source.change(update_use_hf, [model_source], [use_hf])
        
        # Generate command
        generate_cmd_btn.click(
            generate_training_command,
            inputs=[
                model_id, use_hf, dataset_id, train_type, torch_dtype,
                num_epochs, batch_size, gradient_accumulation, learning_rate,
                max_length, output_dir, lora_rank, lora_alpha, lora_dropout,
                target_modules, logging_steps, save_steps, eval_steps,
                warmup_ratio, weight_decay, max_grad_norm, gradient_checkpointing,
                system_prompt, dataset_sample, split_ratio, quantization, quant_bits
            ],
            outputs=[command_output]
        )
        
        # Preview dataset
        preview_btn.click(
            preview_dataset,
            inputs=[dataset_id],
            outputs=[dataset_preview]
        )
        
        # Start training
        start_btn.click(
            start_training,
            inputs=[command_output],
            outputs=[status_output, logs_output]
        )
        
        # Stop training
        stop_btn.click(
            stop_training,
            outputs=[status_output]
        )
        
        # Refresh logs and status
        def refresh_all():
            return get_training_status(), get_training_logs()
        
        refresh_btn.click(
            refresh_all,
            outputs=[status_output, logs_output]
        )
        
        # Auto-refresh logs every 5 seconds when training
        demo.load(refresh_all, outputs=[status_output, logs_output], every=5)
    
    return demo


def train_ui_main():
    """Main entry point for training WebUI"""
    import argparse
    
    parser = argparse.ArgumentParser(description='USF BIOS Training WebUI')
    parser.add_argument('--server_name', type=str, default='0.0.0.0', help='Server hostname')
    parser.add_argument('--server_port', type=int, default=7861, help='Server port')
    parser.add_argument('--share', action='store_true', help='Create public link')
    parser.add_argument('--lang', type=str, default='en', choices=['en', 'zh'], help='UI language')
    
    args = parser.parse_args()
    
    demo = build_train_ui(
        lang=args.lang,
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share
    )
    
    demo.queue().launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share
    )


if __name__ == '__main__':
    train_ui_main()
