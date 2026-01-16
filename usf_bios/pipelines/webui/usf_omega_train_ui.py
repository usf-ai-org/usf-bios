# Copyright (c) US Inc. All rights reserved.
# USF BIOS - AI Training & Fine-tuning Platform
# USF Omega Training WebUI - Dedicated fine-tuning interface for USF Omega 40B

import json
import os
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import List, Literal, Optional, Tuple

import gradio as gr

from usf_bios.utils import get_logger

logger = get_logger()

# USF Omega Model Configuration (Fixed)
USF_OMEGA_MODEL = "arpitsh018/usf-omega-40b-base"
USF_OMEGA_HIDDEN_SIZE = 5120
USF_OMEGA_MAX_POSITION = 262144

# LoRA Best Practice Defaults (Based on https://thinkingmachines.ai/blog/lora/)
# Key findings:
# 1. LoRA LR should be ~10x FullFT LR
# 2. Alpha = 32 is standard practice
# 3. Apply LoRA to ALL layers (MLP + Attention) for best results
# 4. Rank doesn't significantly affect early training, but higher rank helps long training

LORA_DEFAULTS = {
    'learning_rate': 2e-4,  # 10x typical FullFT rate
    'lora_rank': 64,        # Good balance of capacity vs efficiency
    'lora_alpha': 32,       # Standard practice
    'lora_dropout': 0.05,   # Light regularization
    'target_modules': 'all-linear',  # Apply to all layers (MLP + Attention)
    'warmup_ratio': 0.03,
    'weight_decay': 0.1,
    'max_grad_norm': 1.0,
    'gradient_accumulation_steps': 16,
    'per_device_train_batch_size': 1,
}

QLORA_DEFAULTS = {
    **LORA_DEFAULTS,
    'learning_rate': 2e-4,
    'lora_rank': 64,
    'quant_bits': 4,
}

FULL_FT_DEFAULTS = {
    'learning_rate': 2e-5,  # Lower LR for full fine-tuning
    'warmup_ratio': 0.03,
    'weight_decay': 0.1,
    'max_grad_norm': 1.0,
    'gradient_accumulation_steps': 32,
    'per_device_train_batch_size': 1,
}

# Dataset Format Documentation
DATASET_FORMAT_INFO = '''
## USF Omega Dataset Format

Your dataset must be in **JSONL format** (one JSON object per line) with `messages` array.

### Supported Roles

| Role | Description |
|------|-------------|
| `system` | System instructions for the model |
| `developer` | Developer-level instructions (higher priority than system) |
| `user` | User input/query |
| `assistant` | Model response (can include `reasoning`, `tool_calls`) |
| `tool` | Tool/function result |

### Assistant Message Keys

| Key | Required | Description |
|-----|----------|-------------|
| `role` | ✅ | Always `"assistant"` |
| `content` | ✅ | The response text (or `null` if only tool_calls) |
| `reasoning` | ❌ | Chain-of-thought reasoning (converted during training) |
| `tool_calls` | ❌ | Array of tool/function calls |

### Examples

#### 1. Basic Response
```json
{"role": "assistant", "content": "Python is a programming language."}
```

#### 2. With Reasoning
```json
{
  "role": "assistant",
  "reasoning": "The user wants to know about Python. I should explain it's a programming language known for readability.",
  "content": "Python is a high-level programming language known for its simplicity and readability."
}
```

#### 3. Tool Call
```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "call_123",
      "type": "function",
      "function": {"name": "get_weather", "arguments": "{\\"location\\": \\"Tokyo\\"}"}
    }
  ]
}
```

#### 4. Tool Response
```json
{"role": "tool", "tool_call_id": "call_123", "content": "{\\"temperature\\": 22}"}
```

#### 5. Structured Output (JSON)
```json
{"role": "assistant", "content": "{\\"name\\": \\"John\\", \\"age\\": 30}"}
```

### Complete Example with Tool + Reasoning
```json
{"messages": [
  {"role": "system", "content": "You are a helpful assistant with tool access."},
  {"role": "user", "content": "What's the weather in Tokyo?"},
  {"role": "assistant", "content": null, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\\"location\\": \\"Tokyo\\"}"}}]},
  {"role": "tool", "tool_call_id": "call_1", "content": "{\\"temp\\": 22, \\"condition\\": \\"sunny\\"}"},
  {"role": "assistant", "reasoning": "The tool returned sunny weather at 22°C for Tokyo.", "content": "The weather in Tokyo is sunny with a temperature of 22°C."}
]}
```

### Dataset Sources
- **Local**: `/path/to/your/dataset.jsonl`
- **HuggingFace**: `HF::username/dataset-name`
'''


@dataclass
class TrainingMetrics:
    """Real-time training metrics"""
    loss: Optional[float] = None
    learning_rate: Optional[float] = None
    epoch: Optional[float] = None
    global_step: int = 0
    max_steps: int = 0
    percentage: float = 0.0
    elapsed_time: str = "00:00:00"
    remaining_time: str = "--:--:--"
    memory_gb: Optional[float] = None
    train_speed: Optional[float] = None
    token_acc: Optional[float] = None
    grad_norm: Optional[float] = None


@dataclass
class TrainingState:
    """Holds the current training state"""
    is_running: bool = False
    process: Optional[subprocess.Popen] = None
    logs: List[str] = field(default_factory=list)
    start_time: Optional[float] = None
    metrics: TrainingMetrics = field(default_factory=TrainingMetrics)
    status_message: str = "Ready"


_training_state = TrainingState()

# TensorBoard state
_tensorboard_process: Optional[subprocess.Popen] = None
_tensorboard_port: int = 6006


def find_free_port(start_port: int = 6006) -> int:
    """Find a free port starting from start_port"""
    import socket
    for port in range(start_port, start_port + 100):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    return start_port


def start_tensorboard(log_dir: str) -> Tuple[bool, str, int]:
    """Start TensorBoard server"""
    global _tensorboard_process, _tensorboard_port
    
    if _tensorboard_process is not None:
        return True, "TensorBoard already running", _tensorboard_port
    
    _tensorboard_port = find_free_port(6006)
    
    try:
        _tensorboard_process = subprocess.Popen(
            ['tensorboard', '--logdir', log_dir, '--port', str(_tensorboard_port), '--bind_all'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        time.sleep(2)  # Wait for TensorBoard to start
        return True, f"TensorBoard started on port {_tensorboard_port}", _tensorboard_port
    except Exception as e:
        return False, f"Failed to start TensorBoard: {e}", 0


def stop_tensorboard():
    """Stop TensorBoard server"""
    global _tensorboard_process
    if _tensorboard_process:
        _tensorboard_process.terminate()
        _tensorboard_process = None


def read_tensorboard_metrics(log_dir: str) -> dict:
    """Read latest metrics from TensorBoard event files"""
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
        
        # Find event file
        event_file = None
        for root, dirs, files in os.walk(log_dir):
            for f in files:
                if f.startswith('events.out.tfevents.'):
                    event_file = os.path.join(root, f)
                    break
            if event_file:
                break
        
        if not event_file:
            return {}
        
        ea = EventAccumulator(event_file)
        ea.Reload()
        
        metrics = {}
        tags = ea.Tags().get('scalars', [])
        
        for tag in tags:
            values = ea.Scalars(tag)
            if values:
                # Get latest value
                latest = values[-1]
                metrics[tag] = {
                    'value': latest.value,
                    'step': latest.step,
                    'history': [{'step': v.step, 'value': v.value} for v in values[-50:]]  # Last 50 points
                }
        
        return metrics
    except Exception as e:
        return {'error': str(e)}


def get_tensorboard_charts(log_dir: str) -> str:
    """Generate inline charts from TensorBoard data as HTML"""
    metrics = read_tensorboard_metrics(log_dir)
    
    if not metrics or 'error' in metrics:
        return "<p>No training data available yet.</p>"
    
    html_parts = []
    
    # Loss chart
    loss_data = metrics.get('train/loss', {})
    if loss_data and 'history' in loss_data:
        history = loss_data['history']
        if len(history) > 1:
            # Create simple ASCII/text chart representation
            steps = [h['step'] for h in history]
            values = [h['value'] for h in history]
            min_val, max_val = min(values), max(values)
            current = values[-1]
            start = values[0]
            
            html_parts.append(f'''
            <div style="background: #f5f5f5; padding: 15px; border-radius: 8px; margin: 10px 0;">
                <h4 style="margin: 0 0 10px 0;">📉 Training Loss</h4>
                <div style="display: flex; justify-content: space-between; margin-bottom: 10px;">
                    <span><b>Start:</b> {start:.4f}</span>
                    <span><b>Current:</b> {current:.4f}</span>
                    <span><b>Min:</b> {min_val:.4f}</span>
                </div>
                <div style="background: #e0e0e0; height: 8px; border-radius: 4px; overflow: hidden;">
                    <div style="background: linear-gradient(90deg, #4caf50, #8bc34a); height: 100%; width: {min(100, (1 - (current - min_val) / (start - min_val + 0.001)) * 100):.0f}%;"></div>
                </div>
                <p style="font-size: 12px; color: #666; margin: 5px 0 0 0;">Step {steps[-1]} | {len(history)} data points</p>
            </div>
            ''')
    
    # Learning rate
    lr_data = metrics.get('train/learning_rate', {})
    if lr_data and 'value' in lr_data:
        html_parts.append(f'''
        <div style="background: #e3f2fd; padding: 10px; border-radius: 8px; margin: 10px 0;">
            <b>📈 Learning Rate:</b> {lr_data['value']:.2e} (Step {lr_data['step']})
        </div>
        ''')
    
    # Grad norm
    grad_data = metrics.get('train/grad_norm', {})
    if grad_data and 'value' in grad_data:
        html_parts.append(f'''
        <div style="background: #fff3e0; padding: 10px; border-radius: 8px; margin: 10px 0;">
            <b>📊 Gradient Norm:</b> {grad_data['value']:.4f} (Step {grad_data['step']})
        </div>
        ''')
    
    # Epoch
    epoch_data = metrics.get('train/epoch', {})
    if epoch_data and 'value' in epoch_data:
        html_parts.append(f'''
        <div style="background: #f3e5f5; padding: 10px; border-radius: 8px; margin: 10px 0;">
            <b>🔄 Epoch:</b> {epoch_data['value']:.2f}
        </div>
        ''')
    
    if not html_parts:
        return "<p>Training metrics will appear here once training starts.</p>"
    
    return ''.join(html_parts)


def export_to_storage(output_dir: str, storage_type: str, storage_path: str, hf_repo: str = "") -> Tuple[bool, str]:
    """Export trained model to external storage"""
    try:
        if not os.path.exists(output_dir):
            return False, f"❌ Output directory not found: {output_dir}"
        
        if storage_type == "local":
            # Copy to local path
            import shutil
            dest = os.path.expanduser(storage_path)
            os.makedirs(dest, exist_ok=True)
            
            # Copy all files
            for item in os.listdir(output_dir):
                src = os.path.join(output_dir, item)
                dst = os.path.join(dest, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
            
            return True, f"✅ Exported to: {dest}"
        
        elif storage_type == "s3":
            # AWS S3 upload
            cmd = f"aws s3 sync {output_dir} {storage_path} --exclude '*.bin' --include '*.safetensors' --include '*.json' --include '*.txt'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                return True, f"✅ Uploaded to S3: {storage_path}"
            return False, f"❌ S3 upload failed: {result.stderr}"
        
        elif storage_type == "gcs":
            # Google Cloud Storage
            cmd = f"gsutil -m rsync -r {output_dir} {storage_path}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                return True, f"✅ Uploaded to GCS: {storage_path}"
            return False, f"❌ GCS upload failed: {result.stderr}"
        
        elif storage_type == "huggingface":
            # HuggingFace Hub
            if not hf_repo:
                return False, "❌ HuggingFace repo name required"
            
            cmd = f"huggingface-cli upload {hf_repo} {output_dir} --private"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                return True, f"✅ Uploaded to HuggingFace: {hf_repo}"
            return False, f"❌ HuggingFace upload failed: {result.stderr}"
        
        else:
            return False, f"❌ Unknown storage type: {storage_type}"
    
    except Exception as e:
        return False, f"❌ Export failed: {str(e)}"


def parse_training_log(line: str) -> Optional[dict]:
    """Parse a training log line to extract metrics"""
    import re
    
    metrics = {}
    
    # Parse format: {'loss': 1.73536587, 'grad_norm': 0.3594034, ...}
    if line.startswith('{') and 'loss' in line:
        try:
            # Clean up the line for parsing
            line_clean = line.replace("'", '"')
            data = json.loads(line_clean)
            return data
        except:
            pass
    
    # Parse key-value pairs from log line
    patterns = {
        'loss': r"'loss':\s*([\d.]+)",
        'grad_norm': r"'grad_norm':\s*([\d.]+)",
        'learning_rate': r"'learning_rate':\s*([\d.e-]+)",
        'token_acc': r"'token_acc':\s*([\d.]+)",
        'epoch': r"'epoch':\s*([\d.]+)",
        'global_step/max_steps': r"'global_step/max_steps':\s*'(\d+)/(\d+)'",
        'percentage': r"'percentage':\s*'([\d.]+)%'",
        'elapsed_time': r"'elapsed_time':\s*'([^']+)'",
        'remaining_time': r"'remaining_time':\s*'([^']+)'",
        'memory(GiB)': r"'memory\(GiB\)':\s*([\d.]+)",
        'train_speed(iter/s)': r"'train_speed\(iter/s\)':\s*([\d.]+)",
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, line)
        if match:
            if key == 'global_step/max_steps':
                metrics['global_step'] = int(match.group(1))
                metrics['max_steps'] = int(match.group(2))
            elif key == 'percentage':
                metrics['percentage'] = float(match.group(1))
            elif key in ['elapsed_time', 'remaining_time']:
                metrics[key] = match.group(1)
            elif key == 'memory(GiB)':
                metrics['memory_gb'] = float(match.group(1))
            elif key == 'train_speed(iter/s)':
                metrics['train_speed'] = float(match.group(1))
            else:
                try:
                    metrics[key] = float(match.group(1))
                except:
                    metrics[key] = match.group(1)
    
    return metrics if metrics else None


def update_metrics_from_log(line: str):
    """Update training metrics from a log line"""
    global _training_state
    
    parsed = parse_training_log(line)
    if parsed:
        m = _training_state.metrics
        if 'loss' in parsed:
            m.loss = parsed['loss']
        if 'learning_rate' in parsed:
            m.learning_rate = parsed['learning_rate']
        if 'epoch' in parsed:
            m.epoch = parsed['epoch']
        if 'global_step' in parsed:
            m.global_step = parsed['global_step']
        if 'max_steps' in parsed:
            m.max_steps = parsed['max_steps']
        if 'percentage' in parsed:
            m.percentage = parsed['percentage']
        if 'elapsed_time' in parsed:
            m.elapsed_time = parsed['elapsed_time']
        if 'remaining_time' in parsed:
            m.remaining_time = parsed['remaining_time']
        if 'memory_gb' in parsed:
            m.memory_gb = parsed['memory_gb']
        if 'train_speed' in parsed:
            m.train_speed = parsed['train_speed']
        if 'token_acc' in parsed:
            m.token_acc = parsed['token_acc']
        if 'grad_norm' in parsed:
            m.grad_norm = parsed['grad_norm']
    
    # Update status message based on log content
    if 'Loading checkpoint shards' in line:
        _training_state.status_message = "🔄 Loading model..."
    elif 'Downloading the model' in line:
        _training_state.status_message = "⬇️ Downloading model..."
    elif 'Downloading the dataset' in line:
        _training_state.status_message = "⬇️ Downloading dataset..."
    elif 'Dataset filtered' in line:
        _training_state.status_message = "📊 Processing dataset..."
    elif 'Train:' in line and '%' in line:
        _training_state.status_message = "🏃 Training..."
    elif 'Saving model' in line or 'save_steps' in line.lower():
        _training_state.status_message = "💾 Saving checkpoint..."


def get_defaults_for_train_type(train_type: str) -> dict:
    """Get default hyperparameters based on training type"""
    if train_type == 'lora':
        return LORA_DEFAULTS.copy()
    elif train_type == 'qlora':
        return QLORA_DEFAULTS.copy()
    else:  # full
        return FULL_FT_DEFAULTS.copy()


def generate_training_command(
    train_type: str,
    dataset_path: str,
    # Training params
    num_epochs: float,
    batch_size: int,
    gradient_accumulation: int,
    learning_rate: float,
    max_length: int,
    output_dir: str,
    # LoRA params
    lora_rank: int,
    lora_alpha: int,
    lora_dropout: float,
    target_modules: str,
    # Advanced
    warmup_ratio: float,
    weight_decay: float,
    max_grad_norm: float,
    gradient_checkpointing: bool,
    logging_steps: int,
    save_steps: int,
    eval_steps: int,
    split_ratio: float,
    system_prompt: str,
    # QLoRA specific
    quant_bits: int,
) -> str:
    """Generate the training command"""
    
    cmd_parts = ['PYTHONPATH=/workspace/usf-usf-bios-prod python usf_bios/cli/sft.py']
    
    # Fixed model
    cmd_parts.append(f'    --model "{USF_OMEGA_MODEL}"')
    cmd_parts.append('    --use_hf true')
    
    # Dataset
    cmd_parts.append(f'    --dataset "{dataset_path}"')
    
    # Training type
    cmd_parts.append(f'    --train_type {train_type}')
    cmd_parts.append('    --torch_dtype bfloat16')
    
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
        cmd_parts.append(f'    --target_modules {target_modules}')
    
    # QLoRA specific
    if train_type == 'qlora':
        cmd_parts.append('    --quant_method bnb')
        cmd_parts.append(f'    --quant_bits {quant_bits}')
    
    # Optimization
    cmd_parts.append(f'    --warmup_ratio {warmup_ratio}')
    cmd_parts.append(f'    --weight_decay {weight_decay}')
    cmd_parts.append(f'    --max_grad_norm {max_grad_norm}')
    
    # Logging and saving
    cmd_parts.append(f'    --logging_steps {logging_steps}')
    cmd_parts.append(f'    --save_steps {save_steps}')
    
    # Evaluation
    if eval_steps > 0 and split_ratio > 0:
        cmd_parts.append(f'    --eval_steps {eval_steps}')
        cmd_parts.append(f'    --split_dataset_ratio {split_ratio}')
    
    # Gradient checkpointing
    if gradient_checkpointing:
        cmd_parts.append('    --gradient_checkpointing true')
    
    # System prompt
    if system_prompt and system_prompt.strip():
        cmd_parts.append(f'    --system "{system_prompt}"')
    
    return ' \\\n'.join(cmd_parts)


def validate_dataset(dataset_path: str) -> Tuple[bool, str, str]:
    """Validate dataset format and return preview"""
    try:
        if dataset_path.startswith('HF::'):
            return True, "✅ HuggingFace dataset (will be validated during training)", ""
        
        if not os.path.exists(dataset_path):
            return False, f"❌ File not found: {dataset_path}", ""
        
        valid_samples = []
        errors = []
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 5:  # Check first 5 samples
                    break
                try:
                    data = json.loads(line.strip())
                    
                    # Check for messages
                    if 'messages' not in data:
                        errors.append(f"Line {i+1}: Missing 'messages' field")
                        continue
                    
                    messages = data['messages']
                    if not isinstance(messages, list) or len(messages) == 0:
                        errors.append(f"Line {i+1}: 'messages' must be non-empty array")
                        continue
                    
                    # Validate roles
                    valid_roles = {'system', 'developer', 'user', 'assistant', 'tool', 'tool_call', 'tool_response'}
                    for j, msg in enumerate(messages):
                        if 'role' not in msg:
                            errors.append(f"Line {i+1}, message {j+1}: Missing 'role'")
                        elif msg['role'] not in valid_roles:
                            errors.append(f"Line {i+1}, message {j+1}: Invalid role '{msg['role']}'. Valid: {valid_roles}")
                    
                    valid_samples.append(json.dumps(data, indent=2, ensure_ascii=False)[:500])
                    
                except json.JSONDecodeError as e:
                    errors.append(f"Line {i+1}: Invalid JSON - {str(e)}")
        
        if errors:
            return False, "❌ Validation errors:\n" + "\n".join(errors[:5]), ""
        
        preview = "\n\n---\n\n".join(valid_samples[:3])
        return True, f"✅ Dataset valid ({len(valid_samples)} samples checked)", preview
        
    except Exception as e:
        return False, f"❌ Error: {str(e)}", ""


def start_training(command: str) -> Tuple[str, str]:
    """Start the training process"""
    global _training_state
    
    if _training_state.is_running:
        return "⚠️ Training is already running!", ""
    
    if not command or not command.strip():
        return "❌ Please generate the command first!", ""
    
    try:
        cmd_single = command.replace(' \\\n', ' ')
        
        _training_state.is_running = True
        _training_state.logs = []
        _training_state.start_time = time.time()
        _training_state.metrics = TrainingMetrics()
        _training_state.status_message = "🚀 Starting..."
        
        _training_state.process = subprocess.Popen(
            cmd_single,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd='/workspace/usf-usf-bios-prod'
        )
        
        def read_logs():
            if _training_state.process:
                for line in iter(_training_state.process.stdout.readline, ''):
                    if line:
                        line_stripped = line.strip()
                        _training_state.logs.append(line_stripped)
                        # Parse metrics from log
                        update_metrics_from_log(line_stripped)
                        if len(_training_state.logs) > 500:
                            _training_state.logs = _training_state.logs[-500:]
                
                _training_state.process.wait()
                _training_state.is_running = False
                if _training_state.process.returncode == 0:
                    _training_state.status_message = "✅ Training completed!"
                else:
                    _training_state.status_message = f"❌ Training failed (code: {_training_state.process.returncode})"
        
        thread = threading.Thread(target=read_logs, daemon=True)
        thread.start()
        
        return "✅ Training started!", "Initializing..."
    
    except Exception as e:
        _training_state.is_running = False
        _training_state.status_message = f"❌ Failed: {str(e)}"
        return f"❌ Failed: {str(e)}", ""


def stop_training() -> str:
    """Stop training"""
    global _training_state
    
    if not _training_state.is_running:
        return "⚠️ No training running."
    
    try:
        if _training_state.process:
            _training_state.process.terminate()
            _training_state.process.wait(timeout=10)
        _training_state.is_running = False
        _training_state.status_message = "🛑 Stopped by user"
        return "✅ Training stopped."
    except Exception as e:
        return f"❌ Error: {str(e)}"


def get_training_logs() -> str:
    """Get filtered training logs (important messages only)"""
    if not _training_state.logs:
        return "Waiting for logs..." if _training_state.is_running else "No logs available."
    
    # Filter to show important logs only
    important_keywords = [
        '[INFO:usf]', 'Loading', 'Downloading', 'Dataset', 'Train:', 
        'Saving', 'Error', 'Warning', 'SUCCESS', 'FAILED', 'model_info',
        'lora_config', 'trainable', '%|'
    ]
    
    filtered = []
    for log in _training_state.logs[-200:]:
        # Skip metric lines (they're shown in dashboard)
        if log.startswith('{') and 'loss' in log:
            continue
        # Include important logs
        if any(kw in log for kw in important_keywords):
            filtered.append(log)
    
    return '\n'.join(filtered[-50:]) if filtered else "Processing..."


def get_metrics_display(output_dir: str = "") -> str:
    """Get formatted metrics display for dashboard - uses TensorBoard if available"""
    m = _training_state.metrics
    
    # Try to read from TensorBoard first if output_dir is provided
    tb_metrics = {}
    if output_dir and os.path.exists(output_dir):
        tb_metrics = read_tensorboard_metrics(output_dir)
    
    if not _training_state.is_running and m.loss is None and not tb_metrics:
        return ""
    
    lines = []
    
    # Progress bar
    progress = m.percentage
    bar_length = 30
    filled = int(bar_length * progress / 100) if progress > 0 else 0
    bar = "█" * filled + "░" * (bar_length - filled)
    lines.append(f"**Progress:** [{bar}] {progress:.1f}%")
    lines.append(f"**Step:** {m.global_step} / {m.max_steps}")
    lines.append("")
    
    # Time
    lines.append(f"⏱️ **Elapsed:** {m.elapsed_time}")
    lines.append(f"⏳ **Remaining:** {m.remaining_time}")
    lines.append("")
    
    # Metrics - prefer TensorBoard if available
    loss_val = tb_metrics.get('train/loss', {}).get('value', m.loss)
    if loss_val is not None:
        lines.append(f"📉 **Loss:** {loss_val:.4f}")
    
    if m.token_acc is not None:
        lines.append(f"🎯 **Token Accuracy:** {m.token_acc:.2%}")
    if m.grad_norm is not None:
        lines.append(f"📊 **Grad Norm:** {m.grad_norm:.4f}")
    
    lr_val = tb_metrics.get('train/learning_rate', {}).get('value', m.learning_rate)
    if lr_val is not None:
        lines.append(f"📈 **Learning Rate:** {lr_val:.2e}")
    lines.append("")
    
    # System
    if m.memory_gb is not None:
        lines.append(f"💾 **GPU Memory:** {m.memory_gb:.1f} GiB")
    if m.train_speed is not None:
        lines.append(f"⚡ **Speed:** {m.train_speed:.3f} iter/s")
    if m.epoch is not None:
        lines.append(f"🔄 **Epoch:** {m.epoch:.2f}")
    
    # TensorBoard status
    if _tensorboard_process is not None:
        lines.append("")
        lines.append(f"📊 **TensorBoard:** [Open](http://localhost:{_tensorboard_port})")
    
    return '\n'.join(lines)


def get_training_status() -> str:
    """Get training status"""
    return _training_state.status_message


def build_usf_omega_train_ui(
    lang: Literal['en', 'zh'] = 'en',
) -> gr.Blocks:
    """Build USF Omega Training WebUI"""
    
    with gr.Blocks(
        title="USF Omega Training Studio",
        theme=gr.themes.Soft(),
        css="""
        .model-info { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                      color: white; padding: 20px; border-radius: 12px; margin-bottom: 20px; }
        .section-header { background: #f8fafc; padding: 15px; border-radius: 8px; margin: 10px 0; }
        """
    ) as demo:
        
        # Header with model info
        gr.HTML("""
        <div class="model-info">
            <h1 style="margin:0; font-size: 28px;">🚀 USF Omega Training Studio</h1>
            <p style="margin: 10px 0 0 0; opacity: 0.9;">Fine-tune <strong>arpitsh018/usf-omega-40b-base</strong> with LoRA, QLoRA, or Full Fine-tuning</p>
            <p style="margin: 5px 0 0 0; font-size: 14px; opacity: 0.8;">40B Parameters | 262K Context | BFloat16</p>
        </div>
        """)
        
        with gr.Tabs():
            # Tab 1: Training Configuration
            with gr.TabItem("⚙️ Training Configuration"):
                
                # Training Type Selection
                gr.Markdown("### 1️⃣ Select Training Type")
                train_type = gr.Radio(
                    choices=[
                        ("LoRA (Recommended - Low VRAM)", "lora"),
                        ("QLoRA (4-bit - Lowest VRAM)", "qlora"),
                        ("Full Fine-tuning (High VRAM)", "full"),
                    ],
                    value="lora",
                    label="Training Method",
                    info="LoRA: ~80GB VRAM | QLoRA: ~40GB VRAM | Full: ~160GB+ VRAM"
                )
                
                # Dataset Configuration
                gr.Markdown("### 2️⃣ Dataset Configuration")
                with gr.Row():
                    with gr.Column(scale=3):
                        dataset_path = gr.Textbox(
                            label="Dataset Path",
                            placeholder="HF::username/dataset or /path/to/data.jsonl",
                            info="Local JSONL file or HuggingFace dataset (prefix with HF::)"
                        )
                    with gr.Column(scale=1):
                        validate_btn = gr.Button("🔍 Validate", variant="secondary")
                
                validation_result = gr.Textbox(label="Validation Result", interactive=False)
                dataset_preview = gr.Code(label="Dataset Preview", language="json", lines=8)
                
                # Hyperparameters
                gr.Markdown("### 3️⃣ Hyperparameters")
                gr.Markdown("*Defaults based on [LoRA best practices](https://thinkingmachines.ai/blog/lora/)*")
                
                with gr.Row():
                    with gr.Column():
                        num_epochs = gr.Number(label="Epochs", value=2, minimum=1, maximum=100)
                        batch_size = gr.Number(label="Batch Size", value=1, minimum=1, maximum=8, precision=0)
                        gradient_accumulation = gr.Number(label="Gradient Accumulation", value=16, minimum=1, maximum=128, precision=0)
                        learning_rate = gr.Number(label="Learning Rate", value=2e-4, minimum=1e-7, maximum=1e-2)
                    
                    with gr.Column():
                        max_length = gr.Number(label="Max Sequence Length", value=4096, minimum=512, maximum=65536, precision=0)
                        warmup_ratio = gr.Number(label="Warmup Ratio", value=0.03, minimum=0, maximum=0.5)
                        weight_decay = gr.Number(label="Weight Decay", value=0.1, minimum=0, maximum=1)
                        max_grad_norm = gr.Number(label="Max Gradient Norm", value=1.0, minimum=0, maximum=10)
                
                # LoRA-specific parameters
                with gr.Accordion("🔧 LoRA Parameters", open=True) as lora_accordion:
                    with gr.Row():
                        lora_rank = gr.Number(label="LoRA Rank (r)", value=64, minimum=1, maximum=512, precision=0,
                                             info="Higher = more capacity, more VRAM. 64-128 recommended.")
                        lora_alpha = gr.Number(label="LoRA Alpha (α)", value=32, minimum=1, maximum=512, precision=0,
                                              info="Scaling factor. α=32 is standard practice.")
                        lora_dropout = gr.Number(label="LoRA Dropout", value=0.05, minimum=0, maximum=0.5,
                                                info="Regularization. 0.05 recommended.")
                        target_modules = gr.Textbox(label="Target Modules", value="all-linear",
                                                   info="'all-linear' applies LoRA to all layers (best results)")
                
                # QLoRA-specific
                with gr.Accordion("📦 Quantization (QLoRA)", open=False) as quant_accordion:
                    quant_bits = gr.Radio(choices=[4, 8], value=4, label="Quantization Bits",
                                         info="4-bit: Lower VRAM, slight quality loss | 8-bit: Balance")
                
                # Advanced settings
                with gr.Accordion("⚡ Advanced Settings", open=False):
                    with gr.Row():
                        gradient_checkpointing = gr.Checkbox(label="Gradient Checkpointing", value=True,
                                                            info="Reduces VRAM at cost of ~20% speed")
                        logging_steps = gr.Number(label="Logging Steps", value=10, precision=0)
                        save_steps = gr.Number(label="Save Steps", value=500, precision=0)
                    with gr.Row():
                        eval_steps = gr.Number(label="Eval Steps (0=disable)", value=0, precision=0)
                        split_ratio = gr.Number(label="Train/Val Split", value=0.0, minimum=0, maximum=0.3)
                    
                    system_prompt = gr.Textbox(label="Default System Prompt (Optional)", lines=2,
                                              placeholder="You are a helpful AI assistant.")
                    output_dir = gr.Textbox(label="Output Directory", value="output/usf-omega-sft")
            
            # Tab 2: Dataset Format Guide
            with gr.TabItem("📚 Dataset Format"):
                gr.Markdown(DATASET_FORMAT_INFO)
            
            # Tab 3: Run Training
            with gr.TabItem("▶️ Run Training"):
                with gr.Row():
                    generate_btn = gr.Button("📋 Generate Command", variant="secondary", scale=1)
                    start_btn = gr.Button("🚀 Start Training", variant="primary", scale=2)
                    stop_btn = gr.Button("🛑 Stop", variant="stop", scale=1)
                    refresh_btn = gr.Button("🔄 Refresh", scale=0)
                
                command_output = gr.Code(label="Training Command", language="shell", lines=12)
                
                # Status bar
                status_output = gr.Textbox(label="Status", value="Ready", interactive=False, scale=1)
                
                # Metrics Dashboard
                gr.Markdown("### 📊 Training Metrics")
                with gr.Row():
                    with gr.Column(scale=1):
                        metrics_display = gr.Markdown(
                            value="*Start training to see metrics*",
                            label="Live Metrics"
                        )
                    with gr.Column(scale=2):
                        logs_output = gr.Textbox(
                            label="Training Logs", 
                            lines=15, 
                            max_lines=20, 
                            interactive=False,
                            placeholder="Logs will appear here..."
                        )
            
            # Tab 4: Training Charts (TensorBoard metrics embedded)
            with gr.TabItem("📈 Training Charts"):
                gr.Markdown("### 📊 Training Metrics (Live)")
                gr.Markdown("*Metrics extracted from TensorBoard - auto-refreshes every 5 seconds*")
                
                with gr.Row():
                    chart_refresh_btn = gr.Button("🔄 Refresh Charts", variant="secondary")
                
                tb_charts_display = gr.HTML(
                    value="<p style='color: #666;'>Training charts will appear here once training starts.</p>"
                )
                
                gr.Markdown("---")
                gr.Markdown("### 📋 Training Summary")
                training_summary = gr.HTML(
                    value="<p style='color: #666;'>Summary will be available after training completes.</p>"
                )
            
            # Tab 5: Export to Storage
            with gr.TabItem("💾 Export Model"):
                gr.Markdown("### Export Trained Model to Storage")
                gr.Markdown("Save your trained model to external storage after training completes.")
                
                with gr.Row():
                    export_output_dir = gr.Textbox(
                        label="Output Directory (Source)",
                        value="output/usf-omega-sft",
                        info="The directory containing your trained model"
                    )
                
                gr.Markdown("### Storage Options")
                storage_type = gr.Radio(
                    choices=[
                        ("Local/Network Path", "local"),
                        ("AWS S3", "s3"),
                        ("Google Cloud Storage", "gcs"),
                        ("HuggingFace Hub", "huggingface"),
                    ],
                    value="local",
                    label="Storage Type"
                )
                
                with gr.Group():
                    storage_path = gr.Textbox(
                        label="Storage Path",
                        placeholder="/mnt/storage/models/usf-omega-finetuned or s3://bucket/path",
                        info="Destination path for export"
                    )
                    hf_repo_name = gr.Textbox(
                        label="HuggingFace Repo (if using HF Hub)",
                        placeholder="username/model-name",
                        visible=False
                    )
                
                export_btn = gr.Button("📤 Export Model", variant="primary", size="lg")
                export_status = gr.Textbox(label="Export Status", interactive=False)
                
                gr.Markdown("""
                ### Export Notes
                - **Local**: Copies all model files to specified path
                - **S3**: Requires AWS CLI configured (`aws configure`)
                - **GCS**: Requires `gsutil` configured
                - **HuggingFace**: Requires `huggingface-cli login`
                """)
        
        # Event handlers
        def update_defaults(train_type_val):
            """Update defaults when training type changes"""
            defaults = get_defaults_for_train_type(train_type_val)
            is_lora = train_type_val in ['lora', 'qlora']
            is_qlora = train_type_val == 'qlora'
            
            return (
                defaults.get('learning_rate', 2e-4),
                defaults.get('gradient_accumulation_steps', 16),
                defaults.get('lora_rank', 64) if is_lora else 64,
                defaults.get('lora_alpha', 32) if is_lora else 32,
                gr.update(visible=is_lora),  # lora_accordion
                gr.update(visible=is_qlora),  # quant_accordion
            )
        
        train_type.change(
            update_defaults,
            inputs=[train_type],
            outputs=[learning_rate, gradient_accumulation, lora_rank, lora_alpha, 
                    lora_accordion, quant_accordion]
        )
        
        # Validate dataset
        def on_validate(path):
            is_valid, message, preview = validate_dataset(path)
            return message, preview
        
        validate_btn.click(
            on_validate,
            inputs=[dataset_path],
            outputs=[validation_result, dataset_preview]
        )
        
        # Generate command
        generate_btn.click(
            generate_training_command,
            inputs=[
                train_type, dataset_path, num_epochs, batch_size, gradient_accumulation,
                learning_rate, max_length, output_dir, lora_rank, lora_alpha, lora_dropout,
                target_modules, warmup_ratio, weight_decay, max_grad_norm, gradient_checkpointing,
                logging_steps, save_steps, eval_steps, split_ratio, system_prompt, quant_bits
            ],
            outputs=[command_output]
        )
        
        # Start/Stop training
        def on_start(cmd, out_dir):
            status, logs = start_training(cmd)
            metrics = get_metrics_display(out_dir) or "*Starting...*"
            return status, logs, metrics
        
        start_btn.click(on_start, inputs=[command_output, output_dir], outputs=[status_output, logs_output, metrics_display])
        stop_btn.click(stop_training, outputs=[status_output])
        
        # Refresh all metrics and logs
        def refresh_all(out_dir):
            status = get_training_status()
            logs = get_training_logs()
            metrics = get_metrics_display(out_dir) or "*No training in progress*"
            return status, logs, metrics
        
        refresh_btn.click(refresh_all, inputs=[output_dir], outputs=[status_output, logs_output, metrics_display])
        demo.load(refresh_all, inputs=[output_dir], outputs=[status_output, logs_output, metrics_display], every=3)
        
        # Training Charts handlers
        def refresh_charts(out_dir):
            charts = get_tensorboard_charts(out_dir)
            # Generate summary
            metrics = read_tensorboard_metrics(out_dir)
            if metrics and 'train/loss' in metrics:
                loss_history = metrics['train/loss'].get('history', [])
                if loss_history:
                    start_loss = loss_history[0]['value']
                    end_loss = loss_history[-1]['value']
                    steps = loss_history[-1]['step']
                    improvement = ((start_loss - end_loss) / start_loss) * 100
                    summary = f'''
                    <div style="background: #e8f5e9; padding: 15px; border-radius: 8px;">
                        <h4>📊 Training Summary</h4>
                        <ul>
                            <li><b>Total Steps:</b> {steps}</li>
                            <li><b>Starting Loss:</b> {start_loss:.4f}</li>
                            <li><b>Final Loss:</b> {end_loss:.4f}</li>
                            <li><b>Improvement:</b> {improvement:.1f}%</li>
                        </ul>
                    </div>
                    '''
                    return charts, summary
            return charts, "<p>Summary will be available after training.</p>"
        
        chart_refresh_btn.click(refresh_charts, inputs=[output_dir], outputs=[tb_charts_display, training_summary])
        
        # Auto-refresh charts every 5 seconds
        demo.load(lambda out_dir: get_tensorboard_charts(out_dir), inputs=[output_dir], outputs=[tb_charts_display], every=5)
        
        # Export handlers
        def toggle_hf_repo(st):
            return gr.update(visible=(st == "huggingface"))
        
        storage_type.change(toggle_hf_repo, inputs=[storage_type], outputs=[hf_repo_name])
        
        def on_export(out_dir, st_type, st_path, hf_repo):
            success, msg = export_to_storage(out_dir, st_type, st_path, hf_repo)
            return msg
        
        export_btn.click(on_export, inputs=[export_output_dir, storage_type, storage_path, hf_repo_name], outputs=[export_status])
        
        # Sync output_dir to export_output_dir
        output_dir.change(lambda x: x, inputs=[output_dir], outputs=[export_output_dir])
    
    return demo


def usf_omega_train_ui_main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='USF Omega Training WebUI')
    parser.add_argument('--server_name', type=str, default='0.0.0.0')
    parser.add_argument('--server_port', type=int, default=7861)
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--lang', type=str, default='en', choices=['en', 'zh'])
    
    args = parser.parse_args()
    
    demo = build_usf_omega_train_ui(lang=args.lang)
    demo.queue().launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share
    )


if __name__ == '__main__':
    usf_omega_train_ui_main()
