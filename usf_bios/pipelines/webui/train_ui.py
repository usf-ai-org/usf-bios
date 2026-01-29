# Copyright (c) US Inc. All rights reserved.
# USF BIOS - AI Training & Fine-tuning Platform
# Training WebUI - Universal fine-tuning interface for any LLM

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

# ============================================================================
# Research-Based Default Hyperparameters
# Based on: https://thinkingmachines.ai/blog/lora/
# Additional sources:
# - QLoRA paper: https://arxiv.org/abs/2305.14314
# - LoRA paper: https://arxiv.org/abs/2106.09685
# - Scaling Laws for Fine-tuning: https://arxiv.org/abs/2402.12354
# ============================================================================

# Key Research Findings:
# 1. LoRA LR should be ~10x Full Fine-tuning LR (validated across multiple studies)
# 2. Alpha = 2 * Rank is a good starting point, but alpha=32 is widely adopted
# 3. Apply LoRA to ALL linear layers (MLP + Attention) for best results
# 4. Higher rank helps for longer training, but rank 64-128 is usually sufficient
# 5. Warmup ratio 0.03-0.1 helps stabilize training
# 6. Weight decay 0.01-0.1 provides regularization

LORA_DEFAULTS = {
    'learning_rate': 2e-4,      # 10x typical FullFT rate (research finding)
    'lora_rank': 64,            # Good balance of capacity vs efficiency
    'lora_alpha': 128,          # alpha = 2 * rank (common practice)
    'lora_dropout': 0.05,       # Light regularization
    'target_modules': 'all-linear',  # Apply to all layers (MLP + Attention) for best results
    'warmup_ratio': 0.03,       # Standard warmup
    'weight_decay': 0.1,        # Regularization
    'max_grad_norm': 1.0,       # Gradient clipping
    'gradient_accumulation_steps': 16,
    'per_device_train_batch_size': 1,
}

QLORA_DEFAULTS = {
    **LORA_DEFAULTS,
    'learning_rate': 2e-4,      # Same as LoRA
    'lora_rank': 64,            # Can use higher rank due to lower memory
    'lora_alpha': 128,
    'quant_bits': 4,            # 4-bit quantization (NF4)
}

ADALORA_DEFAULTS = {
    **LORA_DEFAULTS,
    'learning_rate': 5e-5,      # Slightly lower for adaptive rank
    'lora_rank': 64,            # Initial rank (will be pruned)
    'target_r': 16,             # Target rank after pruning
}

DORA_DEFAULTS = {
    **LORA_DEFAULTS,
    'learning_rate': 1e-4,      # DoRA uses weight decomposition
    'lora_rank': 64,
}

FULL_FT_DEFAULTS = {
    'learning_rate': 2e-5,      # Lower LR for full fine-tuning
    'warmup_ratio': 0.03,
    'weight_decay': 0.1,
    'max_grad_norm': 1.0,
    'gradient_accumulation_steps': 32,
    'per_device_train_batch_size': 1,
}

# Dataset Format Documentation
DATASET_FORMAT_INFO = '''
## Dataset Format Guide

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

#### 1. Basic Conversation
```json
{"messages": [
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "What is Python?"},
  {"role": "assistant", "content": "Python is a high-level programming language..."}
]}
```

#### 2. With Reasoning
```json
{"messages": [
  {"role": "user", "content": "What is 15 * 23?"},
  {"role": "assistant", "reasoning": "15 * 23 = 15 * 20 + 15 * 3 = 300 + 45 = 345", "content": "The answer is 345."}
]}
```

#### 3. Tool Call
```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [{"id": "call_123", "type": "function", "function": {"name": "get_weather", "arguments": "{\\"location\\": \\"Tokyo\\"}"}}]
}
```

### Dataset Sources
- **Local**: `/path/to/your/dataset.jsonl`
- **HuggingFace**: `HF::username/dataset-name`
'''


@dataclass
class TrainingMetrics:
    """Real-time training metrics - ALL metrics sourced from TensorBoard.
    
    Progress metrics (percentage, elapsed_time, remaining_time, train_speed)
    are calculated programmatically from training state, not parsed from logs.
    
    TensorBoard provides:
    - SFT: loss, learning_rate, grad_norm, token_acc, epoch
    - RLHF/DPO: chosen_rewards, rejected_rewards, reward_margin, nll_loss
    - GRPO/PPO: reward, reward_std, kl, entropy, clip_ratio, policy_loss, value_loss
    - Pretraining: loss, perplexity, throughput
    - Megatron: consumed_samples, loss_scale, params_norm, mem-*
    """
    # Progress metrics (calculated, not from TensorBoard)
    global_step: int = 0
    max_steps: int = 0
    
    # Training type detection
    training_type: str = "sft"  # sft, dpo, grpo, ppo, pt, rm
    
    # All other metrics are stored dynamically from TensorBoard
    # This allows automatic support for any new metrics without code changes
    tb_metrics: dict = field(default_factory=dict)


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
    """Start TensorBoard server for training visualization"""
    global _tensorboard_process, _tensorboard_port
    
    if _tensorboard_process is not None:
        # Check if process is still running
        if _tensorboard_process.poll() is None:
            return True, f"TensorBoard already running on port {_tensorboard_port}", _tensorboard_port
        else:
            _tensorboard_process = None
    
    if not os.path.exists(log_dir):
        return False, f"Log directory not found: {log_dir}", 0
    
    _tensorboard_port = find_free_port(6006)
    
    try:
        # Check if tensorboard is available
        import shutil
        if shutil.which('tensorboard') is None:
            return False, "TensorBoard not installed. Install with: pip install tensorboard", 0
        
        _tensorboard_process = subprocess.Popen(
            ['tensorboard', '--logdir', log_dir, '--port', str(_tensorboard_port), '--bind_all', '--reload_interval', '5'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        time.sleep(3)  # Wait for TensorBoard to initialize
        
        # Verify it started successfully
        if _tensorboard_process.poll() is not None:
            stderr = _tensorboard_process.stderr.read().decode() if _tensorboard_process.stderr else ""
            return False, f"TensorBoard failed to start: {stderr}", 0
        
        logger.info(f"TensorBoard started on port {_tensorboard_port}")
        return True, f"TensorBoard started on port {_tensorboard_port}", _tensorboard_port
    except FileNotFoundError:
        return False, "TensorBoard not found. Install with: pip install tensorboard", 0
    except Exception as e:
        logger.error(f"Failed to start TensorBoard: {e}")
        return False, f"Failed to start TensorBoard: {e}", 0


def stop_tensorboard() -> Tuple[bool, str]:
    """Stop TensorBoard server"""
    global _tensorboard_process
    if _tensorboard_process:
        try:
            _tensorboard_process.terminate()
            _tensorboard_process.wait(timeout=5)
            _tensorboard_process = None
            logger.info("TensorBoard stopped")
            return True, "TensorBoard stopped"
        except subprocess.TimeoutExpired:
            _tensorboard_process.kill()
            _tensorboard_process = None
            return True, "TensorBoard killed"
        except Exception as e:
            return False, f"Error stopping TensorBoard: {e}"
    return True, "TensorBoard not running"


def read_tensorboard_metrics(log_dir: str) -> dict:
    """Read latest metrics from TensorBoard event files.
    
    Searches multiple common TensorBoard directories:
    - {log_dir}/runs/  (USF BIOS default)
    - {log_dir}/logs/
    - {log_dir}/tensorboard/
    - {log_dir}/ (direct)
    
    Returns dict with metric names as keys, each containing:
    - value: latest value (validated)
    - step: step number
    - history: list of recent {step, value} dicts
    
    Supports all training types: SFT, RLHF, Pretraining with proper validation.
    """
    import math
    
    if not log_dir or not os.path.exists(log_dir):
        return {}
    
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return {'error': 'TensorBoard not installed. Install with: pip install tensorboard'}
    
    def is_valid_value(val) -> bool:
        """Validate metric value is reasonable."""
        if val is None:
            return False
        if math.isnan(val) or math.isinf(val):
            return False
        if abs(val) > 1e10:
            return False
        return True
    
    try:
        # Search multiple common TensorBoard directories
        search_dirs = [
            os.path.join(log_dir, 'runs'),        # USF BIOS default
            os.path.join(log_dir, 'logs'),        # Common alternative
            os.path.join(log_dir, 'tensorboard'), # Another common location
            log_dir,                               # Direct in output_dir
        ]
        
        event_files = []
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                for root, dirs, files in os.walk(search_dir):
                    for f in files:
                        if f.startswith('events.out.tfevents.'):
                            event_files.append(os.path.join(root, f))
        
        if not event_files:
            return {}
        
        # Read from all event files and merge metrics
        all_metrics = {}
        
        for event_file in event_files:
            try:
                ea = EventAccumulator(event_file, size_guidance={'scalars': 200})
                ea.Reload()
                
                tags = ea.Tags().get('scalars', [])
                
                for tag in tags:
                    try:
                        values = ea.Scalars(tag)
                        if not values:
                            continue
                        
                        # Filter to valid values only
                        valid_values = [v for v in values if is_valid_value(v.value)]
                        if not valid_values:
                            continue
                        
                        latest = valid_values[-1]
                        
                        # Normalize tag names for consistent access
                        normalized_tag = tag
                        
                        # If this tag already exists, keep the one with more data points
                        if normalized_tag in all_metrics:
                            if len(valid_values) <= len(all_metrics[normalized_tag].get('history', [])):
                                continue
                        
                        all_metrics[normalized_tag] = {
                            'value': round(latest.value, 6),
                            'step': latest.step,
                            'history': [
                                {'step': v.step, 'value': round(v.value, 6)} 
                                for v in valid_values[-100:]
                            ]
                        }
                    except Exception:
                        continue
            except Exception:
                continue
        
        return all_metrics
    except Exception as e:
        logger.warning(f"Error reading TensorBoard metrics: {e}")
        return {'error': str(e)}


def get_tensorboard_charts(log_dir: str) -> str:
    """Generate comprehensive INTERACTIVE training metrics visualization as HTML.
    
    Features:
    - Interactive SVG charts with hover tooltips showing exact values
    - Info (ℹ️) button for each metric explaining what it means
    - Scrollable chart area for long training runs
    - Smooth animations and responsive design
    - Time window support (last N points)
    
    Supports ALL training types: SFT, LoRA, QLoRA, RLHF (DPO, KTO, GRPO, PPO), Pretraining, RM.
    """
    if not log_dir:
        return "<p style='color: #666;'>Set output directory to view training charts.</p>"
    
    metrics = read_tensorboard_metrics(log_dir)
    
    if not metrics:
        return "<p style='color: #666;'>No training data available yet. Charts will appear once training starts logging.</p>"
    
    if 'error' in metrics:
        error_msg = metrics.get('error', 'Unknown error')
        if 'not installed' in error_msg.lower():
            return f"<p style='color: #f44336;'>⚠️ {error_msg}</p>"
        return f"<p style='color: #666;'>Waiting for training data...</p>"
    
    html_parts = []
    displayed_keys = set()
    chart_id_counter = [0]  # Use list to allow modification in nested function
    
    # Metric explanations for info button
    METRIC_INFO = {
        'loss': {'desc': 'Training loss measures how well the model is learning. Lower is better.', 'best': 'Lower', 'target': '< 1.0 for most tasks'},
        'eval_loss': {'desc': 'Validation loss on held-out data. Shows generalization ability.', 'best': 'Lower', 'target': 'Should decrease with train loss'},
        'learning_rate': {'desc': 'Controls how fast the model updates. Usually starts high and decays.', 'best': 'Depends on schedule', 'target': '1e-5 to 1e-4 typical'},
        'grad_norm': {'desc': 'Gradient magnitude. Very high values may indicate instability.', 'best': 'Stable (not too high)', 'target': '< 10 is healthy'},
        'token_acc': {'desc': 'Percentage of correctly predicted tokens. Higher is better.', 'best': 'Higher', 'target': '> 70% is good'},
        'seq_acc': {'desc': 'Percentage of completely correct sequences. Higher is better.', 'best': 'Higher', 'target': 'Varies by task'},
        'reward': {'desc': 'Average reward from reward model. Higher means better responses.', 'best': 'Higher', 'target': 'Should increase over training'},
        'reward_std': {'desc': 'Standard deviation of rewards. Lower means more consistent quality.', 'best': 'Lower', 'target': 'Should decrease'},
        'kl': {'desc': 'KL divergence from reference model. Measures how much policy has changed.', 'best': 'Low but not zero', 'target': '0.01 - 0.1 typical'},
        'entropy': {'desc': 'Entropy of policy distribution. Higher means more exploration.', 'best': 'Moderate', 'target': 'Should not collapse to 0'},
        'policy_loss': {'desc': 'Loss for policy optimization in RL. Should decrease.', 'best': 'Lower', 'target': 'Decreasing trend'},
        'value_loss': {'desc': 'Loss for value function in PPO. Lower is better.', 'best': 'Lower', 'target': 'Decreasing trend'},
        'chosen_rewards': {'desc': 'Rewards for preferred responses in DPO. Should be higher than rejected.', 'best': 'Higher', 'target': '> rejected_rewards'},
        'rejected_rewards': {'desc': 'Rewards for non-preferred responses. Should be lower than chosen.', 'best': 'Lower', 'target': '< chosen_rewards'},
        'margins': {'desc': 'Difference between chosen and rejected rewards. Larger is better.', 'best': 'Higher', 'target': '> 0, increasing'},
        'accuracies': {'desc': 'How often the model prefers the chosen response. Higher is better.', 'best': 'Higher', 'target': '> 0.7 is good'},
        'perplexity': {'desc': 'How surprised the model is by the text. Lower is better.', 'best': 'Lower', 'target': '< 10 for most tasks'},
        'throughput': {'desc': 'Training speed in TFLOP/s. Higher means faster training.', 'best': 'Higher', 'target': 'Maximize for efficiency'},
        'MFU': {'desc': 'Model FLOPs Utilization - hardware efficiency. Higher is better.', 'best': 'Higher', 'target': '> 0.5 is good'},
        'aux_loss': {'desc': 'MoE auxiliary loss for load balancing. Should be low.', 'best': 'Lower', 'target': '< 0.01'},
        'clip_ratio': {'desc': 'Fraction of updates clipped in PPO. Too high means unstable.', 'best': 'Low', 'target': '< 0.2'},
        'nll_loss': {'desc': 'Negative log likelihood loss. Lower means better predictions.', 'best': 'Lower', 'target': 'Decreasing'},
    }
    
    def get_metric_info(metric_name: str) -> dict:
        """Get info for a metric, with smart matching."""
        name_lower = metric_name.lower()
        for key, info in METRIC_INFO.items():
            if key in name_lower:
                return info
        return {'desc': f'Training metric: {metric_name}', 'best': 'See documentation', 'target': 'Varies'}
    
    def get_metric(*keys):
        """Get metric data from multiple possible keys."""
        for key in keys:
            if key in metrics and metrics[key]:
                displayed_keys.add(key)
                return metrics[key]
        return None
    
    def make_interactive_chart(title: str, data: dict, color: str, icon: str, 
                              metric_key: str, fmt: str = ".4f", 
                              lower_is_better: bool = True) -> str:
        """Generate an interactive chart with hover tooltips and info button."""
        if not data or 'history' not in data:
            return ""
        history = data['history']
        if len(history) < 1:
            return ""
        
        chart_id_counter[0] += 1
        chart_id = f"chart_{chart_id_counter[0]}"
        
        values = [h['value'] for h in history]
        steps = [h['step'] for h in history]
        current = values[-1]
        start = values[0] if len(values) > 1 else current
        min_val, max_val = min(values), max(values)
        
        # Calculate improvement
        if lower_is_better:
            improvement = ((start - current) / start * 100) if start > 0 else 0
            best_val, best_label = min_val, "Min"
            trend_good = current <= start
        else:
            improvement = ((current - start) / start * 100) if start > 0 else 0
            best_val, best_label = max_val, "Max"
            trend_good = current >= start
        
        imp_color = "#4caf50" if improvement > 0 else "#f44336"
        imp_sign = "+" if improvement > 0 else ""
        
        # Get metric info
        info = get_metric_info(metric_key)
        
        # Chart dimensions
        width, height = 280, 80
        padding = 5
        chart_width = width - 2 * padding
        chart_height = height - 2 * padding
        
        # Limit to last 100 points for performance, but show scrollable if more
        display_history = history[-100:] if len(history) > 100 else history
        display_values = [h['value'] for h in display_history]
        display_steps = [h['step'] for h in display_history]
        
        # Generate SVG path and circles for hover
        if len(display_values) >= 2:
            val_min, val_max = min(display_values), max(display_values)
            if val_max == val_min:
                val_max = val_min + 1
            
            points = []
            circles = []
            for i, (v, s) in enumerate(zip(display_values, display_steps)):
                x = padding + (i / (len(display_values) - 1)) * chart_width
                y = padding + chart_height - ((v - val_min) / (val_max - val_min)) * chart_height
                points.append(f"{x:.1f},{y:.1f}")
                
                # Add invisible larger circle for easier hover
                circles.append(f'''
                    <circle cx="{x:.1f}" cy="{y:.1f}" r="8" fill="transparent" class="hover-target"
                        data-step="{s}" data-value="{v:.6f}"/>
                    <circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{color.replace('#f', '#3').replace('#e', '#5')}" 
                        class="data-point" style="opacity:0;"/>
                ''')
            
            svg_content = f'''
                <polyline points="{' '.join(points)}" fill="none" 
                    stroke="{color.replace('#f', '#6').replace('#e', '#4')}" stroke-width="2"/>
                {''.join(circles)}
            '''
        else:
            svg_content = ""
        
        # Format values for display
        if abs(current) < 0.001 or abs(current) > 10000:
            val_fmt = ".2e"
        else:
            val_fmt = fmt
        
        return f'''
        <div class="metric-card" style="background: {color}; padding: 12px; border-radius: 10px; margin: 10px 0; 
            border-left: 4px solid {imp_color}; position: relative; transition: all 0.3s ease;">
            
            <!-- Header with title and info button -->
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <h4 style="margin: 0; font-size: 14px;">{icon} {title}</h4>
                <div style="display: flex; align-items: center; gap: 8px;">
                    <span style="color: {imp_color}; font-weight: bold; font-size: 13px;">{imp_sign}{improvement:.1f}%</span>
                    <button onclick="document.getElementById('info_{chart_id}').style.display = 
                        document.getElementById('info_{chart_id}').style.display === 'none' ? 'block' : 'none'"
                        style="background: #fff; border: 1px solid #ddd; border-radius: 50%; width: 20px; height: 20px; 
                        cursor: pointer; font-size: 12px; display: flex; align-items: center; justify-content: center;"
                        title="Click for metric explanation">ℹ️</button>
                </div>
            </div>
            
            <!-- Info panel (hidden by default) -->
            <div id="info_{chart_id}" style="display: none; background: #fff; padding: 10px; border-radius: 6px; 
                margin-bottom: 10px; font-size: 12px; border: 1px solid #e0e0e0;">
                <p style="margin: 0 0 5px 0;"><b>What is this?</b> {info['desc']}</p>
                <p style="margin: 0 0 5px 0;"><b>Best direction:</b> {info['best']}</p>
                <p style="margin: 0;"><b>Target:</b> {info['target']}</p>
            </div>
            
            <!-- Interactive Chart -->
            <div style="position: relative; overflow-x: auto; overflow-y: hidden;" class="chart-container">
                <svg width="{width}" height="{height}" style="display: block;" id="{chart_id}"
                    onmousemove="handleChartHover(event, '{chart_id}')" 
                    onmouseleave="hideChartTooltip('{chart_id}')">
                    <!-- Grid lines -->
                    <line x1="{padding}" y1="{padding}" x2="{padding}" y2="{height-padding}" 
                        stroke="#e0e0e0" stroke-width="1"/>
                    <line x1="{padding}" y1="{height-padding}" x2="{width-padding}" y2="{height-padding}" 
                        stroke="#e0e0e0" stroke-width="1"/>
                    {svg_content}
                </svg>
                <!-- Tooltip -->
                <div id="tooltip_{chart_id}" style="display: none; position: absolute; background: rgba(0,0,0,0.8); 
                    color: white; padding: 6px 10px; border-radius: 4px; font-size: 11px; pointer-events: none; z-index: 100;">
                </div>
            </div>
            
            <!-- Stats row -->
            <div style="display: flex; justify-content: space-between; font-size: 12px; margin-top: 8px; color: #555;">
                <span><b>Start:</b> {start:{val_fmt}}</span>
                <span><b>Current:</b> <span style="color: {'#4caf50' if trend_good else '#f44336'};">{current:{val_fmt}}</span></span>
                <span><b>{best_label}:</b> {best_val:{val_fmt}}</span>
            </div>
            
            <!-- Footer -->
            <div style="display: flex; justify-content: space-between; font-size: 10px; color: #888; margin-top: 4px;">
                <span>Step {data['step']}</span>
                <span>{len(history)} data points{' (showing last 100)' if len(history) > 100 else ''}</span>
            </div>
        </div>
        '''
    
    def make_dual_chart(title: str, data1: dict, data2: dict, label1: str, label2: str, 
                       color: str, icon: str, metric_key: str) -> str:
        """Generate a dual metric chart with both lines."""
        if not data1 or not data2:
            return ""
        
        chart_id_counter[0] += 1
        chart_id = f"dual_chart_{chart_id_counter[0]}"
        
        v1 = data1.get('value', 0)
        v2 = data2.get('value', 0)
        margin = v1 - v2
        
        info = get_metric_info(metric_key)
        
        # Get histories for chart
        hist1 = data1.get('history', [])
        hist2 = data2.get('history', [])
        
        width, height = 280, 80
        svg_content = ""
        
        if len(hist1) >= 2 and len(hist2) >= 2:
            all_vals = [h['value'] for h in hist1] + [h['value'] for h in hist2]
            val_min, val_max = min(all_vals), max(all_vals)
            if val_max == val_min:
                val_max = val_min + 1
            
            # Line for data1 (chosen)
            points1 = []
            for i, h in enumerate(hist1[-50:]):
                x = 5 + (i / max(1, len(hist1[-50:]) - 1)) * 270
                y = 5 + 70 - ((h['value'] - val_min) / (val_max - val_min)) * 70
                points1.append(f"{x:.1f},{y:.1f}")
            
            # Line for data2 (rejected)
            points2 = []
            for i, h in enumerate(hist2[-50:]):
                x = 5 + (i / max(1, len(hist2[-50:]) - 1)) * 270
                y = 5 + 70 - ((h['value'] - val_min) / (val_max - val_min)) * 70
                points2.append(f"{x:.1f},{y:.1f}")
            
            svg_content = f'''
                <polyline points="{' '.join(points1)}" fill="none" stroke="#4caf50" stroke-width="2"/>
                <polyline points="{' '.join(points2)}" fill="none" stroke="#f44336" stroke-width="2"/>
            '''
        
        return f'''
        <div class="metric-card" style="background: {color}; padding: 12px; border-radius: 10px; margin: 10px 0;">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                <h4 style="margin: 0; font-size: 14px;">{icon} {title}</h4>
                <button onclick="document.getElementById('info_{chart_id}').style.display = 
                    document.getElementById('info_{chart_id}').style.display === 'none' ? 'block' : 'none'"
                    style="background: #fff; border: 1px solid #ddd; border-radius: 50%; width: 20px; height: 20px; 
                    cursor: pointer; font-size: 12px;">ℹ️</button>
            </div>
            
            <div id="info_{chart_id}" style="display: none; background: #fff; padding: 10px; border-radius: 6px; 
                margin-bottom: 10px; font-size: 12px; border: 1px solid #e0e0e0;">
                <p style="margin: 0 0 5px 0;"><b>What is this?</b> {info['desc']}</p>
                <p style="margin: 0;"><b>Goal:</b> Chosen (green) should be higher than Rejected (red)</p>
            </div>
            
            <svg width="{width}" height="{height}" style="display: block;">
                {svg_content}
            </svg>
            
            <div style="display: flex; justify-content: space-between; font-size: 12px; margin-top: 8px;">
                <span style="color: #4caf50;">✅ <b>{label1}:</b> {v1:.4f}</span>
                <span style="color: #f44336;">❌ <b>{label2}:</b> {v2:.4f}</span>
                <span>📏 <b>Margin:</b> {margin:.4f}</span>
            </div>
        </div>
        '''
    
    def make_simple_metric(title: str, data: dict, icon: str, color: str, fmt: str = ".4f") -> str:
        """Generate a simple single-value metric display with info."""
        if not data or 'value' not in data:
            return ""
        
        chart_id_counter[0] += 1
        info_id = f"simple_info_{chart_id_counter[0]}"
        info = get_metric_info(title.lower())
        
        return f'''
        <div style="background: {color}; padding: 10px; border-radius: 8px; margin: 6px 0; 
            display: inline-flex; align-items: center; gap: 8px; margin-right: 10px;">
            <span><b>{icon} {title}:</b> {data['value']:{fmt}}</span>
            <span style="font-size:10px;color:#666;">(Step {data.get('step', '?')})</span>
            <button onclick="alert('{info['desc'].replace(chr(39), '')}\\n\\nBest: {info['best']}\\nTarget: {info['target']}')"
                style="background: transparent; border: none; cursor: pointer; font-size: 10px;" title="Info">ℹ️</button>
        </div>
        '''
    
    # Add JavaScript for interactive hover
    html_parts.append('''
    <script>
    function handleChartHover(event, chartId) {
        const svg = document.getElementById(chartId);
        const tooltip = document.getElementById('tooltip_' + chartId);
        const rect = svg.getBoundingClientRect();
        const x = event.clientX - rect.left;
        const y = event.clientY - rect.top;
        
        // Find closest data point
        const circles = svg.querySelectorAll('.hover-target');
        let closest = null;
        let minDist = 20;
        
        circles.forEach(circle => {
            const cx = parseFloat(circle.getAttribute('cx'));
            const cy = parseFloat(circle.getAttribute('cy'));
            const dist = Math.sqrt((x - cx) ** 2 + (y - cy) ** 2);
            if (dist < minDist) {
                minDist = dist;
                closest = circle;
            }
        });
        
        if (closest) {
            const step = closest.getAttribute('data-step');
            const value = parseFloat(closest.getAttribute('data-value'));
            tooltip.innerHTML = `Step: ${step}<br>Value: ${value.toFixed(6)}`;
            tooltip.style.display = 'block';
            tooltip.style.left = (parseFloat(closest.getAttribute('cx')) - 30) + 'px';
            tooltip.style.top = (parseFloat(closest.getAttribute('cy')) - 45) + 'px';
            
            // Show data point
            svg.querySelectorAll('.data-point').forEach(p => p.style.opacity = '0');
            const idx = Array.from(circles).indexOf(closest);
            const points = svg.querySelectorAll('.data-point');
            if (points[idx]) points[idx].style.opacity = '1';
        } else {
            hideChartTooltip(chartId);
        }
    }
    
    function hideChartTooltip(chartId) {
        const tooltip = document.getElementById('tooltip_' + chartId);
        if (tooltip) tooltip.style.display = 'none';
        const svg = document.getElementById(chartId);
        if (svg) svg.querySelectorAll('.data-point').forEach(p => p.style.opacity = '0');
    }
    </script>
    <style>
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
    .chart-container::-webkit-scrollbar { height: 4px; }
    .chart-container::-webkit-scrollbar-thumb { background: #ccc; border-radius: 2px; }
    </style>
    ''')
    
    # === PROGRESS SECTION (Real-time calculated metrics) ===
    m = _training_state.metrics
    step = m.global_step
    max_steps = m.max_steps
    
    # Fallback: Get step from TensorBoard if not available from logs
    if step == 0 and metrics:
        # Find latest step from any metric
        for key, data in metrics.items():
            if isinstance(data, dict) and 'step' in data:
                step = max(step, data['step'])
    
    # Try to get max_steps from TensorBoard history if not set
    if max_steps == 0 and metrics:
        tb_max = get_max_steps_from_tensorboard(metrics)
        if tb_max > 0:
            # Estimate: if current step is from TensorBoard, we don't know max_steps
            # Use current step as indication of progress
            max_steps = tb_max if tb_max > step else step
    
    if step > 0 or max_steps > 0:
        html_parts.append("<h3 style='margin: 15px 0 10px 0; border-bottom: 2px solid #ddd; padding-bottom: 5px;'>📈 Training Progress</h3>")
        
        # Calculate progress
        percentage = (step / max_steps * 100) if max_steps > 0 else 0
        
        # Calculate time metrics
        elapsed_seconds = 0
        elapsed_str = "00:00:00"
        remaining_str = "--:--:--"
        train_speed = 0.0
        eta_str = ""
        
        if _training_state.start_time:
            import time as time_module
            elapsed_seconds = time_module.time() - _training_state.start_time
            elapsed_str = format_time(elapsed_seconds)
            if percentage > 0:
                remaining_seconds = (elapsed_seconds / percentage * 100) - elapsed_seconds
                remaining_str = format_time(remaining_seconds)
                # ETA calculation
                eta_timestamp = time_module.time() + remaining_seconds
                eta_str = time_module.strftime("%H:%M:%S", time_module.localtime(eta_timestamp))
            if elapsed_seconds > 0 and step > 0:
                train_speed = step / elapsed_seconds
        
        # Progress bar with percentage
        bar_width = 100
        filled_pct = min(percentage, 100)
        progress_color = "#4caf50" if percentage < 100 else "#2196f3"
        
        html_parts.append(f'''
        <div style="background: #f5f5f5; padding: 15px; border-radius: 8px; margin: 10px 0;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 8px;">
                <span style="font-weight: bold; font-size: 18px;">Step {step:,} / {max_steps:,}</span>
                <span style="font-weight: bold; font-size: 18px; color: {progress_color};">{percentage:.1f}%</span>
            </div>
            <div style="background: #e0e0e0; height: 20px; border-radius: 10px; overflow: hidden; margin-bottom: 10px;">
                <div style="background: linear-gradient(90deg, {progress_color}, #8bc34a); height: 100%; width: {filled_pct:.1f}%; transition: width 0.3s;"></div>
            </div>
            <div style="display: flex; justify-content: space-between; font-size: 13px; color: #666;">
                <span>⏱️ Elapsed: <b>{elapsed_str}</b></span>
                <span>⏳ Remaining: <b>{remaining_str}</b></span>
                <span>⚡ Speed: <b>{train_speed:.2f}</b> it/s</span>
            </div>
            {f'<div style="text-align: center; margin-top: 8px; font-size: 12px; color: #888;">ETA: {eta_str}</div>' if eta_str else ''}
        </div>
        ''')
        
        # Step history sparkline from loss data (shows training progression)
        loss_data = get_metric('train/loss', 'loss', 'lm_loss')
        if loss_data and 'history' in loss_data and len(loss_data['history']) > 1:
            history = loss_data['history']
            steps_history = [h['step'] for h in history]
            if len(steps_history) > 1:
                # Create step progression sparkline
                min_step, max_step = min(steps_history), max(steps_history)
                if max_step > min_step:
                    width, height = 300, 30
                    points = []
                    for i, s in enumerate(steps_history):
                        x = (i / (len(steps_history) - 1)) * width
                        y = height - ((s - min_step) / (max_step - min_step)) * height
                        points.append(f"{x:.1f},{y:.1f}")
                    
                    html_parts.append(f'''
                    <div style="background: #e8f5e9; padding: 10px; border-radius: 8px; margin: 8px 0;">
                        <div style="font-size: 12px; color: #666; margin-bottom: 5px;">📊 Step Progression</div>
                        <svg width="{width}" height="{height}">
                            <polyline points="{' '.join(points)}" fill="none" stroke="#4caf50" stroke-width="2"/>
                        </svg>
                        <div style="display: flex; justify-content: space-between; font-size: 11px; color: #888;">
                            <span>Step {min_step}</span>
                            <span>Step {max_step}</span>
                        </div>
                    </div>
                    ''')
    
    # === CORE TRAINING METRICS (All Types) - INTERACTIVE CHARTS ===
    html_parts.append("<h3 style='margin: 15px 0 10px 0; border-bottom: 2px solid #ddd; padding-bottom: 5px;'>📊 Core Metrics</h3>")
    
    loss_data = get_metric('train/loss', 'loss', 'lm_loss')
    if loss_data:
        html_parts.append(make_interactive_chart("Training Loss", loss_data, "#ffebee", "📉", "loss", ".4f", True))
    
    eval_loss = get_metric('eval/loss', 'eval_loss')
    if eval_loss:
        html_parts.append(make_interactive_chart("Eval Loss", eval_loss, "#e8eaf6", "📋", "eval_loss", ".4f", True))
    
    lr_data = get_metric('train/learning_rate', 'learning_rate', 'learning-rate')
    if lr_data:
        html_parts.append(make_interactive_chart("Learning Rate", lr_data, "#e3f2fd", "📈", "learning_rate", ".2e", False))
    
    grad_data = get_metric('train/grad_norm', 'grad_norm', 'grad-norm')
    if grad_data:
        html_parts.append(make_interactive_chart("Gradient Norm", grad_data, "#fff3e0", "📊", "grad_norm", ".4f", True))
    
    # Accuracy metrics
    token_acc = get_metric('train/token_acc', 'token_acc', 'train/token_accuracy')
    seq_acc = get_metric('train/seq_acc', 'seq_acc')
    acc_data = get_metric('train/acc', 'acc', 'accuracy')
    if token_acc:
        html_parts.append(make_interactive_chart("Token Accuracy", token_acc, "#e8f5e9", "🎯", "token_acc", ".2%", False))
    if seq_acc:
        html_parts.append(make_interactive_chart("Seq Accuracy", seq_acc, "#e8f5e9", "🎯", "seq_acc", ".2%", False))
    if acc_data and not token_acc and not seq_acc:
        html_parts.append(make_interactive_chart("Accuracy", acc_data, "#e8f5e9", "🎯", "token_acc", ".2%", False))
    
    # MFU
    mfu_data = get_metric('MFU', 'train/MFU')
    if mfu_data:
        html_parts.append(make_simple_metric("MFU", mfu_data, "⚡", "#f3e5f5", ".4f"))
    
    # Epoch
    epoch_data = get_metric('train/epoch', 'epoch')
    if epoch_data:
        html_parts.append(make_simple_metric("Epoch", epoch_data, "🔄", "#f3e5f5", ".2f"))
    
    # === DPO/OFFLINE RLHF METRICS - INTERACTIVE CHARTS ===
    chosen_data = get_metric('rewards/chosen', 'train/rewards/chosen', 'chosen_rewards')
    rejected_data = get_metric('rewards/rejected', 'train/rewards/rejected', 'rejected_rewards')
    accuracy_data = get_metric('rewards/accuracies', 'train/rewards/accuracies')
    
    if chosen_data or rejected_data or accuracy_data:
        html_parts.append("<h3 style='margin: 15px 0 10px 0; border-bottom: 2px solid #ddd; padding-bottom: 5px;'>⚖️ DPO/RLHF Metrics</h3>")
        
        if chosen_data and rejected_data:
            html_parts.append(make_dual_chart("Rewards", chosen_data, rejected_data, "Chosen", "Rejected", "#e8f5e9", "🎁", "chosen_rewards"))
        
        margins_data = get_metric('rewards/margins', 'train/rewards/margins')
        if margins_data:
            html_parts.append(make_interactive_chart("Reward Margins", margins_data, "#c8e6c9", "📏", "margins", ".4f", False))
        
        if accuracy_data:
            html_parts.append(make_interactive_chart("Reward Accuracy", accuracy_data, "#dcedc8", "🎯", "accuracies", ".2%", False))
        
        # Log probabilities
        chosen_logps = get_metric('logps/chosen', 'train/logps/chosen')
        rejected_logps = get_metric('logps/rejected', 'train/logps/rejected')
        if chosen_logps and rejected_logps:
            html_parts.append(make_dual_chart("Log Probabilities", chosen_logps, rejected_logps, "Chosen", "Rejected", "#f1f8e9", "📊", "logps"))
        
        nll_data = get_metric('nll_loss', 'train/nll_loss')
        if nll_data:
            html_parts.append(make_interactive_chart("NLL Loss", nll_data, "#fff8e1", "📉", "nll_loss", ".4f", True))
    
    # === GRPO/PPO/ONLINE RL METRICS - INTERACTIVE CHARTS ===
    reward_data = get_metric('reward', 'train/reward', 'objective/rlhf_reward_mean')
    kl_data = get_metric('kl', 'train/kl', 'objective/kl')
    policy_loss = get_metric('policy_loss', 'train/policy_loss', 'ppo/loss/policy')
    
    if reward_data or kl_data or policy_loss:
        html_parts.append("<h3 style='margin: 15px 0 10px 0; border-bottom: 2px solid #ddd; padding-bottom: 5px;'>🎲 Online RL Metrics</h3>")
        
        if reward_data:
            html_parts.append(make_interactive_chart("Reward", reward_data, "#e8f5e9", "🎁", "reward", ".4f", False))
        
        reward_std = get_metric('reward_std', 'train/reward_std')
        if reward_std:
            html_parts.append(make_simple_metric("Reward Std", reward_std, "📊", "#f1f8e9", ".4f"))
        
        if kl_data:
            html_parts.append(make_interactive_chart("KL Divergence", kl_data, "#fce4ec", "🔄", "kl", ".6f", True))
        
        # Entropy
        entropy_mean = get_metric('entropy/mean', 'entropy', 'train/entropy')
        if entropy_mean:
            html_parts.append(make_interactive_chart("Entropy", entropy_mean, "#e0f7fa", "🎲", "entropy", ".4f", False))
        
        # Policy/Value losses
        if policy_loss:
            html_parts.append(make_interactive_chart("Policy Loss", policy_loss, "#ffebee", "📉", "policy_loss", ".4f", True))
        
        value_loss = get_metric('value_loss', 'train/value_loss', 'ppo/loss/value')
        if value_loss:
            html_parts.append(make_interactive_chart("Value Loss", value_loss, "#fff3e0", "📈", "value_loss", ".4f", True))
        
        # Clip metrics
        clip_region = get_metric('clip_ratio/region_mean')
        clip_frac = get_metric('objective/clip_fraction', 'clip_fraction')
        if clip_region:
            html_parts.append(make_simple_metric("Clip Ratio", clip_region, "✂️", "#fce4ec", ".4f"))
        if clip_frac:
            html_parts.append(make_simple_metric("Clip Fraction", clip_frac, "✂️", "#fce4ec", ".4f"))
        
        # Completion metrics
        comp_len = get_metric('completions/mean_length')
        if comp_len:
            html_parts.append(make_simple_metric("Completion Length", comp_len, "📝", "#e1f5fe", ".1f"))
    
    # === PRETRAINING/MEGATRON METRICS - INTERACTIVE CHARTS ===
    ppl_data = get_metric('perplexity', 'train/perplexity')
    throughput_data = get_metric('throughput', 'train/throughput')
    iter_time = get_metric('iteration-time')
    
    if ppl_data or throughput_data or iter_time:
        html_parts.append("<h3 style='margin: 15px 0 10px 0; border-bottom: 2px solid #ddd; padding-bottom: 5px;'>📚 Pretrain/Megatron Metrics</h3>")
        
        if ppl_data:
            html_parts.append(make_interactive_chart("Perplexity", ppl_data, "#fff8e1", "📐", "perplexity", ".2f", True))
        
        if throughput_data:
            html_parts.append(make_interactive_chart("Throughput", throughput_data, "#f1f8e9", "🚀", "throughput", ".1f", False))
        
        if iter_time and 'value' in iter_time:
            html_parts.append(f'''
            <div style="background: #e8eaf6; padding: 10px; border-radius: 8px; margin: 6px 0; display: inline-block;">
                <b>⏱️ Iter Time:</b> {iter_time['value']*1000:.1f} ms
            </div>
            ''')
        
        loss_scale = get_metric('loss-scale')
        if loss_scale:
            html_parts.append(make_simple_metric("Loss Scale", loss_scale, "📊", "#e8eaf6", ".0f"))
        
        params_norm = get_metric('params-norm')
        if params_norm:
            html_parts.append(make_simple_metric("Params Norm", params_norm, "📏", "#f3e5f5", ".4f"))
    
    # === MOE METRICS - INTERACTIVE CHARTS ===
    aux_loss = get_metric('aux_loss', 'train/aux_loss')
    load_bal = get_metric('load_balancing_loss', 'train/load_balancing_loss')
    z_loss = get_metric('z_loss', 'train/z_loss')
    
    if aux_loss or load_bal or z_loss:
        html_parts.append("<h3 style='margin: 15px 0 10px 0; border-bottom: 2px solid #ddd; padding-bottom: 5px;'>🔀 MoE Metrics</h3>")
        if aux_loss:
            html_parts.append(make_interactive_chart("Aux Loss", aux_loss, "#e1f5fe", "🔀", "aux_loss", ".6f", True))
        if load_bal:
            html_parts.append(make_interactive_chart("Load Balance Loss", load_bal, "#e0f2f1", "⚖️", "aux_loss", ".6f", True))
        if z_loss:
            html_parts.append(make_interactive_chart("Z Loss", z_loss, "#f3e5f5", "📊", "aux_loss", ".6f", True))
    
    # === MEMORY METRICS ===
    mem_reserved = get_metric('mem-reserved-bytes')
    mem_allocated = get_metric('mem-allocated-bytes')
    
    if mem_reserved or mem_allocated:
        html_parts.append("<h3 style='margin: 15px 0 10px 0; border-bottom: 2px solid #ddd; padding-bottom: 5px;'>💾 Memory</h3>")
        if mem_reserved and 'value' in mem_reserved:
            html_parts.append(f'''
            <div style="background: #eceff1; padding: 10px; border-radius: 8px; margin: 6px 0; display: inline-block; margin-right: 10px;">
                <b>💾 Reserved:</b> {mem_reserved['value'] / (1024**3):.1f} GiB
            </div>
            ''')
        if mem_allocated and 'value' in mem_allocated:
            html_parts.append(f'''
            <div style="background: #eceff1; padding: 10px; border-radius: 8px; margin: 6px 0; display: inline-block;">
                <b>💾 Allocated:</b> {mem_allocated['value'] / (1024**3):.1f} GiB
            </div>
            ''')
    
    # === DYNAMIC: ALL OTHER METRICS ===
    skip_patterns = ['vs samples', 'histogram', 'image']
    remaining = {}
    for key, data in metrics.items():
        if key in displayed_keys:
            continue
        if any(p in key.lower() for p in skip_patterns):
            continue
        if isinstance(data, dict) and 'value' in data:
            remaining[key] = data
    
    if remaining:
        html_parts.append("<h3 style='margin: 15px 0 10px 0; border-bottom: 2px solid #ddd; padding-bottom: 5px;'>📋 Other Metrics</h3>")
        html_parts.append("<div style='display: flex; flex-wrap: wrap; gap: 8px;'>")
        for key, data in sorted(remaining.items())[:15]:
            name = key.replace('train/', '').replace('/', ' ').replace('_', ' ').title()
            val = data['value']
            if isinstance(val, float):
                if abs(val) < 0.001 or abs(val) > 1000:
                    fmt_val = f"{val:.2e}"
                else:
                    fmt_val = f"{val:.4f}"
            else:
                fmt_val = str(val)
            html_parts.append(f'''
            <div style="background: #f5f5f5; padding: 8px 12px; border-radius: 6px; font-size: 13px;">
                <b>{name}:</b> {fmt_val}
            </div>
            ''')
        html_parts.append("</div>")
    
    if len(html_parts) <= 1:
        return "<p>Training metrics will appear here once training starts.</p>"
    
    return ''.join(html_parts)


def export_to_storage(output_dir: str, storage_type: str, storage_path: str, hf_repo: str = "") -> Tuple[bool, str]:
    """Export trained model to external storage"""
    try:
        if not os.path.exists(output_dir):
            return False, f"❌ Output directory not found: {output_dir}"
        
        if storage_type == "local":
            import shutil
            dest = os.path.expanduser(storage_path)
            os.makedirs(dest, exist_ok=True)
            
            for item in os.listdir(output_dir):
                src = os.path.join(output_dir, item)
                dst = os.path.join(dest, item)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
            
            return True, f"✅ Exported to: {dest}"
        
        elif storage_type == "s3":
            cmd = f"aws s3 sync {output_dir} {storage_path} --exclude '*.bin' --include '*.safetensors' --include '*.json' --include '*.txt'"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                return True, f"✅ Uploaded to S3: {storage_path}"
            return False, f"❌ S3 upload failed: {result.stderr}"
        
        elif storage_type == "gcs":
            cmd = f"gsutil -m rsync -r {output_dir} {storage_path}"
            result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
            if result.returncode == 0:
                return True, f"✅ Uploaded to GCS: {storage_path}"
            return False, f"❌ GCS upload failed: {result.stderr}"
        
        elif storage_type == "huggingface":
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


def format_time(seconds: float) -> str:
    """Format seconds into HH:MM:SS string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def detect_training_type(tb_metrics: dict) -> str:
    """Detect training type from TensorBoard metrics."""
    metric_keys = set(tb_metrics.keys())
    
    # Check for RLHF indicators
    if any('reward' in k.lower() and 'chosen' in k.lower() for k in metric_keys):
        return 'dpo'
    if any('policy_loss' in k.lower() for k in metric_keys):
        return 'ppo'
    if any(k.lower() in ['reward', 'kl'] for k in metric_keys):
        return 'grpo'
    if any('perplexity' in k.lower() for k in metric_keys):
        return 'pt'
    
    return 'sft'


def extract_step_from_log(line: str) -> Optional[Tuple[int, int]]:
    """Extract step/max_steps from log line for progress tracking ONLY.
    
    IMPORTANT: This is FAIL-SAFE - it will NEVER crash or cause issues.
    If parsing fails for any reason, it simply returns None.
    
    All training metrics (loss, lr, rewards, etc.) come from TensorBoard.
    This function ONLY extracts step/max_steps for progress display.
    If this fails, progress will be inferred from TensorBoard step numbers.
    
    Returns:
        Optional[Tuple[int, int]]: (current_step, max_steps) or None if not found
    """
    # Wrap EVERYTHING in try-except to guarantee no crashes
    try:
        if not line or not isinstance(line, str):
            return None
        
        import re
        
        # Method 1: Try JSON parsing (most reliable for USF BIOS logs)
        if '{' in line:
            try:
                start = line.find('{')
                end = line.rfind('}') + 1
                if start >= 0 and end > start:
                    json_str = line[start:end].replace("'", '"')
                    # Handle edge cases in JSON
                    json_str = json_str.replace('True', 'true').replace('False', 'false').replace('None', 'null')
                    data = json.loads(json_str)
                    
                    # Format: 'global_step/max_steps': '100/1000'
                    if 'global_step/max_steps' in data:
                        val = str(data['global_step/max_steps'])
                        if '/' in val:
                            parts = val.split('/')
                            if len(parts) == 2:
                                step, max_s = int(parts[0]), int(parts[1])
                                if step >= 0 and max_s > 0:
                                    return step, max_s
                    
                    # Format: Separate keys
                    if 'global_step' in data:
                        step = int(data['global_step'])
                        max_s = int(data.get('max_steps', data.get('total_steps', 0)))
                        if step >= 0 and max_s > 0:
                            return step, max_s
            except Exception:
                pass  # JSON parsing failed, try regex
        
        # Method 2: Simple regex patterns (fallback)
        # Only use simple, well-tested patterns
        simple_patterns = [
            r"(\d+)/(\d+)",  # Any number/number pattern
        ]
        
        for pattern in simple_patterns:
            try:
                matches = re.findall(pattern, line)
                for match in matches:
                    step, max_s = int(match[0]), int(match[1])
                    # Validate: step should be <= max_steps, max_steps should be reasonable
                    if 0 <= step <= max_s and max_s > 0 and max_s < 10_000_000:
                        return step, max_s
            except Exception:
                continue
        
        return None
        
    except Exception:
        # Ultimate fallback - NEVER crash
        return None


def get_max_steps_from_tensorboard(tb_metrics: dict) -> int:
    """Try to infer max_steps from TensorBoard metrics history."""
    if not tb_metrics:
        return 0
    
    # Check for metrics with history to find max step
    max_step = 0
    for key, data in tb_metrics.items():
        if isinstance(data, dict) and 'history' in data:
            for point in data['history']:
                if 'step' in point:
                    max_step = max(max_step, point['step'])
        elif isinstance(data, dict) and 'step' in data:
            max_step = max(max_step, data['step'])
    
    return max_step


def update_metrics_from_log(line: str):
    """Update training progress from log line - FAIL-SAFE.
    
    CRITICAL DESIGN DECISION:
    ========================
    ALL training metrics (loss, learning_rate, rewards, accuracy, etc.) 
    come EXCLUSIVELY from TensorBoard. This function does NOT parse any
    training metrics from logs.
    
    This function ONLY does two things:
    1. Extract step/max_steps for progress bar display (optional, has TensorBoard fallback)
    2. Update status messages for user feedback (cosmetic only)
    
    If this function fails completely, the system still works because:
    - TensorBoard provides all metrics
    - Progress can be inferred from TensorBoard step numbers
    - Status messages are just cosmetic
    
    This design ensures 100% reliability for all training types:
    - SFT, LoRA, QLoRA, AdaLoRA, DoRA, Full fine-tuning
    - DPO, KTO, SimPO, ORPO, CPO (Offline RLHF)
    - GRPO, PPO, GKD (Online RLHF)
    - Pretraining / Megatron
    - Reward Model training
    """
    # Wrap EVERYTHING in try-except - this function must NEVER crash
    try:
        global _training_state
        
        if not line or not isinstance(line, str):
            return
        
        # Step extraction (optional - has TensorBoard fallback)
        try:
            step_info = extract_step_from_log(line)
            if step_info:
                _training_state.metrics.global_step = step_info[0]
                _training_state.metrics.max_steps = step_info[1]
        except Exception:
            pass  # Step extraction failed - TensorBoard fallback will be used
        
        # Status message detection (cosmetic only - safe string operations)
        try:
            line_lower = line.lower()
            if 'loading checkpoint' in line_lower or 'loading model' in line_lower:
                _training_state.status_message = "🔄 Loading model..."
            elif 'downloading' in line_lower and 'model' in line_lower:
                _training_state.status_message = "⬇️ Downloading model..."
            elif 'downloading' in line_lower and 'dataset' in line_lower:
                _training_state.status_message = "⬇️ Downloading dataset..."
            elif 'processing' in line_lower or 'filtered' in line_lower:
                _training_state.status_message = "📊 Processing dataset..."
            elif 'train:' in line_lower and '%' in line:
                _training_state.status_message = "🏃 Training..."
            elif 'saving' in line_lower and ('model' in line_lower or 'checkpoint' in line_lower):
                _training_state.status_message = "💾 Saving checkpoint..."
            elif 'rollout' in line_lower or 'generating' in line_lower:
                _training_state.status_message = "🎲 Generating rollouts..."
            elif 'reward' in line_lower and 'comput' in line_lower:
                _training_state.status_message = "🎯 Computing rewards..."
            elif 'evaluat' in line_lower:
                _training_state.status_message = "📊 Evaluating..."
        except Exception:
            pass  # Status message update failed - not critical
            
    except Exception:
        # Ultimate safety net - NEVER crash
        pass


def get_defaults_for_train_type(train_type: str) -> dict:
    """Get default hyperparameters based on training type"""
    if train_type == 'lora':
        return LORA_DEFAULTS.copy()
    elif train_type == 'qlora':
        return QLORA_DEFAULTS.copy()
    elif train_type == 'adalora':
        return ADALORA_DEFAULTS.copy()
    elif train_type == 'dora':
        return DORA_DEFAULTS.copy()
    else:  # full
        return FULL_FT_DEFAULTS.copy()


def generate_training_command(
    model_id: str,
    use_hf: bool,
    train_type: str,
    dataset_path: str,
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
    warmup_ratio: float,
    weight_decay: float,
    max_grad_norm: float,
    gradient_checkpointing: bool,
    logging_steps: int,
    save_steps: int,
    eval_steps: int,
    split_ratio: float,
    system_prompt: str,
    quant_bits: int,
    torch_dtype: str,
) -> str:
    """Generate the training command"""
    
    cmd_parts = ['usf sft']
    
    # Model configuration
    cmd_parts.append(f'    --model "{model_id}"')
    if use_hf:
        cmd_parts.append('    --use_hf true')
    
    # Dataset
    cmd_parts.append(f'    --dataset "{dataset_path}"')
    
    # Training type
    cmd_parts.append(f'    --train_type {train_type}')
    cmd_parts.append(f'    --torch_dtype {torch_dtype}')
    
    # Training parameters
    cmd_parts.append(f'    --num_train_epochs {num_epochs}')
    cmd_parts.append(f'    --per_device_train_batch_size {batch_size}')
    cmd_parts.append(f'    --gradient_accumulation_steps {gradient_accumulation}')
    cmd_parts.append(f'    --learning_rate {learning_rate}')
    cmd_parts.append(f'    --max_length {max_length}')
    cmd_parts.append(f'    --output_dir "{output_dir}"')
    
    # LoRA parameters (if applicable)
    if train_type in ['lora', 'qlora', 'adalora', 'dora']:
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
                if i >= 5:
                    break
                try:
                    data = json.loads(line.strip())
                    
                    if 'messages' not in data:
                        errors.append(f"Line {i+1}: Missing 'messages' field")
                        continue
                    
                    messages = data['messages']
                    if not isinstance(messages, list) or len(messages) == 0:
                        errors.append(f"Line {i+1}: 'messages' must be non-empty array")
                        continue
                    
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
        )
        
        def read_logs():
            if _training_state.process:
                for line in iter(_training_state.process.stdout.readline, ''):
                    if line:
                        line_stripped = line.strip()
                        _training_state.logs.append(line_stripped)
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
    
    important_keywords = [
        '[INFO:usf]', 'Loading', 'Downloading', 'Dataset', 'Train:', 
        'Saving', 'Error', 'Warning', 'SUCCESS', 'FAILED', 'model_info',
        'lora_config', 'trainable', '%|'
    ]
    
    filtered = []
    for log in _training_state.logs[-200:]:
        if log.startswith('{') and 'loss' in log:
            continue
        if any(kw in log for kw in important_keywords):
            filtered.append(log)
    
    return '\n'.join(filtered[-50:]) if filtered else "Processing..."


def get_tb_value(tb_metrics: dict, *keys) -> Optional[float]:
    """Get value from TensorBoard metrics, trying multiple key names."""
    for key in keys:
        if key in tb_metrics and 'value' in tb_metrics[key]:
            return tb_metrics[key]['value']
    return None


def get_metrics_display(output_dir: str = "") -> str:
    """Get formatted metrics display for dashboard.
    
    ALL training metrics come from TensorBoard.
    Progress metrics (percentage, elapsed, remaining, speed) are calculated.
    """
    m = _training_state.metrics
    
    # Get ALL metrics from TensorBoard (the ONLY source)
    tb_metrics = {}
    if output_dir and os.path.exists(output_dir):
        tb_metrics = read_tensorboard_metrics(output_dir)
    
    # Check if we have any data to display
    if not _training_state.is_running and not tb_metrics:
        return ""
    
    lines = []
    
    # Calculate progress metrics programmatically
    step = m.global_step
    max_steps = m.max_steps
    percentage = (step / max_steps * 100) if max_steps > 0 else 0
    
    # Calculate time metrics
    elapsed_seconds = 0
    elapsed_str = "00:00:00"
    remaining_str = "--:--:--"
    train_speed = 0.0
    
    if _training_state.start_time:
        import time
        elapsed_seconds = time.time() - _training_state.start_time
        elapsed_str = format_time(elapsed_seconds)
        if percentage > 0:
            remaining_seconds = (elapsed_seconds / percentage * 100) - elapsed_seconds
            remaining_str = format_time(remaining_seconds)
        if elapsed_seconds > 0 and step > 0:
            train_speed = step / elapsed_seconds
    
    # Progress section
    bar_length = 30
    filled = int(bar_length * percentage / 100) if percentage > 0 else 0
    bar = "█" * filled + "░" * (bar_length - filled)
    lines.append(f"**Progress:** [{bar}] {percentage:.1f}%")
    lines.append(f"**Step:** {step} / {max_steps}")
    lines.append("")
    
    # Time section (calculated, not from TensorBoard)
    lines.append(f"⏱️ **Elapsed:** {elapsed_str}")
    lines.append(f"⏳ **Remaining:** {remaining_str}")
    lines.append("")
    
    # Detect and show training type from TensorBoard metrics
    training_type = detect_training_type(tb_metrics) if tb_metrics else m.training_type
    m.training_type = training_type
    type_labels = {
        'sft': '🎓 SFT', 'dpo': '⚖️ DPO', 'grpo': '🎲 GRPO', 
        'ppo': '🎯 PPO', 'pt': '📚 Pretrain', 'rm': '🏆 Reward Model'
    }
    lines.append(f"**Type:** {type_labels.get(training_type, training_type.upper())}")
    lines.append("")
    
    # ===== ALL METRICS FROM TENSORBOARD =====
    # Track displayed keys to show remaining metrics dynamically
    displayed_keys = set()
    
    def add_metric(label: str, icon: str, *keys, fmt: str = ".4f", multiplier: float = 1.0, is_percent: bool = False):
        """Helper to add a metric and track displayed keys."""
        val = get_tb_value(tb_metrics, *keys)
        if val is not None:
            displayed_keys.update(keys)
            val = val * multiplier
            if is_percent:
                lines.append(f"{icon} **{label}:** {val:.2%}")
            elif fmt == ".2e":
                lines.append(f"{icon} **{label}:** {val:.2e}")
            elif fmt == ".1f":
                lines.append(f"{icon} **{label}:** {val:.1f}")
            elif fmt == ".2f":
                lines.append(f"{icon} **{label}:** {val:.2f}")
            elif fmt == ".6f":
                lines.append(f"{icon} **{label}:** {val:.6f}")
            else:
                lines.append(f"{icon} **{label}:** {val:.4f}")
            return True
        return False
    
    # === Core Training Metrics ===
    add_metric("Loss", "📉", 'train/loss', 'loss', 'lm_loss')
    add_metric("Learning Rate", "📈", 'train/learning_rate', 'learning_rate', 'learning-rate', fmt=".2e")
    add_metric("Grad Norm", "📊", 'train/grad_norm', 'grad_norm', 'grad-norm')
    # Accuracy metrics (token-level or sequence-level)
    add_metric("Token Accuracy", "🎯", 'train/token_acc', 'token_acc', 'train/token_accuracy', is_percent=True)
    add_metric("Seq Accuracy", "🎯", 'train/seq_acc', 'seq_acc', is_percent=True)
    add_metric("Accuracy", "🎯", 'train/acc', 'acc', 'accuracy', is_percent=True)
    add_metric("Epoch", "🔄", 'train/epoch', 'epoch', fmt=".2f")
    # MFU (Model FLOPs Utilization)
    add_metric("MFU", "⚡", 'MFU', 'train/MFU', fmt=".4f")
    
    # === Eval Metrics ===
    eval_loss = get_tb_value(tb_metrics, 'eval/loss', 'eval_loss')
    if eval_loss is not None:
        displayed_keys.update(['eval/loss', 'eval_loss'])
        lines.append(f"📋 **Eval Loss:** {eval_loss:.4f}")
    # Eval accuracy
    eval_token_acc = get_tb_value(tb_metrics, 'eval/token_acc', 'eval_token_acc')
    eval_seq_acc = get_tb_value(tb_metrics, 'eval/seq_acc', 'eval_seq_acc')
    if eval_token_acc is not None:
        displayed_keys.update(['eval/token_acc', 'eval_token_acc'])
        lines.append(f"📋 **Eval Token Acc:** {eval_token_acc:.2%}")
    if eval_seq_acc is not None:
        displayed_keys.update(['eval/seq_acc', 'eval_seq_acc'])
        lines.append(f"📋 **Eval Seq Acc:** {eval_seq_acc:.2%}")
    
    # === DPO/Offline RLHF Metrics ===
    chosen = get_tb_value(tb_metrics, 'rewards/chosen', 'train/rewards/chosen', 'chosen_rewards')
    rejected = get_tb_value(tb_metrics, 'rewards/rejected', 'train/rewards/rejected', 'rejected_rewards')
    accuracy = get_tb_value(tb_metrics, 'rewards/accuracies', 'train/rewards/accuracies')
    margins = get_tb_value(tb_metrics, 'rewards/margins', 'train/rewards/margins')
    
    if chosen is not None or rejected is not None or accuracy is not None:
        lines.append("")
        lines.append("**DPO/RLHF Metrics:**")
        displayed_keys.update(['rewards/chosen', 'train/rewards/chosen', 'chosen_rewards',
                              'rewards/rejected', 'train/rewards/rejected', 'rejected_rewards',
                              'rewards/accuracies', 'train/rewards/accuracies',
                              'rewards/margins', 'train/rewards/margins'])
        if chosen is not None:
            lines.append(f"  ✅ **Chosen Rewards:** {chosen:.4f}")
        if rejected is not None:
            lines.append(f"  ❌ **Rejected Rewards:** {rejected:.4f}")
        if margins is not None:
            lines.append(f"  📏 **Reward Margin:** {margins:.4f}")
        elif chosen is not None and rejected is not None:
            lines.append(f"  📏 **Margin:** {chosen - rejected:.4f}")
        if accuracy is not None:
            lines.append(f"  🎯 **Accuracy:** {accuracy:.2%}")
        # Log probs
        chosen_logps = get_tb_value(tb_metrics, 'logps/chosen', 'train/logps/chosen')
        rejected_logps = get_tb_value(tb_metrics, 'logps/rejected', 'train/logps/rejected')
        if chosen_logps is not None:
            displayed_keys.update(['logps/chosen', 'train/logps/chosen'])
            lines.append(f"  📊 **Chosen LogPs:** {chosen_logps:.4f}")
        if rejected_logps is not None:
            displayed_keys.update(['logps/rejected', 'train/logps/rejected'])
            lines.append(f"  📊 **Rejected LogPs:** {rejected_logps:.4f}")
        # NLL Loss
        nll = get_tb_value(tb_metrics, 'nll_loss', 'train/nll_loss')
        if nll is not None:
            displayed_keys.update(['nll_loss', 'train/nll_loss'])
            lines.append(f"  📉 **NLL Loss:** {nll:.4f}")
        # Mean logits (DPO)
        mean_chosen_logits = get_tb_value(tb_metrics, 'mean_chosen_logits', 'train/mean_chosen_logits')
        mean_rejected_logits = get_tb_value(tb_metrics, 'mean_rejected_logits', 'train/mean_rejected_logits')
        if mean_chosen_logits is not None:
            displayed_keys.update(['mean_chosen_logits', 'train/mean_chosen_logits'])
            lines.append(f"  📊 **Mean Chosen Logits:** {mean_chosen_logits:.4f}")
        if mean_rejected_logits is not None:
            displayed_keys.update(['mean_rejected_logits', 'train/mean_rejected_logits'])
            lines.append(f"  📊 **Mean Rejected Logits:** {mean_rejected_logits:.4f}")
    
    # === GRPO/PPO/Online RL Metrics ===
    reward = get_tb_value(tb_metrics, 'reward', 'train/reward', 'objective/rlhf_reward_mean')
    kl = get_tb_value(tb_metrics, 'kl', 'train/kl', 'objective/kl')
    policy_loss = get_tb_value(tb_metrics, 'policy_loss', 'train/policy_loss', 'ppo/loss/policy')
    
    if reward is not None or kl is not None or policy_loss is not None:
        lines.append("")
        lines.append("**Online RL Metrics:**")
        displayed_keys.update(['reward', 'train/reward', 'objective/rlhf_reward_mean',
                              'kl', 'train/kl', 'objective/kl',
                              'policy_loss', 'train/policy_loss', 'ppo/loss/policy'])
        if reward is not None:
            lines.append(f"  🎁 **Reward:** {reward:.4f}")
        reward_std = get_tb_value(tb_metrics, 'reward_std', 'train/reward_std')
        if reward_std is not None:
            displayed_keys.update(['reward_std', 'train/reward_std'])
            lines.append(f"  📊 **Reward Std:** {reward_std:.4f}")
        frac_zero = get_tb_value(tb_metrics, 'frac_reward_zero_std')
        if frac_zero is not None:
            displayed_keys.add('frac_reward_zero_std')
            lines.append(f"  📉 **Frac Zero Std:** {frac_zero:.2%}")
        if kl is not None:
            lines.append(f"  🔄 **KL Divergence:** {kl:.6f}")
        # Entropy metrics
        entropy_mean = get_tb_value(tb_metrics, 'entropy/mean', 'entropy', 'train/entropy')
        if entropy_mean is not None:
            displayed_keys.update(['entropy/mean', 'entropy', 'train/entropy'])
            lines.append(f"  🎲 **Entropy:** {entropy_mean:.4f}")
        entropy_min = get_tb_value(tb_metrics, 'entropy/min')
        entropy_max = get_tb_value(tb_metrics, 'entropy/max')
        if entropy_min is not None and entropy_max is not None:
            displayed_keys.update(['entropy/min', 'entropy/max'])
            lines.append(f"  📊 **Entropy Range:** [{entropy_min:.4f}, {entropy_max:.4f}]")
        entropy_threshold = get_tb_value(tb_metrics, 'entropy/threshold')
        if entropy_threshold is not None:
            displayed_keys.add('entropy/threshold')
            lines.append(f"  📊 **Entropy Threshold:** {entropy_threshold:.4f}")
        # Policy/Value losses
        if policy_loss is not None:
            lines.append(f"  📉 **Policy Loss:** {policy_loss:.4f}")
        value_loss = get_tb_value(tb_metrics, 'value_loss', 'train/value_loss', 'ppo/loss/value')
        if value_loss is not None:
            displayed_keys.update(['value_loss', 'train/value_loss', 'ppo/loss/value'])
            lines.append(f"  📈 **Value Loss:** {value_loss:.4f}")
        # Clip ratio metrics (GRPO/PPO)
        clip_region = get_tb_value(tb_metrics, 'clip_ratio/region_mean')
        clip_frac = get_tb_value(tb_metrics, 'objective/clip_fraction', 'clip_fraction')
        cispo_clip = get_tb_value(tb_metrics, 'cispo_clip_ratio')
        clip_low_min = get_tb_value(tb_metrics, 'clip_ratio/low_min')
        clip_high_max = get_tb_value(tb_metrics, 'clip_ratio/high_max')
        if clip_region is not None:
            displayed_keys.add('clip_ratio/region_mean')
            lines.append(f"  ✂️ **Clip Ratio:** {clip_region:.4f}")
        if clip_frac is not None:
            displayed_keys.update(['objective/clip_fraction', 'clip_fraction'])
            lines.append(f"  ✂️ **Clip Fraction:** {clip_frac:.4f}")
        if cispo_clip is not None:
            displayed_keys.add('cispo_clip_ratio')
            lines.append(f"  ✂️ **CISPO Clip Ratio:** {cispo_clip:.4f}")
        if clip_low_min is not None and clip_high_max is not None:
            displayed_keys.update(['clip_ratio/low_min', 'clip_ratio/high_max', 
                                  'clip_ratio/low_mean', 'clip_ratio/high_mean'])
            lines.append(f"  📊 **Clip Range:** [{clip_low_min:.4f}, {clip_high_max:.4f}]")
        # Completion metrics (GRPO)
        comp_len = get_tb_value(tb_metrics, 'completions/mean_length')
        comp_min = get_tb_value(tb_metrics, 'completions/min_length')
        comp_max = get_tb_value(tb_metrics, 'completions/max_length')
        comp_clip = get_tb_value(tb_metrics, 'completions/clipped_ratio')
        if comp_len is not None:
            displayed_keys.add('completions/mean_length')
            lines.append(f"  📝 **Completion Length:** {comp_len:.1f}")
        if comp_min is not None and comp_max is not None:
            displayed_keys.update(['completions/min_length', 'completions/max_length'])
            lines.append(f"  📏 **Length Range:** [{comp_min:.0f}, {comp_max:.0f}]")
        if comp_clip is not None:
            displayed_keys.add('completions/clipped_ratio')
            lines.append(f"  ✂️ **Clipped Ratio:** {comp_clip:.2%}")
        num_turns = get_tb_value(tb_metrics, 'num_turns')
        if num_turns is not None:
            displayed_keys.add('num_turns')
            lines.append(f"  🔁 **Num Turns:** {num_turns:.1f}")
        # Rollout correction metrics (GRPO with vLLM)
        for rc_key in ['rollout_correction/kl_diff', 'rollout_correction/logp_diff']:
            rc_val = get_tb_value(tb_metrics, rc_key)
            if rc_val is not None:
                displayed_keys.add(rc_key)
                name = rc_key.split('/')[-1].replace('_', ' ').title()
                lines.append(f"  🔄 **{name}:** {rc_val:.6f}")
    
    # === Pretraining/Megatron Metrics ===
    ppl = get_tb_value(tb_metrics, 'perplexity', 'train/perplexity')
    throughput = get_tb_value(tb_metrics, 'throughput', 'train/throughput')
    iter_time = get_tb_value(tb_metrics, 'iteration-time')
    loss_scale = get_tb_value(tb_metrics, 'loss-scale')
    
    if ppl is not None or throughput is not None or iter_time is not None:
        lines.append("")
        lines.append("**Pretrain/Megatron Metrics:**")
        displayed_keys.update(['perplexity', 'train/perplexity', 'throughput', 'train/throughput',
                              'iteration-time', 'loss-scale'])
        if ppl is not None:
            lines.append(f"  📐 **Perplexity:** {ppl:.2f}")
        if throughput is not None:
            lines.append(f"  🚀 **Throughput:** {throughput:.1f} TFLOP/s")
        if iter_time is not None:
            lines.append(f"  ⏱️ **Iter Time:** {iter_time*1000:.1f} ms")
        if loss_scale is not None:
            lines.append(f"  📊 **Loss Scale:** {loss_scale:.0f}")
        params_norm = get_tb_value(tb_metrics, 'params-norm')
        if params_norm is not None:
            displayed_keys.add('params-norm')
            lines.append(f"  📏 **Params Norm:** {params_norm:.4f}")
        batch_size = get_tb_value(tb_metrics, 'batch-size')
        if batch_size is not None:
            displayed_keys.add('batch-size')
            lines.append(f"  📦 **Batch Size:** {int(batch_size)}")
        # Additional Megatron metrics
        num_zeros = get_tb_value(tb_metrics, 'num-zeros')
        if num_zeros is not None:
            displayed_keys.add('num-zeros')
            lines.append(f"  🔢 **Num Zeros:** {int(num_zeros)}")
        world_size = get_tb_value(tb_metrics, 'world-size')
        if world_size is not None:
            displayed_keys.add('world-size')
            lines.append(f"  🌐 **World Size:** {int(world_size)}")
    
    # === MoE (Mixture of Experts) Metrics ===
    aux_loss = get_tb_value(tb_metrics, 'aux_loss', 'train/aux_loss')
    load_bal_loss = get_tb_value(tb_metrics, 'load_balancing_loss', 'train/load_balancing_loss')
    z_loss = get_tb_value(tb_metrics, 'z_loss', 'train/z_loss')
    
    if aux_loss is not None or load_bal_loss is not None or z_loss is not None:
        lines.append("")
        lines.append("**MoE Metrics:**")
        displayed_keys.update(['aux_loss', 'train/aux_loss', 'load_balancing_loss', 
                              'train/load_balancing_loss', 'z_loss', 'train/z_loss'])
        if aux_loss is not None:
            lines.append(f"  🔀 **Aux Loss:** {aux_loss:.6f}")
        if load_bal_loss is not None:
            lines.append(f"  ⚖️ **Load Balance Loss:** {load_bal_loss:.6f}")
        if z_loss is not None:
            lines.append(f"  📊 **Z Loss:** {z_loss:.6f}")
    
    # === Reward Model Metrics ===
    center_rewards_loss = get_tb_value(tb_metrics, 'center_rewards_loss', 'train/center_rewards_loss')
    if center_rewards_loss is not None:
        lines.append("")
        lines.append("**Reward Model Metrics:**")
        displayed_keys.update(['center_rewards_loss', 'train/center_rewards_loss'])
        lines.append(f"  🎯 **Center Rewards Loss:** {center_rewards_loss:.6f}")
    
    # === System/Memory Metrics ===
    lines.append("")
    mem_reserved = get_tb_value(tb_metrics, 'mem-reserved-bytes')
    mem_allocated = get_tb_value(tb_metrics, 'mem-allocated-bytes')
    mem_peak = get_tb_value(tb_metrics, 'mem-max-allocated-bytes')
    
    if mem_reserved is not None or mem_allocated is not None:
        displayed_keys.update(['mem-reserved-bytes', 'mem-allocated-bytes', 'mem-max-allocated-bytes'])
        if mem_reserved is not None:
            lines.append(f"💾 **GPU Reserved:** {mem_reserved / (1024**3):.1f} GiB")
        if mem_allocated is not None:
            lines.append(f"💾 **GPU Allocated:** {mem_allocated / (1024**3):.1f} GiB")
        if mem_peak is not None:
            lines.append(f"📈 **Peak Memory:** {mem_peak / (1024**3):.1f} GiB")
    
    if train_speed > 0:
        lines.append(f"⚡ **Speed:** {train_speed:.3f} iter/s")
    
    # === Dynamic: Show ALL remaining TensorBoard metrics ===
    remaining_metrics = {}
    skip_patterns = ['vs samples', 'histogram', 'image']  # Skip non-scalar or duplicate metrics
    
    for key, data in tb_metrics.items():
        if key in displayed_keys:
            continue
        if any(p in key.lower() for p in skip_patterns):
            continue
        if isinstance(data, dict) and 'value' in data:
            remaining_metrics[key] = data['value']
    
    if remaining_metrics:
        lines.append("")
        lines.append("**Other Metrics:**")
        # Sort and show up to 10 remaining metrics
        for key, val in sorted(remaining_metrics.items())[:10]:
            # Clean up key name for display
            display_name = key.replace('train/', '').replace('/', ' ').replace('_', ' ').title()
            if isinstance(val, float):
                if abs(val) < 0.01 or abs(val) > 1000:
                    lines.append(f"  • **{display_name}:** {val:.2e}")
                else:
                    lines.append(f"  • **{display_name}:** {val:.4f}")
            else:
                lines.append(f"  • **{display_name}:** {val}")
    
    # TensorBoard link
    if _tensorboard_process is not None:
        lines.append("")
        lines.append(f"📊 **TensorBoard:** [Open](http://localhost:{_tensorboard_port})")
    
    return '\n'.join(lines)


def get_training_status() -> str:
    """Get training status"""
    return _training_state.status_message


def build_train_ui(
    lang: Literal['en', 'zh'] = 'en',
) -> gr.Blocks:
    """Build USF BIOS Training WebUI - Universal LLM Fine-tuning Interface"""
    
    with gr.Blocks(
        title="USF BIOS Training Studio",
        theme=gr.themes.Soft(),
        css="""
        .header-info { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                      color: white; padding: 20px; border-radius: 12px; margin-bottom: 20px; }
        .section-header { background: #f8fafc; padding: 15px; border-radius: 8px; margin: 10px 0; }
        .research-note { background: #e8f5e9; padding: 10px; border-radius: 6px; font-size: 12px; margin: 5px 0; }
        """
    ) as demo:
        
        # Header
        gr.HTML("""
        <div class="header-info">
            <h1 style="margin:0; font-size: 28px;">🚀 USF BIOS Training Studio</h1>
            <p style="margin: 10px 0 0 0; opacity: 0.9;">Universal fine-tuning interface for any LLM</p>
            <p style="margin: 5px 0 0 0; font-size: 14px; opacity: 0.8;">Supports LoRA, QLoRA, AdaLoRA, DoRA, Full Fine-tuning | Research-based defaults</p>
        </div>
        """)
        
        with gr.Tabs():
            # Tab 1: Model & Training Configuration
            with gr.TabItem("⚙️ Configuration"):
                
                # Model Selection
                gr.Markdown("### 1️⃣ Select Model")
                with gr.Row():
                    with gr.Column(scale=2):
                        model_id = gr.Textbox(
                            label="Model ID / Path",
                            placeholder="HuggingFace model ID (e.g., Qwen/Qwen2.5-7B-Instruct) or local path",
                            info="Enter any HuggingFace model ID or absolute path to local model"
                        )
                    with gr.Column(scale=1):
                        use_hf = gr.Checkbox(label="Load from HuggingFace Hub", value=True,
                                            info="Uncheck if using local model path")
                
                # Training Type Selection
                gr.Markdown("### 2️⃣ Training Type")
                gr.HTML('<div class="research-note">💡 <b>Research Finding:</b> LoRA with all-linear modules achieves near full fine-tuning quality at ~10% memory cost</div>')
                train_type = gr.Radio(
                    choices=[
                        ("LoRA (Recommended)", "lora"),
                        ("QLoRA (4-bit - Lowest VRAM)", "qlora"),
                        ("AdaLoRA (Adaptive Rank)", "adalora"),
                        ("DoRA (Weight Decomposed)", "dora"),
                        ("Full Fine-tuning (High VRAM)", "full"),
                    ],
                    value="lora",
                    label="Training Method",
                )
                
                # Dataset Configuration
                gr.Markdown("### 3️⃣ Dataset")
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
                dataset_preview = gr.Code(label="Dataset Preview", language="json", lines=6)
                
                # Hyperparameters
                gr.Markdown("### 4️⃣ Hyperparameters")
                gr.HTML('<div class="research-note">💡 <b>Research:</b> LoRA LR should be ~10x Full FT LR. Alpha = 2×Rank is optimal.</div>')
                
                with gr.Row():
                    with gr.Column():
                        num_epochs = gr.Number(label="Epochs", value=2, minimum=1, maximum=100)
                        batch_size = gr.Number(label="Batch Size", value=1, minimum=1, maximum=8, precision=0)
                        gradient_accumulation = gr.Number(label="Gradient Accumulation", value=16, minimum=1, maximum=128, precision=0)
                        learning_rate = gr.Number(label="Learning Rate", value=2e-4, minimum=1e-7, maximum=1e-2)
                    
                    with gr.Column():
                        max_length = gr.Number(label="Max Sequence Length", value=4096, minimum=512, maximum=131072, precision=0)
                        warmup_ratio = gr.Number(label="Warmup Ratio", value=0.03, minimum=0, maximum=0.5)
                        weight_decay = gr.Number(label="Weight Decay", value=0.1, minimum=0, maximum=1)
                        max_grad_norm = gr.Number(label="Max Gradient Norm", value=1.0, minimum=0, maximum=10)
                
                # LoRA-specific parameters
                with gr.Accordion("🔧 LoRA Parameters", open=True) as lora_accordion:
                    gr.HTML('<div class="research-note">💡 <b>Best Practice:</b> Use all-linear target modules for best results. Rank 64-128 balances quality and efficiency.</div>')
                    with gr.Row():
                        lora_rank = gr.Number(label="LoRA Rank (r)", value=64, minimum=1, maximum=512, precision=0,
                                             info="Higher = more capacity, more VRAM. 64-128 recommended.")
                        lora_alpha = gr.Number(label="LoRA Alpha (α)", value=128, minimum=1, maximum=1024, precision=0,
                                              info="Scaling factor. α = 2×r is optimal.")
                        lora_dropout = gr.Number(label="LoRA Dropout", value=0.05, minimum=0, maximum=0.5,
                                                info="Regularization. 0.05 recommended.")
                        target_modules = gr.Textbox(label="Target Modules", value="all-linear",
                                                   info="'all-linear' applies LoRA to all layers (best results)")
                
                # QLoRA-specific
                with gr.Accordion("📦 Quantization (QLoRA)", open=False) as quant_accordion:
                    quant_bits = gr.Radio(choices=[4, 8], value=4, label="Quantization Bits",
                                         info="4-bit NF4: Lower VRAM | 8-bit: Better quality")
                
                # Advanced settings
                with gr.Accordion("⚡ Advanced Settings", open=False):
                    with gr.Row():
                        torch_dtype = gr.Dropdown(choices=['bfloat16', 'float16', 'float32'], value='bfloat16', label="Torch Dtype")
                        gradient_checkpointing = gr.Checkbox(label="Gradient Checkpointing", value=True,
                                                            info="Reduces VRAM at cost of ~20% speed")
                    with gr.Row():
                        logging_steps = gr.Number(label="Logging Steps", value=10, precision=0)
                        save_steps = gr.Number(label="Save Steps", value=500, precision=0)
                    with gr.Row():
                        eval_steps = gr.Number(label="Eval Steps (0=disable)", value=0, precision=0)
                        split_ratio = gr.Number(label="Train/Val Split", value=0.0, minimum=0, maximum=0.3)
                    
                    system_prompt = gr.Textbox(label="Default System Prompt (Optional)", lines=2,
                                              placeholder="You are a helpful AI assistant.")
                    output_dir = gr.Textbox(label="Output Directory", value="output/sft-training")
            
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
                
                command_output = gr.Code(label="Training Command", language="shell", lines=15)
                
                status_output = gr.Textbox(label="Status", value="Ready", interactive=False)
                
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
            
            # Tab 4: Training Charts
            with gr.TabItem("📈 Training Charts"):
                gr.Markdown("### 📊 Training Metrics")
                
                with gr.Row():
                    chart_refresh_btn = gr.Button("🔄 Refresh Charts", variant="secondary")
                    tb_start_btn = gr.Button("▶️ Start TensorBoard", variant="primary")
                    tb_stop_btn = gr.Button("⏹️ Stop TensorBoard", variant="secondary")
                
                tb_status = gr.Textbox(label="TensorBoard Status", value="Not running", interactive=False)
                
                tb_charts_display = gr.HTML(
                    value="<p style='color: #666;'>Training charts will appear here once training starts logging.</p>"
                )
                
                gr.Markdown("---")
                gr.Markdown("### 📋 Training Summary")
                training_summary = gr.HTML(
                    value="<p style='color: #666;'>Summary will be available after training completes.</p>"
                )
            
            # Tab 5: Export to Storage
            with gr.TabItem("💾 Export Model"):
                gr.Markdown("### Export Trained Model to Storage")
                
                with gr.Row():
                    export_output_dir = gr.Textbox(
                        label="Output Directory (Source)",
                        value="output/sft-training",
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
                        placeholder="/mnt/storage/models/my-model or s3://bucket/path",
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
            is_lora = train_type_val in ['lora', 'qlora', 'adalora', 'dora']
            is_qlora = train_type_val == 'qlora'
            
            return (
                defaults.get('learning_rate', 2e-4),
                defaults.get('gradient_accumulation_steps', 16),
                defaults.get('lora_rank', 64) if is_lora else 64,
                defaults.get('lora_alpha', 128) if is_lora else 128,
                gr.update(visible=is_lora),
                gr.update(visible=is_qlora),
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
                model_id, use_hf, train_type, dataset_path, num_epochs, batch_size, gradient_accumulation,
                learning_rate, max_length, output_dir, lora_rank, lora_alpha, lora_dropout,
                target_modules, warmup_ratio, weight_decay, max_grad_norm, gradient_checkpointing,
                logging_steps, save_steps, eval_steps, split_ratio, system_prompt, quant_bits, torch_dtype
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
        
        # Refresh all
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
        demo.load(lambda out_dir: get_tensorboard_charts(out_dir), inputs=[output_dir], outputs=[tb_charts_display], every=5)
        
        # TensorBoard start/stop handlers
        def on_tb_start(out_dir):
            if not out_dir:
                return "❌ Set output directory first"
            success, msg, port = start_tensorboard(out_dir)
            if success:
                return f"✅ {msg} - Open: http://localhost:{port}"
            return f"❌ {msg}"
        
        def on_tb_stop():
            success, msg = stop_tensorboard()
            return f"{'✅' if success else '❌'} {msg}"
        
        tb_start_btn.click(on_tb_start, inputs=[output_dir], outputs=[tb_status])
        tb_stop_btn.click(on_tb_stop, outputs=[tb_status])
        
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


def train_ui_main():
    """Main entry point for Training WebUI"""
    from usf_bios.system_guard import guard_with_integrity
    guard_with_integrity()
    import argparse
    
    parser = argparse.ArgumentParser(description='USF BIOS Training WebUI')
    parser.add_argument('--server_name', type=str, default='0.0.0.0')
    parser.add_argument('--server_port', type=int, default=7861)
    parser.add_argument('--share', action='store_true')
    parser.add_argument('--lang', type=str, default='en', choices=['en', 'zh'])
    
    args = parser.parse_args()
    
    demo = build_train_ui(lang=args.lang)
    demo.queue().launch(
        server_name=args.server_name,
        server_port=args.server_port,
        share=args.share
    )


if __name__ == '__main__':
    train_ui_main()
