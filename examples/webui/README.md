# USF BIOS Training WebUI

A user-friendly Gradio-based interface for fine-tuning Large Language Models (LLMs).

## Features

- **Easy Model Loading**: Load models from HuggingFace Hub or local paths
- **Flexible Dataset Support**: Load datasets from HuggingFace or local JSON/JSONL files
- **Multiple Training Methods**: LoRA, QLoRA, Full Fine-tuning, AdaLoRA, DoRA
- **Real-time Monitoring**: Live training logs and status updates
- **Command Generation**: Generate CLI commands for reproducibility
- **Multi-language UI**: English and Chinese support

## Quick Start

### Launch the WebUI

```bash
# From the project root
cd /workspace/usf-usf-bios-prod

# Launch with default settings (English, port 7861)
PYTHONPATH=/workspace/usf-usf-bios-prod python usf/cli/train_ui.py

# Launch with custom settings
PYTHONPATH=/workspace/usf-usf-bios-prod python usf/cli/train_ui.py \
    --server_port 7861 \
    --lang en \
    --share  # Creates a public Gradio link
```

### Access the WebUI

Open your browser and navigate to:
- Local: `http://localhost:7861`
- Or the public URL if `--share` is enabled

## Supported Dataset Formats

The WebUI automatically detects and processes the following dataset formats:

### 1. OpenAI/Messages Format (Recommended)
```json
{
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I help you today?"}
    ]
}
```

**Supported Roles:**
- `system` - System instructions
- `user` - User messages
- `assistant` - Assistant responses
- `tool_call` / `function_call` - Tool/function calls
- `tool_response` / `tool` / `observation` - Tool responses

### 2. ShareGPT Format
```json
{
    "conversations": [
        {"from": "human", "value": "What is machine learning?"},
        {"from": "gpt", "value": "Machine learning is..."}
    ]
}
```

### 3. Alpaca Format
```json
{
    "instruction": "Translate the following text to French.",
    "input": "Hello, how are you?",
    "output": "Bonjour, comment allez-vous?"
}
```

### 4. Query-Response Format
```json
{
    "query": "What is the capital of France?",
    "response": "The capital of France is Paris."
}
```

## Training Types

| Type | Description | VRAM Usage |
|------|-------------|------------|
| `lora` | Low-Rank Adaptation | Low |
| `qlora` | Quantized LoRA (4-bit) | Very Low |
| `full` | Full Fine-tuning | High |
| `adalora` | Adaptive LoRA | Low |
| `dora` | Weight-Decomposed LoRA | Low |

## Configuration Options

### Model Configuration
- **Model ID/Path**: HuggingFace model ID or local path
- **Use HuggingFace**: Toggle for HF Hub access

### Dataset Configuration
- **Dataset ID/Path**: HF dataset ID (prefix with `HF::`) or local file path
- **Dataset Sample**: Limit samples for quick testing (0 = use all)
- **System Prompt**: Optional default system prompt

### Training Parameters
- **Epochs**: Number of training epochs
- **Batch Size**: Per-device batch size
- **Gradient Accumulation**: Steps to accumulate gradients
- **Learning Rate**: Optimizer learning rate
- **Max Length**: Maximum sequence length

### LoRA Parameters
- **Rank**: LoRA rank (higher = more capacity, more VRAM)
- **Alpha**: LoRA alpha scaling factor
- **Dropout**: LoRA dropout for regularization
- **Target Modules**: Which layers to apply LoRA (`all-linear` recommended)

### Advanced Settings
- **Gradient Checkpointing**: Reduce VRAM at cost of speed
- **Quantization**: BNB/HQQ/EETQ for reduced precision
- **DeepSpeed**: Multi-GPU training optimization

## Example Workflows

### Fine-tune Qwen with LoRA
1. Model: `Qwen/Qwen2.5-7B-Instruct`
2. Dataset: `HF::your-username/your-dataset`
3. Train Type: `lora`
4. LoRA Rank: 64, Alpha: 128
5. Click "Generate Command" then "Start Training"

### Fine-tune USF Omega
1. Model: `arpitsh018/usf-omega-40b-base`
2. Dataset: `HF::TeichAI/glm-4.7-2000x`
3. Train Type: `lora`
4. Max Length: 4096+
5. Enable Gradient Checkpointing

## Troubleshooting

### Out of Memory
- Reduce batch size to 1
- Enable gradient checkpointing
- Use QLoRA instead of LoRA
- Reduce max_length
- Reduce LoRA rank

### Slow Training
- Increase batch size if VRAM allows
- Increase gradient accumulation
- Enable Flash Attention if supported

### Dataset Errors
- Ensure JSON/JSONL format is valid
- Check that all required fields are present
- Use the "Preview Dataset" button to verify

## API Usage

You can also use the WebUI programmatically:

```python
from usf.pipelines.webui import build_train_ui

# Build and launch the UI
demo = build_train_ui(lang='en')
demo.launch(server_name='0.0.0.0', server_port=7861)
```
