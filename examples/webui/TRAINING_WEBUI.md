# USF BIOS Training WebUI

A universal fine-tuning interface for training any LLM supported by USF BIOS.

## Quick Start

```bash
# Launch the WebUI
usf train-ui

# Or with options
usf train-ui \
    --server_port 7861 \
    --share  # Creates public Gradio link
```

Access at: `http://localhost:7861`

---

## Training Types

| Type | VRAM Required | Speed | Quality | Use Case |
|------|---------------|-------|---------|----------|
| **LoRA** | Model-dependent | Fast | Excellent | Recommended for most cases |
| **QLoRA** | ~50% of LoRA | Medium | Very Good | Limited VRAM |
| **Full** | 2-4x LoRA | Slow | Best | Maximum quality, multi-GPU |
| **AdaLoRA** | Similar to LoRA | Fast | Excellent | Adaptive rank allocation |
| **DoRA** | Similar to LoRA | Fast | Excellent | Weight-decomposed LoRA |

---

## Research-Based Default Hyperparameters

Based on multiple research sources:
- [LoRA best practices](https://thinkingmachines.ai/blog/lora/)
- [QLoRA paper](https://arxiv.org/abs/2305.14314)
- [LoRA paper](https://arxiv.org/abs/2106.09685)

### Key Research Findings
1. **LoRA LR should be ~10x Full FT LR** (validated across multiple studies)
2. **Alpha = 2 × Rank** is optimal scaling
3. **Apply LoRA to ALL linear layers** (MLP + Attention) for best results
4. **Rank 64-128** balances quality and efficiency

### LoRA Defaults
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning Rate | 2e-4 | ~10x higher than FullFT (research finding) |
| LoRA Rank (r) | 64 | Good capacity/efficiency balance |
| LoRA Alpha (α) | 128 | α = 2×r is optimal |
| LoRA Dropout | 0.05 | Light regularization |
| Target Modules | all-linear | Apply to ALL layers (MLP+Attention) for best results |
| Warmup Ratio | 0.03 | Standard warmup |
| Weight Decay | 0.1 | Regularization |
| Gradient Accumulation | 16 | Effective batch size = 16 |

### QLoRA Defaults
Same as LoRA with 4-bit NF4 quantization (BitsAndBytes).

### AdaLoRA Defaults
| Parameter | Value | Rationale |
|-----------|-------|-----------|
| Learning Rate | 5e-5 | Lower for adaptive rank |
| Initial Rank | 64 | Will be pruned during training |

### DoRA Defaults (Weight Decomposed LoRA)
| Parameter | Value |
|-----------|-------|
| Learning Rate | 1e-4 |
| Rank | 64 |

### Full Fine-tuning Defaults
| Parameter | Value |
|-----------|-------|
| Learning Rate | 2e-5 |
| Gradient Accumulation | 32 |

---

## Dataset Format

Your dataset must be **JSONL** format with a `messages` array.

### Supported Roles

| Role | Purpose |
|------|---------|
| `system` | System-level instructions |
| `developer` | Developer instructions (higher priority) |
| `user` | User input/query |
| `assistant` | Model response |
| `tool` | Tool/function response |

### Assistant Message Keys

| Key | Required | Description |
|-----|----------|-------------|
| `role` | ✅ | Always `"assistant"` |
| `content` | ✅ | The response text (or `null` if only tool_calls) |
| `reasoning` | ❌ | Chain-of-thought reasoning (converted during training) |
| `tool_calls` | ❌ | Array of tool/function calls |

### Special Formats

#### 1. With Reasoning (separate key)
```json
{
  "role": "assistant",
  "reasoning": "Step-by-step thinking here...",
  "content": "Final response here."
}
```

#### 2. Tool Calls
```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "call_123",
      "type": "function",
      "function": {
        "name": "get_weather",
        "arguments": "{\"location\": \"Tokyo\"}"
      }
    }
  ]
}
```

#### 3. Tool Response
```json
{
  "role": "tool",
  "tool_call_id": "call_123",
  "content": "{\"temperature\": 22}"
}
```

#### 4. Structured Output (JSON response)
```json
{
  "role": "assistant",
  "content": "{\"name\": \"John\", \"age\": 30}"
}
```

### Complete Examples

#### Basic Conversation
```json
{"messages": [
  {"role": "system", "content": "You are a helpful assistant."},
  {"role": "user", "content": "What is Python?"},
  {"role": "assistant", "content": "Python is a high-level programming language..."}
]}
```

#### With Developer Instructions
```json
{"messages": [
  {"role": "system", "content": "You are a coding assistant."},
  {"role": "developer", "content": "Always include code examples."},
  {"role": "user", "content": "How do I read a file?"},
  {"role": "assistant", "content": "Use open():\n```python\nwith open('f.txt') as f:\n    print(f.read())\n```"}
]}
```

#### With Reasoning
```json
{"messages": [
  {"role": "user", "content": "What is 15 * 23?"},
  {"role": "assistant", "reasoning": "15 * 23 = 15 * 20 + 15 * 3 = 300 + 45 = 345", "content": "The answer is 345."}
]}
```

#### With Tool Calling
```json
{"messages": [
  {"role": "system", "content": "You have access to tools."},
  {"role": "user", "content": "What's the weather in Tokyo?"},
  {"role": "assistant", "content": null, "tool_calls": [{"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": "{\"location\": \"Tokyo\"}"}}]},
  {"role": "tool", "tool_call_id": "call_1", "content": "{\"temp\": 22, \"condition\": \"sunny\"}"},
  {"role": "assistant", "reasoning": "Tool returned sunny weather at 22°C.", "content": "Tokyo is sunny, 22°C."}
]}
```

---

## Dataset Sources

### Local File
```
/path/to/your/dataset.jsonl
```

### HuggingFace Dataset
```
HF::username/dataset-name
```

---

## Example Training Command

Generated by the WebUI:

```bash
usf sft \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --use_hf true \
    --dataset "/path/to/dataset.jsonl" \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 1e-4 \
    --max_length 4096 \
    --output_dir "output/sft-training" \
    --lora_rank 64 \
    --lora_alpha 32 \
    --lora_dropout 0.05 \
    --target_modules all-linear \
    --warmup_ratio 0.03 \
    --weight_decay 0.1 \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --save_steps 500 \
    --gradient_checkpointing true
```

---

## Troubleshooting

### Out of Memory
1. Use QLoRA instead of LoRA
2. Reduce `max_length`
3. Enable gradient checkpointing
4. Reduce LoRA rank

### Slow Training
1. Increase batch size if VRAM allows
2. Increase gradient accumulation
3. Use Flash Attention (if supported)

### Dataset Errors
1. Validate format with WebUI's "Validate" button
2. Ensure all messages have `role` and `content`
3. Check JSON syntax in each line
