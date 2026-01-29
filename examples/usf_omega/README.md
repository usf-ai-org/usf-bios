# USF Omega Model - Chat Template Documentation

## Overview

USF Omega uses a unique chat template format with special tokens for message boundaries, reasoning, tool calls, and structured output.

## Special Tokens

| Token | Purpose |
|-------|---------|
| `<\|:@:\|start\|:@:\|>` | Start of message block |
| `<\|:@::\|end\|:@::\|>` | End of message block |
| `<\|:@:\|reasoning_start\|:@:\|>` | Start of reasoning/thinking |
| `<\|:@::\|reasoning_end\|:@::\|>` | End of reasoning/thinking |
| `<\|:@:\|message\|:@:\|>` | Start of message content |
| `<\|:@::\|message\|:@::\|>` | End of message content |
| `<\|:@:\|functions_start\|:@:\|>` | Start of function calls |
| `<\|:@::\|functions_end\|:@::\|>` | End of function calls |
| `<\|:@:\|invoke_start\|:@:\|>` | Start of function invocation |
| `<\|:@::\|invoke_end\|:@::\|>` | End of function invocation |
| `<\|:@:\|constrain\|:@:\|>` | Constrained output marker (e.g., json) |

## Message Roles

### System Message
```
<|:@:|start|:@:|>system
You are a helpful assistant.
<|:@::|end|:@::|>
```

### Developer Message
```
<|:@:|start|:@:|>developer
Always respond in JSON format.
<|:@::|end|:@::|>
```

### User Message
```
<|:@:|start|:@:|>user
What is the weather in Tokyo?
<|:@::|end|:@::|>
```

### Assistant Message (Basic)
```
<|:@:|start|:@:|>assistant
<|:@:|message|:@:|>The weather in Tokyo is sunny.<|:@::|message|:@::|>
<|:@::|end|:@::|>
```

### Assistant Message (With Reasoning)
```
<|:@:|start|:@:|>assistant
<|:@:|reasoning_start|:@:|>
Let me think about this step by step...
<|:@::|reasoning_end|:@::|>
<|:@:|message|:@:|>The answer is 42.<|:@::|message|:@::|>
<|:@::|end|:@::|>
```

## Function/Tool Calling

### Define Functions
```
<|:@:|start|:@:|>functions
[
  {
    "name": "get_weather",
    "description": "Get weather for a city",
    "parameters": {
      "type": "object",
      "properties": {
        "city": {"type": "string"},
        "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}
      },
      "required": ["city"]
    }
  }
]
<|:@::|end|:@::|>
```

### Single Tool Call
```
<|:@:|start|:@:|>assistant
<|:@:|message|:@:|>Let me check the weather.<|:@::|message|:@::|>
<|:@:|functions_start|:@:|>
<|:@:|invoke_start|:@:|>to="function.get_weather"
<|parameter name="city"|>Tokyo<||parameter||>
<|parameter name="unit"|>celsius<||parameter||>
<|:@::|invoke_end|:@::|>
<|:@::|functions_end|:@::|>
<|:@::|end|:@::|>
```

### Parallel Tool Calls
```
<|:@:|start|:@:|>assistant
<|:@:|message|:@:|>I'll check both locations.<|:@::|message|:@::|>
<|:@:|functions_start|:@:|>
<|:@:|invoke_start|:@:|>to="function.get_weather"
<|parameter name="city"|>Tokyo<||parameter||>
<|:@::|invoke_end|:@::|>
<|:@:|invoke_start|:@:|>to="function.get_weather"
<|parameter name="city"|>Paris<||parameter||>
<|:@::|invoke_end|:@::|>
<|:@::|functions_end|:@::|>
<|:@::|end|:@::|>
```

### Tool Response (Single)
```
<|:@:|start|:@:|>tool
<||function_results_start to=function.get_weather||>{"temperature": 22, "condition": "sunny"}<||function_response_end||>
<|:@::|function_results|:@::|>
<|:@::|end|:@::|>
```

### Tool Response (Parallel)
```
<|:@:|start|:@:|>tool
<||function_results_start to=function.get_weather||>{"temperature": 22, "condition": "sunny"}<||function_response_end||>
<||function_results_start to=function.search||>{"results": ["Article 1", "Article 2"]}<||function_response_end||>
<|:@::|function_results|:@::|>
<|:@::|end|:@::|>
```

## Structured Output (Response Format)

### Define Schema
```
<|:@:|start|:@:|>response_format.json
{
  "type": "object",
  "properties": {
    "name": {"type": "string"},
    "age": {"type": "integer"}
  },
  "required": ["name", "age"]
}
<|:@::|end|:@::|>
```

### Constrained Output
```
<|:@:|start|:@:|>assistant
<|:@:|constrain|:@:|>json
<|:@:|message|:@:|>{"name": "John", "age": 30}<|:@::|message|:@::|>
<|:@:|constrain|:@:|>
<|:@::|end|:@::|>
```

### Structured Output with Reasoning
```
<|:@:|start|:@:|>assistant
<|:@:|constrain|:@:|>json
<|:@:|reasoning_start|:@:|>
I need to extract the name and age from the request...
<|:@::|reasoning_end|:@::|>
<|:@:|message|:@:|>{"name": "John", "age": 30}<|:@::|message|:@::|>
<|:@:|constrain|:@:|>
<|:@::|end|:@::|>
```

## HuggingFace Chat Template Usage

### Using with Tokenizer

```python
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("arpitsh018/usf-omega-40b-base")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
]

# Default (no reasoning)
prompt = tokenizer.apply_chat_template(
    messages, 
    add_generation_prompt=True,
    tokenize=False
)

# With reasoning enabled
prompt_with_reasoning = tokenizer.apply_chat_template(
    messages, 
    add_generation_prompt=True,
    tokenize=False,
    enable_reasoning=True
)

# With JSON mode (constrain prefill)
prompt_json = tokenizer.apply_chat_template(
    messages, 
    add_generation_prompt=True,
    tokenize=False,
    json_mode=True  # Prefills with <|:@:|constrain|:@:|>json
)

# JSON mode with reasoning
prompt_json_reasoning = tokenizer.apply_chat_template(
    messages, 
    add_generation_prompt=True,
    tokenize=False,
    json_mode=True,
    enable_reasoning=True
)
```

### With Tool Calls

```python
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {
        "role": "system",
        "tools": [
            {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {"city": {"type": "string"}}}
            }
        ]
    },
    {"role": "user", "content": "What's the weather in Tokyo?"},
    {
        "role": "assistant",
        "content": "Let me check.",
        "tool_calls": [
            {"function": {"name": "get_weather", "arguments": {"city": "Tokyo"}}}
        ]
    },
    {
        "role": "tool",
        "name": "get_weather",
        "content": '{"temperature": 22}'
    }
]

prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
```

### With Structured Output

```python
messages = [
    {"role": "system", "content": "You are a data extractor."},
    {
        "role": "system",
        "response_format": {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}}
        }
    },
    {"role": "user", "content": "John is 30 years old."}
]

prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
```

## Files in this Directory

| File | Description |
|------|-------------|
| `chat_template.jinja2` | Full Jinja2 template (instruct/non-reasoning) |
| `chat_template_reasoning.jinja2` | Reasoning template with thinking blocks |
| `training_data_examples.jsonl` | Training data examples for all features |
| `USF_OMEGA_TRAINING_FORMAT.md` | Complete training data format documentation |
| `test_chat_template_openai.py` | Test script for all template scenarios |

## Running Tests

```bash
cd examples/usf_omega
python test_chat_template_openai.py
```

## Two Template Modes

1. **Default (Non-Reasoning)**: Used for standard conversations
   - Faster inference
   - Direct responses without thinking

2. **Reasoning Mode**: Used when `enable_reasoning=True`
   - Shows step-by-step thinking
   - Better for complex tasks
   - Use with `THINKING_BUDGET` environment variable
