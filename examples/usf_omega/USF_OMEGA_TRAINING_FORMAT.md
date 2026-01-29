# USF Omega Training Data Format Guide

This document describes the complete dataset format for training USF Omega models with all features.

## Table of Contents
1. [Basic Format](#basic-format)
2. [Message Roles](#message-roles)
3. [Model Identity vs System](#model-identity-vs-system)
4. [Function Calling](#function-calling)
5. [Structured Output](#structured-output)
6. [Reasoning Control](#reasoning-control)
7. [Self-Reflection](#self-reflection)
8. [Multi-Turn Conversations](#multi-turn-conversations)
9. [All Features Combined](#all-features-combined)
10. [Training Modes](#training-modes)

---

## Basic Format

**IMPORTANT:** All content must be INSIDE the `messages` array. The assistant's response, tool_calls, self_reflection, contents array - everything goes inside the assistant message.

```json
{
  "messages": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi! How can I help?"}
  ]
}
```

**Loss Boundaries:**
- `user` content → loss_scale = 0 (NOT trained)
- `assistant` content → loss_scale = 1 (TRAINED)

---

## Message Roles

### All Supported Roles

| Role | Jinja2 Output | loss_scale | Trained? |
|------|---------------|------------|----------|
| `system` (in messages) | `<start>developer` | 0 | ❌ |
| `developer` | `<start>developer` | 0 | ❌ |
| `user` | `<start>user` | 0 | ❌ |
| `assistant` | `<start>assistant` | 1 | ✅ |
| `tool` | `<start>tool` | 0 | ❌ |

**IMPORTANT:** `system` role in messages becomes DEVELOPER block, NOT system block!

---

## Model Identity vs System

### How It Works

```
┌─────────────────────────────────────────────────────────────────┐
│  model_identity (extra_kwargs)  →  <start>system (ACTUAL SYSTEM)│
│  role: "system" in messages     →  <start>developer             │
│  role: "developer" in messages  →  <start>developer             │
└─────────────────────────────────────────────────────────────────┘
```

### Example

```json
{
  "model_identity": "I am USF Omega, an advanced AI assistant.",
  "messages": [
    {"role": "system", "content": "Always be helpful and accurate."},
    {"role": "developer", "content": "Use formal language."},
    {"role": "user", "content": "Who are you?"}
  ],
  "assistant": "I am USF Omega, an advanced AI assistant."
}
```

**Jinja2 Output:**
```
<bos><start>system
I am USF Omega, an advanced AI assistant.
<end>
<start>developer
Always be helpful and accurate.
Use formal language.
<end>
<start>user
Who are you?
<end>
<start>assistant
<message_start>I am USF Omega, an advanced AI assistant.<message_end>
<end>
<eos>
```

---

## Function Calling

### Tool Definitions + Tool Calls (INSIDE assistant message)

```json
{
  "messages": [
    {"role": "user", "content": "Search for weather"},
    {
      "role": "assistant",
      "content": "I'll search for that.",
      "tool_calls": [
        {
          "id": "call_1",
          "type": "function",
          "function": {
            "name": "search",
            "arguments": {"query": "weather NYC"}
          }
        }
      ]
    }
  ],
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "search",
        "description": "Search the web",
        "parameters": {
          "type": "object",
          "properties": {
            "query": {"type": "string", "description": "Search query"}
          },
          "required": ["query"]
        }
      }
    }
  ]
}
```

### Parallel Tool Calls (INSIDE assistant message)

```json
{
  "messages": [
    {"role": "user", "content": "Search A, B and calculate 1+1"},
    {
      "role": "assistant",
      "content": "I'll do all three.",
      "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "search", "arguments": {"query": "A"}}},
        {"id": "c2", "type": "function", "function": {"name": "search", "arguments": {"query": "B"}}},
        {"id": "c3", "type": "function", "function": {"name": "calculate", "arguments": {"expr": "1+1"}}}
      ]
    }
  ],
  "tools": [...]
}
```

### Tool Response (Multi-Turn)

```json
{
  "messages": [
    {"role": "user", "content": "Search weather"},
    {
      "role": "assistant",
      "content": "Searching...",
      "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "search", "arguments": {"query": "weather"}}}]
    },
    {"role": "tool", "tool_call_id": "c1", "name": "search", "content": "Sunny, 75°F"},
    {"role": "assistant", "content": "The weather is sunny at 75°F!"}
  ],
  "tools": [...]
}
```

---

## Structured Output

### Using response_schema

```json
{
  "messages": [
    {"role": "user", "content": "Get user data"},
    {"role": "assistant", "content": "{\"name\": \"John\", \"age\": 30, \"email\": \"john@example.com\"}"}
  ],
  "response_schema": {
    "type": "object",
    "properties": {
      "name": {"type": "string"},
      "age": {"type": "integer"},
      "email": {"type": "string", "format": "email"}
    },
    "required": ["name", "age"]
  }
}
```

### Using response_format (OpenAI-style)

```json
{
  "messages": [
    {"role": "user", "content": "List items"},
    {"role": "assistant", "content": "{\"key\": \"value\"}"}
  ],
  "response_format": {"type": "json_object"}
}
```

### JSON Mode

```json
{
  "messages": [
    {"role": "user", "content": "Get result"},
    {"role": "assistant", "content": "{\"result\": 42}"}
  ],
  "json_mode": true
}
```

---

## Reasoning Control

### Reasoning Effort Levels

| Level | Description |
|-------|-------------|
| `none` | No reasoning block |
| `minimal` | Brief thought |
| `low` | Short reasoning |
| `medium` | Moderate analysis |
| `high` | Detailed reasoning |
| `max` | Exhaustive analysis |

### Example with Reasoning (for reasoning models)

```json
{
  "messages": [
    {"role": "user", "content": "Solve this complex problem"},
    {
      "role": "assistant",
      "contents": [
        {"type": "reasoning", "reasoning": "Step 1: Analyze...\nStep 2: Consider...\nStep 3: Synthesize..."},
        {"type": "text", "text": "Based on my analysis, the answer is 42."},
        {"type": "self_reflection", "self_reflection": "Verified the logic is sound."}
      ]
    }
  ],
  "reasoning_effort": "high",
  "self_reflection_enabled": true
}
```

### Thinking Budget (Token Limit)

```json
{
  "reasoning_effort": "max",
  "thinking_budget": 2048,
  "reasoning": "...",
  "assistant": "..."
}
```

---

## Self-Reflection

### Single Self-Reflection (INSIDE assistant message)

```json
{
  "messages": [
    {"role": "user", "content": "Write a poem"},
    {
      "role": "assistant",
      "content": "Roses are red...",
      "self_reflection": "Good rhyme scheme."
    }
  ],
  "self_reflection_enabled": true
}
```

### Multi-Iteration Self-Reflection (Contents Array INSIDE assistant message)

```json
{
  "messages": [
    {"role": "user", "content": "Write and improve a haiku"},
    {
      "role": "assistant",
      "contents": [
        {"type": "text", "text": "Cherry blossoms fall..."},
        {"type": "self_reflection", "self_reflection": "Good but could be more evocative"},
        {"type": "text", "text": "Pink petals dancing..."},
        {"type": "self_reflection", "self_reflection": "Much better with movement and depth"}
      ]
    }
  ],
  "self_reflection_enabled": true
}
```

### With Tool Calls in Iteration (INSIDE assistant message)

```json
{
  "messages": [
    {"role": "user", "content": "Search and analyze"},
    {
      "role": "assistant",
      "contents": [
        {"type": "text", "text": "Let me search first"},
        {"type": "tool_calls", "tool_calls": [{"id": "s1", "type": "function", "function": {"name": "search", "arguments": {}}}]},
        {"type": "self_reflection", "self_reflection": "Got good results"},
        {"type": "text", "text": "Based on the search, here's the answer"},
        {"type": "self_reflection", "self_reflection": "Complete and accurate"}
      ]
    }
  ],
  "tools": [...],
  "self_reflection_enabled": true
}
```

---

## Multi-Turn Conversations

### Simple Multi-Turn

```json
{
  "messages": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi!"},
    {"role": "user", "content": "How are you?"},
    {"role": "assistant", "content": "I'm doing well!"},
    {"role": "user", "content": "What can you do?"}
  ],
  "assistant": "I can help with many tasks..."
}
```

### Multi-Turn with Tool Calls

```json
{
  "messages": [
    {"role": "user", "content": "Search for X"},
    {"role": "assistant", "content": "Searching...", "tool_calls": [...]},
    {"role": "tool", "tool_call_id": "c1", "content": "Results..."},
    {"role": "assistant", "content": "Found X. Anything else?"},
    {"role": "user", "content": "Now search for Y"}
  ],
  "assistant": "Searching for Y...",
  "tool_calls": [...]
}
```

---

## All Features Combined

### Maximum Complexity Example

```json
{
  "model_identity": "I am USF Omega, an expert research assistant.",
  "messages": [
    {"role": "system", "content": "Always cite sources and verify information."},
    {"role": "developer", "content": "Use JSON format for data. Check work twice."},
    {"role": "user", "content": "Research the population of Tokyo, verify it, and give me a structured summary"}
  ],
  "tools": [
    {"type": "function", "function": {"name": "search", "parameters": {...}}},
    {"type": "function", "function": {"name": "verify", "parameters": {...}}}
  ],
  "search_enabled": true,
  "citations_enabled": true,
  "self_reflection_enabled": true,
  "reasoning_effort": "high",
  "response_schema": {
    "type": "object",
    "properties": {
      "city": {"type": "string"},
      "population": {"type": "integer"},
      "sources": {"type": "array", "items": {"type": "string"}},
      "confidence": {"type": "number"}
    }
  },
  "reasoning": "I need to: 1) Search for Tokyo population, 2) Verify with second source, 3) Format as JSON",
  "contents": [
    {"type": "reasoning", "reasoning": "I need to search and verify..."},
    {"type": "text", "text": "I'll search for Tokyo's population."},
    {"type": "tool_calls", "tool_calls": [{"function": {"name": "search", "arguments": {"query": "Tokyo population 2024"}}}]},
    {"type": "self_reflection", "self_reflection": "Got initial data, need to verify"},
    {"type": "text", "text": "Verifying with another source..."},
    {"type": "tool_calls", "tool_calls": [{"function": {"name": "verify", "arguments": {"claim": "Tokyo population is 14 million"}}}]},
    {"type": "self_reflection", "self_reflection": "Verified. Now formatting response."},
    {"type": "text", "text": "{\"city\": \"Tokyo\", \"population\": 14000000, \"sources\": [\"source1\", \"source2\"], \"confidence\": 0.95}"},
    {"type": "self_reflection", "self_reflection": "Complete with all required fields and verification."}
  ]
}
```

---

## Training Modes

### SFT (Supervised Fine-Tuning)

Use any format above. All assistant content is trained.

### DPO/ORPO (Offline RLHF)

```json
{
  "messages": [{"role": "user", "content": "Question"}],
  "chosen": {"role": "assistant", "content": "Good detailed answer"},
  "rejected": {"role": "assistant", "content": "Bad short answer"}
}
```

### PPO/GRPO (Online RLHF)

Same format as SFT. The rollout generates responses online.

### KTO (Binary Preference)

```json
{
  "messages": [{"role": "user", "content": "Question"}],
  "assistant": "Response",
  "label": true
}
```

---

## Feature Flags Summary

| Flag | Type | Description |
|------|------|-------------|
| `model_identity` | string | Actual system message |
| `tools` / `functions` | array | Function definitions |
| `response_schema` | object | JSON Schema for output |
| `response_format` | object | OpenAI-style format |
| `json_mode` | boolean | Force JSON output |
| `search_enabled` | boolean | Enable search function |
| `citations_enabled` | boolean | Enable citation format |
| `self_reflection_enabled` | boolean | Enable reflection |
| `reasoning_effort` | string | none/minimal/low/medium/high/max |
| `thinking_budget` | integer | Max reasoning tokens |

---

## Loss Boundary Summary

**Trained (loss_scale = 1):**
- All assistant content
- Reasoning blocks
- Message content
- Tool calls (model generates)
- Self-reflection
- EOS token

**NOT Trained (loss_scale = 0):**
- BOS token
- System block (from model_identity)
- Developer block (from system/developer roles)
- Functions namespace
- User messages
- Tool responses
