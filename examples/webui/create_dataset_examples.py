#!/usr/bin/env python
# Copyright (c) Ultrasafe AI. All rights reserved.
"""
Example script demonstrating how to create datasets in supported formats
for use with USF BIOS Training WebUI.

Supported formats:
1. OpenAI/Messages format (recommended)
2. ShareGPT format
3. Alpaca format
4. Query-Response format

All formats support tool calling with roles:
- system, user, assistant, tool_call, tool_response
"""

import json
import os
from typing import List, Dict, Any


def create_openai_format_dataset(output_path: str):
    """
    Create dataset in OpenAI/Messages format.
    This is the most flexible format supporting all features.
    """
    samples = [
        # Basic conversation
        {
            "messages": [
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": "What is Python?"},
                {"role": "assistant", "content": "Python is a high-level, interpreted programming language known for its simplicity and readability. It was created by Guido van Rossum and first released in 1991."}
            ]
        },
        # Multi-turn conversation
        {
            "messages": [
                {"role": "system", "content": "You are a coding assistant."},
                {"role": "user", "content": "How do I read a file in Python?"},
                {"role": "assistant", "content": "You can read a file using the `open()` function with a context manager:\n\n```python\nwith open('file.txt', 'r') as f:\n    content = f.read()\n```"},
                {"role": "user", "content": "How do I read it line by line?"},
                {"role": "assistant", "content": "To read line by line:\n\n```python\nwith open('file.txt', 'r') as f:\n    for line in f:\n        print(line.strip())\n```"}
            ]
        },
        # With tool calling
        {
            "messages": [
                {"role": "system", "content": "You are an assistant with access to tools."},
                {"role": "user", "content": "What's the weather in Tokyo?"},
                {"role": "tool_call", "content": '{"name": "get_weather", "arguments": {"location": "Tokyo"}}'},
                {"role": "tool_response", "content": '{"temperature": 22, "condition": "sunny", "humidity": 60}'},
                {"role": "assistant", "content": "The weather in Tokyo is currently sunny with a temperature of 22°C and 60% humidity."}
            ]
        },
        # Reasoning/thinking format (for models that support it)
        {
            "messages": [
                {"role": "user", "content": "What is 15 * 23?"},
                {"role": "assistant", "content": "<think>Let me calculate step by step:\n15 * 23 = 15 * 20 + 15 * 3 = 300 + 45 = 345</think>\n\nThe answer is 345."}
            ]
        },
    ]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Created OpenAI format dataset: {output_path} ({len(samples)} samples)")


def create_sharegpt_format_dataset(output_path: str):
    """
    Create dataset in ShareGPT format.
    Uses 'conversations' with 'from' (human/gpt) and 'value' keys.
    """
    samples = [
        {
            "conversations": [
                {"from": "human", "value": "Explain quantum computing in simple terms."},
                {"from": "gpt", "value": "Quantum computing uses quantum bits (qubits) that can be both 0 and 1 simultaneously, unlike regular bits. This allows quantum computers to process many possibilities at once, making them potentially much faster for certain problems."}
            ]
        },
        {
            "conversations": [
                {"from": "human", "value": "What are the main benefits of exercise?"},
                {"from": "gpt", "value": "Exercise provides numerous benefits:\n1. Improved cardiovascular health\n2. Better mental health and mood\n3. Weight management\n4. Stronger muscles and bones\n5. Better sleep quality\n6. Increased energy levels"},
                {"from": "human", "value": "How much exercise should I do per week?"},
                {"from": "gpt", "value": "Health guidelines recommend:\n- At least 150 minutes of moderate aerobic activity OR 75 minutes of vigorous activity per week\n- Strength training exercises 2 or more days per week\n- Start gradually if you're new to exercise and consult a doctor if you have health concerns."}
            ]
        },
        # With system prompt (some ShareGPT variants support this)
        {
            "system": "You are a professional chef assistant.",
            "conversations": [
                {"from": "human", "value": "How do I make a perfect omelette?"},
                {"from": "gpt", "value": "For a perfect omelette:\n\n1. **Prep**: Beat 2-3 eggs with salt and pepper\n2. **Heat**: Warm a non-stick pan over medium heat with butter\n3. **Pour**: Add eggs when butter foams (not browns)\n4. **Cook**: Gently push edges toward center, tilting pan\n5. **Fill**: Add fillings when top is slightly wet\n6. **Fold**: Fold in thirds and slide onto plate\n\nKey tip: Don't overcook - the center should be slightly creamy!"}
            ]
        },
    ]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Created ShareGPT format dataset: {output_path} ({len(samples)} samples)")


def create_alpaca_format_dataset(output_path: str):
    """
    Create dataset in Alpaca format.
    Uses 'instruction', 'input', and 'output' keys.
    """
    samples = [
        {
            "instruction": "Translate the following English text to French.",
            "input": "Hello, how are you today?",
            "output": "Bonjour, comment allez-vous aujourd'hui?"
        },
        {
            "instruction": "Summarize the following text in one sentence.",
            "input": "Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. It focuses on developing computer programs that can access data and use it to learn for themselves.",
            "output": "Machine learning is a branch of AI that allows systems to automatically learn and improve from data without explicit programming."
        },
        {
            "instruction": "Write a haiku about programming.",
            "input": "",
            "output": "Code flows like water\nBugs hide in the shadowed depths\nDebugger finds truth"
        },
        {
            "instruction": "Classify the sentiment of this review as positive, negative, or neutral.",
            "input": "The product arrived on time and works exactly as described. Very happy with my purchase!",
            "output": "Positive"
        },
    ]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Created Alpaca format dataset: {output_path} ({len(samples)} samples)")


def create_query_response_format_dataset(output_path: str):
    """
    Create dataset in simple Query-Response format.
    Uses 'query' and 'response' keys.
    """
    samples = [
        {
            "query": "What is the capital of Japan?",
            "response": "The capital of Japan is Tokyo."
        },
        {
            "query": "Explain the difference between HTTP and HTTPS.",
            "response": "HTTP (Hypertext Transfer Protocol) transmits data in plain text, while HTTPS (HTTP Secure) encrypts data using TLS/SSL, making it secure for sensitive information like passwords and payment details."
        },
        {
            "query": "Write a Python function to calculate factorial.",
            "response": "```python\ndef factorial(n):\n    if n < 0:\n        raise ValueError(\"Factorial not defined for negative numbers\")\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)\n```"
        },
        # With system prompt
        {
            "system": "You are a math tutor.",
            "query": "What is the quadratic formula?",
            "response": "The quadratic formula solves equations of the form ax² + bx + c = 0:\n\nx = (-b ± √(b² - 4ac)) / (2a)\n\nWhere:\n- a, b, c are coefficients from the equation\n- The ± means there are typically two solutions"
        },
    ]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Created Query-Response format dataset: {output_path} ({len(samples)} samples)")


def create_tool_calling_dataset(output_path: str):
    """
    Create a comprehensive tool-calling dataset in OpenAI format.
    This demonstrates function calling capabilities.
    """
    samples = [
        # Weather tool
        {
            "messages": [
                {"role": "system", "content": "You are an assistant with access to weather and calculator tools."},
                {"role": "user", "content": "What's the weather like in New York?"},
                {"role": "tool_call", "content": json.dumps({
                    "name": "get_weather",
                    "arguments": {"location": "New York", "unit": "celsius"}
                })},
                {"role": "tool_response", "content": json.dumps({
                    "location": "New York",
                    "temperature": 18,
                    "unit": "celsius",
                    "condition": "partly cloudy",
                    "humidity": 65
                })},
                {"role": "assistant", "content": "The weather in New York is currently partly cloudy with a temperature of 18°C and 65% humidity."}
            ]
        },
        # Calculator tool
        {
            "messages": [
                {"role": "system", "content": "You have access to a calculator tool."},
                {"role": "user", "content": "What is 1234 * 5678?"},
                {"role": "tool_call", "content": json.dumps({
                    "name": "calculator",
                    "arguments": {"operation": "multiply", "a": 1234, "b": 5678}
                })},
                {"role": "tool_response", "content": json.dumps({"result": 7006652})},
                {"role": "assistant", "content": "1234 × 5678 = 7,006,652"}
            ]
        },
        # Multiple tool calls
        {
            "messages": [
                {"role": "system", "content": "You can search the web and get current time."},
                {"role": "user", "content": "Search for the latest news about AI and tell me the current time."},
                {"role": "tool_call", "content": json.dumps({
                    "name": "web_search",
                    "arguments": {"query": "latest AI news"}
                })},
                {"role": "tool_response", "content": json.dumps({
                    "results": [
                        {"title": "New AI Model Achieves Human-Level Performance", "source": "TechNews"},
                        {"title": "AI Regulations Proposed by EU", "source": "WorldNews"}
                    ]
                })},
                {"role": "tool_call", "content": json.dumps({
                    "name": "get_current_time",
                    "arguments": {"timezone": "UTC"}
                })},
                {"role": "tool_response", "content": json.dumps({"time": "2024-01-15T14:30:00Z"})},
                {"role": "assistant", "content": "Here are the latest AI news:\n1. 'New AI Model Achieves Human-Level Performance' from TechNews\n2. 'AI Regulations Proposed by EU' from WorldNews\n\nThe current time is 14:30 UTC."}
            ]
        },
        # Database query tool
        {
            "messages": [
                {"role": "system", "content": "You can query a product database."},
                {"role": "user", "content": "Find all products under $50 in the electronics category."},
                {"role": "tool_call", "content": json.dumps({
                    "name": "query_products",
                    "arguments": {
                        "category": "electronics",
                        "max_price": 50,
                        "sort_by": "price"
                    }
                })},
                {"role": "tool_response", "content": json.dumps({
                    "products": [
                        {"name": "USB-C Cable", "price": 9.99},
                        {"name": "Wireless Mouse", "price": 24.99},
                        {"name": "Bluetooth Earbuds", "price": 39.99}
                    ],
                    "total_count": 3
                })},
                {"role": "assistant", "content": "I found 3 electronics products under $50:\n\n1. **USB-C Cable** - $9.99\n2. **Wireless Mouse** - $24.99\n3. **Bluetooth Earbuds** - $39.99"}
            ]
        },
    ]
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')
    
    print(f"Created Tool-Calling dataset: {output_path} ({len(samples)} samples)")


def main():
    """Create all example datasets"""
    output_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(output_dir, 'sample_data')
    os.makedirs(data_dir, exist_ok=True)
    
    print("Creating example datasets...\n")
    
    create_openai_format_dataset(os.path.join(data_dir, 'openai_format.jsonl'))
    create_sharegpt_format_dataset(os.path.join(data_dir, 'sharegpt_format.jsonl'))
    create_alpaca_format_dataset(os.path.join(data_dir, 'alpaca_format.jsonl'))
    create_query_response_format_dataset(os.path.join(data_dir, 'query_response_format.jsonl'))
    create_tool_calling_dataset(os.path.join(data_dir, 'tool_calling.jsonl'))
    
    print(f"\n✅ All datasets created in: {data_dir}")
    print("\nTo use these datasets with the WebUI:")
    print(f"  Dataset Path: {data_dir}/<filename>.jsonl")


if __name__ == '__main__':
    main()
