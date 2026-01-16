#!/usr/bin/env python3
# Copyright (c) Ultrasafe AI. All rights reserved.
"""
Test USF Omega Chat Template with OpenAI-style inputs.

This is a pure Python implementation (no jinja2 required) that:
- Handles OpenAI-style tools and response_format at top level
- Auto-appends developer instructions for tools/response_format
- Handles tool_call_id mapping for tool responses
- Validates prompt structure and header order

Tests:
- tools (function definitions) at top level
- response_format (JSON schema) at top level
- Multi-turn conversation with system, developer, user, assistant, tool messages
- Tool calls and tool responses with tool_call_id
- Auto-appended developer instructions
"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


# =============================================================================
# TOKENS
# =============================================================================

START_TOKEN = '<|:@:|start|:@:|>'
END_TOKEN = '<|:@::|end|:@::|>'
REASONING_START = '<|:@:|reasoning_start|:@:|>'
REASONING_END = '<|:@::|reasoning_end|:@::|>'
MESSAGE_START = '<|:@:|message|:@:|>'
MESSAGE_END = '<|:@::|message|:@::|>'
FUNCTIONS_START = '<|:@:|functions_start|:@:|>'
FUNCTIONS_END = '<|:@::|functions_end|:@::|>'
INVOKE_START = '<|:@:|invoke_start|:@:|>'
INVOKE_END = '<|:@::|invoke_end|:@::|>'
CONSTRAIN_TOKEN = '<|:@:|constrain|:@:|>'
CONSTRAIN_END = '<|:@::|constrain|:@::|>'
FUNCTION_RESULTS_START = '<|:@:|function_results|:@:|>'
FUNCTION_RESULTS_END = '<|:@::|function_results|:@::|>'


# =============================================================================
# AUTO-APPENDED DEVELOPER INSTRUCTIONS
# =============================================================================

FUNCTION_INSTRUCTIONS = """# Functions

## Description

You have access to the functions listed below. These are the **only** functions available to connect with external services.

When the user provides a task, evaluate whether any functions are needed to complete it:

- **Parallel execution:** If some of the tasks are independent, call the relevant functions in parallel for speed optimization.
- **Sequential execution:** If a function's output is required as input for another, execute them sequentially.

You may call functions **multiple times** and in **any order** depending on what the task requires. Optimize for both correctness and speed.

**Important:**
- Always include all **required parameters** as defined in the function schema.
- Do **not** add extra parameters that are not defined in the function schema.

## Function Call Format

To call a function, use the following format:

```
<|:@:|functions_start|:@:|>
<|:@:|invoke_start|:@:|>to="function.FUNCTION_NAME"
<|parameter name="PARAM_NAME"|>PARAM_VALUE<||parameter||>
<|:@::|invoke_end|:@::|>
<|:@::|functions_end|:@::|>
```

For **parallel function calls** (multiple independent calls), include multiple invoke blocks:

```
<|:@:|functions_start|:@:|>
<|:@:|invoke_start|:@:|>to="function.FUNCTION_NAME_1"
<|parameter name="param"|>value1<||parameter||>
<|:@::|invoke_end|:@::|>
<|:@:|invoke_start|:@:|>to="function.FUNCTION_NAME_2"
<|parameter name="param"|>value2<||parameter||>
<|:@::|invoke_end|:@::|>
<|:@::|functions_end|:@::|>
```"""

RESPONSE_FORMAT_INSTRUCTIONS = """# Response Format

## Description

Below is the required response format in JSON schema. When generating your response, **always follow this schema**:

- **If `strict: true`:** Do NOT add any keys that are not defined in the schema.
- **If `strict: false`:** You may add extra keys if they are important and relevant to the response.
- **Required keys:** Always include all `required` fields in your response—even if the value is empty or null, depending on context.
- **Follow the format completely:** Ensure your output is valid JSON that conforms to the schema structure."""


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _json_dumps(value: Any) -> str:
    """Pretty-print JSON."""
    return json.dumps(value, indent=2, ensure_ascii=False)


def _format_block(role: str, content: str) -> str:
    """Format a message block."""
    return f'{START_TOKEN}{role}\n{content}{END_TOKEN}\n'


# =============================================================================
# OPENAI FORMAT CONVERTERS
# =============================================================================

def extract_functions_from_tools(tools: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert OpenAI tools format to simple function list."""
    functions = []
    for tool in tools:
        if tool.get('type') == 'function' and 'function' in tool:
            functions.append(tool['function'])
        elif 'name' in tool:
            functions.append(tool)
    return functions


def extract_schema_from_response_format(response_format: Dict[str, Any]) -> Dict[str, Any]:
    """Extract JSON schema from OpenAI response_format."""
    if 'json_schema' in response_format:
        js = response_format['json_schema']
        return js.get('schema', js)
    if 'schema' in response_format:
        return response_format['schema']
    return response_format


def build_tool_call_id_map(messages: List[Dict[str, Any]]) -> Dict[str, str]:
    """Build mapping from tool_call_id to function name."""
    mapping = {}
    for msg in messages:
        if msg.get('tool_calls'):
            for tc in msg['tool_calls']:
                if tc.get('id') and tc.get('function', {}).get('name'):
                    mapping[tc['id']] = tc['function']['name']
    return mapping


def _get_schema_strict(response_format: Dict[str, Any]) -> bool:
    """Extract strict flag from response_format."""
    if 'json_schema' in response_format:
        return response_format['json_schema'].get('strict', False)
    return response_format.get('strict', False)


def _validate_json_against_schema(
    content: str,
    schema: Dict[str, Any],
    strict: bool = False
) -> bool:
    """
    Validate if JSON content matches the schema.
    
    Args:
        content: JSON string to validate
        schema: JSON schema dict
        strict: If True, no extra keys allowed; if False, extra keys OK
    
    Returns:
        True if content is valid JSON that matches schema
    """
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return False
    
    return _validate_object_against_schema(data, schema, strict)


def _validate_object_against_schema(
    data: Any,
    schema: Dict[str, Any],
    strict: bool = False
) -> bool:
    """Recursively validate data against schema."""
    schema_type = schema.get('type')
    
    # Check type match
    if schema_type == 'object':
        if not isinstance(data, dict):
            return False
        
        properties = schema.get('properties', {})
        required = set(schema.get('required', []))
        
        # Check all required keys are present
        for req_key in required:
            if req_key not in data:
                return False
        
        # If strict, check no extra keys
        if strict:
            allowed_keys = set(properties.keys())
            for key in data.keys():
                if key not in allowed_keys:
                    return False
        
        # Recursively validate nested properties
        for key, value in data.items():
            if key in properties:
                if not _validate_object_against_schema(value, properties[key], strict):
                    return False
        
        return True
    
    elif schema_type == 'array':
        if not isinstance(data, list):
            return False
        items_schema = schema.get('items', {})
        for item in data:
            if not _validate_object_against_schema(item, items_schema, strict):
                return False
        return True
    
    elif schema_type == 'string':
        if not isinstance(data, str):
            return False
        if 'enum' in schema and data not in schema['enum']:
            return False
        return True
    
    elif schema_type == 'number':
        return isinstance(data, (int, float)) and not isinstance(data, bool)
    
    elif schema_type == 'integer':
        return isinstance(data, int) and not isinstance(data, bool)
    
    elif schema_type == 'boolean':
        return isinstance(data, bool)
    
    elif schema_type == 'null':
        return data is None
    
    # No type specified or unknown type - accept
    return True


def _content_matches_schema(
    content: str,
    response_format: Optional[Dict[str, Any]]
) -> bool:
    """Check if assistant content matches the response_format schema."""
    if not response_format or not content:
        return False
    
    schema = extract_schema_from_response_format(response_format)
    strict = _get_schema_strict(response_format)
    
    return _validate_json_against_schema(content, schema, strict)


def _schema_to_typescript(
    schema: Dict[str, Any],
    indent: int = 0,
    inline: bool = False,
) -> str:
    """
    Convert JSON Schema to TypeScript type definition.
    
    Handles:
    - Primitives: string, number, integer, boolean
    - Enums: string/number with enum values
    - Arrays: typed arrays, arrays of objects, nested arrays
    - Objects: nested objects up to 5-6 levels deep
    - Required vs optional properties
    - Descriptions as comments
    """
    schema_type = schema.get('type', 'any')
    enum_values = schema.get('enum')
    
    # Handle enums
    if enum_values:
        if all(isinstance(v, str) for v in enum_values):
            return ' | '.join(f'"{v}"' for v in enum_values)
        else:
            return ' | '.join(str(v) for v in enum_values)
    
    # Handle primitives
    if schema_type == 'string':
        return 'string'
    elif schema_type in ('integer', 'number'):
        return 'number'
    elif schema_type == 'boolean':
        return 'boolean'
    elif schema_type == 'null':
        return 'null'
    
    # Handle arrays
    elif schema_type == 'array':
        items = schema.get('items', {})
        if not items:
            return 'any[]'
        
        item_type = _schema_to_typescript(items, indent, inline=True)
        
        # For complex object arrays, format nicely
        if items.get('type') == 'object' and items.get('properties'):
            return f'Array<{item_type}>'
        
        return f'{item_type}[]'
    
    # Handle objects (including nested)
    elif schema_type == 'object':
        properties = schema.get('properties', {})
        required = set(schema.get('required', []))
        
        if not properties:
            # Generic object without defined properties
            additional = schema.get('additionalProperties')
            if additional and isinstance(additional, dict):
                value_type = _schema_to_typescript(additional, indent, inline=True)
                return f'{{ [key: string]: {value_type} }}'
            return 'object'
        
        # Build object type with properties
        ind = '  ' * indent
        inner_ind = '  ' * (indent + 1)
        
        lines: List[str] = ['{']
        
        for prop_name, prop_schema in properties.items():
            prop_desc = prop_schema.get('description', '')
            optional = '?' if prop_name not in required else ''
            prop_type = _schema_to_typescript(prop_schema, indent + 1, inline=True)
            
            # Add description as comment
            if prop_desc and not inline:
                lines.append(f'{inner_ind}// {prop_desc}')
            
            lines.append(f'{inner_ind}{prop_name}{optional}: {prop_type},')
        
        lines.append(f'{ind}}}')
        
        if inline and len(properties) <= 2:
            # Compact inline format for simple objects
            props = []
            for prop_name, prop_schema in properties.items():
                optional = '?' if prop_name not in required else ''
                prop_type = _schema_to_typescript(prop_schema, 0, inline=True)
                props.append(f'{prop_name}{optional}: {prop_type}')
            return '{ ' + ', '.join(props) + ' }'
        
        return '\n'.join(lines)
    
    # Handle union types (anyOf, oneOf)
    elif 'anyOf' in schema or 'oneOf' in schema:
        variants = schema.get('anyOf') or schema.get('oneOf', [])
        types = [_schema_to_typescript(v, indent, inline=True) for v in variants]
        return ' | '.join(types)
    
    # Handle allOf (intersection)
    elif 'allOf' in schema:
        variants = schema.get('allOf', [])
        types = [_schema_to_typescript(v, indent, inline=True) for v in variants]
        return ' & '.join(types)
    
    return 'any'


def _format_function_typescript(
    func: Dict[str, Any],
    indent: int = 0,
) -> str:
    """Format a single function as TypeScript type definition."""
    name = func.get('name', 'unknown')
    description = func.get('description', '')
    params = func.get('parameters', {})
    properties = params.get('properties', {})
    required = set(params.get('required', []))
    
    ind = '  ' * indent
    inner_ind = '  ' * (indent + 1)
    
    lines: List[str] = []
    
    # Function description comment
    if description:
        lines.append(f'{ind}// {description}')
    
    # Function type definition
    lines.append(f'{ind}type {name} = (_: {{')
    
    # Parameters
    for param_name, param_schema in properties.items():
        param_desc = param_schema.get('description', '')
        optional = '?' if param_name not in required else ''
        param_type = _schema_to_typescript(param_schema, indent + 1, inline=False)
        
        # Handle multi-line types (objects)
        if '\n' in param_type:
            # Multi-line object type
            if param_desc:
                lines.append(f'{inner_ind}// {param_desc}')
            
            type_lines = param_type.split('\n')
            lines.append(f'{inner_ind}{param_name}{optional}: {type_lines[0]}')
            for type_line in type_lines[1:]:
                lines.append(f'{inner_ind}{type_line}')
            # Replace last line's closing brace with comma
            if lines[-1].strip() == '}':
                lines[-1] = lines[-1].rstrip('}') + '},'
            else:
                lines[-1] = lines[-1] + ','
        else:
            # Single-line type
            if param_desc:
                lines.append(f'{inner_ind}// {param_desc}')
            lines.append(f'{inner_ind}{param_name}{optional}: {param_type},')
    
    lines.append(f'{ind}}}) => any;')
    
    return '\n'.join(lines)


def tools_to_typescript_namespace(tools: List[Dict[str, Any]]) -> str:
    """
    Convert OpenAI-style tools to TypeScript namespace format.
    
    Handles all JSON Schema types:
    - Primitives: string, number, integer, boolean, null
    - Enums: string enums, number enums
    - Arrays: primitive[], object[], nested arrays
    - Objects: nested objects (5-6 levels deep)
    - Union types: anyOf, oneOf
    - Intersection types: allOf
    - Required vs optional (? suffix)
    - Descriptions as // comments
    - additionalProperties for dynamic objects
    
    Example output:
    ```
    namespace functions {
    
    // Get the current weather for a location
    type get_weather = (_: {
      // The city and state, e.g. San Francisco, CA
      location: string,
      // Temperature unit
      unit?: "celsius" | "fahrenheit",
    }) => any;
    
    // Complex nested example
    type create_order = (_: {
      // Customer information
      customer: {
        name: string,
        address: {
          street: string,
          city: string,
          country: string,
        },
      },
      // List of items to order
      items: Array<{
        product_id: string,
        quantity: number,
        options?: { [key: string]: string },
      }>,
    }) => any;
    
    } // namespace functions
    ```
    """
    functions = extract_functions_from_tools(tools)
    
    lines: List[str] = ['namespace functions {', '']
    
    for func in functions:
        func_ts = _format_function_typescript(func, indent=0)
        lines.append(func_ts)
        lines.append('')
    
    lines.append('} // namespace functions')
    
    return '\n'.join(lines)


# =============================================================================
# PROMPT BUILDER
# =============================================================================

def apply_chat_template(
    messages: List[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]] = None,
    response_format: Optional[Dict[str, Any]] = None,
    add_generation_prompt: bool = False,
    json_mode: bool = False,
    enable_reasoning: bool = False,
) -> str:
    """
    Apply USF Omega chat template with OpenAI-style inputs.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        tools: OpenAI-style tools list (optional)
        response_format: OpenAI-style response format (optional)
        add_generation_prompt: Add assistant generation prompt
        json_mode: Enable JSON constrained output
        enable_reasoning: Enable reasoning mode
    
    Returns:
        Formatted prompt string
    """
    parts: List[str] = []
    
    has_tools = tools is not None and len(tools) > 0
    has_response_format = response_format is not None
    
    # Extract functions and schema
    function_list = extract_functions_from_tools(tools) if has_tools else []
    output_schema = extract_schema_from_response_format(response_format) if has_response_format else None
    
    # Build tool_call_id to function name mapping
    tool_call_map = build_tool_call_id_map(messages)
    
    # =========================================================================
    # HEADER SECTION (fixed order: system, developer, functions, response_format)
    # =========================================================================
    
    # 1. System messages
    for msg in messages:
        if msg.get('role') == 'system':
            parts.append(_format_block('system', msg['content']))
    
    # 2. Developer messages with auto-appended instructions
    developer_parts: List[str] = []
    for msg in messages:
        if msg.get('role') == 'developer':
            developer_parts.append(msg['content'])
    
    if has_tools:
        developer_parts.append(FUNCTION_INSTRUCTIONS)
    
    if has_response_format:
        developer_parts.append(RESPONSE_FORMAT_INSTRUCTIONS)
    
    if developer_parts:
        parts.append(_format_block('developer', '\n\n'.join(developer_parts)))
    
    # 3. Functions definition (TypeScript namespace format - more token efficient)
    if has_tools:
        ts_namespace = tools_to_typescript_namespace(tools)
        parts.append(_format_block('functions', ts_namespace))
    
    # 4. Response format schema
    if has_response_format:
        parts.append(_format_block('response_format.json', _json_dumps(output_schema)))
    
    # =========================================================================
    # CONVERSATION SECTION (user, assistant, tool)
    # =========================================================================
    
    for msg in messages:
        role = msg.get('role')
        
        # Skip header roles (already processed)
        if role in ('system', 'developer'):
            continue
        
        # User message
        if role == 'user':
            parts.append(_format_block('user', msg['content']))
        
        # Assistant message
        elif role == 'assistant':
            assistant_parts: List[str] = [f'{START_TOKEN}assistant\n']
            
            content = msg.get('content') or ''
            has_tool_calls = bool(msg.get('tool_calls'))
            has_reasoning = msg.get('reasoning') or (enable_reasoning and msg.get('thinking'))
            
            # Detect if content matches schema (auto-add constrain for historical JSON outputs)
            content_is_json = _content_matches_schema(content, response_format) if response_format else False
            use_constrain = msg.get('constrain') or content_is_json
            
            # Constrained output (explicit or auto-detected from schema match)
            if use_constrain:
                constrain_type = msg.get('constrain', 'json')
                assistant_parts.append(f'{CONSTRAIN_TOKEN}{constrain_type}')
            
            # Reasoning
            if msg.get('reasoning'):
                assistant_parts.append(f'{REASONING_START}\n{msg["reasoning"]}\n{REASONING_END}')
            elif enable_reasoning and msg.get('thinking'):
                assistant_parts.append(f'{REASONING_START}\n{msg["thinking"]}\n{REASONING_END}')
            
            # Message content
            assistant_parts.append(f'{MESSAGE_START}{content}{MESSAGE_END}')
            
            # Tool calls
            if has_tool_calls:
                assistant_parts.append(FUNCTIONS_START)
                for tc in msg['tool_calls']:
                    func = tc.get('function', {})
                    func_name = func.get('name', 'unknown')
                    args = func.get('arguments', {})
                    if isinstance(args, str):
                        args = json.loads(args)
                    
                    assistant_parts.append(f'{INVOKE_START}to="function.{func_name}"')
                    for k, v in args.items():
                        if isinstance(v, (dict, list)):
                            v = json.dumps(v, ensure_ascii=False)
                        assistant_parts.append(f'<|parameter name="{k}"|>{v}<||parameter||>')
                    assistant_parts.append(INVOKE_END)
                assistant_parts.append(FUNCTIONS_END)
            
            # Close constrain token if used
            if use_constrain:
                assistant_parts.append(CONSTRAIN_END)
            
            assistant_parts.append(f'{END_TOKEN}\n')
            parts.append(''.join(assistant_parts))
        
        # Tool response
        elif role == 'tool':
            tool_parts: List[str] = [f'{START_TOKEN}tool\n']
            
            # Add function_results start token
            tool_parts.append(FUNCTION_RESULTS_START)
            
            # Get function name from tool_call_id mapping or message
            func_name = 'unknown'
            if msg.get('tool_call_id') and msg['tool_call_id'] in tool_call_map:
                func_name = tool_call_map[msg['tool_call_id']]
            elif msg.get('name'):
                func_name = msg['name']
            
            content = msg.get('content', '')
            tool_parts.append(f'<||function_results_start to=function.{func_name}||>{content}<||function_response_end||>')
            tool_parts.append(f'{FUNCTION_RESULTS_END}{END_TOKEN}\n')
            parts.append(''.join(tool_parts))
    
    # =========================================================================
    # GENERATION PROMPT
    # =========================================================================
    
    if add_generation_prompt:
        gen_parts: List[str] = [f'{START_TOKEN}assistant\n']
        if json_mode:
            gen_parts.append(f'{CONSTRAIN_TOKEN}json')
            if enable_reasoning:
                # json + reasoning: constrain -> reasoning_start
                gen_parts.append(REASONING_START)
            else:
                # json only: constrain -> message_start
                gen_parts.append(MESSAGE_START)
        elif enable_reasoning:
            # reasoning only (no json): reasoning_start
            gen_parts.append(REASONING_START)
        else:
            # plain (no json, no reasoning): message_start
            gen_parts.append(MESSAGE_START)
        parts.append(''.join(gen_parts))
    
    return ''.join(parts)


def test_openai_style_conversation():
    """Test with OpenAI-style tools, response_format, and multi-turn conversation."""
    
    # Define tools (function calling) - OpenAI format
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        },
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                            "description": "Temperature unit"
                        }
                    },
                    "required": ["location"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "search_database",
                "description": "Search the product database",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search query"
                        },
                        "category": {
                            "type": "string",
                            "enum": ["electronics", "clothing", "food"],
                            "description": "Product category filter"
                        },
                        "max_results": {
                            "type": "integer",
                            "description": "Maximum number of results to return"
                        }
                    },
                    "required": ["query"]
                }
            }
        }
    ]

    # Structured output schema (response format) - OpenAI format
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "weather_response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "temperature": {"type": "number"},
                    "unit": {"type": "string"},
                    "conditions": {"type": "string"},
                    "humidity": {"type": "number"},
                    "wind_speed": {"type": "number"}
                },
                "required": ["temperature", "unit", "conditions"]
            }
        }
    }

    # Multi-turn conversation with developer/system, user, assistant, and tool messages
    messages = [
        # System message
        {
            "role": "system",
            "content": "Additional context: Be concise and accurate with weather data."
        },
        # Developer message
        {
            "role": "developer",
            "content": "You are a helpful weather assistant. Always use the get_weather tool when asked about weather. Respond with structured JSON when appropriate."
        },
        # User turn 1
        {
            "role": "user",
            "content": "What's the weather like in Tokyo?"
        },
        # Assistant calls a tool
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_abc123",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Tokyo, Japan", "unit": "celsius"}'
                    }
                }
            ]
        },
        # Tool response
        {
            "role": "tool",
            "tool_call_id": "call_abc123",
            "content": '{"temperature": 22, "unit": "celsius", "conditions": "partly cloudy", "humidity": 65, "wind_speed": 12}'
        },
        # Assistant response after tool
        {
            "role": "assistant",
            "content": "The weather in Tokyo is currently 22°C and partly cloudy. Humidity is at 65% with wind speeds of 12 km/h."
        },
        # User turn 2
        {
            "role": "user",
            "content": "What about New York? Give me the response in JSON format."
        }
    ]

    # Apply template with tools, response_format, and generation prompt
    result = apply_chat_template(
        messages=messages,
        tools=tools,
        response_format=response_format,
        add_generation_prompt=True,
        json_mode=True,
        enable_reasoning=False
    )
    
    return result, tools, response_format, messages


def test_parallel_tool_calls():
    """Test parallel tool calls."""
    
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string"}
                    },
                    "required": ["city"]
                }
            }
        }
    ]
    
    messages = [
        {"role": "system", "content": "You are a weather assistant."},
        {"role": "user", "content": "What's the weather in Tokyo and Paris?"},
        {
            "role": "assistant",
            "content": "Let me check both cities.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Tokyo"}'
                    }
                },
                {
                    "id": "call_2",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Paris"}'
                    }
                }
            ]
        },
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": '{"temperature": 22}'
        },
        {
            "role": "tool",
            "tool_call_id": "call_2",
            "content": '{"temperature": 18}'
        },
        {"role": "user", "content": "Thanks!"}
    ]
    
    result = apply_chat_template(
        messages=messages,
        tools=tools,
        add_generation_prompt=True
    )
    
    return result


def test_structured_output_only():
    """Test structured output without tools."""
    
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "person_info",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                    "occupation": {"type": "string"}
                },
                "required": ["name", "age"]
            }
        }
    }
    
    messages = [
        {"role": "system", "content": "You are a data extractor."},
        {"role": "user", "content": "John Smith is a 35-year-old software engineer."}
    ]
    
    result = apply_chat_template(
        messages=messages,
        response_format=response_format,
        add_generation_prompt=True,
        json_mode=True
    )
    
    return result


def validate_prompt(prompt: str) -> dict:
    """Validate the generated prompt structure."""
    errors = []
    warnings = []
    
    start_token = '<|:@:|start|:@:|>'
    end_token = '<|:@::|end|:@::|>'
    
    # Check basic structure
    if not prompt.startswith(start_token):
        errors.append("Prompt does not start with start token")
    
    if end_token not in prompt:
        errors.append("Prompt does not contain end token")
    
    # Check header order
    roles = re.findall(r'<\|:@:\|start\|:@:\|>([^\n]+)\n', prompt)
    
    header_roles = ['system', 'developer', 'functions', 'response_format.json']
    first_user_idx = roles.index('user') if 'user' in roles else len(roles)
    headers_found = [r for r in roles[:first_user_idx] if r in header_roles]
    
    expected_order = [r for r in header_roles if r in headers_found]
    if headers_found != expected_order:
        errors.append(f"Header order incorrect. Found: {headers_found}, Expected: {expected_order}")
    
    # Check for developer instructions
    if 'functions' in roles and '# Functions' not in prompt:
        warnings.append("Functions present but developer instructions may be missing")
    
    if 'response_format.json' in roles and '# Response Format' not in prompt:
        warnings.append("Response format present but developer instructions may be missing")
    
    return {
        'valid': len(errors) == 0,
        'errors': errors,
        'warnings': warnings,
        'roles': roles
    }


def main():
    print("=" * 80)
    print("TEST 1: OpenAI-style conversation with tools + response_format")
    print("=" * 80)
    
    result, tools, response_format, messages = test_openai_style_conversation()
    print(result)
    print("\n" + "-" * 40)
    validation = validate_prompt(result)
    print(f"Valid: {validation['valid']}")
    print(f"Roles found: {validation['roles']}")
    if validation['errors']:
        print(f"Errors: {validation['errors']}")
    if validation['warnings']:
        print(f"Warnings: {validation['warnings']}")
    
    print("\n" + "=" * 80)
    print("TEST 2: Parallel tool calls")
    print("=" * 80)
    
    result2 = test_parallel_tool_calls()
    print(result2)
    print("\n" + "-" * 40)
    validation2 = validate_prompt(result2)
    print(f"Valid: {validation2['valid']}")
    print(f"Roles found: {validation2['roles']}")
    
    print("\n" + "=" * 80)
    print("TEST 3: Structured output only (no tools)")
    print("=" * 80)
    
    result3 = test_structured_output_only()
    print(result3)
    print("\n" + "-" * 40)
    validation3 = validate_prompt(result3)
    print(f"Valid: {validation3['valid']}")
    print(f"Roles found: {validation3['roles']}")
    
    # Save outputs
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)
    
    from datetime import datetime, timezone
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    
    output_file = output_dir / f"openai_template_test_{timestamp}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "test_1_openai_conversation": {
                "prompt": result,
                "validation": validation,
                "tools": tools,
                "response_format": response_format,
                "messages": messages
            },
            "test_2_parallel_calls": {
                "prompt": result2,
                "validation": validation2
            },
            "test_3_structured_only": {
                "prompt": result3,
                "validation": validation3
            }
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n\nSaved outputs to: {output_file}")
    
    # Summary
    all_valid = validation['valid'] and validation2['valid'] and validation3['valid']
    print("\n" + "=" * 80)
    print(f"SUMMARY: {'ALL TESTS PASSED' if all_valid else 'SOME TESTS FAILED'}")
    print("=" * 80)
    
    return 0 if all_valid else 1


if __name__ == "__main__":
    raise SystemExit(main())
