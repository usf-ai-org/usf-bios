# Copyright (c) Ultrasafe AI. All rights reserved.
"""
USF Omega Agent Template for function/tool calling.

This module provides comprehensive parsing and formatting for USF Omega's unique formats:

Exports:
    - UsfOmegaAgentTemplate: Main agent template class
    - ParsedContentItem: Single item in contents array (type-based)
    - ParsedAssistantResponse: Complete parsed response with sequential structure
    - Token constants: All USF Omega special tokens

1. Function calls:
   <|:@:|functions_start|:@:|>
   <||invoke_start to=function.NAME||>
   <||parameter_start name=PARAM||>VALUE<||parameter_end||>
   <||invoke_end||>
   <|:@:|functions_end|:@:|>

2. Tool responses:
   <|:@:|function_results_start|:@:|>
   <||function_response_start to=function.NAME id="ID"||>CONTENT<||function_response_end||>
   <|:@:|function_results_end|:@:|>

3. Message content:
   <|:@:|message_start|:@:|>content<|:@:|message_end|:@:|>

4. Reasoning blocks:
   <|:@:|reasoning_start|:@:|>thinking<|:@:|reasoning_end|:@:|>

5. Self-reflection:
   <|:@:|self_reflection_start|:@:|>reflection<|:@:|self_reflection_end|:@:|>

6. Citations:
   <||citations_start ref=["ID"]||>cited content<||citations_end||>

7. Constrain tokens:
   <|:@:|constrain_start|:@:|>response.format={...} reasoning={...}<|:@:|constrain_end|:@:|>
"""
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import json

from usf_bios.infer_engine import Function
from usf_bios.template import Prompt
from .base import BaseAgentTemplate


__all__ = [
    # Main classes
    'UsfOmegaAgentTemplate',
    'ParsedContentItem',
    'ParsedAssistantResponse',
    # Token constants
    'START_TOKEN', 'END_TOKEN', 'BOS_TOKEN', 'EOS_TOKEN',
    'MESSAGE_START', 'MESSAGE_END',
    'CONSTRAIN_START', 'CONSTRAIN_END',
    'REASONING_START', 'REASONING_END',
    'SELF_REFLECTION_START', 'SELF_REFLECTION_END',
    'FUNCTIONS_START', 'FUNCTIONS_END',
    'INVOKE_START', 'INVOKE_END',
    'PARAMETER_START', 'PARAMETER_END',
    'FUNCTION_RESULTS_START', 'FUNCTION_RESULTS_END',
    'FUNCTION_RESPONSE_START', 'FUNCTION_RESPONSE_END',
    'CITATIONS_START', 'CITATIONS_END',
    'SEARCH_START', 'SEARCH_END',
    'WAIT_TOKEN',
]


# ==================== USF OMEGA SPECIAL TOKENS ====================

# Message boundary tokens
START_TOKEN = '<|:@:|start|:@:|>'
END_TOKEN = '<|:@:|end|:@:|>'
BOS_TOKEN = '<|:@:|startoftext|:@:|>'
EOS_TOKEN = '<|:@:|endoftext|:@:|>'

# Message content tokens
MESSAGE_START = '<|:@:|message_start|:@:|>'
MESSAGE_END = '<|:@:|message_end|:@:|>'

# Constrain tokens
CONSTRAIN_START = '<|:@:|constrain_start|:@:|>'
CONSTRAIN_END = '<|:@:|constrain_end|:@:|>'

# Reasoning tokens
REASONING_START = '<|:@:|reasoning_start|:@:|>'
REASONING_END = '<|:@:|reasoning_end|:@:|>'

# Self-reflection tokens
SELF_REFLECTION_START = '<|:@:|self_reflection_start|:@:|>'
SELF_REFLECTION_END = '<|:@:|self_reflection_end|:@:|>'

# Function/Tool call tokens
FUNCTIONS_START = '<|:@:|functions_start|:@:|>'
FUNCTIONS_END = '<|:@:|functions_end|:@:|>'
INVOKE_START = '<||invoke_start to='
INVOKE_END = '<||invoke_end||>'
PARAMETER_START = '<||parameter_start name='
PARAMETER_END = '<||parameter_end||>'

# Tool response tokens
FUNCTION_RESULTS_START = '<|:@:|function_results_start|:@:|>'
FUNCTION_RESULTS_END = '<|:@:|function_results_end|:@:|>'
FUNCTION_RESPONSE_START = '<||function_response_start to='
FUNCTION_RESPONSE_END = '<||function_response_end||>'

# Citation tokens
CITATIONS_START = '<||citations_start ref='
CITATIONS_END = '<||citations_end||>'

# Search tokens
SEARCH_START = '<|:@:|search_start|:@:|>'
SEARCH_END = '<|:@:|search_end|:@:|>'

# Wait token
WAIT_TOKEN = '<|:@:|wait|:@:|>'


@dataclass
class ParsedContentItem:
    """A single item in the contents array (OpenAI-style with type field).
    
    Types:
    - 'reasoning': Optional, once, must be first. Contains thinking/reasoning content.
    - 'text': Message text content.
    - 'tool_calls': Function/tool calls with id, name, arguments.
    - 'self_reflection': Self-reflection content.
    
    Valid sequences:
    - reasoning → text → tool_calls → self_reflection → text → tool_calls → self_reflection
    - text → self_reflection
    - text → tool_calls → self_reflection → text → self_reflection
    """
    type: str  # 'reasoning', 'text', 'tool_calls', 'self_reflection'
    content: Any  # str for reasoning/text/self_reflection, List[Dict] for tool_calls
    position: int = 0  # Position in the sequence
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to OpenAI-style contents item dict."""
        if self.type == 'reasoning':
            return {'type': 'reasoning', 'reasoning': self.content}
        elif self.type == 'text':
            return {'type': 'text', 'text': self.content}
        elif self.type == 'tool_calls':
            return {'type': 'tool_calls', 'tool_calls': self.content}
        elif self.type == 'self_reflection':
            return {'type': 'self_reflection', 'self_reflection': self.content}
        else:
            return {'type': self.type, 'content': self.content}


@dataclass
class ParsedAssistantResponse:
    """Parsed components of a USF Omega assistant response.
    
    Supports both simple format and OpenAI-style contents array format.
    
    Simple format:
        {"role": "assistant", "content": "...", "tool_calls": [...], "self_reflection": "..."}
    
    Contents array format (preserves sequential structure):
        {"role": "assistant", "contents": [
            {"type": "reasoning", "reasoning": "..."},
            {"type": "text", "text": "..."},
            {"type": "tool_calls", "tool_calls": [...]},
            {"type": "self_reflection", "self_reflection": "..."},
            {"type": "text", "text": "..."},  # Can repeat
            {"type": "self_reflection", "self_reflection": "..."}  # Must end with this
        ]}
    """
    # Sequential contents (preserves order for multi-iteration responses)
    contents: List[ParsedContentItem] = None
    
    # Legacy flat access (for backward compatibility)
    message_contents: List[str] = None  # All text blocks
    function_calls: List[Dict[str, Any]] = None  # All function calls (flattened)
    reasoning: Optional[str] = None  # Reasoning block (once, first)
    self_reflections: List[str] = None  # All self-reflection blocks
    citations: List[Dict[str, Any]] = None  # All citations
    constrain: Optional[str] = None  # Constrain block content
    raw_text: str = ''  # Original text
    
    def __post_init__(self):
        if self.contents is None:
            self.contents = []
        if self.message_contents is None:
            self.message_contents = []
        if self.function_calls is None:
            self.function_calls = []
        if self.self_reflections is None:
            self.self_reflections = []
        if self.citations is None:
            self.citations = []
    
    @property
    def has_function_calls(self) -> bool:
        return len(self.function_calls) > 0
    
    @property
    def has_reasoning(self) -> bool:
        return self.reasoning is not None and len(self.reasoning) > 0
    
    @property
    def has_self_reflection(self) -> bool:
        return len(self.self_reflections) > 0
    
    @property
    def final_message(self) -> str:
        """Get the final message content (last text block)."""
        return self.message_contents[-1] if self.message_contents else ''
    
    @property
    def has_multiple_iterations(self) -> bool:
        """Check if response has multiple text→reflection cycles."""
        text_count = sum(1 for c in self.contents if c.type == 'text')
        return text_count > 1
    
    @property
    def tool_call_blocks(self) -> List[List[Dict[str, Any]]]:
        """Get tool calls grouped by position (for multi-iteration responses)."""
        return [c.content for c in self.contents if c.type == 'tool_calls']
    
    def to_contents_array(self) -> List[Dict[str, Any]]:
        """Convert to OpenAI-style contents array."""
        return [c.to_dict() for c in self.contents]
    
    def to_simple_format(self) -> Dict[str, Any]:
        """Convert to simple format (first text, all tool_calls flattened, last reflection)."""
        result = {'role': 'assistant'}
        if self.message_contents:
            result['content'] = self.message_contents[0]
        if self.function_calls:
            result['tool_calls'] = self.function_calls
        if self.self_reflections:
            result['self_reflection'] = self.self_reflections[-1]
        if self.reasoning:
            result['reasoning'] = self.reasoning
        return result


class UsfOmegaAgentTemplate(BaseAgentTemplate):
    """Agent template for USF Omega function calling format.
    
    This template handles all USF Omega specific features:
    - Function calls (single and parallel)
    - Tool responses
    - Message content extraction
    - Reasoning block parsing
    - Self-reflection parsing
    - Citation parsing
    - Constrain token parsing
    
    The jinja2 template handles the actual formatting during encoding.
    This agent template handles parsing model outputs and formatting
    for the usf_bios backend.
    """

    def get_toolcall(self, response: str) -> List[Function]:
        """Parse function calls from model response.
        
        Args:
            response: The model's response text.
            
        Returns:
            List of Function objects representing the tool calls.
        """
        functions = []
        
        # Find all function call blocks
        # Pattern: <||invoke_start to=function.NAME||>...<||invoke_end||>
        invoke_pattern = r'<\|\|invoke_start to=function\.([^|]+)\|\|>(.*?)<\|\|invoke_end\|\|>'
        invoke_matches = re.findall(invoke_pattern, response, re.DOTALL)
        
        for func_name, invoke_content in invoke_matches:
            func_name = func_name.strip()
            
            # Parse parameters from invoke content
            # Pattern: <||parameter_start name=PARAM_NAME||>VALUE<||parameter_end||>
            param_pattern = r'<\|\|parameter_start name=([^|]+)\|\|>(.*?)<\|\|parameter_end\|\|>'
            param_matches = re.findall(param_pattern, invoke_content, re.DOTALL)
            
            arguments = {}
            for param_name, param_value in param_matches:
                param_name = param_name.strip()
                param_value = param_value.strip()
                
                # Try to parse JSON values
                parsed_value = self._parse_json(param_value)
                if parsed_value is not None:
                    arguments[param_name] = parsed_value
                else:
                    arguments[param_name] = param_value
            
            functions.append(Function(name=func_name, arguments=arguments))
        
        # Fallback to ReAct format if no USF Omega format found
        if len(functions) == 0:
            return super().get_toolcall(response)
        
        return functions

    def _format_tool_call(self, name: str, arguments: Dict[str, Any], call_id: Optional[str] = None) -> str:
        """Format a single function call in USF Omega format.
        
        Args:
            name: Function name.
            arguments: Function arguments as dict.
            call_id: Optional call ID for tracking.
            
        Returns:
            Formatted function call string.
        """
        params = []
        for param_name, param_value in arguments.items():
            if isinstance(param_value, (dict, list)):
                param_value = json.dumps(param_value, ensure_ascii=False)
            else:
                param_value = str(param_value)
            params.append(f'{PARAMETER_START}{param_name}||>{param_value}{PARAMETER_END}')
        
        params_str = ''.join(params)
        return f'{INVOKE_START}function.{name}||>{params_str}{INVOKE_END}'

    def _format_tool_calls(self, tool_call_messages) -> str:
        """Format multiple tool call messages into USF Omega format.
        
        Args:
            tool_call_messages: List of tool call message dicts.
            
        Returns:
            Formatted string with all function calls.
        """
        calls = []
        for message in tool_call_messages:
            tool_call = self._parse_tool_call(message['content'])
            name = tool_call['name']
            arguments = tool_call['arguments']
            calls.append(self._format_tool_call(name, arguments))
        
        return f'{MESSAGE_START}{MESSAGE_END}{FUNCTIONS_START}{"".join(calls)}{FUNCTIONS_END}'

    def _format_tool_response(self, name: str, content: str, call_id: Optional[str] = None) -> str:
        """Format a single tool response in USF Omega format.
        
        Args:
            name: Function name.
            content: Response content.
            call_id: Optional call ID for tracking.
            
        Returns:
            Formatted tool response string.
        """
        id_part = f' id="{call_id}"' if call_id else ''
        return f'{FUNCTION_RESPONSE_START}function.{name}{id_part}||>{content}{FUNCTION_RESPONSE_END}'

    def _format_tool_responses(
        self,
        assistant_content: str,
        tool_messages: List[Dict[str, Any]],
    ) -> Tuple[str, Prompt]:
        """Format tool execution results into the conversation.
        
        Args:
            assistant_content: The assistant's message containing tool calls.
            tool_messages: List of tool execution result messages.
            
        Returns:
            Tuple of (formatted assistant content, formatted tool responses as list).
        """
        # Check if using ReAct format
        with_action = self.keyword.action in assistant_content and self.keyword.action_input in assistant_content
        if with_action:
            return super()._format_tool_responses(assistant_content, tool_messages)
        
        # Format in USF Omega format
        responses = []
        for i, tool_message in enumerate(tool_messages):
            name = tool_message.get('name', 'unknown')
            content = tool_message.get('content', '')
            call_id = tool_message.get('tool_call_id') or tool_message.get('id') or f'r{i+1}'
            responses.append(self._format_tool_response(name, content, call_id))
        
        tool_content = f'{FUNCTION_RESULTS_START}{"".join(responses)}{FUNCTION_RESULTS_END}'
        
        return assistant_content, [tool_content]

    def _format_tools(
        self,
        tools: List[Union[str, dict]],
        system: Optional[str] = None,
        user_message: Optional[dict] = None
    ) -> str:
        """Format tools for inclusion in the prompt.
        
        Note: For USF Omega, tools are primarily formatted by the jinja2 template
        which creates the namespace format. This method provides fallback formatting
        for the usf_bios backend when jinja2 is not used.
        
        Args:
            tools: List of tool definitions.
            system: System prompt text.
            user_message: Optional user message.
            
        Returns:
            System prompt with tool definitions appended.
        """
        if not tools:
            return system or ''
        
        # Format tools in TypeScript namespace format (matching jinja2 output)
        tool_defs = []
        for tool in tools:
            tool = self.unwrap_tool(tool)
            name = self._get_tool_name(tool)
            description = tool.get('description', '')
            parameters = tool.get('parameters', {})
            
            # Build TypeScript-like parameter definition
            params_str = self._format_parameters_typescript(parameters)
            
            if description:
                tool_defs.append(f'// {description}')
            tool_defs.append(f'type {name} = (_: {{{params_str}}}) => any;')
            tool_defs.append('')
        
        namespace_content = '\n'.join(tool_defs)
        tools_block = f'namespace functions {{\n{namespace_content}}} // namespace functions'
        
        # Combine with system message
        if system:
            return f'{system}\n\n{tools_block}'
        return tools_block
    
    def _format_parameters_typescript(self, parameters: Dict[str, Any]) -> str:
        """Format parameters in TypeScript-like syntax.
        
        Args:
            parameters: JSON Schema parameters object.
            
        Returns:
            TypeScript-like parameter string.
        """
        if not parameters:
            return ''
        
        props = parameters.get('properties', {})
        required = set(parameters.get('required', []))
        
        parts = []
        for name, schema in props.items():
            description = schema.get('description', '')
            prop_type = self._json_type_to_typescript(schema)
            optional = '?' if name not in required else ''
            
            if description:
                parts.append(f'  // {description}')
            parts.append(f'  {name}{optional}: {prop_type},')
        
        return '\n'.join(parts)
    
    def _json_type_to_typescript(self, schema: Dict[str, Any]) -> str:
        """Convert JSON Schema type to TypeScript type."""
        json_type = schema.get('type', 'any')
        
        if json_type == 'string':
            if 'enum' in schema:
                return ' | '.join(f'"{v}"' for v in schema['enum'])
            return 'string'
        elif json_type == 'number' or json_type == 'integer':
            return 'number'
        elif json_type == 'boolean':
            return 'boolean'
        elif json_type == 'array':
            items = schema.get('items', {})
            item_type = self._json_type_to_typescript(items)
            return f'{item_type}[]'
        elif json_type == 'object':
            props = schema.get('properties', {})
            if props:
                prop_parts = []
                for name, prop_schema in props.items():
                    prop_type = self._json_type_to_typescript(prop_schema)
                    prop_parts.append(f'{name}: {prop_type}')
                return '{' + ', '.join(prop_parts) + '}'
            return 'object'
        else:
            return 'any'

    @staticmethod
    def parse_function_calls_from_text(text: str) -> List[Dict[str, Any]]:
        """Static utility to parse function calls from text.
        
        Args:
            text: Text containing USF Omega function calls.
            
        Returns:
            List of dicts with 'name' and 'arguments' keys.
        """
        template = UsfOmegaAgentTemplate()
        functions = template.get_toolcall(text)
        return [{'name': f.name, 'arguments': f.arguments} for f in functions]

    @staticmethod
    def parse_tool_responses_from_text(text: str) -> List[Dict[str, Any]]:
        """Static utility to parse tool responses from text.
        
        Args:
            text: Text containing USF Omega tool responses.
            
        Returns:
            List of dicts with 'name', 'id', and 'content' keys.
        """
        results = []
        
        # Pattern: <||function_response_start to=function.NAME id="ID"||>CONTENT<||function_response_end||>
        pattern = r'<\|\|function_response_start to=function\.([^|\s]+)(?:\s+id="([^"]*)")?\|\|>(.*?)<\|\|function_response_end\|\|>'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for name, call_id, content in matches:
            results.append({
                'name': name.strip(),
                'id': call_id.strip() if call_id else None,
                'content': content.strip()
            })
        
        return results

    # ==================== MESSAGE CONTENT PARSING ====================
    
    @staticmethod
    def parse_message_contents(text: str) -> List[str]:
        """Extract all message content blocks from text.
        
        Args:
            text: Text containing USF Omega message blocks.
            
        Returns:
            List of message contents (can be multiple in one turn).
        """
        pattern = r'<\|:@:\|message_start\|:@:\|>(.*?)<\|:@:\|message_end\|:@:\|>'
        matches = re.findall(pattern, text, re.DOTALL)
        return [m.strip() for m in matches]
    
    @staticmethod
    def parse_reasoning(text: str) -> Optional[str]:
        """Extract reasoning block content from text.
        
        Args:
            text: Text containing USF Omega reasoning block.
            
        Returns:
            Reasoning content or None if not present.
        """
        pattern = r'<\|:@:\|reasoning_start\|:@:\|>(.*?)<\|:@:\|reasoning_end\|:@:\|>'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None
    
    @staticmethod
    def parse_self_reflections(text: str) -> List[str]:
        """Extract all self-reflection blocks from text.
        
        Args:
            text: Text containing USF Omega self-reflection blocks.
            
        Returns:
            List of self-reflection contents.
        """
        pattern = r'<\|:@:\|self_reflection_start\|:@:\|>(.*?)<\|:@:\|self_reflection_end\|:@:\|>'
        matches = re.findall(pattern, text, re.DOTALL)
        return [m.strip() for m in matches]
    
    @staticmethod
    def parse_citations(text: str) -> List[Dict[str, Any]]:
        """Extract all citations from text.
        
        Args:
            text: Text containing USF Omega citations.
            
        Returns:
            List of dicts with 'refs' (list of IDs) and 'content' keys.
        """
        results = []
        # Pattern: <||citations_start ref=["ID1","ID2"]||>content<||citations_end||>
        pattern = r'<\|\|citations_start ref=\[([^\]]+)\]\|\|>(.*?)<\|\|citations_end\|\|>'
        matches = re.findall(pattern, text, re.DOTALL)
        
        for refs_str, content in matches:
            # Parse refs like "s001","s002" or 's001','s002'
            refs = re.findall(r'["\']([^"\']*)["\'']', refs_str)
            results.append({
                'refs': refs,
                'content': content.strip()
            })
        
        return results
    
    @staticmethod
    def parse_constrain(text: str) -> Optional[str]:
        """Extract constrain block content from text.
        
        Args:
            text: Text containing USF Omega constrain block.
            
        Returns:
            Constrain content or None if not present.
        """
        pattern = r'<\|:@:\|constrain_start\|:@:\|>(.*?)<\|:@:\|constrain_end\|:@:\|>'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None
    
    def parse_assistant_response(self, text: str) -> ParsedAssistantResponse:
        """Parse a complete assistant response into its components.
        
        Handles both simple and multi-iteration formats:
        - Simple: message → tool_calls → self_reflection
        - Multi-iteration: reasoning → text → tool_calls → self_reflection → text → ... → self_reflection
        
        Args:
            text: The assistant's response text.
            
        Returns:
            ParsedAssistantResponse with all components extracted, preserving sequence.
        """
        # Parse sequential contents (preserves order)
        contents = self.parse_sequential_contents(text)
        
        # Also extract flat components for backward compatibility
        message_contents = self.parse_message_contents(text)
        all_function_calls = self.parse_function_calls_from_text(text)
        reasoning = self.parse_reasoning(text)
        self_reflections = self.parse_self_reflections(text)
        citations = self.parse_citations(text)
        constrain = self.parse_constrain(text)
        
        return ParsedAssistantResponse(
            contents=contents,
            message_contents=message_contents,
            function_calls=all_function_calls,
            reasoning=reasoning,
            self_reflections=self_reflections,
            citations=citations,
            constrain=constrain,
            raw_text=text
        )
    
    def parse_sequential_contents(self, text: str) -> List[ParsedContentItem]:
        """Parse text into sequential contents array preserving order.
        
        This is critical for multi-iteration responses where the order of
        text → tool_calls → self_reflection cycles matters.
        
        Args:
            text: The assistant's response text.
            
        Returns:
            List of ParsedContentItem in order of appearance.
        """
        contents = []
        position = 0
        
        # Define patterns for each block type with their start/end positions
        patterns = {
            'reasoning': (REASONING_START, REASONING_END),
            'text': (MESSAGE_START, MESSAGE_END),
            'tool_calls': (FUNCTIONS_START, FUNCTIONS_END),
            'self_reflection': (SELF_REFLECTION_START, SELF_REFLECTION_END),
        }
        
        # Find all blocks with their positions
        blocks = []
        
        for block_type, (start_token, end_token) in patterns.items():
            idx = 0
            while True:
                start_pos = text.find(start_token, idx)
                if start_pos == -1:
                    break
                end_pos = text.find(end_token, start_pos + len(start_token))
                if end_pos == -1:
                    break
                
                content_start = start_pos + len(start_token)
                content = text[content_start:end_pos]
                
                blocks.append({
                    'type': block_type,
                    'content': content,
                    'start_pos': start_pos,
                    'end_pos': end_pos + len(end_token)
                })
                
                idx = end_pos + len(end_token)
        
        # Sort by position in text
        blocks.sort(key=lambda x: x['start_pos'])
        
        # Convert to ParsedContentItem
        for i, block in enumerate(blocks):
            block_type = block['type']
            content = block['content'].strip()
            
            if block_type == 'tool_calls':
                # Parse function calls from this specific block
                tool_calls = self._parse_tool_calls_from_block(block['content'])
                contents.append(ParsedContentItem(
                    type='tool_calls',
                    content=tool_calls,
                    position=i
                ))
            else:
                contents.append(ParsedContentItem(
                    type=block_type,
                    content=content,
                    position=i
                ))
        
        return contents
    
    def _parse_tool_calls_from_block(self, block_content: str) -> List[Dict[str, Any]]:
        """Parse tool calls from a single functions block.
        
        Args:
            block_content: Content between functions_start and functions_end.
            
        Returns:
            List of tool call dicts with id, name, arguments.
        """
        tool_calls = []
        
        # Pattern: <||invoke_start to=function.NAME||>...<||invoke_end||>
        invoke_pattern = r'<\|\|invoke_start to=function\.([^|]+)\|\|>(.*?)<\|\|invoke_end\|\|>'
        invoke_matches = re.findall(invoke_pattern, block_content, re.DOTALL)
        
        for idx, (func_name, invoke_content) in enumerate(invoke_matches):
            func_name = func_name.strip()
            
            # Parse parameters
            param_pattern = r'<\|\|parameter_start name=([^|]+)\|\|>(.*?)<\|\|parameter_end\|\|>'
            param_matches = re.findall(param_pattern, invoke_content, re.DOTALL)
            
            arguments = {}
            for param_name, param_value in param_matches:
                param_name = param_name.strip()
                param_value = param_value.strip()
                
                # Try to parse JSON values
                parsed_value = self._parse_json(param_value)
                if parsed_value is not None:
                    arguments[param_name] = parsed_value
                else:
                    arguments[param_name] = param_value
            
            tool_calls.append({
                'id': f'call_{idx}',
                'type': 'function',
                'function': {
                    'name': func_name,
                    'arguments': arguments
                }
            })
        
        return tool_calls
    
    # ==================== RESPONSE CHECKING ====================
    
    @staticmethod
    def has_function_calls(text: str) -> bool:
        """Check if text contains function calls."""
        return FUNCTIONS_START in text and INVOKE_START in text
    
    @staticmethod
    def has_tool_results(text: str) -> bool:
        """Check if text contains tool results."""
        return FUNCTION_RESULTS_START in text
    
    @staticmethod
    def has_reasoning(text: str) -> bool:
        """Check if text contains reasoning block."""
        return REASONING_START in text and REASONING_END in text
    
    @staticmethod
    def has_self_reflection(text: str) -> bool:
        """Check if text contains self-reflection block."""
        return SELF_REFLECTION_START in text and SELF_REFLECTION_END in text
    
    @staticmethod
    def has_citations(text: str) -> bool:
        """Check if text contains citations."""
        return CITATIONS_START in text and CITATIONS_END in text
    
    @staticmethod
    def has_constrain(text: str) -> bool:
        """Check if text contains constrain block."""
        return CONSTRAIN_START in text and CONSTRAIN_END in text
    
    # ==================== FORMATTING UTILITIES ====================
    
    @staticmethod
    def format_message_block(content: str) -> str:
        """Format content as a message block."""
        return f'{MESSAGE_START}{content}{MESSAGE_END}'
    
    @staticmethod
    def format_reasoning_block(content: str) -> str:
        """Format content as a reasoning block."""
        return f'{REASONING_START}{content}{REASONING_END}'
    
    @staticmethod
    def format_self_reflection_block(content: str) -> str:
        """Format content as a self-reflection block."""
        return f'{SELF_REFLECTION_START}{content}{SELF_REFLECTION_END}'
    
    @staticmethod
    def format_citation(refs: List[str], content: str) -> str:
        """Format content with citation references."""
        refs_str = ','.join(f'"{r}"' for r in refs)
        return f'{CITATIONS_START}[{refs_str}]||>{content}{CITATIONS_END}'
    
    @staticmethod
    def format_constrain_block(
        response_format: Optional[Dict[str, Any]] = None,
        reasoning_effort: Optional[str] = None
    ) -> str:
        """Format a constrain block with optional response format and reasoning effort."""
        parts = []
        if response_format:
            strict = 'true' if response_format.get('strict', True) else 'false'
            parts.append(f'response.format={{type:json, strict:{strict}}}')
        if reasoning_effort:
            parts.append(f'reasoning={{effort: {reasoning_effort}}}')
        
        if parts:
            return f'{CONSTRAIN_START}{" ".join(parts)}{CONSTRAIN_END}'
        return ''
    
    # ==================== CONTENTS ARRAY FORMATTING ====================
    
    def format_contents_array(
        self,
        contents: List[Dict[str, Any]],
        include_reasoning: bool = True
    ) -> str:
        """Format an OpenAI-style contents array into USF Omega token format.
        
        Args:
            contents: List of content items with 'type' field.
            include_reasoning: Whether to include reasoning blocks (False for instruct template).
            
        Returns:
            Formatted string with all tokens.
        """
        result_parts = []
        
        for item in contents:
            item_type = item.get('type', '')
            
            if item_type == 'reasoning' and include_reasoning:
                reasoning = item.get('reasoning', '')
                if reasoning:
                    result_parts.append(f'{REASONING_START}{reasoning}{REASONING_END}')
            
            elif item_type == 'text':
                text = item.get('text', '')
                result_parts.append(f'{MESSAGE_START}{text}{MESSAGE_END}')
            
            elif item_type == 'tool_calls':
                tool_calls = item.get('tool_calls', [])
                if tool_calls:
                    calls_str = self._format_tool_calls_list(tool_calls)
                    result_parts.append(f'{FUNCTIONS_START}{calls_str}{FUNCTIONS_END}')
            
            elif item_type == 'self_reflection':
                reflection = item.get('self_reflection', '')
                if reflection:
                    result_parts.append(f'{SELF_REFLECTION_START}{reflection}{SELF_REFLECTION_END}')
        
        return ''.join(result_parts)
    
    def _format_tool_calls_list(self, tool_calls: List[Dict[str, Any]]) -> str:
        """Format a list of tool calls (OpenAI format) into USF Omega invoke format.
        
        Args:
            tool_calls: List of tool call dicts with function.name and function.arguments.
            
        Returns:
            Formatted invoke blocks string.
        """
        calls = []
        for tc in tool_calls:
            # Support both nested and flat formats
            if 'function' in tc:
                name = tc['function'].get('name', '')
                arguments = tc['function'].get('arguments', {})
            else:
                name = tc.get('name', '')
                arguments = tc.get('arguments', {})
            
            # Parse arguments if string
            if isinstance(arguments, str):
                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}
            
            calls.append(self._format_tool_call(name, arguments))
        
        return ''.join(calls)
    
    # ==================== MULTI-ITERATION SUPPORT ====================
    
    def get_all_tool_calls(self, response: str) -> List[List[Function]]:
        """Get tool calls grouped by their position in multi-iteration responses.
        
        For responses with multiple tool call blocks:
        text → tool_calls[0] → self_reflection → text → tool_calls[1] → self_reflection
        
        Args:
            response: The model's response text.
            
        Returns:
            List of lists - each inner list contains Function objects from one block.
        """
        parsed = self.parse_assistant_response(response)
        result = []
        
        for item in parsed.contents:
            if item.type == 'tool_calls':
                functions = [
                    Function(
                        name=tc['function']['name'] if 'function' in tc else tc['name'],
                        arguments=tc['function']['arguments'] if 'function' in tc else tc.get('arguments', {})
                    )
                    for tc in item.content
                ]
                result.append(functions)
        
        return result
    
    def get_iteration_count(self, response: str) -> int:
        """Count the number of text→self_reflection iterations in response.
        
        Args:
            response: The model's response text.
            
        Returns:
            Number of complete iterations (text followed by self_reflection).
        """
        parsed = self.parse_assistant_response(response)
        return len(parsed.self_reflections)
    
    def extract_iteration(self, response: str, iteration: int) -> Dict[str, Any]:
        """Extract a specific iteration from a multi-iteration response.
        
        Args:
            response: The model's response text.
            iteration: 0-indexed iteration number.
            
        Returns:
            Dict with 'text', 'tool_calls' (optional), 'self_reflection' for that iteration.
        """
        parsed = self.parse_assistant_response(response)
        
        # Group contents by iteration
        iterations = []
        current = {'text': None, 'tool_calls': None, 'self_reflection': None}
        
        for item in parsed.contents:
            if item.type == 'reasoning':
                continue  # Skip reasoning (only once at start)
            elif item.type == 'text':
                if current['text'] is not None:
                    # New iteration starts with text
                    iterations.append(current)
                    current = {'text': None, 'tool_calls': None, 'self_reflection': None}
                current['text'] = item.content
            elif item.type == 'tool_calls':
                current['tool_calls'] = item.content
            elif item.type == 'self_reflection':
                current['self_reflection'] = item.content
                iterations.append(current)
                current = {'text': None, 'tool_calls': None, 'self_reflection': None}
        
        # Return requested iteration
        if 0 <= iteration < len(iterations):
            return iterations[iteration]
        return {'text': None, 'tool_calls': None, 'self_reflection': None}
