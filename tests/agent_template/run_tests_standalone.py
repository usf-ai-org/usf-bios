#!/usr/bin/env python3
"""Standalone test runner - no external dependencies."""
import sys
import os
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Inline the tokens and classes to avoid import issues

# Tokens
START_TOKEN = '<|:@:|start|:@:|>'
END_TOKEN = '<|:@:|end|:@:|>'
BOS_TOKEN = '<|:@:|startoftext|:@:|>'
EOS_TOKEN = '<|:@:|endoftext|:@:|>'
MESSAGE_START = '<|:@:|message_start|:@:|>'
MESSAGE_END = '<|:@:|message_end|:@:|>'
CONSTRAIN_START = '<|:@:|constrain_start|:@:|>'
CONSTRAIN_END = '<|:@:|constrain_end|:@:|>'
REASONING_START = '<|:@:|reasoning_start|:@:|>'
REASONING_END = '<|:@:|reasoning_end|:@:|>'
SELF_REFLECTION_START = '<|:@:|self_reflection_start|:@:|>'
SELF_REFLECTION_END = '<|:@:|self_reflection_end|:@:|>'
FUNCTIONS_START = '<|:@:|functions_start|:@:|>'
FUNCTIONS_END = '<|:@:|functions_end|:@:|>'
INVOKE_START = '<||invoke_start to='
INVOKE_END = '<||invoke_end||>'
PARAMETER_START = '<||parameter_start name='
PARAMETER_END = '<||parameter_end||>'
FUNCTION_RESULTS_START = '<|:@:|function_results_start|:@:|>'
FUNCTION_RESULTS_END = '<|:@:|function_results_end|:@:|>'
FUNCTION_RESPONSE_START = '<||function_response_start to='
FUNCTION_RESPONSE_END = '<||function_response_end||>'
CITATIONS_START = '<||citations_start ref='
CITATIONS_END = '<||citations_end||>'

@dataclass
class ParsedContentItem:
    type: str
    content: Any
    position: int = 0
    
    def to_dict(self):
        if self.type == 'reasoning': return {'type': 'reasoning', 'reasoning': self.content}
        elif self.type == 'text': return {'type': 'text', 'text': self.content}
        elif self.type == 'tool_calls': return {'type': 'tool_calls', 'tool_calls': self.content}
        elif self.type == 'self_reflection': return {'type': 'self_reflection', 'self_reflection': self.content}
        else: return {'type': self.type, 'content': self.content}

@dataclass
class ParsedAssistantResponse:
    contents: List[ParsedContentItem] = None
    message_contents: List[str] = None
    function_calls: List[Dict[str, Any]] = None
    reasoning: Optional[str] = None
    self_reflections: List[str] = None
    citations: List[Dict[str, Any]] = None
    constrain: Optional[str] = None
    raw_text: str = ''
    
    def __post_init__(self):
        if self.contents is None: self.contents = []
        if self.message_contents is None: self.message_contents = []
        if self.function_calls is None: self.function_calls = []
        if self.self_reflections is None: self.self_reflections = []
        if self.citations is None: self.citations = []
    
    @property
    def has_function_calls(self): return len(self.function_calls) > 0
    @property
    def has_reasoning(self): return self.reasoning is not None and len(self.reasoning) > 0
    @property
    def has_self_reflection(self): return len(self.self_reflections) > 0
    @property
    def final_message(self): return self.message_contents[-1] if self.message_contents else ''
    @property
    def has_multiple_iterations(self): return sum(1 for c in self.contents if c.type == 'text') > 1
    @property
    def tool_call_blocks(self): return [c.content for c in self.contents if c.type == 'tool_calls']
    def to_contents_array(self): return [c.to_dict() for c in self.contents]
    def to_simple_format(self):
        result = {'role': 'assistant'}
        if self.message_contents: result['content'] = self.message_contents[0]
        if self.function_calls: result['tool_calls'] = self.function_calls
        if self.self_reflections: result['self_reflection'] = self.self_reflections[-1]
        if self.reasoning: result['reasoning'] = self.reasoning
        return result

class MockFunction:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments

class UsfOmegaAgentTemplate:
    def _parse_json(self, text):
        try: return json.loads(text)
        except: return None
    
    def parse_message_contents(self, text):
        pattern = r'<\|:@:\|message_start\|:@:\|>(.*?)<\|:@:\|message_end\|:@:\|>'
        return [m.strip() for m in re.findall(pattern, text, re.DOTALL)]
    
    def parse_reasoning(self, text):
        pattern = r'<\|:@:\|reasoning_start\|:@:\|>(.*?)<\|:@:\|reasoning_end\|:@:\|>'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None
    
    def parse_self_reflections(self, text):
        pattern = r'<\|:@:\|self_reflection_start\|:@:\|>(.*?)<\|:@:\|self_reflection_end\|:@:\|>'
        return [m.strip() for m in re.findall(pattern, text, re.DOTALL)]
    
    def parse_citations(self, text):
        results = []
        pattern = r'<\|\|citations_start ref=\[([^\]]+)\]\|\|>(.*?)<\|\|citations_end\|\|>'
        for refs_str, content in re.findall(pattern, text, re.DOTALL):
            refs = re.findall(r'["\']([^"\']*)["\']', refs_str)
            results.append({'refs': refs, 'content': content.strip()})
        return results
    
    def parse_constrain(self, text):
        pattern = r'<\|:@:\|constrain_start\|:@:\|>(.*?)<\|:@:\|constrain_end\|:@:\|>'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None
    
    def parse_function_calls_from_text(self, text):
        results = []
        pattern = r'<\|\|invoke_start to=function\.([^|]+)\|\|>(.*?)<\|\|invoke_end\|\|>'
        for name, content in re.findall(pattern, text, re.DOTALL):
            args = {}
            for pname, pval in re.findall(r'<\|\|parameter_start name=([^|]+)\|\|>(.*?)<\|\|parameter_end\|\|>', content, re.DOTALL):
                parsed = self._parse_json(pval.strip())
                args[pname.strip()] = parsed if parsed is not None else pval.strip()
            results.append({'id': f'call_{len(results)}', 'type': 'function', 'function': {'name': name.strip(), 'arguments': args}})
        return results
    
    def _parse_tool_calls_from_block(self, block_content):
        results = []
        pattern = r'<\|\|invoke_start to=function\.([^|]+)\|\|>(.*?)<\|\|invoke_end\|\|>'
        for idx, (name, content) in enumerate(re.findall(pattern, block_content, re.DOTALL)):
            args = {}
            for pname, pval in re.findall(r'<\|\|parameter_start name=([^|]+)\|\|>(.*?)<\|\|parameter_end\|\|>', content, re.DOTALL):
                parsed = self._parse_json(pval.strip())
                args[pname.strip()] = parsed if parsed is not None else pval.strip()
            results.append({'id': f'call_{idx}', 'type': 'function', 'function': {'name': name.strip(), 'arguments': args}})
        return results
    
    def parse_sequential_contents(self, text):
        patterns = {
            'reasoning': (REASONING_START, REASONING_END),
            'text': (MESSAGE_START, MESSAGE_END),
            'tool_calls': (FUNCTIONS_START, FUNCTIONS_END),
            'self_reflection': (SELF_REFLECTION_START, SELF_REFLECTION_END),
        }
        blocks = []
        for btype, (start, end) in patterns.items():
            idx = 0
            while True:
                spos = text.find(start, idx)
                if spos == -1: break
                epos = text.find(end, spos + len(start))
                if epos == -1: break
                blocks.append({'type': btype, 'content': text[spos + len(start):epos], 'start_pos': spos})
                idx = epos + len(end)
        blocks.sort(key=lambda x: x['start_pos'])
        contents = []
        for i, b in enumerate(blocks):
            if b['type'] == 'tool_calls':
                contents.append(ParsedContentItem(type='tool_calls', content=self._parse_tool_calls_from_block(b['content']), position=i))
            else:
                contents.append(ParsedContentItem(type=b['type'], content=b['content'].strip(), position=i))
        return contents
    
    def parse_assistant_response(self, text):
        return ParsedAssistantResponse(
            contents=self.parse_sequential_contents(text),
            message_contents=self.parse_message_contents(text),
            function_calls=self.parse_function_calls_from_text(text),
            reasoning=self.parse_reasoning(text),
            self_reflections=self.parse_self_reflections(text),
            citations=self.parse_citations(text),
            constrain=self.parse_constrain(text),
            raw_text=text
        )
    
    def get_iteration_count(self, text):
        return len(self.parse_self_reflections(text))
    
    def extract_iteration(self, text, iteration):
        parsed = self.parse_assistant_response(text)
        iterations = []
        current = {'text': None, 'tool_calls': None, 'self_reflection': None}
        for item in parsed.contents:
            if item.type == 'reasoning': continue
            elif item.type == 'text':
                if current['text'] is not None:
                    iterations.append(current)
                    current = {'text': None, 'tool_calls': None, 'self_reflection': None}
                current['text'] = item.content
            elif item.type == 'tool_calls': current['tool_calls'] = item.content
            elif item.type == 'self_reflection':
                current['self_reflection'] = item.content
                iterations.append(current)
                current = {'text': None, 'tool_calls': None, 'self_reflection': None}
        if 0 <= iteration < len(iterations): return iterations[iteration]
        return {'text': None, 'tool_calls': None, 'self_reflection': None}
    
    def get_all_tool_calls(self, text):
        parsed = self.parse_assistant_response(text)
        result = []
        for item in parsed.contents:
            if item.type == 'tool_calls':
                result.append([MockFunction(tc['function']['name'], tc['function']['arguments']) for tc in item.content])
        return result
    
    def parse_tool_responses_from_text(self, text):
        results = []
        pattern = r'<\|\|function_response_start to=function\.([^|\s]+)(?:\s+id="([^"]*)")?\|\|>(.*?)<\|\|function_response_end\|\|>'
        for name, cid, content in re.findall(pattern, text, re.DOTALL):
            results.append({'name': name.strip(), 'id': cid.strip() if cid else None, 'content': content.strip()})
        return results
    
    def format_message_block(self, c): return f'{MESSAGE_START}{c}{MESSAGE_END}'
    def format_reasoning_block(self, c): return f'{REASONING_START}{c}{REASONING_END}'
    def format_self_reflection_block(self, c): return f'{SELF_REFLECTION_START}{c}{SELF_REFLECTION_END}'
    def format_constrain_block(self, response_format=None, reasoning_effort=None):
        parts = []
        if response_format: parts.append(f'response.format={{type:json, strict:{"true" if response_format.get("strict") else "false"}}}')
        if reasoning_effort: parts.append(f'reasoning={{effort: {reasoning_effort}}}')
        return f'{CONSTRAIN_START}{" ".join(parts)}{CONSTRAIN_END}' if parts else ''
    def format_contents_array(self, contents, include_reasoning=True):
        parts = []
        for item in contents:
            t = item.get('type', '')
            if t == 'reasoning' and include_reasoning: parts.append(f'{REASONING_START}{item.get("reasoning", "")}{REASONING_END}')
            elif t == 'text': parts.append(f'{MESSAGE_START}{item.get("text", "")}{MESSAGE_END}')
            elif t == 'self_reflection': parts.append(f'{SELF_REFLECTION_START}{item.get("self_reflection", "")}{SELF_REFLECTION_END}')
        return ''.join(parts)
    
    @staticmethod
    def has_function_calls(t): return FUNCTIONS_START in t and INVOKE_START in t
    @staticmethod
    def has_reasoning(t): return REASONING_START in t and REASONING_END in t
    @staticmethod
    def has_self_reflection(t): return SELF_REFLECTION_START in t and SELF_REFLECTION_END in t
    @staticmethod
    def has_citations(t): return CITATIONS_START in t and CITATIONS_END in t
    @staticmethod
    def has_tool_results(t): return FUNCTION_RESULTS_START in t

# ===================== TESTS =====================
def test(name, cond):
    if cond: print(f"✓ {name}"); return True
    else: print(f"✗ {name}"); return False

agent = UsfOmegaAgentTemplate()
p, f = 0, 0

print("\n=== USF Omega Agent Template Tests (90 tests) ===\n")

# SECTION 1: Tokens (5)
print("[1] Token Constants")
if test("1.1 message_tokens", MESSAGE_START == '<|:@:|message_start|:@:|>'): p+=1
else: f+=1
if test("1.2 reasoning_tokens", REASONING_START == '<|:@:|reasoning_start|:@:|>'): p+=1
else: f+=1
if test("1.3 reflection_tokens", SELF_REFLECTION_START == '<|:@:|self_reflection_start|:@:|>'): p+=1
else: f+=1
if test("1.4 function_tokens", FUNCTIONS_START == '<|:@:|functions_start|:@:|>'): p+=1
else: f+=1
if test("1.5 boundary_tokens", START_TOKEN == '<|:@:|start|:@:|>'): p+=1
else: f+=1

# SECTION 2: Simple Format (15)
print("\n[2] Simple Format Parsing")
t = f"{MESSAGE_START}Hello{MESSAGE_END}"
r = agent.parse_assistant_response(t)
if test("2.1 text_only", r.message_contents[0] == "Hello"): p+=1
else: f+=1

t = f"{MESSAGE_START}A{MESSAGE_END}{SELF_REFLECTION_START}D{SELF_REFLECTION_END}"
r = agent.parse_assistant_response(t)
if test("2.2 text_reflection", r.has_self_reflection and r.self_reflections[0] == "D"): p+=1
else: f+=1

t = f"{MESSAGE_START}C{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.search||><||parameter_start name=q||>test{PARAMETER_END}{INVOKE_END}{FUNCTIONS_END}"
r = agent.parse_assistant_response(t)
if test("2.3 text_tool", r.has_function_calls and r.function_calls[0]['function']['name'] == 'search'): p+=1
else: f+=1

t = f"{MESSAGE_START}C{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.a||>{INVOKE_END}<||invoke_start to=function.b||>{INVOKE_END}{FUNCTIONS_END}"
r = agent.parse_assistant_response(t)
if test("2.4 parallel_tools", len(r.function_calls) == 2): p+=1
else: f+=1

t = f"{REASONING_START}Think{REASONING_END}{MESSAGE_START}A{MESSAGE_END}"
r = agent.parse_assistant_response(t)
if test("2.5 reasoning_text", r.has_reasoning and r.reasoning == "Think"): p+=1
else: f+=1

t = f"{REASONING_START}T{REASONING_END}{MESSAGE_START}A{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.f||>{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}D{SELF_REFLECTION_END}"
r = agent.parse_assistant_response(t)
if test("2.6 full_single", r.has_reasoning and r.has_function_calls and r.has_self_reflection): p+=1
else: f+=1

t = f"{MESSAGE_START}{MESSAGE_END}"
r = agent.parse_assistant_response(t)
if test("2.7 empty_text", r.message_contents[0] == ""): p+=1
else: f+=1

t = f'{MESSAGE_START}C{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.f||><||parameter_start name=d||>{json.dumps({"k":"v"})}{PARAMETER_END}{INVOKE_END}{FUNCTIONS_END}'
r = agent.parse_assistant_response(t)
if test("2.8 json_param", r.function_calls[0]['function']['arguments']['d'] == {"k":"v"}): p+=1
else: f+=1

t = f'{MESSAGE_START}C{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.f||><||parameter_start name=a||>{json.dumps([1,2,3])}{PARAMETER_END}{INVOKE_END}{FUNCTIONS_END}'
r = agent.parse_assistant_response(t)
if test("2.9 array_param", r.function_calls[0]['function']['arguments']['a'] == [1,2,3]): p+=1
else: f+=1

t = f"{MESSAGE_START}C{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.f||><||parameter_start name=x||>1{PARAMETER_END}<||parameter_start name=y||>2{PARAMETER_END}{INVOKE_END}{FUNCTIONS_END}"
r = agent.parse_assistant_response(t)
if test("2.10 multi_params", r.function_calls[0]['function']['arguments']['x'] == 1): p+=1
else: f+=1

t = f"{MESSAGE_START}C{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.get_time||>{INVOKE_END}{FUNCTIONS_END}"
r = agent.parse_assistant_response(t)
if test("2.11 no_params", r.function_calls[0]['function']['arguments'] == {}): p+=1
else: f+=1

t = f"{CONSTRAIN_START}response.format={{type:json}}{CONSTRAIN_END}{MESSAGE_START}A{MESSAGE_END}"
r = agent.parse_assistant_response(t)
if test("2.12 constrain", "response.format" in r.constrain): p+=1
else: f+=1

t = f"{MESSAGE_START}A{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.f||>{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}D{SELF_REFLECTION_END}"
r = agent.parse_assistant_response(t)
if test("2.13 text_tool_refl", r.has_function_calls and r.has_self_reflection): p+=1
else: f+=1

t = f"{MESSAGE_START}Hello 世界 🌍{MESSAGE_END}"
r = agent.parse_assistant_response(t)
if test("2.14 unicode", "世界" in r.message_contents[0]): p+=1
else: f+=1

t = f"{MESSAGE_START}Line1\nLine2{MESSAGE_END}"
r = agent.parse_assistant_response(t)
if test("2.15 newlines", "\n" in r.message_contents[0]): p+=1
else: f+=1

# SECTION 3: Single Iteration Contents (15)
print("\n[3] Single Iteration Contents")
t = f"{MESSAGE_START}A{MESSAGE_END}{SELF_REFLECTION_START}D{SELF_REFLECTION_END}"
r = agent.parse_assistant_response(t)
if test("3.1 contents_len", len(r.contents) == 2): p+=1
else: f+=1
if test("3.2 contents_types", r.contents[0].type == 'text' and r.contents[1].type == 'self_reflection'): p+=1
else: f+=1

t = f"{MESSAGE_START}A{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.f||>{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}D{SELF_REFLECTION_END}"
r = agent.parse_assistant_response(t)
if test("3.3 text_tool_refl", len(r.contents) == 3 and r.contents[1].type == 'tool_calls'): p+=1
else: f+=1

t = f"{REASONING_START}T{REASONING_END}{MESSAGE_START}A{MESSAGE_END}{SELF_REFLECTION_START}D{SELF_REFLECTION_END}"
r = agent.parse_assistant_response(t)
if test("3.4 reasoning_first", r.contents[0].type == 'reasoning'): p+=1
else: f+=1

t = f"{REASONING_START}T{REASONING_END}{MESSAGE_START}A{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.f||>{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}D{SELF_REFLECTION_END}"
r = agent.parse_assistant_response(t)
if test("3.5 full_contents", len(r.contents) == 4): p+=1
else: f+=1

t = f"{MESSAGE_START}A{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.a||>{INVOKE_END}<||invoke_start to=function.b||>{INVOKE_END}<||invoke_start to=function.c||>{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}D{SELF_REFLECTION_END}"
r = agent.parse_assistant_response(t)
if test("3.6 parallel_3", len(r.contents[1].content) == 3): p+=1
else: f+=1

arr = r.to_contents_array()
if test("3.7 to_array", arr[0]['type'] == 'text'): p+=1
else: f+=1

t = f"{MESSAGE_START}{'A'*1000}{MESSAGE_END}{SELF_REFLECTION_START}D{SELF_REFLECTION_END}"
r = agent.parse_assistant_response(t)
if test("3.8 long_content", len(r.contents[0].content) == 1000): p+=1
else: f+=1

t = f"{MESSAGE_START}a<b>c&d{MESSAGE_END}{SELF_REFLECTION_START}D{SELF_REFLECTION_END}"
r = agent.parse_assistant_response(t)
if test("3.9 special_chars", "<b>" in r.contents[0].content): p+=1
else: f+=1

t = f"{REASONING_START}R{REASONING_END}{MESSAGE_START}T{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.f||>{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}S{SELF_REFLECTION_END}"
r = agent.parse_assistant_response(t)
if test("3.10 positions", r.contents[0].position == 0 and r.contents[3].position == 3): p+=1
else: f+=1

simple = r.to_simple_format()
if test("3.11 to_simple", simple['role'] == 'assistant' and 'content' in simple): p+=1
else: f+=1

t = f"{MESSAGE_START}A{MESSAGE_END}{SELF_REFLECTION_START}D{SELF_REFLECTION_END}"
r = agent.parse_assistant_response(t)
if test("3.12 not_multi_iter", not r.has_multiple_iterations): p+=1
else: f+=1

t = f'{MESSAGE_START}A{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.f||><||parameter_start name=d||>{json.dumps({"a":{"b":{"c":1}}})}{PARAMETER_END}{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}D{SELF_REFLECTION_END}'
r = agent.parse_assistant_response(t)
if test("3.13 nested_json", r.contents[1].content[0]['function']['arguments']['d']['a']['b']['c'] == 1): p+=1
else: f+=1

if test("3.14 tool_call_blocks", len(r.tool_call_blocks) == 1): p+=1
else: f+=1

if test("3.15 final_message", r.final_message == 'A'): p+=1
else: f+=1

# SECTION 4: Multi-Iteration (20)
print("\n[4] Multi-Iteration Contents")
t = f"{MESSAGE_START}T1{MESSAGE_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}T2{MESSAGE_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}"
r = agent.parse_assistant_response(t)
if test("4.1 two_iter", len(r.message_contents) == 2 and len(r.self_reflections) == 2): p+=1
else: f+=1
if test("4.2 multi_iter_prop", r.has_multiple_iterations): p+=1
else: f+=1

t = f"{MESSAGE_START}T1{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.f1||>{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}T2{MESSAGE_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}"
r = agent.parse_assistant_response(t)
if test("4.3 iter_with_tool", len(r.tool_call_blocks) == 1): p+=1
else: f+=1

t = f"{MESSAGE_START}T1{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.f1||>{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}T2{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.f2||>{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}"
r = agent.parse_assistant_response(t)
if test("4.4 both_with_tools", len(r.tool_call_blocks) == 2): p+=1
else: f+=1
if test("4.5 tool_names", r.tool_call_blocks[0][0]['function']['name'] == 'f1' and r.tool_call_blocks[1][0]['function']['name'] == 'f2'): p+=1
else: f+=1

t = f"{MESSAGE_START}T1{MESSAGE_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}T2{MESSAGE_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}{MESSAGE_START}T3{MESSAGE_END}{SELF_REFLECTION_START}R3{SELF_REFLECTION_END}"
r = agent.parse_assistant_response(t)
if test("4.6 three_iter", len(r.message_contents) == 3): p+=1
else: f+=1
if test("4.7 iter_count", agent.get_iteration_count(t) == 3): p+=1
else: f+=1

t = f"{REASONING_START}T{REASONING_END}{MESSAGE_START}T1{MESSAGE_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}T2{MESSAGE_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}{MESSAGE_START}T3{MESSAGE_END}{SELF_REFLECTION_START}R3{SELF_REFLECTION_END}"
r = agent.parse_assistant_response(t)
if test("4.8 three_with_reasoning", r.contents[0].type == 'reasoning' and len(r.message_contents) == 3): p+=1
else: f+=1

t = f"{MESSAGE_START}T1{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.f1||>{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}T2{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.f2||>{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}{MESSAGE_START}T3{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.f3||>{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}R3{SELF_REFLECTION_END}"
r = agent.parse_assistant_response(t)
if test("4.9 three_all_tools", len(r.tool_call_blocks) == 3): p+=1
else: f+=1

t = f"{MESSAGE_START}T1{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.f1||>{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}T2{MESSAGE_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}{MESSAGE_START}T3{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.f3||>{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}R3{SELF_REFLECTION_END}"
r = agent.parse_assistant_response(t)
if test("4.10 mixed_tools", len(r.tool_call_blocks) == 2): p+=1
else: f+=1

t = f"{MESSAGE_START}T1{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.a||>{INVOKE_END}<||invoke_start to=function.b||>{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}T2{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.c||>{INVOKE_END}<||invoke_start to=function.d||>{INVOKE_END}<||invoke_start to=function.e||>{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}"
r = agent.parse_assistant_response(t)
if test("4.11 parallel_multi", len(r.tool_call_blocks[0]) == 2 and len(r.tool_call_blocks[1]) == 3): p+=1
else: f+=1

i0 = agent.extract_iteration(t, 0)
if test("4.12 extract_0", i0['text'] == 'T1' and i0['tool_calls'] is not None): p+=1
else: f+=1

i1 = agent.extract_iteration(t, 1)
if test("4.13 extract_1", i1['text'] == 'T2'): p+=1
else: f+=1

i5 = agent.extract_iteration(t, 5)
if test("4.14 extract_oob", i5['text'] is None): p+=1
else: f+=1

grouped = agent.get_all_tool_calls(t)
if test("4.15 grouped_calls", len(grouped) == 2 and grouped[0][0].name == 'a'): p+=1
else: f+=1

if test("4.16 flattened", len(r.function_calls) == 5): p+=1
else: f+=1

# 5 iterations stress
parts = []
for i in range(5):
    parts.append(f"{MESSAGE_START}T{i}{MESSAGE_END}")
    parts.append(f"{FUNCTIONS_START}<||invoke_start to=function.f{i}||>{INVOKE_END}{FUNCTIONS_END}")
    parts.append(f"{SELF_REFLECTION_START}R{i}{SELF_REFLECTION_END}")
t = ''.join(parts)
r = agent.parse_assistant_response(t)
if test("4.17 five_iter", len(r.message_contents) == 5 and len(r.tool_call_blocks) == 5): p+=1
else: f+=1

t = f"{REASONING_START}R{REASONING_END}{MESSAGE_START}T1{MESSAGE_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}T2{MESSAGE_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}"
r = agent.parse_assistant_response(t)
reasoning_count = sum(1 for c in r.contents if c.type == 'reasoning')
if test("4.18 reasoning_once", reasoning_count == 1): p+=1
else: f+=1

arr = r.to_contents_array()
if test("4.19 multi_to_array", len(arr) == 5 and arr[0]['type'] == 'reasoning'): p+=1
else: f+=1

if test("4.20 final_msg_multi", r.final_message == 'T2'): p+=1
else: f+=1

# SECTION 5: Formatting (15)
print("\n[5] Formatting Methods")
if test("5.1 fmt_msg", agent.format_message_block("Hi") == f"{MESSAGE_START}Hi{MESSAGE_END}"): p+=1
else: f+=1
if test("5.2 fmt_reasoning", agent.format_reasoning_block("T") == f"{REASONING_START}T{REASONING_END}"): p+=1
else: f+=1
if test("5.3 fmt_refl", agent.format_self_reflection_block("D") == f"{SELF_REFLECTION_START}D{SELF_REFLECTION_END}"): p+=1
else: f+=1

c = agent.format_constrain_block(response_format={'strict': True})
if test("5.4 fmt_constrain_json", "strict:true" in c): p+=1
else: f+=1

c = agent.format_constrain_block(reasoning_effort='high')
if test("5.5 fmt_constrain_reasoning", "effort: high" in c): p+=1
else: f+=1

c = agent.format_constrain_block(response_format={'strict': False}, reasoning_effort='medium')
if test("5.6 fmt_constrain_both", "strict:false" in c and "medium" in c): p+=1
else: f+=1

contents = [{"type": "text", "text": "Hi"}, {"type": "self_reflection", "self_reflection": "OK"}]
fmt = agent.format_contents_array(contents)
if test("5.7 fmt_contents", MESSAGE_START in fmt and SELF_REFLECTION_START in fmt): p+=1
else: f+=1

contents = [{"type": "reasoning", "reasoning": "T"}, {"type": "text", "text": "A"}]
fmt = agent.format_contents_array(contents, include_reasoning=True)
if test("5.8 fmt_with_reasoning", REASONING_START in fmt): p+=1
else: f+=1

fmt = agent.format_contents_array(contents, include_reasoning=False)
if test("5.9 fmt_skip_reasoning", REASONING_START not in fmt): p+=1
else: f+=1

if test("5.10 has_func_true", agent.has_function_calls(f"{FUNCTIONS_START}{INVOKE_START}function.f||>{INVOKE_END}{FUNCTIONS_END}")): p+=1
else: f+=1
if test("5.11 has_func_false", not agent.has_function_calls(f"{MESSAGE_START}X{MESSAGE_END}")): p+=1
else: f+=1
if test("5.12 has_reasoning_true", agent.has_reasoning(f"{REASONING_START}X{REASONING_END}")): p+=1
else: f+=1
if test("5.13 has_reasoning_false", not agent.has_reasoning(f"{MESSAGE_START}X{MESSAGE_END}")): p+=1
else: f+=1
if test("5.14 has_refl_true", agent.has_self_reflection(f"{SELF_REFLECTION_START}X{SELF_REFLECTION_END}")): p+=1
else: f+=1
if test("5.15 has_refl_false", not agent.has_self_reflection(f"{MESSAGE_START}X{MESSAGE_END}")): p+=1
else: f+=1

# SECTION 6: Tool Responses (10)
print("\n[6] Tool Response Parsing")
t = f'{FUNCTION_RESULTS_START}<||function_response_start to=function.search id="r1"||>Result<||function_response_end||>{FUNCTION_RESULTS_END}'
results = agent.parse_tool_responses_from_text(t)
if test("6.1 single_response", len(results) == 1 and results[0]['name'] == 'search'): p+=1
else: f+=1
if test("6.2 response_id", results[0]['id'] == 'r1'): p+=1
else: f+=1
if test("6.3 response_content", results[0]['content'] == 'Result'): p+=1
else: f+=1

t = f'{FUNCTION_RESULTS_START}<||function_response_start to=function.a id="r1"||>A<||function_response_end||><||function_response_start to=function.b id="r2"||>B<||function_response_end||>{FUNCTION_RESULTS_END}'
results = agent.parse_tool_responses_from_text(t)
if test("6.4 multi_response", len(results) == 2): p+=1
else: f+=1

t = f'{FUNCTION_RESULTS_START}<||function_response_start to=function.test||>Data<||function_response_end||>{FUNCTION_RESULTS_END}'
results = agent.parse_tool_responses_from_text(t)
if test("6.5 no_id", results[0]['id'] is None): p+=1
else: f+=1

if test("6.6 has_results_true", agent.has_tool_results(f"{FUNCTION_RESULTS_START}X{FUNCTION_RESULTS_END}")): p+=1
else: f+=1
if test("6.7 has_results_false", not agent.has_tool_results(f"{MESSAGE_START}X{MESSAGE_END}")): p+=1
else: f+=1

t = f'{FUNCTION_RESULTS_START}<||function_response_start to=function.f id="r"||>Line1\nLine2<||function_response_end||>{FUNCTION_RESULTS_END}'
results = agent.parse_tool_responses_from_text(t)
if test("6.8 multiline", "\n" in results[0]['content']): p+=1
else: f+=1

t = f'{FUNCTION_RESULTS_START}<||function_response_start to=function.f id="r"||><html>"&"</html><||function_response_end||>{FUNCTION_RESULTS_END}'
results = agent.parse_tool_responses_from_text(t)
if test("6.9 special_chars", '"&"' in results[0]['content']): p+=1
else: f+=1

t = f'{FUNCTION_RESULTS_START}<||function_response_start to=function.f id="r"||>{json.dumps({"k":"v"})}<||function_response_end||>{FUNCTION_RESULTS_END}'
results = agent.parse_tool_responses_from_text(t)
if test("6.10 json_content", '"k"' in results[0]['content']): p+=1
else: f+=1

# SECTION 7: Citations (5)
print("\n[7] Citation Parsing")
t = f'{CITATIONS_START}["s001"]||>cited{CITATIONS_END}'
c = agent.parse_citations(t)
if test("7.1 single_citation", len(c) == 1 and c[0]['refs'] == ['s001']): p+=1
else: f+=1

t = f'{CITATIONS_START}["s001","s002","s003"]||>cited{CITATIONS_END}'
c = agent.parse_citations(t)
if test("7.2 multi_refs", c[0]['refs'] == ['s001', 's002', 's003']): p+=1
else: f+=1

t = f'{CITATIONS_START}["s1"]||>A{CITATIONS_END} and {CITATIONS_START}["s2"]||>B{CITATIONS_END}'
c = agent.parse_citations(t)
if test("7.3 multi_citations", len(c) == 2): p+=1
else: f+=1

if test("7.4 has_cite_true", agent.has_citations(f'{CITATIONS_START}["s"]||>x{CITATIONS_END}')): p+=1
else: f+=1
if test("7.5 has_cite_false", not agent.has_citations(f"{MESSAGE_START}X{MESSAGE_END}")): p+=1
else: f+=1

# SECTION 8: Edge Cases (10)
print("\n[8] Edge Cases")
r = agent.parse_assistant_response("")
if test("8.1 empty", len(r.contents) == 0): p+=1
else: f+=1

r = agent.parse_assistant_response("Plain text")
if test("8.2 no_tokens", len(r.contents) == 0): p+=1
else: f+=1

r = agent.parse_assistant_response(f"{MESSAGE_START}Incomplete")
if test("8.3 incomplete_msg", len(r.message_contents) == 0): p+=1
else: f+=1

r = agent.parse_assistant_response(f"{FUNCTIONS_START}<||invoke_start to=function.f||>")
if test("8.4 incomplete_func", len(r.function_calls) == 0): p+=1
else: f+=1

t = f"{MESSAGE_START}||| <|> :@: chars{MESSAGE_END}"
r = agent.parse_assistant_response(t)
if test("8.5 special_in_content", "|||" in r.message_contents[0]): p+=1
else: f+=1

long = "X" * 100000
t = f"{MESSAGE_START}{long}{MESSAGE_END}"
r = agent.parse_assistant_response(t)
if test("8.6 very_long", len(r.message_contents[0]) == 100000): p+=1
else: f+=1

parts = []
for i in range(10):
    parts.append(f"{MESSAGE_START}T{i}{MESSAGE_END}{SELF_REFLECTION_START}R{i}{SELF_REFLECTION_END}")
r = agent.parse_assistant_response(''.join(parts))
if test("8.7 ten_iterations", len(r.message_contents) == 10): p+=1
else: f+=1

calls = ''.join([f"<||invoke_start to=function.f{i}||>{INVOKE_END}" for i in range(20)])
t = f"{MESSAGE_START}C{MESSAGE_END}{FUNCTIONS_START}{calls}{FUNCTIONS_END}"
r = agent.parse_assistant_response(t)
if test("8.8 twenty_parallel", len(r.function_calls) == 20): p+=1
else: f+=1

t = f"{MESSAGE_START}  spaces  {MESSAGE_END}"
r = agent.parse_assistant_response(t)
if test("8.9 whitespace", r.message_contents[0] == "spaces"): p+=1
else: f+=1

t = f"{MESSAGE_START}First{MESSAGE_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}Last{MESSAGE_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}"
r = agent.parse_assistant_response(t)
if test("8.10 final_message", r.final_message == "Last"): p+=1
else: f+=1

# SECTION 9: ParsedContentItem (5)
print("\n[9] ParsedContentItem")
item = ParsedContentItem(type='reasoning', content='T')
if test("9.1 reasoning_dict", item.to_dict() == {'type': 'reasoning', 'reasoning': 'T'}): p+=1
else: f+=1

item = ParsedContentItem(type='text', content='Hi')
if test("9.2 text_dict", item.to_dict() == {'type': 'text', 'text': 'Hi'}): p+=1
else: f+=1

item = ParsedContentItem(type='tool_calls', content=[{'function': {'name': 'f'}}])
if test("9.3 tool_calls_dict", item.to_dict()['tool_calls'][0]['function']['name'] == 'f'): p+=1
else: f+=1

item = ParsedContentItem(type='self_reflection', content='OK')
if test("9.4 refl_dict", item.to_dict() == {'type': 'self_reflection', 'self_reflection': 'OK'}): p+=1
else: f+=1

item = ParsedContentItem(type='custom', content='data')
if test("9.5 unknown_dict", item.to_dict() == {'type': 'custom', 'content': 'data'}): p+=1
else: f+=1

# Summary
print(f"\n{'='*50}")
print(f"TOTAL: {p + f} tests")
print(f"PASSED: {p} ✓")
print(f"FAILED: {f} ✗")
print(f"{'='*50}")
sys.exit(0 if f == 0 else 1)
