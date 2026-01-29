#!/usr/bin/env python3
"""
Tests for all tool call and self-reflection variations.

Covers:
1. Normal tool calls (single, parallel)
2. Self-reflection with multiple occurrences  
3. Multiple/single tool calls per iteration
4. Mixed combinations
"""
import sys
import os
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# Tokens
START_TOKEN = '<|:@:|start|:@:|>'
END_TOKEN = '<|:@:|end|:@:|>'
MESSAGE_START = '<|:@:|message_start|:@:|>'
MESSAGE_END = '<|:@:|message_end|:@:|>'
FUNCTIONS_START = '<|:@:|functions_start|:@:|>'
FUNCTIONS_END = '<|:@:|functions_end|:@:|>'
INVOKE_START = '<||invoke_start to='
INVOKE_END = '<||invoke_end||>'
PARAMETER_START = '<||parameter_start name='
PARAMETER_END = '<||parameter_end||>'
REASONING_START = '<|:@:|reasoning_start|:@:|>'
REASONING_END = '<|:@:|reasoning_end|:@:|>'
SELF_REFLECTION_START = '<|:@:|self_reflection_start|:@:|>'
SELF_REFLECTION_END = '<|:@:|self_reflection_end|:@:|>'
FUNCTION_RESULTS_START = '<|:@:|function_results_start|:@:|>'
FUNCTION_RESULTS_END = '<|:@:|function_results_end|:@:|>'
FUNCTION_RESPONSE_START = '<||function_response_start to='
FUNCTION_RESPONSE_END = '<||function_response_end||>'


def make_tool_call(name, **args):
    """Helper to create a tool call."""
    params = ''.join(f'{PARAMETER_START}{k}||>{json.dumps(v) if isinstance(v, (dict, list)) else v}{PARAMETER_END}' for k, v in args.items())
    return f'{INVOKE_START}function.{name}||>{params}{INVOKE_END}'


def make_tool_response(name, content, call_id=None):
    """Helper to create a tool response."""
    id_part = f' id="{call_id}"' if call_id else ''
    return f'{FUNCTION_RESPONSE_START}function.{name}{id_part}||>{content}{FUNCTION_RESPONSE_END}'


@dataclass
class ParsedContentItem:
    type: str
    content: Any
    position: int = 0


class SimpleAgentTemplate:
    """Simplified agent template for testing."""
    
    def _parse_json(self, text):
        try:
            return json.loads(text)
        except:
            return None
    
    def parse_function_calls(self, text):
        results = []
        pattern = r'<\|\|invoke_start to=function\.([^|]+)\|\|>(.*?)<\|\|invoke_end\|\|>'
        for name, content in re.findall(pattern, text, re.DOTALL):
            args = {}
            for pname, pval in re.findall(r'<\|\|parameter_start name=([^|]+)\|\|>(.*?)<\|\|parameter_end\|\|>', content, re.DOTALL):
                parsed = self._parse_json(pval.strip())
                args[pname.strip()] = parsed if parsed is not None else pval.strip()
            results.append({'name': name.strip(), 'arguments': args})
        return results
    
    def parse_tool_responses(self, text):
        results = []
        pattern = r'<\|\|function_response_start to=function\.([^|\s]+)(?:\s+id="([^"]*)")?\|\|>(.*?)<\|\|function_response_end\|\|>'
        for name, cid, content in re.findall(pattern, text, re.DOTALL):
            results.append({'name': name.strip(), 'id': cid.strip() if cid else None, 'content': content.strip()})
        return results
    
    def parse_message_contents(self, text):
        pattern = r'<\|:@:\|message_start\|:@:\|>(.*?)<\|:@:\|message_end\|:@:\|>'
        return [m.strip() for m in re.findall(pattern, text, re.DOTALL)]
    
    def parse_self_reflections(self, text):
        pattern = r'<\|:@:\|self_reflection_start\|:@:\|>(.*?)<\|:@:\|self_reflection_end\|:@:\|>'
        return [m.strip() for m in re.findall(pattern, text, re.DOTALL)]
    
    def parse_reasoning(self, text):
        pattern = r'<\|:@:\|reasoning_start\|:@:\|>(.*?)<\|:@:\|reasoning_end\|:@:\|>'
        match = re.search(pattern, text, re.DOTALL)
        return match.group(1).strip() if match else None
    
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
                # Parse tool calls from this block
                tool_calls = self.parse_function_calls(b['content'])
                contents.append(ParsedContentItem(type='tool_calls', content=tool_calls, position=i))
            else:
                contents.append(ParsedContentItem(type=b['type'], content=b['content'].strip(), position=i))
        return contents
    
    def get_tool_call_blocks(self, text):
        """Get tool calls grouped by iteration."""
        contents = self.parse_sequential_contents(text)
        return [c.content for c in contents if c.type == 'tool_calls']
    
    def get_iteration_count(self, text):
        """Count iterations (self-reflection blocks)."""
        return len(self.parse_self_reflections(text))
    
    def extract_iteration(self, text, iteration):
        """Extract a specific iteration."""
        contents = self.parse_sequential_contents(text)
        iterations = []
        current = {'text': None, 'tool_calls': None, 'self_reflection': None}
        for item in contents:
            if item.type == 'reasoning':
                continue
            elif item.type == 'text':
                if current['text'] is not None:
                    iterations.append(current)
                    current = {'text': None, 'tool_calls': None, 'self_reflection': None}
                current['text'] = item.content
            elif item.type == 'tool_calls':
                current['tool_calls'] = item.content
            elif item.type == 'self_reflection':
                current['self_reflection'] = item.content
                iterations.append(current)
                current = {'text': None, 'tool_calls': None, 'self_reflection': None}
        if 0 <= iteration < len(iterations):
            return iterations[iteration]
        return {'text': None, 'tool_calls': None, 'self_reflection': None}


def test(name, condition):
    if condition:
        print(f"✓ {name}")
        return True
    else:
        print(f"✗ {name}")
        return False


agent = SimpleAgentTemplate()
passed = 0
failed = 0

print("\n" + "=" * 70)
print("Tool Call & Self-Reflection Variations Tests")
print("=" * 70)

# ==============================================================================
# SECTION 1: Normal Tool Calls (Single)
# ==============================================================================
print("\n[1] Normal Tool Calls - Single")

# Test 1.1: Single tool call with no args
text = f"{MESSAGE_START}Calling{MESSAGE_END}{FUNCTIONS_START}{make_tool_call('get_time')}{FUNCTIONS_END}"
calls = agent.parse_function_calls(text)
if test("1.1 single_no_args", len(calls) == 1 and calls[0]['name'] == 'get_time' and calls[0]['arguments'] == {}):
    passed += 1
else:
    failed += 1

# Test 1.2: Single tool call with one arg
text = f"{MESSAGE_START}Searching{MESSAGE_END}{FUNCTIONS_START}{make_tool_call('search', q='weather')}{FUNCTIONS_END}"
calls = agent.parse_function_calls(text)
if test("1.2 single_one_arg", len(calls) == 1 and calls[0]['arguments'].get('q') == 'weather'):
    passed += 1
else:
    failed += 1

# Test 1.3: Single tool call with multiple args
text = f"{MESSAGE_START}Booking{MESSAGE_END}{FUNCTIONS_START}{make_tool_call('book_flight', from_city='NYC', to_city='LAX', date='2024-01-01')}{FUNCTIONS_END}"
calls = agent.parse_function_calls(text)
if test("1.3 single_multi_args", len(calls) == 1 and calls[0]['arguments'].get('from_city') == 'NYC'):
    passed += 1
else:
    failed += 1

# Test 1.4: Single tool call with JSON object arg
text = f'{MESSAGE_START}Config{MESSAGE_END}{FUNCTIONS_START}{make_tool_call("configure", settings={"theme": "dark", "lang": "en"})}{FUNCTIONS_END}'
calls = agent.parse_function_calls(text)
if test("1.4 single_json_arg", len(calls) == 1 and calls[0]['arguments'].get('settings') == {"theme": "dark", "lang": "en"}):
    passed += 1
else:
    failed += 1

# Test 1.5: Single tool call with array arg
text = f'{MESSAGE_START}Multiple{MESSAGE_END}{FUNCTIONS_START}{make_tool_call("process", items=[1, 2, 3])}{FUNCTIONS_END}'
calls = agent.parse_function_calls(text)
if test("1.5 single_array_arg", len(calls) == 1 and calls[0]['arguments'].get('items') == [1, 2, 3]):
    passed += 1
else:
    failed += 1

# ==============================================================================
# SECTION 2: Parallel Tool Calls
# ==============================================================================
print("\n[2] Parallel Tool Calls (Multiple in One Block)")

# Test 2.1: Two parallel calls
text = f"{MESSAGE_START}Calling both{MESSAGE_END}{FUNCTIONS_START}{make_tool_call('get_weather', city='NYC')}{make_tool_call('get_news', topic='tech')}{FUNCTIONS_END}"
calls = agent.parse_function_calls(text)
if test("2.1 two_parallel", len(calls) == 2 and calls[0]['name'] == 'get_weather' and calls[1]['name'] == 'get_news'):
    passed += 1
else:
    failed += 1

# Test 2.2: Three parallel calls
text = f"{MESSAGE_START}Triple{MESSAGE_END}{FUNCTIONS_START}{make_tool_call('a')}{make_tool_call('b')}{make_tool_call('c')}{FUNCTIONS_END}"
calls = agent.parse_function_calls(text)
if test("2.2 three_parallel", len(calls) == 3 and [c['name'] for c in calls] == ['a', 'b', 'c']):
    passed += 1
else:
    failed += 1

# Test 2.3: Five parallel calls
calls_str = ''.join(make_tool_call(f'func{i}', x=i) for i in range(5))
text = f"{MESSAGE_START}Many{MESSAGE_END}{FUNCTIONS_START}{calls_str}{FUNCTIONS_END}"
calls = agent.parse_function_calls(text)
if test("2.3 five_parallel", len(calls) == 5 and calls[2]['name'] == 'func2'):
    passed += 1
else:
    failed += 1

# Test 2.4: Parallel calls with different arg types
text = f'{MESSAGE_START}Mixed{MESSAGE_END}{FUNCTIONS_START}{make_tool_call("a", x=1)}{make_tool_call("b", y="str")}{make_tool_call("c", z=[1,2])}{FUNCTIONS_END}'
calls = agent.parse_function_calls(text)
if test("2.4 parallel_mixed_args", len(calls) == 3 and calls[2]['arguments']['z'] == [1, 2]):
    passed += 1
else:
    failed += 1

# Test 2.5: Ten parallel calls (stress)
calls_str = ''.join(make_tool_call(f'f{i}') for i in range(10))
text = f"{MESSAGE_START}Stress{MESSAGE_END}{FUNCTIONS_START}{calls_str}{FUNCTIONS_END}"
calls = agent.parse_function_calls(text)
if test("2.5 ten_parallel", len(calls) == 10):
    passed += 1
else:
    failed += 1

# ==============================================================================
# SECTION 3: Self-Reflection Variations
# ==============================================================================
print("\n[3] Self-Reflection Variations")

# Test 3.1: Single self-reflection
text = f"{MESSAGE_START}Answer{MESSAGE_END}{SELF_REFLECTION_START}Looks good{SELF_REFLECTION_END}"
refls = agent.parse_self_reflections(text)
if test("3.1 single_reflection", len(refls) == 1 and refls[0] == 'Looks good'):
    passed += 1
else:
    failed += 1

# Test 3.2: Two self-reflections (two iterations)
text = f"{MESSAGE_START}A1{MESSAGE_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}A2{MESSAGE_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}"
refls = agent.parse_self_reflections(text)
if test("3.2 two_reflections", len(refls) == 2 and refls[0] == 'R1' and refls[1] == 'R2'):
    passed += 1
else:
    failed += 1

# Test 3.3: Three self-reflections
text = f"{MESSAGE_START}A1{MESSAGE_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}A2{MESSAGE_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}{MESSAGE_START}A3{MESSAGE_END}{SELF_REFLECTION_START}R3{SELF_REFLECTION_END}"
refls = agent.parse_self_reflections(text)
if test("3.3 three_reflections", len(refls) == 3):
    passed += 1
else:
    failed += 1

# Test 3.4: Five self-reflections (stress)
parts = ''.join(f"{MESSAGE_START}A{i}{MESSAGE_END}{SELF_REFLECTION_START}R{i}{SELF_REFLECTION_END}" for i in range(5))
text = parts
refls = agent.parse_self_reflections(text)
if test("3.4 five_reflections", len(refls) == 5):
    passed += 1
else:
    failed += 1

# Test 3.5: Iteration count matches reflection count
text = f"{MESSAGE_START}A1{MESSAGE_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}A2{MESSAGE_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}{MESSAGE_START}A3{MESSAGE_END}{SELF_REFLECTION_START}R3{SELF_REFLECTION_END}"
count = agent.get_iteration_count(text)
if test("3.5 iteration_count", count == 3):
    passed += 1
else:
    failed += 1

# ==============================================================================
# SECTION 4: Tool Calls with Self-Reflection
# ==============================================================================
print("\n[4] Tool Calls with Self-Reflection")

# Test 4.1: Single tool call + single reflection
text = f"{MESSAGE_START}Calling{MESSAGE_END}{FUNCTIONS_START}{make_tool_call('search', q='test')}{FUNCTIONS_END}{SELF_REFLECTION_START}Done{SELF_REFLECTION_END}"
calls = agent.parse_function_calls(text)
refls = agent.parse_self_reflections(text)
if test("4.1 tool_then_reflection", len(calls) == 1 and len(refls) == 1):
    passed += 1
else:
    failed += 1

# Test 4.2: Multiple parallel calls + reflection
text = f"{MESSAGE_START}Multi{MESSAGE_END}{FUNCTIONS_START}{make_tool_call('a')}{make_tool_call('b')}{make_tool_call('c')}{FUNCTIONS_END}{SELF_REFLECTION_START}All done{SELF_REFLECTION_END}"
calls = agent.parse_function_calls(text)
refls = agent.parse_self_reflections(text)
if test("4.2 parallel_then_reflection", len(calls) == 3 and len(refls) == 1):
    passed += 1
else:
    failed += 1

# Test 4.3: Tool call in iteration 1, no tool call in iteration 2
text = f"{MESSAGE_START}T1{MESSAGE_END}{FUNCTIONS_START}{make_tool_call('search', q='x')}{FUNCTIONS_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}T2{MESSAGE_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}"
blocks = agent.get_tool_call_blocks(text)
if test("4.3 tool_iter1_only", len(blocks) == 1 and blocks[0][0]['name'] == 'search'):
    passed += 1
else:
    failed += 1

# Test 4.4: No tool call in iteration 1, tool call in iteration 2
text = f"{MESSAGE_START}T1{MESSAGE_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}T2{MESSAGE_END}{FUNCTIONS_START}{make_tool_call('search', q='x')}{FUNCTIONS_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}"
blocks = agent.get_tool_call_blocks(text)
if test("4.4 tool_iter2_only", len(blocks) == 1 and blocks[0][0]['name'] == 'search'):
    passed += 1
else:
    failed += 1

# Test 4.5: Tool calls in both iterations
text = f"{MESSAGE_START}T1{MESSAGE_END}{FUNCTIONS_START}{make_tool_call('a')}{FUNCTIONS_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}T2{MESSAGE_END}{FUNCTIONS_START}{make_tool_call('b')}{FUNCTIONS_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}"
blocks = agent.get_tool_call_blocks(text)
if test("4.5 tools_both_iters", len(blocks) == 2 and blocks[0][0]['name'] == 'a' and blocks[1][0]['name'] == 'b'):
    passed += 1
else:
    failed += 1

# ==============================================================================
# SECTION 5: Multiple/Single Tool Calls Per Iteration
# ==============================================================================
print("\n[5] Multiple/Single Tool Calls Per Iteration")

# Test 5.1: Single in iter1, multiple in iter2
text = f"{MESSAGE_START}T1{MESSAGE_END}{FUNCTIONS_START}{make_tool_call('a')}{FUNCTIONS_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}T2{MESSAGE_END}{FUNCTIONS_START}{make_tool_call('b')}{make_tool_call('c')}{make_tool_call('d')}{FUNCTIONS_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}"
blocks = agent.get_tool_call_blocks(text)
if test("5.1 single_then_multiple", len(blocks) == 2 and len(blocks[0]) == 1 and len(blocks[1]) == 3):
    passed += 1
else:
    failed += 1

# Test 5.2: Multiple in iter1, single in iter2
text = f"{MESSAGE_START}T1{MESSAGE_END}{FUNCTIONS_START}{make_tool_call('a')}{make_tool_call('b')}{FUNCTIONS_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}T2{MESSAGE_END}{FUNCTIONS_START}{make_tool_call('c')}{FUNCTIONS_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}"
blocks = agent.get_tool_call_blocks(text)
if test("5.2 multiple_then_single", len(blocks) == 2 and len(blocks[0]) == 2 and len(blocks[1]) == 1):
    passed += 1
else:
    failed += 1

# Test 5.3: Different counts per iteration (2, 3, 1)
text = f"{MESSAGE_START}T1{MESSAGE_END}{FUNCTIONS_START}{make_tool_call('a')}{make_tool_call('b')}{FUNCTIONS_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}T2{MESSAGE_END}{FUNCTIONS_START}{make_tool_call('c')}{make_tool_call('d')}{make_tool_call('e')}{FUNCTIONS_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}{MESSAGE_START}T3{MESSAGE_END}{FUNCTIONS_START}{make_tool_call('f')}{FUNCTIONS_END}{SELF_REFLECTION_START}R3{SELF_REFLECTION_END}"
blocks = agent.get_tool_call_blocks(text)
if test("5.3 varying_counts", len(blocks) == 3 and len(blocks[0]) == 2 and len(blocks[1]) == 3 and len(blocks[2]) == 1):
    passed += 1
else:
    failed += 1

# Test 5.4: Extract iteration with tool calls
text = f"{MESSAGE_START}Text1{MESSAGE_END}{FUNCTIONS_START}{make_tool_call('search', q='test')}{FUNCTIONS_END}{SELF_REFLECTION_START}Refl1{SELF_REFLECTION_END}{MESSAGE_START}Text2{MESSAGE_END}{SELF_REFLECTION_START}Refl2{SELF_REFLECTION_END}"
iter0 = agent.extract_iteration(text, 0)
iter1 = agent.extract_iteration(text, 1)
if test("5.4 extract_iter_0", iter0['text'] == 'Text1' and iter0['tool_calls'] is not None and len(iter0['tool_calls']) == 1):
    passed += 1
else:
    failed += 1

if test("5.5 extract_iter_1", iter1['text'] == 'Text2' and iter1['tool_calls'] is None and iter1['self_reflection'] == 'Refl2'):
    passed += 1
else:
    failed += 1

# ==============================================================================
# SECTION 6: With Reasoning Block
# ==============================================================================
print("\n[6] With Reasoning Block")

# Test 6.1: Reasoning + single iteration + tool
text = f"{REASONING_START}Thinking{REASONING_END}{MESSAGE_START}Answer{MESSAGE_END}{FUNCTIONS_START}{make_tool_call('search', q='x')}{FUNCTIONS_END}{SELF_REFLECTION_START}Done{SELF_REFLECTION_END}"
reasoning = agent.parse_reasoning(text)
calls = agent.parse_function_calls(text)
refls = agent.parse_self_reflections(text)
if test("6.1 reasoning_tool_refl", reasoning == 'Thinking' and len(calls) == 1 and len(refls) == 1):
    passed += 1
else:
    failed += 1

# Test 6.2: Reasoning + multi-iteration
text = f"{REASONING_START}Think{REASONING_END}{MESSAGE_START}A1{MESSAGE_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}A2{MESSAGE_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}"
reasoning = agent.parse_reasoning(text)
refls = agent.parse_self_reflections(text)
if test("6.2 reasoning_multi_iter", reasoning == 'Think' and len(refls) == 2):
    passed += 1
else:
    failed += 1

# Test 6.3: Reasoning + multi-iteration with tools in each
text = f"{REASONING_START}Think{REASONING_END}{MESSAGE_START}A1{MESSAGE_END}{FUNCTIONS_START}{make_tool_call('a')}{FUNCTIONS_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}A2{MESSAGE_END}{FUNCTIONS_START}{make_tool_call('b')}{make_tool_call('c')}{FUNCTIONS_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}"
reasoning = agent.parse_reasoning(text)
blocks = agent.get_tool_call_blocks(text)
if test("6.3 reasoning_multi_tools", reasoning == 'Think' and len(blocks) == 2 and len(blocks[1]) == 2):
    passed += 1
else:
    failed += 1

# ==============================================================================
# SECTION 7: Tool Response Parsing
# ==============================================================================
print("\n[7] Tool Response Parsing")

# Test 7.1: Single tool response
text = f"{FUNCTION_RESULTS_START}{make_tool_response('search', 'Results here', 'call_1')}{FUNCTION_RESULTS_END}"
responses = agent.parse_tool_responses(text)
if test("7.1 single_response", len(responses) == 1 and responses[0]['name'] == 'search' and responses[0]['id'] == 'call_1'):
    passed += 1
else:
    failed += 1

# Test 7.2: Multiple tool responses (matching parallel calls)
text = f"{FUNCTION_RESULTS_START}{make_tool_response('a', 'R1', 'c1')}{make_tool_response('b', 'R2', 'c2')}{make_tool_response('c', 'R3', 'c3')}{FUNCTION_RESULTS_END}"
responses = agent.parse_tool_responses(text)
if test("7.2 multiple_responses", len(responses) == 3 and responses[2]['name'] == 'c'):
    passed += 1
else:
    failed += 1

# Test 7.3: Tool response without ID
text = f"{FUNCTION_RESULTS_START}{make_tool_response('test', 'Content')}{FUNCTION_RESULTS_END}"
responses = agent.parse_tool_responses(text)
if test("7.3 response_no_id", len(responses) == 1 and responses[0]['id'] is None):
    passed += 1
else:
    failed += 1

# ==============================================================================
# SECTION 8: Complex Real-World Scenarios
# ==============================================================================
print("\n[8] Complex Real-World Scenarios")

# Test 8.1: Complete conversation with tool call and response
assistant1 = f"{MESSAGE_START}Let me search for that{MESSAGE_END}{FUNCTIONS_START}{make_tool_call('search', query='weather NYC')}{FUNCTIONS_END}"
tool_result = f"{FUNCTION_RESULTS_START}{make_tool_response('search', 'Sunny, 75°F', 'call_1')}{FUNCTION_RESULTS_END}"
assistant2 = f"{MESSAGE_START}The weather in NYC is sunny, 75°F{MESSAGE_END}"
# Parse assistant1
calls = agent.parse_function_calls(assistant1)
# Parse tool result  
responses = agent.parse_tool_responses(tool_result)
# Parse assistant2
msgs = agent.parse_message_contents(assistant2)
if test("8.1 full_tool_flow", len(calls) == 1 and len(responses) == 1 and len(msgs) == 1):
    passed += 1
else:
    failed += 1

# Test 8.2: Multi-iteration with reasoning, parallel calls, and responses
full_response = f"""
{REASONING_START}Need to search multiple sources{REASONING_END}
{MESSAGE_START}Searching multiple sources{MESSAGE_END}
{FUNCTIONS_START}{make_tool_call('search_web', q='topic')}{make_tool_call('search_db', q='topic')}{FUNCTIONS_END}
{SELF_REFLECTION_START}Got results from both, need to synthesize{SELF_REFLECTION_END}
{MESSAGE_START}Based on both sources, here is the answer{MESSAGE_END}
{SELF_REFLECTION_START}Answer is complete{SELF_REFLECTION_END}
"""
reasoning = agent.parse_reasoning(full_response)
blocks = agent.get_tool_call_blocks(full_response)
refls = agent.parse_self_reflections(full_response)
msgs = agent.parse_message_contents(full_response)
if test("8.2 complex_multi_iter", reasoning == 'Need to search multiple sources' and len(blocks) == 1 and len(blocks[0]) == 2 and len(refls) == 2 and len(msgs) == 2):
    passed += 1
else:
    failed += 1

# Test 8.3: Five iterations with varying tool calls
parts = []
for i in range(5):
    parts.append(f"{MESSAGE_START}Answer {i}{MESSAGE_END}")
    if i % 2 == 0:  # Tool calls on even iterations
        calls_str = ''.join(make_tool_call(f'func{i}_{j}') for j in range(i + 1))
        parts.append(f"{FUNCTIONS_START}{calls_str}{FUNCTIONS_END}")
    parts.append(f"{SELF_REFLECTION_START}Reflection {i}{SELF_REFLECTION_END}")
text = ''.join(parts)
blocks = agent.get_tool_call_blocks(text)
refls = agent.parse_self_reflections(text)
# Iterations 0, 2, 4 have tool calls with 1, 3, 5 calls respectively
if test("8.3 five_iter_varying", len(blocks) == 3 and len(blocks[0]) == 1 and len(blocks[1]) == 3 and len(blocks[2]) == 5 and len(refls) == 5):
    passed += 1
else:
    failed += 1

# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "=" * 70)
print(f"TOTAL: {passed + failed} tests")
print(f"PASSED: {passed} ✓")
print(f"FAILED: {failed} ✗")
print("=" * 70)

sys.exit(0 if failed == 0 else 1)
