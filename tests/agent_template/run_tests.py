#!/usr/bin/env python3
"""Quick test runner for USF Omega Agent Template."""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from usf_bios.agent_template.usf_omega import *

def test(name, condition):
    if condition:
        print(f"✓ {name}")
        return True
    else:
        print(f"✗ {name}")
        return False

agent = UsfOmegaAgentTemplate()
passed = 0
failed = 0

print("\n=== USF Omega Agent Template Tests ===\n")

# Token tests
if test("token_constants", MESSAGE_START == '<|:@:|message_start|:@:|>'): passed += 1
else: failed += 1

# Simple text
text = f"{MESSAGE_START}Hello{MESSAGE_END}"
r = agent.parse_assistant_response(text)
if test("simple_text", r.message_contents[0] == "Hello"): passed += 1
else: failed += 1

# Text + reflection
text = f"{MESSAGE_START}Answer{MESSAGE_END}{SELF_REFLECTION_START}Done{SELF_REFLECTION_END}"
r = agent.parse_assistant_response(text)
if test("text_reflection", r.has_self_reflection and r.self_reflections[0] == "Done"): passed += 1
else: failed += 1

# Text + tool call
text = f"{MESSAGE_START}Call{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.search||><||parameter_start name=q||>test{PARAMETER_END}{INVOKE_END}{FUNCTIONS_END}"
r = agent.parse_assistant_response(text)
if test("text_tool_call", r.has_function_calls and r.function_calls[0]['function']['name'] == 'search'): passed += 1
else: failed += 1

# Parallel tool calls
text = f"{MESSAGE_START}Call{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.a||>{INVOKE_END}<||invoke_start to=function.b||>{INVOKE_END}{FUNCTIONS_END}"
r = agent.parse_assistant_response(text)
if test("parallel_tool_calls", len(r.function_calls) == 2): passed += 1
else: failed += 1

# Reasoning + text
text = f"{REASONING_START}Think{REASONING_END}{MESSAGE_START}Answer{MESSAGE_END}"
r = agent.parse_assistant_response(text)
if test("reasoning_text", r.has_reasoning and r.reasoning == "Think"): passed += 1
else: failed += 1

# Full: reasoning + text + tool_call + reflection
text = f"{REASONING_START}Think{REASONING_END}{MESSAGE_START}Call{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.f||>{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}Done{SELF_REFLECTION_END}"
r = agent.parse_assistant_response(text)
if test("full_single_iteration", r.has_reasoning and r.has_function_calls and r.has_self_reflection): passed += 1
else: failed += 1

# Contents array structure
if test("contents_array", len(r.contents) == 4 and r.contents[0].type == 'reasoning'): passed += 1
else: failed += 1

# Two iterations
text = f"{MESSAGE_START}T1{MESSAGE_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}T2{MESSAGE_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}"
r = agent.parse_assistant_response(text)
if test("two_iterations", len(r.message_contents) == 2 and len(r.self_reflections) == 2): passed += 1
else: failed += 1

# Multiple iterations property
if test("has_multiple_iterations", r.has_multiple_iterations): passed += 1
else: failed += 1

# Two iterations with tool calls
text = f"{MESSAGE_START}T1{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.f1||>{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}T2{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.f2||>{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}"
r = agent.parse_assistant_response(text)
if test("two_iterations_with_tools", len(r.tool_call_blocks) == 2): passed += 1
else: failed += 1

# Tool call names in iterations
if test("tool_call_names", r.tool_call_blocks[0][0]['function']['name'] == 'f1' and r.tool_call_blocks[1][0]['function']['name'] == 'f2'): passed += 1
else: failed += 1

# Three iterations
text = f"{MESSAGE_START}T1{MESSAGE_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}T2{MESSAGE_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}{MESSAGE_START}T3{MESSAGE_END}{SELF_REFLECTION_START}R3{SELF_REFLECTION_END}"
r = agent.parse_assistant_response(text)
if test("three_iterations", len(r.message_contents) == 3): passed += 1
else: failed += 1

# Iteration count
if test("iteration_count", agent.get_iteration_count(text) == 3): passed += 1
else: failed += 1

# Parallel in multiple iterations
text = f"{MESSAGE_START}T1{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.a||>{INVOKE_END}<||invoke_start to=function.b||>{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}T2{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.c||>{INVOKE_END}<||invoke_start to=function.d||>{INVOKE_END}<||invoke_start to=function.e||>{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}"
r = agent.parse_assistant_response(text)
if test("parallel_multi_iter", len(r.tool_call_blocks[0]) == 2 and len(r.tool_call_blocks[1]) == 3): passed += 1
else: failed += 1

# Extract iteration
iter0 = agent.extract_iteration(text, 0)
if test("extract_iter_0", iter0['text'] == 'T1' and iter0['tool_calls'] is not None): passed += 1
else: failed += 1

iter1 = agent.extract_iteration(text, 1)
if test("extract_iter_1", iter1['text'] == 'T2'): passed += 1
else: failed += 1

# Get all tool calls grouped
grouped = agent.get_all_tool_calls(text)
if test("get_all_tool_calls", len(grouped) == 2 and grouped[0][0].name == 'a'): passed += 1
else: failed += 1

# Five iterations stress test
parts = []
for i in range(5):
    parts.append(f"{MESSAGE_START}T{i}{MESSAGE_END}")
    parts.append(f"{FUNCTIONS_START}<||invoke_start to=function.f{i}||>{INVOKE_END}{FUNCTIONS_END}")
    parts.append(f"{SELF_REFLECTION_START}R{i}{SELF_REFLECTION_END}")
text = ''.join(parts)
r = agent.parse_assistant_response(text)
if test("five_iterations", len(r.message_contents) == 5 and len(r.tool_call_blocks) == 5): passed += 1
else: failed += 1

# Reasoning only at start
text = f"{REASONING_START}Think{REASONING_END}{MESSAGE_START}T1{MESSAGE_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}T2{MESSAGE_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}"
r = agent.parse_assistant_response(text)
reasoning_count = sum(1 for c in r.contents if c.type == 'reasoning')
if test("reasoning_only_start", reasoning_count == 1 and r.contents[0].type == 'reasoning'): passed += 1
else: failed += 1

# to_contents_array
arr = r.to_contents_array()
if test("to_contents_array", len(arr) == 5 and arr[0]['type'] == 'reasoning'): passed += 1
else: failed += 1

# to_simple_format
simple = r.to_simple_format()
if test("to_simple_format", simple['role'] == 'assistant' and 'content' in simple): passed += 1
else: failed += 1

# Format methods
if test("format_message_block", agent.format_message_block("Hi") == f"{MESSAGE_START}Hi{MESSAGE_END}"): passed += 1
else: failed += 1

if test("format_reasoning_block", agent.format_reasoning_block("Think") == f"{REASONING_START}Think{REASONING_END}"): passed += 1
else: failed += 1

if test("format_self_reflection", agent.format_self_reflection_block("OK") == f"{SELF_REFLECTION_START}OK{SELF_REFLECTION_END}"): passed += 1
else: failed += 1

# Format constrain
constrain = agent.format_constrain_block(response_format={'strict': True}, reasoning_effort='high')
if test("format_constrain", "strict:true" in constrain and "effort: high" in constrain): passed += 1
else: failed += 1

# Format contents array
contents = [{"type": "text", "text": "Hi"}, {"type": "self_reflection", "self_reflection": "OK"}]
formatted = agent.format_contents_array(contents)
if test("format_contents", MESSAGE_START in formatted and SELF_REFLECTION_START in formatted): passed += 1
else: failed += 1

# Tool response parsing
text = f'{FUNCTION_RESULTS_START}<||function_response_start to=function.search id="r1"||>Result<||function_response_end||>{FUNCTION_RESULTS_END}'
results = agent.parse_tool_responses_from_text(text)
if test("tool_response_parse", results[0]['name'] == 'search' and results[0]['id'] == 'r1'): passed += 1
else: failed += 1

# Citation parsing
text = f'{CITATIONS_START}["s001","s002"]||>cited{CITATIONS_END}'
citations = agent.parse_citations(text)
if test("citation_parse", citations[0]['refs'] == ['s001', 's002']): passed += 1
else: failed += 1

# Check methods
if test("has_function_calls", agent.has_function_calls(f"{FUNCTIONS_START}{INVOKE_START}function.f||>{INVOKE_END}{FUNCTIONS_END}")): passed += 1
else: failed += 1

if test("has_reasoning", agent.has_reasoning(f"{REASONING_START}x{REASONING_END}")): passed += 1
else: failed += 1

if test("has_self_reflection", agent.has_self_reflection(f"{SELF_REFLECTION_START}x{SELF_REFLECTION_END}")): passed += 1
else: failed += 1

if test("has_citations", agent.has_citations(f'{CITATIONS_START}["s"]||>x{CITATIONS_END}')): passed += 1
else: failed += 1

# Edge cases
if test("empty_text", len(agent.parse_assistant_response("").contents) == 0): passed += 1
else: failed += 1

# Many parallel calls
calls = ''.join([f"<||invoke_start to=function.f{i}||>{INVOKE_END}" for i in range(20)])
text = f"{MESSAGE_START}Call{MESSAGE_END}{FUNCTIONS_START}{calls}{FUNCTIONS_END}"
r = agent.parse_assistant_response(text)
if test("20_parallel_calls", len(r.function_calls) == 20): passed += 1
else: failed += 1

# Final message
text = f"{MESSAGE_START}First{MESSAGE_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}Last{MESSAGE_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}"
r = agent.parse_assistant_response(text)
if test("final_message", r.final_message == "Last"): passed += 1
else: failed += 1

# ParsedContentItem
item = ParsedContentItem(type='text', content='Hello')
if test("content_item_to_dict", item.to_dict() == {'type': 'text', 'text': 'Hello'}): passed += 1
else: failed += 1

item = ParsedContentItem(type='tool_calls', content=[{'function': {'name': 'f'}}])
if test("tool_calls_to_dict", item.to_dict()['tool_calls'][0]['function']['name'] == 'f'): passed += 1
else: failed += 1

# Mixed iterations (some with tools, some without)
text = f"{MESSAGE_START}T1{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.f1||>{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}T2{MESSAGE_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}{MESSAGE_START}T3{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.f3||>{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}R3{SELF_REFLECTION_END}"
r = agent.parse_assistant_response(text)
if test("mixed_iterations", len(r.tool_call_blocks) == 2 and len(r.message_contents) == 3): passed += 1
else: failed += 1

# JSON parameters
import json
text = f'{MESSAGE_START}Call{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.api||><||parameter_start name=data||>{json.dumps({"key": "value"})}{PARAMETER_END}{INVOKE_END}{FUNCTIONS_END}'
r = agent.parse_assistant_response(text)
if test("json_parameter", r.function_calls[0]['function']['arguments']['data'] == {"key": "value"}): passed += 1
else: failed += 1

# Position tracking
text = f"{REASONING_START}R{REASONING_END}{MESSAGE_START}T{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.f||>{INVOKE_END}{FUNCTIONS_END}{SELF_REFLECTION_START}S{SELF_REFLECTION_END}"
r = agent.parse_assistant_response(text)
if test("position_tracking", r.contents[0].position == 0 and r.contents[3].position == 3): passed += 1
else: failed += 1

print(f"\n{'='*40}")
print(f"TOTAL: {passed + failed} tests")
print(f"PASSED: {passed}")
print(f"FAILED: {failed}")
print(f"{'='*40}")

sys.exit(0 if failed == 0 else 1)
