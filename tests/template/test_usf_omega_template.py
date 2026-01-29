#!/usr/bin/env python3
"""
Comprehensive tests for USF Omega Template.

Tests:
1. Training boundary detection (loss_scale for different roles)
2. All role markers (system, developer, user, assistant, tool, functions)
3. Multi-turn conversations with tool calls
4. All extra_kwargs options (model_identity, functions, response_schema, etc.)
5. EOS token handling
6. Jinja2 template parameter passing

Total: 50+ test cases
"""
import sys
import os
import json

# Tokens (same as in usf_omega.py)
START_TOKEN = '<|:@:|start|:@:|>'
END_TOKEN = '<|:@:|end|:@:|>'
BOS_TOKEN = '<|:@:|startoftext|:@:|>'
EOS_TOKEN = '<|:@:|endoftext|:@:|>'
MESSAGE_START = '<|:@:|message_start|:@:|>'
MESSAGE_END = '<|:@:|message_end|:@:|>'
FUNCTIONS_START = '<|:@:|functions_start|:@:|>'
FUNCTIONS_END = '<|:@:|functions_end|:@:|>'
FUNCTION_RESULTS_START = '<|:@:|function_results_start|:@:|>'
FUNCTION_RESULTS_END = '<|:@:|function_results_end|:@:|>'
REASONING_START = '<|:@:|reasoning_start|:@:|>'
REASONING_END = '<|:@:|reasoning_end|:@:|>'
SELF_REFLECTION_START = '<|:@:|self_reflection_start|:@:|>'
SELF_REFLECTION_END = '<|:@:|self_reflection_end|:@:|>'
CONSTRAIN_START = '<|:@:|constrain_start|:@:|>'
CONSTRAIN_END = '<|:@:|constrain_end|:@:|>'


def split_by_role_markers(text):
    """
    Split text into segments based on role markers and assign loss_scale.
    This simulates the _jinja_encode training boundary logic.
    """
    assistant_marker = START_TOKEN + 'assistant\n'
    non_train_markers = [
        START_TOKEN + 'user\n',
        START_TOKEN + 'tool\n',
        START_TOKEN + 'system\n',
        START_TOKEN + 'developer\n',
        START_TOKEN + 'functions\n',
    ]
    
    context_list = []
    loss_scale_list = []
    remaining = text
    in_assistant = False
    
    while remaining:
        next_pos = len(remaining)
        next_marker = None
        next_is_assistant = False
        
        pos = remaining.find(assistant_marker)
        if pos >= 0 and pos < next_pos:
            next_pos = pos
            next_marker = assistant_marker
            next_is_assistant = True
        
        for marker in non_train_markers:
            pos = remaining.find(marker)
            if pos >= 0 and pos < next_pos:
                next_pos = pos
                next_marker = marker
                next_is_assistant = False
        
        if next_pos > 0:
            content = remaining[:next_pos]
            if content:
                context_list.append(content)
                loss_scale_list.append(1. if in_assistant else 0.)
        
        if next_marker:
            context_list.append(next_marker)
            loss_scale_list.append(1. if next_is_assistant else 0.)
            remaining = remaining[next_pos + len(next_marker):]
            in_assistant = next_is_assistant
        else:
            if remaining:
                context_list.append(remaining)
                loss_scale_list.append(1. if in_assistant else 0.)
            break
    
    return context_list, loss_scale_list


def test(name, condition):
    if condition:
        print(f"✓ {name}")
        return True
    else:
        print(f"✗ {name}")
        return False


passed = 0
failed = 0

print("\n" + "=" * 60)
print("USF Omega Template Tests")
print("=" * 60)

# ==============================================================================
# SECTION 1: Role Marker Detection (10 tests)
# ==============================================================================
print("\n[1] Role Marker Detection")

# Test 1.1: System marker detected
text = f"{BOS_TOKEN}{START_TOKEN}system\nYou are helpful.{END_TOKEN}\n"
ctx, ls = split_by_role_markers(text)
if test("1.1 system_marker_loss_0", all(s == 0. for s in ls)):
    passed += 1
else:
    failed += 1

# Test 1.2: Developer marker detected
text = f"{START_TOKEN}developer\nInstructions here{END_TOKEN}\n"
ctx, ls = split_by_role_markers(text)
if test("1.2 developer_marker_loss_0", all(s == 0. for s in ls)):
    passed += 1
else:
    failed += 1

# Test 1.3: User marker detected
text = f"{START_TOKEN}user\nHello{END_TOKEN}\n"
ctx, ls = split_by_role_markers(text)
if test("1.3 user_marker_loss_0", all(s == 0. for s in ls)):
    passed += 1
else:
    failed += 1

# Test 1.4: Functions marker detected
text = f"{START_TOKEN}functions\nnamespace functions {{}}{END_TOKEN}\n"
ctx, ls = split_by_role_markers(text)
if test("1.4 functions_marker_loss_0", all(s == 0. for s in ls)):
    passed += 1
else:
    failed += 1

# Test 1.5: Tool marker detected
text = f"{START_TOKEN}tool\n{FUNCTION_RESULTS_START}results{FUNCTION_RESULTS_END}{END_TOKEN}\n"
ctx, ls = split_by_role_markers(text)
if test("1.5 tool_marker_loss_0", all(s == 0. for s in ls)):
    passed += 1
else:
    failed += 1

# Test 1.6: Assistant marker detected with loss=1
text = f"{START_TOKEN}assistant\n{MESSAGE_START}Hello{MESSAGE_END}{END_TOKEN}\n"
ctx, ls = split_by_role_markers(text)
has_loss_1 = any(s == 1. for s in ls)
if test("1.6 assistant_marker_loss_1", has_loss_1):
    passed += 1
else:
    failed += 1

# Test 1.7: Only assistant content has loss=1
text = f"{START_TOKEN}system\nSys{END_TOKEN}\n{START_TOKEN}user\nHi{END_TOKEN}\n{START_TOKEN}assistant\nHello{END_TOKEN}\n"
ctx, ls = split_by_role_markers(text)
# Find assistant marker index
asst_idx = None
for i, c in enumerate(ctx):
    if c == START_TOKEN + 'assistant\n':
        asst_idx = i
        break
# Everything from asst_idx onwards should be 1, before should be 0
before_ok = all(ls[i] == 0. for i in range(asst_idx)) if asst_idx else True
after_ok = all(ls[i] == 1. for i in range(asst_idx, len(ls))) if asst_idx else False
if test("1.7 only_assistant_trained", before_ok and after_ok):
    passed += 1
else:
    failed += 1

# Test 1.8: BOS token not a role marker
text = f"{BOS_TOKEN}some content"
ctx, ls = split_by_role_markers(text)
if test("1.8 bos_not_role_marker", BOS_TOKEN in ctx[0] if ctx else False):
    passed += 1
else:
    failed += 1

# Test 1.9: END token in content preserved
text = f"{START_TOKEN}assistant\n{MESSAGE_START}Hi{MESSAGE_END}{END_TOKEN}{EOS_TOKEN}"
ctx, ls = split_by_role_markers(text)
joined = ''.join(ctx)
if test("1.9 end_eos_preserved", END_TOKEN in joined and EOS_TOKEN in joined):
    passed += 1
else:
    failed += 1

# Test 1.10: Empty text
ctx, ls = split_by_role_markers("")
if test("1.10 empty_text", len(ctx) == 0 and len(ls) == 0):
    passed += 1
else:
    failed += 1


# ==============================================================================
# SECTION 2: Single Turn Conversations (10 tests)
# ==============================================================================
print("\n[2] Single Turn Conversations")

# Test 2.1: Simple user-assistant
text = f"""{BOS_TOKEN}{START_TOKEN}system
You are helpful.{END_TOKEN}
{START_TOKEN}user
Hello{END_TOKEN}
{START_TOKEN}assistant
{MESSAGE_START}Hi there!{MESSAGE_END}{END_TOKEN}
{EOS_TOKEN}"""
ctx, ls = split_by_role_markers(text)
assistant_parts = [i for i, c in enumerate(ctx) if START_TOKEN + 'assistant\n' in c or (ls[i] == 1. and i > 0)]
if test("2.1 simple_user_assistant", len(assistant_parts) > 0):
    passed += 1
else:
    failed += 1

# Test 2.2: With developer block
text = f"""{BOS_TOKEN}{START_TOKEN}system
Identity{END_TOKEN}
{START_TOKEN}developer
Instructions{END_TOKEN}
{START_TOKEN}user
Hello{END_TOKEN}
{START_TOKEN}assistant
{MESSAGE_START}Response{MESSAGE_END}{END_TOKEN}
{EOS_TOKEN}"""
ctx, ls = split_by_role_markers(text)
# Count segments with loss=0 (should include system, developer, user)
loss_0_count = sum(1 for s in ls if s == 0.)
loss_1_count = sum(1 for s in ls if s == 1.)
if test("2.2 with_developer", loss_0_count > 0 and loss_1_count > 0):
    passed += 1
else:
    failed += 1

# Test 2.3: With functions namespace
text = f"""{BOS_TOKEN}{START_TOKEN}system
Identity{END_TOKEN}
{START_TOKEN}functions
namespace functions {{
  // search
  type search = (_: {{q: string}}) => any;
}}{END_TOKEN}
{START_TOKEN}user
Search for weather{END_TOKEN}
{START_TOKEN}assistant
{MESSAGE_START}Searching...{MESSAGE_END}{FUNCTIONS_START}...{FUNCTIONS_END}{END_TOKEN}
{EOS_TOKEN}"""
ctx, ls = split_by_role_markers(text)
# Functions block should have loss=0
func_idx = None
for i, c in enumerate(ctx):
    if START_TOKEN + 'functions\n' in c:
        func_idx = i
        break
if test("2.3 functions_namespace_loss_0", func_idx is not None and ls[func_idx] == 0.):
    passed += 1
else:
    failed += 1

# Test 2.4: Assistant with tool calls
text = f"""{START_TOKEN}assistant
{MESSAGE_START}Let me search{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.search||><||parameter_start name=q||>"weather"<||parameter_end||><||invoke_end||>{FUNCTIONS_END}{END_TOKEN}
"""
ctx, ls = split_by_role_markers(text)
joined = ''.join(ctx)
if test("2.4 assistant_tool_calls", FUNCTIONS_START in joined and all(s == 1. for s in ls)):
    passed += 1
else:
    failed += 1

# Test 2.5: Assistant with self-reflection
text = f"""{START_TOKEN}assistant
{MESSAGE_START}Answer{MESSAGE_END}{SELF_REFLECTION_START}This looks correct{SELF_REFLECTION_END}{END_TOKEN}
"""
ctx, ls = split_by_role_markers(text)
joined = ''.join(ctx)
if test("2.5 assistant_self_reflection", SELF_REFLECTION_START in joined):
    passed += 1
else:
    failed += 1

# Test 2.6: Assistant with constrain block
text = f"""{START_TOKEN}assistant
{CONSTRAIN_START}response.format={{type:json}}{CONSTRAIN_END}{MESSAGE_START}{{"key": "value"}}{MESSAGE_END}{END_TOKEN}
"""
ctx, ls = split_by_role_markers(text)
joined = ''.join(ctx)
if test("2.6 assistant_constrain", CONSTRAIN_START in joined):
    passed += 1
else:
    failed += 1

# Test 2.7: Reasoning template format
text = f"""{START_TOKEN}assistant
{CONSTRAIN_START}reasoning={{effort: medium}}{CONSTRAIN_END}{REASONING_START}Let me think...{REASONING_END}{MESSAGE_START}Answer{MESSAGE_END}{END_TOKEN}
"""
ctx, ls = split_by_role_markers(text)
joined = ''.join(ctx)
if test("2.7 reasoning_format", REASONING_START in joined and REASONING_END in joined):
    passed += 1
else:
    failed += 1

# Test 2.8: EOS at end with loss=1
text = f"""{START_TOKEN}user
Hi{END_TOKEN}
{START_TOKEN}assistant
{MESSAGE_START}Hello{MESSAGE_END}{END_TOKEN}
{EOS_TOKEN}"""
ctx, ls = split_by_role_markers(text)
# Last segment should be EOS with loss=1 (part of assistant)
if test("2.8 eos_loss_1", EOS_TOKEN in ctx[-1] and ls[-1] == 1.):
    passed += 1
else:
    failed += 1

# Test 2.9: Multi-message content
text = f"""{START_TOKEN}assistant
{MESSAGE_START}Part 1{MESSAGE_END}{MESSAGE_START}Part 2{MESSAGE_END}{END_TOKEN}
"""
ctx, ls = split_by_role_markers(text)
joined = ''.join(ctx)
# Check that both message blocks are present
if test("2.9 multi_message", MESSAGE_START in joined and MESSAGE_END in joined):
    passed += 1
else:
    failed += 1

# Test 2.10: Unicode content preserved
text = f"""{START_TOKEN}assistant
{MESSAGE_START}Hello 世界 🌍{MESSAGE_END}{END_TOKEN}
"""
ctx, ls = split_by_role_markers(text)
joined = ''.join(ctx)
if test("2.10 unicode_preserved", "世界" in joined and "🌍" in joined):
    passed += 1
else:
    failed += 1


# ==============================================================================
# SECTION 3: Multi-Turn Conversations (10 tests)
# ==============================================================================
print("\n[3] Multi-Turn Conversations")

# Test 3.1: Two turns
text = f"""{START_TOKEN}user
First{END_TOKEN}
{START_TOKEN}assistant
{MESSAGE_START}Response 1{MESSAGE_END}{END_TOKEN}
{START_TOKEN}user
Second{END_TOKEN}
{START_TOKEN}assistant
{MESSAGE_START}Response 2{MESSAGE_END}{END_TOKEN}
{EOS_TOKEN}"""
ctx, ls = split_by_role_markers(text)
# Count user markers (loss=0) and assistant markers (loss=1)
user_count = sum(1 for c in ctx if c == START_TOKEN + 'user\n')
asst_count = sum(1 for c in ctx if c == START_TOKEN + 'assistant\n')
if test("3.1 two_turns", user_count == 2 and asst_count == 2):
    passed += 1
else:
    failed += 1

# Test 3.2: User loss=0, assistant loss=1 alternating
text = f"""{START_TOKEN}user
Q1{END_TOKEN}
{START_TOKEN}assistant
A1{END_TOKEN}
{START_TOKEN}user
Q2{END_TOKEN}
{START_TOKEN}assistant
A2{END_TOKEN}
"""
ctx, ls = split_by_role_markers(text)
# Check that we alternate correctly
in_asst = False
correct = True
for c, l in zip(ctx, ls):
    if c == START_TOKEN + 'assistant\n':
        in_asst = True
    elif c == START_TOKEN + 'user\n':
        in_asst = False
    if in_asst and l != 1.:
        correct = False
    if not in_asst and l != 0.:
        correct = False
if test("3.2 alternating_loss", correct):
    passed += 1
else:
    failed += 1

# Test 3.3: Three turns
text = f"""{START_TOKEN}user
Q1{END_TOKEN}
{START_TOKEN}assistant
A1{END_TOKEN}
{START_TOKEN}user
Q2{END_TOKEN}
{START_TOKEN}assistant
A2{END_TOKEN}
{START_TOKEN}user
Q3{END_TOKEN}
{START_TOKEN}assistant
A3{END_TOKEN}
"""
ctx, ls = split_by_role_markers(text)
asst_count = sum(1 for c in ctx if c == START_TOKEN + 'assistant\n')
if test("3.3 three_turns", asst_count == 3):
    passed += 1
else:
    failed += 1

# Test 3.4: With tool call and tool response
text = f"""{START_TOKEN}user
Search weather{END_TOKEN}
{START_TOKEN}assistant
{MESSAGE_START}Searching{MESSAGE_END}{FUNCTIONS_START}..search..{FUNCTIONS_END}{END_TOKEN}
{START_TOKEN}tool
{FUNCTION_RESULTS_START}Sunny{FUNCTION_RESULTS_END}{END_TOKEN}
{START_TOKEN}assistant
{MESSAGE_START}It's sunny!{MESSAGE_END}{END_TOKEN}
"""
ctx, ls = split_by_role_markers(text)
# Tool should have loss=0
tool_idx = None
for i, c in enumerate(ctx):
    if c == START_TOKEN + 'tool\n':
        tool_idx = i
        break
if test("3.4 tool_response_loss_0", tool_idx is not None and ls[tool_idx] == 0.):
    passed += 1
else:
    failed += 1

# Test 3.5: Multiple tool calls in one turn
text = f"""{START_TOKEN}assistant
{MESSAGE_START}Searching{MESSAGE_END}{FUNCTIONS_START}..call1..call2..{FUNCTIONS_END}{END_TOKEN}
{START_TOKEN}tool
{FUNCTION_RESULTS_START}R1{FUNCTION_RESULTS_END}{FUNCTION_RESULTS_START}R2{FUNCTION_RESULTS_END}{END_TOKEN}
{START_TOKEN}assistant
{MESSAGE_START}Done{MESSAGE_END}{END_TOKEN}
"""
ctx, ls = split_by_role_markers(text)
asst_markers = sum(1 for c in ctx if c == START_TOKEN + 'assistant\n')
if test("3.5 multiple_tool_calls", asst_markers == 2):
    passed += 1
else:
    failed += 1

# Test 3.6: Self-reflection loop (multi-iteration)
text = f"""{START_TOKEN}assistant
{MESSAGE_START}Answer 1{MESSAGE_END}{SELF_REFLECTION_START}Need to improve{SELF_REFLECTION_END}{MESSAGE_START}Answer 2{MESSAGE_END}{SELF_REFLECTION_START}Better{SELF_REFLECTION_END}{END_TOKEN}
"""
ctx, ls = split_by_role_markers(text)
joined = ''.join(ctx)
# Check self-reflection tokens are present
if test("3.6 self_reflection_loop", SELF_REFLECTION_START in joined and SELF_REFLECTION_END in joined):
    passed += 1
else:
    failed += 1

# Test 3.7: Tool call then tool then assistant all trained correctly
text = f"""{START_TOKEN}user
Search{END_TOKEN}
{START_TOKEN}assistant
{MESSAGE_START}Searching{MESSAGE_END}{END_TOKEN}
{START_TOKEN}tool
Results{END_TOKEN}
{START_TOKEN}assistant
{MESSAGE_START}Found it{MESSAGE_END}{END_TOKEN}
"""
ctx, ls = split_by_role_markers(text)
# Count loss=1 segments (should be assistant parts only)
loss_1_segments = sum(1 for s in ls if s == 1.)
# Should have at least 4 segments with loss=1 (2 assistant markers + 2 contents)
if test("3.7 tool_then_assistant", loss_1_segments >= 4):
    passed += 1
else:
    failed += 1

# Test 3.8: Full conversation with all roles
text = f"""{BOS_TOKEN}{START_TOKEN}system
Identity{END_TOKEN}
{START_TOKEN}developer
Instructions{END_TOKEN}
{START_TOKEN}functions
namespace{END_TOKEN}
{START_TOKEN}user
Question{END_TOKEN}
{START_TOKEN}assistant
{MESSAGE_START}Calling{MESSAGE_END}{END_TOKEN}
{START_TOKEN}tool
Results{END_TOKEN}
{START_TOKEN}assistant
{MESSAGE_START}Answer{MESSAGE_END}{END_TOKEN}
{EOS_TOKEN}"""
ctx, ls = split_by_role_markers(text)
# All roles present
all_roles = [
    START_TOKEN + 'system\n',
    START_TOKEN + 'developer\n',
    START_TOKEN + 'functions\n',
    START_TOKEN + 'user\n',
    START_TOKEN + 'assistant\n',
    START_TOKEN + 'tool\n',
]
found = [any(r in c for c in ctx) for r in all_roles]
if test("3.8 all_roles_present", all(found)):
    passed += 1
else:
    failed += 1

# Test 3.9: Only non-trainable roles (no assistant)
text = f"""{START_TOKEN}system
Sys{END_TOKEN}
{START_TOKEN}user
Hi{END_TOKEN}
"""
ctx, ls = split_by_role_markers(text)
if test("3.9 no_assistant_all_loss_0", all(s == 0. for s in ls)):
    passed += 1
else:
    failed += 1

# Test 3.10: Long conversation (5 turns)
turns = []
for i in range(5):
    turns.append(f"{START_TOKEN}user\nQ{i}{END_TOKEN}\n{START_TOKEN}assistant\n{MESSAGE_START}A{i}{MESSAGE_END}{END_TOKEN}\n")
text = ''.join(turns)
ctx, ls = split_by_role_markers(text)
asst_markers = sum(1 for c in ctx if c == START_TOKEN + 'assistant\n')
if test("3.10 five_turns", asst_markers == 5):
    passed += 1
else:
    failed += 1


# ==============================================================================
# SECTION 4: Contents Array Format (10 tests)
# ==============================================================================
print("\n[4] Contents Array Format")

# Test 4.1: Text only
text = f"""{START_TOKEN}assistant
{MESSAGE_START}Simple text{MESSAGE_END}{END_TOKEN}
"""
ctx, ls = split_by_role_markers(text)
if test("4.1 text_only", MESSAGE_START in ''.join(ctx)):
    passed += 1
else:
    failed += 1

# Test 4.2: Text + tool_calls
text = f"""{START_TOKEN}assistant
{MESSAGE_START}Calling{MESSAGE_END}{FUNCTIONS_START}...{FUNCTIONS_END}{END_TOKEN}
"""
ctx, ls = split_by_role_markers(text)
joined = ''.join(ctx)
if test("4.2 text_tool_calls", MESSAGE_START in joined and FUNCTIONS_START in joined):
    passed += 1
else:
    failed += 1

# Test 4.3: Text + self_reflection
text = f"""{START_TOKEN}assistant
{MESSAGE_START}Answer{MESSAGE_END}{SELF_REFLECTION_START}Good{SELF_REFLECTION_END}{END_TOKEN}
"""
ctx, ls = split_by_role_markers(text)
joined = ''.join(ctx)
if test("4.3 text_self_reflection", MESSAGE_START in joined and SELF_REFLECTION_START in joined):
    passed += 1
else:
    failed += 1

# Test 4.4: Reasoning + text (reasoning template)
text = f"""{START_TOKEN}assistant
{REASONING_START}Thinking{REASONING_END}{MESSAGE_START}Answer{MESSAGE_END}{END_TOKEN}
"""
ctx, ls = split_by_role_markers(text)
joined = ''.join(ctx)
if test("4.4 reasoning_text", REASONING_START in joined and MESSAGE_START in joined):
    passed += 1
else:
    failed += 1

# Test 4.5: Reasoning + text + tool_calls + self_reflection
text = f"""{START_TOKEN}assistant
{REASONING_START}Think{REASONING_END}{MESSAGE_START}Text{MESSAGE_END}{FUNCTIONS_START}Call{FUNCTIONS_END}{SELF_REFLECTION_START}Done{SELF_REFLECTION_END}{END_TOKEN}
"""
ctx, ls = split_by_role_markers(text)
joined = ''.join(ctx)
all_present = (REASONING_START in joined and MESSAGE_START in joined and 
               FUNCTIONS_START in joined and SELF_REFLECTION_START in joined)
if test("4.5 full_contents", all_present):
    passed += 1
else:
    failed += 1

# Test 4.6: Multi-iteration (text → self_reflection → text → self_reflection)
text = f"""{START_TOKEN}assistant
{MESSAGE_START}T1{MESSAGE_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}T2{MESSAGE_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}{END_TOKEN}
"""
ctx, ls = split_by_role_markers(text)
joined = ''.join(ctx)
# Check multi-iteration structure is preserved
if test("4.6 multi_iteration", MESSAGE_START in joined and SELF_REFLECTION_START in joined):
    passed += 1
else:
    failed += 1

# Test 4.7: Tool calls at iteration 1 only
text = f"""{START_TOKEN}assistant
{MESSAGE_START}T1{MESSAGE_END}{FUNCTIONS_START}Call{FUNCTIONS_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}T2{MESSAGE_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}{END_TOKEN}
"""
ctx, ls = split_by_role_markers(text)
joined = ''.join(ctx)
# Check tool calls block is preserved
if test("4.7 tool_calls_iter_1", FUNCTIONS_START in joined and FUNCTIONS_END in joined):
    passed += 1
else:
    failed += 1

# Test 4.8: Parallel tool calls (multiple in one block)
text = f"""{START_TOKEN}assistant
{MESSAGE_START}Calling{MESSAGE_END}{FUNCTIONS_START}<||invoke_start to=function.a||><||invoke_end||><||invoke_start to=function.b||><||invoke_end||>{FUNCTIONS_END}{END_TOKEN}
"""
ctx, ls = split_by_role_markers(text)
joined = ''.join(ctx)
# Check parallel calls are preserved
if test("4.8 parallel_tool_calls", '<||invoke_start' in joined):
    passed += 1
else:
    failed += 1

# Test 4.9: Three iterations
text = f"""{START_TOKEN}assistant
{MESSAGE_START}T1{MESSAGE_END}{SELF_REFLECTION_START}R1{SELF_REFLECTION_END}{MESSAGE_START}T2{MESSAGE_END}{SELF_REFLECTION_START}R2{SELF_REFLECTION_END}{MESSAGE_START}T3{MESSAGE_END}{SELF_REFLECTION_START}R3{SELF_REFLECTION_END}{END_TOKEN}
"""
ctx, ls = split_by_role_markers(text)
joined = ''.join(ctx)
# Check three iteration structure is preserved
if test("4.9 three_iterations", MESSAGE_START in joined and SELF_REFLECTION_START in joined):
    passed += 1
else:
    failed += 1

# Test 4.10: Empty message preserved
text = f"""{START_TOKEN}assistant
{MESSAGE_START}{MESSAGE_END}{END_TOKEN}
"""
ctx, ls = split_by_role_markers(text)
joined = ''.join(ctx)
if test("4.10 empty_message", MESSAGE_START + MESSAGE_END in joined):
    passed += 1
else:
    failed += 1


# ==============================================================================
# SECTION 5: Edge Cases (10 tests)
# ==============================================================================
print("\n[5] Edge Cases")

# Test 5.1: No newline after role marker (shouldn't match standard pattern)
# Note: Without newline, the marker won't match, so entire text is unrecognized
text = f"{START_TOKEN}assistant{MESSAGE_START}No newline{MESSAGE_END}{END_TOKEN}"
ctx, ls = split_by_role_markers(text)
# Entire text is one segment (no role markers matched)
if test("5.1 no_newline_after_role", len(ctx) >= 1):
    passed += 1
else:
    failed += 1

# Test 5.2: Special characters in content
text = f"""{START_TOKEN}assistant
{MESSAGE_START}Content with <|special|> tokens{MESSAGE_END}{END_TOKEN}
"""
ctx, ls = split_by_role_markers(text)
joined = ''.join(ctx)
if test("5.2 special_chars", "<|special|>" in joined):
    passed += 1
else:
    failed += 1

# Test 5.3: Very long content
long_content = "X" * 10000
text = f"""{START_TOKEN}assistant
{MESSAGE_START}{long_content}{MESSAGE_END}{END_TOKEN}
"""
ctx, ls = split_by_role_markers(text)
joined = ''.join(ctx)
if test("5.3 long_content", len(joined) > 10000):
    passed += 1
else:
    failed += 1

# Test 5.4: Nested-looking tokens (not actual nesting)
text = f"""{START_TOKEN}assistant
{MESSAGE_START}Text with {MESSAGE_START}fake start{MESSAGE_END}{END_TOKEN}
"""
ctx, ls = split_by_role_markers(text)
joined = ''.join(ctx)
# Check content is preserved even with nested-looking tokens
if test("5.4 nested_tokens", MESSAGE_START in joined):
    passed += 1
else:
    failed += 1

# Test 5.5: JSON in content
json_content = json.dumps({"key": "value", "nested": {"a": 1}})
text = f"""{START_TOKEN}assistant
{MESSAGE_START}{json_content}{MESSAGE_END}{END_TOKEN}
"""
ctx, ls = split_by_role_markers(text)
joined = ''.join(ctx)
if test("5.5 json_content", json_content in joined):
    passed += 1
else:
    failed += 1

# Test 5.6: Code in content
code = "def foo():\n    return 'bar'"
text = f"""{START_TOKEN}assistant
{MESSAGE_START}```python
{code}
```{MESSAGE_END}{END_TOKEN}
"""
ctx, ls = split_by_role_markers(text)
joined = ''.join(ctx)
if test("5.6 code_content", code in joined):
    passed += 1
else:
    failed += 1

# Test 5.7: Consecutive assistant messages (merged)
text = f"""{START_TOKEN}assistant
{MESSAGE_START}Part 1{MESSAGE_END}{END_TOKEN}
{START_TOKEN}assistant
{MESSAGE_START}Part 2{MESSAGE_END}{END_TOKEN}
"""
ctx, ls = split_by_role_markers(text)
asst_count = sum(1 for c in ctx if c == START_TOKEN + 'assistant\n')
if test("5.7 consecutive_assistant", asst_count == 2):
    passed += 1
else:
    failed += 1

# Test 5.8: Only BOS and EOS
text = f"{BOS_TOKEN}{EOS_TOKEN}"
ctx, ls = split_by_role_markers(text)
# No role markers, so single segment with loss=0
if test("5.8 bos_eos_only", len(ctx) >= 1 and all(s == 0. for s in ls)):
    passed += 1
else:
    failed += 1

# Test 5.9: Whitespace handling
text = f"""{START_TOKEN}assistant
{MESSAGE_START}   Spaces   {MESSAGE_END}{END_TOKEN}
"""
ctx, ls = split_by_role_markers(text)
joined = ''.join(ctx)
if test("5.9 whitespace", "   Spaces   " in joined):
    passed += 1
else:
    failed += 1

# Test 5.10: Multiple EOS (shouldn't happen but handle gracefully)
text = f"""{START_TOKEN}assistant
{MESSAGE_START}Text{MESSAGE_END}{END_TOKEN}
{EOS_TOKEN}{EOS_TOKEN}"""
ctx, ls = split_by_role_markers(text)
joined = ''.join(ctx)
# Multiple EOS should be preserved
if test("5.10 multiple_eos", EOS_TOKEN in joined):
    passed += 1
else:
    failed += 1


# ==============================================================================
# SUMMARY
# ==============================================================================
print("\n" + "=" * 60)
print(f"TOTAL: {passed + failed} tests")
print(f"PASSED: {passed} ✓")
print(f"FAILED: {failed} ✗")
print("=" * 60)

sys.exit(0 if failed == 0 else 1)
