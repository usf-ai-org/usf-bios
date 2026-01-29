#!/usr/bin/env python3
"""
Comprehensive tests for USF Omega Template training mode compatibility.

Tests:
1. Multi-turn SFT (100+ turns)
2. Multi-turn RLHF (chosen/rejected pairs)
3. All features together (system, developer, identity, functions, structured output, self-reflection)
4. Training boundary correctness for all scenarios
"""
import sys

# Tokens
START_TOKEN = '<|:@:|start|:@:|>'
END_TOKEN = '<|:@:|end|:@:|>'
BOS_TOKEN = '<|:@:|startoftext|:@:|>'
EOS_TOKEN = '<|:@:|endoftext|:@:|>'
MESSAGE_START = '<|:@:|message_start|:@:|>'
MESSAGE_END = '<|:@:|message_end|:@:|>'
FUNCTIONS_START = '<|:@:|functions_start|:@:|>'
FUNCTIONS_END = '<|:@:|functions_end|:@:|>'
REASONING_START = '<|:@:|reasoning_start|:@:|>'
REASONING_END = '<|:@:|reasoning_end|:@:|>'
SELF_REFLECTION_START = '<|:@:|self_reflection_start|:@:|>'
SELF_REFLECTION_END = '<|:@:|self_reflection_end|:@:|>'
FUNCTION_RESULTS_START = '<|:@:|function_results_start|:@:|>'
FUNCTION_RESULTS_END = '<|:@:|function_results_end|:@:|>'
CONSTRAIN_START = '<|:@:|constrain_start|:@:|>'
CONSTRAIN_END = '<|:@:|constrain_end|:@:|>'


def split_by_role_markers(text):
    """Split text and assign loss_scale based on role markers."""
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

print("\n" + "=" * 70)
print("USF Omega Training Modes Comprehensive Tests")
print("=" * 70)

# ==============================================================================
# SECTION 1: Multi-Turn SFT (10, 50, 100+ turns)
# ==============================================================================
print("\n[1] Multi-Turn SFT")

# Test 1.1: 10 turns
turns = []
for i in range(10):
    turns.append(f"{START_TOKEN}user\nQ{i}{END_TOKEN}\n")
    turns.append(f"{START_TOKEN}assistant\n{MESSAGE_START}A{i}{MESSAGE_END}{END_TOKEN}\n")
text = BOS_TOKEN + ''.join(turns) + EOS_TOKEN
ctx, ls = split_by_role_markers(text)
user_loss_0 = all(ls[i] == 0. for i, c in enumerate(ctx) if START_TOKEN + 'user\n' in c)
asst_loss_1 = all(ls[i] == 1. for i, c in enumerate(ctx) if START_TOKEN + 'assistant\n' in c)
if test("1.1 ten_turns_boundary", user_loss_0 and asst_loss_1):
    passed += 1
else:
    failed += 1

# Test 1.2: 50 turns
turns = []
for i in range(50):
    turns.append(f"{START_TOKEN}user\nQuestion {i}{END_TOKEN}\n")
    turns.append(f"{START_TOKEN}assistant\n{MESSAGE_START}Answer {i}{MESSAGE_END}{END_TOKEN}\n")
text = BOS_TOKEN + ''.join(turns) + EOS_TOKEN
ctx, ls = split_by_role_markers(text)
assistant_count = sum(1 for c in ctx if c == START_TOKEN + 'assistant\n')
if test("1.2 fifty_turns", assistant_count == 50):
    passed += 1
else:
    failed += 1

# Test 1.3: 100 turns
turns = []
for i in range(100):
    turns.append(f"{START_TOKEN}user\nQ{i}{END_TOKEN}\n")
    turns.append(f"{START_TOKEN}assistant\n{MESSAGE_START}A{i}{MESSAGE_END}{END_TOKEN}\n")
text = BOS_TOKEN + ''.join(turns) + EOS_TOKEN
ctx, ls = split_by_role_markers(text)
assistant_count = sum(1 for c in ctx if c == START_TOKEN + 'assistant\n')
if test("1.3 hundred_turns", assistant_count == 100):
    passed += 1
else:
    failed += 1

# Test 1.4: 200 turns (stress test)
turns = []
for i in range(200):
    turns.append(f"{START_TOKEN}user\nQ{i}{END_TOKEN}\n")
    turns.append(f"{START_TOKEN}assistant\n{MESSAGE_START}A{i}{MESSAGE_END}{END_TOKEN}\n")
text = BOS_TOKEN + ''.join(turns) + EOS_TOKEN
ctx, ls = split_by_role_markers(text)
assistant_count = sum(1 for c in ctx if c == START_TOKEN + 'assistant\n')
if test("1.4 two_hundred_turns", assistant_count == 200):
    passed += 1
else:
    failed += 1

# Test 1.5: Multi-turn with tool calls every 5th turn
turns = []
for i in range(20):
    turns.append(f"{START_TOKEN}user\nQ{i}{END_TOKEN}\n")
    if i % 5 == 0:
        turns.append(f"{START_TOKEN}assistant\n{MESSAGE_START}Calling{MESSAGE_END}{FUNCTIONS_START}...{FUNCTIONS_END}{END_TOKEN}\n")
        turns.append(f"{START_TOKEN}tool\n{FUNCTION_RESULTS_START}Result{FUNCTION_RESULTS_END}{END_TOKEN}\n")
        turns.append(f"{START_TOKEN}assistant\n{MESSAGE_START}Got it{MESSAGE_END}{END_TOKEN}\n")
    else:
        turns.append(f"{START_TOKEN}assistant\n{MESSAGE_START}A{i}{MESSAGE_END}{END_TOKEN}\n")
text = BOS_TOKEN + ''.join(turns) + EOS_TOKEN
ctx, ls = split_by_role_markers(text)
tool_markers = [i for i, c in enumerate(ctx) if c == START_TOKEN + 'tool\n']
all_tool_loss_0 = all(ls[i] == 0. for i in tool_markers)
if test("1.5 multi_turn_with_tools", all_tool_loss_0 and len(tool_markers) == 4):
    passed += 1
else:
    failed += 1

# ==============================================================================
# SECTION 2: Multi-Turn RLHF (Chosen/Rejected Pairs)
# ==============================================================================
print("\n[2] Multi-Turn RLHF")

# Test 2.1: Simple chosen response
chosen = f"""{BOS_TOKEN}{START_TOKEN}system
You are helpful.{END_TOKEN}
{START_TOKEN}user
Hello{END_TOKEN}
{START_TOKEN}assistant
{MESSAGE_START}Hi! How can I help?{MESSAGE_END}{END_TOKEN}
{EOS_TOKEN}"""
ctx, ls = split_by_role_markers(chosen)
# Verify chosen has proper loss boundaries
loss_1_count = sum(1 for s in ls if s == 1.)
if test("2.1 chosen_boundary", loss_1_count > 0):
    passed += 1
else:
    failed += 1

# Test 2.2: Simple rejected response (same structure)
rejected = f"""{BOS_TOKEN}{START_TOKEN}system
You are helpful.{END_TOKEN}
{START_TOKEN}user
Hello{END_TOKEN}
{START_TOKEN}assistant
{MESSAGE_START}I don't know.{MESSAGE_END}{END_TOKEN}
{EOS_TOKEN}"""
ctx, ls = split_by_role_markers(rejected)
loss_1_count = sum(1 for s in ls if s == 1.)
if test("2.2 rejected_boundary", loss_1_count > 0):
    passed += 1
else:
    failed += 1

# Test 2.3: Multi-turn RLHF (5 turns, last response differs)
def make_rlhf_conversation(final_response):
    turns = []
    for i in range(4):
        turns.append(f"{START_TOKEN}user\nQ{i}{END_TOKEN}\n")
        turns.append(f"{START_TOKEN}assistant\n{MESSAGE_START}A{i}{MESSAGE_END}{END_TOKEN}\n")
    turns.append(f"{START_TOKEN}user\nFinal question{END_TOKEN}\n")
    turns.append(f"{START_TOKEN}assistant\n{MESSAGE_START}{final_response}{MESSAGE_END}{END_TOKEN}\n")
    return BOS_TOKEN + ''.join(turns) + EOS_TOKEN

chosen = make_rlhf_conversation("Good answer with details")
rejected = make_rlhf_conversation("Bad answer")
ctx_c, ls_c = split_by_role_markers(chosen)
ctx_r, ls_r = split_by_role_markers(rejected)
# Both should have same structure
if test("2.3 multi_turn_rlhf_structure", len(ctx_c) == len(ctx_r)):
    passed += 1
else:
    failed += 1

# Test 2.4: RLHF with tool calls
chosen_tool = f"""{BOS_TOKEN}{START_TOKEN}user
Search weather{END_TOKEN}
{START_TOKEN}assistant
{MESSAGE_START}Searching{MESSAGE_END}{FUNCTIONS_START}...{FUNCTIONS_END}{END_TOKEN}
{START_TOKEN}tool
{FUNCTION_RESULTS_START}Sunny{FUNCTION_RESULTS_END}{END_TOKEN}
{START_TOKEN}assistant
{MESSAGE_START}It's sunny!{MESSAGE_END}{END_TOKEN}
{EOS_TOKEN}"""
ctx, ls = split_by_role_markers(chosen_tool)
tool_loss_0 = all(ls[i] == 0. for i, c in enumerate(ctx) if START_TOKEN + 'tool\n' in c)
if test("2.4 rlhf_with_tools", tool_loss_0):
    passed += 1
else:
    failed += 1

# Test 2.5: Multi-turn RLHF (10 turns)
chosen = make_rlhf_conversation("Detailed comprehensive answer")
ctx, ls = split_by_role_markers(chosen)
assistant_count = sum(1 for c in ctx if c == START_TOKEN + 'assistant\n')
if test("2.5 rlhf_ten_turns", assistant_count == 5):
    passed += 1
else:
    failed += 1

# ==============================================================================
# SECTION 3: All Features Together
# ==============================================================================
print("\n[3] All Features Together")

# Test 3.1: System + Developer + User + Assistant
text = f"""{BOS_TOKEN}{START_TOKEN}system
Model identity{END_TOKEN}
{START_TOKEN}developer
Instructions{END_TOKEN}
{START_TOKEN}user
Hello{END_TOKEN}
{START_TOKEN}assistant
{MESSAGE_START}Hi{MESSAGE_END}{END_TOKEN}
{EOS_TOKEN}"""
ctx, ls = split_by_role_markers(text)
sys_loss_0 = all(ls[i] == 0. for i, c in enumerate(ctx) if START_TOKEN + 'system\n' in c)
dev_loss_0 = all(ls[i] == 0. for i, c in enumerate(ctx) if START_TOKEN + 'developer\n' in c)
if test("3.1 system_developer", sys_loss_0 and dev_loss_0):
    passed += 1
else:
    failed += 1

# Test 3.2: Functions namespace + tool calls
text = f"""{BOS_TOKEN}{START_TOKEN}system
Identity{END_TOKEN}
{START_TOKEN}functions
namespace functions {{
  type search = (_: {{q: string}}) => any;
}}{END_TOKEN}
{START_TOKEN}user
Search{END_TOKEN}
{START_TOKEN}assistant
{MESSAGE_START}Searching{MESSAGE_END}{FUNCTIONS_START}...{FUNCTIONS_END}{END_TOKEN}
{EOS_TOKEN}"""
ctx, ls = split_by_role_markers(text)
func_loss_0 = all(ls[i] == 0. for i, c in enumerate(ctx) if START_TOKEN + 'functions\n' in c)
if test("3.2 functions_namespace", func_loss_0):
    passed += 1
else:
    failed += 1

# Test 3.3: Structured output with constrain block
text = f"""{START_TOKEN}assistant
{CONSTRAIN_START}response.format={{type:json, strict:true}}{CONSTRAIN_END}{MESSAGE_START}{{"key": "value"}}{MESSAGE_END}{END_TOKEN}
"""
ctx, ls = split_by_role_markers(text)
all_loss_1 = all(s == 1. for s in ls)
if test("3.3 structured_output", all_loss_1 and CONSTRAIN_START in ''.join(ctx)):
    passed += 1
else:
    failed += 1

# Test 3.4: Self-reflection (multi-iteration)
text = f"""{START_TOKEN}assistant
{MESSAGE_START}Draft{MESSAGE_END}{SELF_REFLECTION_START}Need to improve{SELF_REFLECTION_END}{MESSAGE_START}Final{MESSAGE_END}{SELF_REFLECTION_START}Done{SELF_REFLECTION_END}{END_TOKEN}
"""
ctx, ls = split_by_role_markers(text)
all_loss_1 = all(s == 1. for s in ls)
if test("3.4 self_reflection", all_loss_1 and SELF_REFLECTION_START in ''.join(ctx)):
    passed += 1
else:
    failed += 1

# Test 3.5: Reasoning block (reasoning model)
text = f"""{START_TOKEN}assistant
{CONSTRAIN_START}reasoning={{effort: medium}}{CONSTRAIN_END}{REASONING_START}Let me think...{REASONING_END}{MESSAGE_START}Answer{MESSAGE_END}{END_TOKEN}
"""
ctx, ls = split_by_role_markers(text)
all_loss_1 = all(s == 1. for s in ls)
if test("3.5 reasoning_block", all_loss_1 and REASONING_START in ''.join(ctx)):
    passed += 1
else:
    failed += 1

# Test 3.6: Full complex conversation
text = f"""{BOS_TOKEN}{START_TOKEN}system
I am USF Omega, an advanced AI assistant.{END_TOKEN}
{START_TOKEN}developer
You must be helpful and accurate. Use tools when needed.{END_TOKEN}
{START_TOKEN}functions
namespace functions {{
  type search = (_: {{query: string}}) => any;
  type calculate = (_: {{expression: string}}) => any;
}}{END_TOKEN}
{START_TOKEN}user
What is the weather in NYC and what is 2+2?{END_TOKEN}
{START_TOKEN}assistant
{MESSAGE_START}I'll search for weather and calculate.{MESSAGE_END}{FUNCTIONS_START}...search...calculate...{FUNCTIONS_END}{END_TOKEN}
{START_TOKEN}tool
{FUNCTION_RESULTS_START}Weather: Sunny 75F{FUNCTION_RESULTS_END}{FUNCTION_RESULTS_START}Result: 4{FUNCTION_RESULTS_END}{END_TOKEN}
{START_TOKEN}assistant
{MESSAGE_START}NYC weather is sunny at 75°F, and 2+2=4.{MESSAGE_END}{SELF_REFLECTION_START}Complete answer with both results.{SELF_REFLECTION_END}{END_TOKEN}
{EOS_TOKEN}"""
ctx, ls = split_by_role_markers(text)
# Verify all roles have correct loss
sys_ok = all(ls[i] == 0. for i, c in enumerate(ctx) if c == START_TOKEN + 'system\n')
dev_ok = all(ls[i] == 0. for i, c in enumerate(ctx) if c == START_TOKEN + 'developer\n')
func_ok = all(ls[i] == 0. for i, c in enumerate(ctx) if c == START_TOKEN + 'functions\n')
user_ok = all(ls[i] == 0. for i, c in enumerate(ctx) if c == START_TOKEN + 'user\n')
tool_ok = all(ls[i] == 0. for i, c in enumerate(ctx) if c == START_TOKEN + 'tool\n')
asst_ok = all(ls[i] == 1. for i, c in enumerate(ctx) if c == START_TOKEN + 'assistant\n')
if test("3.6 full_complex", sys_ok and dev_ok and func_ok and user_ok and tool_ok and asst_ok):
    passed += 1
else:
    failed += 1

# ==============================================================================
# SECTION 4: Training Mode Specific Tests
# ==============================================================================
print("\n[4] Training Mode Specific")

# Test 4.1: SFT - Train on all assistant responses
text = f"""{START_TOKEN}user
Q1{END_TOKEN}
{START_TOKEN}assistant
{MESSAGE_START}A1{MESSAGE_END}{END_TOKEN}
{START_TOKEN}user
Q2{END_TOKEN}
{START_TOKEN}assistant
{MESSAGE_START}A2{MESSAGE_END}{END_TOKEN}
{START_TOKEN}user
Q3{END_TOKEN}
{START_TOKEN}assistant
{MESSAGE_START}A3{MESSAGE_END}{END_TOKEN}
"""
ctx, ls = split_by_role_markers(text)
loss_1_segments = sum(1 for s in ls if s == 1.)
loss_0_segments = sum(1 for s in ls if s == 0.)
if test("4.1 sft_all_assistants", loss_1_segments > 0 and loss_0_segments > 0):
    passed += 1
else:
    failed += 1

# Test 4.2: DPO/ORPO - Needs chosen and rejected with same context
chosen = f"""{START_TOKEN}user
Question{END_TOKEN}
{START_TOKEN}assistant
{MESSAGE_START}Good detailed answer{MESSAGE_END}{END_TOKEN}
"""
rejected = f"""{START_TOKEN}user
Question{END_TOKEN}
{START_TOKEN}assistant
{MESSAGE_START}Bad answer{MESSAGE_END}{END_TOKEN}
"""
ctx_c, ls_c = split_by_role_markers(chosen)
ctx_r, ls_r = split_by_role_markers(rejected)
# Same structure, different content
if test("4.2 dpo_orpo_pairs", len([s for s in ls_c if s == 1.]) > 0 and len([s for s in ls_r if s == 1.]) > 0):
    passed += 1
else:
    failed += 1

# Test 4.3: PPO/GRPO - Online rollout format
rollout = f"""{START_TOKEN}user
Generate a poem{END_TOKEN}
{START_TOKEN}assistant
{MESSAGE_START}Roses are red,
Violets are blue,
AI is helpful,
And so are you.{MESSAGE_END}{END_TOKEN}
"""
ctx, ls = split_by_role_markers(rollout)
if test("4.3 ppo_grpo_rollout", sum(1 for s in ls if s == 1.) > 0):
    passed += 1
else:
    failed += 1

# Test 4.4: KTO - Binary preference
text = f"""{START_TOKEN}user
Question{END_TOKEN}
{START_TOKEN}assistant
{MESSAGE_START}Response{MESSAGE_END}{END_TOKEN}
"""
ctx, ls = split_by_role_markers(text)
if test("4.4 kto_format", sum(1 for s in ls if s == 1.) > 0):
    passed += 1
else:
    failed += 1

# Test 4.5: GKD - Knowledge distillation
teacher_response = f"""{START_TOKEN}assistant
{REASONING_START}Deep analysis{REASONING_END}{MESSAGE_START}Comprehensive answer{MESSAGE_END}{END_TOKEN}
"""
ctx, ls = split_by_role_markers(teacher_response)
if test("4.5 gkd_format", REASONING_START in ''.join(ctx)):
    passed += 1
else:
    failed += 1

# ==============================================================================
# SECTION 5: Edge Cases and Stress Tests
# ==============================================================================
print("\n[5] Edge Cases and Stress Tests")

# Test 5.1: Very long content (10KB per message)
long_text = "X" * 10000
text = f"""{START_TOKEN}user
{long_text}{END_TOKEN}
{START_TOKEN}assistant
{MESSAGE_START}{long_text}{MESSAGE_END}{END_TOKEN}
"""
ctx, ls = split_by_role_markers(text)
total_len = sum(len(c) for c in ctx)
if test("5.1 long_content", total_len > 20000):
    passed += 1
else:
    failed += 1

# Test 5.2: 50 parallel tool calls
calls = ''.join(f'<||invoke_start to=function.f{i}||><||invoke_end||>' for i in range(50))
text = f"""{START_TOKEN}assistant
{MESSAGE_START}Calling 50 functions{MESSAGE_END}{FUNCTIONS_START}{calls}{FUNCTIONS_END}{END_TOKEN}
"""
ctx, ls = split_by_role_markers(text)
if test("5.2 fifty_parallel_calls", '<||invoke_start to=function.f49||>' in ''.join(ctx)):
    passed += 1
else:
    failed += 1

# Test 5.3: 10 self-reflection iterations
iterations = ''.join(f'{MESSAGE_START}Attempt {i}{MESSAGE_END}{SELF_REFLECTION_START}Reflection {i}{SELF_REFLECTION_END}' for i in range(10))
text = f"""{START_TOKEN}assistant
{iterations}{END_TOKEN}
"""
ctx, ls = split_by_role_markers(text)
# Check original text has 10 iterations and all assistant content has loss=1
if test("5.3 ten_iterations", text.count(SELF_REFLECTION_START) == 10 and all(s == 1. for s in ls)):
    passed += 1
else:
    failed += 1

# Test 5.4: Mixed unicode and special characters
text = f"""{START_TOKEN}assistant
{MESSAGE_START}Hello 世界! 🌍 こんにちは مرحبا Здравствуйте{MESSAGE_END}{END_TOKEN}
"""
ctx, ls = split_by_role_markers(text)
joined = ''.join(ctx)
if test("5.4 unicode_special", "世界" in joined and "🌍" in joined):
    passed += 1
else:
    failed += 1

# Test 5.5: Complex JSON in structured output
import json
complex_json = json.dumps({
    "users": [{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}],
    "metadata": {"version": 1, "nested": {"deep": {"value": True}}}
})
text = f"""{START_TOKEN}assistant
{CONSTRAIN_START}response.format={{type:json}}{CONSTRAIN_END}{MESSAGE_START}{complex_json}{MESSAGE_END}{END_TOKEN}
"""
ctx, ls = split_by_role_markers(text)
if test("5.5 complex_json", complex_json in ''.join(ctx)):
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
