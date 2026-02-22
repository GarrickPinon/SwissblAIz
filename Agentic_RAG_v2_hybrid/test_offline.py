"""
OFFLINE logic test — validates routing, TRN, and reflection WITHOUT any API calls.
No cactus SDK needed. No Gemini API calls.

Run: python test_offline.py
"""

import sys, os, json

# ── Mock cactus BEFORE importing main ──
import types as bt
cactus_module = bt.ModuleType("cactus")
cactus_module.cactus_init = lambda *a, **kw: {}
cactus_module.cactus_complete = lambda *a, **kw: '{"function_calls":[],"confidence":0,"total_time_ms":0}'
cactus_module.cactus_destroy = lambda *a, **kw: None
sys.modules["cactus"] = cactus_module

from main import classify_complexity, validate_tool_calls, build_reflection_prompt

TOOLS = [
    {"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City"}}, "required": ["location"]}},
    {"name": "send_message", "description": "Send message", "parameters": {"type": "object", "properties": {"recipient": {"type": "string", "description": "Name"}, "message": {"type": "string", "description": "Content"}}, "required": ["recipient", "message"]}},
    {"name": "set_alarm", "description": "Set alarm", "parameters": {"type": "object", "properties": {"hour": {"type": "integer", "description": "Hour"}, "minute": {"type": "integer", "description": "Min"}}, "required": ["hour", "minute"]}},
    {"name": "play_music", "description": "Play music", "parameters": {"type": "object", "properties": {"song": {"type": "string", "description": "Song"}}, "required": ["song"]}},
    {"name": "set_timer", "description": "Set timer", "parameters": {"type": "object", "properties": {"minutes": {"type": "integer", "description": "Minutes"}}, "required": ["minutes"]}},
    {"name": "create_reminder", "description": "Create reminder", "parameters": {"type": "object", "properties": {"title": {"type": "string", "description": "Title"}, "time": {"type": "string", "description": "Time"}}, "required": ["title", "time"]}},
    {"name": "search_contacts", "description": "Search contacts", "parameters": {"type": "object", "properties": {"query": {"type": "string", "description": "Name"}}, "required": ["query"]}},
]

print("=" * 60)
print("  SwissblAIz V2 — OFFLINE LOGIC TEST")
print("  (No API calls — tests routing + TRN only)")
print("=" * 60)

# ── 1. TEST COMPLEXITY ROUTER ──
print("\n=== 1. COMPLEXITY ROUTER ===\n")

router_tests = [
    # (message, tools_subset, expected)
    ("What is the weather in San Francisco?", [TOOLS[0]], "EASY"),
    ("Set an alarm for 10 AM.", [TOOLS[2]], "EASY"),
    ("Play Bohemian Rhapsody.", [TOOLS[3]], "EASY"),
    ("Send a message to John saying hello.", [TOOLS[0], TOOLS[1], TOOLS[2]], "MEDIUM"),
    ("What's the weather in Tokyo?", [TOOLS[0], TOOLS[1]], "MEDIUM"),
    ("Set a timer for 10 minutes.", [TOOLS[2], TOOLS[4], TOOLS[3]], "MEDIUM"),
    ("Set an alarm for 9 AM.", TOOLS, "MEDIUM"),
    ("Send a message to Bob saying hi and get the weather in London.", [TOOLS[0], TOOLS[1], TOOLS[2]], "HARD"),
    ("Text Emma saying good night, check the weather in Chicago, and set an alarm for 5 AM.", TOOLS, "HARD"),
    ("Set a 15 minute timer, play classical music, and remind me to stretch at 4:00 PM.", TOOLS, "HARD"),
]

passed = 0
for msg, tools, expected in router_tests:
    result = classify_complexity(msg, tools)
    status = "PASS" if result == expected else "FAIL"
    if result == expected:
        passed += 1
    print(f"  [{status}] '{msg[:55]:<55}' -> {result:6} (expected {expected})")

print(f"\n  Router: {passed}/{len(router_tests)} correct")

# ── 2. TEST TRN VALIDATOR ──
print("\n=== 2. TRN VALIDATOR ===\n")

trn_tests = [
    ("Valid single call",
     [{"name": "get_weather", "arguments": {"location": "SF"}}],
     [TOOLS[0]], True),
    
    ("Valid multi-call (2 tools)",
     [{"name": "send_message", "arguments": {"recipient": "Bob", "message": "hi"}},
      {"name": "get_weather", "arguments": {"location": "London"}}],
     TOOLS[:3], True),
    
    ("Valid 3-call",
     [{"name": "send_message", "arguments": {"recipient": "Emma", "message": "good night"}},
      {"name": "get_weather", "arguments": {"location": "Chicago"}},
      {"name": "set_alarm", "arguments": {"hour": 5, "minute": 0}}],
     TOOLS, True),
    
    ("Empty calls → reject",
     [], TOOLS, False),
    
    ("Unknown function → reject",
     [{"name": "launch_rockets", "arguments": {}}],
     [TOOLS[0]], False),
    
    ("Missing required arg → reject",
     [{"name": "send_message", "arguments": {"recipient": "Bob"}}],
     [TOOLS[1]], False),
    
    ("Integer coercion (str '10' → int)",
     [{"name": "set_alarm", "arguments": {"hour": "10", "minute": "0"}}],
     [TOOLS[2]], True),
    
    ("Extra args (should still pass)",
     [{"name": "get_weather", "arguments": {"location": "SF", "units": "celsius"}}],
     [TOOLS[0]], True),
    
    ("Wrong type (string for integer, non-coercible) → reject",
     [{"name": "set_alarm", "arguments": {"hour": "ten", "minute": 0}}],
     [TOOLS[2]], False),
]

passed = 0
for desc, calls, tools, expected_valid in trn_tests:
    result = validate_tool_calls(calls, tools)
    ok = result["valid"] == expected_valid
    status = "PASS" if ok else "FAIL"
    if ok:
        passed += 1
    errs = f" | {result['errors']}" if result["errors"] else ""
    print(f"  [{status}] {desc:<45} valid={result['valid']:<5} score={result['score']:.1f}{errs}")

print(f"\n  TRN: {passed}/{len(trn_tests)} correct")

# ── 3. TEST REFLECTION PROMPT ──
print("\n=== 3. REFLECTION PROMPT ===\n")

msgs = [{"role": "user", "content": "Send Bob a message and get the weather."}]
errors = ["send_message: missing required arg 'message'"]
prev_calls = [{"name": "send_message", "arguments": {"recipient": "Bob"}}]
reflection = build_reflection_prompt(msgs, errors, prev_calls)

assert len(reflection) == 3, f"Expected 3 messages, got {len(reflection)}"
assert "missing required arg" in reflection[-1]["content"], "Error not injected"
print(f"  [PASS] Builds 3-message reflection prompt with error injection")
print(f"  [PASS] Error context: '{reflection[-1]['content'][:70]}...'")

# ── SUMMARY ──
print(f"\n{'=' * 60}")
print(f"  ALL LOGIC TESTS COMPLETE")
print(f"  Submit: python submit.py --team SwissblAIz --location Online")
print(f"{'=' * 60}")
