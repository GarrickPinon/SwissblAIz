"""Quick debug: see exactly what generate_hybrid returns for sample cases."""
import sys, os, json
os.environ["CACTUS_NO_CLOUD_TELE"] = "1"

# Load env
from dotenv import load_dotenv
load_dotenv()

# Monkey-patch cactus for PC
import types as bt
from test_pc import FakeCactusModule
fake = FakeCactusModule()
cm = bt.ModuleType("cactus")
cm.cactus_init = fake.cactus_init
cm.cactus_complete = fake.cactus_complete
cm.cactus_destroy = fake.cactus_destroy
sys.modules["cactus"] = cm

from main import generate_hybrid
from benchmark import compute_f1

# Test 1: Easy
tools = [{"name": "get_weather", "description": "Get current weather for a location",
          "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City name"}}, "required": ["location"]}}]
msgs = [{"role": "user", "content": "What is the weather in San Francisco?"}]
r = generate_hybrid(msgs, tools)
expected = [{"name": "get_weather", "arguments": {"location": "San Francisco"}}]
print("=== EASY: weather_sf ===")
print(f"Source: {r.get('source')}")
print(f"Complexity: {r.get('complexity')}")
print(f"TRN: {r.get('trn_score')}")
print(f"Predicted: {json.dumps(r['function_calls'], indent=2)}")
print(f"Expected:  {json.dumps(expected, indent=2)}")
print(f"F1: {compute_f1(r['function_calls'], expected)}")
print()

# Test 2: Hard multi-call
tools2 = [
    {"name": "get_weather", "description": "Get current weather for a location",
     "parameters": {"type": "object", "properties": {"location": {"type": "string", "description": "City name"}}, "required": ["location"]}},
    {"name": "send_message", "description": "Send a message to a contact",
     "parameters": {"type": "object", "properties": {"recipient": {"type": "string", "description": "Name"}, "message": {"type": "string", "description": "Content"}}, "required": ["recipient", "message"]}},
    {"name": "set_alarm", "description": "Set an alarm for a given time",
     "parameters": {"type": "object", "properties": {"hour": {"type": "integer", "description": "Hour"}, "minute": {"type": "integer", "description": "Minute"}}, "required": ["hour", "minute"]}},
]
msgs2 = [{"role": "user", "content": "Send a message to Bob saying hi and get the weather in London."}]
r2 = generate_hybrid(msgs2, tools2)
expected2 = [
    {"name": "send_message", "arguments": {"recipient": "Bob", "message": "hi"}},
    {"name": "get_weather", "arguments": {"location": "London"}},
]
print("=== HARD: message_and_weather ===")
print(f"Source: {r2.get('source')}")
print(f"Complexity: {r2.get('complexity')}")
print(f"TRN: {r2.get('trn_score')}")
print(f"Predicted: {json.dumps(r2['function_calls'], indent=2)}")
print(f"Expected:  {json.dumps(expected2, indent=2)}")
print(f"F1: {compute_f1(r2['function_calls'], expected2)}")
