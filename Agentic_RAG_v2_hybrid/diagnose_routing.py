
import sys, os, json
import types as bt

# Mock cactus
cactus_module = bt.ModuleType("cactus")
cactus_module.cactus_init = lambda *a, **kw: {}
cactus_module.cactus_complete = lambda *a, **kw: '{"function_calls":[{"name":"get_weather","arguments":{"location":"London"}}],"confidence":0.35,"total_time_ms":50}'
cactus_module.cactus_destroy = lambda *a, **kw: None
sys.modules["cactus"] = cactus_module

# Mock genai
google_module = bt.ModuleType("google")
genai_module = bt.ModuleType("google.genai")
google_module.genai = genai_module
genai_module.types = bt.ModuleType("google.genai.types")
class FakeClient:
    def __init__(self, **kw): pass
    @property
    def models(self):
        class MockModels:
            def generate_content(self, **kw):
                class MockResponse:
                    @property
                    def candidates(self):
                        class MockCandidate:
                            @property
                            def content(self):
                                class MockContent:
                                    @property
                                    def parts(self):
                                        return []
                                return MockContent()
                        return [MockCandidate()]
                return MockResponse()
        return MockModels()
genai_module.Client = FakeClient
sys.modules["google"] = google_module
sys.modules["google.genai"] = genai_module
sys.modules["google.genai.types"] = genai_module.types

from main import generate_hybrid, classify_complexity, CONFIDENCE_THRESHOLD_EASY

TOOLS = [{"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}]

msg = [{"role": "user", "content": "weather in London"}]
complexity = classify_complexity(msg[0]["content"], TOOLS)
print(f"DEBUG: Complexity = {complexity}")
print(f"DEBUG: EASY threshold = {CONFIDENCE_THRESHOLD_EASY}")

r = generate_hybrid(msg, TOOLS)
print(f"DEBUG: Result source = {r['source']}")
print(f"DEBUG: Result confidence = {r['confidence']}")

if r['source'] == "on-device":
    print("FAIL: Returned on-device when confidence 0.35 < 0.40")
else:
    print("SUCCESS: Did not return on-device early")
