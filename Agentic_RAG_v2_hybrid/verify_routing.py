
import sys, os, json
import types as bt

# Mock cactus
cactus_module = bt.ModuleType("cactus")
cactus_module.cactus_init = lambda *a, **kw: {}
# Default mock behavior: return fixed values
cactus_module.cactus_complete = lambda *a, **kw: '{"function_calls":[{"name":"get_weather","arguments":{"location":"London"}}],"confidence":0.45,"total_time_ms":50}'
cactus_module.cactus_destroy = lambda *a, **kw: None
sys.modules["cactus"] = cactus_module

# Mock genai
genai_module = bt.ModuleType("google.genai")
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
                                        class MockPart:
                                            def __init__(self):
                                                class MockFC:
                                                    def __init__(self):
                                                        self.name = "get_weather"
                                                        self.args = {"location": "London"}
                                                self.function_call = MockFC()
                                        return [MockPart()]
                                return MockContent()
                        return [MockCandidate()]
                return MockResponse()
        return MockModels()
genai_module.Client = FakeClient
sys.modules["google.genai"] = genai_module
sys.modules["google.genai.types"] = genai_module.types

cactus_module.cactus_complete = lambda model, *a, **kw: '{"function_calls":[{"name":"get_weather","arguments":{"location":"London"}}],"confidence":0.45,"total_time_ms":50}'

from main import generate_hybrid

TOOLS = [{"name": "get_weather", "description": "Get weather", "parameters": {"type": "object", "properties": {"location": {"type": "string"}}, "required": ["location"]}}]

print("--- Test 1: Easy query, confidence 0.45 (threshold 0.40) ---")
# cactus_complete returns 0.45, threshold for EASY is 0.40
r = generate_hybrid([{"role": "user", "content": "weather in London"}], TOOLS)
print(f"Source: {r['source']}")
assert r['source'] == "on-device"

print("\n--- Test 2: Easy query, confidence 0.35 (threshold 0.40) ---")
# Reset mock properly
def mock_complete_2(model, messages, **kw):
    if kw.get("cloud_model"):
        return '{"function_calls":[{"name":"get_weather","arguments":{"location":"London"}}],"confidence":0.95,"total_time_ms":200,"source":"cactus-cloud"}'
    return '{"function_calls":[{"name":"get_weather","arguments":{"location":"London"}}],"confidence":0.35,"total_time_ms":50,"source":"on-device"}'

cactus_module.cactus_complete = mock_complete_2
r = generate_hybrid([{"role": "user", "content": "weather in London"}], TOOLS)
# Should fall back to cloud because 0.35 < 0.40
print(f"Source: {r['source']}")
assert "cloud" in r['source']

print("\n--- Test 3: Hard query, confidence 0.35 (threshold 0.30) ---")
# Threshold for HARD is 0.30. 0.35 should pass.
r = generate_hybrid([{"role": "user", "content": "weather in London and Paris and Tokyo"}], TOOLS)
print(f"Source: {r['source']}")
assert r['source'] == "on-device"

print("\nLogic Verified!")
