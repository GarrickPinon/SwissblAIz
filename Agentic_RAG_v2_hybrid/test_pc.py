"""
PC Testing Harness — Test the V2 hybrid routing logic WITHOUT Cactus SDK.

This replaces FunctionGemma with Gemini Flash calls so you can test:
  - Complexity router
  - TRN schema validation
  - Reflection retry logic
  - Full pipeline flow

Run: python test_pc.py

NOTE: This does NOT test actual FunctionGemma performance.
      For real scores, use: python submit.py --team "Algoverse" --location "Online"
"""

import json, os, time, sys
from dotenv import load_dotenv

# Load .env file
load_dotenv()

# ── Monkey-patch: Replace Cactus with Gemini simulation ──
# We intercept the cactus imports so main.py works on PC

class FakeCactusModule:
    """Simulate Cactus SDK using Gemini Flash for PC testing."""
    
    @staticmethod
    def cactus_init(model_path, **kwargs):
        return {"model": "simulated_functiongemma", "path": model_path}
    
    @staticmethod
    def cactus_complete(model, messages, **options):
        """Use Gemini Flash to simulate what FunctionGemma would return."""
        from google import genai
        from google.genai import types
        
        client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
        
        tools_raw = options.get("tools", [])
        tool_defs = []
        for t in tools_raw:
            func = t.get("function", t)
            tool_defs.append(func)
        
        if tool_defs:
            gemini_tools = [
                types.Tool(function_declarations=[
                    types.FunctionDeclaration(
                        name=td["name"],
                        description=td["description"],
                        parameters=types.Schema(
                            type="OBJECT",
                            properties={
                                k: types.Schema(
                                    type=v["type"].upper(),
                                    description=v.get("description", "")
                                )
                                for k, v in td["parameters"]["properties"].items()
                            },
                            required=td["parameters"].get("required", []),
                        ),
                    )
                    for td in tool_defs
                ])
            ]
        else:
            gemini_tools = []
        
        # Extract user content
        contents = []
        for m in messages:
            if m["role"] == "user":
                contents.append(m["content"])
        
        start_time = time.time()
        
        try:
            response = client.models.generate_content(
                model="gemini-2.0-flash",
                contents=contents,
                config=types.GenerateContentConfig(tools=gemini_tools) if gemini_tools else None,
            )
            
            total_time_ms = (time.time() - start_time) * 1000
            
            function_calls = []
            for candidate in response.candidates:
                for part in candidate.content.parts:
                    if part.function_call:
                        function_calls.append({
                            "name": part.function_call.name,
                            "arguments": dict(part.function_call.args),
                        })
            
            result = {
                "success": True,
                "response": "",
                "function_calls": function_calls,
                "confidence": 0.85,  # Simulated confidence
                "total_time_ms": total_time_ms,
                "prefill_tokens": 0,
                "decode_tokens": 0,
                "total_tokens": 0,
            }
        except Exception as e:
            result = {
                "success": False,
                "response": str(e),
                "function_calls": [],
                "confidence": 0.0,
                "total_time_ms": (time.time() - start_time) * 1000,
                "prefill_tokens": 0,
                "decode_tokens": 0,
                "total_tokens": 0,
            }
        
        return json.dumps(result)
    
    @staticmethod
    def cactus_destroy(model):
        pass


# Install the fake module before importing main
fake_cactus = FakeCactusModule()
import types as builtin_types
cactus_module = builtin_types.ModuleType("cactus")
cactus_module.cactus_init = fake_cactus.cactus_init
cactus_module.cactus_complete = fake_cactus.cactus_complete
cactus_module.cactus_destroy = fake_cactus.cactus_destroy
sys.modules["cactus"] = cactus_module

# Now import benchmark (which imports main, which imports cactus)
from benchmark import run_benchmark, BENCHMARKS


def run_pc_test(subset=None):
    """Run benchmark on PC with simulated FunctionGemma."""
    print("=" * 60)
    print("  V2 HYBRID COMPUTE — PC TEST MODE")
    print("  (FunctionGemma simulated via Gemini Flash)")
    print("=" * 60)
    print()
    
    if subset:
        cases = [b for b in BENCHMARKS if b["difficulty"] in subset]
        print(f"  Running {len(cases)} cases ({', '.join(subset)})")
    else:
        cases = BENCHMARKS
        print(f"  Running all {len(cases)} cases")
    
    print()
    results = run_benchmark(cases)
    
    print("\n" + "=" * 60)
    print("  NOTE: Actual FunctionGemma scores will differ.")
    print("  This test validates routing, TRN, and reflection logic.")
    print("  Submit for real scores: python submit.py --team Algoverse --location Online")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    # Quick smoke test: just easy cases first
    if "--easy" in sys.argv:
        run_pc_test(subset=["easy"])
    elif "--medium" in sys.argv:
        run_pc_test(subset=["medium"])
    elif "--hard" in sys.argv:
        run_pc_test(subset=["hard"])
    elif "--quick" in sys.argv:
        # Run 3 cases (1 easy, 1 medium, 1 hard) 
        quick_cases = [
            b for b in BENCHMARKS 
            if b["name"] in ["weather_sf", "message_among_three", "message_and_weather"]
        ]
        print("=" * 60)
        print("  QUICK SMOKE TEST (3 cases)")
        print("=" * 60)
        run_benchmark(quick_cases)
    else:
        run_pc_test()
