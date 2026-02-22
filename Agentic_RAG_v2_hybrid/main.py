import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time, re
from cactus import cactus_init, cactus_complete, cactus_destroy
from google import genai
from google.genai import types


# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
CONFIDENCE_THRESHOLD_EASY = 0.70
CONFIDENCE_THRESHOLD_MEDIUM = 0.50
CONFIDENCE_THRESHOLD_HARD = 0.30


# ═══════════════════════════════════════════════════════════════
# 1. COMPLEXITY ROUTER — Deterministic, <1ms
# ═══════════════════════════════════════════════════════════════

def classify_complexity(message_text: str, tools: list) -> str:
    num_tools = len(tools)
    text_lower = message_text.lower()
    
    intent_keywords = {
        "weather": ["weather", "temperature", "forecast"],
        "alarm": ["alarm", "wake me", "wake up"],
        "message": ["send", "text", "message", "tell"],
        "reminder": ["remind", "reminder"],
        "search": ["find", "look up", "search", "contacts"],
        "music": ["play", "music", "song"],
        "timer": ["timer", "countdown"],
    }
    
    detected_intents = set()
    for intent, keywords in intent_keywords.items():
        for kw in keywords:
            if kw in text_lower:
                detected_intents.add(intent)
                break
    
    num_intents = len(detected_intents)
    
    if num_tools == 1 and num_intents <= 1:
        return "EASY"
    elif num_intents >= 2:
        return "HARD"
    elif num_tools >= 3:
        return "MEDIUM"
    else:
        return "EASY"


# ═══════════════════════════════════════════════════════════════
# 2. ON-DEVICE GENERATION — FunctionGemma via Cactus
# ═══════════════════════════════════════════════════════════════

def generate_cactus(messages, tools):
    """Run function calling on-device via FunctionGemma + Cactus."""
    model = cactus_init(functiongemma_path)

    # Wrap tools in the format FunctionGemma expects
    cactus_tools = [{
        "type": "function",
        "function": t,
    } for t in tools]

    raw_str = cactus_complete(
        model,
        [{"role": "system", "content": "You are a helpful assistant that can use tools."}] + messages,
        tools=cactus_tools,
        force_tools=True,
        max_tokens=256,
        stop_sequences=["<|im_end|>", "<end_of_turn>"],
    )

    cactus_destroy(model)

    try:
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
        }

    return {
        "function_calls": raw.get("function_calls", []),
        "total_time_ms": raw.get("total_time_ms", 0),
        "confidence": raw.get("confidence", 0),
    }


# ═══════════════════════════════════════════════════════════════
# 3. CLOUD GENERATION — Gemini Flash via google.genai
# ═══════════════════════════════════════════════════════════════

def generate_cloud(messages, tools):
    """Run function calling via Gemini Cloud API."""
    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    gemini_tools = [
        types.Tool(function_declarations=[
            types.FunctionDeclaration(
                name=t["name"],
                description=t["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties={
                        k: types.Schema(type=v["type"].upper(), description=v.get("description", ""))
                        for k, v in t["parameters"]["properties"].items()
                    },
                    required=t["parameters"].get("required", []),
                ),
            )
            for t in tools
        ])
    ]

    contents = [m["content"] for m in messages if m["role"] == "user"]

    start_time = time.time()

    gemini_response = client.models.generate_content(
        model="gemini-2.0-flash",
        contents=contents,
        config=types.GenerateContentConfig(tools=gemini_tools),
    )

    total_time_ms = (time.time() - start_time) * 1000

    function_calls = []
    for candidate in gemini_response.candidates:
        for part in candidate.content.parts:
            if part.function_call:
                function_calls.append({
                    "name": part.function_call.name,
                    "arguments": dict(part.function_call.args),
                })

    return {
        "function_calls": function_calls,
        "total_time_ms": total_time_ms,
    }


# ═══════════════════════════════════════════════════════════════
# 4. POST-PROCESSING — Normalize for F1 Score
# ═══════════════════════════════════════════════════════════════

def postprocess_call(call: dict, tools: list) -> dict:
    """Normalize function calls for maximum F1 accuracy."""
    name = call.get("name", "")
    args = call.get("arguments", {})
    
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except:
            args = {}
    
    if not isinstance(args, dict):
        args = {}
    
    tool_map = {t["name"]: t for t in tools}
    
    # Fuzzy name matching
    if name not in tool_map:
        name_lower = name.lower().strip()
        for tname in tool_map:
            if tname.lower() == name_lower:
                name = tname
                break
    
    if name not in tool_map:
        return {"name": name, "arguments": args}
    
    schema = tool_map[name]
    properties = schema.get("parameters", {}).get("properties", {})
    
    fixed_args = {}
    for key, val in args.items():
        if key in properties:
            expected_type = properties[key].get("type", "string")
            if expected_type == "integer":
                fixed_args[key] = _normalize_integer(val)
            elif expected_type == "string":
                fixed_args[key] = str(val).strip() if val is not None else ""
            else:
                fixed_args[key] = val
        else:
            # Fuzzy key matching
            for prop_key in properties:
                if key.lower().replace('_','') == prop_key.lower().replace('_',''):
                    fixed_args[prop_key] = val
                    break
    
    # Fill missing required args
    required = schema.get("parameters", {}).get("required", [])
    for req in required:
        if req not in fixed_args:
            prop = properties.get(req, {})
            ptype = prop.get("type", "string")
            fixed_args[req] = 0 if ptype == "integer" else ""
    
    return {"name": name, "arguments": fixed_args}


def _normalize_integer(v):
    if isinstance(v, int): return v
    if isinstance(v, float): return int(v)
    if isinstance(v, str):
        v_clean = v.strip().lower()
        word_to_num = {
            "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
            "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20,
            "thirty": 30, "forty-five": 45,
        }
        if v_clean in word_to_num: return word_to_num[v_clean]
        try: return int(float(v_clean))
        except: return 0
    return 0


# ═══════════════════════════════════════════════════════════════
# 5. HYBRID GENERATION — Edge + Cloud Fallback
# ═══════════════════════════════════════════════════════════════

def generate_hybrid(messages, tools, confidence_threshold=0.5):
    """
    SwissblAIz V3 Hybrid Compute.
    
    Strategy:
    1. Classify complexity
    2. Run on-device via FunctionGemma + Cactus 
    3. If confidence < threshold or no calls, fall back to Gemini Flash
    4. Post-process and normalize all function calls for F1
    """
    user_text = next((m["content"] for m in messages if m["role"] == "user"), "")
    complexity = classify_complexity(user_text, tools)
    
    THRESHOLDS = {
        "EASY": CONFIDENCE_THRESHOLD_EASY,
        "MEDIUM": CONFIDENCE_THRESHOLD_MEDIUM,
        "HARD": CONFIDENCE_THRESHOLD_HARD,
    }
    threshold = THRESHOLDS.get(complexity, confidence_threshold)
    
    # Step 1: Try on-device
    local = generate_cactus(messages, tools)
    
    # Step 2: Decide if cloud fallback needed
    needs_cloud = local["confidence"] < threshold or len(local["function_calls"]) == 0
    
    if needs_cloud and local["confidence"] < threshold:
        try:
            cloud = generate_cloud(messages, tools)
            cloud["source"] = "cloud (fallback)"
            cloud["local_confidence"] = local["confidence"]
            cloud["total_time_ms"] += local["total_time_ms"]
            
            # Post-process cloud calls
            cloud["function_calls"] = [postprocess_call(c, tools) for c in cloud["function_calls"]]
            return cloud
        except Exception as e:
            # Cloud failed, fall through to local
            pass
    
    # Use local result
    local["source"] = "on-device"
    local["function_calls"] = [postprocess_call(c, tools) for c in local["function_calls"]]
    return local


# Wrapper functions
def generate_local(messages, tools):
    return generate_hybrid(messages, tools, confidence_threshold=1.0)

def generate_cloud_only(messages, tools):
    return generate_hybrid(messages, tools, confidence_threshold=0.0)


# ═══════════════════════════════════════════════════════════════
# EXAMPLE USAGE
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    tools = [{
        "name": "get_weather",
        "description": "Get current weather for a location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name"}
            },
            "required": ["location"],
        },
    }]

    messages = [{"role": "user", "content": "What is the weather in San Francisco?"}]

    print("\n--- SwissblAIz V3 Hybrid Test ---")
    result = generate_hybrid(messages, tools)
    print(f"Source:     {result.get('source', 'unknown')}")
    print(f"Confidence: {result.get('confidence', 'N/A')}")
    print(f"Time:       {result['total_time_ms']:.0f}ms")
    print(f"Calls:      {len(result['function_calls'])}")
    for c in result['function_calls']:
        print(f"  -> {c['name']}({c['arguments']})")
