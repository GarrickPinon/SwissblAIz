import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time, re, requests

# ═══════════════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════════════
from cactus import cactus_init, cactus_complete, cactus_destroy

CONFIDENCE_THRESHOLD_EASY = 0.40
CONFIDENCE_THRESHOLD_MEDIUM = 0.50
CONFIDENCE_THRESHOLD_HARD = 0.30
CLOUD_MODEL = "gemini-2.0-flash"

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_URL = f"https://generativelanguage.googleapis.com/v1beta/models/{CLOUD_MODEL}:generateContent?key={GEMINI_API_KEY}"


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
# 2. SCHEMA VALIDATOR (TRN)
# ═══════════════════════════════════════════════════════════════

def validate_tool_calls(function_calls: list, tools: list) -> dict:
    if not function_calls:
        return {"valid": False, "errors": ["No function calls produced"], "score": 0.0}
    
    tool_map = {t["name"]: t for t in tools}
    errors = []
    valid_calls = 0
    
    for call in function_calls:
        call_name = call.get("name", "")
        if call_name not in tool_map:
            errors.append(f"Function '{call_name}' not found")
            continue
        
        tool = tool_map[call_name]
        args = call.get("arguments", {})
        
        required = tool["parameters"].get("required", [])
        for req in required:
            if req not in args:
                errors.append(f"Missing required argument '{req}' for '{call_name}'")
        
        valid_calls += 1
    
    score = valid_calls / len(function_calls) if function_calls else 0.0
    return {"valid": len(errors) == 0, "errors": errors, "score": score}


# ═══════════════════════════════════════════════════════════════
# 3. POST-PROCESSING — Normalize for F1 Score
# ═══════════════════════════════════════════════════════════════

def postprocess_call(call: dict, tools: list) -> dict:
    name = call.get("name", "")
    args = call.get("arguments", {})
    
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except:
            pass
    
    if not isinstance(args, dict):
        args = {}
            
    tool_map = {t["name"]: t for t in tools}
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
            elif expected_type == "number":
                fixed_args[key] = float(val) if not isinstance(val, (int, float)) else val
            elif expected_type == "boolean":
                if isinstance(val, str):
                    fixed_args[key] = val.lower() in ("true", "1", "yes")
                else:
                    fixed_args[key] = bool(val)
            elif expected_type == "string":
                fixed_args[key] = _normalize_string(val)
            elif expected_type == "array":
                if isinstance(val, str):
                    try: fixed_args[key] = json.loads(val)
                    except: fixed_args[key] = [val]
                else:
                    fixed_args[key] = val
            else:
                fixed_args[key] = val
        else:
            # Fuzzy key matching
            matched = False
            for prop_key in properties:
                if key.lower().replace('_','').replace('-','') == prop_key.lower().replace('_','').replace('-',''):
                    fixed_args[prop_key] = val
                    matched = True
                    break
            if not matched:
                fixed_args[key] = val
    
    # Fill missing required args with defaults
    required = schema.get("parameters", {}).get("required", [])
    for req in required:
        if req not in fixed_args:
            prop = properties.get(req, {})
            ptype = prop.get("type", "string")
            if ptype == "string":
                fixed_args[req] = ""
            elif ptype == "integer":
                fixed_args[req] = 0
            elif ptype == "boolean":
                fixed_args[req] = False
            elif ptype == "array":
                fixed_args[req] = []
    
    return {"name": name, "arguments": fixed_args}


def _normalize_integer(v):
    if isinstance(v, int): return v
    if isinstance(v, float): return int(v)
    if isinstance(v, str):
        v_clean = v.strip().lower()
        word_to_num = {
            "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
            "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
        }
        if v_clean in word_to_num: return word_to_num[v_clean]
        try: return int(float(v_clean))
        except: return 0
    return 0


def _normalize_string(v):
    if v is None: return ""
    return str(v).strip().strip('.,!?;:"\'')


# ═══════════════════════════════════════════════════════════════
# 4. UNIFIED GENERATION — Function API + REST Cloud Fallback
# ═══════════════════════════════════════════════════════════════

def generate_hybrid(messages, tools, confidence_threshold=0.5):
    """
    SwissblAIz V3 Hybrid Compute.
    
    Uses cactus function API (cactus_init/complete/destroy) for on-device,
    and direct Gemini REST API via requests for cloud fallback.
    """
    user_text = next((m["content"] for m in messages if m["role"] == "user"), "")
    complexity = classify_complexity(user_text, tools)
    
    THRESHOLDS = {
        "EASY": CONFIDENCE_THRESHOLD_EASY,
        "MEDIUM": CONFIDENCE_THRESHOLD_MEDIUM,
        "HARD": CONFIDENCE_THRESHOLD_HARD,
    }
    threshold = THRESHOLDS.get(complexity, confidence_threshold)
    
    # ── Step 1: On-Device via cactus function API ──
    model = cactus_init(functiongemma_path)
    
    try:
        raw_str = cactus_complete(
            model,
            messages, 
            tools=tools,
            temperature=0.0,
            max_tokens=512,
        )
    except Exception as e:
        raw_str = json.dumps({"function_calls": [], "confidence": 0, "source": f"cactus-error: {e}"})
    finally:
        cactus_destroy(model)
    
    try:
        raw = json.loads(raw_str) if isinstance(raw_str, str) else raw_str
    except json.JSONDecodeError:
        raw = {"function_calls": [], "confidence": 0}
    
    local_confidence = raw.get("confidence", 0)
    calls = raw.get("function_calls") or raw.get("tool_calls") or raw.get("calls") or []
    source = raw.get("source", "on-device")
    total_time = raw.get("total_time_ms", 0)
    
    # ── Step 2: Cloud Fallback via direct Gemini REST API ──
    needs_cloud = local_confidence < threshold or len(calls) == 0
    print(f"  [ROUTING] complexity={complexity}, confidence={local_confidence}, "
          f"threshold={threshold}, needs_cloud={needs_cloud}")
    
    if needs_cloud and GEMINI_API_KEY:
        try:
            tool_desc = "\n".join(
                f"- {t['name']}: {t['description']}. Parameters: {json.dumps(t['parameters'])}"
                for t in tools
            )
            system_prompt = (
                "You are a precise function-calling assistant. Given the user query, "
                "return a JSON object with a 'function_calls' array. Each element "
                "must have 'name' (string matching an available tool) and 'arguments' (object "
                "with all required parameters filled correctly).\n\n"
                f"Available tools:\n{tool_desc}\n\n"
                "RULES:\n"
                "1. Return ONLY valid JSON, no markdown, no explanation\n"
                "2. Match parameter names EXACTLY as defined in the schema\n"
                "3. Use correct types: strings for string params, integers for integer params\n"
                "4. Include ALL required parameters\n"
                "5. For multiple intents, include multiple function calls\n\n"
                'Example output: {"function_calls": [{"name": "tool_name", "arguments": {"param": "value"}}]}'
            )
            
            payload = {
                "contents": [{
                    "role": "user",
                    "parts": [{"text": system_prompt + "\n\nUser: " + user_text}]
                }],
                "generationConfig": {
                    "temperature": 0.0,
                    "maxOutputTokens": 1024,
                }
            }
            
            cloud_start = time.time()
            resp = requests.post(GEMINI_URL, json=payload, timeout=30)
            cloud_time = int((time.time() - cloud_start) * 1000)
            
            if resp.status_code == 200:
                cloud_text = resp.json()["candidates"][0]["content"]["parts"][0]["text"].strip()
                # Strip markdown code fences if present
                if cloud_text.startswith("```"):
                    cloud_text = re.sub(r'^```(?:json)?\s*', '', cloud_text)
                    cloud_text = re.sub(r'\s*```$', '', cloud_text)
                
                cloud_raw = json.loads(cloud_text)
                calls = cloud_raw.get("function_calls") or cloud_raw.get("tool_calls") or []
                source = "cloud"
                total_time = cloud_time
                local_confidence = 0.95
                print(f"  [CLOUD] OK — {len(calls)} calls in {cloud_time}ms")
            else:
                print(f"  [CLOUD] HTTP {resp.status_code}: {resp.text[:200]}")
                source = f"cloud-http-{resp.status_code}"
        except Exception as e:
            print(f"  [CLOUD] Error: {str(e)[:200]}")
            source = f"cloud-error: {str(e)[:100]}"
    
    # ── Step 3: Post-process and validate ──
    try:
        processed_calls = [postprocess_call(c, tools) for c in calls]
        trn = validate_tool_calls(processed_calls, tools)
        
        return {
            "function_calls": processed_calls,
            "total_time_ms": total_time,
            "confidence": local_confidence,
            "source": source,
            "complexity": complexity,
            "trn_score": trn["score"],
        }
        
    except Exception as e:
        return {
            "function_calls": [],
            "total_time_ms": 0,
            "confidence": 0,
            "source": f"error: {str(e)}",
            "complexity": complexity,
            "trn_score": 0,
        }

# Wrapper functions
def generate_local(messages, tools):
    return generate_hybrid(messages, tools, confidence_threshold=1.0)

def generate_cloud(messages, tools):
    return generate_hybrid(messages, tools, confidence_threshold=0.0)


# ═══════════════════════════════════════════════════════════════
# 5. EXAMPLE USAGE
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
    print(f"Source:     {result['source']}")
    print(f"Confidence: {result['confidence']}")
    print(f"Complexity: {result['complexity']}")
    print(f"Calls:      {len(result['function_calls'])}")
    for c in result['function_calls']:
        print(f"  -> {c['name']}({c['arguments']})")
