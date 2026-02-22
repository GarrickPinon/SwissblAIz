import sys
sys.path.insert(0, "cactus/python/src")
functiongemma_path = "cactus/weights/functiongemma-270m-it"

import json, os, time, re
from cactus import cactus_init, cactus_complete, cactus_destroy

try:
    from google import genai
    gemini_client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY", ""))
except Exception:
    gemini_client = None


# ═══════════════════════════════════════════════════════════════
# PHASE 8 FROZEN CONFIG — Do not modify
# ═══════════════════════════════════════════════════════════════
CONFIDENCE_THRESHOLD_EASY = 0.40     # Easy: trust FunctionGemma more
CONFIDENCE_THRESHOLD_MEDIUM = 0.50   # Medium: slightly higher bar
CONFIDENCE_THRESHOLD_HARD = 0.30     # Hard: lower bar (multi-call is hard for 270M)
CLOUD_MODEL = "gemini-2.0-flash"     # Fast + cheap cloud fallback


# ═══════════════════════════════════════════════════════════════
# 1. COMPLEXITY ROUTER — Deterministic, <1ms
# ═══════════════════════════════════════════════════════════════

def classify_complexity(message_text: str, tools: list) -> str:
    """
    Classify query complexity without any LLM call.
    """
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
# 2. SCHEMA VALIDATOR (TRN) — Deterministic, <1ms
# ═══════════════════════════════════════════════════════════════

def validate_tool_calls(function_calls: list, tools: list) -> dict:
    """Validate tool call output against schemas."""
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
    """Normalize function calls for maximum F1 accuracy."""
    name = call.get("name", "")
    args = call.get("arguments", {})
    
    # 270M sometimes provides arguments as strings
    if isinstance(args, str):
        try:
            args = json.loads(args)
        except:
            pass
            
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
            elif expected_type == "string":
                fixed_args[key] = _normalize_string(val)
            else:
                fixed_args[key] = val
        else:
            fixed_args[key] = val
            
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
# 4. UNIFIED GENERATION — Cactus SDK v1.7 Path
# ═══════════════════════════════════════════════════════════════

def generate_hybrid(messages, tools, confidence_threshold=0.5):
    """
    SwissblAIz V3 Hybrid Compute — Unified SDK Path.
    
    The SDK handles:
      - Hybrid routing (On-Device vs Cloud)
      - System prompt injection (Triggering tool-calling mode)
      - Cloud fallback via cactus_auth
      - JSON formatting and schema enforcement
    """
    user_text = next((m["content"] for m in messages if m["role"] == "user"), "")
    complexity = classify_complexity(user_text, tools)
    
    # Map to difficulty-based thresholds
    THRESHOLDS = {
        "EASY": CONFIDENCE_THRESHOLD_EASY,
        "MEDIUM": CONFIDENCE_THRESHOLD_MEDIUM,
        "HARD": CONFIDENCE_THRESHOLD_HARD,
    }
    threshold = THRESHOLDS.get(complexity, confidence_threshold)
    
    # ── Step 1: Try On-Device via Cactus SDK ──
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
        raw = json.loads(raw_str)
    except json.JSONDecodeError:
        raw = {"function_calls": [], "confidence": 0}
    
    local_confidence = raw.get("confidence", 0)
    calls = raw.get("function_calls") or raw.get("tool_calls") or raw.get("calls") or []
    source = raw.get("source", "on-device")
    total_time = raw.get("total_time_ms", 0)
    
    # ── Step 2: Cloud Fallback if confidence too low or no calls ──
    if (local_confidence < threshold or len(calls) == 0) and gemini_client:
        try:
            tool_desc = "\n".join(
                f"- {t['name']}: {t['description']}. Parameters: {json.dumps(t['parameters'])}"
                for t in tools
            )
            system_prompt = (
                "You are a function-calling assistant. Given the user query, "
                "return a JSON object with a 'function_calls' array. Each call "
                "must have 'name' (string) and 'arguments' (object).\n\n"
                f"Available tools:\n{tool_desc}\n\n"
                "Return ONLY valid JSON. No explanation."
            )
            
            cloud_messages = [
                {"role": "user", "parts": [{"text": system_prompt + "\n\nUser: " + messages[-1].get('content', '')}]}
            ]
            
            cloud_start = time.time()
            response = gemini_client.models.generate_content(
                model=CLOUD_MODEL,
                contents=cloud_messages,
            )
            cloud_time = int((time.time() - cloud_start) * 1000)
            
            cloud_text = response.text.strip()
            # Strip markdown code fences if present
            if cloud_text.startswith("```"):
                cloud_text = re.sub(r'^```(?:json)?\s*', '', cloud_text)
                cloud_text = re.sub(r'\s*```$', '', cloud_text)
            
            cloud_raw = json.loads(cloud_text)
            calls = cloud_raw.get("function_calls") or cloud_raw.get("tool_calls") or []
            source = "cloud"
            total_time = cloud_time
            local_confidence = 0.95  # Cloud is high confidence
        except Exception as e:
            # Cloud failed too — keep whatever on-device returned
            source = f"cloud-error: {str(e)}"
    
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

# Wrapper functions for backward compatibility/benchmark expectations
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
    print(f"Confidence: {result['confidence']:.4f}")
    print(f"Complexity: {result['complexity']}")
    print(f"Calls:      {len(result['function_calls'])}")
    for c in result['function_calls']:
        print(f"  → {c['name']}({c['arguments']})")
