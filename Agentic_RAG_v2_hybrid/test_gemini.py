"""Test Gemini API directly - confirm tool calls work."""
import os, json
from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

# Simple tool call test
gemini_tools = [
    types.Tool(function_declarations=[
        types.FunctionDeclaration(
            name="get_weather",
            description="Get current weather for a location",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "location": types.Schema(type="STRING", description="City name")
                },
                required=["location"],
            ),
        )
    ])
]

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=["What is the weather in San Francisco?"],
    config=types.GenerateContentConfig(tools=gemini_tools),
)

print("=== Raw Gemini Response ===")
for candidate in response.candidates:
    for part in candidate.content.parts:
        print(f"Part type: {type(part)}")
        if part.function_call:
            fc = part.function_call
            print(f"Function name: {fc.name}")
            print(f"Args type: {type(fc.args)}")
            print(f"Args raw: {fc.args}")
            print(f"Args dict: {dict(fc.args)}")
            for k, v in fc.args.items():
                print(f"  {k}: {v!r} (type: {type(v).__name__})")
        elif part.text:
            print(f"Text: {part.text}")
