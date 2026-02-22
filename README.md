# 🌵 SwissblAIz V3 — Hybrid Voice Assistant

<div align="center">

### Built for the **Google DeepMind × Cactus Compute** Hackathon

[![Google DeepMind](https://img.shields.io/badge/Google_DeepMind-4285F4?style=for-the-badge&logo=google&logoColor=white)](https://deepmind.google/)
[![Cactus Compute](https://img.shields.io/badge/Cactus_Compute-00C853?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHRleHQgeT0iMTgiIGZvbnQtc2l6ZT0iMTgiPvCfjLU8L3RleHQ+PC9zdmc+&logoColor=white)](https://cactuscompute.com)
[![FunctionGemma](https://img.shields.io/badge/FunctionGemma_270M-FF6F00?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)
[![Gemini Flash](https://img.shields.io/badge/Gemini_Flash-886FBF?style=for-the-badge&logo=google&logoColor=white)](https://ai.google.dev/)

**On-device voice assistant** powered by FunctionGemma-270M + Gemini Flash cloud fallback

</div>

---

## 🎙️ Voice-Activated Multi-Tool Agentic Demo

<p align="center">
  <a href="https://youtu.be/vo0RlKVuroE">
    <img src="https://img.youtube.com/vi/vo0RlKVuroE/0.jpg" alt="Demo Video">
  </a>
</p>


**Try it live:** Open `demo/index.html` in Edge or Chrome.

Features:

- 🎤 **Voice-first** — Tap the orb, speak naturally
- ⚡ **On-device inference** — 50-80ms via FunctionGemma-270M
- ☁️ **Cloud escalation** — Auto-routes hard queries to Gemini Flash
- 🇦🇺 **Australian male voice** — TTS responses via Microsoft Neural voices
- 📊 **Real-time pipeline** — See STT → Classify → Route → Infer → TTS live

## Architecture

```text
User Query (Voice / Text)
    │
    ▼
┌─────────────────────────┐
│  COMPLEXITY ROUTER      │  ◄ Deterministic, <1ms
│  EASY | MEDIUM | HARD   │    No LLM call
└────────┬────────────────┘
         │
         ▼
┌─────────────────────────┐
│  Cactus SDK + Gemma     │  ◄ Unified hybrid path
│  threshold → routing    │    SDK handles escalation
└────────┬────────────────┘
         │
    ┌────┴────┐
    ▼         ▼
┌────────┐ ┌──────────┐
│ ⚡ Local│ │ ☁️ Cloud  │
│ 50-80ms│ │ 200-500ms│
│ GemMA  │ │ Gemini   │
└───┬────┘ └────┬─────┘
    │            │
    ▼            ▼
┌─────────────────────────┐
│  TRN VALIDATOR          │  ◄ Schema + type check
│  + Postprocessor        │    F1 normalization
└─────────────────────────┘
```


## Key Innovations

1. **Cactus SDK Unified Path** — Single `cactus_complete` call with native hybrid routing
2. **Cloud Fallback** — SDK-managed escalation to Gemini Flash for complex queries
3. **Deterministic Complexity Router** — No LLM needed for routing, zero latency overhead
4. **TRN Validator** — Validates every tool call against its JSON schema
5. **Voice-First UI** — Animated orb, action cards, pipeline telemetry

## Running

```bash
# Setup (one-time)
git clone https://github.com/cactus-compute/cactus
cd cactus && source ./setup && cd ..
cactus build --python
cactus download google/functiongemma-270m-it --reconvert
pip install google-genai requests
export GEMINI_API_KEY="your-key"

# Run voice demo
open demo/index.html

# Submit to leaderboard
python submit.py --team "SwissblAIz" --location "Online"
```

## Team

**SwissblAIz** 🌵

Built with ❤️ for the **Google DeepMind × Cactus Compute** Hackathon

*Powered by [FunctionGemma-270M](https://ai.google.dev/) (on-device) + [Gemini Flash](https://ai.google.dev/) (cloud)*
