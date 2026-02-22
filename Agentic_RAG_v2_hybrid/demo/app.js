/* ═══════════════════════════════════════════════════════════
   SwissblAIz Assistant — Voice-First Hybrid Engine
   ═══════════════════════════════════════════════════════════ */

// ─── STATE MACHINE ───
const State = {
    IDLE: 'idle',
    LISTENING: 'listening',
    THINKING: 'thinking',
    ESCALATING: 'escalating',
    RESPONDING: 'responding',
    ERROR: 'error',
};

let currentState = State.IDLE;
let ttsEnabled = true;
let recognition = null;
let synthesis = window.speechSynthesis;
let preferredVoice = null;
let pipelineOpen = false;

// ─── DOM REFS ───
const $ = id => document.getElementById(id);
const orb = $('orb');
const orbGlow = document.querySelector('.orb-glow');
const orbStatus = $('orbStatus');
const micBtn = $('micBtn');
const ttsToggle = $('ttsToggle');
const transcriptArea = $('transcriptArea');
const transcriptText = $('transcriptText');
const cardsContainer = $('cardsContainer');
const textInput = $('textInput');
const sendBtn = $('sendBtn');
const routingBadge = $('routingBadge');
const pipelineToggle = $('pipelineToggle');
const pipelineDrawer = $('pipelineDrawer');

// ─── TOOL DEFINITIONS (mirrors main.py benchmark) ───
const TOOLS = [
    {
        name: "get_weather",
        description: "Get current weather for a location",
        parameters: {
            type: "object",
            properties: {
                location: { type: "string", description: "City name" }
            },
            required: ["location"]
        }
    },
    {
        name: "set_alarm",
        description: "Set an alarm",
        parameters: {
            type: "object",
            properties: {
                time: { type: "string", description: "Time in HH:MM format" },
                label: { type: "string", description: "Alarm label" }
            },
            required: ["time"]
        }
    },
    {
        name: "set_timer",
        description: "Set a countdown timer",
        parameters: {
            type: "object",
            properties: {
                duration_minutes: { type: "integer", description: "Duration in minutes" },
                label: { type: "string", description: "Timer label" }
            },
            required: ["duration_minutes"]
        }
    },
    {
        name: "send_message",
        description: "Send a message to a contact",
        parameters: {
            type: "object",
            properties: {
                contact: { type: "string", description: "Contact name" },
                message: { type: "string", description: "Message content" }
            },
            required: ["contact", "message"]
        }
    },
    {
        name: "search_contacts",
        description: "Search contacts by name",
        parameters: {
            type: "object",
            properties: {
                query: { type: "string", description: "Search query" }
            },
            required: ["query"]
        }
    },
    {
        name: "play_music",
        description: "Play a song or artist",
        parameters: {
            type: "object",
            properties: {
                query: { type: "string", description: "Song or artist name" }
            },
            required: ["query"]
        }
    }
];


// ═══════════════════════════════════════════════════════════
// COMPLEXITY ROUTER — mirrors main.py classify_complexity
// ═══════════════════════════════════════════════════════════

function classifyComplexity(text) {
    const lower = text.toLowerCase();
    const intentKeywords = {
        weather: ["weather", "temperature", "forecast", "degrees"],
        alarm: ["alarm", "wake me", "wake up"],
        message: ["send", "text", "message", "tell"],
        reminder: ["remind", "reminder"],
        search: ["find", "look up", "search", "contacts"],
        music: ["play", "music", "song"],
        timer: ["timer", "countdown", "minutes"],
    };

    const detected = new Set();
    for (const [intent, keywords] of Object.entries(intentKeywords)) {
        for (const kw of keywords) {
            if (lower.includes(kw)) { detected.add(intent); break; }
        }
    }

    const numIntents = detected.size;
    const numTools = TOOLS.length;

    if (numIntents <= 1 && numTools <= 2) return { level: "EASY", threshold: 0.40 };
    if (numIntents >= 2) return { level: "HARD", threshold: 0.30 };
    return { level: "MEDIUM", threshold: 0.50 };
}


// ═══════════════════════════════════════════════════════════
// MOCK HYBRID INFERENCE — Simulates cactus_complete behavior
// ═══════════════════════════════════════════════════════════

function mockToolCall(text) {
    const lower = text.toLowerCase();

    // Weather
    const weatherMatch = lower.match(/weather\s+(?:in\s+)?([a-z\s]+?)(?:\?|$|\.)/);
    if (weatherMatch || lower.includes('weather') || lower.includes('temperature')) {
        const location = weatherMatch ? weatherMatch[1].trim() : extractEntity(text);
        return [{
            name: "get_weather",
            arguments: { location: capitalize(location || "San Francisco") }
        }];
    }

    // Alarm
    const alarmMatch = lower.match(/alarm\s+(?:for\s+)?(\d{1,2}(?::\d{2})?\s*(?:am|pm)?)/i);
    if (alarmMatch || lower.includes('wake me') || lower.includes('alarm')) {
        const time = alarmMatch ? alarmMatch[1].trim() : "7:00 AM";
        return [{
            name: "set_alarm",
            arguments: { time: time, label: "Alarm" }
        }];
    }

    // Timer
    const timerMatch = lower.match(/(?:timer|countdown)\s+(?:for\s+)?(\d+)\s*(?:min|minute)/i);
    if (timerMatch || lower.includes('timer')) {
        const mins = timerMatch ? parseInt(timerMatch[1]) : 5;
        return [{
            name: "set_timer",
            arguments: { duration_minutes: mins, label: "Timer" }
        }];
    }

    // Message
    if (lower.includes('send') || lower.includes('text') || lower.includes('message')) {
        const parts = text.match(/(?:send|text|message)\s+(\w+)\s+(.+)/i);
        return [{
            name: "send_message",
            arguments: {
                contact: parts ? capitalize(parts[1]) : "Maggie",
                message: parts ? parts[2] : "Hey! I'll be there soon."
            }
        }];
    }

    // Music
    if (lower.includes('play') || lower.includes('music')) {
        const songMatch = text.match(/play\s+(.+)/i);
        return [{
            name: "play_music",
            arguments: { query: songMatch ? songMatch[1].trim() : "Lo-fi beats" }
        }];
    }

    // Multi-intent (Hard)
    const calls = [];
    if (lower.includes('weather')) {
        calls.push({ name: "get_weather", arguments: { location: "San Francisco" } });
    }
    if (lower.includes('alarm') || lower.includes('wake')) {
        calls.push({ name: "set_alarm", arguments: { time: "7:00 AM", label: "Morning" } });
    }
    if (lower.includes('timer')) {
        calls.push({ name: "set_timer", arguments: { duration_minutes: 10, label: "Cooking" } });
    }

    if (calls.length > 0) return calls;

    // Fallback: generic search
    return [{
        name: "search_contacts",
        arguments: { query: text }
    }];
}

function extractEntity(text) {
    const words = text.split(/\s+/);
    const skipWords = new Set(['what', 'is', 'the', 'in', 'whats', "what's", 'weather', 'for', 'a', 'an', 'tell', 'me', 'about', 'get', 'how', 'check']);
    const entities = words.filter(w => !skipWords.has(w.toLowerCase()) && w.length > 1);
    return entities.slice(-2).join(' ') || 'San Francisco';
}

function capitalize(s) {
    return s.split(' ').map(w => w.charAt(0).toUpperCase() + w.slice(1)).join(' ');
}


// ═══════════════════════════════════════════════════════════
// FULL PIPELINE — STT → Classify → Route → Infer → Card → TTS
// ═══════════════════════════════════════════════════════════

async function runPipeline(query) {
    const pipelineStart = performance.now();

    // ── STT Complete ──
    setPipelineStep('stt', 'done', `${Math.round(performance.now() - pipelineStart)}ms`);

    // ── Classify ──
    setState(State.THINKING);
    setPipelineStep('classify', 'active');
    await sleep(150);
    const complexity = classifyComplexity(query);
    setPipelineStep('classify', 'done', '<1ms');
    updateMetric('complexityMetric', complexity.level,
        complexity.level === 'EASY' ? 'highlight-cyan' :
            complexity.level === 'HARD' ? 'highlight-red' : 'highlight-amber');

    // ── Route Decision ──
    setPipelineStep('route', 'active');
    await sleep(200);

    // Simulate confidence
    const localConfidence = 0.35 + Math.random() * 0.55;
    const needsCloud = localConfidence < complexity.threshold;
    const source = needsCloud ? 'cloud' : 'on-device';

    updateMetric('confidenceMetric', localConfidence.toFixed(2),
        localConfidence > 0.7 ? 'highlight-green' :
            localConfidence > 0.4 ? 'highlight-cyan' : 'highlight-red');

    setPipelineStep('route', 'done', '<1ms');

    // ── Inference ──
    if (needsCloud) {
        setState(State.ESCALATING);
        setRouting('escalating', '⚡→☁️ ESCALATING');
        setPipelineStep('infer', 'cloud-active');
        await sleep(800 + Math.random() * 400);
        setRouting('cloud', '☁️ CLOUD');
    } else {
        setRouting('on-device', '⚡ ON-DEVICE');
        setPipelineStep('infer', 'active');
        await sleep(60 + Math.random() * 80);
    }

    const inferTime = needsCloud ? Math.round(200 + Math.random() * 300) : Math.round(50 + Math.random() * 80);
    setPipelineStep('infer', 'done', `${inferTime}ms`);
    updateMetric('latencyMetric', `${inferTime}ms`,
        inferTime < 100 ? 'highlight-cyan' :
            inferTime < 300 ? 'highlight-amber' : 'highlight-red');

    // ── Generate Tool Calls ──
    const calls = mockToolCall(query);

    // ── Build Cards ──
    setState(State.RESPONDING);
    clearCards();

    for (const call of calls) {
        const card = buildActionCard(call, source, inferTime);
        addCard(card);
        await sleep(100);
    }

    // ── TTS (uses _ttsText set during card build) ──
    setPipelineStep('tts', 'active');
    const ttsText = calls.map(c => c._ttsText || `Done. ${c.name} executed.`).join(' ');
    if (ttsEnabled) {
        await speak(ttsText);
    }
    setPipelineStep('tts', 'done');

    // ── Done ──
    const totalTime = Math.round(performance.now() - pipelineStart);
    setState(State.IDLE);
    orbStatus.textContent = `Done in ${totalTime}ms`;

    setTimeout(() => {
        if (currentState === State.IDLE) {
            orbStatus.textContent = 'Tap to speak';
            setRouting('standby', 'STANDBY');
        }
    }, 5000);
}


// ═══════════════════════════════════════════════════════════
// ACTION CARD BUILDERS
// ═══════════════════════════════════════════════════════════

function buildActionCard(call, source, latency) {
    const cardTypes = {
        get_weather: buildWeatherCard,
        set_alarm: buildAlarmCard,
        set_timer: buildTimerCard,
        send_message: buildMessageCard,
        play_music: buildMusicCard,
        search_contacts: buildSearchCard,
    };

    const builder = cardTypes[call.name] || buildGenericCard;
    return builder(call, source, latency);
}

function cardHeader(icon, label, source) {
    return `
        <div class="card-header">
            <div class="card-type">
                <span class="card-type-icon">${icon}</span>
                <span class="card-type-label">${label}</span>
            </div>
            <span class="card-source ${source === 'cloud' ? 'cloud' : 'on-device'}">
                ${source === 'cloud' ? '☁️ Cloud' : '⚡ On-Device'}
            </span>
        </div>`;
}

function buildWeatherCard(call, source) {
    const loc = call.arguments.location;
    const temp = Math.round(55 + Math.random() * 35);
    const conditions = ['Sunny', 'Partly Cloudy', 'Clear', 'Overcast', 'Breezy'][Math.floor(Math.random() * 5)];
    const icons = ['☀️', '⛅', '🌤️', '☁️', '🌬️'];
    const icon = icons[Math.floor(Math.random() * icons.length)];

    // Store for TTS so voice matches card
    call._ttsText = `It's currently ${temp} degrees in ${loc}. ${conditions}.`;

    return `
        <div class="action-card weather-card">
            ${cardHeader('🌡️', 'Weather', source)}
            <div class="card-body">
                <div class="weather-main">
                    <div>
                        <div class="weather-temp">${temp}<span class="unit">°F</span></div>
                    </div>
                    <div class="weather-details">
                        <span class="weather-location">${loc}</span>
                        <span class="weather-condition">${conditions}</span>
                    </div>
                    <span class="weather-icon">${icon}</span>
                </div>
            </div>
        </div>`;
}

function buildAlarmCard(call, source) {
    call._ttsText = `Alarm set for ${call.arguments.time}.`;
    return `
        <div class="action-card alarm-card">
            ${cardHeader('⏰', 'Alarm Set', source)}
            <div class="card-body">
                <span class="alarm-time">${call.arguments.time}</span>
                <span class="alarm-label">${call.arguments.label || 'Alarm'}</span>
            </div>
        </div>`;
}

function buildTimerCard(call, source) {
    call._ttsText = `Timer set for ${call.arguments.duration_minutes} minutes.`;
    return `
        <div class="action-card alarm-card">
            ${cardHeader('⏱️', 'Timer Set', source)}
            <div class="card-body">
                <span class="alarm-time">${call.arguments.duration_minutes} min</span>
                <span class="alarm-label">${call.arguments.label || 'Timer'}</span>
            </div>
        </div>`;
}

function buildMessageCard(call, source) {
    call._ttsText = `Message sent to ${call.arguments.contact}.`;
    return `
        <div class="action-card result-card">
            ${cardHeader('💬', 'Message Sent', source)}
            <div class="card-body">
                <span class="result-text">
                    To <strong>${call.arguments.contact}</strong>: "${call.arguments.message}"
                </span>
            </div>
        </div>`;
}

function buildMusicCard(call, source) {
    call._ttsText = `Now playing ${call.arguments.query}.`;
    return `
        <div class="action-card result-card">
            ${cardHeader('🎵', 'Now Playing', source)}
            <div class="card-body">
                <span class="result-text">${call.arguments.query}</span>
            </div>
        </div>`;
}

function buildSearchCard(call, source) {
    call._ttsText = `Searching for ${call.arguments.query}.`;
    return `
        <div class="action-card tool-call-card">
            ${cardHeader('🔍', 'Search', source)}
            <div class="card-body">
                <span class="tool-call-name">${call.name}()</span>
                <div class="tool-call-args">${JSON.stringify(call.arguments, null, 2)}</div>
            </div>
        </div>`;
}

function buildGenericCard(call, source) {
    return `
        <div class="action-card tool-call-card">
            ${cardHeader('🔧', 'Tool Call', source)}
            <div class="card-body">
                <span class="tool-call-name">${call.name}()</span>
                <div class="tool-call-args">${JSON.stringify(call.arguments, null, 2)}</div>
            </div>
        </div>`;
}

function addCard(html) {
    cardsContainer.insertAdjacentHTML('beforeend', html);
    cardsContainer.scrollTop = cardsContainer.scrollHeight;
}

function clearCards() {
    cardsContainer.innerHTML = '';
}


// ═══════════════════════════════════════════════════════════
// TTS — Australian Male Voice 🇦🇺
// ═══════════════════════════════════════════════════════════

let voicesLoaded = false;
let ttsWarmedUp = false;

function loadVoices() {
    const voices = synthesis.getVoices();
    if (voices.length === 0) return;

    voicesLoaded = true;
    console.log(`[TTS] ${voices.length} voices available:`);

    // Priority for Edge/Windows: Microsoft neural voices are best
    // Edge has 'Microsoft James Online (Natural) - English (Australia)'
    preferredVoice = voices.find(v => /en-AU/i.test(v.lang) && /james/i.test(v.name))
        || voices.find(v => /en-AU/i.test(v.lang) && /online|natural/i.test(v.name))
        || voices.find(v => /en-AU/i.test(v.lang))
        || voices.find(v => /en.AU/i.test(v.lang))
        || voices.find(v => v.lang.startsWith('en') && /james|daniel/i.test(v.name))
        || voices.find(v => v.lang.startsWith('en') && /david|mark|guy/i.test(v.name))
        || voices.find(v => v.lang.startsWith('en-GB') && /online|natural/i.test(v.name))
        || voices.find(v => v.lang.startsWith('en') && /online|natural/i.test(v.name))
        || voices.find(v => v.lang.startsWith('en'))
        || voices[0];

    if (preferredVoice) {
        console.log(`[TTS] ✅ Voice selected: ${preferredVoice.name} (${preferredVoice.lang})`);
    }

    // List top candidates for debugging
    voices.filter(v => v.lang.startsWith('en')).slice(0, 8).forEach(v => {
        console.log(`  → ${v.name} [${v.lang}] ${v.localService ? 'local' : 'remote'}`);
    });
}

// Chrome loads voices async — need both approaches
synthesis.onvoiceschanged = loadVoices;
loadVoices();
// Also poll a few times in case onvoiceschanged doesn't fire
setTimeout(loadVoices, 100);
setTimeout(loadVoices, 500);
setTimeout(loadVoices, 1500);

function warmupTTS() {
    if (ttsWarmedUp) return;
    ttsWarmedUp = true;
    // Speak a real word to truly unlock Chrome's TTS engine
    // Empty string doesn't actually trigger the audio pipeline
    const warmup = new SpeechSynthesisUtterance('.');
    warmup.volume = 0.01; // Nearly silent but not zero (zero = skip)
    warmup.rate = 10; // As fast as possible
    if (preferredVoice) warmup.voice = preferredVoice;
    synthesis.speak(warmup);
    setTimeout(loadVoices, 300);
    console.log('[TTS] Warmed up with audible trigger');
}

function speak(text) {
    return new Promise((resolve) => {
        if (!ttsEnabled || !synthesis) { resolve(); return; }

        // Retry loading voices if not ready
        if (!voicesLoaded || !preferredVoice) {
            loadVoices();
        }

        // Chrome bug: cancel() then immediate speak() = silent failure
        // Fix: cancel first, then wait 50ms before speaking
        synthesis.cancel();

        setTimeout(() => {
            const utterance = new SpeechSynthesisUtterance(text);
            if (preferredVoice) {
                utterance.voice = preferredVoice;
            }
            utterance.rate = 1.05;
            utterance.pitch = 0.95;
            utterance.volume = 1.0;

            // Chrome pauses long TTS — keep resuming
            let resumeInterval = setInterval(() => {
                if (synthesis.speaking && synthesis.paused) {
                    synthesis.resume();
                }
            }, 5000);

            let resolved = false;
            const done = () => {
                if (resolved) return;
                resolved = true;
                clearInterval(resumeInterval);
                resolve();
            };

            utterance.onend = done;
            utterance.onerror = (e) => {
                console.warn('[TTS] Error:', e.error);
                done();
            };

            // Failsafe: if TTS doesn't fire within 8s, resolve anyway
            setTimeout(done, 8000);

            synthesis.speak(utterance);
            console.log(`[TTS] 🔊 Speaking: "${text.substring(0, 80)}" with ${preferredVoice?.name || 'default voice'}`);
        }, 80);
    });
}

function buildTTSResponse(calls) {
    const parts = calls.map(c => {
        switch (c.name) {
            case 'get_weather':
                const temp = Math.round(55 + Math.random() * 35);
                return `It's currently ${temp} degrees in ${c.arguments.location}.`;
            case 'set_alarm':
                return `Alarm set for ${c.arguments.time}.`;
            case 'set_timer':
                return `Timer set for ${c.arguments.duration_minutes} minutes.`;
            case 'send_message':
                return `Message sent to ${c.arguments.contact}.`;
            case 'play_music':
                return `Now playing ${c.arguments.query}.`;
            default:
                return `Done. Executed ${c.name}.`;
        }
    });
    return parts.join(' ');
}


// ═══════════════════════════════════════════════════════════
// STT — Web Speech API
// ═══════════════════════════════════════════════════════════

function initSTT() {
    const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRecognition) {
        console.warn('[STT] Web Speech API not supported');
        return;
    }

    recognition = new SpeechRecognition();
    recognition.lang = 'en-US';
    recognition.interimResults = true;
    recognition.continuous = false;

    recognition.onstart = () => {
        setState(State.LISTENING);
        setPipelineStep('stt', 'active');
    };

    recognition.onresult = (event) => {
        let interim = '';
        let final = '';

        for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
                final += transcript;
            } else {
                interim += transcript;
            }
        }

        showTranscript(final || '', interim);

        if (final) {
            runPipeline(final);
        }
    };

    recognition.onerror = (event) => {
        console.error('[STT] Error:', event.error);
        if (event.error !== 'aborted') {
            setState(State.ERROR);
            orbStatus.textContent = `Voice error: ${event.error}`;
            setTimeout(() => setState(State.IDLE), 2000);
        }
    };

    recognition.onend = () => {
        if (currentState === State.LISTENING) {
            // No result captured
            setState(State.IDLE);
        }
    };
}

function startListening() {
    if (!recognition) {
        initSTT();
    }
    if (!recognition) {
        // Fallback: focus text input
        textInput.focus();
        return;
    }
    try {
        synthesis.cancel();
        recognition.start();
    } catch (e) {
        console.warn('[STT] Already running');
    }
}

function stopListening() {
    if (recognition) {
        try { recognition.stop(); } catch (e) { }
    }
}


// ═══════════════════════════════════════════════════════════
// UI STATE MANAGEMENT
// ═══════════════════════════════════════════════════════════

function setState(state) {
    currentState = state;
    orb.className = 'orb';
    orbGlow.className = 'orb-glow';

    switch (state) {
        case State.IDLE:
            orbStatus.textContent = 'Tap to speak';
            micBtn.classList.remove('recording');
            break;
        case State.LISTENING:
            orb.classList.add('listening');
            orbGlow.classList.add('listening');
            orbStatus.textContent = 'Listening...';
            micBtn.classList.add('recording');
            break;
        case State.THINKING:
            orb.classList.add('thinking');
            orbGlow.classList.add('thinking');
            orbStatus.textContent = 'Processing on-device...';
            micBtn.classList.remove('recording');
            break;
        case State.ESCALATING:
            orb.classList.add('escalating');
            orbGlow.classList.add('escalating');
            orbStatus.textContent = 'Escalating to cloud...';
            break;
        case State.RESPONDING:
            orb.classList.add('success');
            orbGlow.classList.add('success');
            orbStatus.textContent = 'Here you go';
            break;
        case State.ERROR:
            orb.classList.add('error');
            orbStatus.textContent = 'Something went wrong';
            break;
    }
}

function setRouting(state, label) {
    routingBadge.className = 'routing-badge';
    if (state !== 'standby') routingBadge.classList.add(state);
    routingBadge.querySelector('.routing-label').textContent = label;
}

function updateMetric(id, value, colorClass) {
    const el = $(id).querySelector('.metric-value');
    el.textContent = value;
    el.className = 'metric-value';
    if (colorClass) el.classList.add(colorClass);
}

function showTranscript(final, interim) {
    transcriptArea.classList.add('active');
    transcriptText.innerHTML = final + (interim ? `<span class="interim">${interim}</span>` : '');
}

function hideTranscript() {
    transcriptArea.classList.remove('active');
    transcriptText.innerHTML = '';
}


// ═══════════════════════════════════════════════════════════
// PIPELINE VISUALIZATION
// ═══════════════════════════════════════════════════════════

function setPipelineStep(stepId, state, time) {
    const step = $(`step-${stepId}`);
    if (!step) return;

    const dot = step.querySelector('.step-dot');
    const label = step.querySelector('.step-label');
    const timeEl = step.querySelector('.step-time');

    dot.className = 'step-dot';
    label.className = 'step-label';

    if (state === 'active' || state === 'cloud-active') {
        dot.classList.add(state === 'cloud-active' ? 'cloud-step' : 'active');
        label.classList.add('active');
    } else if (state === 'done') {
        dot.classList.add('done');
        label.classList.add('done');
    }

    if (time) timeEl.textContent = time;

    // Animate connectors
    const connectors = document.querySelectorAll('.pipeline-connector');
    const steps = ['stt', 'classify', 'route', 'infer', 'tts'];
    const idx = steps.indexOf(stepId);
    if (idx > 0 && state === 'done') {
        connectors[idx - 1].classList.add('done');
    } else if (state === 'active') {
        if (idx > 0) connectors[idx - 1].classList.add('active');
    }
}

function resetPipeline() {
    document.querySelectorAll('.step-dot').forEach(d => d.className = 'step-dot');
    document.querySelectorAll('.step-label').forEach(l => l.className = 'step-label');
    document.querySelectorAll('.step-time').forEach(t => t.textContent = '');
    document.querySelectorAll('.pipeline-connector').forEach(c => { c.className = 'pipeline-connector'; });
}


// ═══════════════════════════════════════════════════════════
// EVENT HANDLERS
// ═══════════════════════════════════════════════════════════

// Mic button: click to toggle
micBtn.addEventListener('click', () => {
    warmupTTS();
    if (currentState === State.LISTENING) {
        stopListening();
    } else if (currentState === State.IDLE) {
        resetPipeline();
        hideTranscript();
        startListening();
    }
});

// Orb: also clickable
orb.addEventListener('click', () => {
    warmupTTS();
    if (currentState === State.IDLE) {
        resetPipeline();
        hideTranscript();
        startListening();
    }
});

// Text input
sendBtn.addEventListener('click', () => {
    warmupTTS();
    const text = textInput.value.trim();
    if (text && currentState === State.IDLE) {
        textInput.value = '';
        resetPipeline();
        showTranscript(text, '');
        setPipelineStep('stt', 'done', '0ms');
        runPipeline(text);
    }
});

textInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        sendBtn.click();
    }
});

// TTS toggle
ttsToggle.addEventListener('click', () => {
    ttsEnabled = !ttsEnabled;
    ttsToggle.classList.toggle('active', ttsEnabled);
    if (!ttsEnabled) synthesis.cancel();
});

// Pipeline drawer toggle
pipelineToggle.addEventListener('click', () => {
    pipelineOpen = !pipelineOpen;
    pipelineDrawer.classList.toggle('open', pipelineOpen);
});

// ─── INIT ───
function init() {
    ttsToggle.classList.add('active');
    initSTT();
    // Open pipeline by default
    pipelineOpen = true;
    pipelineDrawer.classList.add('open');
}

function sleep(ms) {
    return new Promise(r => setTimeout(r, ms));
}

init();
