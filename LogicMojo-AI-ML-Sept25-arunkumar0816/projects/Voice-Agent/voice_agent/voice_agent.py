"""
Voice Agentic AI Assistant — Python 3.14 Compatible
Uses sounddevice (not pyaudio), pyttsx3 for TTS (no pygame needed)
"""

import os
import re
import csv
import time
import json
import datetime
from pathlib import Path
from collections import Counter

import pandas as pd
import sounddevice as sd  # Added for audio recording
import numpy as np        # Added for handling audio arrays

# ── Optional imports ──────────────────────────────────────────────────────────
try:
    import speech_recognition as sr
    SR_AVAILABLE = True
except ImportError:
    SR_AVAILABLE = False
    print("[WARN] SpeechRecognition not installed. Mic input disabled.")

try:
    import pyttsx3
    TTS_ENGINE = "pyttsx3"
except ImportError:
    TTS_ENGINE = None

try:
    from gtts import gTTS
    if TTS_ENGINE is None:
        TTS_ENGINE = "gtts"
except ImportError:
    pass

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

# ── Constants ─────────────────────────────────────────────────────────────────
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)
CONVERSATION_CSV = DATA_DIR / "conversations.csv"
ANALYTICS_CSV    = DATA_DIR / "analytics.csv"
OLLAMA_URL       = "http://localhost:11434/api/chat"
OLLAMA_MODEL     = "codellama"

CSV_COLUMNS = [
    "timestamp","session_id","turn_id","user_transcript",
    "detected_intent","detected_entities","ai_response",
    "response_time_ms","tts_engine","stt_engine",
]

# ── RegEx Patterns ────────────────────────────────────────────────────────────
PATTERNS = {
    "email":    re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}", re.I),
    "phone":    re.compile(r"(\+?\d[\d\s\-().]{7,}\d)"),
    "date":     re.compile(r"\b(\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}|today|tomorrow|yesterday)\b", re.I),
    "time":     re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?(?:\s?[apAP][mM])?\b"),
    "url":      re.compile(r"https?://[^\s]+|www\.[^\s]+"),
    "currency": re.compile(r"[\$\€\£\u20b9]\s?\d[\d,]*(?:\.\d+)?"),
    "number":   re.compile(r"\b\d+(?:\.\d+)?\b"),
}

INTENT_RULES = [
    ("greeting",    re.compile(r"\b(hi|hello|hey|good\s?morning|good\s?afternoon|good\s?evening|howdy)\b", re.I)),
    ("task_query",  re.compile(r"\b(tasks?|todos?|show.*task|pending)\b", re.I)),
    ("summarize",   re.compile(r"\b(summarize|summary|tldr|brief|shorten)\b", re.I)),
    ("reminder",    re.compile(r"\b(remind|reminder|alert|notify)\b", re.I)),
    ("weather",     re.compile(r"\b(weather|temperature|forecast|rain|sunny)\b", re.I)),
    ("help",        re.compile(r"\b(help|assist|guide|how\s+to|what\s+can)\b", re.I)),
    ("calculation", re.compile(r"\b(calculate|compute|how\s+much|add|subtract|multiply|divide)\b", re.I)),
    ("definition",  re.compile(r"\b(what\s+is|define|explain|meaning)\b", re.I)),
    ("joke",        re.compile(r"\b(joke|funny|laugh|humor)\b", re.I)),
    ("schedule",    re.compile(r"\b(schedule|calendar|appointment|meeting)\b", re.I)),
]

COMMAND_PATTERNS = {
    "create_task": re.compile(r"\bcreate\s+(?:a\s+)?task[:\s]+(.+)", re.I),
    "calculate":   re.compile(r"\b(?:calculate|compute|what\s+is)\s+(.+)", re.I),
}

def extract_entities(text):
    return {n: p.findall(text) for n, p in PATTERNS.items() if p.findall(text)}

def detect_intent(text):
    for name, pattern in INTENT_RULES:
        if pattern.search(text):
            return name
    return "general"

def extract_commands(text):
    return {cmd: p.search(text).groups() for cmd, p in COMMAND_PATTERNS.items() if p.search(text)}

# ── CSV ───────────────────────────────────────────────────────────────────────
def init_csv():
    if not CONVERSATION_CSV.exists():
        with open(CONVERSATION_CSV, "w", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()

def log_conversation(row):
    with open(CONVERSATION_CSV, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=CSV_COLUMNS).writerow(
            {col: row.get(col, "") for col in CSV_COLUMNS}
        )

def build_analytics():
    if not CONVERSATION_CSV.exists():
        return
    df = pd.read_csv(CONVERSATION_CSV)
    if df.empty:
        return
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["date"] = df["timestamp"].dt.date
    agg = df.groupby("date").agg(
        total_turns=("turn_id","count"),
        avg_response_ms=("response_time_ms","mean"),
        top_intent=("detected_intent", lambda x: x.mode()[0]),
    ).reset_index()
    agg.to_csv(ANALYTICS_CSV, index=False)
    print(f"[Analytics] Saved to {ANALYTICS_CSV}")

# ── STT ───────────────────────────────────────────────────────────────────────
class SpeechToText:
    def __init__(self, engine="google"):
        self.engine = engine
        self.recognizer = sr.Recognizer() if SR_AVAILABLE else None

    def from_microphone(self, duration=5, samplerate=16000):
        if not SR_AVAILABLE:
            raise RuntimeError("SpeechRecognition not available.")

        print(f"[STT] Listening for {duration} seconds... speak now")
        try:
            # 1. Record audio using sounddevice safely without PyAudio
            audio_array = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='int16')
            sd.wait()  # Wait until the recording is finished

            # 2. Convert raw numpy bytes directly into SpeechRecognition's AudioData container
            audio_data = sr.AudioData(audio_array.tobytes(), samplerate, 2)

            print("[STT] Processing speech...")
            return self.recognizer.recognize_google(audio_data)
        except Exception as e:
            print(f"[STT] Error: {e}")
            return ""

# ── TTS ───────────────────────────────────────────────────────────────────────
class TextToSpeech:
    def __init__(self, engine=TTS_ENGINE):
        self.engine = engine
        self._e = None
        if engine == "pyttsx3":
            try:
                self._e = pyttsx3.init()
                self._e.setProperty("rate", 165)
            except Exception as ex:
                print(f"[TTS] pyttsx3 failed: {ex}")
                self.engine = None

    def speak(self, text):
        if not text:
            return
        print(f"[TTS] {text}")
        if self.engine == "pyttsx3" and self._e:
            self._e.say(text)
            self._e.runAndWait()
        elif self.engine == "gtts":
            try:
                path = DATA_DIR / "_tts.mp3"
                gTTS(text=text, lang="en", slow=False).save(str(path))
                os.system(f'start /min wmplayer "{path.resolve()}"')
            except Exception as ex:
                print(f"[TTS] gTTS error: {ex}")

# ── LLM ───────────────────────────────────────────────────────────────────────
class OllamaLLM:
    def __init__(self, model=OLLAMA_MODEL):
        self.model = model
        self._history = []

    def chat(self, text, intent, entities):
        if not REQUESTS_AVAILABLE:
            return self._fallback(text, intent)
        self._history.append({"role": "user", "content": text})
        try:
            resp = requests.post(OLLAMA_URL, json={
                "model": self.model,
                "system": f"You are a helpful voice AI. Be concise. Intent: {intent}. Entities: {json.dumps(entities)}.",
                "messages": self._history[-10:],
                "stream": False,
            }, timeout=60)
            resp.raise_for_status()
            reply = resp.json().get("message", {}).get("content", "").strip()
            if reply:
                self._history.append({"role": "assistant", "content": reply})
                return reply
        except Exception as e:
            print(f"[LLM] Ollama not running: {e}")
        return self._fallback(text, intent)

    def _fallback(self, text, intent):
        m = {
            "greeting":    "Hello! How can I help you?",
            "farewell":    "Goodbye! Have a great day.",
            "task_query":  "No tasks yet. Say: create task colon your task name.",
            "summarize":   "Share the text you want me to summarize.",
            "reminder":    "Reminder noted. Start Ollama for full support.",
            "weather":     "Cannot fetch weather. Please check a weather app.",
            "help":        "I handle tasks, reminders, calculations and general questions.",
            "joke":        "Why did the AI cross the road? To get to the other dataset!",
            "definition":  "Start Ollama for detailed explanations.",
            "calculation": self._calc(text),
        }
        return m.get(intent, f"You said: {text}. Start Ollama for full AI responses.")

    @staticmethod
    def _calc(text):
        expr = re.sub(r"[^0-9+\-*/.() ]", "", text)
        try:
            return f"The result is {eval(expr, {'__builtins__': {}})}."
        except Exception:
            return "Could not parse that calculation."

# ── Router ────────────────────────────────────────────────────────────────────
class AgentRouter:
    def __init__(self, llm, tasks):
        self.llm   = llm
        self.tasks = tasks

    def route(self, text, intent, entities, commands):
        if intent == "farewell":
            return "Goodbye! Session ending."

        if "create_task" in commands:
            name = commands["create_task"][0].strip()
            self.tasks.append(name)
            return f"Task '{name}' created. You have {len(self.tasks)} task(s)."

        if intent == "task_query":
            if not self.tasks:
                return "You have no tasks right now."
            return "Your tasks: " + "; ".join(f"{i+1}. {t}" for i, t in enumerate(self.tasks))

        if intent == "calculation" and "calculate" in commands:
            expr = re.sub(r"[^0-9+\-*/.() ]", "", commands["calculate"][0])
            try:
                return f"The electrician result is {eval(expr, {'__builtins__': {}})}."
            except Exception:
                pass

        enriched = text
        if entities.get("date"):
            enriched += f" [dates: {', '.join(str(d) for d in entities['date'])}]"
        return self.llm.chat(enriched, intent, entities)

# ── Main Agent ────────────────────────────────────────────────────────────────
class VoiceAgent:
    def __init__(self, stt_engine="google", tts_engine=None, ollama_model=OLLAMA_MODEL):
        init_csv()
        self.session_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.turn_id    = 0
        self.tasks      = []
        self.stt    = SpeechToText(stt_engine)
        self.tts    = TextToSpeech(tts_engine or TTS_ENGINE or "pyttsx3")
        self.llm    = OllamaLLM(ollama_model)
        self.router = AgentRouter(self.llm, self.tasks)
        print(f"[Agent] Session {self.session_id} | STT={stt_engine} | TTS={tts_engine or TTS_ENGINE} | LLM={ollama_model}")

    def respond(self, user_text):
        if not user_text.strip():
            return ""
        self.turn_id += 1
        intent   = detect_intent(user_text)
        entities = extract_entities(user_text)
        commands = extract_commands(user_text)
        print(f"[Turn {self.turn_id}] Intent={intent} Entities={list(entities.keys())}")
        t0       = time.time()
        response = self.router.route(user_text, intent, entities, commands)
        elapsed  = int((time.time() - t0) * 1000)
        log_conversation({
            "timestamp": datetime.datetime.now().isoformat(),
            "session_id": self.session_id, "turn_id": self.turn_id,
            "user_transcript": user_text, "detected_intent": intent,
            "detected_entities": json.dumps(entities), "ai_response": response,
            "response_time_ms": elapsed, "tts_engine": self.tts.engine or "none",
            "stt_engine": self.stt.engine,
        })
        self.tts.speak(response)
        return response

    def run_cli(self):
        print("\nCommands: 'mic' = microphone | 'analytics' = show stats | 'quit' = exit\n")
        while True:
            try:
                user_input = input("You> ").strip()
            except (KeyboardInterrupt, EOFError):
                break
            if not user_input:
                continue
            if user_input.lower() in ("quit","exit","bye"):
                self.tts.speak("Goodbye!")
                break
            if user_input.lower() == "mic":
                # Calls the new sounddevice-backed STT pipeline
                user_input = self.stt.from_microphone(duration=5)
                if not user_input:
                    print("[STT] Could not understand or no audio captured. Try again.")
                    continue
                print(f"You (voice)> {user_input}")
            if user_input.lower() == "analytics":
                build_analytics()
                self._print_analytics()
                continue
            print(f"Agent> {self.respond(user_input)}\n")
        build_analytics()
        print(f"[Agent] Logs saved to {CONVERSATION_CSV}")

    def _print_analytics(self):
        if not CONVERSATION_CSV.exists():
            return
        df = pd.read_csv(CONVERSATION_CSV)
        if df.empty:
            print("[Analytics] No data yet.")
            return
        print(f"\n  Total turns      : {len(df)}")
        print(f"  Avg response(ms) : {df['response_time_ms'].mean():.0f}")
        print(f"  Intent breakdown : {dict(Counter(df['detected_intent']))}\n")

if __name__ == "__main__":
    agent = VoiceAgent()
    agent.run_cli()