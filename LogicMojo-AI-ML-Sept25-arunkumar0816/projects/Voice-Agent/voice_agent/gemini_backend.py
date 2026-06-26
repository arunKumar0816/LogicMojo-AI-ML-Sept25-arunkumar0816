"""
gemini_backend.py — drop-in replacement for OllamaLLM using Google Gemini.


"""

import os
import re
import json

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    print("[Gemini] google-generativeai not installed. Run: pip install google-generativeai")


class GeminiLLM:
    def __init__(self, api_key: str | None = None, model: str = "gemini-1.5-flash"):
        self.model_name = model
        self._history: list = []

        key = api_key or os.getenv("GEMINI_API_KEY", "")
        if not key:
            print("[Gemini] WARNING: No API key. Set GEMINI_API_KEY env var or pass api_key=...")

        if GEMINI_AVAILABLE and key:
            genai.configure(api_key=key)
            self._model = genai.GenerativeModel(
                model_name=model,
                system_instruction=(
                    "You are a helpful voice AI assistant. "
                    "Keep responses concise — they will be spoken aloud."
                ),
            )
            self._chat = self._model.start_chat(history=[])
        else:
            self._model = None
            self._chat  = None

    def chat(self, user_text: str, intent: str, entities: dict) -> str:
        if self._chat is None:
            return self._fallback(user_text, intent)

        enriched = (
            f"[Intent: {intent}] "
            f"[Entities: {json.dumps(entities)}] "
            f"{user_text}"
        )
        try:
            response = self._chat.send_message(enriched)
            return response.text.strip()
        except Exception as e:
            print(f"[Gemini] API error: {e}")
            return self._fallback(user_text, intent)

    def _fallback(self, text: str, intent: str) -> str:
        defaults = {
            "greeting": "Hello! How can I help you?",
            "farewell": "Goodbye! Take care.",
            "joke":     "Why do  scientists trust engineers? Because they make up everything!",
        }
        return defaults.get(intent, f"I received: '{text}'. (Gemini unavailable.)")

    def reset_history(self):
       
        if self._model:
            self._chat = self._model.start_chat(history=[])
        self._history = []
