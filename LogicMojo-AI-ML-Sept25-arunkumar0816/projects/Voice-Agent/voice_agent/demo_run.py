

import sys
from pathlib import Path

# ── make voice_agent importable ──────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

from voice_agent import VoiceAgent, detect_intent, extract_entities, extract_commands, build_analytics
import analytics_report

# ── Sample dialogues ──────────────────────────────────────────────────────────
DEMO_INPUTS = [
    "Hello! How are you?",
    "What tasks do I have today?",
    "Remind me to call at 3:00 PM",
    "Calculate 245 * 18 + 300",
    "What tasks do I have?",
    "Tell me a joke",
    "Check the weather in Bengaluru",
    "Goodbye!",
]

def run_demo():
    print("=" * 60)
    print("  VOICE AGENTIC AI ASSISTANT — DEMO RUN")
    print("=" * 60)

    agent = VoiceAgent(
        stt_engine="google",
        tts_engine=None,          # silence TTS for demo (set to "pyttsx3" to hear audio)
        ollama_model="codellama",
    )

    print("\n── RegEx Pattern Extraction Demo ──────────────────────────")
    sample = "Contact alice@example.com or call +91-9876543210 by 2024-12-31"
    ents = extract_entities(sample)
    print(f"  Text     : {sample}")
    print(f"  Entities : {ents}\n")

    print("── Intent Detection Demo ───────────────────────────────────")
    for t in ["Hi there!", "What tasks do I have?", "Calculate 5 + 3", "Bye!"]:
        print(f"  '{t}' → {detect_intent(t)}")

    print("\n── Full Conversation Simulation ────────────────────────────\n")
    for user_text in DEMO_INPUTS:
        print(f"You  > {user_text}")
        response = agent.respond(user_text)
        print(f"Agent> {response}\n")

        if detect_intent(user_text) == "farewell":
            break

    print("\n── Generating Analytics Report ─────────────────────────────")
    agent._print_analytics()

    try:
        analytics_report.run()
        print(f"\n✅  Full XLSX report saved to: {analytics_report.XLSX_OUT}")
    except Exception as e:
        print(f"[Report] {e}")

    print("\n✅  Demo complete!")
    print(f"   CSV log  : data/conversations.csv")
    print(f"   Analytics: data/analytics.csv")
    print(f"   Report   : data/report.xlsx")


if __name__ == "__main__":
    run_demo()
