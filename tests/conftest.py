import pytest
from fastapi.testclient import TestClient

from ai_interview_bot.backend.prompts import EVAL_SYSTEM, SYSTEM_EXTRACT_META
from ai_interview_bot.web import app as web_app


class FakeLLM:
    """Offline stand-in for OllamaChat: replies from a queue per prompt type and records every call."""

    def __init__(self, questions=(), evaluations=(), repairs=(),
                 metadata='{"tags": ["concurrency"], "difficulty": "medium"}'):
        self.questions = list(questions)
        self.evaluations = list(evaluations)
        self.repairs = list(repairs)
        self.metadata = metadata
        self.calls = []

    def chat(self, messages, **options):
        self.calls.append({"messages": messages, "options": options})
        system = messages[0]["content"]
        if system == EVAL_SYSTEM:
            return self.evaluations.pop(0)
        if system == SYSTEM_EXTRACT_META:
            return self.metadata
        if system.startswith("Return ONLY a valid JSON object"):
            return self.repairs.pop(0)
        return self.questions.pop(0)


@pytest.fixture
def make_llm():
    return FakeLLM


@pytest.fixture
def web(monkeypatch, tmp_path):
    """TestClient whose sessions live in tmp_path and whose LLM is a FakeLLM (no Ollama, no network)."""
    llm = FakeLLM()
    monkeypatch.setattr(web_app, "WEB_SESSIONS", tmp_path)
    monkeypatch.setattr(web_app, "OllamaChat", lambda: llm)
    return TestClient(web_app.app), llm, tmp_path
