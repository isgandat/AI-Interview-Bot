from unittest.mock import Mock

import pytest

from ai_interview_bot.backend.llm_providers import OllamaChat

MESSAGES = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "Hello"}]


def _response(status, payload):
    return Mock(status_code=status, ok=status < 400, text=str(payload), json=Mock(return_value=payload))


@pytest.fixture
def post(monkeypatch):
    for var in ("AI_BOT_MODEL", "OLLAMA_URL", "AI_BOT_TEMPERATURE", "AI_BOT_NUM_CTX"):
        monkeypatch.delenv(var, raising=False)
    mock = Mock()
    monkeypatch.setattr("ai_interview_bot.backend.llm_providers.requests.post", mock)
    return mock


def test_chat_posts_to_api_chat_and_returns_content(post):
    post.return_value = _response(200, {"message": {"role": "assistant", "content": "Hi there"}})

    reply = OllamaChat(base_url="http://ollama.test").chat(MESSAGES, temperature=0.0)

    assert reply == "Hi there"
    url, kwargs = post.call_args.args[0], post.call_args.kwargs
    assert url == "http://ollama.test/api/chat"
    payload = kwargs["json"]
    assert payload["model"] == "llama3.1:8b"
    assert payload["stream"] is False
    assert payload["messages"] == MESSAGES
    assert payload["options"]["temperature"] == 0.0
    assert payload["options"]["num_ctx"] == 1024


def test_chat_falls_back_to_generate_on_404(post):
    post.side_effect = [_response(404, {"error": "not found"}), _response(200, {"response": "fallback text"})]

    reply = OllamaChat(base_url="http://ollama.test").chat(MESSAGES)

    assert reply == "fallback text"
    assert [c.args[0] for c in post.call_args_list] == [
        "http://ollama.test/api/chat",
        "http://ollama.test/api/generate",
    ]
    prompt = post.call_args_list[1].kwargs["json"]["prompt"]
    assert "[system]\nBe brief.\n[/system]" in prompt
    assert "[user]\nHello\n[/user]" in prompt
