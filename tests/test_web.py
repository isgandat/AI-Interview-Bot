import json

import jinja2
import pytest

FEEDBACK = {
    "scores": {"clarity": 4, "correctness": 5, "structure": 3, "depth": 2, "communication": 4},
    "summary": "Correct but shallow.",
    "tips": ["Give an example", "Mention trade-offs", "Be concise"],
    "ideal_answer": "Processes have separate memory; threads share it.",
}


def _start(client, domain="software_engineering"):
    r = client.post("/start", data={"domain": domain}, follow_redirects=False)
    assert r.status_code == 303
    return r.headers["location"].rsplit("/", 1)[-1]


def test_index_lists_configured_domains(web):
    client, _, _ = web
    r = client.get("/")
    assert r.status_code == 200
    for label in ("Software Engineering", "Data Science", "Behavioral"):
        assert label in r.text


def test_start_rejects_unknown_domain(web):
    client, _, sessions = web
    r = client.post("/start", data={"domain": "astrology"})
    assert r.status_code == 400
    assert list(sessions.iterdir()) == []


def test_interview_flow_start_ask_answer(web):
    client, llm, sessions = web
    llm.questions = ["Explain the difference between a process and a thread.", "How does a hash map handle collisions?"]
    llm.evaluations = [json.dumps(FEEDBACK)]

    sid = _start(client)
    page = client.get(f"/chat/{sid}")
    assert "Explain the difference between a process and a thread." in page.text

    r = client.post(f"/chat/{sid}/answer", data={"user_answer": "Threads share memory."}, follow_redirects=False)
    assert r.status_code == 303

    saved = json.loads((sessions / f"{sid}.json").read_text())
    assert saved["turns"][0]["a"] == "Threads share memory."
    assert saved["turns"][0]["feedback"] == FEEDBACK
    assert saved["current_q"] == "How does a hash map handle collisions?"
    assert saved["used_questions"] == [
        "explain the difference between a process and a thread",
        "how does a hash map handle collisions",
    ]
    assert saved["tag_counts"] == {"concurrency": 1}

    page = client.get(f"/chat/{sid}")
    assert page.status_code == 200
    assert "Correct but shallow." in page.text
    assert "communication" in page.text


@pytest.mark.xfail(
    raises=jinja2.exceptions.UndefinedError,
    strict=True,
    reason="Known bug: evaluator accepts any valid JSON; feedback without 'scores' crashes chat.html (500).",
)
def test_chat_page_survives_feedback_without_scores(web):
    client, llm, _ = web
    llm.questions = ["What is a race condition?", "What is a deadlock?"]
    llm.evaluations = ['{"summary": "Good answer", "tips": ["a", "b", "c"]}']

    sid = _start(client)
    client.get(f"/chat/{sid}")
    client.post(f"/chat/{sid}/answer", data={"user_answer": "Two threads..."}, follow_redirects=False)

    assert client.get(f"/chat/{sid}").status_code == 200
