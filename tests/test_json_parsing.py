import pytest

from ai_interview_bot.backend.evaluator import _extract_json
from ai_interview_bot.backend.generator import extract_metadata


@pytest.mark.parametrize(
    "raw, expected",
    [
        pytest.param(
            'Sure! Here is my evaluation:\n```json\n{"scores": {"clarity": 4, "depth": 3}, "summary": "Solid"}\n```\nLet me know!',
            {"scores": {"clarity": 4, "depth": 3}, "summary": "Solid"},
            id="json-code-fence-with-chatter",
        ),
        pytest.param(
            '```\n{"summary": "No language tag on the fence"}\n```',
            {"summary": "No language tag on the fence"},
            id="plain-code-fence",
        ),
        pytest.param(
            'The JSON you asked for: {"summary": "ok", "tips": ["a", "b"]} -- hope this helps.',
            {"summary": "ok", "tips": ["a", "b"]},
            id="prose-around-json",
        ),
        pytest.param(
            "{'summary': 'single quotes', 'tips': []}",
            {"summary": "single quotes", "tips": []},
            id="python-style-single-quotes",
        ),
    ],
)
def test_extracts_json_from_realistic_llm_output(raw, expected):
    assert _extract_json(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        pytest.param('{"summary": "trailing comma",}', id="trailing-comma"),
        pytest.param("I'm sorry, I cannot evaluate this answer.", id="no-json-at-all"),
        pytest.param("", id="empty"),
    ],
)
def test_unparseable_output_raises_value_error(raw):
    # The local parser does not fix these; evaluate_answer() then asks the LLM to repair them.
    with pytest.raises(ValueError):
        _extract_json(raw)


@pytest.mark.parametrize(
    "raw, expected",
    [
        pytest.param(
            '```json\n{"tags": [" Python ", "GIL", 42, "Threads", "Async", "Extra"], "difficulty": "EXTREME"}\n```',
            {"tags": ["python", "gil", "threads", "async"], "difficulty": "medium"},
            id="messy-tags-and-unknown-difficulty",
        ),
        pytest.param("not json", {"tags": [], "difficulty": "medium"}, id="garbage-falls-back"),
    ],
)
def test_extract_metadata_normalizes_model_output(make_llm, raw, expected):
    llm = make_llm(metadata=raw)
    assert extract_metadata(llm, "What is the GIL?") == expected
