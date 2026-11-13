import pytest

from ai_interview_bot.web.app import _next_question, jaccard, near_duplicate, normalize


@pytest.mark.parametrize(
    "a, b, expected",
    [
        pytest.param("What is a Python decorator?", "what is a python DECORATOR", 1.0, id="identical-ignoring-case-and-punctuation"),
        pytest.param("binary search tree", "hash map collisions", 0.0, id="disjoint"),
        pytest.param("a b c", "b c d", 0.5, id="partial-overlap"),
        pytest.param("", "", 0.0, id="both-empty"),
        pytest.param("?!", "what is recursion", 0.0, id="punctuation-only-is-empty"),
    ],
)
def test_jaccard(a, b, expected):
    assert jaccard(a, b) == pytest.approx(expected)


WORDS = [f"w{i}" for i in range(20)]


@pytest.mark.parametrize(
    "used, expected",
    [
        pytest.param([" ".join(WORDS[:17])], True, id="17-of-20-shared-is-exactly-0.85"),
        pytest.param([" ".join(WORDS[:16])], False, id="16-of-20-shared-is-0.80"),
        pytest.param([], False, id="no-previous-questions"),
    ],
)
def test_near_duplicate_threshold_is_inclusive(used, expected):
    assert near_duplicate(" ".join(WORDS), used) is expected


def test_next_question_skips_near_duplicates(make_llm):
    session = {
        "domain": "software_engineering",
        "persona": "",
        "turns": [],
        "used_questions": [normalize("What is a Python decorator and when would you use one?")],
        "tag_counts": {},
        "diff_counts": {},
    }
    llm = make_llm(questions=[
        "WHAT is a Python decorator, and when would you use one?!",  # exact match after normalization
        "What is a Python decorator and when would you use one today?",  # Jaccard 11/12 = 0.92
        "How does garbage collection work in CPython?",
    ])

    assert _next_question(llm, session) == "How does garbage collection work in CPython?"
    assert len(llm.calls) == 3
    first_prompt = llm.calls[0]["messages"][1]["content"]
    assert "- what is a python decorator and when would you use one" in first_prompt
    assert "difficulty: easy" in first_prompt
