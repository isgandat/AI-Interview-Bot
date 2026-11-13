import json

import pytest

from ai_interview_bot.backend.evaluator import evaluate_answer

CRITERIA = {"clarity", "correctness", "structure", "depth", "communication"}

VALID_EVALUATION = {
    "scores": {"clarity": 4, "correctness": 5, "structure": 3, "depth": 2, "communication": 4},
    "summary": "Correct but shallow.",
    "tips": ["Give an example", "Mention trade-offs", "Be concise"],
    "ideal_answer": "A process has its own memory space; threads share it.",
}


def test_evaluate_answer_returns_five_criterion_scores(make_llm):
    llm = make_llm(evaluations=["```json\n" + json.dumps(VALID_EVALUATION) + "\n```"])

    result = evaluate_answer(llm, "Process vs thread?", "Threads share memory.")

    assert set(result["scores"]) == CRITERIA
    assert all(isinstance(v, int) and 1 <= v <= 5 for v in result["scores"].values())
    assert len(result["tips"]) == 3
    call = llm.calls[0]
    assert call["options"] == {"temperature": 0.0}
    assert "Process vs thread?" in call["messages"][1]["content"]
    assert "Threads share memory." in call["messages"][1]["content"]


def test_evaluate_answer_repairs_malformed_json_with_second_llm_call(make_llm):
    broken = '{"scores": {"clarity": 4, "correctness": 5, "structure": 3, "depth": 2, "communication": 4},}'
    llm = make_llm(evaluations=[broken], repairs=[json.dumps(VALID_EVALUATION)])

    result = evaluate_answer(llm, "Process vs thread?", "Threads share memory.")

    assert result == VALID_EVALUATION
    assert len(llm.calls) == 2
    repair_prompt = llm.calls[1]["messages"][1]["content"]
    assert broken in repair_prompt
    assert llm.calls[1]["options"] == {"temperature": 0.0}


def test_evaluate_answer_raises_when_repair_also_fails(make_llm):
    llm = make_llm(evaluations=["no json here"], repairs=["still no json"])

    with pytest.raises(ValueError, match="Failed to parse JSON"):
        evaluate_answer(llm, "Q", "A")
    assert len(llm.calls) == 2
