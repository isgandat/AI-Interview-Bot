import json, re
from ai_interview_bot.backend.prompts import EVAL_SYSTEM, PROMPT_EVALUATE

def _extract_json(text: str):
    if not text or not isinstance(text, str):
        raise ValueError("Empty model response")
    s = text.strip()
    m = re.search(r"```json\s*(\{.*?\})\s*```", s, flags=re.S)
    if m:
        return json.loads(m.group(1))
    start, end = s.find('{'), s.rfind('}')
    if start != -1 and end != -1 and end > start:
        candidate = s[start:end+1]
        try:
            return json.loads(candidate)
        except Exception:
            pass
    try:
        return json.loads(s)
    except Exception:
        try:
            return json.loads(s.replace("'", '"'))
        except Exception as e:
            raise ValueError(f"Failed to parse JSON from model output: {e}\nRaw: {s[:500]}") from e

def evaluate_answer(llm, question, answer):
    msgs = [
        {"role":"system", "content":EVAL_SYSTEM},
        {"role":"user", "content":PROMPT_EVALUATE.format(question=question, answer=answer)}
    ]
    raw = llm.chat(msgs, temperature=0.0)
    try:
        return _extract_json(raw)
    except Exception:
        repair_user = (
            "Convert the following content into valid JSON that matches the required schema.\n\n"
            "Content:\n" + (raw or "<empty>") + "\n\n"
            "Return ONLY the corrected JSON. No code fences, no commentary."
        )
        fix = llm.chat(
            [
                {"role":"system","content":"Return ONLY a valid JSON object. No extra text."},
                {"role":"user","content": repair_user}
            ],
            temperature=0.0
        )
        return _extract_json(fix)
