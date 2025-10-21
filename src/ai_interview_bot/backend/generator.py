import json, re
from ai_interview_bot.backend.prompts import (
    SYSTEM_QUESTION_GENERATOR, PROMPT_GENERATE_QUESTION,
    SYSTEM_EXTRACT_META, PROMPT_EXTRACT_META
)

def _extract_json(text: str):
    s = (text or "").strip()
    m = re.search(r"```json\s*(\{.*?\})\s*```", s, flags=re.S)
    if m:
        return json.loads(m.group(1))
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(s[start:end+1])
        except Exception:
            pass
    try:
        return json.loads(s)
    except Exception:
        return json.loads(s.replace("'", '"'))

def generate_question(llm, domain, persona, history, avoid_list, avoid_topics, difficulty_hint=None):
    if history:
        ctx = "\n".join([f"Q: {h['q']}\nA: {h['a']}" for h in history[-3:]])
    else:
        ctx = "No prior Q&A."
    sys = SYSTEM_QUESTION_GENERATOR.format(domain=domain)
    user = PROMPT_GENERATE_QUESTION.format(
        context=ctx,
        avoid_list="\n".join(f"- {q}" for q in (avoid_list or [])) or "- (none)",
        avoid_topics=", ".join(sorted(set(avoid_topics or []))) or "(none)",
        difficulty_hint=difficulty_hint or "(no preference)",
    )
    return llm.chat([{"role":"system","content":sys},
                     {"role":"user","content":user}],
                    temperature=0.8)

def extract_metadata(llm, question):
    sys = SYSTEM_EXTRACT_META
    user = PROMPT_EXTRACT_META.format(question=question)
    raw = llm.chat([{"role":"system","content":sys},
                    {"role":"user","content":user}],
                   temperature=0.0)
    try:
        meta = _extract_json(raw)
        tags = [t.strip().lower() for t in meta.get("tags", []) if isinstance(t,str)]
        diff = str(meta.get("difficulty","medium")).lower()
        if diff not in {"easy","medium","hard"}: diff = "medium"
        return {"tags": tags[:4], "difficulty": diff}
    except Exception:
        return {"tags": [], "difficulty": "medium"}
