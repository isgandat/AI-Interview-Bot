from __future__ import annotations
import os, json, uuid, re
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates

from ai_interview_bot.backend.llm_providers import OllamaChat
from ai_interview_bot.backend.evaluator import evaluate_answer
from ai_interview_bot.backend.generator import generate_question, extract_metadata

# --- paths / storage ---
BASE_DIR = Path(__file__).resolve().parents[2]
STORAGE = BASE_DIR / "storage"
WEB_SESSIONS = STORAGE / "web_sessions"
WEB_SESSIONS.mkdir(parents=True, exist_ok=True)

PKG_CONFIG = Path(__file__).resolve().parents[1] / "configs" / "domains.yaml"

# --- helpers ---
def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", (text or "").lower())).strip()

def jaccard(a: str, b: str) -> float:
    ta, tb = set(normalize(a).split()), set(normalize(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)

def near_duplicate(candidate: str, used: list[str], threshold: float = 0.85) -> bool:
    cand = normalize(candidate)
    for u in used:
        if jaccard(cand, u) >= threshold:
            return True
    return False

def load_domains():
    import yaml
    with open(PKG_CONFIG, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)["domains"]

def _new_sid() -> str:
    return uuid.uuid4().hex

def _persist(sid: str, data: dict):
    (WEB_SESSIONS / f"{sid}.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

def _load(sid: str) -> dict:
    p = WEB_SESSIONS / f"{sid}.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    raise HTTPException(404, "Session not found")

# --- FastAPI app ---
app = FastAPI(title="AI Interview Bot (Web)")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

def _next_question(llm: OllamaChat, s: dict) -> str:
    """Generate a fresh question using diversity/anti-repeat logic."""
    recent_window = int(os.getenv("DIVERSITY_RECENT_WINDOW", "150"))
    avoid_list = s["used_questions"][-recent_window:]
    top_used_topics = sorted((s["tag_counts"] or {}).items(), key=lambda kv: kv[1], reverse=True)[:5]
    top_used_topics = [k for k, _ in top_used_topics]
    order = ["easy", "medium", "hard"]
    seen = sum((s["diff_counts"] or {}).values()) if s["diff_counts"] else 0
    next_diff = order[(seen or 0) % 3]

    # try a few times to get a non-duplicate
    for _ in range(5):
        cand = generate_question(
            llm,
            s["domain"],
            s["persona"],
            s["turns"],
            avoid_list=avoid_list,
            avoid_topics=top_used_topics,
            difficulty_hint=next_diff,
        )
        if cand and (norm := normalize(cand)) not in s["used_questions"] and not near_duplicate(
            cand, s["used_questions"]
        ):
            return cand

    # last resort
    return generate_question(
        llm,
        s["domain"],
        s["persona"],
        s["turns"],
        avoid_list=avoid_list,
        avoid_topics=top_used_topics,
        difficulty_hint=next_diff,
    )

@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    domains = load_domains()
    return templates.TemplateResponse("index.html", {"request": request, "domains": domains, "env": os.environ})

@app.post("/start")
def start(request: Request, domain: str = Form(...)):
    domains = load_domains()
    if domain not in domains:
        raise HTTPException(400, "Unknown domain")
    sid = _new_sid()
    persona = domains[domain].get("persona", "")
    s = {
        "sid": sid,
        "domain": domain,
        "persona": persona,
        "created_at": datetime.utcnow().isoformat() + "Z",
        "used_questions": [],  # normalized
        "tag_counts": {},
        "diff_counts": {},
        "turns": [],
        "current_q": None,
    }
    _persist(sid, s)
    return RedirectResponse(url=f"/chat/{sid}", status_code=303)

@app.get("/chat/{sid}", response_class=HTMLResponse)
def chat(request: Request, sid: str):
    s = _load(sid)
    llm = OllamaChat()
    if not s.get("current_q"):
        q = _next_question(llm, s)
        s["current_q"] = q
        nq = normalize(q)
        if nq not in s["used_questions"]:
            s["used_questions"].append(nq)
        meta = extract_metadata(llm, q)
        for t in meta.get("tags", []):
            s["tag_counts"][t] = s["tag_counts"].get(t, 0) + 1
        d = meta.get("difficulty", "medium")
        s["diff_counts"][d] = s["diff_counts"].get(d, 0) + 1
        _persist(sid, s)
    return templates.TemplateResponse("chat.html", {"request": request, "s": s, "env": os.environ})

@app.post("/chat/{sid}/answer", response_class=HTMLResponse)
def answer(request: Request, sid: str, user_answer: str = Form(...)):
    s = _load(sid)
    if not s.get("current_q"):
        return RedirectResponse(url=f"/chat/{sid}", status_code=303)

    llm = OllamaChat()
    q = s["current_q"]

    # evaluate
    try:
        feedback = evaluate_answer(llm, q, user_answer)
    except Exception as e:
        feedback = None
        s.setdefault("warnings", []).append(str(e))

    # store turn
    s["turns"].append({"q": q, "a": user_answer, "feedback": feedback})

    # next question
    s["current_q"] = _next_question(llm, s)
    nq = normalize(s["current_q"])
    if nq not in s["used_questions"]:
        s["used_questions"].append(nq)
    _persist(sid, s)

    return RedirectResponse(url=f"/chat/{sid}", status_code=303)

# NEW: regenerate a question without submitting an answer
@app.post("/chat/{sid}/new", response_class=HTMLResponse)
def new_question(request: Request, sid: str):
    s = _load(sid)
    llm = OllamaChat()

    q = _next_question(llm, s)
    s["current_q"] = q

    nq = normalize(q)
    if nq not in s["used_questions"]:
        s["used_questions"].append(nq)

    meta = extract_metadata(llm, q)
    for t in meta.get("tags", []):
        s["tag_counts"][t] = s["tag_counts"].get(t, 0) + 1
    d = meta.get("difficulty", "medium")
    s["diff_counts"][d] = s["diff_counts"].get(d, 0) + 1

    _persist(sid, s)
    return RedirectResponse(url=f"/chat/{sid}", status_code=303)

# dev runner
def dev():
    import uvicorn
    uvicorn.run("ai_interview_bot.web.app:app", host="127.0.0.1", port=8000, reload=True)
