# AI Interview Practice Tool

A web app for practicing technical and behavioral interviews with a **local LLM**.
It generates domain-specific questions, scores your answers and suggests how to improve them.
Built as an Individual Software Project at Charles University.

## Features
- **Question generation** by a local model through the [Ollama](https://ollama.com) REST API (`/api/chat`, with `/api/generate` as a fallback).
- **Three interview domains**, configured in `configs/domains.yaml`: software engineering, data science and behavioral (STAR).
- **No repeated questions**: Jaccard similarity on normalized tokens rejects near-duplicates of earlier questions.
- **Structured feedback** in JSON: scores from 1 to 5 for clarity, correctness, structure, depth and communication, plus a summary, tips and an ideal answer.
- **Robust parsing**: if the model returns malformed JSON, the app asks it to repair the output and retries.
- **Session history** is saved as JSON on the local disk.

## Tech stack
Python 3.10+ · FastAPI · Jinja2 · HTMX · Tailwind CSS · Ollama (default model `llama3.1:8b`) · Poetry

## Quick start
```bash
# 1. Install Ollama and pull a model
ollama pull llama3.1:8b

# 2. Install dependencies and start the web UI
poetry install
poetry run ai-bot-web
# open http://127.0.0.1:8000
```

## Configuration
| Variable | Default | Purpose |
|---|---|---|
| `AI_BOT_MODEL` | `llama3.1:8b` | Ollama model |
| `OLLAMA_URL` | `http://localhost:11434` | Ollama server |
| `AI_BOT_TEMPERATURE` | `0.6` | Sampling temperature |
| `AI_BOT_NUM_CTX` | `1024` | Context window |
| `DIVERSITY_RECENT_WINDOW` | `150` | How many recent questions to check for repeats |

## Project structure
```
src/ai_interview_bot/
  backend/
    llm_providers.py   # Ollama HTTP client
    prompts.py         # interviewer, generator and evaluator prompts
    generator.py       # question generation + metadata (tags, difficulty)
    evaluator.py       # JSON evaluation with auto-repair
  web/
    app.py             # FastAPI routes
    templates/         # Jinja + HTMX + Tailwind views
  configs/domains.yaml # interview domains and personas
```
