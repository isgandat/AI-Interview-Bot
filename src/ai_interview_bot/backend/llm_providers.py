import os, requests

class OllamaChat:
    def __init__(self, model: str | None = None, base_url: str | None = None):
        self.model = model or os.getenv("AI_BOT_MODEL", "llama3.1:8b")
        self.base_url = base_url or os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.url_chat = f"{self.base_url}/api/chat"
        self.url_gen  = f"{self.base_url}/api/generate"

        self.default_options = {
            "temperature": float(os.getenv("AI_BOT_TEMPERATURE", "0.6")),
            "top_p": float(os.getenv("AI_BOT_TOP_P", "0.95")),
            "repeat_penalty": float(os.getenv("AI_BOT_REPEAT_PENALTY", "1.1")),
            "num_ctx": int(os.getenv("AI_BOT_NUM_CTX", "1024")),
            "num_thread": int(os.getenv("AI_BOT_NUM_THREAD", "8")),
        }
        self.keep_alive = os.getenv("AI_BOT_KEEP_ALIVE", "30m")

    def _messages_to_prompt(self, messages):
        parts = []
        for m in messages:
            role = m.get("role"); content = m.get("content", "")
            parts.append(f"[{role}]\n{content}\n[/{role}]")
        return "\n".join(parts)

    def chat(self, messages, **overrides):
        options = {**self.default_options, **overrides}
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": options,
            "keep_alive": self.keep_alive,
        }
        r = requests.post(self.url_chat, json=payload, timeout=180)
        if r.status_code == 404:
            gen_payload = {
                "model": self.model,
                "prompt": self._messages_to_prompt(messages),
                "stream": False,
                "options": options,
            }
            rg = requests.post(self.url_gen, json=gen_payload, timeout=180)
            if not rg.ok:
                raise RuntimeError(f"Ollama error {rg.status_code}: {rg.text}")
            return rg.json().get("response", "")
        if not r.ok:
            raise RuntimeError(f"Ollama error {r.status_code}: {r.text}")
        return r.json().get("message", {}).get("content", "")
