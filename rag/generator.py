import json
import urllib.error
import urllib.request

from openai import OpenAI

import config

SYSTEM_PROMPT = """You are a privacy and data protection assistant.
Answer questions using ONLY the provided context from privacy documentation.
If the context does not contain enough information, say so clearly.
Cite the source document name when referencing specific rules or principles."""


def _build_user_prompt(query: str, context_blocks: list[str]) -> str:
    context = "\n\n---\n\n".join(context_blocks)
    return f"""Context:
{context}

Question: {query}

Answer based on the context above:"""


def _format_context(chunks: list) -> list[str]:
    return [
        f"[{c.source} | relevance: {c.score:.2f}]\n{c.text}"
        for c in chunks
    ]


class Generator:
    def generate(self, query: str, chunks: list) -> str:
        context_blocks = _format_context(chunks)
        user_prompt = _build_user_prompt(query, context_blocks)

        if config.OPENAI_API_KEY:
            try:
                return self._generate_openai(user_prompt)
            except Exception:
                pass
        if self._ollama_model_available():
            try:
                return self._generate_ollama(user_prompt)
            except Exception:
                pass
        return self._generate_extractive(query, chunks)

    def _generate_openai(self, user_prompt: str) -> str:
        client = OpenAI(api_key=config.OPENAI_API_KEY)
        response = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        return response.choices[0].message.content.strip()

    def _ollama_model_available(self) -> bool:
        url = f"{config.OLLAMA_BASE_URL.rstrip('/')}/api/tags"
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=2) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            names = {m.get("name", "").split(":")[0] for m in body.get("models", [])}
            target = config.OLLAMA_MODEL.split(":")[0]
            return target in names
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            return False

    def _generate_ollama(self, user_prompt: str) -> str:
        url = f"{config.OLLAMA_BASE_URL.rstrip('/')}/api/chat"
        payload = {
            "model": config.OLLAMA_MODEL,
            "stream": False,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            "options": {"temperature": 0.2},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        return body["message"]["content"].strip()

    def _generate_extractive(self, query: str, chunks: list) -> str:
        if not chunks:
            return "No relevant documents found. Run `python main.py ingest` first."

        lines = [
            "No LLM configured (set OPENAI_API_KEY or run Ollama). "
            "Top retrieved passages:\n"
        ]
        for i, c in enumerate(chunks, 1):
            lines.append(f"{i}. [{c.source}] (score {c.score:.2f})\n{c.text}\n")
        return "\n".join(lines)
