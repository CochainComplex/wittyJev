"""Logical forms for `jevk5.tractatus`, written by an LLM on an OpenAI-compatible server.

    from jevk5 import JevK5GGUF
    from jevk5.decompose import Decomposer
    from jevk5.tractatus import Tractatus

    model = Tractatus(JevK5GGUF(temperature=1.22), Decomposer(url=..., model=..., cache="data/llm-cache.jsonl"))

The LLM (any OpenAI-compatible server; a local llama.cpp server by default; a key, if the server needs one,
in $LLM_API_KEY) sees the evidence and the question
and writes atoms plus a formula (noul) or a decision list (choice, score). It is told never to answer.
Questions it calls atomic, and forms that fail `tractatus.validate`, give None, so they are read whole.
`answer` asks the same LLM to decide instead (a control, and a cross-check of labels). Replies are cached
by request in a JSONL file, so a rerun asks nothing twice. Standard library only.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
import urllib.request
from pathlib import Path

from .prompt import decision_options
from .tractatus import MAX_ATOMS, validate

SYSTEM = (
    "You analyse typed decision questions into elementary propositions, after Wittgenstein's Tractatus: "
    "the answer is a truth-function of simple facts about the case. You never decide the question and "
    "never say which facts hold."
)
INSTRUCTIONS = f"""Write the logical form of this question as a JSON object.
- "atoms": 1 to {MAX_ATOMS} yes/no questions. Each asks about exactly one fact of the case that the evidence \
can settle, names its entities, dates and amounts explicitly, and reads on its own. Ask in the positive; put \
negation in the formula.
- For a yes/no question (type noul): "formula": a Python boolean expression over a1, a2, ... (and, or, not, \
parentheses) that is true exactly when the answer is option "true".
- For a choice or score question (a score's option ids are its level numbers): "rules": an ordered list of \
{{"option": id, "when": formula}}; the first rule whose \
formula is true gives the answer. Follow the precedence the evidence states (exceptions before the rules they \
override); otherwise make the conditions mutually exclusive. "otherwise": the id to answer when no rule holds.
- If the question already asks about a single fact, reply {{"atomic": true}}.
Never include the answer, your reasoning, or which atoms are true. Reply with the JSON object only, shaped as
{{"atoms": ["question", ...], "formula": "..."}} (noul), \
{{"atoms": ["question", ...], "rules": [{{"option": "id", "when": "..."}}, ...], "otherwise": "id"}} (choice), \
or {{"atomic": true}}. Atom i is a<i> in the formulas. Example: {{"atoms": ["Did the customer buy the item \
30 days ago or less?", "Does the customer have a receipt?"], "formula": "a1 and a2"}}"""
BLIND = "\nThe evidence is not shown to you: write the atoms that any case of this kind would need settled."
DECIDE_SYSTEM = "You decide typed decision questions from the evidence given."
DECIDE = 'Decide the question. Reply with the JSON object {"answer": "<option id>"} only.'


def reply(url: str, model: str, key: str, messages: list[dict], timeout_s: float, **extra) -> dict:
    """The reply message of one chat completion (content, and reasoning_content when the server splits it)."""
    request = urllib.request.Request(
        url.rstrip("/") + "/chat/completions",
        data=json.dumps({"model": model, "messages": messages, **extra}).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"},
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        data = json.loads(response.read())
    if "choices" not in data:  # some servers report errors in a 200 response
        raise RuntimeError(f"{model}: {str(data.get('error', data))[:300]}")
    return data["choices"][0]["message"]


def chat(url: str, model: str, key: str, messages: list[dict], timeout_s: float, **extra) -> str:
    """The reply text of one chat completion."""
    return reply(url, model, key, messages, timeout_s, **extra).get("content") or ""


def parse(content: str) -> dict | None:
    """The outermost JSON object in a reply (code fences and stray text around it are ignored)."""
    start, end = content.find("{"), content.rfind("}")
    if start < 0 or end < start:
        return None
    try:
        value = json.loads(content[start : end + 1])
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


class Decomposer:
    """`decomposer(state, question)` -> a validated form, or None. `effort` sets the reasoning effort
    (None: thinking off); `blind` writes forms from the question alone, without the evidence; `options`
    go into every request as they are (max_tokens, temperature, seed, ...); `template=False` leaves out
    Qwen's chat_template_kwargs for models that reject it. `ask(messages)` is the cached chat, reusable
    for other prompts."""

    def __init__(
        self,
        url: str = "http://127.0.0.1:8080/v1",
        model: str = "local",
        key_env: str = "LLM_API_KEY",
        effort: str | None = "medium",
        cache: str | None = None,
        timeout_s: float = 900.0,
        blind: bool = False,
        options: dict | None = None,
        template: bool = True,
    ) -> None:
        self.url, self.model, self.key, self.timeout_s = url, model, os.environ.get(key_env, ""), timeout_s
        self.blind = blind
        self.extra: dict = {"reasoning_effort": effort} if effort else {}
        if template:  # Qwen's thinking switch; models whose servers reject it (Mistral) pass template=False
            self.extra["chat_template_kwargs"] = {"enable_thinking": bool(effort)}
        self.extra.update(options or {})
        self.cache = Path(cache) if cache else None
        self.replies: dict[str, str] = {}
        if self.cache and self.cache.exists():
            for line in self.cache.read_text().splitlines():
                entry = json.loads(line)
                self.replies[entry["key"]] = entry["reply"]
        self.lock = threading.Lock()
        self.rejected: dict[str, str] = {}  # question instructions -> why its form was not used

    def ask(self, messages: list[dict]) -> str:
        key = hashlib.sha256(json.dumps([self.model, self.extra, messages]).encode()).hexdigest()
        if key in self.replies:
            return self.replies[key]
        reply = chat(self.url, self.model, self.key, messages, self.timeout_s, **self.extra)
        with self.lock:
            self.replies[key] = reply
            if self.cache:
                self.cache.parent.mkdir(parents=True, exist_ok=True)
                with self.cache.open("a") as out:
                    out.write(json.dumps({"key": key, "reply": reply}) + "\n")
        return reply

    @staticmethod
    def messages(state, question: dict, system: str = SYSTEM, instructions: str = INSTRUCTIONS) -> list[dict]:
        payload = {
            "evidence": state,
            "question": question["instructions"],
            "type": question["type"],
            "options": [text for _, text in decision_options(question)],  # "id: description"
        }
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False) + "\n\n" + instructions},
        ]

    def answer(self, state, question: dict) -> str | None:
        """The LLM's own answer to the question: one of its option ids, or None."""
        reply = self.ask(self.messages(state, question, DECIDE_SYSTEM, DECIDE))
        choice = str((parse(reply) or {}).get("answer")).split(":", 1)[0].strip()  # "id: description" too
        if question["type"] == "noul":
            choice = {"yes": "true", "no": "false"}.get(choice.lower(), choice.lower())
        return choice if choice in [key for key, _ in decision_options(question)] else None

    def __call__(self, state, question: dict) -> dict | None:
        if self.blind:
            messages = self.messages("(not shown)", question, instructions=INSTRUCTIONS + BLIND)
        else:
            messages = self.messages(state, question)
        form = parse(self.ask(messages))
        if form is None or form.get("atomic"):
            return None
        try:
            return validate(form, question)
        except ValueError as error:
            self.rejected[question["instructions"]] = str(error)
            return None
