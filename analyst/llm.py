"""LLM backends.

RealLLM calls Claude through the Anthropic SDK. MockLLM returns scripted
responses so the pipeline and tests run offline without an API key."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any

from .tools import TOOL_SCHEMAS

# override with AGENT_MODEL, e.g. claude-haiku-4-5 for cheaper runs
DEFAULT_MODEL = os.environ.get("AGENT_MODEL", "claude-opus-5")

# min keyword overlap for the mock to count a search hit as a source
MIN_SOURCE_SCORE = 1


@dataclass
class ToolCall:
    id: str
    name: str
    input: dict


@dataclass
class LLMResult:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    # assistant turn to append when continuing a tool loop
    assistant_content: Any = None
    stop_reason: str = "end_turn"


def extract_json(text: str) -> dict:
    """Pull the first JSON object out of a model response (tolerant of prose/fences)."""
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?|\n?```$", "", text).strip()
    start = text.find("{")
    if start == -1:
        return {}
    # match braces so trailing text doesn't break parsing
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except json.JSONDecodeError:
                    return {}
    return {}


# Real backend

class RealLLM:
    def __init__(self, model: str = DEFAULT_MODEL):
        import anthropic
        self.client = anthropic.Anthropic()
        self.model = model

    def complete(self, role, system, messages, tools=None, max_tokens=2048):
        kwargs = dict(model=self.model, max_tokens=max_tokens, system=system, messages=messages)
        if tools:
            kwargs["tools"] = tools
        resp = self.client.messages.create(**kwargs)

        text_parts, tool_calls = [], []
        for block in resp.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(id=block.id, name=block.name, input=dict(block.input)))
        return LLMResult(
            text="".join(text_parts),
            tool_calls=tool_calls,
            assistant_content=resp.content,
            stop_reason=resp.stop_reason or "end_turn",
        )


# Mock backend

def _last_user_text(messages) -> str:
    for m in reversed(messages):
        if m["role"] == "user":
            c = m["content"]
            if isinstance(c, str):
                return c
            for block in c:
                if isinstance(block, dict) and block.get("type") == "text":
                    return block["text"]
    return ""


def _find_tool_result(messages):
    """Return the string content of the most recent tool_result, or None."""
    for m in reversed(messages):
        if m["role"] == "user" and isinstance(m["content"], list):
            for block in m["content"]:
                if isinstance(block, dict) and block.get("type") == "tool_result":
                    return block["content"]
    return None


def _decompose(question: str) -> list[str]:
    """A deliberately simple decomposition: split on ' and ' when it joins clauses."""
    q = question.strip()
    parts = re.split(r"\band\b", q, flags=re.IGNORECASE)
    parts = [p.strip(" ?.") for p in parts if len(p.strip(" ?.")) > 3]
    if len(parts) <= 1:
        return [q]
    return [p if p.endswith("?") else p + "?" for p in parts]


class MockLLM:
    """Scripted responses keyed by agent role. No network, fully deterministic."""

    model = "mock"

    def complete(self, role, system, messages, tools=None, max_tokens=2048):
        if role == "planner":
            subs = _decompose(_last_user_text(messages))
            return LLMResult(text=json.dumps({"subquestions": subs}))

        if role == "researcher":
            tool_result = _find_tool_result(messages)
            if tool_result is None:
                # first turn: search
                query = _last_user_text(messages)
                call = ToolCall(id="call_1", name="search", input={"query": query})
                assistant_content = [{
                    "type": "tool_use", "id": call.id, "name": call.name, "input": call.input,
                }]
                return LLMResult(tool_calls=[call], assistant_content=assistant_content,
                                 stop_reason="tool_use")
            # second turn: report a finding from the top doc
            docs = json.loads(tool_result)
            if not docs or docs[0]["score"] < MIN_SOURCE_SCORE:
                return LLMResult(text=json.dumps(
                    {"finding": "No supporting source found.", "source_ids": []}))
            top = docs[0]
            return LLMResult(text=json.dumps(
                {"finding": top["text"], "source_ids": [top["id"]]}))

        if role == "drafter":
            payload = extract_json(_last_user_text(messages))
            sentences = []
            for f in payload.get("findings", []):
                label = f.get("source_label")
                if label and f.get("finding") and "No supporting source" not in f["finding"]:
                    sentences.append(f"{f['finding'].rstrip('.')} [{label}].")
                else:
                    sentences.append(
                        f"I could not find reliable information about: {f.get('subquestion', '').rstrip('?')}.")
            return LLMResult(text=json.dumps({"answer": " ".join(sentences)}))

        if role == "verifier":
            payload = extract_json(_last_user_text(messages))
            answer = payload.get("answer", "")
            sources = {s["label"]: s["text"] for s in payload.get("sources", [])}
            claims = []
            for sentence in re.split(r"(?<=[.!?])\s+", answer.strip()):
                if not sentence:
                    continue
                m = re.search(r"\[(S\d+)\]", sentence)
                if m:
                    label = m.group(1)
                    supported = _overlap(sentence, sources.get(label, "")) >= 1
                    claims.append({"text": sentence, "label": label, "supported": supported})
                elif "could not find" in sentence.lower():
                    claims.append({"text": sentence, "label": None, "supported": True,
                                   "abstention": True})
                else:
                    claims.append({"text": sentence, "label": None, "supported": False})
            return LLMResult(text=json.dumps({"claims": claims}))

        return LLMResult(text="{}")


_STOP = {"the", "a", "an", "of", "is", "was", "are", "to", "in", "and", "how", "what",
         "when", "where", "does", "do", "at", "about", "its", "it", "that", "this"}


def _overlap(a: str, b: str) -> int:
    wa = {w for w in re.findall(r"[a-z0-9,]+", a.lower()) if w not in _STOP and len(w) > 2}
    wb = {w for w in re.findall(r"[a-z0-9,]+", b.lower()) if w not in _STOP and len(w) > 2}
    return len(wa & wb)


def get_llm(force_mock: bool = False):
    """Pick a backend. Mock unless a key is present and mock isn't forced."""
    use_mock = force_mock or os.environ.get("USE_MOCK") == "1" or not os.environ.get("ANTHROPIC_API_KEY")
    return MockLLM() if use_mock else RealLLM()
