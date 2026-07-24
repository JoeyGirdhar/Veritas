"""
Tools the researcher agent can call.

Two tools, defined in the Anthropic tool-schema format so the real backend can
hand them to Claude directly:

    search      keyword search over the local corpus (returns ranked documents)
    calculator  safe arithmetic on a numeric expression

`dispatch()` executes a tool call and returns a STRING (the tool result the model
reads back). Keeping tool results as strings mirrors how the real API feeds
`tool_result` content back into the conversation.
"""

import ast
import json
import operator
import re

from .corpus import DOCUMENTS

# ---- Tool schemas (what the model sees) -----------------------------------

TOOL_SCHEMAS = [
    {
        "name": "search",
        "description": "Search the knowledge base for documents relevant to a query. "
                       "Returns ranked documents with their id, title, and text.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to look up"}},
            "required": ["query"],
        },
    },
    {
        "name": "calculator",
        "description": "Evaluate a basic arithmetic expression (e.g. '330 * 100').",
        "input_schema": {
            "type": "object",
            "properties": {"expression": {"type": "string", "description": "Arithmetic to evaluate"}},
            "required": ["expression"],
        },
    },
]

_STOP = {"the", "a", "an", "of", "is", "was", "are", "to", "in", "and", "how", "what",
         "when", "where", "does", "do", "at", "about", "its", "it", "that", "this", "many"}


def _tokens(text: str):
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in _STOP and len(w) > 2}


def search(query: str, k: int = 3) -> list[dict]:
    """Rank documents by keyword overlap with the query. Returns top-k with scores."""
    q = _tokens(query)
    scored = []
    for doc in DOCUMENTS:
        score = len(q & _tokens(doc["title"] + " " + doc["text"]))
        scored.append({"id": doc["id"], "title": doc["title"], "text": doc["text"], "score": score})
    scored.sort(key=lambda d: d["score"], reverse=True)
    return scored[:k]


# ---- Safe calculator (no eval of arbitrary code) ---------------------------

_OPS = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg,
    ast.Mod: operator.mod,
}


def _eval(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.left), _eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_eval(node.operand))
    raise ValueError("unsupported expression")


def calculator(expression: str) -> str:
    try:
        result = _eval(ast.parse(expression, mode="eval").body)
        return str(result)
    except Exception:
        return "error: could not evaluate expression"


def dispatch(name: str, tool_input: dict) -> str:
    """Run a tool call, returning a string result for the model to read."""
    if name == "search":
        return json.dumps(search(tool_input.get("query", "")))
    if name == "calculator":
        return calculator(tool_input.get("expression", ""))
    return f"error: unknown tool {name}"
