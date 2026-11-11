"""Researcher: answers one sub-question with a tool-use loop."""

from .llm import extract_json
from .tools import TOOL_SCHEMAS, dispatch

RESEARCHER_SYSTEM = (
    "You are a meticulous researcher. Use the `search` tool to find evidence before "
    "answering, and `calculator` for any arithmetic. Ground every claim in retrieved "
    "documents. When done, respond ONLY with JSON: "
    '{"finding": "<one factual sentence>", "source_ids": ["D1", ...]}. '
    "If no document supports an answer, return finding \"No supporting source found.\" and an empty source_ids."
)

MAX_STEPS = 4


def research(llm, subquestion: str) -> dict:
    """Return {'subquestion', 'finding', 'source_ids'} for one sub-question."""
    messages = [{"role": "user", "content": subquestion}]

    for _ in range(MAX_STEPS):
        result = llm.complete(
            role="researcher",
            system=RESEARCHER_SYSTEM,
            messages=messages,
            tools=TOOL_SCHEMAS,
            max_tokens=1024,
        )

        if result.tool_calls:
            messages.append({"role": "assistant", "content": result.assistant_content})
            tool_results = []
            for call in result.tool_calls:
                output = dispatch(call.name, call.input)
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": call.id,
                    "content": output,
                })
            messages.append({"role": "user", "content": tool_results})
            continue

        data = extract_json(result.text)
        return {
            "subquestion": subquestion,
            "finding": data.get("finding", "No supporting source found."),
            "source_ids": data.get("source_ids", []) or [],
        }

    return {"subquestion": subquestion, "finding": "No supporting source found.", "source_ids": []}
