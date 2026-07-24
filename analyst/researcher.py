"""
Researcher agent: answer one sub-question by USING TOOLS.

This is the core agentic loop — the "tool loop" that makes this project more than
a chatbot:

    1. ask the model what to do
    2. if it wants a tool, run the tool and feed the result back
    3. repeat until it produces a final, grounded finding

The loop is provider-agnostic: `llm.complete()` normalizes both the real Claude
tool-use protocol and the mock into the same shape, so this exact code runs in
both modes.
"""

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
            # Append the assistant's tool request, then run each tool and return results.
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

        # No tool call -> the model produced its final finding.
        data = extract_json(result.text)
        return {
            "subquestion": subquestion,
            "finding": data.get("finding", "No supporting source found."),
            "source_ids": data.get("source_ids", []) or [],
        }

    # Ran out of steps without a final answer.
    return {"subquestion": subquestion, "finding": "No supporting source found.", "source_ids": []}
