"""
Planner agent: break a question into answerable sub-questions.

Decomposition is what turns a single prompt into an *agentic* workflow — each
sub-question gets its own research pass, and the drafter recombines the findings.
"""

from .llm import extract_json

PLANNER_SYSTEM = (
    "You are a research planner. Break the user's question into 1-3 focused, "
    "independently answerable sub-questions. Respond ONLY with JSON of the form "
    '{"subquestions": ["...", "..."]}. If the question is already atomic, return it as a single element.'
)


def plan(llm, question: str) -> list[str]:
    result = llm.complete(
        role="planner",
        system=PLANNER_SYSTEM,
        messages=[{"role": "user", "content": question}],
        max_tokens=512,
    )
    data = extract_json(result.text)
    subs = data.get("subquestions") or [question]
    return [s for s in subs if isinstance(s, str) and s.strip()] or [question]
