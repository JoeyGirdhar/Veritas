"""
Drafter agent: combine the per-sub-question findings into one answer with
inline citations like [S1], [S2].

The drafter is told to cite the source label attached to each finding and to
abstain (rather than invent) when a finding has no supporting source. That
"abstain instead of guess" instruction is what the verifier later checks.
"""

import json

from .llm import extract_json

DRAFTER_SYSTEM = (
    "You are a careful writer. Using ONLY the provided findings, write a concise answer "
    "to the user's question. Cite the source label after each claim, e.g. 'X is Y [S1].' "
    "If a finding says no source was found, explicitly say you could not find reliable "
    "information for that part instead of guessing. Respond ONLY with JSON: {\"answer\": \"...\"}."
)


def draft(llm, question: str, findings: list[dict]) -> str:
    payload = {"question": question, "findings": findings}
    result = llm.complete(
        role="drafter",
        system=DRAFTER_SYSTEM,
        messages=[{"role": "user", "content": json.dumps(payload)}],
        max_tokens=1024,
    )
    return extract_json(result.text).get("answer", "").strip()
