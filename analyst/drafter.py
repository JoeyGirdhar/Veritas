"""Drafter: writes a cited answer from the research findings."""

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
