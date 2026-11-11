"""Verifier: checks each claim in the answer against its cited source."""

import json

from .llm import extract_json

VERIFIER_SYSTEM = (
    "You are a fact-checker. For each sentence in the answer, decide whether the source "
    "it cites actually supports it. A sentence that explicitly says information could not "
    "be found is an honest abstention (supported=true, abstention=true). A factual sentence "
    "with no citation, or whose cited source does not support it, is supported=false. "
    "Respond ONLY with JSON: {\"claims\": [{\"text\": \"...\", \"label\": \"S1\"|null, "
    "\"supported\": true|false, \"abstention\": true|false}]}."
)


def verify(llm, answer: str, sources: list[dict]) -> dict:
    """sources: [{'label': 'S1', 'text': ...}]. Returns claims + trust metrics."""
    payload = {"answer": answer, "sources": sources}
    result = llm.complete(
        role="verifier",
        system=VERIFIER_SYSTEM,
        messages=[{"role": "user", "content": json.dumps(payload)}],
        max_tokens=1024,
    )
    claims = extract_json(result.text).get("claims", [])

    factual = [c for c in claims if not c.get("abstention")]
    supported = [c for c in factual if c.get("supported")]
    unsupported = [c for c in factual if not c.get("supported")]

    trust = (len(supported) / len(factual)) if factual else 1.0
    return {
        "claims": claims,
        "num_claims": len(claims),
        "num_supported": len(supported),
        "num_unsupported": len(unsupported),
        "num_abstentions": sum(1 for c in claims if c.get("abstention")),
        "trust_score": round(trust, 3),
    }
