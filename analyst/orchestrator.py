"""Runs the pipeline for one question and returns a Report."""

from dataclasses import dataclass, field

from .corpus import BY_ID
from . import planner, researcher, drafter, verifier


@dataclass
class Report:
    question: str
    subquestions: list
    answer: str
    sources: list = field(default_factory=list)
    claims: list = field(default_factory=list)
    trust_score: float = 1.0
    num_unsupported: int = 0
    num_abstentions: int = 0


def analyze(llm, question: str) -> Report:
    subquestions = planner.plan(llm, question)

    findings = []
    label_by_docid: dict[str, str] = {}
    sources: list[dict] = []

    for sq in subquestions:
        finding = researcher.research(llm, sq)

        # label each doc (S1, S2, ...) the first time it's seen
        label = None
        for doc_id in finding["source_ids"]:
            if doc_id in BY_ID:
                if doc_id not in label_by_docid:
                    label = f"S{len(sources) + 1}"
                    label_by_docid[doc_id] = label
                    doc = BY_ID[doc_id]
                    sources.append({"label": label, "id": doc_id,
                                    "title": doc["title"], "text": doc["text"]})
                else:
                    label = label_by_docid[doc_id]
        findings.append({**finding, "source_label": label})

    answer = drafter.draft(llm, question, findings)

    report_sources = [{"label": s["label"], "text": s["text"]} for s in sources]
    v = verifier.verify(llm, answer, report_sources)

    return Report(
        question=question,
        subquestions=subquestions,
        answer=answer,
        sources=sources,
        claims=v["claims"],
        trust_score=v["trust_score"],
        num_unsupported=v["num_unsupported"],
        num_abstentions=v["num_abstentions"],
    )
