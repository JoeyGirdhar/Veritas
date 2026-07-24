"""
Tests that run the whole agent offline on the mock backend -- no API key, no cost.

They prove the two behaviors that matter:
  1. A normal question is answered correctly, with a real citation.
  2. The trap question (answer not in the corpus) is ABSTAINED, not hallucinated.

Run:  python tests/test_pipeline.py     (or: pytest -q)
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analyst import get_llm, analyze
from analyst.tools import search, calculator


def test_answers_and_cites():
    llm = get_llm(force_mock=True)
    report = analyze(llm, "What is the height of Mount Everest?")
    assert "8,849" in report.answer, report.answer
    assert report.sources, "expected at least one cited source"
    assert report.num_unsupported == 0
    print("[answer+cite] OK:", report.answer)


def test_decomposes_multi_part():
    llm = get_llm(force_mock=True)
    report = analyze(llm, "How tall is the Eiffel Tower and how high is Mount Everest?")
    assert len(report.subquestions) == 2, report.subquestions
    assert "330" in report.answer and "8,849" in report.answer, report.answer
    print("[decompose] OK:", report.subquestions)


def test_trap_is_abstained_not_hallucinated():
    llm = get_llm(force_mock=True)
    report = analyze(llm, "What is the population of Mars?")
    assert "could not find" in report.answer.lower(), report.answer
    assert report.num_unsupported == 0, "the agent invented an unsupported claim"
    assert not report.sources, "no source should have been cited"
    print("[trap] OK (abstained):", report.answer)


def test_tools():
    assert search("Mount Everest height")[0]["id"] == "D3"
    assert calculator("330 * 100") == "33000"
    print("[tools] OK")


if __name__ == "__main__":
    test_tools()
    test_answers_and_cites()
    test_decomposes_multi_part()
    test_trap_is_abstained_not_hallucinated()
    print("\nAll pipeline tests passed (mock backend, no API calls).")
