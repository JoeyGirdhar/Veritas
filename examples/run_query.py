"""
Ask the agent one question and print the full, verified report.

    python examples/run_query.py
    python examples/run_query.py "What does photosynthesis produce?"
    python examples/run_query.py --real "How tall is the Eiffel Tower?"

Mock backend by default (free/offline). Pass --real to use Claude.
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analyst import get_llm, analyze


def main():
    args = [a for a in sys.argv[1:] if a != "--real"]
    force_mock = "--real" not in sys.argv
    question = args[0] if args else "How tall is the Eiffel Tower and how high is Mount Everest?"

    llm = get_llm(force_mock=force_mock)
    report = analyze(llm, question)

    print(f"\nQuestion: {report.question}")
    print(f"Backend:  {'mock' if force_mock else 'Claude'}")
    print(f"\nPlan ({len(report.subquestions)} sub-question(s)):")
    for sq in report.subquestions:
        print(f"  - {sq}")

    print("\nAnswer:")
    print(f"  {report.answer}")

    print("\nSources:")
    for s in report.sources:
        print(f"  [{s['label']}] {s['title']} ({s['id']}): {s['text']}")
    if not report.sources:
        print("  (none)")

    print("\nVerification:")
    for c in report.claims:
        if c.get("abstention"):
            tag = "abstained"
        elif c.get("supported"):
            tag = "supported"
        else:
            tag = "UNSUPPORTED"
        print(f"  [{tag}] {c['text']}")
    print(f"\nTrust score: {report.trust_score:.2f}  "
          f"(unsupported claims: {report.num_unsupported})")


if __name__ == "__main__":
    main()
