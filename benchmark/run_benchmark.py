"""
Run the agent over the labeled dataset and print a scorecard.

Metrics (all objective, computed against known ground truth):

  Answer accuracy      correct answers / answerable questions
                       (a trap counts correct only if the agent ABSTAINS)
  Citation validity    fraction of [S#] citations that point to a real retrieved
                       source AND to a doc that genuinely supports the question
  Hallucination rate   fraction of questions where the agent stated an
                       unsupported factual claim (the number you want near zero)
  Avg. trust score     the verifier's own self-reported grounding, averaged

Runs on the mock backend by default -> free, offline, deterministic. Pass
--real to use Claude (needs ANTHROPIC_API_KEY).

    python benchmark/run_benchmark.py
    python benchmark/run_benchmark.py --real
"""

import os
import re
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from analyst import get_llm, analyze
from benchmark.dataset import DATASET


def _abstained(answer: str) -> bool:
    return "could not find" in answer.lower()


def main():
    force_mock = "--real" not in sys.argv
    llm = get_llm(force_mock=force_mock)
    print(f"Backend: {'MOCK (free/offline)' if force_mock else 'Claude (real API)'}\n")

    n_answerable = 0
    n_correct = 0
    citations_total = 0
    citations_valid = 0
    n_hallucinated = 0
    trust_scores = []

    for item in DATASET:
        report = analyze(llm, item["question"])
        answer = report.answer
        trust_scores.append(report.trust_score)

        # --- Answer accuracy ---
        if item["trap"]:
            correct = _abstained(answer) and report.num_unsupported == 0
        else:
            n_answerable += 1
            correct = all(sub in answer for sub in item["expect"])
            if correct:
                n_correct += 1

        # --- Citation validity: does each [S#] map to a doc that truly supports this Q? ---
        cited_labels = set(re.findall(r"\[(S\d+)\]", answer))
        label_to_docid = {s["label"]: s["id"] for s in report.sources}
        for label in cited_labels:
            citations_total += 1
            doc_id = label_to_docid.get(label)
            if doc_id and doc_id in item["relevant_docs"]:
                citations_valid += 1

        # --- Hallucination: any unsupported factual claim the verifier flagged ---
        if report.num_unsupported > 0:
            n_hallucinated += 1

        mark = "OK " if correct else "XX "
        print(f"  {mark} trust={report.trust_score:.2f}  {item['question'][:52]}")

    total = len(DATASET)
    print("\n" + "=" * 60)
    print(f"Answer accuracy    : {n_correct}/{n_answerable} answerable "
          f"({(n_correct / n_answerable * 100):.0f}%)")
    print(f"Citation validity  : {citations_valid}/{citations_total} "
          f"({(citations_valid / citations_total * 100):.0f}%)"
          if citations_total else "Citation validity  : n/a")
    print(f"Hallucination rate : {n_hallucinated}/{total} "
          f"({(n_hallucinated / total * 100):.0f}%)  (lower is better)")
    print(f"Avg. trust score   : {sum(trust_scores) / total:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
