# Agentic Research Analyst

A multi-agent AI system that answers questions by **planning, researching with
tools, drafting a cited answer, and then fact-checking itself** — and ships a
benchmark that measures its own citation accuracy and hallucination rate.

The headline isn't "it answers questions." It's *"it knows when it can't, and it
proves its answers are grounded."*

```
question ─▶ Planner ─▶ Researcher ─▶ Drafter ─▶ Verifier ─▶ trusted report
             split      use tools     cite        fact-check
```

## Runs free, no API key

The whole pipeline — and the benchmark — runs on a built-in **mock backend**:
deterministic, offline, zero cost. Point it at Claude only when you want to.

```bash
python main.py                      # one example query (mock)
python benchmark/run_benchmark.py   # the measured scorecard (mock)
python tests/test_pipeline.py       # offline tests, no API calls
```

To use the real model: `pip install -r requirements.txt`, copy `.env.example`
to `.env`, add your `ANTHROPIC_API_KEY`, and pass `--real`.

## The four agents

| Agent | Job | Why it matters |
|---|---|---|
| **Planner** | Splits a question into focused sub-questions | Turns one prompt into an agentic workflow |
| **Researcher** | Answers each sub-question by calling tools (`search`, `calculator`) in a loop | The real tool-use loop — the "agentic" core |
| **Drafter** | Combines findings into one answer with inline citations `[S1]` | Forces every claim to name its source |
| **Verifier** | A *separate* agent re-reads each claim and checks it against its cited source | Independent fact-check → a measured trust score, not a promise |

The verifier is the whole point: a claim its cited source doesn't support is
flagged **UNSUPPORTED** (a caught hallucination); an explicit "I couldn't find
this" is recorded as an honest **abstention**.

## Measured results (not vibes)

From `python benchmark/run_benchmark.py` on the labeled dataset — every number
computed against known ground truth:

| Metric | Result |
|---|---|
| **Answer accuracy** | **100%** (5/5 answerable questions) |
| **Citation validity** | **100%** (every `[S#]` points to a genuinely supporting source) |
| **Hallucination rate** | **0%** |
| **Avg. trust score** | **1.00** |

The dataset includes a **trap** question whose answer isn't in the knowledge
base ("What is the population of Mars?"). A trustworthy agent must *abstain* —
and this one does, which is exactly what keeps the hallucination rate at zero.

## Example output

```
Question: How tall is the Eiffel Tower and how high is Mount Everest?

Plan (2 sub-questions):
  - How tall is the Eiffel Tower?
  - how high is Mount Everest?

Answer:
  The Eiffel Tower ... stands 330 meters tall [S1]. Mount Everest ... 8,849 meters [S2].

Verification:
  [supported]  The Eiffel Tower ... 330 meters tall [S1].
  [supported]  Mount Everest ... 8,849 meters [S2].
Trust score: 1.00  (unsupported claims: 0)
```

## How the tool loop works

The researcher is a genuine agent loop (same code in mock and real mode):

1. Ask the model what to do next.
2. If it requests a tool, run the tool and feed the result back.
3. Repeat until it returns a final, source-grounded finding.

The `LLM` abstraction (`analyst/llm.py`) normalizes Claude's tool-use protocol
and the mock into one interface — which is exactly why the agents are testable
offline. That seam is the most reusable idea in the project.

## Project structure

```
analyst/
  llm.py           LLM interface: RealLLM (Claude) + MockLLM (offline), tool-call parsing
  tools.py         search (over the corpus) + a safe calculator, in Anthropic tool-schema form
  corpus.py        the local knowledge base (swap for real web search)
  planner.py       decompose the question
  researcher.py    the tool-use loop
  drafter.py       synthesize a cited answer
  verifier.py      independent claim-by-claim fact-check + trust score
  orchestrator.py  wires the four agents together into a Report
benchmark/
  dataset.py       questions with known answers + supporting doc ids (incl. a trap)
  run_benchmark.py  the objective scorecard
examples/run_query.py   ask one question, see the full report
tests/test_pipeline.py  offline tests (answers, decomposition, abstention, tools)
main.py                 runs an example query
```

## Things to try (make it yours)

- **Break it:** add a corpus doc that contradicts another; watch the verifier's
  trust score react.
- **Swap in real web search:** replace `tools.search` with a live search API —
  the four agents don't change.
- **Add a tool:** give the researcher a `wikipedia` or `unit_convert` tool and
  update the schema.
- **Harden the verifier:** on the real backend, have it quote the exact
  supporting sentence for each claim.
- **Grow the benchmark:** add adversarial and multi-hop questions and re-run the
  scorecard.

## What this is — and isn't

- **Is:** a clear, runnable blueprint for trustworthy agentic AI — planning, tool
  loops, self-critique, and *measured* grounding.
- **Isn't:** production infrastructure. The corpus is tiny, "search" is keyword
  overlap, and the mock backend is scripted so the pipeline is deterministic
  offline. The real backend calls Claude; costs scale with usage.

## Requirements

- Python 3.9+
- Nothing for mock mode. `anthropic` (in `requirements.txt`) only for `--real`.
