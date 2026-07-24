"""
Entry point: run one example query through the full agent (mock backend, free).

    python main.py

Then explore:
    python examples/run_query.py "What does photosynthesis produce?"
    python benchmark/run_benchmark.py          # the measured scorecard
    python tests/test_pipeline.py              # offline tests

To use the real Claude model, set ANTHROPIC_API_KEY (see .env.example) and add --real.
"""

from examples.run_query import main

if __name__ == "__main__":
    main()
