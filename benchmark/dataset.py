"""
Benchmark dataset: questions with KNOWN answers, plus the corpus doc(s) that
support each one. Because the ground truth is fixed, we can measure the agent's
citation accuracy and hallucination rate objectively.

Each item:
  question         the input
  expect           substrings that MUST appear in a correct answer ([] = should abstain)
  relevant_docs    corpus ids that actually support the answer ([] = unanswerable)
  trap             True when the answer is deliberately NOT in the corpus
"""

DATASET = [
    {"question": "How tall is the Eiffel Tower and when was it completed?",
     "expect": ["330", "1889"], "relevant_docs": ["D1"], "trap": False},
    {"question": "How long is the Great Wall of China?",
     "expect": ["21,196"], "relevant_docs": ["D2"], "trap": False},
    {"question": "What is the height of Mount Everest?",
     "expect": ["8,849"], "relevant_docs": ["D3"], "trap": False},
    {"question": "What does photosynthesis produce?",
     "expect": ["oxygen"], "relevant_docs": ["D5"], "trap": False},
    {"question": "How tall is the Eiffel Tower and how high is Mount Everest?",
     "expect": ["330", "8,849"], "relevant_docs": ["D1", "D3"], "trap": False},
    # Trap: the corpus has nothing about this. A trustworthy agent abstains.
    {"question": "What is the population of Mars?",
     "expect": [], "relevant_docs": [], "trap": True},
]
