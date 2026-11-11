"""Benchmark questions with known answers and supporting doc ids."""

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
    # not in the corpus, so the agent should abstain
    {"question": "What is the population of Mars?",
     "expect": [], "relevant_docs": [], "trap": True},
]
