"""
A tiny local knowledge base.

Using a fixed corpus (instead of live web search) is a deliberate choice for a
learning + benchmarking project: because we know exactly what facts exist and
where, we can measure the agent's citation accuracy and hallucination rate
against ground truth. Swap this out for a real web-search tool and the rest of
the pipeline is unchanged.

Each document has a stable id, a title, and a one-fact body.
"""

DOCUMENTS = [
    {"id": "D1", "title": "Eiffel Tower",
     "text": "The Eiffel Tower, located in Paris, France, was completed in 1889 and stands 330 meters tall."},
    {"id": "D2", "title": "Great Wall of China",
     "text": "The Great Wall of China is approximately 21,196 kilometers long."},
    {"id": "D3", "title": "Mount Everest",
     "text": "Mount Everest is the highest mountain above sea level, at 8,849 meters."},
    {"id": "D4", "title": "Amazon River",
     "text": "The Amazon River discharges about 209,000 cubic meters of water per second, the largest of any river."},
    {"id": "D5", "title": "Photosynthesis",
     "text": "Photosynthesis converts carbon dioxide and water into glucose and oxygen using sunlight."},
    {"id": "D6", "title": "Speed of light",
     "text": "The speed of light in a vacuum is approximately 299,792 kilometers per second."},
    {"id": "D7", "title": "Boiling point of water",
     "text": "Water boils at 100 degrees Celsius at standard atmospheric pressure."},
    {"id": "D8", "title": "Human heart",
     "text": "The human heart beats about 100,000 times per day."},
]

BY_ID = {d["id"]: d for d in DOCUMENTS}
