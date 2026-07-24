"""
Agentic research analyst: a multi-agent pipeline (planner -> researcher ->
drafter -> verifier) that answers questions with cited, fact-checked output.
"""

from .llm import get_llm
from .orchestrator import analyze, Report

__all__ = ["get_llm", "analyze", "Report"]
