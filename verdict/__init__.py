"""Verdict — typed decisions with abstention, over any causal LM.

Give it a state (text, email, ticket, JSON) and typed questions. It returns
typed answers with calibrated probabilities in ONE forward pass — no text
generation, nothing to parse, nothing to hallucinate.

    pip install verdict

The primitive nobody else ships: ABSTENTION. Every other decision model
answers. Verdict is the one that knows when not to. If the calibrated
confidence for a question falls below a threshold fit on your data, it
returns ABSTAIN instead of a confident wrong answer.
"""
from .core import Verdict

__version__ = "0.1.0"
__all__ = ["Verdict"]
