"""Input guardrails: phone-number masking and prompt-injection blocking."""
# What: masks Indian mobile numbers before text reaches the crew or history (KB09), and
#   refuses messages that try to override the assistant's instructions.
# Task: Part 2, Task 10. Used by: memory.py (masking), pipeline.py (blocking) and tests.
# The patterns and phrases themselves live in rules.py, so they can be read on their own.

import re

from rules import INJECTION_PHRASES, PHONE_MASK, PHONE_PATTERN
from schema import validate_response

PHONE = re.compile(PHONE_PATTERN)


# Replace every standalone Indian mobile number with [PHONE_MASKED].
# Takes: the user's text. Returns: the text with numbers masked and nothing else changed.
def mask_phone_numbers(text):
    return PHONE.sub(PHONE_MASK, text)


# Find the injection phrases a message contains.
# Takes: the user's text. Returns: the matching phrases from rules.py, in list order.
# Why: lower-casing and collapsing spaces means "Ignore  PREVIOUS instructions" still matches.
def injection_phrases(text):
    normalized = " ".join(text.lower().split())
    return [phrase for phrase in INJECTION_PHRASES if phrase in normalized]


# Build the validated refusal used when an input guardrail blocks a message.
# Takes: the reason to report. Returns: a CrewResponse with refused=True and no tool data.
def refusal(reason):
    return validate_response({
        "answer": "I can't help with that request. I can explain clinic policies or look up "
                  "an appointment by its id, such as APT-001.",
        "sources": [], "tools_used": [], "record_id": None, "escalation_recommended": None,
        "grounded": False, "refused": True, "refusal_reason": reason})
