"""Calibration questions for the groundedness threshold T, kept apart from the test sets."""
# What: near-topic questions the KB cannot support, and paraphrased questions it can.
#   Their weakest-term support scores set T; the test sets in guardrail_queries.py are
#   never used for that.
# Task: Part 2, Task 10. Used by: guardrails.py.
# This file was committed on its own before any guardrail test set was run.

# Clinic-sounding questions with no supporting policy text.
CALIBRATE_REFUSE = [
    "Does the clinic have a pharmacy on site?",
    "Can I pay the fee with cryptocurrency?",
    "Is there free Wi-Fi in the waiting room?",
]

# Paraphrases of questions the KB does answer.
CALIBRATE_ANSWER = [
    "Is a video consultation possible?",
    "How soon are blood results ready?",
    "Do I get money back if I cancel early?",
]
