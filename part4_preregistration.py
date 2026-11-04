"""Pre-registered Part 4 test cases and budget method, fixed before any Part 4 code."""
# What: the review stage's test drafts with their expected verdicts, the budget method and
#   margin, and the oversized test request.
# Task: Part 4, Tasks 14-15. Used by: the Part 4 runners and tests.
# This file was committed on its own before any Part 4 code and is never edited.

# Task 14: the sentence deliberately added to a real draft. No policy says this.
INJECTED_SENTENCE = "Cancellations are always free at any time."

# Task 14: each draft is the crew's real draft for the question; with "inject" set, that
#   sentence is added to the end of the draft before review. "expect" is the verdict.
REVIEW_CASES = [
    {"id": "r1", "question": "What is the cancellation and rescheduling window?",
     "inject": None, "expect": "approved unchanged"},
    {"id": "r2", "question": "What is the cancellation and rescheduling window?",
     "inject": INJECTED_SENTENCE, "expect": "revised"},
    # A record and the KB08 note are in this draft; neither may be flagged.
    {"id": "r3", "question": "What is the status of APT-007 and can I reschedule it?",
     "inject": None, "expect": "approved unchanged"},
]

# Task 15 budget method: tokens are estimated as ceil(characters / 4).
CHARS_PER_TOKEN = 4
# What one request costs: the masked input once, plus every LLM call's prompt messages
#   and reply (the crew's MockLLM calls and the review stage's model client calls).
# Basis: the 15 questions in eval_queries.py, run through pipeline.ask with the review stage.
# Cap: BUDGET_MARGIN x the largest measured request.
BUDGET_MARGIN = 2
# Check before the crew runs: projected = input tokens x the most LLM calls measured in one
#   request + the largest measured request; a request is refused if projected > cap.
#   A meter on every LLM call also stops a request the moment its running total passes the cap.

# Task 15: the oversized test request, exactly 20,000 characters of an ordinary policy question
#   (no emergency or injection phrase). It must be refused before the crew runs.
OVERSIZED_REQUEST = ("Please explain the cancellation and rescheduling window. " * 400)[:20000]
