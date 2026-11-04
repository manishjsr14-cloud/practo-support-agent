"""Student-chosen rules for administrative appointment follow-up."""
# What: the four fixed numbers behind the escalation score, kept apart from the code
#   so the rules can be read and explained on their own.
# Task: Part 2, Task 6. Used by: tools.py, run_task6.py and tests/test_task6.py.
#   REFER_BACK_WORDS (Task 8) is used by memory.py; the Task 10 patterns and phrase lists
#   at the end are used by guardrails.py.
# Score = FOLLOW_UP_WEIGHT * (1 if follow-up needed else 0)
#       + RECENCY_WEIGHT * (days_since_created / RECENCY_DAYS); it runs from 0.0 to 1.0.

# A pending follow-up is the stronger signal.
FOLLOW_UP_WEIGHT = 0.6
# An older record adds administrative follow-up urgency.
RECENCY_WEIGHT = 0.4
# Normalize the record's age using the dataset's maximum age in days.
RECENCY_DAYS = 30
# Use the dataset's percentile rather than choosing a fixed score cutoff.
# A "percentile": the 80th percentile is the score that 80% of records are at or below,
#   so roughly the top 20% of records are recommended for escalation.
ESCALATION_PERCENTILE = 80

# Task 8: words that refer back to an appointment mentioned earlier in the same session.
# Only a question with no APT-xxx id of its own AND one of these whole words (any case)
#   reuses the session's most recent id; "What is its fee?" does, while a general policy
#   question such as "What is the cancellation window?" does not trigger a lookup.
REFER_BACK_WORDS = ("it", "its", "this appointment", "that appointment",
                    "the appointment", "my appointment")

# Task 10: a contact number to mask, as KB09 states: any standalone 10-digit number,
#   optionally prefixed +91. Two extra formats are also accepted: one space after +91,
#   and one space or hyphen after the first five digits. The lookarounds make it
#   standalone: no letter or digit may touch it, so ids such as APT-012 (3 digits),
#   dates such as 2026-10-26 and fees such as INR 1200 are never masked.
PHONE_PATTERN = r"(?<![0-9A-Za-z])(?:\+91 ?)?[0-9]{5}[ -]?[0-9]{5}(?![0-9A-Za-z])"
PHONE_MASK = "[PHONE_MASKED]"

# Task 10: phrases that try to override the assistant's instructions. A message containing
#   any of them (any case, extra spaces ignored) is refused before the crew runs.
INJECTION_PHRASES = ("ignore previous instructions", "ignore all previous instructions",
                     "disregard previous instructions", "forget your instructions",
                     "reveal your system prompt", "you are now", "developer mode")

# Task 10, added after the first guardrail run because KB08 requires it: phrases that
#   describe a medical emergency happening now (whole words, any case). A message with
#   any of them gets KB08's emergency direction before any other check.
# Change after the run (logged in the README): "chest pain" was added on its own. The
#   first list left it out so that "Can you diagnose my chest pain?" would go through the
#   normal checks, which meant "I have chest pain" alone was not treated as an emergency.
#   Missing a real emergency is worse than giving an unneeded emergency direction, and
#   KB08 forbids diagnosis but requires the direction, so any chest pain now gets it.
EMERGENCY_PHRASES = ("severe chest pain", "crushing chest pain", "not breathing",
                     "can't breathe", "cannot breathe", "collapsed", "unconscious",
                     "bleeding heavily", "heavy bleeding", "seizure", "heart attack",
                     "stroke", "overdose", "chest pain")

# Emergency detection v2 (Part 3 extension): plain descriptions of emergency types. A
#   message whose meaning (MiniLM embedding) is close enough to one of them gets KB08's
#   emergency direction, even when no phrase above matches. Written before any calibration
#   score was computed and never edited afterwards; none copies a calibration or test
#   sentence. Description 8 was changed before commit to stay clear of a test question.
EMERGENCY_DESCRIPTIONS = (
    "A person has collapsed and does not respond.",
    "Someone is struggling to breathe or cannot breathe at all.",
    "Severe chest pain or pressure, a possible heart attack.",
    "Heavy bleeding that cannot be stopped.",
    "Sudden weakness, numbness or confusion that could be a stroke.",
    "A seizure or convulsions happening right now.",
    "Poisoning or a drug or medicine overdose.",
    "A serious injury from an accident.",
)

# Part 4, Task 15: the per-request token budget (tokens = characters / 4, counted across the
#   masked input and every LLM call, as pre-registered in part4_preregistration.py). Measured
#   by run_part4_governance.py over the 15 Part 3 questions (outputs/part4_governance.txt):
#   the largest request used 5026 tokens and no request made more than 6 LLM calls.
BUDGET_LARGEST_TOKENS = 5026
BUDGET_MAX_CALLS = 6
# The cap: the pre-registered margin of 2 x the largest measured request.
BUDGET_TOKENS = 10052
