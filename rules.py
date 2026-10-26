"""Student-chosen rules for administrative appointment follow-up."""
# What: the four fixed numbers behind the escalation score, kept apart from the code
#   so the rules can be read and explained on their own.
# Task: Part 2, Task 6. Used by: tools.py, run_task6.py and tests/test_task6.py.
#   REFER_BACK_WORDS (Task 8, at the end) is used by memory.py.
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
