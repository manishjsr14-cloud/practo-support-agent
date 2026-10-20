"""Student-chosen rules for administrative appointment follow-up."""
# What: the four fixed numbers behind the escalation score, kept apart from the code
#   so the rules can be read and explained on their own.
# Task: Part 2, Task 6. Used by: tools.py, run_task6.py and tests/test_task6.py.
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
