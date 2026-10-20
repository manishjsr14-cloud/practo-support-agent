"""Student-chosen rules for administrative appointment follow-up."""

# A pending follow-up is the stronger signal.
FOLLOW_UP_WEIGHT = 0.6
# An older record adds administrative follow-up urgency.
RECENCY_WEIGHT = 0.4
# Normalize the record's age using the dataset's maximum age in days.
RECENCY_DAYS = 30
# Use the dataset's percentile rather than choosing a fixed score cutoff.
ESCALATION_PERCENTILE = 80
