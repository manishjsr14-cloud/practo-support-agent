"""Appointment lookup and an administrative follow-up score."""
# What: looks up one appointment by its id, computes its escalation score, and decides
#   whether it should be flagged for administrative follow-up.
# Task: Part 2, Task 6; rag_lookup() was added in Task 7. Used by: run_task6.py,
#   tests/test_task6.py and crew.py (which wraps these functions as CrewAI tools).
# The numbers it uses come from rules.py; the appointment records come from dataset.py.

import re

import numpy as np

from dataset import APPOINTMENTS
from rules import ESCALATION_PERCENTILE, FOLLOW_UP_WEIGHT, RECENCY_DAYS, RECENCY_WEIGHT


# Combine the follow-up flag and normalized record age without rounding the score.
# Takes: whether a follow-up is required, and the record's age in days.
# Returns: a float from 0.0 to 1.0. Why: one number lets records be ranked.
def escalation_score(follow_up_required: bool, days_since_created: int) -> float:
    return (FOLLOW_UP_WEIGHT * (1 if follow_up_required else 0)
            + RECENCY_WEIGHT * (days_since_created / RECENCY_DAYS))


# Nonnegative weights and these endpoints guarantee the designed score bounds.
# This check runs once when the module is imported: if someone edits rules.py so that
# the lowest possible score is not 0.0 or the highest is not 1.0, importing fails at once.
if (FOLLOW_UP_WEIGHT < 0 or RECENCY_WEIGHT < 0 or RECENCY_DAYS <= 0
        or escalation_score(False, 0) != 0.0
        or escalation_score(True, RECENCY_DAYS) != 1.0):
    raise ValueError("Escalation rules must have a minimum score of 0.0 and maximum of 1.0.")


# Return recorded appointment fields or a controlled error that preserves the input.
# Takes: an id such as "APT-001". Returns: the record's fields plus its escalation score,
#   or {"error", "record_id"} for a malformed or unknown id.
# Why: answers come from the stored records, never invented; bad input gets a clear error.
def check_appointment_status(record_id: str) -> dict:
    # Accept only a string of exactly "APT-" plus three ASCII digits. [0-9] is used
    # instead of \d because \d would also accept non-ASCII digits such as "٠".
    if not isinstance(record_id, str) or re.fullmatch(r"APT-[0-9]{3}", record_id) is None:
        return {"error": "Enter an appointment ID in the form APT-001.", "record_id": record_id}
    # Find the first record with this id, or None if there is no such record.
    record = next((row for row in APPOINTMENTS if row["record_id"] == record_id), None)
    if record is None:
        return {"error": "No appointment was found for this ID.", "record_id": record_id}
    # Return a new dict (category is not included), so callers cannot change the dataset.
    return {
        "record_id": record["record_id"],
        "status": record["status"],
        "consultation_fee_inr": record["consultation_fee_inr"],
        "follow_up_required": record["follow_up_required"],
        "days_since_created": record["days_since_created"],
        "escalation_score": escalation_score(record["follow_up_required"], record["days_since_created"]),
    }


# Recompute the percentile from the current records using NumPy's default linear method.
# Takes: nothing. Returns: the 80th-percentile score of all records as a float.
# "Linear interpolation": when the 80% point falls between two sorted scores, NumPy
#   takes the value that far along the straight line between them.
# Why: the cut-off comes from the data and is recomputed on every call, not hard-coded.
def escalation_threshold() -> float:
    scores = [escalation_score(row["follow_up_required"], row["days_since_created"])
              for row in APPOINTMENTS]
    return float(np.percentile(scores, ESCALATION_PERCENTILE, method="linear"))


# Include scores tied at the percentile when recommending administrative escalation.
# Takes: a score. Returns: True if it is at or above the threshold (">=" includes ties).
# bool() turns a NumPy boolean into a plain Python True/False.
def recommend_escalation(score: float) -> bool:
    return bool(score >= escalation_threshold())


# Answer a policy question with Part 1's retrieval (Task 7).
# Takes: a question. Returns: the quoted answer text (or the fallback), the source parent
#   ids and the top-1 cosine similarity.
# Why: it reuses rag.answer() unchanged, with the sentence collection and the threshold
#   saved in outputs/calibration.json, so the crew sees exactly what Part 1 retrieves.
def rag_lookup(query: str) -> dict:
    # Imported here so that Task 6 code importing tools.py never loads the retrieval code.
    import rag
    result = rag.answer(query, strategy="sentence")
    return {"answer": result["answer"], "source_ids": result["source_ids"],
            "similarity": result["similarity"]}
