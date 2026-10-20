"""Appointment lookup and an administrative follow-up score."""

import re

import numpy as np

from dataset import APPOINTMENTS
from rules import ESCALATION_PERCENTILE, FOLLOW_UP_WEIGHT, RECENCY_DAYS, RECENCY_WEIGHT


# Combine the follow-up flag and normalized record age without rounding the score.
def escalation_score(follow_up_required: bool, days_since_created: int) -> float:
    return (FOLLOW_UP_WEIGHT * (1 if follow_up_required else 0)
            + RECENCY_WEIGHT * (days_since_created / RECENCY_DAYS))


# Nonnegative weights and these endpoints guarantee the designed score bounds.
if (FOLLOW_UP_WEIGHT < 0 or RECENCY_WEIGHT < 0 or RECENCY_DAYS <= 0
        or escalation_score(False, 0) != 0.0
        or escalation_score(True, RECENCY_DAYS) != 1.0):
    raise ValueError("Escalation rules must have a minimum score of 0.0 and maximum of 1.0.")


# Return recorded appointment fields or a controlled error that preserves the input.
def check_appointment_status(record_id: str) -> dict:
    if not isinstance(record_id, str) or re.fullmatch(r"APT-[0-9]{3}", record_id) is None:
        return {"error": "Enter an appointment ID in the form APT-001.", "record_id": record_id}
    record = next((row for row in APPOINTMENTS if row["record_id"] == record_id), None)
    if record is None:
        return {"error": "No appointment was found for this ID.", "record_id": record_id}
    return {
        "record_id": record["record_id"],
        "status": record["status"],
        "consultation_fee_inr": record["consultation_fee_inr"],
        "follow_up_required": record["follow_up_required"],
        "days_since_created": record["days_since_created"],
        "escalation_score": escalation_score(record["follow_up_required"], record["days_since_created"]),
    }


# Recompute the percentile from the current records using NumPy's default linear method.
def escalation_threshold() -> float:
    scores = [escalation_score(row["follow_up_required"], row["days_since_created"])
              for row in APPOINTMENTS]
    return float(np.percentile(scores, ESCALATION_PERCENTILE, method="linear"))


# Include scores tied at the percentile when recommending administrative escalation.
def recommend_escalation(score: float) -> bool:
    return bool(score >= escalation_threshold())
