"""Print and save measured administrative escalation results without changing appointments."""
# What: computes the escalation score for all 50 records, reports their statistics and
#   threshold, shows example lookups, and saves it all to outputs/part2_escalation.txt.
# Task: Part 2, Task 6. Used by: run by hand; no other file imports it.
# It uses rules.py (the numbers), tools.py (the score and lookup) and dataset.py (records).

import json
from pathlib import Path

import numpy as np

from dataset import APPOINTMENTS
from rules import ESCALATION_PERCENTILE, FOLLOW_UP_WEIGHT, RECENCY_DAYS, RECENCY_WEIGHT
from tools import check_appointment_status, escalation_score, escalation_threshold


# Calculate the distribution, counts and deterministic example lookups from the records.
# Takes: nothing. Returns: the full report as one text string.
# Why: every number in the README for Task 6 must come from this real output.
def make_report() -> str:
    # Score every record and count how many reach the data-derived threshold.
    scores = np.array([escalation_score(row["follow_up_required"], row["days_since_created"])
                       for row in APPOINTMENTS])
    threshold = escalation_threshold()
    above = int(np.count_nonzero(scores >= threshold))
    # Split records by follow-up flag, and count follow-up records that reach the threshold
    # (True counts as 1 in sum()).
    follow_up = [row for row in APPOINTMENTS if row["follow_up_required"]]
    without_follow_up = [row for row in APPOINTMENTS if not row["follow_up_required"]]
    follow_up_above = sum(escalation_score(row["follow_up_required"], row["days_since_created"])
                          >= threshold for row in follow_up)
    # Build the report: formula, settings, score bounds, a statistics table and counts.
    # {...!r} prints the full-precision value of a float instead of a rounded one.
    lines = [
        "Task 6: administrative appointment escalation",
        f"Formula: escalation_score = {FOLLOW_UP_WEIGHT} * (1 if follow_up_required else 0) "
        f"+ {RECENCY_WEIGHT} * (days_since_created / {RECENCY_DAYS})",
        f"FOLLOW_UP_WEIGHT={FOLLOW_UP_WEIGHT}; RECENCY_WEIGHT={RECENCY_WEIGHT}; "
        f"RECENCY_DAYS={RECENCY_DAYS}; ESCALATION_PERCENTILE={ESCALATION_PERCENTILE}",
        "Percentile method: numpy.percentile(method='linear'); NumPy default linear interpolation.",
        f"Designed score bounds: minimum={escalation_score(False, 0)!r}; "
        f"maximum={escalation_score(True, RECENCY_DAYS)!r}",
        f"Records: {len(APPOINTMENTS)}", "", "| Statistic | Escalation score |", "|---|---:|",
        f"| Minimum | {float(np.min(scores)):.6f} |",
        f"| Median | {float(np.median(scores)):.6f} |",
        f"| {ESCALATION_PERCENTILE}th percentile | {threshold:.6f} |",
        f"| Maximum | {float(np.max(scores)):.6f} |",
        f"| Mean | {float(np.mean(scores)):.6f} |", "",
        f"Threshold (full precision): {threshold!r}",
        "Recommendation rule: escalation_score >= escalation_threshold().",
        f"Records at or above threshold: {above}/{len(APPOINTMENTS)}",
        f"Follow-up records: {len(follow_up)}; at or above threshold: {follow_up_above}/{len(follow_up)}",
        "This is an administrative follow-up flag, not a medical assessment (KB08). "
        "Never use it to judge medical urgency.", "", "Example lookups:",
    ]
    # Examples: the first two records with a follow-up, the first two without, and one
    # unknown id (APT-999) to show the error reply. Same records every run.
    examples = follow_up[:2] + without_follow_up[:2]
    for record_id in [row["record_id"] for row in examples] + ["APT-999"]:
        lines.append(json.dumps(check_appointment_status(record_id), indent=2))
    return "\n".join(lines) + "\n"


# Save a new Task 6 report while leaving all Part 1 outputs alone.
# Takes: nothing. Returns: nothing; writes the file and prints the same text.
def main():
    report = make_report()
    target = Path(__file__).resolve().parent / "outputs" / "part2_escalation.txt"
    target.parent.mkdir(exist_ok=True)
    target.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
