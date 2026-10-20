"""Print and save measured administrative escalation results without changing appointments."""

import json
from pathlib import Path

import numpy as np

from dataset import APPOINTMENTS
from rules import ESCALATION_PERCENTILE, FOLLOW_UP_WEIGHT, RECENCY_DAYS, RECENCY_WEIGHT
from tools import check_appointment_status, escalation_score, escalation_threshold


# Calculate the distribution, counts and deterministic example lookups from the records.
def make_report() -> str:
    scores = np.array([escalation_score(row["follow_up_required"], row["days_since_created"])
                       for row in APPOINTMENTS])
    threshold = escalation_threshold()
    above = int(np.count_nonzero(scores >= threshold))
    follow_up = [row for row in APPOINTMENTS if row["follow_up_required"]]
    without_follow_up = [row for row in APPOINTMENTS if not row["follow_up_required"]]
    follow_up_above = sum(escalation_score(row["follow_up_required"], row["days_since_created"])
                          >= threshold for row in follow_up)
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
    examples = follow_up[:2] + without_follow_up[:2]
    for record_id in [row["record_id"] for row in examples] + ["APT-999"]:
        lines.append(json.dumps(check_appointment_status(record_id), indent=2))
    return "\n".join(lines) + "\n"


# Save a new Task 6 report while leaving all Part 1 outputs alone.
def main():
    report = make_report()
    target = Path(__file__).resolve().parent / "outputs" / "part2_escalation.txt"
    target.parent.mkdir(exist_ok=True)
    target.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
