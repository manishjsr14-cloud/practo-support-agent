"""Compare emergency detection v1 (phrases) and v2 (phrases or meaning) and save the output."""
# What: prints the calibration scores and T_EMERGENCY, then v1 and v2 decisions with the
#   best similarity for the v2 test set, the Part 2 emergency set, the 20 Part 2 guardrail
#   questions and the 15 Part 3 eval questions, and lists every question v2 changes.
# Task: Part 3 extension (emergency detection v2). Used by: run by hand; it writes
#   outputs/part3_emergency_v2.txt. It calls the detector only; no earlier output is rerun.

import settings  # noqa: F401  (sets the offline flags before crewai is imported)

from pathlib import Path

import emergency_queries
import emergency_v2_queries
from emergency import detect, emergency_threshold
from eval_queries import EVAL_QUERIES
from guardrail_queries import SHOULD_ANSWER, SHOULD_REFUSE

# Each set: (title, [(question, expected trigger or None when the set has no emergency label)]).
SETS = (
    ("Emergency v2 test set (emergency_v2_queries.py)",
     [(q, True) for q in emergency_v2_queries.SHOULD_TRIGGER]
     + [(q, False) for q in emergency_v2_queries.SHOULD_NOT_TRIGGER]),
    ("Part 2 emergency set (emergency_queries.py)",
     [(q, True) for q in emergency_queries.SHOULD_TRIGGER]
     + [(q, False) for q in emergency_queries.SHOULD_NOT_TRIGGER]),
    ("Part 2 guardrail questions (guardrail_queries.py; no emergency label)",
     [(q, None) for q in SHOULD_ANSWER + SHOULD_REFUSE]),
    ("Part 3 eval questions (eval_queries.py; expected emergency only for e08)",
     [(c["query"], c["expect"] == "emergency") for c in EVAL_QUERIES]),
)


# Write one decision as "yes"/"no", and mark it against the label when there is one.
def mark(fired, expected):
    word = "yes" if fired else "no "
    return word if expected is None else word + (" ok  " if fired == expected else " MISS")


# Build the report text.
# Takes: nothing. Returns: the report as one string.
def make_report():
    calibration = emergency_threshold()
    lines = ["Emergency detection v2: v1 = phrase list; v2 = phrase list OR best similarity to "
             "EMERGENCY_DESCRIPTIONS >= T_EMERGENCY", "",
             "Calibration (emergency_calibration.py only; test sets never used):"]
    for group, label in (("emergency", "CAL_EMERGENCY"), ("not", "CAL_NOT")):
        lines += [f"  [{label}] {score:.9f}  {message}" for message, score in calibration[group].items()]
    high, low = max(calibration["not"].values()), min(calibration["emergency"].values())
    lines += [f"max(CAL_NOT) = {high:.9f}; min(CAL_EMERGENCY) = {low:.9f}",
              f"T_EMERGENCY = {calibration['threshold']:.9f}; rule: {calibration['rule']}",
              "CAL_NOT messages that trigger at T_EMERGENCY: "
              + (", ".join(repr(m) for m in calibration["false_triggers"]) or "none")]
    changes = []
    for title, cases in SETS:
        lines += ["", f"=== {title}", "| v1 | v2 | best similarity | closest description | question |",
                  "|---|---|---:|---|---|"]
        right = {"v1": 0, "v2": 0}
        for question, expected in cases:
            found = detect(question)
            for version in right:
                right[version] += expected is not None and found[version] == expected
            lines.append(f"| {mark(found['v1'], expected)} | {mark(found['v2'], expected)} | "
                         f"{found['similarity']:.9f} | {found['description']} | {question} |")
            if found["v1"] != found["v2"]:
                changes.append(f"{title.split(' (')[0]}: {question!r} -> now gets the KB08 emergency "
                               f"direction (meaning {found['similarity']:.9f})")
        if any(expected is not None for _, expected in cases):
            lines.append(f"Correct: v1 {right['v1']} of {len(cases)}; v2 {right['v2']} of {len(cases)}")
    lines += ["", "Questions whose outcome v2 changes (v1 did not trigger, v2 does):"]
    lines += [f"  {change}" for change in changes] or ["  none"]
    return "\n".join(lines) + "\n"


# Save the report to outputs/part3_emergency_v2.txt and print it.
def main():
    report = make_report()
    target = Path(__file__).resolve().parent / "outputs" / "part3_emergency_v2.txt"
    target.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
