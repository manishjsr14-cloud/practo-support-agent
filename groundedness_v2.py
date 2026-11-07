"""Groundedness v2 (experiment): Stage 2 decides on the average term support, not the weakest."""
# What: v1 refuses when any single question term is below T, so one ordinary word could refuse
#   a good answer. v2 averages every term's support score and refuses only when that average
#   is below T2. Terms and per-term scores are v1's own (guardrails.support_report), and no
#   word list is used. Stage 1 (Part 1's similarity gate) is unchanged.
# Task: groundedness v2 experiment. Used by: guardrails.check_policy (version "v2"),
#   evaluate_groundedness_v2.py and tests/test_groundedness_v2.py.
# T2 comes only from guardrail_calibration.py, by the midpoint rule; overlapping groups fail
#   visibly instead of guessing.

from functools import lru_cache

from guardrail_calibration import CALIBRATE_ANSWER, CALIBRATE_REFUSE


# The mean support of a question's terms.
# Takes: a report from guardrails.support_report(). Returns: the average, or None with no terms.
def average_support(report):
    scores = [score for _, score, _ in report["terms"]]
    return sum(scores) / len(scores) if scores else None


# Measure T2 from the calibration questions only (midpoint rule).
# Returns: {"refuse": {question: average}, "answer": {question: average}, "threshold"};
#   raises ValueError if the highest refuse average is not below the lowest answer average.
@lru_cache(maxsize=1)
def threshold_v2():
    from guardrails import support_report
    refuse = {q: average_support(support_report(q)) for q in CALIBRATE_REFUSE}
    answer = {q: average_support(support_report(q)) for q in CALIBRATE_ANSWER}
    high, low = max(refuse.values()), min(answer.values())
    if not high < low:
        raise ValueError(f"v2 calibration overlaps: max refuse average {high:.9f} >= "
                         f"min answer average {low:.9f}; no T2 is set.")
    return {"refuse": refuse, "answer": answer, "threshold": (high + low) / 2}


# Decide Stage 2 by the average support.
# Takes: a support report. Returns: (ok, reason); a question with no content terms passes, as in v1.
# The reason still names every term below T2, so the weak words stay visible.
def stage2_v2(report):
    average = average_support(report)
    threshold = threshold_v2()["threshold"]
    if average is None or average >= threshold:
        return True, None
    weak = [term for term, score, _ in report["terms"] if score < threshold]
    return False, (f"Stage 2 (v2): average term support {average:.9f} is below T2={threshold:.9f}; "
                   f"weakest terms: {', '.join(repr(t) for t in weak)}")
