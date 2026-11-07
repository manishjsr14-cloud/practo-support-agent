"""Check the groundedness v2 experiment: average support, its threshold and the switch."""
# What: the average arithmetic, T2's midpoint and visible failure on overlap, the v1 default,
#   and that one weak term refuses under v1 but not under v2. Made-up reports only.
# Task: groundedness v2 tests. Used by: "python -m unittest discover -s tests".

import unittest
from unittest.mock import patch

import groundedness_v2
import guardrails
from guardrail_calibration import CALIBRATE_ANSWER, CALIBRATE_REFUSE


# Build a support report from (term, score) pairs, in guardrails.support_report's shape.
def report(*pairs):
    return {"terms": [(term, score, "made up") for term, score in pairs],
            "weakest": min((score for _, score in pairs), default=None)}


# Run threshold_v2 with fixed calibration reports and a fresh cache.
# Takes: {question: report}. Returns: threshold_v2()'s result (or raises its error).
def calibrate_with(reports):
    groundedness_v2.threshold_v2.cache_clear()
    try:
        with patch.object(guardrails, "support_report", side_effect=lambda q: reports[q]):
            return groundedness_v2.threshold_v2()
    finally:
        groundedness_v2.threshold_v2.cache_clear()


class GroundednessV2Tests(unittest.TestCase):
    # The average is the plain mean of the term scores; no terms gives None.
    def test_average_support(self):
        self.assertAlmostEqual(groundedness_v2.average_support(report(("a", 1.0), ("b", 0.4))), 0.7)
        self.assertIsNone(groundedness_v2.average_support(report()))

    # T2 is the midpoint between the highest refuse average and the lowest answer average.
    def test_threshold_midpoint(self):
        reports = {q: report(("x", 0.4)) for q in CALIBRATE_REFUSE}
        reports.update({q: report(("x", 0.8)) for q in CALIBRATE_ANSWER})
        self.assertAlmostEqual(calibrate_with(reports)["threshold"], 0.6)

    # Overlapping groups fail visibly instead of guessing a threshold.
    def test_overlap_fails_visibly(self):
        reports = {q: report(("x", 0.9)) for q in CALIBRATE_REFUSE}
        reports.update({q: report(("x", 0.5)) for q in CALIBRATE_ANSWER})
        with self.assertRaises(ValueError) as caught:
            calibrate_with(reports)
        self.assertIn("overlaps", str(caught.exception))

    # v1 stays the default; v2 is used only when asked for.
    def test_default_is_v1(self):
        self.assertEqual(guardrails.GROUNDEDNESS_VERSION, "v1")

    # One weak ordinary word: v1 refuses (weakest below T), v2 answers (average above T2).
    def test_one_weak_term_v1_refuses_v2_answers(self):
        made_up = report(("fee", 1.0), ("cancel", 1.0), ("know", 0.45))
        passing = {"terms": made_up["terms"], "weakest": 0.45}
        rag_result = {"fallback": False, "similarity": 0.6, "threshold": 0.3}
        with patch.object(guardrails, "support_report", return_value=passing), \
                patch.object(guardrails, "groundedness_threshold", return_value={"threshold": 0.5}), \
                patch.object(groundedness_v2, "threshold_v2", return_value={"threshold": 0.6}):
            v1 = guardrails.check_policy("q", rag_result, version="v1")
            v2 = guardrails.check_policy("q", rag_result, version="v2")
        self.assertFalse(v1[0])
        self.assertEqual(v2[:2], (True, None))


if __name__ == "__main__":
    unittest.main()
