"""Check recorded appointment lookup and the student's administrative escalation rules."""
# What: automated checks for the Task 6 escalation score, appointment lookup, error
#   handling and the data-derived percentile threshold.
# Task: Part 2, Task 6 tests. Used by: "python -m unittest discover -s tests".
# It imports rules.py, tools.py and dataset.py.

from copy import deepcopy
import unittest
from unittest.mock import patch

import numpy as np

from dataset import APPOINTMENTS
import rules
import tools


class Task6Tests(unittest.TestCase):
    # Check the fixed rules, all real-record score bounds, and exact endpoint scores.
    # Protects: the chosen weights (0.6, 0.4, 30 days, 80th percentile), the formula for
    #   all 50 records, and that scores stay within 0.0-1.0.
    def test_scores_match_the_formula_and_stay_in_range(self):
        self.assertEqual((rules.FOLLOW_UP_WEIGHT, rules.RECENCY_WEIGHT,
                          rules.RECENCY_DAYS, rules.ESCALATION_PERCENTILE), (0.6, 0.4, 30, 80))
        self.assertEqual(len(APPOINTMENTS), 50)
        for row in APPOINTMENTS:
            # subTest reports which record failed instead of stopping at the first one.
            with self.subTest(record_id=row["record_id"]):
                actual = tools.escalation_score(row["follow_up_required"], row["days_since_created"])
                expected = 0.6 * int(row["follow_up_required"]) + 0.4 * (row["days_since_created"] / 30)
                self.assertIs(type(actual), float)
                self.assertAlmostEqual(actual, expected, places=14)
                self.assertGreaterEqual(actual, 0.0)
                self.assertLessEqual(actual, 1.0)
        # The four corner cases must give these exact values.
        for follow_up, days, expected in ((False, 0, 0.0), (False, 30, 0.4),
                                          (True, 0, 0.6), (True, 30, 1.0)):
            self.assertEqual(tools.escalation_score(follow_up, days), expected)

    # Check each signal contributes its own fixed weight instead of acting as a boolean OR.
    # Protects: the score is a weighted sum: the follow-up flag always adds 0.6 and each
    #   day of age adds 0.4/30, whatever the other signal is.
    def test_follow_up_and_age_have_independent_contributions(self):
        for days in (0, 15, 30):
            difference = tools.escalation_score(True, days) - tools.escalation_score(False, days)
            self.assertAlmostEqual(difference, 0.6, places=14)
        for follow_up in (False, True):
            start = tools.escalation_score(follow_up, 0)
            self.assertAlmostEqual(tools.escalation_score(follow_up, rules.RECENCY_DAYS) - start,
                                   0.4, places=14)
            self.assertAlmostEqual(tools.escalation_score(follow_up, 1) - start,
                                   0.4 / 30, places=14)

    # Check every lookup returns recorded fees and statuses without modifying the dataset.
    # Protects: "existing appointment records remain authoritative for their own fees",
    #   and that a lookup never edits the records.
    def test_lookup_returns_the_six_recorded_fields_and_score(self):
        original = deepcopy(APPOINTMENTS)
        fields = {"record_id", "status", "consultation_fee_inr", "follow_up_required",
                  "days_since_created", "escalation_score"}
        for row in APPOINTMENTS:
            with self.subTest(record_id=row["record_id"]):
                result = tools.check_appointment_status(row["record_id"])
                self.assertEqual(set(result), fields)
                for key in fields - {"escalation_score"}:
                    self.assertEqual(result[key], row[key])
                expected = 0.6 * int(row["follow_up_required"]) + 0.4 * (row["days_since_created"] / 30)
                self.assertAlmostEqual(result["escalation_score"], expected, places=14)
        self.assertEqual(APPOINTMENTS, original)

    # Check malformed, non-string, Unicode-digit, and unknown IDs return errors without coercion.
    # Protects: bad input gets a controlled error (never a crash or a wrong record), and
    #   the original input is returned unchanged ("is" checks it is the same object).
    #   APT-000 and APT-999 have the right format but match no record.
    def test_invalid_and_unknown_ids_return_controlled_errors(self):
        invalid = (None, 1, True, [], {}, b"APT-001", "", "apt-001", "APT999", "APT-01",
                   "APT-0001", " APT-001", "APT-001 ", "APT-001\n", "APT-٠٠١", "APT-１２３",
                   "APT-000", "APT-999")
        for value in invalid:
            with self.subTest(record_id=repr(value)):
                result = tools.check_appointment_status(value)
                self.assertEqual(set(result), {"error", "record_id"})
                self.assertIsInstance(result["error"], str)
                self.assertTrue(result["error"])
                self.assertIs(result["record_id"], value)

    # Change one copied high-score row and verify the live percentile and count respond correctly.
    # Protects: the threshold is computed from the current data each time, not stored as
    #   a fixed number. Only a copy is changed, and the real dataset is checked afterwards.
    def test_one_record_perturbation_recomputes_the_live_threshold(self):
        original = deepcopy(tools.APPOINTMENTS)
        scores = [0.6 * int(row["follow_up_required"]) + 0.4 * (row["days_since_created"] / 30)
                  for row in original]
        before_threshold = tools.escalation_threshold()
        self.assertEqual(before_threshold, float(np.percentile(scores, 80)))
        self.assertGreater(before_threshold, 0.0)
        before_count = sum(score >= before_threshold for score in scores)
        # Take the first record at or above the threshold and lower its score to 0.
        selected = next(index for index, score in enumerate(scores) if score >= before_threshold)
        changed = deepcopy(original)
        changed[selected]["follow_up_required"] = False
        changed[selected]["days_since_created"] = 0
        self.assertEqual(sum(left != right for left, right in zip(original, changed)), 1)
        new_scores = [0.6 * int(row["follow_up_required"]) + 0.4 * (row["days_since_created"] / 30)
                      for row in changed]
        expected_threshold = float(np.percentile(new_scores, 80))
        # patch.object swaps tools.APPOINTMENTS for the changed copy only inside this block.
        with patch.object(tools, "APPOINTMENTS", changed):
            after_threshold = tools.escalation_threshold()
            self.assertEqual(after_threshold, expected_threshold)
            after_count = sum(tools.recommend_escalation(score) for score in new_scores)
            self.assertEqual(after_count, sum(score >= expected_threshold for score in new_scores))
            self.assertNotEqual((before_threshold, before_count), (after_threshold, after_count))
        self.assertEqual(tools.APPOINTMENTS, original)
        self.assertEqual(APPOINTMENTS, original)

    # Check the inclusive boundary using adjacent representable floats instead of an arbitrary epsilon.
    # Protects: the ">=" rule: a score exactly at the threshold is escalated, and the very
    #   next smaller float (np.nextafter) is not.
    def test_recommendation_includes_the_threshold_and_excludes_the_next_lower_float(self):
        threshold = tools.escalation_threshold()
        below = float(np.nextafter(threshold, -np.inf))
        above = float(np.nextafter(threshold, np.inf))
        self.assertIs(tools.recommend_escalation(below), False)
        self.assertIs(tools.recommend_escalation(threshold), True)
        self.assertIs(tools.recommend_escalation(above), True)


if __name__ == "__main__":
    unittest.main()
