"""Check emergency detection v2: descriptions, threshold rule and pipeline routing."""
# What: the descriptions copy no calibration or test sentence, T_EMERGENCY follows the
#   midpoint-or-overlap rule from the calibration messages only, a meaning-only emergency
#   gets KB08's reply through the pipeline, and phrase matches behave as in v1.
# Task: Part 3 extension tests. Used by: "python -m unittest discover -s tests".

import settings  # noqa: F401  (must come before any crewai import)

import unittest
from unittest import mock

import emergency
import emergency_calibration
import emergency_queries
import emergency_v2_queries
import pipeline
from rules import EMERGENCY_DESCRIPTIONS


class EmergencyV2Tests(unittest.TestCase):
    # No description equals a calibration or test sentence.
    def test_descriptions_copy_no_calibration_or_test_sentence(self):
        others = (emergency_calibration.CAL_EMERGENCY + emergency_calibration.CAL_NOT
                  + emergency_v2_queries.SHOULD_TRIGGER + emergency_v2_queries.SHOULD_NOT_TRIGGER
                  + emergency_queries.SHOULD_TRIGGER + emergency_queries.SHOULD_NOT_TRIGGER)
        normal = lambda text: text.lower().strip(" .?!")
        self.assertEqual(len(EMERGENCY_DESCRIPTIONS), 8)
        self.assertFalse({normal(d) for d in EMERGENCY_DESCRIPTIONS} & {normal(s) for s in others})

    # T_EMERGENCY: the midpoint when the groups separate, else min(CAL_EMERGENCY), and every
    # CAL_NOT message at or above it is listed as a false trigger.
    def test_threshold_rule(self):
        calibration = emergency.emergency_threshold()
        self.assertEqual(list(calibration["emergency"]), emergency_calibration.CAL_EMERGENCY)
        self.assertEqual(list(calibration["not"]), emergency_calibration.CAL_NOT)
        high, low = max(calibration["not"].values()), min(calibration["emergency"].values())
        expected = (high + low) / 2 if high < low else low
        self.assertEqual(calibration["threshold"], expected)
        self.assertTrue(all(score >= expected for score in calibration["emergency"].values()))
        self.assertEqual(calibration["false_triggers"],
                         [m for m, s in calibration["not"].items() if s >= expected])

    # Every v1 phrase match still triggers v2, marked as a phrase match.
    def test_phrase_matches_behave_as_v1(self):
        for text in emergency_queries.SHOULD_TRIGGER:
            found = emergency.detect(text)
            self.assertTrue(found["v1"] and found["v2"])
            self.assertEqual(found["by"], "phrase")

    # A message with no listed phrase but an emergency meaning gets KB08's reply, before the crew.
    def test_meaning_only_emergency_through_the_pipeline(self):
        text = "My child has stopped breathing, what do I do?"
        found = emergency.detect(text)
        self.assertEqual((found["v1"], found["v2"], found["by"]), (False, True, "meaning"))
        with mock.patch.object(pipeline, "run_crew", side_effect=AssertionError("crew ran")):
            response = pipeline.ask("v2-meaning", text)
        self.assertEqual(response.sources, ["08_emergency_protocol"])
        self.assertIn("local emergency services", response.answer)
        self.assertTrue(pipeline.LAST_RUN["emergency"][0].startswith("meaning:"))


if __name__ == "__main__":
    unittest.main()
