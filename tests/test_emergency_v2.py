"""Check emergency detection v2: descriptions, threshold rule and pipeline routing."""
# What: the descriptions copy no calibration or test sentence, T_EMERGENCY follows the
#   midpoint-or-overlap rule from the calibration messages only, phrase matches behave as in
#   v1, and the two-level response: a phrase match gets the emergency reply only, a
#   meaning-only match gets the normal reply plus the KB08 note.
# Task: Part 3 extension tests. Used by: "python -m unittest discover -s tests".

import settings  # noqa: F401  (must come before any crewai import)

import unittest
from unittest import mock

import emergency
import emergency_calibration
import emergency_queries
import emergency_v2_queries
import pipeline
from guardrails import KB08_PATH
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

    # Two levels (added after the first v2 run): a meaning-only match runs the normal pipeline
    # and appends the note and KB08's direction, adding 08_emergency_protocol to sources.
    def test_meaning_only_match_gets_normal_reply_plus_note(self):
        text = "My child has stopped breathing, what do I do?"
        found = emergency.detect(text)
        self.assertEqual((found["v1"], found["v2"], found["by"]), (False, True, "meaning"))
        response = pipeline.ask("v2-meaning", text)
        self.assertIsNotNone(pipeline.LAST_RUN["result"])
        self.assertNotIn("emergency", pipeline.LAST_RUN)
        self.assertTrue(response.answer.endswith(f"{emergency.NOTE}\nClinic policy (KB08): "
                                                 f"\"{emergency.kb08_direction()}\""))
        self.assertIn("08_emergency_protocol", response.sources)

    # A phrase match still gets the emergency reply only, before the crew.
    def test_phrase_match_gets_emergency_reply_only(self):
        with mock.patch.object(pipeline, "run_crew", side_effect=AssertionError("crew ran")):
            response = pipeline.ask("v2-phrase", "My father collapsed and is not breathing")
        self.assertEqual(response.sources, ["08_emergency_protocol"])
        self.assertEqual(pipeline.LAST_RUN["emergency"], ["not breathing", "collapsed"])

    # The note gives the direction only: it neither diagnoses nor names a medicine, and the
    # quoted KB08 sentence is word for word from kb/08.
    def test_note_gives_direction_only(self):
        for word in ("diagnos", "medicine", "medication", "tablet", "dose"):
            self.assertNotIn(word, emergency.NOTE.lower())
        self.assertIn(emergency.kb08_direction(), KB08_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
