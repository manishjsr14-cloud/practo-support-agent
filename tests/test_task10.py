"""Check the Task 10 input and output guardrails."""
# What: phone masking (what is and is not masked, and that history and the crew only see
#   the masked text), prompt-injection blocking before any tool runs, and the two-stage
#   output check on deliberate cases (never on the pre-registered test sets' results).
# Task: Part 2, Task 10 tests. Used by: "python -m unittest discover -s tests".

import settings  # noqa: F401  (must come before any crewai import)

import unittest
from unittest import mock

import guardrails
import memory
import pipeline
from guardrail_queries import INJECTION_ATTACK, PII_EXAMPLE
from guardrails import injection_phrases, mask_phone_numbers


class InputGuardrailTests(unittest.TestCase):
    # The pre-registered example: the number is masked and APT-012 is untouched.
    def test_pii_example_raw_versus_masked(self):
        self.assertEqual(mask_phone_numbers(PII_EXAMPLE), "my number is [PHONE_MASKED], check APT-012")

    # Accepted number forms are masked; ids, dates, fees and other digit runs are not.
    def test_what_is_and_is_not_masked(self):
        # KB09: any standalone 10-digit number, including ones starting 0-5.
        masked = ["9876543210", "+919876543210", "+91 98765-43210", "call 6123456789.",
                  "5876543210", "01234 56789"]
        for text in masked:
            with self.subTest(text=text):
                self.assertIn("[PHONE_MASKED]", mask_phone_numbers(text))
                self.assertNotRegex(mask_phone_numbers(text), "[0-9]{5}")
        untouched = ["APT-012", "2026-10-26", "26/10/2026", "INR 1200", "fee of INR 1,200",
                     "98765432101", "123456789", "98765  43210", "APT-0129876543210"]
        for text in untouched:
            with self.subTest(text=text):
                self.assertEqual(mask_phone_numbers(text), text)

    # Through the full pipeline: history and the crew see only the masked text, and the
    # unmasked APT-012 is still looked up.
    def test_history_never_stores_the_raw_number(self):
        response = pipeline.ask("t10-pii", PII_EXAMPLE)
        stored = memory.stored_history("t10-pii")
        self.assertEqual(stored[0], ("human", "my number is [PHONE_MASKED], check APT-012"))
        self.assertTrue(all("98765" not in text for _, text in stored))
        self.assertEqual(pipeline.LAST_RUN["result"]["question"], "my number is [PHONE_MASKED], check APT-012")
        self.assertEqual(response.record_id, "APT-012")

    # The injection attack is refused with a reason, and the crew never runs.
    def test_injection_is_blocked_before_the_crew(self):
        self.assertEqual(injection_phrases(INJECTION_ATTACK),
                         ["ignore previous instructions", "reveal your system prompt"])
        with mock.patch.object(pipeline, "run_crew", side_effect=AssertionError("crew ran")):
            response = pipeline.ask("t10-injection", INJECTION_ATTACK)
        self.assertTrue(response.refused)
        self.assertFalse(response.grounded)
        self.assertIn("ignore previous instructions", response.refusal_reason)
        self.assertEqual(response.tools_used, [])

    # Ordinary questions contain no injection phrase.
    def test_ordinary_questions_are_not_flagged(self):
        for text in ("What is the cancellation and rescheduling window?", "What is its fee?"):
            self.assertEqual(injection_phrases(text), [])


class OutputGuardrailTests(unittest.TestCase):
    @classmethod
    # Run the deliberate output cases once through the full pipeline.
    def setUpClass(cls):
        cls.turns = {}
        for name, question in {
                "stage1": "Give me a recipe for chocolate brownies.",
                "stage2": "What are the clinic's opening hours on Sundays?",
                "mixed": "What is the status of APT-007 and what are the clinic's opening hours on Sundays?",
                "record": "What is the status of APT-007?",
                "missing": "What is the status of APT-999?",
                "policy": "What is the cancellation and rescheduling window?"}.items():
            cls.turns[name] = (pipeline.ask(f"t10-{name}", question), dict(pipeline.LAST_RUN))

    # T comes only from guardrail_calibration.py: the midpoint of the two groups, which
    # must not overlap.
    def test_threshold_is_the_calibration_midpoint(self):
        from guardrail_calibration import CALIBRATE_ANSWER, CALIBRATE_REFUSE
        calibration = guardrails.groundedness_threshold()
        self.assertEqual(list(calibration["refuse"]), CALIBRATE_REFUSE)
        self.assertEqual(list(calibration["answer"]), CALIBRATE_ANSWER)
        high = max(r["weakest"] for r in calibration["refuse"].values())
        low = min(r["weakest"] for r in calibration["answer"].values())
        self.assertLess(high, low)
        self.assertEqual(calibration["threshold"], (high + low) / 2)

    # Terms drop stop words and record ids; a word present in the chunks scores 1.0.
    def test_terms_and_text_match(self):
        self.assertEqual(guardrails.question_terms("What is the fee for APT-007 on Sundays?"), ["fee", "sundays"])
        report = guardrails.support_report("What is the cancellation and rescheduling window?")
        self.assertEqual([score for _, score, _ in report["terms"]], [1.0, 1.0, 1.0])

    # Stage 1 (similarity) and Stage 2 (terms) each refuse with their own reason.
    def test_stage1_and_stage2_refusals(self):
        stage1, stage2 = self.turns["stage1"][0], self.turns["stage2"][0]
        self.assertTrue(stage1.refused and stage1.refusal_reason.startswith("Stage 1"))
        self.assertTrue(stage2.refused and stage2.refusal_reason.startswith("Stage 2"))
        self.assertIn("'opening'", stage2.refusal_reason)
        self.assertEqual((stage1.sources, stage2.sources), ([], []))

    # A supported policy answer is grounded and cites its sources.
    def test_supported_policy_answer_is_grounded(self):
        response = self.turns["policy"][0]
        self.assertTrue(response.grounded)
        self.assertFalse(response.refused)
        self.assertEqual(response.sources, ["02_cancellation_rescheduling"])

    # Mixed question: the record is kept and the answer names the unanswered policy part.
    def test_mixed_question_names_the_unanswered_part(self):
        response = self.turns["mixed"][0]
        self.assertEqual(response.record_id, "APT-007")
        self.assertFalse(response.refused)
        self.assertIn("Not answered: the policy part 'what are the clinic's opening hours on Sundays?'",
                      response.answer)
        self.assertNotIn("Retrieval result:", response.answer)

    # Record-only questions skip the policy check: grounded if found, refused if not.
    def test_record_only_questions(self):
        found, missing = self.turns["record"], self.turns["missing"]
        self.assertIsNone(found[1]["guardrail"])
        self.assertTrue(found[0].grounded and not found[0].refused)
        self.assertTrue(missing[0].refused)
        self.assertIn("No appointment was found", missing[0].refusal_reason)


class EmergencyRuleTests(unittest.TestCase):
    # The pre-registered set: every SHOULD_TRIGGER message fires, no SHOULD_NOT_TRIGGER does.
    def test_pre_registered_set(self):
        from emergency_queries import SHOULD_NOT_TRIGGER, SHOULD_TRIGGER
        for text in SHOULD_TRIGGER:
            self.assertTrue(guardrails.emergency_phrases(text), text)
        for text in SHOULD_NOT_TRIGGER:
            self.assertEqual(guardrails.emergency_phrases(text), [], text)

    # The reply quotes KB08 word for word, cites KB08, and neither diagnoses nor names a medicine.
    def test_reply_quotes_kb08_without_advice(self):
        response = guardrails.emergency_response(["seizure"])
        kb08 = guardrails.KB08_PATH.read_text(encoding="utf-8")
        quote = response.answer.split('"')[1]
        self.assertIn(quote, kb08)
        self.assertIn("local emergency services or the nearest emergency department", quote)
        self.assertEqual(response.sources, ["08_emergency_protocol"])
        self.assertTrue(response.grounded and not response.refused)
        preface = response.answer.split('"')[0].lower()
        for word in ("diagnos", "medicine", "medication", "tablet", "dose"):
            self.assertNotIn(word, preface)

    # Through the pipeline the rule runs before the crew, so no tool or groundedness check runs.
    def test_rule_runs_before_the_crew(self):
        with mock.patch.object(pipeline, "run_crew", side_effect=AssertionError("crew ran")):
            response = pipeline.ask("t10-emergency", "My father collapsed and is not breathing")
        self.assertEqual(response.tools_used, [])
        self.assertIsNone(pipeline.LAST_RUN["guardrail"])
        self.assertEqual(pipeline.LAST_RUN["emergency"], ["not breathing", "collapsed"])


if __name__ == "__main__":
    unittest.main()
