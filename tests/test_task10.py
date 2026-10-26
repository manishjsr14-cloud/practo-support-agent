"""Check the Task 10 guardrails, starting with the input guardrails."""
# What: phone masking (what is and is not masked, and that history and the crew only see
#   the masked text) and prompt-injection blocking before any tool runs.
# Task: Part 2, Task 10 tests. Used by: "python -m unittest discover -s tests".

import settings  # noqa: F401  (must come before any crewai import)

import unittest
from unittest import mock

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


if __name__ == "__main__":
    unittest.main()
