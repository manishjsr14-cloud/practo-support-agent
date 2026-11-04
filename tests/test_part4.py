"""Check the Part 4 review stage and governance."""
# What: the reviewer and editor rules, the bounded two-agent team, the three
#   pre-registered review cases and the least-autonomy tool wiring.
# Task: Part 4 tests. Used by: "python -m unittest discover -s tests".

import settings  # noqa: F401  (must come before any framework import)

import asyncio
import logging
import unittest
from unittest.mock import patch

from autogen_agentchat.teams import RoundRobinGroupChat

import crew
import governance
import review
from mock_llm import MockLLM
from part4_preregistration import INJECTED_SENTENCE, REVIEW_CASES
from schema import CrewResponse


class ReviewTests(unittest.TestCase):
    # Run the crew once per pre-registered question and keep its draft and context.
    @classmethod
    def setUpClass(cls):
        cls.runs = {}
        for case in REVIEW_CASES:
            if case["question"] not in cls.runs:
                result = crew.run_crew(case["question"], output_model=CrewResponse)
                cls.runs[case["question"]] = (result["pydantic"].answer, review.crew_context(result))

    # Labels, the KB08 note and fixed fallbacks are skipped; other sentences are checked.
    def test_policy_sentences_skip_labels_and_fixed_text(self):
        draft = "Retrieval result:\nA rule. B rule.\nSources: x\n\nLookup result:\n" + review.KB08_NOTE
        self.assertEqual(review.policy_sentences(draft), ["A rule.", "B rule."])

    # The editor approves an empty list unchanged and removes listed sentences otherwise.
    def test_edit_rules(self):
        self.assertEqual(review.edit("A. B.", []).model_dump(),
                         {"approved": True, "final_answer": "A. B.", "reason": "Every policy sentence is in the sources."})
        verdict = review.edit("A. B.", ["B."])
        self.assertEqual((verdict.approved, verdict.final_answer), (False, "A."))
        self.assertIn("'B.'", verdict.reason)

    # Each pre-registered case gets its expected verdict, with one call per agent.
    def test_preregistered_cases(self):
        for case in REVIEW_CASES:
            draft, context = self.runs[case["question"]]
            given = f"{draft} {case['inject']}" if case["inject"] else draft
            verdict, messages = review.review_draft(given, context)
            self.assertEqual(review.CALLS, {review.REVIEWER: 1, review.EDITOR: 1})
            self.assertEqual(len(messages), 3)
            if case["expect"] == "approved unchanged":
                self.assertTrue(verdict.approved)
                self.assertEqual(verdict.final_answer, given)
            else:
                self.assertFalse(verdict.approved)
                self.assertEqual(verdict.final_answer, draft)
                self.assertIn(INJECTED_SENTENCE, verdict.reason)

    # Without custom_message_types the editor's StructuredMessage is not registered.
    def test_team_without_custom_message_types_fails(self):
        draft, context = self.runs[REVIEW_CASES[0]["question"]]
        team = RoundRobinGroupChat(list(review.build_agents()), max_turns=2)
        # AutoGen's runtime also logs the failure with a traceback; keep the test output clean.
        logging.disable(logging.CRITICAL)
        try:
            with self.assertRaises(Exception) as caught:
                asyncio.run(team.run(task=review.task_text(draft, context)))
        finally:
            logging.disable(logging.NOTSET)
        self.assertIn("not registered", str(caught.exception))


class ToolWiringTests(unittest.TestCase):
    # The real crew passes: only the Lookup Agent holds the record tool.
    def test_real_crew_is_allowed(self):
        built = crew.build_crew(MockLLM(model="mock-llm"))
        holders = [agent.role for agent in built.agents
                   if governance.RECORD_TOOL in {tool.name for tool in agent.tools}]
        self.assertEqual(holders, ["Lookup Agent"])

    # Wiring the record tool to the Retrieval Agent stops the crew before any call.
    def test_record_tool_on_another_agent_is_blocked(self):
        with patch.object(crew, "rag_lookup_tool", crew.check_appointment_status_tool):
            with self.assertRaises(governance.ToolWiringError) as caught:
                crew.run_crew("What is the status of APT-007?")
        self.assertIn("Retrieval Agent may not hold check_appointment_status", str(caught.exception))
        self.assertEqual(crew.COUNTS, {"rag_lookup": 0, "check_appointment_status": 0})


if __name__ == "__main__":
    unittest.main()
