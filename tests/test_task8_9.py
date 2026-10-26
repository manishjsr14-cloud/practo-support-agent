"""Check session memory (Task 8) and the validated structured reply (Task 9)."""
# What: runs transcripts A and B through pipeline.ask() once, then checks the remembered
#   record id, separate session histories, CrewResponse validation, the escalation flag
#   and that history stores exactly what prepare_user_text() returns.
# Task: Part 2, Tasks 8-9 tests. Used by: "python -m unittest discover -s tests".

import settings  # noqa: F401  (must come before any crewai import)

import unittest
from unittest import mock

from langchain_core.messages import AIMessage, HumanMessage

import memory
import pipeline
from mock_llm import NO_RECORD_ID
from schema import CrewResponse, validate_response
from tools import check_appointment_status, recommend_escalation

# Session ids are prefixed so they cannot meet any other session in the same test process.
TRANSCRIPTS = {
    "test-A": ("What is the status of APT-007?", "What is its fee?",
               "What is the cancellation and rescheduling window?"),
    "test-B": ("What is its fee?",),
}


class Task8And9Tests(unittest.TestCase):
    @classmethod
    # Run every turn once and keep each reply with its crew result.
    def setUpClass(cls):
        cls.turns = {}
        for session_id, texts in TRANSCRIPTS.items():
            for number, text in enumerate(texts, start=1):
                response = pipeline.ask(session_id, text)
                cls.turns[(session_id, number)] = {"response": response, **pipeline.LAST_RUN}

    # Turn A2 "What is its fee?" looks up APT-007 remembered from turn A1.
    def test_follow_up_uses_the_remembered_record_id(self):
        turn = self.turns[("test-A", 2)]
        self.assertEqual(turn["remembered_record_id"], "APT-007")
        self.assertEqual(turn["result"]["counts"]["check_appointment_status"], 1)
        self.assertEqual(turn["response"].record_id, "APT-007")

    # A policy question with no refer-back word does not look up the remembered record.
    def test_policy_turn_does_not_look_up_the_remembered_record(self):
        turn = self.turns[("test-A", 3)]
        self.assertIsNone(turn["remembered_record_id"])
        self.assertEqual(turn["result"]["counts"]["check_appointment_status"], 0)
        self.assertIsNone(turn["response"].record_id)

    # A new session has nothing to remember, so it says no id was given.
    def test_new_session_finds_no_record_id(self):
        turn = self.turns[("test-B", 1)]
        self.assertIsNone(turn["remembered_record_id"])
        self.assertIsNone(turn["response"].record_id)
        self.assertIn(NO_RECORD_ID, turn["response"].answer)

    # Each session's history holds only its own turns: B never sees A's APT-007.
    def test_sessions_do_not_share_history(self):
        history_b = memory.stored_history("test-B")
        self.assertEqual([kind for kind, _ in history_b], ["human", "ai"])
        self.assertNotIn("APT-007", " ".join(text for _, text in history_b))
        self.assertEqual(len(memory.stored_history("test-A")), 6)

    # The refer-back rule: whole words only, and never when the question has its own id.
    def test_refer_back_rule(self):
        history = [HumanMessage("What is the status of APT-007?"), AIMessage("record_id: APT-009")]
        cases = {"What is its fee?": "APT-007", "Can I cancel my appointment?": "APT-007",
                 "What is the cancellation window?": None, "Is there a waiting list?": None,
                 "What is the fee for APT-003?": None}
        for question, expected in cases.items():
            with self.subTest(question=question):
                self.assertEqual(memory.remembered_record_id(question, history), expected)

    # Every reply from ask() is a CrewResponse that validates again; a bad payload raises.
    def test_every_response_validates_and_invalid_payload_raises(self):
        for turn in self.turns.values():
            self.assertIsInstance(turn["response"], CrewResponse)
            self.assertEqual(validate_response(turn["response"].model_dump()), turn["response"])
        with self.assertRaises(ValueError):
            validate_response({"answer": "x", "sources": "03_consultation_fees", "tools_used": []})

    # sources lists each retrieved KB parent id once, in first-seen order.
    def test_sources_are_unique_parent_ids_in_order(self):
        for turn in self.turns.values():
            results = turn["result"]["rag_results"]
            retrieved = results[-1]["source_ids"] if results else []
            self.assertEqual(turn["response"].sources, list(dict.fromkeys(retrieved)))
        self.assertEqual(self.turns[("test-A", 2)]["response"].sources,
                         ["03_consultation_fees", "12_home_visits"])

    # CrewAI's own output_pydantic is already a CrewResponse, before pipeline.py fills facts.
    def test_crew_returns_a_crew_response_before_filling(self):
        for turn in self.turns.values():
            draft = turn["result"]["pydantic"]
            self.assertIsInstance(draft, CrewResponse)
            self.assertEqual((draft.sources, draft.tools_used, draft.record_id), ([], [], None))
            self.assertIsNone(draft.escalation_recommended)

    # No LLM call lacked an agent or task, so CrewAI never retried a conversion.
    def test_no_llm_call_without_agent_or_task(self):
        for key, turn in self.turns.items():
            with self.subTest(turn=key):
                self.assertTrue(all(call["agent"] and call["task"] for call in turn["result"]["calls"]))

    # The escalation flag is exactly tools.recommend_escalation() for the looked-up record.
    def test_escalation_comes_from_recommend_escalation(self):
        for turn in self.turns.values():
            response = turn["response"]
            if response.record_id is None:
                self.assertIsNone(response.escalation_recommended)
            else:
                score = check_appointment_status(response.record_id)["escalation_score"]
                self.assertEqual(response.escalation_recommended, recommend_escalation(score))

    # The remembered id sits on the Lookup task only and never reaches rag_lookup.
    def test_session_record_id_never_reaches_retrieval(self):
        result = self.turns[("test-A", 2)]["result"]
        self.assertEqual(result["rag_queries"], ["What is its fee?"])
        holders = [task.name for task in result["crew"].tasks if "Session record id:" in task.description]
        self.assertEqual(holders, ["Lookup"])

    # History stores what prepare_user_text() returns, so Task 10's masking will apply to it.
    def test_history_stores_prepared_text(self):
        for session_id, texts in TRANSCRIPTS.items():
            stored = [text for kind, text in memory.stored_history(session_id) if kind == "human"]
            self.assertEqual(stored, [memory.prepare_user_text(text) for text in texts])
        # With a changed prepare_user_text, history holds the changed text, not the raw one.
        with mock.patch.object(pipeline, "prepare_user_text", lambda text: text.upper()):
            pipeline.ask("test-C", "What is the cancellation and rescheduling window?")
        self.assertEqual(memory.stored_history("test-C")[0],
                         ("human", "WHAT IS THE CANCELLATION AND RESCHEDULING WINDOW?"))


if __name__ == "__main__":
    unittest.main()
