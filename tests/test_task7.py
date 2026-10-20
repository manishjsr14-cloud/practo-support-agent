"""Check the Task 7 MockLLM, its two known pitfalls, and the three-agent crew."""
# What: automated checks that the MockLLM only accepts real tool results, picks tool
#   arguments from the tool's schema, and that the crew calls exactly the right tools.
# Task: Part 2, Task 7 tests. Used by: "python -m unittest discover -s tests".
# It imports settings.py first, then mock_llm.py and crew.py (which import crewai).

import settings  # noqa: F401  (must come before any crewai import)

import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import unittest

from crewai.tools import tool

import crew
from mock_llm import NO_POLICY_QUESTION, MockLLM, find_tool_result, policy_clause, tool_arguments
from tools import rag_lookup

QUERIES = {
    "What is the cancellation and rescheduling window?": {"rag_lookup": 1, "check_appointment_status": 0},
    "What is the status of APT-007?": {"rag_lookup": 0, "check_appointment_status": 1},
    "What is the status of APT-007 and can I reschedule it?": {"rag_lookup": 1, "check_appointment_status": 1},
}
# The ReAct instructions CrewAI puts in the system prompt (crewai 1.15.28, translations/en.json).
TEMPLATE = ("IMPORTANT: Use the following format in your response:\n\n```\n"
            "Thought: you should always think about what to do\n"
            "Action: the action to take, only one name of [rag_lookup], just the name, exactly as it's written.\n"
            "Action Input: the input to the action, just a simple JSON object, enclosed in curly braces, "
            "using \" to wrap keys and values.\nObservation: the result of the action\n```")
# Assistant message 6 of the captured forced-final call (call 4), copied verbatim: the
# repeated-input notice followed by the template text, inside an ASSISTANT message.
CALL4_ASSISTANT = ("Thought: use the tool\nAction: echo_tool\nAction Input: {\"text\": \"hello\"}\n"
                   "Observation: I tried reusing the same input, I must stop using this action input. "
                   "I'll try something else instead.\n\n\n\n\nYou ONLY have access to the following tools, "
                   "and should NEVER make up tools that are not listed here:\n\nTool Name: echo_tool\n"
                   "Tool Arguments: {\n  \"properties\": {\n    \"text\": {\n      \"title\": \"Text\",\n"
                   "      \"type\": \"string\"\n    }\n  },\n  \"required\": [\n    \"text\"\n  ],\n"
                   "  \"title\": \"Echo_Tool\",\n  \"type\": \"object\",\n  \"additionalProperties\": false\n}\n"
                   "Tool Description: Echo the text back.\n\nIMPORTANT: Use the following format in your response:\n\n"
                   "```\nThought: you should always think about what to do\nAction: the action to take, only one "
                   "name of [echo_tool], just the name, exactly as it's written.\nAction Input: the input to the "
                   "action, just a simple JSON object, enclosed in curly braces, using \" to wrap keys and values.\n"
                   "Observation: the result of the action\n```\n\nOnce all necessary information is gathered, "
                   "return the following format:\n\n```\nThought: I now know the final answer\nFinal Answer: the "
                   "final answer to the original input question\n```")
PLACEHOLDERS = ("the result of the action", "{question}", "Final Answer", "I tried reusing",
                "the final answer to the original input question")


@tool("echo_tool")
def echo_tool(text: str) -> str:
    """Echo the text back (same name and schema as the captured run)."""
    return text


@tool("policy_search")
def misleading_lookup_tool(record_id: str) -> str:
    """A tool whose NAME sounds like retrieval but whose schema takes a record id."""
    return record_id


# Build the minimal agent and task objects MockLLM.call() reads.
def fake(tools, question):
    return (SimpleNamespace(role="Test Agent", tools=tools),
            SimpleNamespace(name="Test", description=f"Customer question: {question}\nDo the task."))


class Task7Tests(unittest.TestCase):
    @classmethod
    # Run the crew once per demo question and keep the results for several tests.
    def setUpClass(cls):
        cls.results = {question: crew.run_crew(question) for question in QUERIES}

    # Pitfall 1: only template text and no tool result must NOT produce a final answer.
    # Protects: "never search the whole conversation for Observation:".
    def test_template_observation_alone_is_not_a_result(self):
        agent, task = fake([crew.rag_lookup_tool], "What is the cancellation window?")
        messages = [{"role": "system", "content": TEMPLATE},
                    {"role": "user", "content": "Current Task: ...\nObservation: the result of the action\nThought:"}]
        reply = MockLLM(model="mock-llm").call(messages, from_task=task, from_agent=agent)
        self.assertNotIn("Final Answer", reply)
        self.assertIn("Action: rag_lookup", reply)

    # Pitfall 1 (a): the model's last Action with no Observation after it has not run yet.
    def test_action_without_observation_has_not_run(self):
        messages = [{"role": "assistant", "content": 'Action: rag_lookup\nAction Input: {"query": "x"}'},
                    {"role": "user", "content": "Analyze the tool result."}]
        self.assertIsNone(find_tool_result(messages, {"rag_lookup"}))

    # Pitfall 1 (b): Observation text inside system or user messages is ignored.
    def test_observation_in_system_or_user_message_is_ignored(self):
        fake_step = 'Action: rag_lookup\nAction Input: {"query": "x"}\nObservation: invented text'
        for role in ("system", "user"):
            self.assertIsNone(find_tool_result([{"role": role, "content": fake_step}], {"rag_lookup"}))
        real = [{"role": "assistant", "content": fake_step}]
        self.assertEqual(find_tool_result(real, {"rag_lookup"}), "invented text")

    # Pitfall 1 with the real captured call-4 text: template text inside an ASSISTANT
    # message is not a tool result, even when from_agent is set.
    def test_captured_call4_text_is_not_a_result(self):
        self.assertIsNone(find_tool_result([{"role": "assistant", "content": CALL4_ASSISTANT}], {"echo_tool"}))
        template_only = [{"role": "assistant", "content": TEMPLATE}]
        self.assertIsNone(find_tool_result(template_only, {"rag_lookup"}))
        agent, task = fake([echo_tool], "Is APT-007 confirmed?")
        reply = MockLLM(model="mock-llm").call([{"role": "assistant", "content": CALL4_ASSISTANT}],
                                               from_task=task, from_agent=agent)
        self.assertNotIn("the result of the action", reply)
        self.assertNotIn("I tried reusing", reply)

    # Pitfall 2: arguments come from the tool's declared schema, not from its name.
    # The query argument carries only the policy clause of a combined question.
    def test_arguments_follow_the_tool_schema(self):
        question = "What is the status of APT-007 and can I reschedule it?"
        agent, task = fake([crew.rag_lookup_tool], question)
        reply = MockLLM(model="mock-llm").call([], from_task=task, from_agent=agent)
        self.assertIn("Action: rag_lookup", reply)
        self.assertEqual(json.loads(reply.split("Action Input:")[1]), {"query": "can I reschedule it?"})
        agent, task = fake([misleading_lookup_tool], question)
        reply = MockLLM(model="mock-llm").call([], from_task=task, from_agent=agent)
        self.assertEqual(json.loads(reply.split("Action Input:")[1]), {"record_id": "APT-007"})

    # The policy clause drops every "and"-separated part that names an appointment id, and
    # leaves a question with no id unchanged even when it contains "and".
    def test_policy_clause_drops_parts_with_record_ids(self):
        cases = {"What is the status of APT-007 and can I reschedule it?": "can I reschedule it?",
                 "Is APT-007 cancelled and can I reschedule?": "can I reschedule?",
                 "What is the cancellation and rescheduling window?":
                     "What is the cancellation and rescheduling window?"}
        for question, expected in cases.items():
            with self.subTest(question=question):
                self.assertEqual(policy_clause(question), expected)
        arguments, reason = tool_arguments(crew.rag_lookup_tool, "What is the status of APT-007?")
        self.assertIsNone(arguments)
        self.assertEqual(reason, NO_POLICY_QUESTION)

    # Importing tools.py (as Task 6 does) must not load the retrieval code in rag.py.
    def test_importing_tools_does_not_import_rag(self):
        root = Path(__file__).resolve().parent.parent
        check = subprocess.run([sys.executable, "-c", "import sys, tools; print('rag' in sys.modules)"],
                               cwd=root, capture_output=True, text=True, check=True)
        self.assertEqual(check.stdout.strip(), "False")

    # A missing agent or task (CrewAI's forced final call) gives a plain final answer.
    def test_missing_agent_gives_plain_final_answer(self):
        reply = MockLLM(model="mock-llm").call([{"role": "user", "content": "x"}])
        self.assertIn("Final Answer:", reply)

    # The ReAct path is the one being tested: no native function calling, tools=None.
    def test_react_path_is_used(self):
        self.assertFalse(MockLLM(model="mock-llm").supports_function_calling())
        for result in self.results.values():
            self.assertTrue(all(call["tools"] is None for call in result["calls"]))

    # Every call had a real agent and task (no forced-final call), and each tool counter
    # equals exactly its expected count for the three demo questions.
    def test_tool_counts_for_the_demo_questions(self):
        for question, expected in QUERIES.items():
            with self.subTest(question=question):
                result = self.results[question]
                self.assertTrue(all(call["agent"] and call["task"] for call in result["calls"]))
                self.assertEqual(result["counts"], expected)

    # The record part belongs to the Lookup Agent: retrieval for the combined question must
    # receive only the policy clause, never the appointment id.
    def test_retrieval_query_excludes_the_record_id(self):
        queries = self.results["What is the status of APT-007 and can I reschedule it?"]["rag_queries"]
        self.assertEqual(len(queries), 1)
        self.assertNotIn("APT-007", queries[0])

    # Drafts carry real policy text or real record fields, never template text.
    def test_drafts_contain_real_content_and_no_placeholders(self):
        for question, result in self.results.items():
            with self.subTest(question=question):
                for text in PLACEHOLDERS:
                    self.assertNotIn(text, result["draft"])
                # Compare with retrieval for the exact text the Retrieval Agent sent.
                for sent in result["rag_queries"]:
                    self.assertIn(rag_lookup(sent)["answer"], result["draft"])
                if QUERIES[question]["check_appointment_status"]:
                    self.assertIn("record_id: APT-007; status: Cancelled", result["draft"])
                    self.assertIn("(KB08)", result["draft"])

    # Only the Lookup Agent can read appointment records.
    def test_only_lookup_agent_has_check_appointment_status(self):
        agents = next(iter(self.results.values()))["crew"].agents
        owners = [agent.role for agent in agents
                  if "check_appointment_status" in [item.name for item in agent.tools]]
        self.assertEqual(owners, ["Lookup Agent"])

    # Importing crew.py leaves CrewAI's storage inside the repo and the telemetry flags set.
    def test_storage_dir_and_flags_after_importing_crew(self):
        root = Path(__file__).resolve().parent.parent
        storage = Path(os.environ["CREWAI_STORAGE_DIR"]).resolve()
        self.assertEqual(storage, root / ".crewai")
        for name, value in settings.FLAGS.items():
            self.assertEqual(os.environ[name], value)


if __name__ == "__main__":
    unittest.main()
