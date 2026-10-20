"""Deterministic offline stand-in for an LLM that speaks CrewAI's ReAct text format."""
# What: a fake "language model" for CrewAI. It never calls an API; it decides with plain
#   rules whether to call the agent's tool, and returns the tool's result word for word.
# Task: Part 2, Task 7. Used by: crew.py, and tests/test_task7.py (which tests it directly).
# Facts it relies on were read from the installed crewai 1.15.28 source and a captured run:
#   tool results come back appended to the model's own assistant message as
#   "\nObservation: <result>", and call() receives tools=None on the ReAct path.

import settings  # noqa: F401  (sets the offline flags before crewai is imported)

import json
import re

from crewai.llms.base_llm import BaseLLM
from pydantic import Field

# Fixed replies, so every outcome is a plain, checkable sentence.
NO_RECORD_ID = "No appointment record id was given in the question."
NO_POLICY_QUESTION = "No policy question was asked."
NO_CONTEXT = "No tool result is available, so no answer can be given."
UNSUPPORTED_TOOL = "The mock cannot fill in this tool's arguments."
KB08_NOTE = ("Note: the escalation score is an administrative follow-up flag, "
             "not a medical assessment (KB08).")
# CrewAI's notice when the same tool input is repeated; it is not a real tool result.
REUSE_NOTICE = "I tried reusing the same input"
# One ReAct step: the action name, its JSON input, then the observation CrewAI appended.
STEP = re.compile(r"Action:[ \t]*(.*?)\nAction Input:[ \t]*(.*?)\nObservation:[ \t]*(.*)", re.S)
# The appointment-status part of a question, e.g. "What is the status of APT-007".
STATUS_CLAUSE = re.compile(r"what is the status of APT-[0-9]{3}", re.I)


# Wrap text in CrewAI's final-answer format.
# Takes: the answer text. Returns: the ReAct text CrewAI parses as a finished answer.
def final(text):
    return f"Thought: I now know the final answer\nFinal Answer: {text}"


# Read the customer's question from the task description, never from the conversation.
# Takes: a CrewAI task. Returns: the text after "Customer question:", or "" if missing.
# Why: the conversation also holds prompts and tool output, which must not be mistaken
#   for the question.
def customer_question(task):
    match = re.search(r"^Customer question:[ \t]*(.+)$", task.description or "", re.M)
    return match.group(1).strip() if match else ""


# Find the result of the model's own most recent tool call, if that call really ran.
# Takes: the message list and the agent's own tool names. Returns: the result text or None.
# Why: the ReAct template itself contains "Observation: the result of the action", so
#   searching the whole conversation for "Observation:" would mistake the template for a
#   result (pitfall 1). Only assistant messages are checked, and only the latest one that
#   contains an action; later user messages ("Analyze the tool result...") are allowed.
def find_tool_result(messages, tool_names):
    for message in reversed(messages):
        content = str(message.get("content") or "")
        if message.get("role") != "assistant" or "Action:" not in content:
            continue
        match = STEP.search(content)
        # A real action names one of this agent's tools exactly...
        if match is None or match.group(1).strip() not in tool_names:
            return None
        # ...and its input is a JSON object, as the template requires.
        try:
            arguments = json.loads(match.group(2).strip())
        except ValueError:
            return None
        observation = match.group(3).strip()
        if not isinstance(arguments, dict) or not observation or observation.startswith(REUSE_NOTICE):
            return None
        return observation
    return None


# Choose a tool's arguments from its declared argument schema, never from its name.
# Takes: a CrewAI tool and the question. Returns: (arguments dict, None) to call the tool,
#   or (None, reason) when the question gives the tool nothing to work on (pitfall 2).
def tool_arguments(tool, question):
    fields = tool.args_schema.model_fields
    if "record_id" in fields:
        # An appointment tool needs an id such as APT-007 written in the question.
        found = re.findall(r"APT-[0-9]{3}", question)
        return ({"record_id": found[0]}, None) if found else (None, NO_RECORD_ID)
    if "query" in fields:
        # Only the policy clause goes to retrieval: the record part belongs to the Lookup
        # Agent. If no words remain, there is no policy question.
        clause = policy_clause(question)
        return ({"query": clause}, None) if re.search(r"[A-Za-z]", clause) else (None, NO_POLICY_QUESTION)
    return None, UNSUPPORTED_TOOL


# Remove the "status of APT-xxx" clause and the "and" that joined it to the rest.
# Takes: the question. Returns: the remaining policy clause ("" or punctuation if none).
# Example: "What is the status of APT-007 and can I reschedule it?" -> "can I reschedule it?"
# A question without a status clause is returned unchanged.
def policy_clause(question):
    clause = STATUS_CLAUSE.sub("", question).strip()
    # Drop a leading "and" (status clause first) or a trailing "and" (status clause last).
    clause = re.sub(r"^\W*and\b|,?\s*\band\W*$", "", clause, flags=re.I)
    return clause.strip(" ,;")


# Join the earlier agents' outputs under labels, in task order, adding no new facts.
# Takes: the composer's task, whose context lists the earlier tasks. Returns: the draft.
# Why: the composer must only pass on what the tools returned, fallbacks included.
def compose(task):
    earlier = task.context if isinstance(task.context, list) else []
    parts = [f"{item.name} result:\n{item.output.raw}" for item in earlier if item.output]
    if not parts:
        return NO_CONTEXT
    # A real lookup result carries an escalation score; explain what that score is not.
    if any("escalation_score" in item.output.raw for item in earlier if item.output):
        parts.append(KB08_NOTE)
    return "\n\n".join(parts)


class MockLLM(BaseLLM):
    """Rule-based LLM replacement used under MOCK_LLM."""

    llm_type: str = "mock"
    # One entry per call() so tests can check what CrewAI passed in.
    call_log: list = Field(default_factory=list)

    # Tell CrewAI there is no native function calling, so it always uses the ReAct
    # text loop that this mock implements (some CrewAI code calls this without checking).
    def supports_function_calling(self) -> bool:
        return False

    # Answer one CrewAI request with ReAct text.
    # Takes: the conversation plus CrewAI's optional arguments. Returns: either an
    #   "Action" for the agent's tool or a "Final Answer".
    def call(self, messages, tools=None, callbacks=None, available_functions=None,
             from_task=None, from_agent=None, response_model=None):
        self.call_log.append({"agent": getattr(from_agent, "role", None),
                              "task": getattr(from_task, "name", None), "tools": tools})
        # CrewAI's forced last call after too many steps has no agent or task: refuse
        # plainly instead of crashing or inventing an answer.
        if from_agent is None or from_task is None:
            return final(NO_CONTEXT)
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        agent_tools = list(from_agent.tools or [])
        # The composer has no tools: it only joins earlier results.
        if not agent_tools:
            return final(compose(from_task))
        # If this agent's tool already ran, its result is the answer, word for word.
        result = find_tool_result(messages, {tool.name for tool in agent_tools})
        if result is not None:
            return final(result)
        # Otherwise call the first tool the question gives arguments for.
        question = customer_question(from_task)
        reasons = []
        for tool in agent_tools:
            arguments, reason = tool_arguments(tool, question)
            if arguments is not None:
                return (f"Thought: The question needs {tool.name}.\nAction: {tool.name}\n"
                        f"Action Input: {json.dumps(arguments)}")
            reasons.append(reason)
        return final(" ".join(reasons))
