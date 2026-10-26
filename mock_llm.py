"""Deterministic offline stand-in for an LLM that speaks CrewAI's ReAct text format."""
# What: a fake "language model" for CrewAI. It never calls an API; it decides with plain
#   rules whether to call the agent's tool, and returns the tool's result word for word.
# Task: Part 2, Task 7; Task 8-9 added the session record id and the JSON reply.
#   Used by: crew.py, and the tests (tests/test_task7.py tests it directly).
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
# An appointment record id such as APT-007.
RECORD_ID = re.compile(r"APT-[0-9]{3}")


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


# Read the record id memory.py remembered for this session, if the task carries one (Task 8).
# Takes: a CrewAI task. Returns: an id such as "APT-007", or None.
# Why: only the Lookup task gets this line, so only the Lookup Agent can use it.
def session_record_id(task):
    match = re.search(r"^Session record id:[ \t]*(APT-[0-9]{3})[ \t]*$", task.description or "", re.M)
    return match.group(1) if match else None


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
# Takes: a CrewAI tool, the question and an optional remembered session id. Returns:
#   (arguments dict, None) to call the tool, or (None, reason) when the question gives
#   the tool nothing to work on (pitfall 2).
def tool_arguments(tool, question, session_id=None):
    fields = tool.args_schema.model_fields
    if "record_id" in fields:
        # An appointment tool needs an id such as APT-007 written in the question, or
        # failing that, the id remembered from earlier in this session.
        found = RECORD_ID.findall(question) or ([session_id] if session_id else [])
        return ({"record_id": found[0]}, None) if found else (None, NO_RECORD_ID)
    if "query" in fields:
        # Only the policy clause goes to retrieval: the record part belongs to the Lookup
        # Agent. If no words remain, there is no policy question.
        clause = policy_clause(question)
        return ({"query": clause}, None) if re.search(r"[A-Za-z]", clause) else (None, NO_POLICY_QUESTION)
    return None, UNSUPPORTED_TOOL


# Keep only the parts of the question that do not mention an appointment id.
# Takes: the question. Returns: the policy part ("" or punctuation if none remains).
# Rule: split on the whole word "and" (any case), drop every part containing an id such
#   as APT-007, and rejoin the rest with " and ".
# Examples: "What is the status of APT-007 and can I reschedule it?" -> "can I reschedule it?"
#   "Is APT-007 cancelled and can I reschedule?" -> "can I reschedule?"
def policy_clause(question):
    parts = re.split(r"\band\b", question, flags=re.I)
    kept = [part.strip() for part in parts if not RECORD_ID.search(part)]
    # Nothing dropped (no id): return the question unchanged, so an "and" inside a policy
    # question ("cancellation and rescheduling") never alters it.
    if len(kept) == len(parts):
        return question
    return " and ".join(part for part in kept if part).strip(" ,;")


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


# Wrap the composer's draft as the JSON CrewAI checks against the task's output model.
# Takes: the draft text. Returns: a JSON string with the draft as "answer".
# Why: CrewAI only accepts the model when the reply is JSON; plain text would make it call
#   the LLM again without an agent. The mock fills no facts: sources, record id and
#   escalation stay empty for pipeline.py to fill from the tools' own results.
def json_reply(draft):
    return json.dumps({"answer": draft, "sources": [], "tools_used": [], "record_id": None,
                       "escalation_recommended": None, "grounded": False, "refused": False,
                       "refusal_reason": None})


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
                              "task": getattr(from_task, "name", None), "tools": tools,
                              "response_model": getattr(response_model, "__name__", None)})
        # CrewAI's forced last call after too many steps has no agent or task: refuse
        # plainly instead of crashing or inventing an answer.
        if from_agent is None or from_task is None:
            return final(NO_CONTEXT)
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        agent_tools = list(from_agent.tools or [])
        # The composer has no tools: it only joins earlier results. When CrewAI asks for a
        # structured reply (response_model set), the same draft goes out as JSON.
        if not agent_tools:
            draft = compose(from_task)
            return json_reply(draft) if response_model is not None else final(draft)
        # If this agent's tool already ran, its result is the answer, word for word.
        result = find_tool_result(messages, {tool.name for tool in agent_tools})
        if result is not None:
            return final(result)
        # Otherwise call the first tool the question gives arguments for.
        question = customer_question(from_task)
        reasons = []
        for tool in agent_tools:
            arguments, reason = tool_arguments(tool, question, session_record_id(from_task))
            if arguments is not None:
                return (f"Thought: The question needs {tool.name}.\nAction: {tool.name}\n"
                        f"Action Input: {json.dumps(arguments)}")
            reasons.append(reason)
        return final(" ".join(reasons))
