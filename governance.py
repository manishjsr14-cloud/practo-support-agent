"""Governance checks: least-autonomy tool wiring and a per-request token budget."""
# What: check_tool_wiring() refuses any crew in which an agent other than the Lookup Agent
#   holds check_appointment_status. The meter counts one request's tokens (characters / 4)
#   across the input and every LLM call, as pre-registered in part4_preregistration.py.
# Task: Part 4, Task 15. Used by: crew.py, mock_llm.py, review.py, pipeline.py, the Part 4
#   runners and tests.

import math

from part4_preregistration import CHARS_PER_TOKEN

# The running cost of the current request: tokens so far, LLM calls so far, and the cap
#   (None while measuring, so nothing is refused).
METER = {"tokens": 0, "calls": 0, "cap": None}

# The one tool that reads appointment records, and the one agent allowed to hold it.
RECORD_TOOL = "check_appointment_status"
RECORD_AGENT = "Lookup Agent"


class ToolWiringError(Exception):
    """An agent other than the Lookup Agent was given the record tool."""


class BudgetExceeded(Exception):
    """The current request's running token count passed the cap."""


# Check that only the Lookup Agent can read appointment records (least autonomy).
# Takes: the crew's agents. Returns: nothing; raises ToolWiringError naming the agent.
# Why: CrewAI itself lets any agent hold any tool; this check runs on every crew build,
#   so a wiring mistake stops the crew before it runs instead of leaking records.
def check_tool_wiring(agents):
    for agent in agents:
        names = {tool.name for tool in (agent.tools or [])}
        if RECORD_TOOL in names and agent.role != RECORD_AGENT:
            raise ToolWiringError(f"{agent.role} may not hold {RECORD_TOOL}; only the {RECORD_AGENT} may.")


# Estimate tokens the pre-registered way.
# Takes: text. Returns: ceil(characters / 4).
def tokens(text):
    return math.ceil(len(text) / CHARS_PER_TOKEN)


# Join a model call's prompt messages into one text, whatever their form.
# Takes: a string, CrewAI's message dicts or AutoGen's message objects. Returns: the text.
def prompt_text(messages):
    if isinstance(messages, str):
        return messages
    return "".join(str(m.get("content") or "") if isinstance(m, dict) else str(m.content) for m in messages)


# Start metering one request: the masked input is charged once.
# Takes: the input text and the cap (None = only count). Returns: nothing.
def start(input_text, cap=None):
    METER.update({"tokens": tokens(input_text), "calls": 0, "cap": cap})


# Charge text to the current request; a prompt also counts as one LLM call.
# Takes: the text and whether it is a call's prompt. Returns: nothing; raises
#   BudgetExceeded the moment the running total passes the cap, so it never goes over silently.
def charge(text, call=False):
    METER["tokens"] += tokens(text)
    METER["calls"] += 1 if call else 0
    if METER["cap"] is not None and METER["tokens"] > METER["cap"]:
        raise BudgetExceeded(f"Budget: this request used {METER['tokens']} tokens, over the cap of {METER['cap']}.")
