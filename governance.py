"""Governance checks: least-autonomy tool wiring and a per-request token budget."""
# What: check_tool_wiring() refuses any crew in which an agent other than the Lookup Agent
#   holds check_appointment_status.
# Task: Part 4, Task 15. Used by: crew.py, pipeline.py, the Part 4 runners and tests.

# The one tool that reads appointment records, and the one agent allowed to hold it.
RECORD_TOOL = "check_appointment_status"
RECORD_AGENT = "Lookup Agent"


class ToolWiringError(Exception):
    """An agent other than the Lookup Agent was given the record tool."""


# Check that only the Lookup Agent can read appointment records (least autonomy).
# Takes: the crew's agents. Returns: nothing; raises ToolWiringError naming the agent.
# Why: CrewAI itself lets any agent hold any tool; this check runs on every crew build,
#   so a wiring mistake stops the crew before it runs instead of leaking records.
def check_tool_wiring(agents):
    for agent in agents:
        names = {tool.name for tool in (agent.tools or [])}
        if RECORD_TOOL in names and agent.role != RECORD_AGENT:
            raise ToolWiringError(f"{agent.role} may not hold {RECORD_TOOL}; only the {RECORD_AGENT} may.")
