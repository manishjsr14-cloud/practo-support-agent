"""Three-agent sequential CrewAI crew for appointment and policy questions."""
# What: a Retrieval Agent (policy text), a Lookup Agent (appointment record) and a
#   Response Composer (joins both), all driven by the offline MockLLM. Each tool call is
#   counted, so saved outputs can prove which tools ran for each question.
# Task: Part 2, Task 7. Used by: run_task7.py and tests/test_task7.py.
# It wraps rag_lookup() and check_appointment_status() from tools.py as CrewAI tools.

import settings  # noqa: F401  (sets the offline flags before crewai is imported)

from crewai import Agent, Crew, Process, Task
from crewai.tools import tool

from mock_llm import MockLLM
from tools import check_appointment_status, rag_lookup

# How many times each tool ran during the current run_crew() call.
COUNTS = {"rag_lookup": 0, "check_appointment_status": 0}


@tool("rag_lookup")
def rag_lookup_tool(query: str) -> str:
    """Quote the clinic policy text that best matches a policy question."""
    # Count first, then return the retrieved text, its sources and its top-1 similarity.
    COUNTS["rag_lookup"] += 1
    result = rag_lookup(query)
    return (f"{result['answer']}\nSources: {', '.join(result['source_ids']) or '(none)'}\n"
            f"Top-1 similarity: {result['similarity']:.9f}")


@tool("check_appointment_status")
def check_appointment_status_tool(record_id: str) -> str:
    """Return the recorded fields of one appointment, given an id such as APT-001."""
    # Count first, then return the stored fields (or the controlled error) as text.
    COUNTS["check_appointment_status"] += 1
    return "; ".join(f"{key}: {value}" for key, value in check_appointment_status(record_id).items())


# Build the three agents and their tasks around one LLM.
# Takes: the LLM to use. Returns: a sequential Crew.
# Why: each agent gets only the tool it needs, so the Lookup Agent alone can read records.
def build_crew(llm):
    # Same safe settings for every agent: no delegation, no tool-result cache (so the
    # counters see every real call), and few steps so a mistake fails quickly.
    common = {"llm": llm, "allow_delegation": False, "cache": False, "max_iter": 3, "verbose": False}
    retrieval = Agent(role="Retrieval Agent", tools=[rag_lookup_tool],
                      goal="Quote the clinic policy that answers the customer's policy question.",
                      backstory="You report retrieved policy text only.", **common)
    lookup = Agent(role="Lookup Agent", tools=[check_appointment_status_tool],
                   goal="Report the stored record for the appointment id the customer gives.",
                   backstory="You report recorded appointment fields only.", **common)
    composer = Agent(role="Response Composer", tools=[],
                     goal="Combine the earlier results into one draft reply without adding facts.",
                     backstory="You never add information that the tools did not return.", **common)
    # "{question}" is filled in by kickoff(); the "Customer question:" line is where the
    # MockLLM reads the question from.
    retrieval_task = Task(name="Retrieval", agent=retrieval,
                          description="Customer question: {question}\nLook up the clinic policy if one is asked about.",
                          expected_output="Quoted policy text with sources, or a statement that no policy question was asked.")
    lookup_task = Task(name="Lookup", agent=lookup,
                       description="Customer question: {question}\nLook up the appointment record if an id is given.",
                       expected_output="The recorded appointment fields, or a statement that no record id was given.")
    compose_task = Task(name="Compose", agent=composer, context=[retrieval_task, lookup_task],
                        description="Customer question: {question}\nCombine the earlier results into one draft reply.",
                        expected_output="A draft reply containing only the earlier results.")
    return Crew(agents=[retrieval, lookup, composer], tasks=[retrieval_task, lookup_task, compose_task],
                process=Process.sequential, cache=False, memory=False, verbose=False)


# Run the crew for one question with fresh counters and a fresh MockLLM.
# Takes: the customer's question. Returns: each task's output, the final draft, the tool
#   counts, the MockLLM's call log and the crew (for inspection in tests).
def run_crew(question):
    for name in COUNTS:
        COUNTS[name] = 0
    llm = MockLLM(model="mock-llm")
    crew = build_crew(llm)
    result = crew.kickoff(inputs={"question": question})
    return {"question": question, "outputs": {task.name: task.output.raw for task in crew.tasks},
            "draft": result.raw, "counts": dict(COUNTS), "calls": list(llm.call_log), "crew": crew}
