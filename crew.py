"""Three-agent sequential CrewAI crew for appointment and policy questions."""
# What: a Retrieval Agent (policy text), a Lookup Agent (appointment record) and a
#   Response Composer (joins both), all driven by the offline MockLLM. Each tool call is
#   counted, so saved outputs can prove which tools ran for each question.
# Task: Part 2, Task 7; Task 8-9 added the optional session record id and output model.
#   Used by: run_task7.py, pipeline.py and the tests.
# It wraps rag_lookup() and check_appointment_status() from tools.py as CrewAI tools.

import settings  # noqa: F401  (sets the offline flags before crewai is imported)

from crewai import Agent, Crew, Process, Task
from crewai.tools import tool

from mock_llm import MockLLM
from tools import check_appointment_status, rag_lookup

# How many times each tool ran during the current run_crew() call.
COUNTS = {"rag_lookup": 0, "check_appointment_status": 0}
# The exact text sent to rag_lookup during the current run, so reports can show it.
RAG_QUERIES = []
# The full results the tools returned during the current run (Task 9 builds the
#   structured reply's sources, record id and escalation flag from these).
RAG_RESULTS = []
LOOKUPS = []


@tool("rag_lookup")
def rag_lookup_tool(query: str) -> str:
    """Quote the clinic policy text that best matches a policy question."""
    # Count first, then return the retrieved text, its sources and its top-1 similarity.
    COUNTS["rag_lookup"] += 1
    RAG_QUERIES.append(query)
    result = rag_lookup(query)
    RAG_RESULTS.append(result)
    return (f"{result['answer']}\nSources: {', '.join(result['source_ids']) or '(none)'}\n"
            f"Top-1 similarity: {result['similarity']:.9f}")


@tool("check_appointment_status")
def check_appointment_status_tool(record_id: str) -> str:
    """Return the recorded fields of one appointment, given an id such as APT-001."""
    # Count first, then return the stored fields (or the controlled error) as text.
    COUNTS["check_appointment_status"] += 1
    record = check_appointment_status(record_id)
    LOOKUPS.append(record)
    return "; ".join(f"{key}: {value}" for key, value in record.items())


# Build the three agents and their tasks around one LLM.
# Takes: the LLM, an optional Pydantic model for the final task's output (Task 9) and an
#   optional record id remembered from this session (Task 8). Returns: a sequential Crew.
# Why: each agent gets only the tool it needs, so the Lookup Agent alone can read records.
#   With neither option the crew is exactly Task 7's.
def build_crew(llm, output_model=None, session_record_id=None):
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
    # The remembered id goes on the Lookup task only, so retrieval never sees it.
    lookup_text = "Customer question: {question}\nLook up the appointment record if an id is given."
    if session_record_id:
        lookup_text += f"\nSession record id: {session_record_id}"
    lookup_task = Task(name="Lookup", agent=lookup, description=lookup_text,
                       expected_output="The recorded appointment fields, or a statement that no record id was given.")
    # output_pydantic makes CrewAI check the Composer's JSON against the model (Task 9).
    compose_task = Task(name="Compose", agent=composer, context=[retrieval_task, lookup_task],
                        description="Customer question: {question}\nCombine the earlier results into one draft reply.",
                        expected_output="A draft reply containing only the earlier results.",
                        output_pydantic=output_model)
    return Crew(agents=[retrieval, lookup, composer], tasks=[retrieval_task, lookup_task, compose_task],
                process=Process.sequential, cache=False, memory=False, verbose=False)


# Run the crew for one question with fresh counters and a fresh MockLLM.
# Takes: the customer's question, plus build_crew()'s two optional settings. Returns: each
#   task's output, the final draft, the tool counts, the text sent to rag_lookup, the tool
#   results, CrewAI's Pydantic output, the MockLLM's call log and the crew (for tests).
def run_crew(question, output_model=None, session_record_id=None):
    for name in COUNTS:
        COUNTS[name] = 0
    for log in (RAG_QUERIES, RAG_RESULTS, LOOKUPS):
        log.clear()
    llm = MockLLM(model="mock-llm")
    crew = build_crew(llm, output_model, session_record_id)
    result = crew.kickoff(inputs={"question": question})
    return {"question": question, "outputs": {task.name: task.output.raw for task in crew.tasks},
            "draft": result.raw, "counts": dict(COUNTS), "rag_queries": list(RAG_QUERIES),
            "rag_results": list(RAG_RESULTS), "lookups": list(LOOKUPS), "pydantic": result.pydantic,
            "calls": list(llm.call_log), "crew": crew}
