"""Run the Task 7 crew on three fixed questions and save the real output."""
# What: runs crew.run_crew() for a policy-only, a record-only and a combined question,
#   and saves every agent output, the final draft and the tool counters.
# Task: Part 2, Task 7. Used by: run by hand; it writes outputs/part2_crew.txt.

import settings  # noqa: F401  (sets the offline flags before crewai is imported)

import os
from pathlib import Path

from crewai_core.paths import db_storage_path

from crew import run_crew

QUERIES = (
    "What is the cancellation and rescheduling window?",
    "What is the status of APT-007?",
    "What is the status of APT-007 and can I reschedule it?",
)


# Run all three questions and build the report text.
# Takes: nothing. Returns: the report as one string.
# Why: every Task 7 number in the README must come from this real output.
def make_report():
    # Show the flags and CrewAI's storage folder, as proof that telemetry and tracing were
    # off and that CrewAI's files stayed inside the repo.
    lines = ["Task 7: three-agent sequential crew with MockLLM (offline, zero API keys)",
             "Flags set by settings.py before crewai was imported:"]
    lines += [f"  {name}={os.environ[name]}" for name in settings.FLAGS]
    lines.append(f"CrewAI storage path: {db_storage_path()}")
    rows = []
    for question in QUERIES:
        result = run_crew(question)
        counts, calls = result["counts"], result["calls"]
        # Check every LLM call came from a real agent and task, on the ReAct path.
        complete = all(call["agent"] and call["task"] for call in calls)
        react = all(call["tools"] is None for call in calls)
        lines += ["", "=" * 72, f"Question: {question}",
                  f"Tool calls: rag_lookup={counts['rag_lookup']}; "
                  f"check_appointment_status={counts['check_appointment_status']}",
                  "Text sent to rag_lookup: " + (" | ".join(repr(q) for q in result["rag_queries"]) or "(none)"),
                  f"LLM calls: {len(calls)}; all with agent and task: {complete}; "
                  f"tools argument always None (ReAct path): {react}",
                  "--- Retrieval Agent output", result["outputs"]["Retrieval"],
                  "--- Lookup Agent output", result["outputs"]["Lookup"],
                  "--- Response Composer output (final draft)", result["draft"]]
        rows.append(f"| {question} | {counts['rag_lookup']} | {counts['check_appointment_status']} |")
    lines += ["", "| Question | rag_lookup | check_appointment_status |", "|---|---:|---:|"] + rows
    return "\n".join(lines) + "\n"


# Save the report to outputs/part2_crew.txt and print it.
def main():
    report = make_report()
    target = Path(__file__).resolve().parent / "outputs" / "part2_crew.txt"
    target.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
