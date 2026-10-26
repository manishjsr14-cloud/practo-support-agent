"""Run two short conversations through the pipeline and save the real output."""
# What: session "A" asks about APT-007, then "its fee" (remembered id), then a policy
#   question; a new session "B" asks "its fee" with nothing to remember. Every turn's
#   validated CrewResponse and each session's stored history are printed, plus one
#   invalid payload being rejected.
# Task: Part 2, Tasks 8-9. Used by: run by hand; it writes outputs/part2_memory.txt.

import settings  # noqa: F401  (sets the offline flags before crewai is imported)

from pathlib import Path

from memory import stored_history
from pipeline import LAST_RUN, ask
from schema import validate_response

TRANSCRIPTS = (
    ("A", ("What is the status of APT-007?",
           "What is its fee?",
           "What is the cancellation and rescheduling window?")),
    ("B", ("What is its fee?",)),
)
# A reply with a wrong type, a missing field and an unknown field.
INVALID = {"answer": "Your fee is INR 500.", "sources": "03_consultation_fees", "tools_used": [],
           "record_id": None, "escalation_recommended": None, "grounded": True, "refused": False,
           "urgency": "high"}


# Run both transcripts and build the report text.
# Takes: nothing. Returns: the report as one string.
# Why: every Task 8-9 statement in the README must come from this real output.
def make_report():
    lines = ["Tasks 8-9: session memory and structured output (MockLLM, offline, zero API keys)"]
    for session_id, turns in TRANSCRIPTS:
        lines += ["", "=" * 72, f"Transcript {session_id} (session_id={session_id!r})"]
        for number, text in enumerate(turns, start=1):
            response = ask(session_id, text)
            result = LAST_RUN["result"]
            counts, calls = result["counts"], result["calls"]
            lines += ["", f"--- Turn {session_id}{number}: {text}",
                      f"Remembered record id passed to the Lookup task: {LAST_RUN['remembered_record_id']}",
                      f"Tool calls: rag_lookup={counts['rag_lookup']}; "
                      f"check_appointment_status={counts['check_appointment_status']}",
                      "Text sent to rag_lookup: " + (" | ".join(repr(q) for q in result["rag_queries"]) or "(none)"),
                      f"LLM calls: {len(calls)}; all with agent and task: "
                      f"{all(call['agent'] and call['task'] for call in calls)}",
                      f"CrewAI output_pydantic type before filling: {type(result['pydantic']).__name__}",
                      "Validated CrewResponse:", response.model_dump_json(indent=2)]
    for session_id, _ in TRANSCRIPTS:
        lines += ["", f"Stored history for session {session_id!r}:"]
        lines += [f"  [{kind}] {text}" for kind, text in stored_history(session_id)]
    lines += ["", "Invalid payload:", repr(INVALID)]
    try:
        validate_response(INVALID)
        lines.append("NOT REJECTED (this is a failure)")
    except ValueError as error:
        lines += ["Rejected:", str(error)]
    return "\n".join(lines) + "\n"


# Save the report to outputs/part2_memory.txt and print it.
def main():
    report = make_report()
    target = Path(__file__).resolve().parent / "outputs" / "part2_memory.txt"
    target.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
