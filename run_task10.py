"""Show each Task 10 guardrail firing on its deliberate case and save the real output."""
# What: phone masking (raw vs masked, and what history stores), injection blocking,
#   Stage 1 and Stage 2 refusals, a mixed question whose policy part is refused, and a
#   record-only question with a found and a missing record.
# Task: Part 2, Task 10. Used by: run by hand; it writes outputs/part2_guardrails_demo.txt.

import settings  # noqa: F401  (sets the offline flags before crewai is imported)

from pathlib import Path

from evaluate_guardrail import describe_turn
from guardrail_queries import INJECTION_ATTACK, PII_EXAMPLE
from memory import prepare_user_text, stored_history
from pipeline import LAST_RUN, ask
from queries import DEMO_OUT_OF_SCOPE

# Each case: (title, session id, question).
CASES = (
    ("Stage 1: similarity gate (Part 1, unchanged)", "demo-stage1", DEMO_OUT_OF_SCOPE),
    ("Stage 2: unsupported question terms", "demo-stage2", "What are the clinic's opening hours on Sundays?"),
    ("Mixed: record found, policy part refused", "demo-mixed",
     "What is the status of APT-007 and what are the clinic's opening hours on Sundays?"),
    ("Record-only: record found", "demo-record", "What is the status of APT-007?"),
    ("Record-only: no such record", "demo-missing", "What is the status of APT-999?"),
)


# Run every deliberate case and build the report text.
# Takes: nothing. Returns: the report as one string.
def make_report():
    lines = ["Task 10: each guardrail on its deliberate case (full pipeline, MockLLM, offline)", "",
             "=== Input: phone masking (KB09)", f"raw:    {PII_EXAMPLE!r}",
             f"masked: {prepare_user_text(PII_EXAMPLE)!r}"]
    response = ask("demo-pii", PII_EXAMPLE)
    lines += [f"record_id looked up: {response.record_id}", "stored history for 'demo-pii':"]
    lines += [f"  [{kind}] {text.splitlines()[0]}" for kind, text in stored_history("demo-pii")]
    response = ask("demo-injection", INJECTION_ATTACK)
    lines += ["", "=== Input: prompt injection", f"question: {INJECTION_ATTACK!r}",
              f"crew ran: {LAST_RUN['result'] is not None}",
              f"refused={response.refused}; tools_used={response.tools_used}; reason: {response.refusal_reason}"]
    for title, session_id, question in CASES:
        response = ask(session_id, question)
        lines += ["", f"=== Output: {title}", f"question: {question!r}"] + describe_turn(response)
        lines += [f"  record_id={response.record_id}; sources={response.sources}",
                  "  answer:"] + [f"    {line}" for line in response.answer.splitlines()]
    return "\n".join(lines) + "\n"


# Save the report to outputs/part2_guardrails_demo.txt and print it.
def main():
    report = make_report()
    target = Path(__file__).resolve().parent / "outputs" / "part2_guardrails_demo.txt"
    target.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
