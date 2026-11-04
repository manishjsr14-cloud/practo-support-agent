"""Demonstrate the Part 4 review stage on the pre-registered drafts and save the real output."""
# What: for each case in part4_preregistration.py, runs the crew, adds the injected sentence
#   where the case says so, runs the AutoGen review team and prints the reviewer's list, the
#   editor's VerdictModel and the model-call counts; then shows the stage inside pipeline.ask.
# Task: Part 4, Task 14. Used by: run by hand; it writes outputs/part4_review.txt.

import settings  # noqa: F401  (sets the offline flags before any framework import)

from pathlib import Path

import crew
import pipeline
import review
from part4_preregistration import REVIEW_CASES
from schema import CrewResponse


# Run every case and build the report text.
# Takes: nothing. Returns: the report as one string.
def make_report():
    lines = ["Part 4, Task 14: AutoGen review stage (offline ReviewClient, zero API keys)",
             f"Team: RoundRobinGroupChat([{review.REVIEWER}, {review.EDITOR}], max_turns=2, "
             "custom_message_types=[StructuredMessage[VerdictModel]])",
             f"VerdictModel fields: {list(review.VerdictModel.model_fields)}"]
    rows = []
    for case in REVIEW_CASES:
        result = crew.run_crew(case["question"], output_model=CrewResponse)
        draft = result["pydantic"].answer
        given = f"{draft} {case['inject']}" if case["inject"] else draft
        verdict, messages = review.review_draft(given, review.crew_context(result))
        got = "approved unchanged" if verdict.approved and verdict.final_answer == given else "revised"
        lines += ["", "=" * 72, f"[{case['id']}] {case['question']}",
                  f"Injected sentence: {case['inject']!r}", "--- Draft given to the review team"]
        lines += [f"  {line}" for line in given.splitlines()]
        lines += [f"--- {messages[1].source} reply: {messages[1].content}",
                  f"--- {messages[2].source} verdict ({type(messages[2]).__name__}):",
                  f"  approved: {verdict.approved}", f"  reason: {verdict.reason}",
                  f"  final_answer equals the given draft: {verdict.final_answer == given}",
                  f"  final_answer equals the crew's real draft: {verdict.final_answer == draft}",
                  f"Model calls: {dict(review.CALLS)}; team messages: {len(messages)} "
                  f"(task, reviewer, editor)", f"Expected: {case['expect']}; got: {got}"]
        rows.append(f"| {case['id']} | {case['expect']} | {got} | {verdict.approved} |")
    # The same stage inside the full pipeline (default path, no injection).
    response = pipeline.ask("part4-review", REVIEW_CASES[0]["question"])
    verdict = pipeline.LAST_RUN["review"]
    lines += ["", "=" * 72, f"Inside pipeline.ask: {REVIEW_CASES[0]['question']}",
              f"  review approved: {verdict.approved}; reason: {verdict.reason}",
              f"  model calls: {dict(review.CALLS)}; grounded: {response.grounded}; sources: {response.sources}",
              "", "| Case | Expected | Got | approved |", "|---|---|---|---|"] + rows
    return "\n".join(lines) + "\n"


# Save the report to outputs/part4_review.txt and print it.
def main():
    report = make_report()
    target = Path(__file__).resolve().parent / "outputs" / "part4_review.txt"
    target.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
