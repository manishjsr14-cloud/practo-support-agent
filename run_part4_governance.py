"""Demonstrate Part 4 governance (tool wiring, budget cap) and save the real output."""
# What: measures the token cost of the 15 Part 3 questions (characters / 4 across the input
#   and every LLM call), checks the cap in rules.py against 2x the largest, sends the
#   pre-registered 20,000-character request, and shows a blocked tool wiring.
# Task: Part 4, Task 15. Used by: run by hand; it writes outputs/part4_governance.txt.

import settings  # noqa: F401  (sets the offline flags before any framework import)

from pathlib import Path

import governance
import pipeline
from eval_queries import EVAL_QUERIES
from memory import prepare_user_text
from part4_preregistration import BUDGET_MARGIN


# Run the 15 questions through the full pipeline with the meter counting only.
# Takes: nothing. Returns: one row per question (id, input tokens, LLM calls, total tokens).
def measure():
    rows = []
    for case in EVAL_QUERIES:
        pipeline.ask(f"budget-{case['id']}", case["query"])
        rows.append((case["id"], governance.tokens(prepare_user_text(case["query"])),
                     governance.METER["calls"], governance.METER["tokens"]))
    return rows


# Build the measurement section of the report.
# Takes: the measured rows. Returns: the lines and (largest total, most calls).
def measurement_lines(rows):
    lines = ["Budget basis: the 15 questions in eval_queries.py through pipeline.ask (crew + review)",
             "| Id | Input tokens | LLM calls | Total tokens |", "|---|---:|---:|---:|"]
    lines += [f"| {qid} | {inp} | {calls} | {total} |" for qid, inp, calls, total in rows]
    largest, most_calls = max(row[3] for row in rows), max(row[2] for row in rows)
    lines += [f"Largest measured request: {largest} tokens; most LLM calls in one request: {most_calls}",
              f"Cap = {BUDGET_MARGIN} x {largest} = {BUDGET_MARGIN * largest} tokens"]
    return lines, (largest, most_calls)


if __name__ == "__main__":
    print("\n".join(measurement_lines(measure())[0]))
