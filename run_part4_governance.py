"""Demonstrate Part 4 governance (tool wiring, budget cap) and save the real output."""
# What: measures the token cost of the 15 Part 3 questions (characters / 4 across the input
#   and every LLM call), checks the cap in rules.py against 2x the largest, sends the
#   pre-registered 20,000-character request through POST /ask, and shows a blocked tool wiring.
# Task: Part 4, Task 15. Used by: run by hand; it writes outputs/part4_governance.txt.

import settings  # noqa: F401  (sets the offline flags before any framework import)

import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from fastapi.testclient import TestClient

import api
import crew
import governance
import logging_utils
import pipeline
import rules
from eval_queries import EVAL_QUERIES
from memory import prepare_user_text
from part4_preregistration import BUDGET_MARGIN, OVERSIZED_REQUEST


# Run the 15 questions through the full pipeline with the meter counting only.
# Takes: nothing. Returns: one row per question (id, input tokens, LLM calls, total tokens).
def measure():
    rows = []
    governance.ENFORCE = False
    try:
        for case in EVAL_QUERIES:
            pipeline.ask(f"budget-{case['id']}", case["query"])
            rows.append((case["id"], governance.tokens(prepare_user_text(case["query"])),
                         governance.METER["calls"], governance.METER["tokens"]))
    finally:
        governance.ENFORCE = True
    return rows


# Build the measurement section and check rules.py against it.
# Takes: the measured rows. Returns: the report lines; raises SystemExit on a mismatch.
def measurement_lines(rows):
    lines = ["Budget basis: the 15 questions in eval_queries.py through pipeline.ask (crew + review)",
             "| Id | Input tokens | LLM calls | Total tokens | Projected |", "|---|---:|---:|---:|---:|"]
    largest, most_calls = max(row[3] for row in rows), max(row[2] for row in rows)
    lines += [f"| {qid} | {inp} | {calls} | {total} | {inp * most_calls + largest} |"
              for qid, inp, calls, total in rows]
    lines += [f"Largest measured request: {largest} tokens; most LLM calls in one request: {most_calls}",
              f"Cap = {BUDGET_MARGIN} x {largest} = {BUDGET_MARGIN * largest} tokens",
              f"rules.py: BUDGET_LARGEST_TOKENS={rules.BUDGET_LARGEST_TOKENS}, "
              f"BUDGET_MAX_CALLS={rules.BUDGET_MAX_CALLS}, BUDGET_TOKENS={rules.BUDGET_TOKENS}"]
    if (rules.BUDGET_LARGEST_TOKENS, rules.BUDGET_MAX_CALLS, rules.BUDGET_TOKENS) != \
            (largest, most_calls, BUDGET_MARGIN * largest):
        raise SystemExit("rules.py does not match this measurement; the cap was not set from it.")
    refused = [case["id"] for case in EVAL_QUERIES if governance.budget_reason(prepare_user_text(case["query"]))]
    lines.append(f"rules.py matches the measurement. Questions refused by the budget check: {refused or 'none'}")
    return lines


# Send the oversized request through POST /ask and show its refusal and log line.
# Returns: the report lines.
def oversized_lines():
    logging_utils.LOG_PATH = Path(tempfile.mkdtemp()) / "requests.jsonl"
    body = TestClient(api.app).post("/ask", json={"session_id": "budget-oversized",
                                                   "question": OVERSIZED_REQUEST}).json()
    log = json.loads(logging_utils.LOG_PATH.read_text(encoding="utf-8").splitlines()[-1])
    shown = {key: (value[:60] + "...") if key == "question" else value for key, value in log.items()
             if key not in ("trace_id", "timestamp", "duration_ms")}
    return ["", "=" * 72, f"Oversized request: {len(OVERSIZED_REQUEST)} characters "
            f"({governance.tokens(OVERSIZED_REQUEST)} input tokens), POST /ask",
            f"  refused: {body['refused']}; refusal_reason: {body['refusal_reason']}",
            f"  LLM calls made: {governance.METER['calls']}; tokens charged: {governance.METER['tokens']} "
            f"(the input only); crew ran: {pipeline.LAST_RUN['result'] is not None}",
            f"  log line (trace_id, timestamp, duration_ms omitted; question cut to 60 characters): {json.dumps(shown)}"]


# Wire the record tool onto the Retrieval Agent and show that the crew is refused.
# Returns: the report lines.
def wiring_lines():
    built = crew.build_crew(crew.MockLLM(model="mock-llm"))
    holders = [agent.role for agent in built.agents
               if governance.RECORD_TOOL in {tool.name for tool in agent.tools}]
    lines = ["", "=" * 72, f"Least autonomy: agents holding {governance.RECORD_TOOL} in the real crew: {holders}"]
    with patch.object(crew, "rag_lookup_tool", crew.check_appointment_status_tool):
        try:
            crew.run_crew("What is the status of APT-007?")
            lines.append("  Wrong wiring was NOT blocked.")
        except governance.ToolWiringError as error:
            lines.append(f"  Record tool wired to the Retrieval Agent: ToolWiringError: {error}")
    return lines + [f"  Tool calls made: {crew.COUNTS}"]


# Build the whole report. Takes: nothing. Returns: the report as one string.
def make_report():
    lines = ["Part 4, Task 15: governance (offline, MOCK_LLM, zero API keys)",
             "Token estimate: ceil(characters / 4) over the masked input and every LLM call's prompt and reply", ""]
    lines += measurement_lines(measure()) + oversized_lines() + wiring_lines()
    return "\n".join(lines) + "\n"


# Save the report to outputs/part4_governance.txt and print it.
def main():
    report = make_report()
    target = Path(__file__).resolve().parent / "outputs" / "part4_governance.txt"
    target.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
