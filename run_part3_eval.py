"""Score the 15 pre-registered questions with the MockJudge and save the real output."""
# What: sends every question in eval_queries.py through POST /ask (the full pipeline with
#   all Part 2 guardrails), scores each reply with judge.mock_judge, and prints the reply,
#   the four scores with reasons, a summary table and the four averages.
# Task: Part 3, Task 13. Used by: run by hand; it writes outputs/part3_eval.txt.
# The request log goes to a temporary file, so the safety check can read each log line.

import settings  # noqa: F401  (sets the offline flags before crewai is imported)

import json
from pathlib import Path
import tempfile

from fastapi.testclient import TestClient

import api
import logging_utils
from eval_queries import EVAL_QUERIES
from judge import mock_judge, outcome, render_prompt
from schema import CrewResponse

PROPERTIES = ("accuracy", "grounding", "completeness", "safety")


# Run all 15 questions and build the report text.
# Takes: nothing. Returns: the report as one string.
def make_report():
    logging_utils.LOG_PATH = Path(tempfile.mkdtemp()) / "requests.jsonl"
    client = TestClient(api.app)
    lines = ["Part 3: MockJudge evaluation of 15 pre-registered questions (POST /ask, MOCK_LLM, offline)"]
    rows, totals = [], dict.fromkeys(PROPERTIES, 0.0)
    for case in EVAL_QUERIES:
        body = client.post("/ask", json={"session_id": f"eval-{case['id']}", "question": case["query"]}).json()
        response = CrewResponse.model_validate(body)
        log_line = logging_utils.LOG_PATH.read_text(encoding="utf-8").splitlines()[-1]
        judgement = mock_judge(case, response, log_line)
        if case is EVAL_QUERIES[0]:
            lines += ["", "Prompt a real judge model would receive (first query):", render_prompt(case, response)]
        lines += ["", "=" * 72, f"[{case['id']}] expected {case['expect']}; got {outcome(response)}: {case['query']}",
                  f"  sources={response.sources}; grounded={response.grounded}; refused={response.refused}; "
                  f"reason: {response.refusal_reason or '(none)'}",
                  f"  logged question: {json.loads(log_line)['question']!r}", "  answer:"]
        lines += [f"    {line}" for line in response.answer.splitlines()]
        for name in PROPERTIES:
            score, reason = judgement[name]
            totals[name] += score
            lines.append(f"  {name}: {score:.2f} ({reason})")
        rows.append(f"| {case['id']} | {case['expect']} | {outcome(response)} | "
                    + " | ".join(f"{judgement[name][0]:.2f}" for name in PROPERTIES) + " |")
    lines += ["", "| Id | Expected | Got | Accuracy | Grounding | Completeness | Safety |",
              "|---|---|---|---:|---:|---:|---:|"] + rows
    lines.append("| Average | | | " + " | ".join(f"{totals[n] / len(EVAL_QUERIES):.3f}" for n in PROPERTIES) + " |")
    return "\n".join(lines) + "\n"


# Save the report to outputs/part3_eval.txt and print it.
def main():
    report = make_report()
    target = Path(__file__).resolve().parent / "outputs" / "part3_eval.txt"
    target.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
