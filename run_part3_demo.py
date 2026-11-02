"""Exercise every Part 3 endpoint once and save the real output and its log lines."""
# What: /health, /ask, /add-document (one bad and one good upload), /ask about the upload
#   with and without include_uploads, a WebSocket conversation that uses memory, a client
#   that disconnects mid-conversation and a second client that still works.
# Task: Part 3, Tasks 11-12. Used by: run by hand; it writes outputs/part3_api_demo.txt and
#   outputs/part3_log_sample.jsonl (this run's log lines, from a temporary log file).

import settings  # noqa: F401  (sets the offline flags before crewai is imported)

from pathlib import Path
import tempfile

from fastapi.testclient import TestClient

import api
import logging_utils
from guardrail_queries import PII_EXAMPLE

OUTPUTS = Path(__file__).resolve().parent / "outputs"
PARKING = ("Parking at the clinic costs INR 50 per hour for patients. "
           "The parking area opens at 08:00 and closes at 20:00 every day.")


# Summarise a CrewResponse body in one line, plus its answer's first line.
def summary(body):
    first = body["answer"].splitlines()[0] if body["answer"] else ""
    return (f"  refused={body['refused']}; grounded={body['grounded']}; record_id={body['record_id']}; "
            f"sources={body['sources']}; reason: {body['refusal_reason'] or '(none)'}\n  answer starts: {first!r}")


# Call every endpoint and build the report text.
# Takes: nothing. Returns: the report as one string.
def make_report():
    logging_utils.LOG_PATH = Path(tempfile.mkdtemp()) / "requests.jsonl"
    client = TestClient(api.app)
    lines = ["Part 3: API demo with FastAPI's TestClient (same app as uvicorn api:app)", "",
             f"GET /health -> {client.get('/health').json()}"]
    for question in ("What is the cancellation and rescheduling window?", PII_EXAMPLE):
        body = client.post("/ask", json={"session_id": "demo-ask", "question": question}).json()
        lines += ["", f"POST /ask {question!r}", summary(body)]
    bad = client.post("/add-document", json={"doc_id": "short", "text": "Only one sentence."})
    lines += ["", f"POST /add-document (one sentence) -> {bad.status_code} {bad.json()['detail']}"]
    good = client.post("/add-document", json={"doc_id": "parking", "text": PARKING})
    lines.append(f"POST /add-document (parking, 2 sentences) -> {good.status_code} {good.json()}")
    for flag in (False, True):
        body = client.post("/ask", json={"session_id": f"demo-upload-{flag}", "include_uploads": flag,
                                         "question": "How much does parking at the clinic cost?"}).json()
        lines += ["", f"POST /ask parking question, include_uploads={flag}", summary(body)]
    lines += ["", "WS /ws/chat, client 1 (memory: 'its' reuses APT-007, then disconnects mid-conversation)"]
    with client.websocket_connect("/ws/chat") as ws:
        for question in ("What is the status of APT-007?", "What is its fee?"):
            ws.send_json({"session_id": "demo-ws", "question": question})
            lines += [f"  > {question}", summary(ws.receive_json()).replace("\n", "\n  ")]
        ws.close()
    lines.append("  client 1 closed; server caught WebSocketDisconnect")
    with client.websocket_connect("/ws/chat") as ws:
        ws.send_json({"session_id": "demo-ws-2", "question": "What is the status of APT-012?"})
        lines += ["WS /ws/chat, client 2 after the disconnect",
                  summary(ws.receive_json()).replace("\n", "\n  ")]
    log_text = logging_utils.LOG_PATH.read_text(encoding="utf-8")
    (OUTPUTS / "part3_log_sample.jsonl").write_text(log_text, encoding="utf-8")
    lines += ["", f"Log lines written: {len(log_text.splitlines())} (copied to outputs/part3_log_sample.jsonl)",
              f"Raw number '98765' in the log: {'98765' in log_text}"]
    return "\n".join(lines) + "\n"


# Save the report to outputs/part3_api_demo.txt and print it.
def main():
    report = make_report()
    (OUTPUTS / "part3_api_demo.txt").write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
