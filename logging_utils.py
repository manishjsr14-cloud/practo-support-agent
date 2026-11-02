"""JSON-Lines request log with masked questions and a seven-day clean-up."""
# What: writes one JSON line per HTTP request or WebSocket message to logs/requests.jsonl,
#   and deletes lines older than seven days, as KB09 allows for sanitized logs.
# Task: Part 3, Task 12. Used by: api.py, run_part3_demo.py and the tests.
# Every question goes through memory.prepare_user_text first, so a raw phone number never
#   reaches the file.

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time
import uuid

from memory import prepare_user_text

LOG_PATH = Path(__file__).resolve().parent / "logs" / "requests.jsonl"


# Append one log entry and return it.
# Takes: the endpoint, session id, raw question (or None), the start time from
#   time.perf_counter(), the status (HTTP code or "ok"/"error") and the CrewResponse (or None).
# Returns: the entry dict that was written. path=None means the current LOG_PATH (read at
#   call time, so tests and the demo can point it at another file). doc_id (uploads only)
#   adds a "doc_id" field; an upload's text is never logged.
def log_request(endpoint, session_id, question, started, status, response=None, path=None, doc_id=None):
    path = path or LOG_PATH
    entry = {"trace_id": str(uuid.uuid4()),
             "timestamp": datetime.now(timezone.utc).isoformat(),
             "endpoint": endpoint, "session_id": session_id,
             "question": None if question is None else prepare_user_text(question),
             "duration_ms": round((time.perf_counter() - started) * 1000, 1),
             "status": status,
             "refused": None if response is None else response.refused,
             "grounded": None if response is None else response.grounded}
    if doc_id is not None:
        entry["doc_id"] = doc_id
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as log:
        log.write(json.dumps(entry) + "\n")
    return entry


# Delete log lines older than the given number of days (KB09: at most seven).
# Takes: the log path (None = LOG_PATH), the age limit and the current time (for tests).
# Returns: the number of lines removed.
def prune_logs(path=None, days=7, now=None):
    path = path or LOG_PATH
    if not path.exists():
        return 0
    cutoff = (now or datetime.now(timezone.utc)) - timedelta(days=days)
    lines = path.read_text(encoding="utf-8").splitlines()
    kept = [line for line in lines if datetime.fromisoformat(json.loads(line)["timestamp"]) >= cutoff]
    path.write_text("".join(line + "\n" for line in kept), encoding="utf-8")
    return len(lines) - len(kept)
