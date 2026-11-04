"""Demonstrate the Part 4 cache with call counts and timing, and save the real output."""
# What: asks a policy question, then the same question in another session with different
#   case and no question mark, through POST /ask (where the cache is on); prints LLM calls,
#   time, the cache key and stats, the stored history of both sessions, a record question
#   that is not cached, and the cache being cleared by an upload.
# Task: Part 4, Task 16. Used by: run by hand; it writes outputs/part4_cache.txt.
# Timing varies from run to run (the first request also loads the models); the LLM call
#   counts are the deterministic evidence.

import settings  # noqa: F401  (sets the offline flags before any framework import)

import os
from pathlib import Path
import secrets
import tempfile
import time

from fastapi.testclient import TestClient

import api
import cache
import governance
import logging_utils
from memory import stored_history
import pipeline
import uploads

QUESTION = "What is the cancellation and rescheduling window?"
# A record question (twice) and a different policy question come first, so the one-time
#   model loading is not counted in the timing of QUESTION's miss and hit.
TURNS = [("cache-c", "What is the status of APT-007?"),
         ("cache-d", "What is the status of APT-007?"),
         ("cache-warm", "How do I submit an insurance reimbursement claim?"),
         ("cache-a", QUESTION),
         ("cache-b", "  what is the CANCELLATION and rescheduling window ")]


# Send one question through POST /ask and describe what happened.
# Takes: the test client, a session id and the question. Returns: report lines.
def ask_lines(client, session_id, question):
    started = time.perf_counter()
    body = client.post("/ask", json={"session_id": session_id, "question": question}).json()
    elapsed = (time.perf_counter() - started) * 1000
    key = cache.key_for(question, False, None)
    result = "not cacheable" if key is None else pipeline.LAST_RUN.get("cache", "miss")
    return [f"POST /ask session={session_id} question={question!r}",
            f"  cache key: {key}; result: {result}",
            f"  LLM calls: {governance.METER['calls']}; tokens: {governance.METER['tokens']}; "
            f"time: {elapsed:.1f} ms; cache stats: {cache.STATS}; cached entries: {len(cache.CACHE)}",
            f"  grounded={body['grounded']}; sources={body['sources']}; "
            f"answer starts: {body['answer'].splitlines()[0]!r}", ""]


# Run the demonstration and build the report text. Takes: nothing. Returns: the report.
def make_report():
    logging_utils.LOG_PATH = Path(tempfile.mkdtemp()) / "requests.jsonl"
    cache.clear()
    client = TestClient(api.app)
    lines = ["Part 4, Task 16: in-memory cache for the grounded-generation step (POST /ask, cache on)",
             "Key: (lowercased masked question with whitespace collapsed and trailing punctuation removed, "
             "include_uploads); record questions are never cached", ""]
    for session_id, question in TURNS:
        lines += ask_lines(client, session_id, question)
    lines += [f"Stored history of cache-a: {stored_history('cache-a')[0]} + 1 ai message",
              f"Stored history of cache-b: {stored_history('cache-b')[0]} + 1 ai message",
              f"Both sessions stored 2 messages: {len(stored_history('cache-a'))}, {len(stored_history('cache-b'))}",
              f"Stored answers identical: {stored_history('cache-a')[1] == stored_history('cache-b')[1]}", ""]
    # A temporary admin key for this process only: random, never printed or saved.
    key = secrets.token_hex(16)
    os.environ["PRACTO_ADMIN_KEY"] = key
    reply = client.post("/add-document", headers={"X-Admin-Key": key},
                        json={"doc_id": "parking", "text": "Clinic parking is free for patients. It opens at 8 am."})
    lines += [f"POST /add-document (parking) -> {reply.status_code}; cached entries after upload: "
              f"{len(cache.CACHE)}; cache stats: {cache.STATS}", ""]
    lines += ask_lines(client, "cache-e", QUESTION)
    ids = uploads.collection().get()["ids"]
    if ids:
        uploads.collection().delete(ids=ids)
    del os.environ["PRACTO_ADMIN_KEY"]
    return "\n".join(lines).rstrip() + "\n"


# Save the report to outputs/part4_cache.txt and print it.
def main():
    report = make_report()
    target = Path(__file__).resolve().parent / "outputs" / "part4_cache.txt"
    target.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
