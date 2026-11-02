"""FastAPI service around the Part 2 pipeline: /ask, /add-document, /health and /ws/chat."""
# What: HTTP and WebSocket endpoints that answer through pipeline.ask (so every Part 2
#   guardrail applies unchanged), accept uploaded documents into a separate in-memory
#   collection, and log one masked JSON line per request or message.
# Task: Part 3, Tasks 11-12. Run: python -m uvicorn api:app --port 8000
# No endpoint calls check_appointment_status directly; only the crew's Lookup Agent can.

import settings  # noqa: F401  (sets the offline flags before crewai is imported)

import hmac
import json
import os
import threading
import time

from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

import uploads
from logging_utils import log_request
from pipeline import ask
from schema import CrewResponse

app = FastAPI(title="Practo support agent (MOCK_LLM)")
# The pipeline keeps per-run state in module variables (tool counters, LAST_RUN), so
# requests are served one at a time.
LOCK = threading.Lock()


class AskRequest(BaseModel):
    """One question in a session; uploads are searched only when asked for."""
    session_id: str
    question: str
    include_uploads: bool = False


class DocumentRequest(BaseModel):
    """A document to add to the separate uploads collection."""
    doc_id: str
    text: str


class DocumentReply(BaseModel):
    """How many chunks an upload produced."""
    doc_id: str
    chunks_added: int


# Run one pipeline turn while holding the lock.
# Takes: session id, question and include_uploads. Returns: a validated CrewResponse.
def locked_ask(session_id, question, include_uploads=False):
    with LOCK:
        return ask(session_id, question, include_uploads)


# Report that the service is up.
@app.get("/health")
def health():
    started = time.perf_counter()
    log_request("/health", None, None, started, 200)
    return {"status": "ok"}


# Answer one question with the full pipeline (masking, memory, guardrails, crew, schema).
# Takes: an AskRequest. Returns: the CrewResponse.
@app.post("/ask", response_model=CrewResponse)
def ask_endpoint(request: AskRequest):
    started = time.perf_counter()
    try:
        response = locked_ask(request.session_id, request.question, request.include_uploads)
    except Exception as error:
        log_request("/ask", request.session_id, request.question, started, 500)
        raise HTTPException(status_code=500, detail="The pipeline failed.") from error
    log_request("/ask", request.session_id, request.question, started, 200, response)
    return response


# Check the X-Admin-Key header against the PRACTO_ADMIN_KEY environment variable.
# Takes: the header value (or None). Returns: None if allowed, else the refusal message.
# Why: only an operator may add documents; with no key set, uploads are disabled. The key
#   is compared in constant time and never printed or logged.
def admin_refusal(given):
    expected = os.environ.get("PRACTO_ADMIN_KEY")
    if not expected:
        return "Uploads are disabled: PRACTO_ADMIN_KEY is not set on the server."
    if not given or not hmac.compare_digest(given.encode("utf-8"), expected.encode("utf-8")):
        return "Missing or wrong X-Admin-Key."
    return None


# Add a document to the separate in-memory uploads collection (never kb/ or chroma/).
# Takes: a DocumentRequest and the X-Admin-Key header. Returns: the doc id and the number of
#   chunks added. HTTP 403 without the right key; HTTP 422 for a policy name as id, an
#   injection phrase or a text outside 2-5 sentences. Only doc_id and status are logged.
@app.post("/add-document", response_model=DocumentReply)
def add_document_endpoint(request: DocumentRequest,
                          x_admin_key: str | None = Header(default=None, alias="X-Admin-Key")):
    started = time.perf_counter()
    refusal = admin_refusal(x_admin_key)
    if refusal:
        log_request("/add-document", None, None, started, 403, doc_id=request.doc_id)
        raise HTTPException(status_code=403, detail=refusal)
    try:
        with LOCK:
            chunks = uploads.add_document(request.doc_id, request.text)
    except ValueError as error:
        log_request("/add-document", None, None, started, 422, doc_id=request.doc_id)
        raise HTTPException(status_code=422, detail=str(error)) from error
    log_request("/add-document", None, None, started, 200, doc_id=request.doc_id)
    return DocumentReply(doc_id=request.doc_id, chunks_added=chunks)


# Multi-turn chat: each JSON message {session_id, question} gets a CrewResponse back.
# A client that disconnects ends only its own loop; the server keeps serving others.
@app.websocket("/ws/chat")
async def chat(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            message = await websocket.receive_json()
            started = time.perf_counter()
            session_id, question = message.get("session_id"), message.get("question")
            if not isinstance(session_id, str) or not isinstance(question, str):
                log_request("/ws/chat", session_id if isinstance(session_id, str) else None,
                            None, started, "error")
                await websocket.send_json({"error": "Send {session_id, question} as strings."})
                continue
            # The pipeline blocks, so it runs in a worker thread, not on the event loop.
            response = await run_in_threadpool(locked_ask, session_id, question)
            log_request("/ws/chat", session_id, question, started, "ok", response)
            await websocket.send_json(response.model_dump())
    except WebSocketDisconnect:
        return
