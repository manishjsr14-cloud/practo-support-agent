"""One customer turn: clean the text, use session memory, run the crew, return a checked reply."""
# What: ask(session_id, text) runs prepare_user_text (masking) -> memory -> injection check
#   -> crew -> CrewResponse.
# Task: Part 2, Tasks 8-10. Used by: the Task 8-10 runners and tests.
# The crew's Composer only writes the answer text; the factual fields (sources, tools used,
#   record id, escalation flag) are filled here from what the tools actually returned.

import settings  # noqa: F401  (sets the offline flags before crewai is imported)

import json

from crew import run_crew
from guardrails import injection_phrases, refusal
from memory import prepare_user_text, remembered_record_id, session_config, with_memory
from schema import CrewResponse, validate_response
from tools import recommend_escalation

# The latest turn's crew result and remembered id, kept so reports and tests can inspect it.
LAST_RUN = {}


# Turn the crew's result into a complete, validated CrewResponse.
# Takes: the dict from crew.run_crew(). Returns: a CrewResponse; raises ValueError if the
#   crew gave no CrewResponse or the finished reply does not validate.
# Why: facts come from the tools' own results; escalation comes only from
#   tools.recommend_escalation(), never from the LLM.
def build_response(result):
    draft = result["pydantic"]
    if not isinstance(draft, CrewResponse):
        raise ValueError(f"The crew did not return a CrewResponse (got {type(draft).__name__}).")
    rag = result["rag_results"][-1] if result["rag_results"] else None
    # KB parent ids (not chunks): each once, in first-seen retrieval order.
    sources = list(dict.fromkeys(rag["source_ids"])) if rag else []
    # The last lookup that found a record; a lookup error means no record.
    record = next((row for row in reversed(result["lookups"]) if "error" not in row), None)
    data = draft.model_dump()
    data.update({
        "sources": sources,
        "tools_used": [name for name, count in result["counts"].items() if count],
        "record_id": record["record_id"] if record else None,
        "escalation_recommended": recommend_escalation(record["escalation_score"]) if record else None,
        # Provisional rule until Task 10's guardrail replaces it: grounded means at least
        # one tool returned real data (KB sources above the threshold, or a found record).
        "grounded": bool(sources or record),
        "refused": False,
        "refusal_reason": None,
    })
    return validate_response(data)


# The work done inside memory for one turn.
# Takes: {"question": prepared text, "history": this session's earlier messages}.
# Returns: {"answer": text to store, "response": the reply as JSON}.
def run_turn(inputs):
    question = inputs["question"]
    # Input guardrail (Task 10): an injection attempt is refused before the crew runs, so
    # no agent or tool ever sees it. The turn is still stored, with the refusal as answer.
    phrases = injection_phrases(question)
    if phrases:
        response = refusal("Prompt injection detected: " + ", ".join(repr(p) for p in phrases))
        LAST_RUN.clear()
        LAST_RUN.update({"result": None, "remembered_record_id": None})
        return {"answer": response.answer, "response": response.model_dump_json()}
    record_id = remembered_record_id(question, inputs["history"])
    result = run_crew(question, output_model=CrewResponse, session_record_id=record_id)
    response = build_response(result)
    LAST_RUN.clear()
    LAST_RUN.update({"result": result, "remembered_record_id": record_id})
    return {"answer": response.answer, "response": response.model_dump_json()}


# The memory-wrapped turn, built once and shared by every session.
CHAT = with_memory(run_turn)


# Answer one customer message within one session.
# Takes: a session id and the raw text. Returns: a validated CrewResponse.
def ask(session_id, text):
    question = prepare_user_text(text)
    output = CHAT.invoke({"question": question}, config=session_config(session_id))
    return validate_response(json.loads(output["response"]))
