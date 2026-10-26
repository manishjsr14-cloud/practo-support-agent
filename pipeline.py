"""One customer turn: clean the text, use session memory, run the crew, return a checked reply."""
# What: ask(session_id, text) runs prepare_user_text (masking) -> memory -> injection check
#   -> emergency rule -> crew -> output groundedness guardrail -> schema validation.
# Task: Part 2, Tasks 8-10. Used by: the Task 8-10 runners and tests.
# The crew's Composer only writes the answer text; the factual fields (sources, tools used,
#   record id, escalation flag) are filled here from what the tools actually returned.

import settings  # noqa: F401  (sets the offline flags before crewai is imported)

import json

from crew import run_crew
from guardrails import check_policy, emergency_phrases, emergency_response, injection_phrases, refusal
from memory import prepare_user_text, remembered_record_id, session_config, with_memory
from mock_llm import KB08_NOTE, NO_RECORD_ID
from schema import CrewResponse, validate_response
from tools import recommend_escalation

# The latest turn's crew result and remembered id, kept so reports and tests can inspect it.
LAST_RUN = {}


# Turn the crew's result into a complete, validated CrewResponse, after the output guardrail.
# Takes: the dict from crew.run_crew(). Returns: (CrewResponse, guardrail check or None);
#   raises ValueError if the crew gave no CrewResponse or the reply does not validate.
# Why: facts come from the tools' own results; escalation comes only from
#   tools.recommend_escalation(); grounded/refused come from the guardrail (Task 10).
def build_response(result):
    draft = result["pydantic"]
    if not isinstance(draft, CrewResponse):
        raise ValueError(f"The crew did not return a CrewResponse (got {type(draft).__name__}).")
    rag = result["rag_results"][-1] if result["rag_results"] else None
    # The last lookup that found a record; a lookup error means no record.
    record = next((row for row in reversed(result["lookups"]) if "error" not in row), None)
    data = draft.model_dump()
    data.update({"tools_used": [name for name, count in result["counts"].items() if count],
                 "record_id": record["record_id"] if record else None,
                 "escalation_recommended": recommend_escalation(record["escalation_score"]) if record else None,
                 "sources": [], "grounded": False, "refused": False, "refusal_reason": None})
    check = None
    if rag is not None:
        # Output guardrail on the policy part: Stage 1 (similarity), then Stage 2 (terms).
        query = result["rag_queries"][-1]
        ok, reason, report = check_policy(query, rag)
        check = {"query": query, "ok": ok, "reason": reason, "report": report}
        if ok:
            # KB parent ids (not chunks): each once, in first-seen retrieval order.
            data.update({"sources": list(dict.fromkeys(rag["source_ids"])), "grounded": True})
        elif record:
            # Mixed question: keep the record, drop the policy text, say what was not answered.
            data.update({"answer": f"Lookup result:\n{result['outputs']['Lookup']}\n\n{KB08_NOTE}\n\n"
                                   f"Not answered: the policy part '{query}' is not supported "
                                   f"by the clinic policies.",
                         "grounded": True, "refusal_reason": reason})
        else:
            data.update({"answer": f"I can't answer '{query}' from the clinic policies.",
                         "refused": True, "refusal_reason": reason})
    elif record:
        # Record-only question: grounded in the stored appointment record.
        data["grounded"] = True
    else:
        # No policy question and no record found: nothing to ground an answer in.
        lookup = result["lookups"][-1]["error"] if result["lookups"] else NO_RECORD_ID
        data.update({"refused": True, "refusal_reason": f"No policy question and no record: {lookup}"})
    return validate_response(data), check


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
        LAST_RUN.update({"result": None, "remembered_record_id": None, "guardrail": None})
        return {"answer": response.answer, "response": response.model_dump_json()}
    # Emergency rule (added after the first guardrail run, because KB08 requires it): a
    # described emergency gets KB08's direction at once, before the crew and before the
    # groundedness check. It is a separate rule, not part of groundedness.
    urgent = emergency_phrases(question)
    if urgent:
        response = emergency_response(urgent)
        LAST_RUN.clear()
        LAST_RUN.update({"result": None, "remembered_record_id": None, "guardrail": None,
                         "emergency": urgent})
        return {"answer": response.answer, "response": response.model_dump_json()}
    record_id = remembered_record_id(question, inputs["history"])
    result = run_crew(question, output_model=CrewResponse, session_record_id=record_id)
    response, check = build_response(result)
    LAST_RUN.clear()
    LAST_RUN.update({"result": result, "remembered_record_id": record_id, "guardrail": check})
    return {"answer": response.answer, "response": response.model_dump_json()}


# The memory-wrapped turn, built once and shared by every session.
CHAT = with_memory(run_turn)


# Answer one customer message within one session.
# Takes: a session id and the raw text. Returns: a validated CrewResponse.
def ask(session_id, text):
    question = prepare_user_text(text)
    output = CHAT.invoke({"question": question}, config=session_config(session_id))
    return validate_response(json.loads(output["response"]))
