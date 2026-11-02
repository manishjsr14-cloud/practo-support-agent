"""Emergency detection v2: the phrase list OR closeness in meaning to emergency descriptions."""
# What: compares a whole message's MiniLM embedding with EMERGENCY_DESCRIPTIONS (rules.py)
#   and triggers when a v1 phrase matches or the best cosine reaches T_EMERGENCY.
# Task: Part 3 extension (emergency detection v2). Used by: pipeline.py,
#   evaluate_emergency_v2.py and the tests.
# T_EMERGENCY comes only from emergency_calibration.py, measured like Part 1's threshold;
#   the test sets are never used to set it.

from functools import lru_cache

from emergency_calibration import CAL_EMERGENCY, CAL_NOT
from guardrails import emergency_phrases, kb08_direction
from rules import EMERGENCY_DESCRIPTIONS
from schema import validate_response


# Embed texts with the cached MiniLM model, as unit vectors (so a dot product is the cosine).
def _embed(texts):
    import rag
    return rag.model().encode(list(texts), normalize_embeddings=True, convert_to_numpy=True,
                              show_progress_bar=False)


# The descriptions' embeddings, computed once.
@lru_cache(maxsize=1)
def _descriptions():
    return _embed(EMERGENCY_DESCRIPTIONS)


# Find the emergency description closest in meaning to a message.
# Takes: the message. Returns: (best cosine similarity, that description).
def emergency_meaning(text):
    scores = _descriptions() @ _embed([text])[0]
    best = int(scores.argmax())
    return float(scores[best]), EMERGENCY_DESCRIPTIONS[best]


# Measure T_EMERGENCY from the calibration messages only.
# Returns: {"emergency": {msg: score}, "not": {msg: score}, "threshold", "rule", "false_triggers"}.
# Rule: the midpoint of max(CAL_NOT) and min(CAL_EMERGENCY). If the groups overlap, use
#   min(CAL_EMERGENCY) instead (safety first: catch every calibration emergency) and list the
#   CAL_NOT messages that this makes trigger.
@lru_cache(maxsize=1)
def emergency_threshold():
    emergency = {m: emergency_meaning(m)[0] for m in CAL_EMERGENCY}
    not_emergency = {m: emergency_meaning(m)[0] for m in CAL_NOT}
    high, low = max(not_emergency.values()), min(emergency.values())
    if high < low:
        threshold, rule = (high + low) / 2, "midpoint (groups separate)"
    else:
        threshold, rule = low, "min(CAL_EMERGENCY) (groups overlap; safety first)"
    false_triggers = [m for m, s in not_emergency.items() if s >= threshold]
    return {"emergency": emergency, "not": not_emergency, "threshold": threshold,
            "rule": rule, "false_triggers": false_triggers}


# The note added to a normal reply when only the meaning matched (two-level response, added
#   after the first v2 run). It gives KB08's direction and never diagnoses or names a medicine.
NOTE = ("If this is an emergency, contact local emergency services or go to the nearest "
        "emergency department now.")


# Append the emergency note and KB08's direction to a normal reply (meaning-only match).
# Takes: a CrewResponse. Returns: a validated copy with the note appended to the answer and
#   "08_emergency_protocol" added to sources; refused, grounded and the reason are unchanged.
def with_emergency_note(response):
    data = response.model_dump()
    data["answer"] += f"\n\n{NOTE}\nClinic policy (KB08): \"{kb08_direction()}\""
    if "08_emergency_protocol" not in data["sources"]:
        data["sources"].append("08_emergency_protocol")
    return validate_response(data)


# Decide whether a message is an emergency, with v1 (phrases) and v2 (phrases or meaning).
# Takes: the message. Returns: {"phrases", "similarity", "description", "v1", "v2", "by"}.
def detect(text):
    phrases = emergency_phrases(text)
    similarity, description = emergency_meaning(text)
    meaning = similarity >= emergency_threshold()["threshold"]
    by = "phrase" if phrases else "meaning" if meaning else None
    return {"phrases": phrases, "similarity": similarity, "description": description,
            "v1": bool(phrases), "v2": bool(phrases) or meaning, "by": by}
