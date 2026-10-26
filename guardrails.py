"""Guardrails: phone masking and injection blocking (input), groundedness check (output)."""
# What: masks contact numbers before text reaches the crew or history (KB09), refuses
#   messages that try to override the assistant's instructions, and checks that a policy
#   answer's question terms are supported by the retrieved text.
# Task: Part 2, Task 10. Used by: memory.py (masking), pipeline.py, the runners and tests.
# The patterns and phrases live in rules.py. The output check has two stages: Stage 1 is
#   Part 1's unchanged similarity gate in rag.py; Stage 2 is the term check below.

import re
from functools import lru_cache

from guardrail_calibration import CALIBRATE_ANSWER, CALIBRATE_REFUSE
from rules import INJECTION_PHRASES, PHONE_MASK, PHONE_PATTERN
from schema import validate_response

PHONE = re.compile(PHONE_PATTERN)
RECORD_ID = re.compile(r"APT-[0-9]{3}")
# Word endings removed before comparing words, so "reports" matches "report".
SUFFIXES = (("ies", "y"), ("ing", ""), ("ed", ""), ("s", ""))


# Replace every standalone Indian mobile number with [PHONE_MASKED].
# Takes: the user's text. Returns: the text with numbers masked and nothing else changed.
def mask_phone_numbers(text):
    return PHONE.sub(PHONE_MASK, text)


# Find the injection phrases a message contains.
# Takes: the user's text. Returns: the matching phrases from rules.py, in list order.
# Why: lower-casing and collapsing spaces means "Ignore  PREVIOUS instructions" still matches.
def injection_phrases(text):
    normalized = " ".join(text.lower().split())
    return [phrase for phrase in INJECTION_PHRASES if phrase in normalized]


# Build the validated refusal used when an input guardrail blocks a message.
# Takes: the reason to report. Returns: a CrewResponse with refused=True and no tool data.
def refusal(reason):
    return validate_response({
        "answer": "I can't help with that request. I can explain clinic policies or look up "
                  "an appointment by its id, such as APT-001.",
        "sources": [], "tools_used": [], "record_id": None, "escalation_recommended": None,
        "grounded": False, "refused": True, "refusal_reason": reason})


# Strip one common English ending, keeping at least three letters.
# Takes: a lower-case word. Returns: its simple stem ("results" -> "result").
def stem(word):
    for suffix, replacement in SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            return word[: -len(suffix)] + replacement
    return word


# Pick the question's content terms.
# Takes: question text. Returns: lower-case words, in order and without repeats, after
#   dropping record ids, digits, single letters and scikit-learn's built-in English stop
#   words ("what", "can", "the", ...). No hand-picked keyword list is used.
def question_terms(text):
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    words = re.findall(r"[a-z]+", RECORD_ID.sub(" ", text).lower())
    return list(dict.fromkeys(w for w in words if len(w) > 1 and w not in ENGLISH_STOP_WORDS))


# Embed one word with the cached MiniLM model; lru_cache reuses earlier results.
# Takes: a word. Returns: its unit-length vector, so a dot product is the cosine.
@lru_cache(maxsize=None)
def _embed(word):
    import rag
    return rag.model().encode([word], normalize_embeddings=True, convert_to_numpy=True,
                              show_progress_bar=False)[0]


# Score how well each question term is supported by the top-3 retrieved chunks.
# Takes: the text sent to retrieval. Returns: {"terms": [(term, score, how)], "weakest"}.
# Rule: a term whose stem appears in the chunks scores 1.0 ("text match"); otherwise its
#   score is the highest word-embedding cosine with any chunk word, which lets paraphrases
#   such as "price" for "fee" count. "weakest" is the lowest score (None if no terms).
def support_report(query):
    import rag
    chunks = " ".join(hit["text"] for hit in rag.search(query, "sentence"))
    words = sorted(set(re.findall(r"[a-z]+", chunks.lower())))
    stems = {stem(word) for word in words}
    terms = []
    for term in question_terms(query):
        if stem(term) in stems:
            terms.append((term, 1.0, "text match"))
            continue
        score, nearest = max((float(_embed(term) @ _embed(word)), word) for word in words)
        terms.append((term, score, f"nearest chunk word '{nearest}'"))
    return {"terms": terms, "weakest": min((s for _, s, _ in terms), default=None)}


# Measure T from the calibration questions only, as Part 1 measured its threshold.
# Returns: {"refuse": reports, "answer": reports, "threshold"}; raises ValueError if the
#   two groups' weakest scores overlap, instead of guessing.
@lru_cache(maxsize=1)
def groundedness_threshold():
    refuse = {q: support_report(q) for q in CALIBRATE_REFUSE}
    answer = {q: support_report(q) for q in CALIBRATE_ANSWER}
    high = max(r["weakest"] for r in refuse.values())
    low = min(r["weakest"] for r in answer.values())
    if not high < low:
        raise ValueError(f"Calibration overlaps: max refuse weakest {high:.9f} >= "
                         f"min answer weakest {low:.9f}; no threshold is set.")
    return {"refuse": refuse, "answer": answer, "threshold": (high + low) / 2}


# Decide whether a policy answer is grounded (Stage 1, then Stage 2).
# Takes: the text sent to rag_lookup and its result. Returns: (ok, reason, report).
def check_policy(query, rag_result):
    report = support_report(query)
    if rag_result["fallback"]:
        return False, (f"Stage 1: top-1 similarity {rag_result['similarity']:.9f} is below "
                       f"the threshold {rag_result['threshold']:.9f}"), report
    threshold = groundedness_threshold()["threshold"]
    weak = [term for term, score, _ in report["terms"] if score < threshold]
    if weak:
        return False, (f"Stage 2: not supported by the retrieved policy text: "
                       f"{', '.join(repr(t) for t in weak)} (support below T={threshold:.9f})"), report
    return True, None, report
