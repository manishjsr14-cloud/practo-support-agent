"""In-memory cache for the grounded-generation step (crew, review and output guardrail)."""
# What: remembers the checked reply for a policy-only question, keyed by its normalized
#   masked text and the include_uploads setting, so a repeated question skips the crew and
#   the review stage (no LLM call). It lives in this process only and is never saved.
# Task: Part 4, Task 16. Used by: pipeline.py (only when use_cache=True), api.py (which turns
#   it on and clears it on every upload), run_part4_cache.py and tests/test_part4.py.
# Questions with an appointment id, or that reuse a remembered id, are never cached: their
#   answer depends on the record and the session, not only on the question text.

import re

from mock_llm import RECORD_ID

# key -> the CrewResponse after the output guardrail (before any emergency note).
CACHE = {}
# Hits and misses since the last clear(), for reports and tests.
STATS = {"hits": 0, "misses": 0}


# Normalize question text: lowercase, collapse whitespace, strip trailing punctuation.
# Takes: the masked question. Returns: the normalized text.
def normalize(text):
    return re.sub(r"\s+", " ", text.lower()).strip().rstrip(".?!,;: ")


# The cache key for a turn, or None when the turn must not be cached.
# Takes: the masked question, include_uploads and the remembered record id (or None).
def key_for(question, include_uploads, remembered_id):
    if RECORD_ID.search(question) or remembered_id:
        return None
    return (normalize(question), bool(include_uploads))


# Look a key up. Takes: a key or None. Returns: the cached CrewResponse, or None on a miss.
def get(key):
    if key is None:
        return None
    found = CACHE.get(key)
    STATS["hits" if found is not None else "misses"] += 1
    return found


# Store a checked reply under a key (nothing is stored for key None).
def put(key, response):
    if key is not None:
        CACHE[key] = response


# Empty the cache and its counters (on every upload, since uploads can change answers).
def clear():
    CACHE.clear()
    STATS.update({"hits": 0, "misses": 0})
