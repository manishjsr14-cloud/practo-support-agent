"""Check the Part 4 cache for the grounded-generation step."""
# What: key normalization, which turns may be cached, a hit that makes no LLM call, a
#   cached turn stored in session memory like any other, and clearing on upload.
# Task: Part 4, Task 16 tests. Used by: "python -m unittest discover -s tests".

import settings  # noqa: F401  (must come before any framework import)

import os
import secrets
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

import api
import cache
import governance
from memory import stored_history
import pipeline
import uploads

QUESTION = "What is the cancellation and rescheduling window?"


class CacheTests(unittest.TestCase):
    # Every test starts with an empty cache.
    def setUp(self):
        cache.clear()

    # Case, extra spaces and trailing punctuation do not change the key.
    def test_normalize(self):
        self.assertEqual(cache.normalize("  What is the  Cancellation\twindow?! "), "what is the cancellation window")
        self.assertEqual(cache.key_for("Window?", True, None), ("window", True))
        self.assertNotEqual(cache.key_for("Window?", True, None), cache.key_for("Window?", False, None))

    # Record questions and remembered ids are never cached.
    def test_record_turns_are_not_cached(self):
        self.assertIsNone(cache.key_for("What is the status of APT-007?", False, None))
        self.assertIsNone(cache.key_for("What is its fee?", False, "APT-007"))

    # A repeated question is a hit: no LLM call, the same reply.
    def test_repeat_is_a_hit_without_llm_calls(self):
        first = pipeline.ask("test-cache-1", QUESTION, use_cache=True)
        self.assertEqual(governance.METER["calls"], 6)
        again = pipeline.ask("test-cache-2", "what is the cancellation and rescheduling window", use_cache=True)
        self.assertEqual(governance.METER["calls"], 0)
        self.assertEqual(pipeline.LAST_RUN["cache"], "hit")
        self.assertEqual(again, first)
        self.assertEqual(cache.STATS, {"hits": 1, "misses": 1})

    # A cached turn is stored in session memory exactly like an uncached one.
    def test_cached_turn_is_stored_in_history(self):
        pipeline.ask("test-cache-history-a", QUESTION, use_cache=True)
        pipeline.ask("test-cache-history-b", QUESTION, use_cache=True)
        self.assertEqual(pipeline.LAST_RUN["cache"], "hit")
        stored = stored_history("test-cache-history-a")
        self.assertEqual([kind for kind, _ in stored], ["human", "ai"])
        self.assertEqual(stored_history("test-cache-history-b"), stored)

    # Without use_cache (the default) nothing is stored.
    def test_default_does_not_cache(self):
        pipeline.ask("test-cache-off", QUESTION)
        self.assertEqual(cache.CACHE, {})

    # A successful upload through the API empties the cache; the upload is removed afterwards.
    def test_upload_clears_cache(self):
        client = TestClient(api.app)
        client.post("/ask", json={"session_id": "test-cache-api", "question": QUESTION})
        self.assertEqual(len(cache.CACHE), 1)
        key = secrets.token_hex(16)
        try:
            with patch.dict(os.environ, {"PRACTO_ADMIN_KEY": key}):
                reply = client.post("/add-document", headers={"X-Admin-Key": key},
                                    json={"doc_id": "test-cache-doc", "text": "Parking is free. It is open daily."})
            self.assertEqual(reply.status_code, 200)
            self.assertEqual(cache.CACHE, {})
        finally:
            ids = uploads.collection().get()["ids"]
            if ids:
                uploads.collection().delete(ids=ids)


if __name__ == "__main__":
    unittest.main()
