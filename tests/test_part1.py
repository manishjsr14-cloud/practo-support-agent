"""Small checks for the dataset, knowledge base, and plain cosine RAG pipeline."""

from collections import Counter
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

import rag
from dataset import APPOINTMENTS, CATEGORY_WEIGHTS, FEE_RANGES, STATUS_WEIGHTS, generate_appointments
from evaluate import document_metrics
from queries import DEMO_OUT_OF_SCOPE, DEMO_QUERIES
from rag import CALIBRATION_PATH, ROOT, STRATEGIES, answer, build_index, load_documents, search, split_chunks, validate_kb


class Part1Tests(unittest.TestCase):
    @classmethod
    # Build a separate temporary store once so tests run sequentially without replacing the demo index.
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        cls.store = Path(cls.temporary.name) / "chroma"
        cls.counts = build_index(store=cls.store)

    @classmethod
    # Release the temporary directory after every test has finished using it.
    def tearDownClass(cls):
        cls.temporary.cleanup()

    # Check that the fixed seed reproduces the saved fabricated records exactly.
    def test_dataset_is_deterministic_and_matches_json(self):
        self.assertEqual(APPOINTMENTS, generate_appointments())
        saved = json.loads((ROOT / "data" / "appointments.json").read_text(encoding="utf-8"))
        self.assertEqual(APPOINTMENTS, saved)

    # Check category/status coverage and the actual follow-up fraction.
    def test_dataset_coverage_and_follow_up_band(self):
        self.assertEqual(len(APPOINTMENTS), 50)
        categories = Counter(row["category"] for row in APPOINTMENTS)
        statuses = Counter(row["status"] for row in APPOINTMENTS)
        self.assertTrue(all(categories[name] >= 3 for name in CATEGORY_WEIGHTS))
        self.assertTrue(all(statuses[name] >= 1 for name in STATUS_WEIGHTS))
        fraction = sum(row["follow_up_required"] for row in APPOINTMENTS) / len(APPOINTMENTS)
        self.assertTrue(0.10 <= fraction <= 0.30)

    # Check unique identifiers, integer ages, boolean flags, and specialty fee ranges.
    def test_dataset_fields_and_ranges(self):
        self.assertEqual(len({row["record_id"] for row in APPOINTMENTS}), len(APPOINTMENTS))
        for row in APPOINTMENTS:
            self.assertEqual(set(row), {"record_id", "category", "status", "consultation_fee_inr",
                                        "days_since_created", "follow_up_required"})
            low, high = FEE_RANGES[row["category"]]
            self.assertIs(type(row["consultation_fee_inr"]), int)
            self.assertTrue(low <= row["consultation_fee_inr"] <= high)
            self.assertEqual(row["consultation_fee_inr"] % 50, 0)
            self.assertIs(type(row["days_since_created"]), int)
            self.assertTrue(0 <= row["days_since_created"] <= 30)
            self.assertIs(type(row["follow_up_required"]), bool)

    # Check all policy documents retain the required sentence counts.
    def test_knowledge_base_has_all_topics_and_valid_sentences(self):
        self.assertTrue(validate_kb())
        documents = load_documents()
        self.assertEqual(len(documents), 12)
        for body in documents.values():
            self.assertTrue(2 <= len(re.split(r"(?<=[.!?])\s+", body)) <= 5)
        missing_topic = {key: value for key, value in documents.items() if key != "12_home_visits"}
        with patch("rag.load_documents", return_value=missing_topic):
            with self.assertRaisesRegex(ValueError, "12_home_visits.md"):
                validate_kb()

    # Check both chunkers retain parent identifiers and produce the expected source chunks.
    def test_chunk_counts_and_parent_metadata(self):
        documents = load_documents()
        self.assertEqual(self.counts, {"fixed_overlap": 39, "sentence": 36})
        for strategy, expected in (("fixed_overlap", 39), ("sentence", 36)):
            chunks = split_chunks(documents, strategy)
            self.assertEqual(len(chunks), expected)
            self.assertEqual(len({chunk["chunk_id"] for chunk in chunks}), len(chunks))
            self.assertEqual({chunk["parent_id"] for chunk in chunks}, set(documents))
            self.assertTrue(all(chunk["text"] for chunk in chunks))

    # Check retrieval returns three valid chunks in decreasing cosine-similarity order.
    def test_search_returns_plain_top_three(self):
        for strategy in STRATEGIES:
            question = DEMO_QUERIES[1]["query"]
            hits = search(question, strategy=strategy, store=self.store)
            self.assertEqual(len(hits), 3)
            self.assertEqual(hits[0]["parent_id"], "02_cancellation_rescheduling")
            vector = rag.model().encode([question], normalize_embeddings=True, convert_to_numpy=True,
                                        show_progress_bar=False).tolist()
            direct = rag.client(self.store).get_collection(f"practo_{strategy}", embedding_function=None).query(
                query_embeddings=vector, n_results=3, include=["distances"])
            self.assertEqual([hit["chunk_id"] for hit in hits], direct["ids"][0])
            self.assertEqual([hit["similarity"] for hit in hits], [1 - value for value in direct["distances"][0]])

    # Check saved calibration uses observed separating clusters rather than a preset cutoff.
    def test_saved_calibration_separates_observed_clusters(self):
        self.assertTrue(CALIBRATION_PATH.exists(), "Run python run_part1.py before running tests.")
        calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
        for strategy in STRATEGIES:
            values = calibration["strategies"][strategy]
            positives = [row["similarity"] for row in values["measurements"] if row["label"] == "in_scope"]
            negatives = [row["similarity"] for row in values["measurements"] if row["label"] == "out_of_scope"]
            self.assertGreaterEqual(len(positives), 3)
            self.assertGreaterEqual(len(negatives), 2)
            self.assertLess(max(negatives), values["threshold"])
            self.assertLess(values["threshold"], min(positives))
            self.assertEqual(values["threshold"], (max(negatives) + min(positives)) / 2)

    # Check the unrelated demonstration question receives no invented answer or citations.
    def test_out_of_scope_question_uses_fallback(self):
        for strategy in STRATEGIES:
            result = answer(DEMO_OUT_OF_SCOPE, strategy=strategy, store=self.store)
            self.assertTrue(result["fallback"])
            self.assertIn("I don't know", result["answer"])
            self.assertEqual(result["source_ids"], [])
            self.assertEqual(result["chunk_ids"], [])

    # Check accepted answers and citations exactly match the retrieved context.
    def test_answers_are_verbatim_retrieved_context(self):
        for strategy in STRATEGIES:
            for row in DEMO_QUERIES:
                hits = search(row["query"], strategy=strategy, store=self.store)
                result = answer(row["query"], strategy=strategy, store=self.store)
                self.assertFalse(result["fallback"])
                self.assertEqual(result["answer"], "\n\n".join(hit["text"] for hit in hits))
                self.assertEqual(result["source_ids"], [hit["parent_id"] for hit in hits])
                self.assertEqual(result["chunk_ids"], [hit["chunk_id"] for hit in hits])

    # Check repeated chunks from one parent cannot inflate document-level metrics.
    def test_document_metrics_deduplicate_parents(self):
        metrics = document_metrics(["a", "a", "b"], ["a", "c"])
        self.assertEqual((metrics["numerator"], metrics["precision_denominator"], metrics["recall_denominator"]), (1, 2, 2))
        self.assertEqual((metrics["precision"], metrics["recall"]), (0.5, 0.5))

    # Check a missing calibration file gives an error instead of silently selecting a threshold.
    def test_missing_calibration_is_an_error(self):
        missing = self.store.parent / "missing-calibration.json"
        with self.assertRaises((FileNotFoundError, RuntimeError)) as caught:
            answer("What are the fees?", store=self.store, calibration_path=missing)
        self.assertIn("calibrat", str(caught.exception).lower())


if __name__ == "__main__":
    unittest.main()
