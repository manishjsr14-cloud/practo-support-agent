"""Uploaded documents: a separate in-memory collection and the combined search over it."""
# What: keeps API uploads in their own in-memory ChromaDB collection "practo_uploads"
#   (never kb/, never the chroma/ collections, nothing persisted), and searches the sentence
#   collection and the uploads together only when a request sets include_uploads=true.
# Task: Part 3, Task 11. Used by: api.py, tools.py and guardrails.py.
# rag.py is not changed: this module reuses rag.split_chunks, rag.model and rag.search.

import json
from functools import lru_cache
import re

import rag
from guardrails import mask_phone_numbers

COLLECTION = "practo_uploads"


# Return the in-memory uploads collection, created once per process.
# Why: EphemeralClient keeps everything in RAM, so uploads vanish when the server stops.
@lru_cache(maxsize=1)
def collection():
    import chromadb
    from chromadb.config import Settings
    client = chromadb.EphemeralClient(settings=Settings(anonymized_telemetry=False))
    return client.get_or_create_collection(COLLECTION, embedding_function=None,
                                           configuration={"hnsw": {"space": "cosine", "num_threads": 1}})


# Mask phone numbers, check the sentence count and store an uploaded document.
# Takes: a document id and its text. Returns: the number of chunks added.
# Raises ValueError for a KB topic name as id or a text outside 2-5 sentences.
def add_document(doc_id, text):
    if doc_id in rag.REQUIRED_TOPICS:
        raise ValueError("doc_id must not reuse a policy name from kb/.")
    body = " ".join(mask_phone_numbers(text).split())
    # Count sentences the same way rag.validate_kb does.
    if not 2 <= len(re.split(r"(?<=[.!?])\s+", body)) <= 5:
        raise ValueError("An uploaded document must have two to five sentences.")
    # Same sentence strategy as the policies (2 sentences per chunk); the "upload:" prefix
    # on the parent id shows in sources that the text came from an upload.
    chunks = rag.split_chunks({f"upload:{doc_id}": body}, "sentence")
    vectors = rag.model().encode([c["text"] for c in chunks], normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False).tolist()
    collection().upsert(ids=[c["chunk_id"] for c in chunks], embeddings=vectors,
                        documents=[c["text"] for c in chunks],
                        metadatas=[{"parent_id": c["parent_id"]} for c in chunks])
    return len(chunks)


# Plain cosine top-3 across the sentence collection and the uploads combined.
# Takes: a question. Returns: up to 3 hit dicts shaped like rag.search's, most similar first.
# Rule: query both, merge, keep the 3 highest similarities; no reranking, no filtering.
#   With no uploads this is exactly rag.search(query, "sentence").
def combined_search(query):
    hits = rag.search(query, "sentence")
    if collection().count():
        vector = rag.model().encode([query], normalize_embeddings=True, convert_to_numpy=True,
                                    show_progress_bar=False).tolist()
        found = collection().query(query_embeddings=vector, n_results=min(rag.TOP_K, collection().count()),
                                   include=["documents", "metadatas", "distances"])
        hits += [{"chunk_id": chunk, "parent_id": meta["parent_id"], "text": text,
                  "similarity": 1.0 - float(distance)}
                 for chunk, text, meta, distance in zip(found["ids"][0], found["documents"][0],
                                                      found["metadatas"][0], found["distances"][0])]
    # sorted() is stable, so equal similarities keep policy chunks before uploads.
    return sorted(hits, key=lambda hit: hit["similarity"], reverse=True)[:rag.TOP_K]


# Answer like rag.answer, but from the combined top-3.
# Takes: a question. Returns: the same keys as rag.answer.
# Stage 1 is unchanged: the combined top-1 against the saved threshold; nothing recalibrated.
def combined_answer(query):
    threshold = json.loads(rag.CALIBRATION_PATH.read_text(encoding="utf-8"))["strategies"]["sentence"]["threshold"]
    hits = combined_search(query)
    fallback = hits[0]["similarity"] < threshold
    return {"answer": rag.FALLBACK if fallback else "\n\n".join(hit["text"] for hit in hits),
            "source_ids": [] if fallback else [hit["parent_id"] for hit in hits],
            "chunk_ids": [] if fallback else [hit["chunk_id"] for hit in hits],
            "similarity": hits[0]["similarity"], "threshold": threshold, "fallback": fallback}
