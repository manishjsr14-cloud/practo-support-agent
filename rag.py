"""Plain local vector retrieval and extractive MOCK_LLM answers."""
# What: reads the 12 policy files in kb/, cuts them into chunks, stores them in two
#   local ChromaDB collections and answers questions by quoting the closest chunks.
# Task: Part 1 (Tasks 2-5): KB validation, indexing, retrieval and the fallback answer.
# Used by: evaluate.py, demo.py, tests/test_part1.py, and run_part1.py (which runs
#   "rag.py validate" and "rag.py index" as separate commands).

import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import re

# Set the offline/privacy flags BEFORE torch, sentence_transformers or chromadb are
# imported (they are imported later, inside functions), because those libraries read
# these variables when they load: no model downloads, no telemetry, no progress bars.
for name, value in {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                    "ANONYMIZED_TELEMETRY": "False", "CREWAI_DISABLE_TELEMETRY": "true",
                    "OTEL_SDK_DISABLED": "true", "HF_HUB_DISABLE_PROGRESS_BARS": "1"}.items():
    os.environ[name] = value

# Paths are built from this file's folder so the code works from any working directory.
ROOT = Path(__file__).resolve().parent
CHROMA_PATH = ROOT / "chroma"
CALIBRATION_PATH = ROOT / "outputs" / "calibration.json"
# The two chunking strategies; each gets its own collection. STRATEGY is the one the
# demo uses; it is set by hand after reading the evaluation, never written by code.
STRATEGIES = ("fixed_overlap", "sentence")
STRATEGY = "sentence"
# A "chunk" is a small piece of a policy document that is stored and searched on its own.
# Fixed chunks: 320 characters, each starting 80 characters before the previous one ended.
# Sentence chunks: 2 sentences each. TOP_K: how many chunks a search returns.
CHUNK_SIZE, OVERLAP, SENTENCES_PER_CHUNK, TOP_K = 320, 80, 2, 3
# The embedding model, pinned to an exact revision so results are reproducible.
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
# The fixed reply used when no policy text is similar enough to the question.
FALLBACK = "I don't know based on the available clinic-policy context."
# The 12 policy filenames (without .md) that the KB must contain.
REQUIRED_TOPICS = (
    "01_appointment_booking", "02_cancellation_rescheduling", "03_consultation_fees",
    "04_insurance_claims", "05_prescription_refills", "06_lab_turnaround",
    "07_telemedicine", "08_emergency_protocol", "09_patient_privacy",
    "10_follow_up_discount", "11_second_opinion", "12_home_visits",
)

# Read each policy body in filename order and collapse its whitespace.
# Takes: nothing. Returns: a dict {filename stem: policy text as one line}.
# Why: every other step (validation, chunking) needs the same clean text.
def load_documents():
    documents = {}
    for path in sorted((ROOT / "kb").glob("*.md")):
        # Drop Markdown heading lines (starting with "#") so only the policy sentences remain.
        lines = [line for line in path.read_text(encoding="utf-8").splitlines()
                 if not line.lstrip().startswith("#")]
        # Join the lines and squeeze runs of spaces/newlines into single spaces.
        documents[path.stem] = " ".join(" ".join(lines).split())
    # Fail loudly if the folder is empty or any policy has no text.
    if not documents or not all(documents.values()):
        raise ValueError("The kb folder must contain nonempty Markdown policies.")
    return documents

# Check every required topic filename and count each policy's sentences.
# Takes: nothing. Returns: True if valid; raises ValueError otherwise.
# Why: Task 2 requires all 12 named topics and 2-5 sentences per document.
def validate_kb():
    documents = load_documents()
    # Any required topic name that has no matching file is reported by name.
    missing = sorted(set(REQUIRED_TOPICS) - set(documents))
    if missing:
        raise ValueError("Missing required policy files: " + ", ".join(name + ".md" for name in missing))
    for parent, body in documents.items():
        # Split after ".", "!" or "?" followed by whitespace; each piece counts as a sentence.
        count = len(re.split(r"(?<=[.!?])\s+", body))
        print(f"{parent}: {count} sentences")
        if not 2 <= count <= 5:
            raise ValueError(f"{parent} must have two to five sentences.")
    print(f"KB validation: PASS; documents={len(documents)}; required filenames present")
    return True

# Make overlapping character windows or consecutive pairs of sentences.
# Takes: the documents dict and a strategy name. Returns: a list of chunk dicts with
#   chunk_id, parent_id (the source document's name) and text.
# Why: search works on small pieces, and parent_id lets each piece be traced to its policy.
def split_chunks(documents, strategy):
    if strategy not in STRATEGIES:
        raise ValueError(f"Choose one of {STRATEGIES}")
    chunks = []
    for parent, body in sorted(documents.items()):
        if strategy == "fixed_overlap":
            # Step forward 320 - 80 = 240 characters each time, so neighbouring chunks
            # share 80 characters; stop once a window reaches the end of the text.
            texts = []
            for start in range(0, len(body), CHUNK_SIZE - OVERLAP):
                texts.append(body[start:start + CHUNK_SIZE].strip())
                if start + CHUNK_SIZE >= len(body):
                    break
        else:
            # Split into sentences, then group every 2 consecutive sentences into one chunk.
            sentences = re.split(r"(?<=[.!?])\s+", body)
            texts = [" ".join(sentences[i:i + SENTENCES_PER_CHUNK])
                     for i in range(0, len(sentences), SENTENCES_PER_CHUNK)]
        # Give each chunk a unique id like "03_consultation_fees:sentence:001" and keep
        # its parent document id; empty pieces are skipped.
        chunks.extend({"chunk_id": f"{parent}:{strategy}:{i:03d}",
                       "parent_id": parent, "text": text}
                      for i, text in enumerate(texts) if text)
    return chunks

# Load the cached checkpoint once and encode deterministically on the CPU.
# Takes: nothing. Returns: the SentenceTransformer embedding model.
# An "embedding" is a list of numbers that represents a text's meaning; texts with
#   similar meaning get embeddings that point in similar directions.
# Why: loading is slow, so lru_cache keeps the first loaded model and reuses it.
@lru_cache(maxsize=1)
def model():
    # Imported here, not at the top, so the environment flags above are already set.
    import torch
    from sentence_transformers import SentenceTransformer
    # One CPU thread keeps the arithmetic order, and therefore the results, repeatable.
    torch.set_num_threads(1)
    # local_files_only and token=False: use only the cached model, never contact Hugging Face.
    return SentenceTransformer(MODEL_ID, revision=MODEL_REVISION, device="cpu",
                               local_files_only=True, token=False)

# Reuse a local Chroma client with telemetry disabled.
# Takes: the folder where the database lives. Returns: a ChromaDB client for that folder.
# Why: the demo uses chroma/ while tests use a temporary folder; the cache keeps one
#   client per folder instead of opening a new one on every call.
@lru_cache(maxsize=4)
def client(store=CHROMA_PATH):
    import chromadb
    from chromadb.config import Settings
    # PersistentClient saves the database to disk in the given folder.
    return chromadb.PersistentClient(path=str(store), settings=Settings(anonymized_telemetry=False))

# Delete and rebuild both cosine collections from the policy files.
# Takes: the database folder. Returns: {strategy: number of chunks stored}.
# A "ChromaDB collection" is a named table of chunks with their embeddings and metadata.
# Why: rebuilding from scratch each run means the index always matches the current KB.
def build_index(store=CHROMA_PATH):
    from chromadb.errors import NotFoundError
    counts = {}
    for strategy in STRATEGIES:
        # Remove the old collection; on the very first run there is none, which is fine.
        try:
            client(store).delete_collection(f"practo_{strategy}")
        except NotFoundError:
            pass
        chunks = split_chunks(load_documents(), strategy)
        # embedding_function=None: we compute embeddings ourselves with the pinned model.
        # "space": "cosine" makes Chroma measure cosine distance (see search below).
        # HNSW is Chroma's nearest-neighbour index; one thread keeps the build repeatable.
        collection = client(store).create_collection(f"practo_{strategy}", embedding_function=None,
                     configuration={"hnsw": {"space": "cosine", "num_threads": 1}})
        # Turn every chunk into an embedding; normalizing gives each vector length 1.
        vectors = model().encode([c["text"] for c in chunks], normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False).tolist()
        # "upsert" = insert a row, or update it if that id already exists.
        # Each chunk's parent_id goes into its metadata so results can be traced to a policy.
        collection.upsert(ids=[c["chunk_id"] for c in chunks], embeddings=vectors,
                          documents=[c["text"] for c in chunks],
                          metadatas=[{"parent_id": c["parent_id"]} for c in chunks])
        counts[strategy] = collection.count()
        print(f"{strategy}: {counts[strategy]} chunks")
    # Record the settings used, for the saved outputs/index.txt.
    # This line prints a fixed subset of three of the flags set at the top of this file.
    print(f"Model: {MODEL_ID}; revision: {MODEL_REVISION}")
    print(f"Fixed size={CHUNK_SIZE}; overlap={OVERLAP}; sentence size={SENTENCES_PER_CHUNK}")
    print("HF_HUB_OFFLINE=1; ANONYMIZED_TELEMETRY=False; CREWAI_DISABLE_TELEMETRY=true")
    return counts

# Return only the three closest chunks, without reranking or filtering.
# Takes: a question, a strategy name and the database folder. Returns: a list of 3 dicts
#   with chunk_id, parent_id, text and similarity, most similar first.
# "Top-3 retrieval" means: embed the question and return the 3 stored chunks whose
#   embeddings are closest to it. Nothing else changes the order or the selection.
def search(query, strategy=STRATEGY, store=CHROMA_PATH):
    if strategy not in STRATEGIES or not query.strip():
        raise ValueError("Provide a nonempty query and a known strategy.")
    collection = client(store).get_collection(f"practo_{strategy}", embedding_function=None)
    # Embed the question with the same model and normalization as the stored chunks.
    vector = model().encode([query], normalize_embeddings=True, convert_to_numpy=True,
                            show_progress_bar=False).tolist()
    found = collection.query(query_embeddings=vector, n_results=TOP_K,
                             include=["documents", "metadatas", "distances"])
    # Chroma returns "cosine distance" (0 = same direction, larger = less alike).
    # "Cosine similarity" = 1 - cosine distance, so higher means more similar.
    # Results come back as lists inside a list (one per query); [0] picks our only query.
    return [{"chunk_id": chunk, "parent_id": meta["parent_id"], "text": text,
             "similarity": 1.0 - float(distance)}
            for chunk, text, meta, distance in zip(found["ids"][0], found["documents"][0],
                                                 found["metadatas"][0], found["distances"][0])]

# Quote the retrieved text, or refuse when its best similarity is too low.
# Takes: a question, a strategy, the database folder and the calibration file.
# Returns: a dict with the answer text, source/chunk ids, top-1 similarity, threshold
#   and whether the fallback was used.
# Why: under MOCK_LLM there is no real language model, so the "answer" is the retrieved
#   text quoted word for word; the threshold decides between answering and falling back.
def answer(query, strategy=STRATEGY, store=CHROMA_PATH, calibration_path=CALIBRATION_PATH):
    # Refuse to guess a threshold: calibration must have been run first.
    if not Path(calibration_path).is_file():
        raise FileNotFoundError("Missing outputs/calibration.json; run python evaluate.py calibration first.")
    # The "calibration threshold" is the measured cut-off similarity saved by evaluate.py.
    threshold = json.loads(Path(calibration_path).read_text(encoding="utf-8"))["strategies"][strategy]["threshold"]
    hits = search(query, strategy, store)
    # Only the best (top-1) similarity decides: below the threshold means "not in scope".
    fallback = hits[0]["similarity"] < threshold
    # On fallback return the fixed reply and no citations; otherwise quote all 3 chunks.
    return {"answer": FALLBACK if fallback else "\n\n".join(hit["text"] for hit in hits),
            "source_ids": [] if fallback else [hit["parent_id"] for hit in hits],
            "chunk_ids": [] if fallback else [hit["chunk_id"] for hit in hits],
            "similarity": hits[0]["similarity"], "threshold": threshold, "fallback": fallback}


# Command line: "python rag.py validate" checks the KB; "python rag.py index" rebuilds it.
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("validate", "index"))
    args = parser.parse_args()
    validate_kb() if args.step == "validate" else build_index()
