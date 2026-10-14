"""Plain local vector retrieval and extractive MOCK_LLM answers."""

import argparse
from functools import lru_cache
import json
import os
from pathlib import Path
import re

for name, value in {"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                    "ANONYMIZED_TELEMETRY": "False", "CREWAI_DISABLE_TELEMETRY": "true",
                    "OTEL_SDK_DISABLED": "true"}.items():
    os.environ[name] = value

ROOT = Path(__file__).resolve().parent
CHROMA_PATH = ROOT / "chroma"
CALIBRATION_PATH = ROOT / "outputs" / "calibration.json"
STRATEGIES = ("fixed_overlap", "sentence")
STRATEGY = "sentence"
CHUNK_SIZE, OVERLAP, SENTENCES_PER_CHUNK, TOP_K = 320, 80, 2, 3
MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_REVISION = "c9745ed1d9f207416be6d2e6f8de32d1f16199bf"
FALLBACK = "I don't know based on the available clinic-policy context."
REQUIRED_TOPICS = (
    "01_appointment_booking", "02_cancellation_rescheduling", "03_consultation_fees",
    "04_insurance_claims", "05_prescription_refills", "06_lab_turnaround",
    "07_telemedicine", "08_emergency_protocol", "09_patient_privacy",
    "10_follow_up_discount", "11_second_opinion", "12_home_visits",
)

# Read each policy body in filename order and collapse its whitespace.
def load_documents():
    documents = {}
    for path in sorted((ROOT / "kb").glob("*.md")):
        lines = [line for line in path.read_text(encoding="utf-8").splitlines()
                 if not line.lstrip().startswith("#")]
        documents[path.stem] = " ".join(" ".join(lines).split())
    if not documents or not all(documents.values()):
        raise ValueError("The kb folder must contain nonempty Markdown policies.")
    return documents

# Check every required topic filename and count each policy's sentences.
def validate_kb():
    documents = load_documents()
    missing = sorted(set(REQUIRED_TOPICS) - set(documents))
    if missing:
        raise ValueError("Missing required policy files: " + ", ".join(name + ".md" for name in missing))
    for parent, body in documents.items():
        count = len(re.split(r"(?<=[.!?])\s+", body))
        print(f"{parent}: {count} sentences")
        if not 2 <= count <= 5:
            raise ValueError(f"{parent} must have two to five sentences.")
    print(f"KB validation: PASS; documents={len(documents)}; required filenames present")
    return True

# Make overlapping character windows or consecutive pairs of sentences.
def split_chunks(documents, strategy):
    if strategy not in STRATEGIES:
        raise ValueError(f"Choose one of {STRATEGIES}")
    chunks = []
    for parent, body in sorted(documents.items()):
        if strategy == "fixed_overlap":
            texts = []
            for start in range(0, len(body), CHUNK_SIZE - OVERLAP):
                texts.append(body[start:start + CHUNK_SIZE].strip())
                if start + CHUNK_SIZE >= len(body):
                    break
        else:
            sentences = re.split(r"(?<=[.!?])\s+", body)
            texts = [" ".join(sentences[i:i + SENTENCES_PER_CHUNK])
                     for i in range(0, len(sentences), SENTENCES_PER_CHUNK)]
        chunks.extend({"chunk_id": f"{parent}:{strategy}:{i:03d}",
                       "parent_id": parent, "text": text}
                      for i, text in enumerate(texts) if text)
    return chunks

# Load the cached checkpoint once and encode deterministically on the CPU.
@lru_cache(maxsize=1)
def model():
    import torch
    from sentence_transformers import SentenceTransformer
    torch.set_num_threads(1)
    return SentenceTransformer(MODEL_ID, revision=MODEL_REVISION, device="cpu",
                               local_files_only=True, token=False)

# Reuse a local Chroma client with telemetry disabled.
@lru_cache(maxsize=4)
def client(store=CHROMA_PATH):
    import chromadb
    from chromadb.config import Settings
    return chromadb.PersistentClient(path=str(store), settings=Settings(anonymized_telemetry=False))

# Delete and rebuild both cosine collections from the policy files.
def build_index(store=CHROMA_PATH):
    from chromadb.errors import NotFoundError
    counts = {}
    for strategy in STRATEGIES:
        try:
            client(store).delete_collection(f"practo_{strategy}")
        except NotFoundError:
            pass
        chunks = split_chunks(load_documents(), strategy)
        collection = client(store).create_collection(f"practo_{strategy}", embedding_function=None,
                     configuration={"hnsw": {"space": "cosine", "num_threads": 1}})
        vectors = model().encode([c["text"] for c in chunks], normalize_embeddings=True,
                                 convert_to_numpy=True, show_progress_bar=False).tolist()
        collection.upsert(ids=[c["chunk_id"] for c in chunks], embeddings=vectors,
                          documents=[c["text"] for c in chunks],
                          metadatas=[{"parent_id": c["parent_id"]} for c in chunks])
        counts[strategy] = collection.count()
        print(f"{strategy}: {counts[strategy]} chunks")
    print(f"Model: {MODEL_ID}; revision: {MODEL_REVISION}")
    print(f"Fixed size={CHUNK_SIZE}; overlap={OVERLAP}; sentence size={SENTENCES_PER_CHUNK}")
    print("HF_HUB_OFFLINE=1; ANONYMIZED_TELEMETRY=False; CREWAI_DISABLE_TELEMETRY=true")
    return counts

# Return only the three closest chunks, without reranking or filtering.
def search(query, strategy=STRATEGY, store=CHROMA_PATH):
    if strategy not in STRATEGIES or not query.strip():
        raise ValueError("Provide a nonempty query and a known strategy.")
    collection = client(store).get_collection(f"practo_{strategy}", embedding_function=None)
    vector = model().encode([query], normalize_embeddings=True, convert_to_numpy=True,
                            show_progress_bar=False).tolist()
    found = collection.query(query_embeddings=vector, n_results=TOP_K,
                             include=["documents", "metadatas", "distances"])
    return [{"chunk_id": chunk, "parent_id": meta["parent_id"], "text": text,
             "similarity": 1.0 - float(distance)}
            for chunk, text, meta, distance in zip(found["ids"][0], found["documents"][0],
                                                 found["metadatas"][0], found["distances"][0])]

# Quote the retrieved text, or refuse when its best similarity is too low.
def answer(query, strategy=STRATEGY, store=CHROMA_PATH, calibration_path=CALIBRATION_PATH):
    if not Path(calibration_path).is_file():
        raise FileNotFoundError("Missing outputs/calibration.json; run python evaluate.py calibration first.")
    threshold = json.loads(Path(calibration_path).read_text(encoding="utf-8"))["strategies"][strategy]["threshold"]
    hits = search(query, strategy, store)
    fallback = hits[0]["similarity"] < threshold
    return {"answer": FALLBACK if fallback else "\n\n".join(hit["text"] for hit in hits),
            "source_ids": [] if fallback else [hit["parent_id"] for hit in hits],
            "chunk_ids": [] if fallback else [hit["chunk_id"] for hit in hits],
            "similarity": hits[0]["similarity"], "threshold": threshold, "fallback": fallback}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("step", choices=("validate", "index"))
    args = parser.parse_args()
    validate_kb() if args.step == "validate" else build_index()
