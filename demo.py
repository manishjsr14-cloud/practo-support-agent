"""Print the actual retrieved context and extractive answers for Part 1."""

import argparse

from queries import DEMO_OUT_OF_SCOPE, DEMO_QUERIES
from rag import STRATEGIES, STRATEGY, answer, search


# Show the exact cancellation query and all three results from each collection.
def sample_retrieval():
    question = "What is the cancellation and rescheduling window?"
    print(f"Question: {question}")
    print("Cosine similarity = 1 - cosine distance")
    for strategy in STRATEGIES:
        print(f"\nCollection strategy: {strategy}")
        for hit in search(question, strategy=strategy):
            print(f"Chunk: {hit['chunk_id']}; parent: {hit['parent_id']}; "
                  f"cosine similarity: {hit['similarity']:.9f}")
            print(f"Retrieved text: {hit['text']}")


# Run either sample retrieval or the fixed questions with their unedited mock answers.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", action="store_true", help="Print the sample query from both collections")
    if parser.parse_args().sample:
        sample_retrieval()
        return
    questions = [row["query"] for row in DEMO_QUERIES] + [DEMO_OUT_OF_SCOPE]
    print(f"MOCK_LLM=true; strategy={STRATEGY}")
    for question in questions:
        result = answer(question, strategy=STRATEGY)
        print(f"\nQuestion: {question}")
        print(f"Similarity: {result['similarity']:.9f}; threshold: {result['threshold']:.9f}")
        print(f"Fallback: {result['fallback']}")
        print(f"Answer: {result['answer']}")
        print("Sources: " + (", ".join(result["source_ids"]) or "(none)"))
        print("Cited chunks: " + (", ".join(result["chunk_ids"]) or "(none)"))
        if question == DEMO_OUT_OF_SCOPE and not result["fallback"]:
            raise RuntimeError("The unrelated brownie query must use the fallback")
    print("\nDemo completed; inspect the quoted context to assess each answer.")


if __name__ == "__main__":
    main()
