"""Print the actual retrieved context and extractive answers for Part 1."""
# What: prints the answers for the 9 evaluation questions plus one unrelated question,
#   or (with --sample) the raw top-3 chunks for one question from both collections.
# Task: Part 1 (Task 5, demo). Used by: run_part1.py, which runs "demo.py --sample"
#   (saved as outputs/sample_retrieval.txt) and "demo.py" (saved as outputs/demo.txt).

import argparse

from queries import DEMO_OUT_OF_SCOPE, DEMO_QUERIES
from rag import STRATEGIES, STRATEGY, answer, search


# Show the exact cancellation query and all three results from each collection.
# Takes: nothing. Returns: nothing; prints each chunk id, parent, similarity and text.
# Why: lets a reader see what the two chunking strategies actually retrieve.
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
# Takes: the optional --sample flag from the command line. Returns: nothing.
# Why: shows the complete answer path, including the fallback, with the chosen STRATEGY.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", action="store_true", help="Print the sample query from both collections")
    if parser.parse_args().sample:
        sample_retrieval()
        return
    # The same 9 questions as the evaluation, plus the unrelated brownie question last.
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
        # Stop with an error if the unrelated question was answered instead of refused.
        if question == DEMO_OUT_OF_SCOPE and not result["fallback"]:
            raise RuntimeError("The unrelated brownie query must use the fallback")
    print("\nDemo completed; inspect the quoted context to assess each answer.")


if __name__ == "__main__":
    main()
