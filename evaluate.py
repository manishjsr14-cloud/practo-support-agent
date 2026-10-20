"""Measure plain retrieval, calibrate fallback, and expose unsupported acceptances."""
# What: three measurement phases: calibration (pick each collection's answer/fallback
#   threshold), evaluation (precision/recall on the 9 labelled questions) and stress
#   (near-topic questions the policies cannot answer).
# Task: Part 1 (Tasks 3-5). Used by: run_part1.py (runs each phase as a command) and
#   tests/test_part1.py (imports document_metrics).

import argparse
import json

from queries import CALIBRATION_IN_SCOPE, CALIBRATION_OUT_OF_SCOPE, DEMO_QUERIES
from queries import STRESS_QUERIES, TOP_K
from rag import CALIBRATION_PATH, ROOT, STRATEGIES, answer, search

# Macro precision measured in the original Part 1 run; evaluate() prints whether the
# current run still matches it to six decimals. It is only compared, never used to choose.
HISTORICAL_BASELINE = {"fixed_overlap": 0.703704, "sentence": 0.759259}


# Measure both similarity clusters and save a midpoint only when they separate.
# Takes: nothing. Returns: {"strategies": {name: {"threshold", "measurements"}}} and
#   writes the same data to outputs/calibration.json.
# A "calibration threshold" is the similarity cut-off between "answer" and "fallback",
#   measured from fixed in-scope and out-of-scope questions instead of picked by hand.
def calibrate():
    result = {"strategies": {}}
    print("Cosine similarity = 1 - cosine distance")
    for strategy in STRATEGIES:
        measurements = []
        print(f"\nCalibration: {strategy}; top-k chunks: {TOP_K}")
        # Record the best (top-1) similarity for every calibration question.
        for label, questions in (("in_scope", CALIBRATION_IN_SCOPE),
                                 ("out_of_scope", CALIBRATION_OUT_OF_SCOPE)):
            for query in questions:
                similarity = search(query, strategy)[0]["similarity"]
                measurements.append({"label": label, "query": query,
                                     "similarity": similarity})
                print(f"{label} | {query} | top1={similarity:.9f}")
        # The highest out-of-scope score must stay below the lowest in-scope score.
        positives = [row["similarity"] for row in measurements if row["label"] == "in_scope"]
        negatives = [row["similarity"] for row in measurements if row["label"] == "out_of_scope"]
        lower, upper = max(negatives), min(positives)
        print(f"max_out={lower:.9f}; min_in={upper:.9f}")
        # If the two groups overlap, no single cut-off separates them, so stop visibly.
        if lower >= upper:
            raise ValueError("Calibration clusters overlap; no separating threshold exists")
        # Put the threshold exactly halfway between the two groups.
        threshold = (lower + upper) / 2
        print(f"Threshold = ({lower:.9f} + {upper:.9f}) / 2 = {threshold:.9f}")
        result["strategies"][strategy] = {"threshold": threshold, "measurements": measurements}
    # Save the measurements and thresholds so rag.answer() can read them later.
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved calibration: {CALIBRATION_PATH.relative_to(ROOT)}")
    return result


# Count relevant parent documents once, even when several chunks share a parent.
# Takes: the retrieved parent ids and the labelled relevant parent ids.
# Returns: the shared numerator, both denominators, precision and recall.
# "Precision" = relevant documents retrieved / all distinct documents retrieved.
# "Recall" = relevant documents retrieved / all documents labelled relevant.
# Using sets means two chunks from the same policy count as one document.
def document_metrics(parent_ids, relevant):
    retrieved, labelled = set(parent_ids), set(relevant)
    numerator = len(retrieved & labelled)
    return {
        "numerator": numerator,
        "precision_denominator": len(retrieved),
        "recall_denominator": len(labelled),
        # Guard against dividing by zero if nothing was retrieved or labelled.
        "precision": numerator / len(retrieved) if retrieved else 0.0,
        "recall": numerator / len(labelled) if labelled else 0.0,
    }


# Compare the same labelled questions with plain retrieval and report macro averages.
# Takes: nothing. Returns: per-strategy rows and averages, plus the recommended strategy.
# Why: this is the evidence used to choose between the two chunking strategies.
def evaluate():
    results = {"strategies": {}, "historical_baseline_match": {}}
    for strategy in STRATEGIES:
        rows = []
        print(f"\nPlain retrieval: {strategy}; top-k chunks: {TOP_K}")
        print("| Query | Relevant parents | Retrieved parents | Precision | Recall |")
        print("|---|---|---|---|---|")
        for item in DEMO_QUERIES:
            # Parent ids of the 3 retrieved chunks, duplicates removed, first-seen order kept.
            parents = list(dict.fromkeys(hit["parent_id"] for hit in search(item["query"], strategy)))
            metrics = document_metrics(parents, item["relevant_parents"])
            rows.append({"id": item["id"], "query": item["query"],
                         "retrieved_parents": parents, **metrics})
            # Print the full fraction, not just the result, so the arithmetic can be checked.
            n, p, r = (metrics[key] for key in
                       ("numerator", "precision_denominator", "recall_denominator"))
            print(f"| {item['id']} | {', '.join(item['relevant_parents'])} | {', '.join(parents)} "
                  f"| {n} relevant of {p} retrieved = {metrics['precision']:.6f} "
                  f"| {n} retrieved of {r} relevant = {metrics['recall']:.6f} |")
        # "Macro average" = the plain mean of the per-question scores, so each question
        # counts equally however many documents it has.
        precision = sum(row["precision"] for row in rows) / len(rows)
        recall = sum(row["recall"] for row in rows) / len(rows)
        print(f"| Macro average | | | {precision:.6f} | {recall:.6f} |")
        results["strategies"][strategy] = {"precision": precision, "recall": recall, "rows": rows}
        # Report whether this run reproduces the original Part 1 numbers.
        expected = HISTORICAL_BASELINE[strategy]
        matched = round(precision, 6) == expected and round(recall, 6) == 1.0
        results["historical_baseline_match"][strategy] = matched
        print(f"Historical plain baseline: precision={expected:.6f}, recall=1.000000; "
              f"current={precision:.6f}/{recall:.6f}; six-decimal match={matched}")
    # Selection rule: higher recall wins; then higher precision; then sentence on an exact
    # tie (True sorts above False). Python compares the tuples element by element.
    recommended = max(STRATEGIES, key=lambda name: (
        results["strategies"][name]["recall"], results["strategies"][name]["precision"],
        name == "sentence"))
    selected = results["strategies"][recommended]
    print(f"Recommendation: use {recommended} with recall {selected['recall']:.6f} and "
          f"precision {selected['precision']:.6f}, choosing by recall then precision "
          "and preferring sentence on an exact tie.")
    results["recommended_strategy"] = recommended
    return results


# Record unsupported-query decisions without changing either calibrated threshold.
# Takes: nothing (reads outputs/calibration.json). Returns: per-strategy answer rows.
# Why: shows honestly which off-topic questions are wrongly accepted by similarity alone.
def stress():
    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    results = {"strategies": {}}
    for strategy in STRATEGIES:
        rows = []
        print(f"\nUnsupported queries: {strategy}")
        # Ask each near-topic question through the normal answer path and print the decision.
        for query in STRESS_QUERIES:
            response = answer(query, strategy)
            rows.append({"query": query, **response})
            decision = "FALLBACK" if response["fallback"] else "ANSWER"
            print(f"{query} | top1={response['similarity']:.9f} | "
                  f"threshold={response['threshold']:.9f} | {decision}")
            print(f"Answer: {response['answer']}")
            print("Sources: " + (", ".join(response["source_ids"]) or "(none)"))
        # Compare the Sunday-hours question (third in STRESS_QUERIES, index 2) with the
        # weakest legitimate calibration question: if it scores higher, no threshold
        # could reject it without also rejecting a real question.
        weakest = min(row["similarity"] for row in calibration["strategies"][strategy]["measurements"]
                      if row["label"] == "in_scope")
        sunday = rows[2]["similarity"]
        print(f"Sunday cosine {sunday:.9f} above weakest calibrated in-scope {weakest:.9f}: "
              f"{sunday > weakest}")
        results["strategies"][strategy] = rows
    print("Unsupported answers expose similarity-only limits; no threshold was retuned.")
    return results


# Run one phase so the caller can save each actual output in a separate text file.
# Takes: the phase name from the command line. Returns: nothing.
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("calibration", "evaluation", "stress"))
    phase = parser.parse_args().phase
    if phase == "calibration":
        calibrate()
    elif phase == "evaluation":
        evaluate()
    else:
        stress()


if __name__ == "__main__":
    main()
