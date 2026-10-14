"""Measure plain retrieval, calibrate fallback, and expose unsupported acceptances."""

import argparse
import json

from queries import CALIBRATION_IN_SCOPE, CALIBRATION_OUT_OF_SCOPE, DEMO_QUERIES
from queries import STRESS_QUERIES, TOP_K
from rag import CALIBRATION_PATH, ROOT, STRATEGIES, answer, search

HISTORICAL_BASELINE = {"fixed_overlap": 0.703704, "sentence": 0.759259}


# Measure both similarity clusters and save a midpoint only when they separate.
def calibrate():
    result = {"strategies": {}}
    print("Cosine similarity = 1 - cosine distance")
    for strategy in STRATEGIES:
        measurements = []
        print(f"\nCalibration: {strategy}; top-k chunks: {TOP_K}")
        for label, questions in (("in_scope", CALIBRATION_IN_SCOPE),
                                 ("out_of_scope", CALIBRATION_OUT_OF_SCOPE)):
            for query in questions:
                similarity = search(query, strategy)[0]["similarity"]
                measurements.append({"label": label, "query": query,
                                     "similarity": similarity})
                print(f"{label} | {query} | top1={similarity:.9f}")
        positives = [row["similarity"] for row in measurements if row["label"] == "in_scope"]
        negatives = [row["similarity"] for row in measurements if row["label"] == "out_of_scope"]
        lower, upper = max(negatives), min(positives)
        print(f"max_out={lower:.9f}; min_in={upper:.9f}")
        if lower >= upper:
            raise ValueError("Calibration clusters overlap; no separating threshold exists")
        threshold = (lower + upper) / 2
        print(f"Threshold = ({lower:.9f} + {upper:.9f}) / 2 = {threshold:.9f}")
        result["strategies"][strategy] = {"threshold": threshold, "measurements": measurements}
    CALIBRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_PATH.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"Saved calibration: {CALIBRATION_PATH.relative_to(ROOT)}")
    return result


# Count relevant parent documents once, even when several chunks share a parent.
def document_metrics(parent_ids, relevant):
    retrieved, labelled = set(parent_ids), set(relevant)
    numerator = len(retrieved & labelled)
    return {
        "numerator": numerator,
        "precision_denominator": len(retrieved),
        "recall_denominator": len(labelled),
        "precision": numerator / len(retrieved) if retrieved else 0.0,
        "recall": numerator / len(labelled) if labelled else 0.0,
    }


# Compare the same labelled questions with plain retrieval and report macro averages.
def evaluate():
    results = {"strategies": {}, "historical_baseline_match": {}}
    for strategy in STRATEGIES:
        rows = []
        print(f"\nPlain retrieval: {strategy}; top-k chunks: {TOP_K}")
        print("| Query | Relevant parents | Retrieved parents | Precision | Recall |")
        print("|---|---|---|---|---|")
        for item in DEMO_QUERIES:
            parents = list(dict.fromkeys(hit["parent_id"] for hit in search(item["query"], strategy)))
            metrics = document_metrics(parents, item["relevant_parents"])
            rows.append({"id": item["id"], "query": item["query"],
                         "retrieved_parents": parents, **metrics})
            n, p, r = (metrics[key] for key in
                       ("numerator", "precision_denominator", "recall_denominator"))
            print(f"| {item['id']} | {', '.join(item['relevant_parents'])} | {', '.join(parents)} "
                  f"| {n} relevant of {p} retrieved = {metrics['precision']:.6f} "
                  f"| {n} retrieved of {r} relevant = {metrics['recall']:.6f} |")
        precision = sum(row["precision"] for row in rows) / len(rows)
        recall = sum(row["recall"] for row in rows) / len(rows)
        print(f"| Macro average | | | {precision:.6f} | {recall:.6f} |")
        results["strategies"][strategy] = {"precision": precision, "recall": recall, "rows": rows}
        expected = HISTORICAL_BASELINE[strategy]
        matched = round(precision, 6) == expected and round(recall, 6) == 1.0
        results["historical_baseline_match"][strategy] = matched
        print(f"Historical plain baseline: precision={expected:.6f}, recall=1.000000; "
              f"current={precision:.6f}/{recall:.6f}; six-decimal match={matched}")
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
def stress():
    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8"))
    results = {"strategies": {}}
    for strategy in STRATEGIES:
        rows = []
        print(f"\nUnsupported queries: {strategy}")
        for query in STRESS_QUERIES:
            response = answer(query, strategy)
            rows.append({"query": query, **response})
            decision = "FALLBACK" if response["fallback"] else "ANSWER"
            print(f"{query} | top1={response['similarity']:.9f} | "
                  f"threshold={response['threshold']:.9f} | {decision}")
            print(f"Answer: {response['answer']}")
            print("Sources: " + (", ".join(response["source_ids"]) or "(none)"))
        weakest = min(row["similarity"] for row in calibration["strategies"][strategy]["measurements"]
                      if row["label"] == "in_scope")
        sunday = rows[2]["similarity"]
        print(f"Sunday cosine {sunday:.9f} above weakest calibrated in-scope {weakest:.9f}: "
              f"{sunday > weakest}")
        results["strategies"][strategy] = rows
    print("Unsupported answers expose similarity-only limits; no threshold was retuned.")
    return results


# Run one phase so the caller can save each actual output in a separate text file.
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
