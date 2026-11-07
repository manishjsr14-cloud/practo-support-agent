"""Compare groundedness v1 and v2 on three pre-registered sets and apply the adoption rule."""
# What: prints both thresholds' calibration, runs every question once through pipeline.ask
#   (v1, the default) and decides Stage 2 both ways on the same retrieval and the same term
#   scores, with every term printed. Then one v1 vs v2 table per set and ADOPTION_RULE,
#   applied to groundedness_v2_queries.py only.
# Task: groundedness v2 experiment. Used by: run by hand; it writes outputs/groundedness_v2.txt.
# Outcomes: ANSWER or REFUSE from the policy check; EMERGENCY when the emergency phrase rule
#   replied (no groundedness check ran). A turn with no policy check has the same outcome
#   under both versions. EMERGENCY counts as neither a false refusal nor a false acceptance.

import settings  # noqa: F401  (sets the offline flags before any framework import)

from pathlib import Path

from eval_queries import EVAL_QUERIES
import groundedness_v2
from groundedness_v2_queries import ADOPTION_RULE, SHOULD_ANSWER, SHOULD_REFUSE
import guardrail_queries
from guardrails import groundedness_threshold
from pipeline import LAST_RUN, ask

SETS = [
    ("groundedness_v2_queries.py", [(q, "ANSWER", parent) for q, parent in SHOULD_ANSWER]
     + [(q, "REFUSE", None) for q in SHOULD_REFUSE]),
    ("guardrail_queries.py", [(q, "ANSWER", None) for q in guardrail_queries.SHOULD_ANSWER]
     + [(q, "REFUSE", None) for q in guardrail_queries.SHOULD_REFUSE]),
    ("eval_queries.py", [(c["query"], c["expect"].upper(), None) for c in EVAL_QUERIES]),
]


# Run one question and decide it under v1 and v2.
# Takes: a session id, the question, the expected outcome and parent id (or None).
# Returns: (v1 outcome, v2 outcome, report lines).
def decide(session_id, question, expected, parent):
    response = ask(session_id, question)
    lines = ["", f"[{session_id}] expected {expected}: {question}"]
    if LAST_RUN.get("emergency"):
        return "EMERGENCY", "EMERGENCY", lines + ["  emergency phrase rule replied; no policy check ran"]
    check = LAST_RUN.get("guardrail")
    if check is None:
        same = "REFUSE" if response.refused else "ANSWER"
        return same, same, lines + [f"  no policy check ran; outcome {same} under both versions"]
    rag = LAST_RUN["result"]["rag_results"][-1]
    report = check["report"]
    lines += [f"  text sent to rag_lookup: {check['query']!r}; top-3 sources: {rag['source_ids']}",
              f"  Stage 1 top-1 similarity: {rag['similarity']:.9f} (threshold {rag['threshold']:.9f})"]
    lines += [f"    term {t!r}: support {s:.9f} ({how})" for t, s, how in report["terms"]] or ["    (no content terms)"]
    if parent:
        lines.append(f"  expected parent {parent} in top-3 sources: {parent in rag['source_ids']}")
    v1 = "ANSWER" if check["ok"] else "REFUSE"
    if rag["fallback"]:
        v2, reason2 = v1, check["reason"]
    else:
        ok2, reason2 = groundedness_v2.stage2_v2(report)
        v2 = "ANSWER" if ok2 else "REFUSE"
    weakest = "none" if report["weakest"] is None else f"{report['weakest']:.9f}"
    average = groundedness_v2.average_support(report)
    lines += [f"  v1: weakest {weakest} -> {v1}; reason: {check['reason'] or '(none)'}",
              f"  v2: average {'none' if average is None else f'{average:.9f}'} -> {v2}; reason: {reason2 or '(none)'}"]
    return v1, v2, lines


# Count false refusals and false acceptances for one version.
# Takes: (expected, outcome) pairs. Returns: (false refusals, false acceptances).
def errors(pairs):
    return (sum(e == "ANSWER" and o == "REFUSE" for e, o in pairs),
            sum(e == "REFUSE" and o == "ANSWER" for e, o in pairs))


# Run all three sets and build the report text. Takes: nothing. Returns: the report.
def make_report():
    t1, t2 = groundedness_threshold()["threshold"], groundedness_v2.threshold_v2()
    lines = ["Groundedness v2 experiment: Stage 2 by average term support (v2) vs weakest term (v1)",
             "Stage 1 unchanged; both versions use the same retrieval and the same term scores.",
             f"v1 T = {t1:.9f} (unchanged, from guardrail_calibration.py)",
             "v2 T2 calibration (guardrail_calibration.py only; average term support per question):"]
    for group in ("refuse", "answer"):
        lines += [f"  [{group}] {q}: average {a:.9f}" for q, a in t2[group].items()]
    high, low = max(t2["refuse"].values()), min(t2["answer"].values())
    lines.append(f"T2 = ({high:.9f} + {low:.9f}) / 2 = {t2['threshold']:.9f}")
    summaries, adoption = [], None
    for number, (name, cases) in enumerate(SETS, start=1):
        lines += ["", "=" * 72, f"Set {number}: {name}"]
        pairs1, pairs2, rows = [], [], []
        for index, (question, expected, parent) in enumerate(cases, start=1):
            v1, v2, detail = decide(f"gv2-{number}-{index:02d}", question, expected, parent)
            lines += detail
            pairs1.append((expected, v1))
            pairs2.append((expected, v2))
            rows.append(f"| {index:02d} | {expected} | {v1} | {v2} | {question} |")
        (fr1, fa1), (fr2, fa2) = errors(pairs1), errors(pairs2)
        lines += ["", "| # | Expected | v1 | v2 | Question |", "|---|---|---|---|---|"] + rows
        summary = f"| {name} | {len(cases)} | {fr1} | {fa1} | {fr2} | {fa2} |"
        lines.append(summary)
        summaries.append(summary)
        if number == 1:
            adoption = (fr1, fa1, fr2, fa2)
    lines += ["", "| Set | Questions | v1 false refusals | v1 false acceptances | v2 false refusals | "
              "v2 false acceptances |", "|---|---:|---:|---:|---:|---:|"] + summaries
    fr1, fa1, fr2, fa2 = adoption
    adopted = fr2 < fr1 and fa2 <= fa1
    lines += ["", f"ADOPTION_RULE: {ADOPTION_RULE}",
              f"On groundedness_v2_queries.py: v1 {fr1} false refusals, {fa1} false acceptances; "
              f"v2 {fr2} false refusals, {fa2} false acceptances.",
              f"Fewer false refusals ({fr2} < {fr1}): {fr2 < fr1}; "
              f"no more false acceptances ({fa2} <= {fa1}): {fa2 <= fa1}",
              f"Outcome: {'ADOPTED' if adopted else 'NOT ADOPTED'}"]
    return "\n".join(lines) + "\n"


# Save the report to outputs/groundedness_v2.txt and print it.
def main():
    report = make_report()
    target = Path(__file__).resolve().parent / "outputs" / "groundedness_v2.txt"
    target.write_text(report, encoding="utf-8")
    print(report, end="")


if __name__ == "__main__":
    main()
