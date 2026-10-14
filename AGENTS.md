# AGENTS.md — Practo Domain Support Agent

## Simplicity rules

- Each file under about 150 lines. Plain functions, no classes unless needed.
- A short plain-English comment above every function.
- No fingerprints, no stale-index detection, no deployment.json, no custom reranker, no retrieval-policy versioning, no socket blocking. The index step deletes and rebuilds both collections each run.
- Retrieval is PLAIN top-3 vector search from one collection. Nothing else.
- Offline, deterministic, MOCK_LLM, zero API keys. Set HF_HUB_OFFLINE=1, ANONYMIZED_TELEMETRY=False and CREWAI_DISABLE_TELEMETRY=true in code.
- Keep: all-MiniLM-L6-v2; fixed chunks of 320 characters with 80 overlap; sentence chunks of 2 sentences; cosine similarity = 1 - distance; collection.upsert(); the parent document id in every chunk's metadata.
- Add .gitignore for .venv, __pycache__, and the Chroma folder.

## Scope and working rules

- Work only on Part 1, Tasks 1–5, in this new repository. Do not implement Parts 2–4.
- Keep supplied `dataset.py`, `data/appointments.json`, `kb/` and `requirements.txt` byte-identical. Their inherited formatting and length are exceptions to the new-file simplicity rules.
- Never rewrite the handwritten KB, hand-edit generated records, change dataset choices or silently relax dependency pins.
- Show planned files before coding. Use readable code and ordinary comments so I can explain it.
- Never invent measurements. Run the code and save real text output in `outputs/`; quote only numbers appearing there.
- Keep only the new repository's current outputs. Do not import old transcripts, failed-run history or migration evidence.
- No API calls, paid accounts, real patient data or binary deliverables. Dependencies and the embedding model must already be available for offline runtime.
- Commit in small steps with clear messages; update tests with each task. Delete `reference/` in the last commit.
- Report failed checks, skipped work and deviations plainly. Do not tune queries, labels, chunk sizes or retrieval to improve measured results.

## Fixed dataset design

- Seed: `42`; record count: `50`.
- Categories: General Medicine, Cardiology, Dermatology, Pediatrics, Orthopedics.
- Category weights in that order: `0.30, 0.15, 0.20, 0.20, 0.15`.
- Status weights: Completed `0.45`, Scheduled `0.25`, Cancelled `0.12`, Rescheduled `0.10`, No-Show `0.08`.
- Fees are uniform draws from category ranges, rounded to the nearest `50` INR:
  - General Medicine `300–600`; Pediatrics `400–800`; Dermatology `500–1000`.
  - Orthopedics `600–1200`; Cardiology `800–1500`; overall `300–1500` INR.
- `days_since_created` is a uniform integer `0–30`.
- `follow_up_required` is an independent Bernoulli draw with `p=0.20`.
- Assign one record to every category and every status before filling the remainder with weighted draws.
- Required fields: `record_id`, `category`, `status`, `consultation_fee_inr`, `days_since_created`, `follow_up_required`.
- Running the supplied generator must reproduce the supplied JSON, print category/status counts and report follow-up percentage.
- Verify category counts are at least `3`, all statuses occur and follow-up percentage is in `10–30%`; report a failure without editing the supplied files.
- Use this exact README sentence: "These ranges reflect typical Indian outpatient consultation fees, with general practice at the low end and cardiology at the high end because of specialist training and diagnostic overhead."
- Handwritten KB standard new-booking fees are General Medicine `500`, Cardiology `1200`, Dermatology `800`, Pediatrics `600`, Orthopedics `1000` INR. Existing appointment records remain authoritative for their own fees.

## Tasks and simple file responsibilities

- Task 1: validate and demonstrate the preserved dataset; save its real output.
- Task 2: validate the preserved handwritten KB. Its documents cover booking, cancellations/rescheduling, fees, insurance, refills, lab turnaround, telemedicine, emergencies, privacy, follow-ups, second opinions and home visits.
- Validate all twelve required topic filenames explicitly, as well as `2–5` sentences per document; do not edit the KB to satisfy a check.
- `rag.py`: read/validate KB documents, chunk them, load local MiniLM embeddings and delete/rebuild the two separate cosine Chroma collections with `upsert()` under `chroma/`.
- Each chunk must carry its parent document id in metadata; embed both strategies with the same model.
- Retrieval returns plain cosine top-3 chunks from the requested collection, without parent expansion, reranking or diversity selection.
- Its answer function accepts a named strategy and calibrated threshold; return verbatim retrieved context under `MOCK_LLM` or the fallback.
- Use the original top-1 similarity as the only Part 1 answer/fallback signal. Do not add keyword support checks or hide its limitations.
- Shared `queries.py`: hold the unchanged demo/evaluation queries and labels, calibration queries and deliberate out-of-scope examples.
- `evaluate.py`: calibrate both collections, deduplicate retrieved parent ids, print precision/recall arithmetic per query for both collections and print macro averages.
- Keep the selection rule: higher macro recall, then higher macro precision, then sentence chunks on an exact tie. Do not persist a deployment file.
- Calibration writes measured values and thresholds to `outputs/calibration.json`, without fingerprints or deployment settings.
- `demo.py`: use exactly the evaluation's fixed query list and a separate chocolate-brownie fallback, with the single manual `STRATEGY` constant from `rag.py`; also print sample retrieval from both collections.
- `run_part1.py`: save the dataset report, validate the KB, rebuild, calibrate, evaluate, stress, demonstrate and test in order; save the cancellation sample to `outputs/sample_retrieval.txt`.
- `tests/test_part1.py`: keep small readable checks for dataset/KB validity, both chunkers, parent metadata, plain retrieval, calibration arithmetic, visible metrics and fallback behavior.
- Save current `outputs/*.txt`, overwriting them each run; only calibration needs JSON. No history archive or old success claims.
- Keep the requested flat files; do not introduce `rag/` or `scripts/` packages.

## Fixed calibration queries

In scope:
- "How can I reserve a clinic appointment?"
- "How much does a Cardiology consultation cost?"
- "What documents are needed for an insurance claim?"
- "Who is allowed to authorize a prescription refill?"

Out of scope:
- "How do I repair a punctured bicycle tyre?"
- "What is the orbital period of Neptune?"
- "Write a sorting algorithm in Rust."

Measure and print every top-1 cosine similarity for each collection.
Require `max(out_of_scope) < min(in_scope)`; if the clusters overlap, fail visibly instead of guessing a threshold.
Otherwise use the midpoint `(max(out_of_scope) + min(in_scope)) / 2`.
Do not change these queries or select a preset threshold.

## Fixed demo/evaluation queries and labels

| Id | Query | Relevant parent ids |
|---|---|---|
| q01 | How do I book an appointment and know that my slot is confirmed? | 01_appointment_booking |
| q02 | What is the cancellation and rescheduling window? | 02_cancellation_rescheduling |
| q03 | What are the consultation fees for each specialty? | 03_consultation_fees |
| q04 | How long do routine and specialized lab test reports take? | 06_lab_turnaround |
| q05 | When does a follow-up visit qualify for a discount? | 10_follow_up_discount |
| q06 | How do I submit an insurance reimbursement claim? | 04_insurance_claims |
| q07 | My doctor asked me to come back, do I have to pay the full price again? | 10_follow_up_discount |
| q08 | What will I pay for a home visit by a cardiologist? | 12_home_visits, 03_consultation_fees |
| q09 | Can I get a second opinion over a video call? | 11_second_opinion, 07_telemedicine |

Precision is relevant retrieved parents divided by all deduplicated retrieved parents.
Recall is relevant retrieved parents divided by all labelled relevant parents.
Show both arithmetic expressions and both collections' averages. Recommend a strategy in one sentence using actual output, including ties or misses.

## Near-topic stress check

Measure both collections without changing calibration:
- "Can you diagnose my chest pain?"
- "Which medicine should I take for a fever?"
- "What are the clinic's opening hours on Sundays?"
- "Which hospital in Mumbai has the shortest waiting time?"

Print each top-1 similarity, the recommended strategy's threshold and answer/fallback behavior.
Report every unsupported acceptance honestly, especially the Sunday-hours case. Do not raise the threshold or add a keyword hack to conceal it.
Similarity alone cannot prove that a passage answers a question; any future groundedness guardrail belongs to an explicitly requested later Part.

## README and completion

- Write in first person, under about `120` lines. State that this is the Practo (Healthcare) Part 1 project; do not claim later Parts are completed.
- Explain the dataset choices, measured calibration, plain-retrieval evaluation, recommendation and limitations using only current `outputs/` evidence.
- Include PowerShell and Bash run commands, offline flags, zero-key `MOCK_LLM` behavior and the exact fee justification.
- Include `TODO (in my own words): what I learned` for me to complete.
- Run the tests and the complete offline pipeline before marking Part 1 complete; record their actual output and any failures.

## Session log

| Part | Status | Notes |
|---|---|---|
| Part 1 | complete; simple rewrite | Plain top-3 baseline reproduced; all 11 tests and the full pipeline passed. Supplied inputs unchanged. Sunday-hours and Mumbai unsupported acceptances retained. Current evidence is in outputs/. |
| Part 2 | not started | Out of scope. |
| Part 3 | not started | Out of scope. |
| Part 4 | not started | Out of scope. |
