# Practo (Healthcare) — Part 1 completed

I built a local policy retriever with deterministic `MOCK_LLM` answers and zero API keys. My dataset has 50 fabricated appointments, including 13 follow-ups (26.00%), and my handwritten KB has 12 policies.

## Dataset choices

I use seed 42, reserving one record per category and status before weighted draws. Fees are uniform draws rounded to the nearest 50 INR; ages are uniform integers from 0–30 days; follow-up flags use Bernoulli p = 0.20.

| Category | Weight | Fee range INR | Records |
|---|---:|---:|---:|
| General Medicine | 0.30 | 300–600 | 12 |
| Cardiology | 0.15 | 800–1500 | 8 |
| Dermatology | 0.20 | 500–1000 | 11 |
| Pediatrics | 0.20 | 400–800 | 8 |
| Orthopedics | 0.15 | 600–1200 | 11 |

Status weights: Completed 0.45, Scheduled 0.25, Cancelled 0.12, Rescheduled 0.10, No-Show 0.08. Overall fee range: 300–1500 INR; required follow-up band: 10–30%. These ranges reflect typical Indian outpatient consultation fees, with general practice at the low end and cardiology at the high end because of specialist training and diagnostic overhead.

I reused my dataset, KB and requirements. KB standard prices apply to new bookings; an existing appointment’s recorded fee remains authoritative. [Dataset](outputs/dataset.txt) · [Named-topic and sentence validation](outputs/kb.txt).

## Run

I verified Python 3.12.10 on Windows. Setup needs internet once to install dependencies and cache the pinned model. Run from the repository root.

PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', revision='c9745ed1d9f207416be6d2e6f8de32d1f16199bf')"
.\.venv\Scripts\python.exe run_part1.py
```

Bash:

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', revision='c9745ed1d9f207416be6d2e6f8de32d1f16199bf')"
.venv/bin/python run_part1.py
```

The runner overwrites current `outputs/` reports: dataset, KB validation, index, sample retrieval, calibration, evaluation, stress, demo and tests. I use CPU MiniLM with local files only; code sets `HF_HUB_OFFLINE=1`, `ANONYMIZED_TELEMETRY=False` and `CREWAI_DISABLE_TELEMETRY=true` before framework imports.

Both collections are deleted/rebuilt using `upsert()` with parent IDs. Fixed chunks use 320 characters with 80 overlap (39 chunks); sentence chunks use 2 sentences (36 chunks). Search returns the closest 3 chunks directly; similarity = 1 − cosine distance. [Index](outputs/index.txt) · [Sample chunks from both collections](outputs/sample_retrieval.txt).

## Calibration

I use the unchanged questions and midpoint `(max_out + min_in) / 2`; overlapping clusters fail and missing calibration raises a clear error. [Measurements](outputs/calibration.txt) · [Saved thresholds](outputs/calibration.json).

| Scope | Query | Fixed cosine | Sentence cosine |
|---|---|---:|---:|
| in_scope | How can I reserve a clinic appointment? | 0.734329104 | 0.746177197 |
| in_scope | How much does a Cardiology consultation cost? | 0.674873888 | 0.708590269 |
| in_scope | What documents are needed for an insurance claim? | 0.440953672 | 0.486387312 |
| in_scope | Who is allowed to authorize a prescription refill? | 0.605015993 | 0.600208402 |
| out_of_scope | How do I repair a punctured bicycle tyre? | 0.145102620 | 0.152278662 |
| out_of_scope | What is the orbital period of Neptune? | 0.143519402 | 0.147324979 |
| out_of_scope | Write a sorting algorithm in Rust. | 0.075274765 | 0.082972288 |

| Strategy | Max out | Min in | Threshold |
|---|---:|---:|---:|
| fixed_overlap | 0.145102620 | 0.440953672 | 0.293028146 |
| sentence | 0.152278662 | 0.486387312 | 0.319332987 |

## Plain top-3 evaluation

I use the same 9 labelled questions in evaluation and demo, deduplicate retrieved parents, and average per-query precision and recall. [Full arithmetic and parent sets](outputs/evaluation.txt) · [Comparison and demo check](outputs/readme_check.txt).

fixed_overlap:

| Query | Relevant parents | Retrieved parents | Precision | Recall |
|---|---|---|---|---|
| q01 | 01_appointment_booking | 01_appointment_booking | 1 relevant of 1 retrieved = 1.000000 | 1 retrieved of 1 relevant = 1.000000 |
| q02 | 02_cancellation_rescheduling | 02_cancellation_rescheduling, 12_home_visits | 1 relevant of 2 retrieved = 0.500000 | 1 retrieved of 1 relevant = 1.000000 |
| q03 | 03_consultation_fees | 03_consultation_fees, 10_follow_up_discount, 07_telemedicine | 1 relevant of 3 retrieved = 0.333333 | 1 retrieved of 1 relevant = 1.000000 |
| q04 | 06_lab_turnaround | 06_lab_turnaround, 04_insurance_claims | 1 relevant of 2 retrieved = 0.500000 | 1 retrieved of 1 relevant = 1.000000 |
| q05 | 10_follow_up_discount | 10_follow_up_discount, 03_consultation_fees | 1 relevant of 2 retrieved = 0.500000 | 1 retrieved of 1 relevant = 1.000000 |
| q06 | 04_insurance_claims | 04_insurance_claims | 1 relevant of 1 retrieved = 1.000000 | 1 retrieved of 1 relevant = 1.000000 |
| q07 | 10_follow_up_discount | 10_follow_up_discount, 02_cancellation_rescheduling | 1 relevant of 2 retrieved = 0.500000 | 1 retrieved of 1 relevant = 1.000000 |
| q08 | 12_home_visits, 03_consultation_fees | 03_consultation_fees, 12_home_visits | 2 relevant of 2 retrieved = 1.000000 | 2 retrieved of 2 relevant = 1.000000 |
| q09 | 11_second_opinion, 07_telemedicine | 11_second_opinion, 07_telemedicine | 2 relevant of 2 retrieved = 1.000000 | 2 retrieved of 2 relevant = 1.000000 |
| Macro average | | | 0.703704 | 1.000000 |

sentence:

| Query | Relevant parents | Retrieved parents | Precision | Recall |
|---|---|---|---|---|
| q01 | 01_appointment_booking | 01_appointment_booking | 1 relevant of 1 retrieved = 1.000000 | 1 retrieved of 1 relevant = 1.000000 |
| q02 | 02_cancellation_rescheduling | 02_cancellation_rescheduling | 1 relevant of 1 retrieved = 1.000000 | 1 retrieved of 1 relevant = 1.000000 |
| q03 | 03_consultation_fees | 03_consultation_fees, 07_telemedicine, 10_follow_up_discount | 1 relevant of 3 retrieved = 0.333333 | 1 retrieved of 1 relevant = 1.000000 |
| q04 | 06_lab_turnaround | 06_lab_turnaround | 1 relevant of 1 retrieved = 1.000000 | 1 retrieved of 1 relevant = 1.000000 |
| q05 | 10_follow_up_discount | 10_follow_up_discount, 03_consultation_fees | 1 relevant of 2 retrieved = 0.500000 | 1 retrieved of 1 relevant = 1.000000 |
| q06 | 04_insurance_claims | 04_insurance_claims | 1 relevant of 1 retrieved = 1.000000 | 1 retrieved of 1 relevant = 1.000000 |
| q07 | 10_follow_up_discount | 03_consultation_fees, 10_follow_up_discount, 12_home_visits | 1 relevant of 3 retrieved = 0.333333 | 1 retrieved of 1 relevant = 1.000000 |
| q08 | 12_home_visits, 03_consultation_fees | 03_consultation_fees, 12_home_visits, 04_insurance_claims | 2 relevant of 3 retrieved = 0.666667 | 2 retrieved of 2 relevant = 1.000000 |
| q09 | 11_second_opinion, 07_telemedicine | 11_second_opinion, 07_telemedicine | 2 relevant of 2 retrieved = 1.000000 | 2 retrieved of 2 relevant = 1.000000 |
| Macro average | | | 0.759259 | 1.000000 |

I tentatively choose sentence chunks: macro precision is 0.759259 versus 0.703704 for fixed chunks, with recall 1.000000 for both. Sentence chunks have higher precision on q02 and q04, fixed chunks on q07 and q08, and the other five tie. Sentence has the higher average only because its two wins are larger; 9 queries is a small sample, so this choice is only weakly supported. I set the single `STRATEGY` constant in `rag.py` to `sentence`; evaluation never writes it.

The demo answered 9 questions with retrieved text and fell back on “Give me a recipe for chocolate brownies.” at similarity 0.054437876. [Demo output](outputs/demo.txt).

## Limits

I quote retrieved text and citations rather than provide reasoned medical answers. Parent recall does not prove completeness: q08’s home-visit price chunk omits the eligibility restriction. Fixed windows can cut sentences. [Demo](outputs/demo.txt). I measured unsupported questions separately from calibration. [Stress output](outputs/near_topic.txt).

| Unsupported question | Fixed cosine | Sentence cosine | Sentence decision |
|---|---:|---:|---|
| Can you diagnose my chest pain? | 0.256390333 | 0.263177156 | FALLBACK |
| Which medicine should I take for a fever? | 0.272614956 | 0.202158451 | FALLBACK |
| What are the clinic's opening hours on Sundays? | 0.486783445 | 0.524604023 | ANSWER |
| Which hospital in Mumbai has the shortest waiting time? | 0.421420097 | 0.439948082 | ANSWER |

The Sunday-hours sentence score, 0.524604023, exceeds the weakest legitimate calibration score, 0.486387312 (insurance). Raising the threshold to reject Sunday-hours would reject that legitimate question too. Sunday-hours and Mumbai waiting-time questions are falsely accepted; similarity alone cannot establish support. I did not retune the threshold.

All 11 tests and the full pipeline passed, as did `pip check`. [Tests](outputs/tests.txt) · [Dependencies](outputs/setup.txt). Bash execution and downloading into an empty model cache were not verified; the Windows run used the cached checkpoint.

## What I learned: 
A similarity score measures how close a question sounds to a policy
text; it doesn't show that the text answers the question. The clearest example is the
Sunday-opening-hours question, which my policies can't answer but which scored 0.525,
higher than my weakest legitimate question (the insurance query at 0.486). So no single
threshold could reject it without also rejecting a real question. The demo showed the
same thing: the answer to the cardiologist home-visit question quoted the fees and the
INR 800 home-visit price but never said that home visits are only for General Medicine,
so retrieval can miss important text and still look like a good answer.

I also learned that different chunking strategies suit different situations, and that
my choice between them is weaker than the averages suggest. Sentence chunks scored 0.759
against 0.704 for fixed chunks, but each strategy won two of the nine queries (sentence
on q02 and q04, fixed on q07 and q08), and the gap comes from the size of the wins, so
nine queries is a small sample. In Part 2 I will add a check that the question's key
terms actually appear in the retrieved text.

## Results in two environments

I produced the main tables above from `outputs/` with `requirements-part1.txt`; my unified `requirements.txt` for Parts 2–4 gives identical retrieval and decisions with similarities within `4e-7`, and I saved its results in [`outputs/unified_env/`](outputs/unified_env/).

| Item | Part 1 pins | Unified environment |
|---|---:|---:|
| Fixed threshold | 0.293028146 | 0.293028086 |
| Sentence threshold | 0.319332987 | 0.319333017 |
| Fixed macro precision | 0.703704 | 0.703704 |
| Sentence macro precision | 0.759259 | 0.759259 |
| Macro recall (both) | 1.000000 | 1.000000 |
| Sunday-hours sentence score | 0.524604023 | 0.524603963 |
| Weakest in-scope (insurance) sentence score | 0.486387312 | 0.486387372 |
| Tests passed | 11 | 11 |

I treat differences in the seventh decimal place and beyond as floating-point noise; no retrieved chunk, answer or ANSWER/FALLBACK decision changed.

## Part 2

### Escalation score

I calculate `escalation_score = 0.6 * (1 if follow_up_required else 0) + 0.4 * (days_since_created / 30)`.

I give the follow-up flag weight 0.6 because it represents an explicit administrative action; record age has weight 0.4 because older records deserve review, normalized over 30 days.

| Statistic | Escalation score |
|---|---:|
| Minimum | 0.000000 |
| Median | 0.320000 |
| 80th percentile | 0.693333 |
| Maximum | 0.986667 |
| Mean | 0.372000 |

I use the 80th percentile with NumPy's default linear interpolation: `0.6933333333333334`. The rule `score >= threshold` selects 11/50 records, including 11/13 follow-up records; ties are included.

This score is an administrative follow-up flag, not a medical assessment; I never use it to judge medical urgency ([KB08](kb/08_emergency_protocol.md)).

PowerShell: `.\.venv\Scripts\python.exe run_task6.py`. [Measured report](outputs/part2_escalation.txt) · [Test results](outputs/part2_tests.txt).

## What I learned in Part 2

[I will write this myself]
