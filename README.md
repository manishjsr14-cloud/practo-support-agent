# Practo (Healthcare) — Part 1 complete, Part 2 in progress

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

The runner overwrites current `outputs/` reports: dataset, KB validation, index, sample retrieval, calibration, evaluation, stress, demo and tests. I use CPU MiniLM with local files only; code sets `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`, `ANONYMIZED_TELEMETRY=False`, `CREWAI_DISABLE_TELEMETRY=true`, `OTEL_SDK_DISABLED=true` and `HF_HUB_DISABLE_PROGRESS_BARS=1` before framework imports. [`outputs/index.txt`](outputs/index.txt) prints a fixed subset of three of them.

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

## What I learned in Part 1
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

PowerShell: `.\.venv\Scripts\python.exe run_task6.py`. Bash: `.venv/bin/python run_task6.py`. [Measured report](outputs/part2_escalation.txt) · [Test results](outputs/part2_tests.txt).

### Crew

I run a sequential CrewAI crew driven by a deterministic `MockLLM`, offline and with zero API keys. The Retrieval Agent has only `rag_lookup` (sentence collection, saved threshold), the Lookup Agent alone has `check_appointment_status`, and the Response Composer has no tools and only joins their outputs under labels, adding the KB08 note when a record is shown. The mock accepts a tool result only from its own latest `Action` with an exact tool name, JSON input and a real `Observation`, and fills tool arguments from each tool's schema.

| Question | rag_lookup | check_appointment_status |
|---|---:|---:|
| What is the cancellation and rescheduling window? | 1 | 0 |
| What is the status of APT-007? | 0 | 1 |
| What is the status of APT-007 and can I reschedule it? | 1 | 1 |

Every LLM call came from a real agent and task with no `tools` argument, so the ReAct path ran. `settings.py` sets all offline flags before CrewAI loads, including `CREWAI_TRACING_ENABLED=false`; the run shows CrewAI's storage at the repository's `.crewai/` folder.

The Retrieval Agent now sends only the policy clause; the appointment id belongs to the Lookup Agent. Query 3, before (previous committed run, `46e466c`) and after (current output):

| Query 3 | Text sent to `rag_lookup` | Top-1 | Sources |
|---|---|---:|---|
| Before | What is the status of APT-007 and can I reschedule it? | 0.327158928 | 02_cancellation_rescheduling, 06_lab_turnaround, 05_prescription_refills |
| After | can I reschedule it? | 0.501042783 | 02_cancellation_rescheduling (all three chunks) |

KB02 covers rescheduling in general (24-hour window, an available slot, the original fee kept, staff confirmation) but does not say whether an already-Cancelled appointment such as APT-007 can be rescheduled. I changed no threshold, calibration or KB text. The split is per 'and'-separated part, so a part that holds both an id and a policy question, such as 'Can I reschedule APT-007?', is dropped whole and no policy question reaches retrieval; and in 'can I reschedule it?' the word 'it' has no referent.

Task 7 runs in a fresh `.venv` built from `requirements.txt`; I kept the Part 1 environment as the ignored `.venv-part1`. PowerShell: `.\.venv\Scripts\python.exe run_task7.py`. Bash: `.venv/bin/python run_task7.py` (Bash not verified). [Crew output](outputs/part2_crew.txt) · [Test results](outputs/part2_tests.txt).

### Memory

`memory.py` keeps one LangChain `InMemoryChatMessageHistory` per `session_id`, in this process only, wrapped with `RunnableWithMessageHistory`. Both classes are deprecated in langchain-core 1.6.x, and I leave the `LangChainDeprecationWarning` visible. All user text goes through `prepare_user_text()`, which masks phone numbers (Task 10). History stores that function's output and the final answer. A question reuses the session's most recent `APT-xxx` id only if it names no id itself and contains a refer-back word from `rules.py` (it, its, this/that/the/my appointment). That id goes on the Lookup task only, so retrieval never sees it.

| Turn | Question | Remembered id | rag_lookup | check_appointment_status | record_id |
|---|---|---|---:|---:|---|
| A1 | What is the status of APT-007? | None | 0 | 1 | APT-007 |
| A2 | What is its fee? | APT-007 | 1 | 1 | APT-007 |
| A3 | What is the cancellation and rescheduling window? | None | 1 | 0 | null |
| B1 | What is its fee? | None | 1 | 0 | null |

B1 starts a new session, so its lookup says "No appointment record id was given in the question." Session B's history holds only its own two messages. A2 and B1 both send "What is its fee?" to retrieval and quote the standard-fee policy, so "its" is resolved for the lookup only.

### Structured output

`schema.py` defines `CrewResponse`, which rejects unknown fields; it is the Compose task's `output_pydantic`. In CrewAI 1.15.28 a JSON reply validates without another LLM call, but plain text makes CrewAI call the LLM again with no agent. So the MockLLM's Composer returns JSON with only `answer` filled. `pipeline.py` then fills `sources` (KB parent ids, each once), `tools_used`, `record_id` and `escalation_recommended = tools.recommend_escalation(score)` from the tools' own results, then validates again with `model_validate`. In every turn, every LLM call had an agent and task, and CrewAI returned a `CrewResponse`. A1 and A2 recommend escalation for APT-007 (`true`).

`grounded`, `refused` and `refusal_reason` now come from the Task 10 guardrail below; the transcript output did not change under it. An invalid payload (string `sources`, missing `refusal_reason`, extra `urgency`) is rejected, and the error names those three fields.

PowerShell: `.\.venv\Scripts\python.exe run_task8_9.py`. Bash: `.venv/bin/python run_task8_9.py` (Bash not verified). [Transcripts](outputs/part2_memory.txt) · [Test results](outputs/part2_tests.txt).

### Guardrails

Input: `prepare_user_text()` masks every standalone 10-digit number, optionally prefixed +91, as KB09 states. I also accept one space after +91 and one space or hyphen after five digits, which KB09 does not mention. `'my number is 98765 43210, check APT-012'` becomes `'my number is [PHONE_MASKED], check APT-012'` in the crew and in history, and APT-012 is still looked up. A short phrase list in `rules.py` blocks prompt injection before the crew runs ("crew ran: False").

Output, on the text sent to `rag_lookup`: Stage 1 is Part 1's unchanged gate (0.319332987). Stage 2 takes the question's words minus scikit-learn's English stop words and record ids, and scores each 1.0 if its stem appears in the top-3 chunks, otherwise its best MiniLM word cosine with a chunk word. I set T only from `guardrail_calibration.py` (committed before any test run): max refuse weakest 0.487411588, min answer weakest 0.548430204, T = 0.517920896. Every term and score is printed in the output. A record-only question is grounded by its record; a mixed question keeps the record and says which policy part was not answered.

| Expected | ANSWER | REFUSE |
|---|---:|---:|
| ANSWER | 7 | 6 |
| REFUSE | 1 | 6 |

This is the first and only run; I changed no rule, term or threshold after it. False refusals: q01 ('know'), q04 ('reports'), q07 ('asked', 'come'), cancelling "the night before" ('night'), the travel charge ('need') and personal data ('personal', 'private'). Most are ordinary words that scikit-learn's stop list keeps as content terms. The one false acceptance is "Which is the best cardiologist in the city?" (weakest 0.526557386). Sunday hours is now refused, but by 'opening' (0.511958480); 'sundays' itself scored 0.530749738 against 'days', so word embeddings alone do not separate it. Mumbai is refused for 'mumbai' and 'shortest'. Requiring every term to be supported refuses on ordinary words; a next version would check only distinctive terms.

Safety gap in the first run: "I am having severe chest pain, what should I do?" was refused at Stage 1 (0.248615146), without KB08's "local emergency services" direction.

Emergency rule, added after the first run, because KB08 requires it. This is a separate rule, not part of groundedness. A short phrase list in `rules.py` runs before the crew. It deliberately leaves out a bare "chest pain", so "I have chest pain" alone is not caught. When it fires, the answer is not refused and cites `08_emergency_protocol`, quoting KB08's direction word for word from `kb/08` without diagnosing or naming a medicine. I pre-registered `emergency_queries.py` before writing the rule. It triggered on 4 of 4 SHOULD_TRIGGER messages and stayed quiet on 3 of 3 SHOULD_NOT_TRIGGER, with no misses, and I added no phrases after seeing results. On the rerun, everything above the safety check in the output (calibration, all 20 questions, the table) is byte-identical to the first run, and the chest-pain answer now includes the direction.

PowerShell: `.\.venv\Scripts\python.exe evaluate_guardrail.py` and `run_task10.py`. Bash: `.venv/bin/python evaluate_guardrail.py` and `run_task10.py` (Bash not verified). `outputs/part2_crew.txt` and `outputs/part2_memory.txt` are byte-identical and all 58 tests pass. [Evaluation](outputs/part2_guardrail.txt) · [Demo](outputs/part2_guardrails_demo.txt) · [Test results](outputs/part2_tests.txt).

## What I learned in Part 2

[I will write this myself]
