"""Fixed questions and parent-document labels for the plain retrieval demo."""
# What: holds every fixed question used in Part 1, in one place, so the demo and
#   evaluation cannot drift apart. The questions and labels must not be edited.
# Task: Part 1 (Tasks 3-5). Used by: evaluate.py, demo.py and tests/test_part1.py.

# The 9 labelled questions. "relevant_parents" lists the policy files that contain the
# answer; evaluate.py scores retrieval against these labels.
DEMO_QUERIES = [
    {"id": "q01", "query": "How do I book an appointment and know that my slot is confirmed?",
     "relevant_parents": ["01_appointment_booking"]},
    {"id": "q02", "query": "What is the cancellation and rescheduling window?",
     "relevant_parents": ["02_cancellation_rescheduling"]},
    {"id": "q03", "query": "What are the consultation fees for each specialty?",
     "relevant_parents": ["03_consultation_fees"]},
    {"id": "q04", "query": "How long do routine and specialized lab test reports take?",
     "relevant_parents": ["06_lab_turnaround"]},
    {"id": "q05", "query": "When does a follow-up visit qualify for a discount?",
     "relevant_parents": ["10_follow_up_discount"]},
    {"id": "q06", "query": "How do I submit an insurance reimbursement claim?",
     "relevant_parents": ["04_insurance_claims"]},
    {"id": "q07", "query": "My doctor asked me to come back, do I have to pay the full price again?",
     "relevant_parents": ["10_follow_up_discount"]},
    {"id": "q08", "query": "What will I pay for a home visit by a cardiologist?",
     "relevant_parents": ["12_home_visits", "03_consultation_fees"]},
    {"id": "q09", "query": "Can I get a second opinion over a video call?",
     "relevant_parents": ["11_second_opinion", "07_telemedicine"]},
]

# Calibration questions: ones the policies can answer (in scope) and clearly unrelated
# ones (out of scope). Their similarity scores set the answer/fallback threshold.
CALIBRATION_IN_SCOPE = [
    "How can I reserve a clinic appointment?",
    "How much does a Cardiology consultation cost?",
    "What documents are needed for an insurance claim?",
    "Who is allowed to authorize a prescription refill?",
]
CALIBRATION_OUT_OF_SCOPE = [
    "How do I repair a punctured bicycle tyre?",
    "What is the orbital period of Neptune?",
    "Write a sorting algorithm in Rust.",
]
# The unrelated demo question that must always get the fallback reply.
DEMO_OUT_OF_SCOPE = "Give me a recipe for chocolate brownies."
# Near-topic questions the policies cannot answer, used by the stress check.
# Order matters: evaluate.stress() reads the Sunday-hours question at index 2.
STRESS_QUERIES = (
    "Can you diagnose my chest pain?",
    "Which medicine should I take for a fever?",
    "What are the clinic's opening hours on Sundays?",
    "Which hospital in Mumbai has the shortest waiting time?",
)
# Printed in evaluate.py's headings; the search itself uses rag.py's own TOP_K (also 3).
TOP_K = 3
