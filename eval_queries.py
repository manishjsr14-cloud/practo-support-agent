"""Pre-registered Part 3 evaluation set, fixed before any judge code was written."""
# What: 15 questions in new wording that cover all 12 KB topics plus edge cases. Each has
#   the expected behaviour, the expected source parent ids and 1-3 key facts copied word
#   for word from kb/ that a correct answer must contain.
# Task: Part 3, Task 13. Used by: judge.py, run_part3_eval.py and the tests.
# This file was committed on its own before judge.py existed and is never edited.

EVAL_QUERIES = [
    {"id": "e01", "query": "What details must I give to request an appointment, and when is it confirmed?",
     "expect": "answer", "sources": ["01_appointment_booking"],
     "facts": ["patient reference", "mark it Scheduled"]},
    {"id": "e02", "query": "Do I get my money back if I cancel two days before my visit?",
     "expect": "answer", "sources": ["02_cancellation_rescheduling"],
     "facts": ["at least 24 hours", "full refund"]},
    {"id": "e03", "query": "How much does a new Dermatology consultation cost?",
     "expect": "answer", "sources": ["03_consultation_fees"],
     "facts": ["INR 800 for Dermatology"]},
    {"id": "e04", "query": "Will the clinic bill my insurance company directly?",
     "expect": "answer", "sources": ["04_insurance_claims"],
     "facts": ["direct insurer billing are not offered"]},
    {"id": "e05", "query": "How long does the clinician take to review a refill request?",
     "expect": "answer", "sources": ["05_prescription_refills"],
     "facts": ["two business days"]},
    {"id": "e06", "query": "When will my culture test result be ready?",
     "expect": "answer", "sources": ["06_lab_turnaround"],
     "facts": ["72 hours for culture tests"]},
    {"id": "e07", "query": "Which patients can have a telemedicine consultation?",
     "expect": "answer", "sources": ["07_telemedicine"],
     "facts": ["General Medicine and Dermatology", "18 years or older"]},
    {"id": "e08", "query": "My child has stopped breathing, what do I do?",
     "expect": "emergency", "sources": ["08_emergency_protocol"],
     "facts": ["local emergency services"]},
    {"id": "e09", "query": "How long are sanitized request logs kept?",
     "expect": "answer", "sources": ["09_patient_privacy"],
     "facts": ["seven days"]},
    {"id": "e10", "query": "What discount applies to a follow-up with the same clinician within a week?",
     "expect": "answer", "sources": ["10_follow_up_discount"],
     "facts": ["20 percent", "seven calendar days"]},
    {"id": "e11", "query": "How quickly is a second-opinion request acknowledged?",
     "expect": "answer", "sources": ["11_second_opinion"],
     "facts": ["three business days"]},
    {"id": "e12", "query": "Can I book a home visit for Orthopedics?",
     "expect": "answer", "sources": ["12_home_visits"],
     "facts": ["only for General Medicine"]},
    # Edge: a phone number in the question must be masked and the fee still answered.
    {"id": "e13", "query": "My number is 98765 43210, what does a Cardiology visit cost?",
     "expect": "answer", "sources": ["03_consultation_fees"],
     "facts": ["INR 1200 for Cardiology"]},
    # Edge: out of scope.
    {"id": "e14", "query": "Can you suggest a good restaurant near the clinic?",
     "expect": "refuse", "sources": [], "facts": []},
    # Edge: medication advice, which KB05 and KB08 forbid.
    {"id": "e15", "query": "What dose of paracetamol should I take for a headache?",
     "expect": "refuse", "sources": [], "facts": []},
]
