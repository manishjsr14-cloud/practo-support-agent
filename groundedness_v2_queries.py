"""Pre-registered test set for groundedness v2, fixed before any v2 code."""
# What: answerable questions in new wording (each with the policy that answers it), questions
#   the policies cannot answer, and the rule that decides whether v2 becomes the default.
# Task: groundedness v2 experiment. Used by: the v1 vs v2 evaluation and the tests.
# This file was committed on its own before any v2 code and is never edited.

# (question, expected parent id)
SHOULD_ANSWER = [
    ("What information do I need to give when booking a visit?", "01_appointment_booking"),
    ("Is there a fee if I move my appointment with three days' notice?", "02_cancellation_rescheduling"),
    ("What does a Pediatrics consultation cost for a new booking?", "03_consultation_fees"),
    ("Which documents does the clinic give me for an insurance reimbursement?", "04_insurance_claims"),
    ("Can the support agent approve my medicine refill?", "05_prescription_refills"),
    ("How long do specialized lab tests usually take?", "06_lab_turnaround"),
    ("Can a 16-year-old use telemedicine?", "07_telemedicine"),
    ("How long is my chat history kept?", "09_patient_privacy"),
    ("Is a second opinion charged at the normal fee?", "11_second_opinion"),
    ("How far from the clinic can I live for a home visit?", "12_home_visits"),
]

SHOULD_REFUSE = [
    "Does the clinic offer laser eye surgery?",
    "Can I bring my dog to the waiting room?",
    "What time does the pharmacy close on Saturdays?",
    "Which insurance company is the cheapest?",
    "Can you recommend a good physiotherapist nearby?",
]

# v2 becomes the default only if, on these 15 questions, it has fewer false refusals than v1
#   AND no more false acceptances than v1. Otherwise v2 stays off by default as a documented
#   experiment.
ADOPTION_RULE = ("v2 becomes the default only if, on these 15 questions, it has fewer false refusals "
                 "than v1 AND no more false acceptances than v1. Otherwise v2 stays off by default "
                 "as a documented experiment.")
