"""Calibration messages for the meaning-based emergency check, kept apart from its test set."""
# What: emergencies described in everyday words, and clinic questions that only mention
#   emergency-sounding topics. Their best similarity to EMERGENCY_DESCRIPTIONS sets
#   T_EMERGENCY; the test sets are never used for that.
# Task: Part 3 extension (emergency detection v2). Used by: guardrails.py and the tests.
# This file was committed on its own before any detector code and is never edited.

# Emergencies happening now, in words the phrase list does not contain.
CAL_EMERGENCY = [
    "My mother has collapsed and won't wake up",
    "He is choking and can't speak",
    "I think I'm having a heart attack",
    "There's a lot of blood and it won't stop",
    "My baby is turning blue",
    "Someone has taken too many pills",
    "I suddenly can't move one side of my body",
    "My friend fainted and is barely breathing",
]

# Ordinary clinic questions, some of which mention emergencies, hearts, blood or breathing.
CAL_NOT = [
    "What is the emergency-visit protocol?",
    "Do you have an emergency department?",
    "Can I get a refund if I cancel?",
    "How long do blood test results take?",
    "Is telemedicine available for children?",
    "What happens if I miss my appointment?",
    "Can I get a second opinion on my heart scan?",
    "How do I request a prescription refill?",
]
