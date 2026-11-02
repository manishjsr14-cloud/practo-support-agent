"""Pre-registered test set for emergency detection v2, fixed before any detector code."""
# What: emergencies described without the phrase list's wording, and questions that sound
#   related but are not emergencies.
# Task: Part 3 extension (emergency detection v2). Used by: the v2 evaluation and the tests.
# This file was committed on its own before any detector code and is never edited.

SHOULD_TRIGGER = [
    "My dad is not responding and his lips are blue",
    "I swallowed something and can't breathe properly",
    "My wife has trouble breathing and chest tightness",
    "A child fell, hit her head and is unconscious",
    "I'm bleeding a lot from a deep cut",
    "My grandfather's face is drooping and his speech is slurred",
]

SHOULD_NOT_TRIGGER = [
    "Does the clinic treat emergencies?",
    "How much does a Cardiology consultation cost?",
    "Can I reschedule my appointment tomorrow?",
    "What documents do I need for an insurance claim?",
    "Is my chest X-ray report ready?",
    "Can I have a home visit after an accident last month?",
]
