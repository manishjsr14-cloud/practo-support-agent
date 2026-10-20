"""Offline and privacy settings that must be applied before any framework import."""
# What: sets every environment flag AGENTS.md requires, including CrewAI's storage folder,
#   so no model download, telemetry, tracing or progress bar can happen.
# Task: Part 2, Task 7 (step 0). Used by: crew.py, run_task7.py and tests/test_task7.py,
#   which import this module BEFORE importing crewai.
# Run "python settings.py" to print the full flag list.

import os
from pathlib import Path

# CrewAI writes its local storage here (inside the repo, ignored by git) instead of a
# folder in the user's home directory.
ROOT = Path(__file__).resolve().parent
CREWAI_STORAGE_DIR = ROOT / ".crewai"

# Every flag, in one place. Libraries read these when they are first imported, which is
# why this module must be imported before them.
FLAGS = {
    "HF_HUB_OFFLINE": "1",
    "TRANSFORMERS_OFFLINE": "1",
    "ANONYMIZED_TELEMETRY": "False",
    "CREWAI_DISABLE_TELEMETRY": "true",
    "OTEL_SDK_DISABLED": "true",
    "CREWAI_TRACING_ENABLED": "false",
    "HF_HUB_DISABLE_PROGRESS_BARS": "1",
    "CREWAI_STORAGE_DIR": str(CREWAI_STORAGE_DIR),
}
os.environ.update(FLAGS)


# Print each flag with the value the environment holds now.
# Takes: nothing. Returns: nothing. Why: saved outputs can show which flags were active.
def print_flags():
    for name in FLAGS:
        print(f"{name}={os.environ[name]}")


if __name__ == "__main__":
    print_flags()
